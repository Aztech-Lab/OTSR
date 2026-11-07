"""Resolution-independent 2D Gaussian rendering (wraps the Image-GS gsplat fork in ``gmod/``).

Representation of one image (all tensors float32, N Gaussians):

* ``uv``     (N, 2)  centre in continuous image coordinates, [0, 1] per axis.
                     Pixel (row i, col j) of an H x W raster samples uv = ((j + .5) / W, (i + .5) / H).
* ``scale``  (N, 2)  standard deviations along the two principal axes, in units of the
                     image long side L = max(H, W).  A Gaussian therefore keeps its physical
                     shape at every output resolution and aspect ratio.
* ``rot``    (N, 1)  rotation of the principal axes, radians.
* ``color``  (N, 3)  RGB.
* ``weight`` (N, 1)  optional positive weight (only used by ``mode='wnorm'``).

Modes
-----
``topk``   Image-GS: colour = sum_k c_k a_k / (sum_k a_k + 1e-4) over the 10 largest a_k at the pixel.
``wnorm``  weighted Nadaraya-Watson mixture over every Gaussian within 3 sigma:
           colour = sum w c a / (sum w a + eps).
``sum``    additive (GaussianImage style): colour = sum c a.

``aa_px`` adds an isotropic pixel-footprint variance (in output pixels) before rasterising.
For ``sum``/``wnorm`` the amplitude is compensated so each Gaussian keeps its integral,
which is the correct prefilter when rendering below the native resolution.
"""
from __future__ import annotations

import math
import os
import sys

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gmod.gsplat.project_gaussians_2d_scale_rot import project_gaussians_2d_scale_rot  # noqa: E402
from gmod.gsplat.rasterize_sum import rasterize_gaussians_sum  # noqa: E402

BLOCK = 16
RADIUS_MULT = 1.0   # >1: cull at RADIUS_MULT x 3 sigma instead of 3 sigma (render-only option, e.g. for S x renders)


def tile_bounds(H: int, W: int):
    return ((W + BLOCK - 1) // BLOCK, (H + BLOCK - 1) // BLOCK, 1)


def conics_from_scale_rot(s_px, rot):
    """Inverse covariance (a, b, c) of R diag(s^2) R^T with the kernel's R = [[cos, sin], [-sin, cos]].
    Closed form: Sigma^-1 = R diag(1/s^2) R^T.
    ``rot`` is either an angle (N,1) or a doubled-angle vector (N,2) ~ (cos 2t, sin 2t) (normalised here),
    which is continuous for ellipse orientation (period pi) and has no atan2 singularity."""
    i1, i2 = 1.0 / (s_px[:, 0:1] ** 2), 1.0 / (s_px[:, 1:2] ** 2)
    if rot.shape[-1] == 2:
        v = rot / torch.sqrt((rot * rot).sum(-1, keepdim=True) + 1e-8)
        C, S = v[:, 0:1], v[:, 1:2]
        a = 0.5 * (1 + C) * i1 + 0.5 * (1 - C) * i2
        b = 0.5 * S * (i2 - i1)
        d = 0.5 * (1 - C) * i1 + 0.5 * (1 + C) * i2
        return torch.cat([a, b, d], dim=1)
    c, s = torch.cos(rot), torch.sin(rot)
    a = c * c * i1 + s * s * i2
    b = -c * s * i1 + s * c * i2
    d = s * s * i1 + c * c * i2
    return torch.cat([a, b, d], dim=1)


def angle(rot):
    """Angle (N,1) from either representation."""
    if rot.shape[-1] == 2:
        return 0.5 * torch.atan2(rot[:, 1:2], rot[:, 0:1])
    return rot.reshape(-1, 1)


KERNEL_GRADS = False   # ablation only (015): take centres / conics from the kernel *with* its (wrong) backward


def project(uv, s_px, rot, H, W):
    """Pixel-space centres, integer radii, conics and tile counts.

    The fork's CUDA backward for scale/rotation is wrong (audited in test_render.py), so only the integer
    culling data (radii, tile counts) come from the kernel; centres and conics are differentiable PyTorch."""
    shift = uv.new_tensor([0.5 / W, 0.5 / H])
    means = (uv.float() - shift).contiguous()
    rot = rot.float()
    rot = rot.reshape(-1, 2) if rot.shape[-1] == 2 else rot.reshape(-1, 1)
    if KERNEL_GRADS:
        return project_gaussians_2d_scale_rot(means, s_px.float().contiguous(), angle(rot).contiguous(), H, W,
                                              tile_bounds(H, W))
    with torch.no_grad():
        _, radii, _, nth = project_gaussians_2d_scale_rot(
            means.detach(), s_px.detach().contiguous(), angle(rot.detach()).contiguous(), H, W, tile_bounds(H, W))
    xys = means * means.new_tensor([float(W), float(H)])
    conics = conics_from_scale_rot(s_px, rot)
    if RADIUS_MULT != 1.0:                 # wider support: radii and tile counts as the kernel would compute them
        with torch.no_grad():
            rad = torch.ceil(3.0 * RADIUS_MULT * s_px.detach().float().amax(1))
            ntx, nty = (W + BLOCK - 1) // BLOCK, (H + BLOCK - 1) // BLOCK
            cx, cy = xys[:, 0].detach(), xys[:, 1].detach()
            tx0 = torch.clamp(torch.trunc((cx - rad) / BLOCK), 0, ntx)
            tx1 = torch.clamp(torch.trunc((cx + rad) / BLOCK + 1), 0, ntx)
            ty0 = torch.clamp(torch.trunc((cy - rad) / BLOCK), 0, nty)
            ty1 = torch.clamp(torch.trunc((cy + rad) / BLOCK + 1), 0, nty)
            vis = radii > 0
            radii = torch.where(vis, rad.to(radii.dtype), radii)
            nth = torch.where(vis, ((tx1 - tx0) * (ty1 - ty0)).to(nth.dtype), nth)
    return xys, radii, conics, nth


def project_cov(uv, cov_px, H, W):
    """Like project() but from a covariance (N,3) = (xx, xy, yy) in pixels^2: conic = inverse (differentiable,
    smooth everywhere for SPD input); the kernel only supplies integer culling data for an isotropic proxy with
    the largest eigenvalue (radius >= the exact one, so nothing is clipped)."""
    shift = uv.new_tensor([0.5 / W, 0.5 / H])
    means = (uv.float() - shift).contiguous()
    a, b, d = cov_px[:, 0:1], cov_px[:, 1:2], cov_px[:, 2:3]
    with torch.no_grad():
        m = 0.5 * (a + d)
        lam = m + torch.sqrt((0.5 * (a - d)) ** 2 + b * b)
        s_eq = torch.sqrt(lam.clamp_min(1e-12)).repeat(1, 2).contiguous()
        _, radii, _, nth = project_gaussians_2d_scale_rot(
            means.detach(), s_eq, torch.zeros_like(a), H, W, tile_bounds(H, W))
    det = (a * d - b * b).clamp_min(1e-12)
    conics = torch.cat([d / det, -b / det, a / det], dim=1)
    xys = means * means.new_tensor([float(W), float(H)])
    return xys, radii, conics, nth


def render_one(uv, scale, rot, color, H: int, W: int, mode: str = "wnorm", weight=None,
               aa_px: float = 0.0, eps: float = 1e-4, bg=None, cov=None):
    """Render one image. Returns (3, H, W). If ``cov`` (N,3) = (xx, xy, yy) in long-side units^2 is given,
    it replaces scale/rot."""
    L = float(max(H, W))
    fac = None
    if cov is not None:
        cpx = cov.float() * (L * L)
        if aa_px > 0:
            det0 = (cpx[:, 0:1] * cpx[:, 2:3] - cpx[:, 1:2] ** 2).clamp_min(1e-12)
            cpx = cpx + cpx.new_tensor([aa_px ** 2, 0.0, aa_px ** 2])
            det1 = (cpx[:, 0:1] * cpx[:, 2:3] - cpx[:, 1:2] ** 2).clamp_min(1e-12)
            fac = torch.sqrt(det0 / det1)
        xys, radii, conics, nth = project_cov(uv, cpx, H, W)
    else:
        s_px = scale.float() * L
        if aa_px > 0:
            s_eff = torch.sqrt(s_px * s_px + aa_px * aa_px)
            fac = (s_px[:, :1] * s_px[:, 1:]) / (s_eff[:, :1] * s_eff[:, 1:])
            s_px = s_eff
        xys, radii, conics, nth = project(uv, s_px, rot, H, W)
    if int(nth.sum().item()) == 0:  # nothing on screen; the CUDA path would crash on empty bins
        out = uv.new_zeros(3, H, W)
        return out if bg is None else out + bg.view(3, 1, 1)
    if mode == "topk":
        img = rasterize_gaussians_sum(xys, radii, conics, nth, color.float(), H, W, BLOCK, BLOCK, topk_norm=True)
    elif mode == "sum":
        c = color.float() if fac is None else color.float() * fac
        img = rasterize_gaussians_sum(xys, radii, conics, nth, c, H, W, BLOCK, BLOCK, topk_norm=False)
    elif mode == "wnorm":
        w = torch.ones_like(color[:, :1]) if weight is None else weight.float().reshape(-1, 1)
        if fac is not None:
            w = w * fac
        feats = torch.cat([color.float() * w, w], dim=1)
        acc = rasterize_gaussians_sum(xys, radii, conics, nth, feats, H, W, BLOCK, BLOCK, topk_norm=False)
        num, den = acc[..., :3], acc[..., 3:]
        if bg is None:
            img = num / (den + eps)
        else:
            img = (num + eps * bg.view(1, 1, 3)) / (den + eps)
    else:
        raise ValueError(mode)
    return img.reshape(H, W, 3).permute(2, 0, 1)


def render_batch(g: dict, H: int, W: int, mode: str = "wnorm", aa_px: float = 0.0, eps: float = 1e-4):
    """Render a list-of-dicts or dict-of-lists batch. ``g`` keys: uv, scale, rot, color, (weight), each a
    list of per-image tensors (variable N allowed) or a stacked (B, N, .) tensor. Returns (B, 3, H, W)."""
    B = len(g["uv"])
    outs = []
    for b in range(B):
        w = g.get("weight")
        cv = g.get("cov")
        outs.append(render_one(g["uv"][b], None if cv is not None else g["scale"][b],
                               None if cv is not None else g["rot"][b], g["color"][b], H, W, mode=mode,
                               weight=None if w is None else w[b], aa_px=aa_px, eps=eps,
                               cov=None if cv is None else cv[b]))
    return torch.stack(outs, 0)


def dense_reference(uv, scale, rot, color, H, W, mode="wnorm", weight=None, aa_px=0.0, eps=1e-4,
                    truncate=True):
    """Pure PyTorch O(N*H*W) reference used only by tests and audits (float64 friendly)."""
    dt = uv.dtype
    L = float(max(H, W))
    s = scale * L
    fac = torch.ones_like(s[:, :1])
    if aa_px > 0:
        se = torch.sqrt(s * s + aa_px ** 2)
        fac = (s[:, :1] * s[:, 1:]) / (se[:, :1] * se[:, 1:])
        s = se
    c, sn = torch.cos(rot.reshape(-1)), torch.sin(rot.reshape(-1))
    # kernel convention (glm is column-major): R = [[cos, sin], [-sin, cos]]
    R = torch.stack([torch.stack([c, sn], -1), torch.stack([-sn, c], -1)], -2)  # (N,2,2)
    S2 = torch.diag_embed(s * s)
    cov = R @ S2 @ R.transpose(-1, -2)
    a, b_, d = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]
    det = a * d - b_ * b_
    ca, cb, cd = d / det, -b_ / det, a / det
    cx = (uv[:, 0] - 0.5 / W) * W
    cy = (uv[:, 1] - 0.5 / H) * H
    ys, xs = torch.meshgrid(torch.arange(H, dtype=dt, device=uv.device),
                            torch.arange(W, dtype=dt, device=uv.device), indexing="ij")
    dx = cx.view(-1, 1, 1) - xs
    dy = cy.view(-1, 1, 1) - ys
    q = 0.5 * (ca.view(-1, 1, 1) * dx * dx + cd.view(-1, 1, 1) * dy * dy) + cb.view(-1, 1, 1) * dx * dy
    alpha = torch.exp(-q)
    if truncate:  # emulate the CUDA tile culling exactly: radius = ceil(3*sqrt(lambda_max)), 16x16 tiles
        bb = 0.5 * (a + d)
        lam = bb + torch.sqrt(torch.clamp(bb * bb - det, min=0.1))
        rad = torch.ceil(3.0 * torch.sqrt(lam))
        ntx, nty = (W + BLOCK - 1) // BLOCK, (H + BLOCK - 1) // BLOCK
        tx0 = torch.clamp(torch.trunc((cx - rad) / BLOCK), 0, ntx)
        tx1 = torch.clamp(torch.trunc((cx + rad) / BLOCK + 1), 0, ntx)
        ty0 = torch.clamp(torch.trunc((cy - rad) / BLOCK), 0, nty)
        ty1 = torch.clamp(torch.trunc((cy + rad) / BLOCK + 1), 0, nty)
        tj = torch.div(xs, BLOCK, rounding_mode="floor")
        ti = torch.div(ys, BLOCK, rounding_mode="floor")
        inside = ((tj >= tx0.view(-1, 1, 1)) & (tj < tx1.view(-1, 1, 1)) &
                  (ti >= ty0.view(-1, 1, 1)) & (ti < ty1.view(-1, 1, 1)))
        alpha = alpha * inside
    if mode == "sum":
        return torch.einsum("nhw,nc->chw", alpha, color * fac)
    if mode == "wnorm":
        w = (torch.ones_like(color[:, :1]) if weight is None else weight.reshape(-1, 1)) * fac
        num = torch.einsum("nhw,nc->chw", alpha, color * w)
        den = torch.einsum("nhw,n->hw", alpha, w.reshape(-1))
        return num / (den + eps)
    if mode == "topk":
        k = min(10, alpha.shape[0])
        vals, idx = alpha.topk(k, dim=0)
        cols = color[idx]  # (k,H,W,3)
        return (cols * vals.unsqueeze(-1)).sum(0).permute(2, 0, 1) / (vals.sum(0) + 1e-4)
    raise ValueError(mode)
