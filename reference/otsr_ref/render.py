"""The OTSR image function, evaluated densely (O(N x pixels)): the specification the WebGPU rasteriser must match.

A model is a dict of tensors for N Gaussians (``PARAM_LAYOUT`` in spec.py):

* ``uv``    (N, 2)  centre in [0, 1]^2, (x, y). Source pixel (row i, col j) covers [j, j+1) x [i, i+1) in source
                    pixel units, so its centre is uv = ((j + .5) / W, (i + .5) / H).
* ``log_s`` (N, 2)  log standard deviations along the two principal axes, in units of the long side L = max(H, W).
* ``rot``   (N, 1)  rotation in radians, kernel convention R = [[cos, sin], [-sin, cos]].
* ``color`` (N, 3)  RGB.
* ``log_w`` (N, 1)  log weight.

Image function at a point x (source pixel units):

    q_i(x)     = 0.5 (x - m_i)^T Sigma_i^-1 (x - m_i),   m_i = uv_i * (W, H),   Sigma_i = R diag(s^2) R^T, s = e^log_s L
    alpha_i(x) = max(exp(-q_i) - exp(-Q_CUT), 0)                                (Q_CUT = 8: 4-sigma support)
    out(x)     = sum_i w_i alpha_i c_i / (sum_i w_i alpha_i + EPS)

A *view* samples this function on an output raster: output pixel (i, j) has its centre at the source position
((j + .5 - tx) / zoom, (i + .5 - ty) / zoom). zoom = S, tx = ty = 0 is the S x upscale; zoom = 1 is the fit raster.
"""
from __future__ import annotations

import math

import torch

from .spec import ALPHA_SHIFT, EPS, Q_CUT


def gaussian_terms(p: dict, H: int, W: int, zoom: float = 1.0, tx: float = 0.0, ty: float = 0.0):
    """Centres in output-pixel index coordinates, conic (a, b, d) in output pixels^-2 and weights."""
    L = max(H, W)
    uv = p["uv"]
    mx = uv[:, 0] * (W * zoom) + (tx - 0.5)
    my = uv[:, 1] * (H * zoom) + (ty - 0.5)
    s = torch.exp(p["log_s"]) * (L * zoom)
    i1, i2 = 1.0 / (s[:, 0] * s[:, 0]), 1.0 / (s[:, 1] * s[:, 1])
    c, sn = torch.cos(p["rot"][:, 0]), torch.sin(p["rot"][:, 0])
    a = c * c * i1 + sn * sn * i2
    b = c * sn * (i2 - i1)
    d = sn * sn * i1 + c * c * i2
    return mx, my, a, b, d, torch.exp(p["log_w"][:, 0])


def render(p: dict, H: int, W: int, out_h: int | None = None, out_w: int | None = None, zoom: float = 1.0,
           tx: float = 0.0, ty: float = 0.0, chunk: int = 256, with_den: bool = False):
    """(3, out_h, out_w) image; with_den also returns D = sum w alpha + EPS (out_h, out_w). Differentiable."""
    out_h = H if out_h is None else out_h
    out_w = W if out_w is None else out_w
    mx, my, a, b, d, w = gaussian_terms(p, H, W, zoom, tx, ty)
    dt, dev = p["uv"].dtype, p["uv"].device
    xs = torch.arange(out_w, dtype=dt, device=dev)
    ys = torch.arange(out_h, dtype=dt, device=dev)
    num = torch.zeros(3, out_h, out_w, dtype=dt, device=dev)
    den = torch.zeros(out_h, out_w, dtype=dt, device=dev)
    for k0 in range(0, mx.shape[0], chunk):
        sl = slice(k0, k0 + chunk)
        dx = mx[sl, None, None] - xs[None, None, :]
        dy = my[sl, None, None] - ys[None, :, None]
        q = 0.5 * (a[sl, None, None] * dx * dx + d[sl, None, None] * dy * dy) + b[sl, None, None] * dx * dy
        u = w[sl, None, None] * (torch.exp(-q) - ALPHA_SHIFT).clamp_min(0)
        num = num + torch.einsum("nhw,nc->chw", u, p["color"][sl])
        den = den + u.sum(0)
    D = den + EPS
    out = num / D
    return (out, D) if with_den else out


def mse(out, gt):
    """The fitting loss: mean over all 3 H W values (F.mse_loss)."""
    return ((out - gt) ** 2).mean()


def psnr(out, gt) -> float:
    """PSNR of the clamped image, as the research logs it."""
    m = float(((out.detach().clamp(0, 1) - gt) ** 2).mean())
    return -10 * math.log10(max(m, 1e-12))


def support_box(p: dict, H: int, W: int, zoom: float = 1.0, tx: float = 0.0, ty: float = 0.0):
    """Axis-aligned half extents (rx, ry) of the 4-sigma ellipse in output pixels: k sqrt(Sigma_xx), k sqrt(Sigma_yy)."""
    L = max(H, W)
    s = torch.exp(p["log_s"]) * (L * zoom)
    c, sn = torch.cos(p["rot"][:, 0]), torch.sin(p["rot"][:, 0])
    sxx = c * c * s[:, 0] ** 2 + sn * sn * s[:, 1] ** 2
    syy = sn * sn * s[:, 0] ** 2 + c * c * s[:, 1] ** 2
    k = math.sqrt(2 * Q_CUT)
    return k * torch.sqrt(sxx), k * torch.sqrt(syy)
