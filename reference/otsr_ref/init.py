"""Optimal-transport initialisation (make_init "ot" of Agent/cc/exp/015_revision/run_ours15.py).

A gy x gx lattice (gy gx ~ K, aspect matched to the image) is moved by the entropic-OT map from the uniform density to
an edge-importance density; each Gaussian takes the lattice cell deformed by the map's Jacobian as its covariance and
the image colour at its new centre. Float32 throughout, as in the research code.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .spec import (K_PER_PX, LUMA, MIN_SCALE_PX, SIGMA_REL, SINKHORN_FINAL_ITERS, SINKHORN_ITERS_PER_STAGE,
                   SINKHORN_SHRINK, n_long_for)


def lattice_dims(N: int, H: int, W: int):
    """Gy x Gx grid with Gy/Gx ~ H/W and Gy*Gx ~ N."""
    gx = max(1, round(math.sqrt(N * W / H)))
    gy = max(1, round(N / gx))
    return gy, gx


def lattice_uv(gy: int, gx: int, device="cpu"):
    ys = (torch.arange(gy, device=device, dtype=torch.float32) + 0.5) / gy
    xs = (torch.arange(gx, device=device, dtype=torch.float32) + 0.5) / gx
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], -1).reshape(-1, 2)


def lam_for_spacing(h_px: float) -> float:
    """Uniform share of the importance density by lattice spacing h (px): 0.3 at h <= 4, 0.1 at h >= 8."""
    return float(min(0.3, max(0.1, 0.3 - 0.2 * (math.log2(max(h_px, 1e-6)) - 2.0))))


def ot_grid_shape(H: int, W: int, n_long: int):
    L = max(H, W)
    return max(8, round(n_long * H / L)), max(8, round(n_long * W / L))


def pixel_density(gt, lam: float):
    """Image-GS importance per pixel (sums to 1): lam uniform + (1 - lam) |dx| + |dy| of luma (forward differences,
    zero in the last column / row)."""
    y = LUMA[0] * gt[0] + LUMA[1] * gt[1] + LUMA[2] * gt[2]
    gxm = F.pad((y[:, 1:] - y[:, :-1]).abs(), (0, 1))
    gym = F.pad((y[1:, :] - y[:-1, :]).abs(), (0, 0, 0, 1))
    g = gxm + gym
    return lam / g.numel() + (1 - lam) * g / g.sum().clamp_min(1e-12)


def pooled_density(gt, lam: float, n_long: int):
    _, H, W = gt.shape
    ny, nx = ot_grid_shape(H, W, n_long)
    p = F.adaptive_avg_pool2d(pixel_density(gt, lam)[None, None], (ny, nx))[0, 0]
    return p / p.sum()


@torch.no_grad()
def transport_map(rho, ex: float = 1.0, ey: float = 1.0, dtype=torch.float32):
    """Entropic OT from the uniform density to rho (ny, nx) on the box [0, ex] x [0, ey] (long side 1), float32
    log-domain Sinkhorn with eps-scaling (common/ot.py transport_map_batched, B = 1). Returns the barycentric map
    T (ny, nx, 2), (x, y) in box units at the cell centres, and a dict with the final eps and the iteration count.
    ``dtype=torch.float64`` gives the precise map the float32 solvers (this one and the WGSL one) are measured
    against: at the final eps the exponents are ~1e4, so float32 keeps only ~1e-3 of absolute precision there."""
    dev = rho.device
    ny, nx = rho.shape
    xs = (torch.arange(nx, device=dev, dtype=dtype) + 0.5) / nx * ex
    ys = (torch.arange(ny, device=dev, dtype=dtype) + 0.5) / ny * ey
    Cx = (xs[:, None] - xs[None, :]) ** 2
    Cy = (ys[:, None] - ys[None, :]) ** 2
    nu = rho.to(dtype).clamp_min(0) + 1e-12
    nu = nu / nu.sum()
    lnu = nu.log()
    lmu = torch.full_like(lnu, -math.log(ny * nx))
    h = max(ex / nx, ey / ny)
    eps = (1.5 * h) ** 2
    f = torch.zeros_like(lnu)
    g = torch.zeros_like(lnu)

    def soft_min(pot, e):     # LSE_b [(pot(b) - c(a, b)) / e], separable: x first, then y
        inner = torch.logsumexp((pot[:, None, :] - Cx[None]) / e, dim=2)              # (ny_b, nx_a)
        return torch.logsumexp(inner[None, :, :] - Cy[:, :, None] / e, dim=1)          # (ny_a, nx_a)

    e = max(ex, ey) ** 2
    n_iter = 0
    while True:
        for _ in range(SINKHORN_FINAL_ITERS if e <= eps else SINKHORN_ITERS_PER_STAGE):
            f = e * lmu - e * soft_min(g, e)
            g = e * lnu - e * soft_min(f, e)
            n_iter += 1
        if e <= eps:
            break
        e = max(eps, e * SINKHORN_SHRINK)
    base = (g[:, None, :] - Cx[None]) / e                                               # (ny_b, nx_a, nx_b)
    inner = torch.logsumexp(base, dim=2)
    inner_x = torch.logsumexp(base + torch.log(xs)[None, None, :], dim=2)
    outer = lambda t: torch.logsumexp(t[None, :, :] - Cy[:, :, None] / e, dim=1)      # noqa: E731
    lse = outer(inner)
    Tx = torch.exp(outer(inner_x) - lse)
    Ty = torch.exp(torch.logsumexp(inner[None, :, :] - Cy[:, :, None] / e + torch.log(ys)[None, :, None], dim=1) - lse)
    return torch.stack([Tx, Ty], -1).float(), {"eps": e, "iters": n_iter, "f": f, "g": g}


def warp_points(uv, T, ex: float = 1.0, ey: float = 1.0):
    """uv (n, 2) -> new uv (n, 2) and the Jacobian dT/dx (n, 2, 2) in box units: bilinear interpolation (border) of
    the displacement field and of its central-difference Jacobian (one-sided at the border)."""
    ny, nx, _ = T.shape
    dev = T.device
    box = T.new_tensor([ex, ey])
    xs = (torch.arange(nx, device=dev, dtype=T.dtype) + 0.5) / nx * ex
    ys = (torch.arange(ny, device=dev, dtype=T.dtype) + 0.5) / ny * ey
    X = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1)                          # (ny, nx, 2)
    D = (T - X).permute(2, 0, 1)[None]
    dTdx = torch.gradient(T, spacing=(ex / nx,), dim=1)[0]
    dTdy = torch.gradient(T, spacing=(ey / ny,), dim=0)[0]
    Jg = torch.stack([dTdx, dTdy], -1).reshape(ny, nx, 4).permute(2, 0, 1)[None]        # dTx/dx, dTx/dy, dTy/dx, dTy/dy
    gs = (uv * 2 - 1).view(1, 1, -1, 2).to(T.dtype)
    d = F.grid_sample(D, gs, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0].T
    j = F.grid_sample(Jg, gs, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0].T
    p = uv.to(T.dtype) * box + d
    return (p / box).clamp(0, 1), j.view(-1, 2, 2)


def cov_to_scale_rot(C):
    """C (n, 2, 2) SPD -> scales (n, 2), larger first, and the kernel angle (n, 1) of the first principal axis."""
    evals, evecs = torch.linalg.eigh(C)
    v1 = evecs[..., :, 1]
    s = torch.sqrt(evals.clamp_min(1e-12)).flip(-1)
    theta = torch.atan2(-v1[:, 1], v1[:, 0]).unsqueeze(1)
    return s, theta


def sample_color(gt, uv):
    """Bilinear sample (border) of gt (3, H, W) at uv (N, 2) -> (N, 3)."""
    grid = (uv * 2 - 1).view(1, 1, -1, 2)
    return F.grid_sample(gt[None], grid, mode="bilinear", align_corners=False, padding_mode="border")[0, :, 0].T


def ot_init(gt, K: int | None = None, k_per_px: float = K_PER_PX, n_long: int | None = None, debug: bool = False):
    """gt (3, H, W) float32 in [0, 1] -> params (gy * gx Gaussians, close to K). debug adds the intermediates."""
    _, H, W = gt.shape
    L = max(H, W)
    K = int(round(k_per_px * H * W)) if K is None else int(K)
    gy, gx = lattice_dims(K, H, W)
    h_px = math.sqrt(H * W / (gy * gx))
    lam = lam_for_spacing(h_px)
    n_long = n_long_for(H, W) if n_long is None else n_long
    rho = pooled_density(gt, lam, n_long)
    T, info = transport_map(rho, W / L, H / L)
    uv, J = warp_points(lattice_uv(gy, gx, gt.device), T, W / L, H / L)
    s, theta = cov_to_scale_rot((SIGMA_REL * h_px / L) ** 2 * (J @ J.transpose(1, 2)))
    p = {"uv": uv.float().clone(), "log_s": torch.log(s.clamp(min=MIN_SCALE_PX / L)).float(), "rot": theta.float(),
         "color": sample_color(gt, uv.float()).clone(), "log_w": torch.zeros(uv.shape[0], 1, device=gt.device)}
    if debug:
        return p, {"gy": gy, "gx": gx, "lam": lam, "n_long": n_long, "rho": rho, "T": T, "J": J, **info}
    return p
