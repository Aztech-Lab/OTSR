"""Optimal-transport layouts for Gaussians on a lattice (Agent/cc 011).

Entropic OT between a source density (default: uniform, i.e. the encoder's lattice) and a target density rho on a
regular grid over the image box [0, W/L] x [0, H/L] (long side = 1, so the quadratic cost is isotropic in pixels).
Log-domain Sinkhorn with eps-scaling; the cost is separable, so every kernel application is two 1-D log-sum-exps
(O(ny nx (ny + nx)) per iteration). The barycentric projection T(x) = E_pi[y | x] approximates the Brenier map: the
gradient of a convex potential, so it has no rotation / shear and keeps the lattice topology (unlike the
Knothe-Rosenblatt map of 003, which is triangular and produced sheared needles).

  T, info = transport_map(rho, ex, ey)          # T: (ny, nx, 2) positions in box units, at the grid cell centres
  uv_new, J = warp_points(uv, T, ex, ey)          # move points given in uv in [0,1]^2; J: (n, 2, 2) Jacobian in box units
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _lse(x, dim):
    return torch.logsumexp(x, dim=dim)


def _grid(n, extent, device):
    return (torch.arange(n, device=device, dtype=torch.float64) + 0.5) / n * extent


def transport_map(rho, ex=1.0, ey=1.0, src=None, eps=None, eps0=None, shrink=0.7, iters_per_stage=12, final_iters=60):
    """rho, src: (ny, nx) non-negative densities (src default uniform). Returns T (ny, nx, 2) as (x, y) in box units
    and a dict with the potentials, eps and the marginal error. float64 internally."""
    dev = rho.device
    ny, nx = rho.shape
    xs, ys = _grid(nx, ex, dev), _grid(ny, ey, dev)
    Cx = (xs[:, None] - xs[None, :]) ** 2            # (nx, nx)
    Cy = (ys[:, None] - ys[None, :]) ** 2            # (ny, ny)
    nu = rho.double().clamp_min(0) + 1e-12
    nu = nu / nu.sum()
    mu = torch.full_like(nu, 1.0 / nu.numel()) if src is None else (src.double().clamp_min(0) + 1e-12)
    mu = mu / mu.sum()
    lmu, lnu = mu.log(), nu.log()
    h = max(ex / nx, ey / ny)
    eps = eps if eps is not None else (1.5 * h) ** 2
    eps0 = eps0 if eps0 is not None else max(ex, ey) ** 2
    f = torch.zeros_like(nu)
    g = torch.zeros_like(nu)

    def soft_min(pot, e):
        # out[a1, a2] = LSE_{b1, b2} [ (pot[b1, b2] - Cy[a1, b1] - Cx[a2, b2]) / e ]
        inner = _lse((pot[:, None, :] - Cx[None, :, :]) / e, dim=2)        # (ny_b, nx_a)
        return _lse(inner[None, :, :] - Cy[:, :, None] / e, dim=1)          # (ny_a, nx_a)

    e = eps0
    while True:
        n_it = final_iters if e <= eps else iters_per_stage
        for _ in range(n_it):
            f = e * lmu - e * soft_min(g, e)
            g = e * lnu - e * soft_min(f, e)
        if e <= eps:
            break
        e = max(eps, e * shrink)
    # barycentric projection: T(x) = sum_y pi(x, y) y / sum_y pi(x, y)
    base = (g[:, None, :] - Cx[None, :, :]) / e                              # (ny_b, nx_a, nx_b)
    inner = _lse(base, dim=2)
    inner_x = _lse(base + torch.log(xs)[None, None, :], dim=2)
    outer = lambda t: _lse(t[None, :, :] - Cy[:, :, None] / e, dim=1)
    lse = outer(inner)
    Tx = torch.exp(outer(inner_x) - lse)
    Ty = torch.exp(_lse(inner[None, :, :] - Cy[:, :, None] / e + torch.log(ys)[None, :, None], dim=1) - lse)
    T = torch.stack([Tx, Ty], -1).float()
    # marginal check: row sums of pi vs mu
    row = torch.exp(f / e + lse)
    info = {"eps": e, "marginal_err": float((row - mu).abs().sum()), "f": f, "g": g}
    return T, info


@torch.no_grad()
def transport_map_batched(rho, ex=1.0, ey=1.0, eps=None, shrink=0.6, iters_per_stage=6, final_iters=30,
                          dtype=torch.float32):
    """Batched version of transport_map for training: rho (B, ny, nx) target densities, uniform source, one shared
    box. Returns T (B, ny, nx, 2). float32 log-domain is accurate enough for layouts (pushforward error ~1-2%)."""
    dev = rho.device
    B, ny, nx = rho.shape
    xs = (torch.arange(nx, device=dev, dtype=dtype) + 0.5) / nx * ex
    ys = (torch.arange(ny, device=dev, dtype=dtype) + 0.5) / ny * ey
    Cx = (xs[:, None] - xs[None, :]) ** 2
    Cy = (ys[:, None] - ys[None, :]) ** 2
    nu = rho.to(dtype).clamp_min(0) + 1e-12
    nu = nu / nu.sum((1, 2), keepdim=True)
    lnu = nu.log()
    lmu = torch.full_like(lnu, -math.log(ny * nx))
    h = max(ex / nx, ey / ny)
    eps = eps if eps is not None else (1.5 * h) ** 2
    f = torch.zeros_like(lnu)
    g = torch.zeros_like(lnu)

    def soft_min(pot, e):     # pot (B, ny, nx) -> LSE_b [(pot(b) - c(a, b)) / e]  (B, ny, nx)
        inner = torch.logsumexp((pot[:, :, None, :] - Cx[None, None]) / e, dim=3)          # (B, ny_b, nx_a)
        return torch.logsumexp(inner[:, None, :, :] - Cy[None, :, :, None] / e, dim=2)      # (B, ny_a, nx_a)

    e = max(ex, ey) ** 2
    while True:
        for _ in range(final_iters if e <= eps else iters_per_stage):
            f = e * lmu - e * soft_min(g, e)
            g = e * lnu - e * soft_min(f, e)
        if e <= eps:
            break
        e = max(eps, e * shrink)
    base = (g[:, :, None, :] - Cx[None, None]) / e                                         # (B, ny_b, nx_a, nx_b)
    inner = torch.logsumexp(base, dim=3)
    inner_x = torch.logsumexp(base + torch.log(xs)[None, None, None, :], dim=3)
    outer = lambda t: torch.logsumexp(t[:, None, :, :] - Cy[None, :, :, None] / e, dim=2)
    lse = outer(inner)
    Tx = torch.exp(outer(inner_x) - lse)
    Ty = torch.exp(torch.logsumexp(inner[:, None, :, :] - Cy[None, :, :, None] / e
                                   + torch.log(ys)[None, None, :, None], dim=2) - lse)
    return torch.stack([Tx, Ty], -1).float()


def transport_map_f32(rho, ex=1.0, ey=1.0):
    """Drop-in for transport_map (same (T, info) return) using the float32 batched solver: ~2.3x faster on GeForce GPUs
    (0.17 s vs 0.40 s on a 128^2 grid); vs the float64 solver the map differs by 0.3-0.7 px on average and the fitted PSNR
    by < 0.05 dB (checked on 4 images x 2 budgets, 2026-10-15)."""
    return transport_map_batched(rho[None], ex, ey)[0], {}


def kr_map(rho, ex=1.0, ey=1.0):
    """Knothe-Rosenblatt (row-first triangular) map from the uniform density to rho on the same grid, in the format
    of transport_map: T (ny, nx, 2) in box units at the grid cell centres. y = F_Y^-1(v), then x = F_{X|y}^-1(u).
    It pushes the uniform density to rho exactly, but its Jacobian is triangular (shear) -- the comparison for OT."""
    ny, nx = rho.shape
    r = rho.double().clamp_min(1e-12)
    r = r / r.sum()
    dev = r.device
    def inv_cdf(w, q):          # w (..., n) cell masses, q (..., m) quantiles in (0,1) -> positions in (0,1)
        c = torch.cumsum(w, -1)
        c = torch.cat([torch.zeros_like(c[..., :1]), c / c[..., -1:]], -1)            # (..., n+1) at cell edges
        n = w.shape[-1]
        idx = torch.searchsorted(c.contiguous(), q.contiguous(), right=True).clamp(1, n)
        c0, c1 = torch.gather(c, -1, idx - 1), torch.gather(c, -1, idx)
        t = (q - c0) / (c1 - c0).clamp_min(1e-15)
        return ((idx - 1).double() + t) / n
    vq = (torch.arange(ny, device=dev, dtype=torch.float64) + 0.5) / ny
    uq = (torch.arange(nx, device=dev, dtype=torch.float64) + 0.5) / nx
    yv = inv_cdf(r.sum(1)[None], vq[None])[0]                                    # (ny,) row positions in (0,1)
    # conditional density of x at each mapped row (linear interpolation between grid rows)
    fy = (yv * ny - 0.5).clamp(0, ny - 1)
    i0 = fy.floor().long().clamp(0, ny - 1)
    i1 = (i0 + 1).clamp(max=ny - 1)
    a = (fy - i0.double())[:, None]
    cond = (1 - a) * r[i0] + a * r[i1]                                           # (ny, nx)
    xu = inv_cdf(cond, uq[None].expand(ny, nx))                                  # (ny, nx)
    T = torch.stack([xu * ex, yv[:, None].expand(ny, nx) * ey], -1)
    return T.float()


def lattice_anchors_ot(img, gy, gx, lam, n_long=64, batched=True):
    """OT-warped lattice anchors for a batch of images (B, 3, H, W): (B, gy, gx, 2) in uv. Density = Image-GS
    importance (|dx| + |dy| of luma, mixed with lam uniform) pooled to the OT grid."""
    B, _, H, W = img.shape
    L = max(H, W)
    ny, nx = ot_grid_shape(H, W, n_long)
    y = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]
    gxm = F.pad((y[:, :, 1:] - y[:, :, :-1]).abs(), (0, 1))
    gym = F.pad((y[:, 1:, :] - y[:, :-1, :]).abs(), (0, 0, 0, 1))
    gpx = (gxm + gym).float()
    gpx = gpx / gpx.sum((1, 2), keepdim=True).clamp_min(1e-12)
    lam_t = torch.as_tensor(lam, device=img.device, dtype=torch.float32).view(-1, 1, 1) if not isinstance(lam, float) \
        else torch.full((B, 1, 1), lam, device=img.device)
    p = lam_t / (H * W) + (1 - lam_t) * gpx
    rho = F.adaptive_avg_pool2d(p[:, None], (ny, nx))[:, 0]
    T = transport_map_batched(rho, W / L, H / L)
    ys = (torch.arange(gy, device=img.device, dtype=torch.float32) + 0.5) / gy
    xs = (torch.arange(gx, device=img.device, dtype=torch.float32) + 0.5) / gx
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    uv0 = torch.stack([xx, yy], -1).reshape(-1, 2)
    out = []
    for b in range(B):
        uv, _ = warp_points(uv0, T[b], W / L, H / L)
        out.append(uv.view(gy, gx, 2))
    return torch.stack(out).float()


def lam_for_spacing(h_px):
    """Uniform share of the importance density by budget (011 test A: 0.1 is best at h >= 8, 0.3 at h = 4)."""
    return float(min(0.3, max(0.1, 0.3 - 0.2 * (math.log2(max(h_px, 1e-6)) - 2.0))))


def warp_points(uv, T, ex=1.0, ey=1.0):
    """uv (n, 2) in [0,1]^2 (x, y) -> new uv (n, 2) and the Jacobian dT/dx (n, 2, 2) in box units, by bilinear
    interpolation of the displacement field and of its finite-difference Jacobian on the grid."""
    ny, nx, _ = T.shape
    dev = T.device
    box = T.new_tensor([ex, ey])
    xs = (torch.arange(nx, device=dev, dtype=T.dtype) + 0.5) / nx * ex
    ys = (torch.arange(ny, device=dev, dtype=T.dtype) + 0.5) / ny * ey
    X = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1)                # (ny, nx, 2)
    D = (T - X).permute(2, 0, 1)[None]                                          # displacement (1, 2, ny, nx)
    # Jacobian of T on the grid (central differences, one-sided at the border)
    dTdx = torch.gradient(T, spacing=(ex / nx,), dim=1)[0]                      # (ny, nx, 2): d(Tx, Ty)/dx
    dTdy = torch.gradient(T, spacing=(ey / ny,), dim=0)[0]
    Jg = torch.stack([dTdx, dTdy], -1).reshape(ny, nx, 4).permute(2, 0, 1)[None]  # [dTx/dx, dTx/dy, dTy/dx, dTy/dy]
    # map uv in [0,1] to grid_sample coordinates; the grid covers the box with cell centres at (i + .5) / n
    gs = (uv * 2 - 1).view(1, 1, -1, 2).to(T.dtype)
    d = F.grid_sample(D, gs, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0].T   # (n, 2)
    j = F.grid_sample(Jg, gs, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0].T  # (n, 4)
    p = uv.to(T.dtype) * box + d
    uv_new = (p / box).clamp(0, 1)
    J = j.view(-1, 2, 2)
    return uv_new, J


def importance_density(img, lam=0.3, blur=2.0, grid=(128, 128)):
    """Image-GS-style importance density lam / |grid| + (1 - lam) * |grad luma| / sum, at the OT grid resolution.
    img (3, H, W) in [0, 1]. Returns (ny, nx)."""
    y = (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2])[None, None]
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=y.dtype, device=y.device).view(1, 1, 3, 3) / 4
    gx = F.conv2d(F.pad(y, (1, 1, 1, 1), mode="replicate"), kx)
    gy = F.conv2d(F.pad(y, (1, 1, 1, 1), mode="replicate"), kx.transpose(-1, -2))
    g = torch.sqrt(gx ** 2 + gy ** 2)
    if blur > 0:
        r = int(math.ceil(3 * blur))
        k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=y.device, dtype=y.dtype) / blur) ** 2)
        k = (k / k.sum()).view(1, 1, 1, -1)
        g = F.conv2d(F.pad(g, (r, r, 0, 0), mode="replicate"), k)
        g = F.conv2d(F.pad(g, (0, 0, r, r), mode="replicate"), k.transpose(-1, -2))
    g = F.adaptive_avg_pool2d(g, grid)[0, 0]
    g = g / g.sum().clamp_min(1e-12)
    return lam / g.numel() + (1 - lam) * g


def ot_grid_shape(H, W, n_long=128):
    L = max(H, W)
    return max(8, round(n_long * H / L)), max(8, round(n_long * W / L))
