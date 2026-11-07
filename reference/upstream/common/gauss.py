"""Gaussian sets: initialisation, parameterisation and per-image optimisation (fitting).

Unconstrained parameter dict (one image):
    uv      (N,2)  centre in [0,1]^2 (clamped to a small margin outside the frame at render time)
    log_s   (N,2)  log std-dev in long-side units
    rot     (N,1)  radians (kernel convention, see render.py)
    color   (N,3)
    log_w   (N,1)  log weight (wnorm only)
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from .render import render_one


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


def sample_color(gt, uv):
    """Bilinear sample of gt (3,H,W) at uv (N,2) -> (N,3)."""
    grid = (uv * 2 - 1).view(1, 1, -1, 2)
    return F.grid_sample(gt[None], grid, mode="bilinear", align_corners=False, padding_mode="border")[0, :, 0].T


def init_params(gt, N: int, mode: str, how: str = "lattice", lam: float = 0.3, sigma_rel: float = 0.6, seed=0):
    """lattice: uniform grid; importance: Image-GS style sampling p ~ lam*uniform + (1-lam)*|grad I|."""
    _, H, W = gt.shape
    dev = gt.device
    L = max(H, W)
    if how == "lattice":
        gy, gx = lattice_dims(N, H, W)
        uv = lattice_uv(gy, gx, dev)
        N = uv.shape[0]
        spacing_px = torch.full((N,), math.sqrt(H * W / N), device=dev)
    elif how == "importance":
        y = (0.299 * gt[0] + 0.587 * gt[1] + 0.114 * gt[2])
        gxm = F.pad((y[:, 1:] - y[:, :-1]).abs(), (0, 1))
        gym = F.pad((y[1:, :] - y[:-1, :]).abs(), (0, 0, 0, 1))
        g = gxm + gym
        p = lam / g.numel() + (1 - lam) * g / g.sum().clamp_min(1e-12)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.multinomial(p.flatten().cpu(), N, replacement=False, generator=gen).to(dev)
        r, c = idx // W, idx % W
        jit = torch.rand(N, 2, generator=gen).to(dev)
        uv = torch.stack([(c.float() + jit[:, 0]) / W, (r.float() + jit[:, 1]) / H], -1)
        spacing_px = 1.0 / torch.sqrt(N * p.flatten()[idx])
    else:
        raise ValueError(how)
    s = (sigma_rel * spacing_px / L).clamp(min=0.35 / L).unsqueeze(1).repeat(1, 2)
    color = sample_color(gt, uv)
    if mode == "sum":  # sum of unit-peak Gaussians: local coverage ~ 2*pi*sigma^2/spacing^2
        cover = 2 * math.pi * (sigma_rel ** 2)
        color = color / cover
    return {
        "uv": uv.clone(), "log_s": torch.log(s), "rot": torch.zeros(N, 1, device=dev),
        "color": color.clone(), "log_w": torch.zeros(N, 1, device=dev),
    }


def from_encoder(g):
    """Encoder output dict (uv, scale, rot[(N,2) doubled-angle or (N,1)], color, weight) -> fit parameters."""
    return {"uv": g["uv"].float(), "log_s": torch.log(g["scale"].float().clamp_min(1e-6)),
            "rot": g["rot"].float(), "color": g["color"].float(),
            "log_w": torch.log(g["weight"].float().clamp_min(1e-8)) if g.get("weight") is not None
            else torch.zeros_like(g["uv"][:, :1])}


def render_params(p, H, W, mode, aa_px=0.0, use_rot=True):
    if use_rot:
        rot = p["rot"]
    else:  # axis-aligned: angle 0 (or the doubled-angle vector (1, 0))
        rot = torch.zeros_like(p["rot"]) if p["rot"].shape[-1] == 1 else torch.cat(
            [torch.ones_like(p["rot"][:, :1]), torch.zeros_like(p["rot"][:, :1])], 1)
    return render_one(p["uv"], torch.exp(p["log_s"]), rot, p["color"], H, W, mode=mode,
                      weight=torch.exp(p["log_w"]) if mode == "wnorm" else None, aa_px=aa_px)


def fit(gt, p0, mode="wnorm", steps=2000, use_rot=True, lr_px=0.25, lr_logs=0.02, lr_rot=0.02, lr_col=0.01,
        lr_logw=0.02, final_lr_frac=0.05, loss="mse", log_every=100, aa_px=0.0, callback=None):
    """Adam on all parameters with cosine decay. Returns (params, curve, seconds)."""
    _, H, W = gt.shape
    L = max(H, W)
    p = {k: v.detach().clone().requires_grad_(True) for k, v in p0.items()}
    groups = [
        {"params": [p["uv"]], "lr": lr_px / L},
        {"params": [p["log_s"]], "lr": lr_logs},
        {"params": [p["color"]], "lr": lr_col},
    ]
    if use_rot:
        groups.append({"params": [p["rot"]], "lr": lr_rot})
    if mode == "wnorm":
        groups.append({"params": [p["log_w"]], "lr": lr_logw})
    opt = torch.optim.Adam(groups, betas=(0.9, 0.99))
    base = [g["lr"] for g in opt.param_groups]
    curve = []
    torch.cuda.synchronize()
    t0 = time.time()
    for it in range(steps + 1):
        frac = it / max(1, steps)
        k = final_lr_frac + (1 - final_lr_frac) * 0.5 * (1 + math.cos(math.pi * frac))
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * k
        out = render_params(p, H, W, mode, aa_px=aa_px, use_rot=use_rot)
        l = F.mse_loss(out, gt) if loss == "mse" else F.l1_loss(out, gt)
        if it % log_every == 0 or it == steps:
            with torch.no_grad():
                m = F.mse_loss(out.clamp(0, 1), gt).item()
            curve.append({"step": it, "psnr": -10 * math.log10(max(m, 1e-12)), "time_s": time.time() - t0})
            if callback:
                callback(it, p, out)
        if it == steps:
            break
        opt.zero_grad(set_to_none=True)
        l.backward()
        opt.step()
        with torch.no_grad():
            p["uv"].clamp_(-0.02, 1.02)
            p["log_s"].clamp_(math.log(0.2 / L), math.log(0.5))
            p["log_w"].clamp_(-8, 8)
    torch.cuda.synchronize()
    return {k: v.detach() for k, v in p.items()}, curve, time.time() - t0
