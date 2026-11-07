"""Per-image fitting: Agent/cc/common/gauss.py ``fit`` (wnorm, rotation on, MSE) with the OTSR image function."""
from __future__ import annotations

import math

import torch

from .render import mse, psnr, render
from .spec import ADAM_EPS, BETAS, LOG_S_MAX, LOG_S_MIN_PX, LOG_W_CLAMP, LR, STEPS, UV_CLAMP, lr_scale

GROUP_ORDER = ("uv", "log_s", "color", "rot", "log_w")     # the param-group order of the research optimiser


def make_optimizer(p: dict, L: int):
    groups = [{"params": [p[k]], "lr": LR[k] / L if k == "uv" else LR[k]} for k in GROUP_ORDER]
    return torch.optim.Adam(groups, betas=BETAS, eps=ADAM_EPS)


@torch.no_grad()
def clamp_(p: dict, L: int):
    p["uv"].clamp_(*UV_CLAMP)
    p["log_s"].clamp_(math.log(LOG_S_MIN_PX / L), LOG_S_MAX)
    p["log_w"].clamp_(*LOG_W_CLAMP)


def fit(gt, p0: dict, steps: int = STEPS, log_every: int = 0, callback=None):
    """Adam with per-group learning rates and cosine decay. Returns (params, curve). ``steps`` optimiser steps are
    taken; the curve logs PSNR of the clamped render before step 0, every ``log_every`` steps and after the last."""
    _, H, W = gt.shape
    L = max(H, W)
    p = {k: v.detach().clone().requires_grad_(True) for k, v in p0.items()}
    opt = make_optimizer(p, L)
    base = [g["lr"] for g in opt.param_groups]
    curve = []
    for it in range(steps + 1):
        k = lr_scale(it, steps)
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * k
        out = render(p, H, W)
        loss = mse(out, gt)
        if (log_every and it % log_every == 0) or it == steps:
            curve.append({"step": it, "psnr": psnr(out, gt), "loss": float(loss)})
            if callback:
                callback(it, p, out)
        if it == steps:
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        clamp_(p, L)
    return {k: v.detach() for k, v in p.items()}, curve
