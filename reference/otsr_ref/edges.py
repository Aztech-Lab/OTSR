"""v3 distance-field edges (paper/ot_workshop/fig_todo/sr_remap.py ``edge_field``), unchanged.

Same principle as distance-field glyph magnification (Green, SIGGRAPH 2007 Talks): inside a window of radius
~ 0.75 S, take the actual colours D and B of the darkest and brightest pixel (luma); t = (Y - Ymin) / (Ymax - Ymin) is a
locally normalised distance field whose 0.5 level is the edge. alpha = smoothstep(1/2 - w, 1/2 + w, t) with
w = 1/2 edge_px |grad t| (the change of t per output pixel, i.e. fwidth), so the transition is edge_px output pixels
wide at every magnification. Output = D + (B - D) alpha, gated by the local luma range so smooth regions are kept.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .spec import EDGE_C0, EDGE_C1, EDGE_PX, EDGE_WINDOW, LUMA


def smoothstep(e0, e1, x):
    u = ((x - e0) / (e1 - e0)).clamp(0, 1)
    return u * u * (3 - 2 * u)


def gate(c, c0, c1):
    return smoothstep(torch.as_tensor(c0), torch.as_tensor(max(c1, c0 + 1e-6)), c)


def window_radius(S: float, window: float = EDGE_WINDOW) -> int:
    return max(1, int(round(S * window)))


def edge_field(img, S: float, edge_px: float = EDGE_PX, window: float = EDGE_WINDOW, c0: float = EDGE_C0,
               c1: float = EDGE_C1):
    """img: (3, H, W) in [0, 1] at the S x output resolution. Edges become edge_px output pixels wide."""
    r = window_radius(S, window)
    _, H, W = img.shape
    y = (img * img.new_tensor(LUMA).view(3, 1, 1)).sum(0, keepdim=True)[None]          # (1, 1, H, W)
    ymax, imax = F.max_pool2d(y, 2 * r + 1, 1, r, return_indices=True)
    ymin, imin = F.max_pool2d(-y, 2 * r + 1, 1, r, return_indices=True)
    ymin = -ymin
    flat = img.reshape(3, -1)
    B = flat[:, imax.view(-1)].view(3, H, W)                                          # colour of the brightest point
    D = flat[:, imin.view(-1)].view(3, H, W)                                          # colour of the darkest point
    rng = (ymax - ymin)[0]
    t = ((y[0] - ymin[0]) / (rng + 1e-6)).clamp(0, 1)
    gx = torch.zeros_like(t)
    gy = torch.zeros_like(t)
    gx[..., 1:-1] = (t[..., 2:] - t[..., :-2]) / 2
    gy[..., 1:-1, :] = (t[..., 2:, :] - t[..., :-2, :]) / 2
    w = (0.5 * edge_px * (gx.abs() + gy.abs())).clamp(1e-3, 0.5)
    a = smoothstep(0.5 - w, 0.5 + w, t)
    sharp = D + (B - D) * a
    return (img + gate(rng, c0, c1) * (sharp - img)).clamp(0, 1)
