"""The OTSR model specification: every constant shared by this reference and the WebGPU engine.

The TypeScript twin of this file is ``src/core/spec.ts``; keep the two in sync (``tests/unit/spec.test.ts`` checks
the values against ``tests/golden/spec.json``, which ``make_golden.py`` writes from this module).
"""
from __future__ import annotations

import math

# --- image function -------------------------------------------------------------------------------------------
K_SIGMA = 4.0                         # support radius in Mahalanobis units
Q_CUT = 0.5 * K_SIGMA ** 2            # q = 0.5 d^T Sigma^-1 d is cut at 8
ALPHA_SHIFT = math.exp(-Q_CUT)        # alpha(q) = max(exp(-q) - exp(-Q_CUT), 0): continuous at the cut
EPS = 1e-4                            # out = sum w a c / (sum w a + EPS)
TILE = 16                             # tile edge in output pixels (engine only; the image function has no tiles)

# --- parameter layout: 9 floats per Gaussian, in this order ----------------------------------------------------
PARAM_LAYOUT = (("uv", 2), ("log_s", 2), ("rot", 1), ("color", 3), ("log_w", 1))
FLOATS_PER_GAUSSIAN = sum(n for _, n in PARAM_LAYOUT)

# --- fitting (Agent/cc/common/gauss.py: fit) -------------------------------------------------------------------
STEPS = 3000
LR = {"uv": 0.25, "log_s": 0.02, "rot": 0.02, "color": 0.01, "log_w": 0.02}   # uv: in pixels of the long side
FINAL_LR_FRAC = 0.05                  # cosine decay from 1 to 0.05
BETAS = (0.9, 0.99)
ADAM_EPS = 1e-8                       # torch.optim.Adam default
UV_CLAMP = (-0.02, 1.02)
LOG_S_MIN_PX = 0.2                    # log_s in [log(0.2 / L), log(0.5)]
LOG_S_MAX = math.log(0.5)
LOG_W_CLAMP = (-8.0, 8.0)

# --- initialisation (exp/011 ot_layout_test.py, exp/015 run_ours15.py: make_init "ot") ------------------------
K_PER_PX = 0.1                        # default budget: 0.1 Gaussians per pixel
SIGMA_REL = 0.6                       # sigma = 0.6 x lattice spacing, deformed by the OT Jacobian
MIN_SCALE_PX = 0.35                   # initial scales are clamped to >= 0.35 px
SINKHORN_SHRINK = 0.6
SINKHORN_ITERS_PER_STAGE = 6
SINKHORN_FINAL_ITERS = 30
LUMA = (0.299, 0.587, 0.114)

# --- v3 distance-field edges (paper/ot_workshop/fig_todo/sr_remap.py: edge_field) -----------------------------
EDGE_PX = 1.4                         # transition width in output pixels (smaller = harder)
EDGE_WINDOW = 0.75                    # window radius = max(1, round(0.75 * S))
EDGE_C0 = 0.08                        # local luma range where the edge pass starts ...
EDGE_C1 = 0.2                         # ... and where it is fully on


def lr_scale(it: int, steps: int) -> float:
    """Cosine factor applied to every base learning rate at iteration ``it`` (set before the step, as in fit)."""
    frac = it / max(1, steps)
    return FINAL_LR_FRAC + (1 - FINAL_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * frac))


def n_long_for(H: int, W: int) -> int:
    """Transport-grid cells on the long side (the SR recipe of sr_mres_tau.base_fit)."""
    return min(256, max(128, round(max(H, W) / 4)))


def as_dict() -> dict:
    """Every constant above, for tests/golden/spec.json."""
    return {k: v for k, v in globals().items() if k.isupper()}
