# Upstream research code (verbatim snapshot)

These files are **unmodified copies** of the research code that OTSR is ported from.
They are kept for provenance: every formula in the WebGPU engine and in
`reference/otsr_ref/` can be traced back to a line here.

They are **not meant to run from this folder**. They import modules that are not part
of OTSR (the CUDA rasteriser `gmod/`, `common.data`, `protocol15`, ...). The runnable,
self-contained reference is [`../otsr_ref/`](../otsr_ref/).

Snapshot taken 2026-10-25 from `E:\Projekt\GaussianEncoder\Agent\cc\` (copied with
`cp -p`, so the modification times below are the originals).

| File | Original path (under `Agent/cc/`) | sha256 (first 16) | Modified | What OTSR takes from it |
|---|---|---|---|---|
| `common/render.py` | `common/render.py` | `fbeaa020e8408c2e` | 2026-10-18 02:08 | pixel-centre convention, conic from (scale, rotation), weighted-normalised (`wnorm`) blend, eps = 1e-4 |
| `common/gauss.py` | `common/gauss.py` | `6d86e1e1d41f4d55` | 2026-10-13 11:26 | lattice dimensions, colour sampling, the fitting loop (Adam, per-group learning rates, cosine decay, clamps) |
| `common/ot.py` | `common/ot.py` | `c3a69e02dc1763b1` | 2026-10-15 19:13 | float32 log-domain Sinkhorn with separable cost (`transport_map_batched`), `warp_points`, `lam_for_spacing`, `ot_grid_shape` |
| `exp/011_ot_layout/ot_layout_test.py` | `exp/011_ot_layout/ot_layout_test.py` | `4a9af0197e8443b2` | 2026-10-15 05:44 | `pixel_density`, `pooled_density`, `cov_to_scale_rot`, `ot_params_from_map`, `residual_monitor` |
| `exp/015_revision/run_ours15.py` | `exp/015_revision/run_ours15.py` | `f9dda5f0a237b611` | 2026-10-18 02:13 | `make_init` (the `"ot"` branch) |
| `paper/ot_workshop/fig_todo/sr_mres_tau.py` | `paper/ot_workshop/fig_todo/sr_mres_tau.py` | `0a8d71a851151de9` | 2026-10-24 21:26 | `base_fit`: the exact SR recipe (0.1 Gaussians per pixel, `n_long = min(256, max(128, round(L/4)))`, 3000 steps) |
| `paper/ot_workshop/fig_todo/sr_hardness.py` | `paper/ot_workshop/fig_todo/sr_hardness.py` | `dbb38aee3726bb6c` | 2026-10-24 10:29 | `render_t`: how the research rendered at S x (5 sigma culling) |
| `paper/ot_workshop/fig_todo/sr_remap.py` | `paper/ot_workshop/fig_todo/sr_remap.py` | `9d004b057bb756fd` | 2026-10-25 00:50 | v3 distance-field edges (`edge_field`) |

## Where OTSR deliberately differs

The research renderer used the gsplat CUDA fork, which culls per 16 x 16 tile at
`ceil(3 sigma)`: a Gaussian contributes its full `exp(-q)` to every pixel of every tile
its 3-sigma box touches. That support depends on the output resolution and is
discontinuous at tile borders. OTSR instead uses one resolution-independent kernel
for fitting **and** display (see `docs/method.md`):

```
alpha(q) = max(exp(-q) - exp(-q_c), 0),   q = 0.5 * d^T Sigma^-1 d,   q_c = 4^2 / 2 = 8
```

i.e. a Gaussian truncated at 4 sigma (Mahalanobis) and shifted down by exp(-8) = 3.4e-4
so that it reaches zero continuously. Everything else (parameterisation, blend, eps,
optimiser, schedule, clamps, OT initialisation, edge pass) follows these files exactly.
