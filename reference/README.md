# Python reference

`otsr_ref/` is the executable specification of the engine: a dense PyTorch implementation that is easy to read and
can run in float64. It has no dependency on the research repository; `upstream/` keeps unmodified copies of the
research files it was derived from, for provenance (see `upstream/README.md`).

| module | contents |
|---|---|
| `spec.py` | every constant of the model, the learning-rate schedule, the transport-grid size |
| `render.py` | the image function (dense, O(N × pixels)), views (zoom / offset), loss, PSNR |
| `fit.py` | Adam with per-parameter learning rates, cosine decay and the clamps |
| `init.py` | the optimal-transport initialisation: densities, Sinkhorn (float32 or float64), warp, eigendecomposition, colours |
| `edges.py` | the distance-field edge pass |
| `fileformat.py` | `.otsr` reader / writer (numpy) |
| `goldenio.py` | the golden-case container used by `make_golden.py` |

Scripts:

- `make_golden.py` — writes `tests/golden/` (deterministic, CPU, about 10 s). Run it after changing anything here.
- `dump_image.py image.png image.rgb.bin` — converts an image to the raw float layout `tools/bench.mjs` reads.

Requirements: Python 3.10+, `torch >= 2.0`, `numpy`; `Pillow` for `dump_image.py`. A CPU-only PyTorch is enough.

Example — load a model saved by the web app and render it at 3×:

```python
import torch
from otsr_ref import fileformat, render, edges

p, W, H, meta = fileformat.load("model.otsr")
params = {k: torch.from_numpy(v) for k, v in p.items()}
img = render.render(params, H, W, out_h=3 * H, out_w=3 * W, zoom=3.0)       # (3, 3H, 3W), float32
sharp = edges.edge_field(img.clamp(0, 1), S=3.0)                              # the distance-field edge pass
```
