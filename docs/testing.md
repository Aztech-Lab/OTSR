# Testing

Three layers, from the specification down to the browser:

1. **Python reference** (`reference/otsr_ref/`): a dense, float64-capable PyTorch implementation of the model, the
   fit, the OT initialisation, the edge pass and the file format. It is the specification; `reference/make_golden.py`
   runs it on deterministic synthetic inputs and writes the results to `tests/golden/` (one `.json` manifest + one
   `.bin` of raw arrays per case, 2.7 MB in total, committed).
2. **Vitest suites**:
   - `tests/unit/` (`npm test`): CPU-only — the constants against the reference's `spec.json`, Python-style
     rounding, the `.otsr` codec against the Python-written sample.
   - `tests/gpu/` (`npm run test:gpu`): the WGSL kernels on a real GPU, in Node, through Dawn (`webgpu` package),
     against the golden data. One worker, files in sequence.
3. **Browser smoke test** (`npm run build && npm run smoke`): drives the built app in headless Chrome through the
   `window.otsr` hook — loads a synthetic image, fits, exports, snapshots the three view modes — and writes the
   snapshots to `tools/out/`.

## What the GPU tests prove (and the accuracy measured on an RTX 5070 Ti, 2026-10-25)

| test | checks | tolerance | measured |
|---|---|---|---|
| `forward.test.ts` | projection terms (centre, conic, weight) of 900 random Gaussians on a 61 × 45 raster, incl. sub-pixel and off-screen ones | rel 2e-6 | ≤ 1.1e-6 |
| | the render (rgb) and D against the float64 dense reference | 2e-5 abs + 2e-6 rel | ≤ 1.2e-6 abs |
| | the MSE loss | rel 1e-6 | 1e-7 |
| | all nine gradient columns against float64 autograd | rel 1e-4 of the column's max | ≤ 1.4e-6 |
| | a zoomed, offset view (2.5×, fractional offsets) | 2e-5 abs | ≤ 1.1e-6 |
| | tile coverage: Σ gcount equals the total entries | exact | exact |
| `adam.test.ts` | three Adam steps with the cosine schedule, bias correction and all clamps, vs `torch.optim.Adam` | 2e-6 abs params, 1e-5 rel moments | ≤ 4.8e-7, 9e-7 |
| `fit.test.ts` | the whole loop, 40 steps from the OT init of a 48 × 40 image: PSNR curve and final parameters | 0.02 dB, rel 2e-4 | identical to 4 decimals, ≤ 1.7e-5 |
| `ot.test.ts` | pooled density; the transport map vs the **float64** solution (must be as close as the float32 torch map); uv, scales, covariance, colours with tolerances derived from the map noise (colour via the bilinear interpolant's Lipschitz bound) | see file | map: 1.60e-3 max / 2.3e-4 rms (torch float32: 1.59e-3 / 2.3e-4) |
| `edges.test.ts` | the edge pass at S = 4 / 2.5 / 8 with three edge widths on a 208 × 160 image | 2e-6 abs | ≤ 8e-7 |
| `export.test.ts` | tiled export (256-px tiles with overlap) equals an untiled render with edges; nearest / bilinear / bicubic reproduce the source at zoom 1 and known blends at zoom 2 | 1 (8-bit), 1e-6 | 1, 0 |

Every test prints its statistics (`max |d|`, rms, `max |ref|`, relative error) so a passing run documents the
accuracy; run with `npx vitest run --project gpu --silent=false` to see them.

## Regenerating the golden data

```bash
python reference/make_golden.py     # any Python 3.10+ with torch >= 2.0 and numpy; CPU, ~10 s
```

The cases are deterministic (fixed seeds, CPU tensors). Regenerate after any change to `reference/otsr_ref/` and
commit the result together with the engine change that motivated it.

## Choosing the GPU for the tests

Dawn picks the system's high-performance adapter. `OTSR_DAWN_FLAGS="adapter=<name part>"` selects another one
(`node tools/dawn-probe.mjs adapter=list` prints the available adapters), e.g. `OTSR_DAWN_FLAGS=adapter=4090` on a
machine with several NVIDIA cards. The tests are accuracy tests and run in a second or two on any GPU; timing
numbers belong in performance.md and are taken on one fixed card.

## Headless Chrome caveats

The smoke test launches Chrome with `--enable-unsafe-webgpu --ignore-gpu-blocklist`. Headless Chrome may use the
integrated GPU, sometimes fails the first `requestAdapter` (the app retries), and does not composite WebGPU canvases
into screenshots or `drawImage` — which is why the test checks the view through `viewer.snapshot()` (the same passes
rendered into an offscreen texture) rather than through pixels of the page. Pass `--headful` to watch it.
