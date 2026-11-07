# OTSR — optimal-transport Gaussian upscaling in the browser

OTSR fits an image with a lattice of anisotropic 2-D Gaussians laid out by optimal transport, and renders that
vector model at **any** resolution. Everything — the initialisation, the 3000-step gradient fit, the interactive
viewer and the export — runs on your own GPU through [WebGPU](https://www.w3.org/TR/webgpu/). No server, no
upload, no neural network weights: a 2048 × 2048 anime frame becomes 419 k Gaussians in about 6 s on an RTX 4090
and can then be viewed or exported at 2×, 4×, 8× … with edges that stay as sharp as the original.

The method comes from our workshop paper on optimal-transport Gaussian lattices (see [CITATION.md](CITATION.md));
this repository is the self-contained, browser-side implementation of it.

## What it does

1. **Optimal-transport initialisation** — an edge-importance density of the image is computed, an entropic OT map
   (Sinkhorn, on the GPU) moves a uniform lattice onto it, and every lattice cell, deformed by the map's Jacobian,
   becomes one Gaussian: dense and elongated along edges, sparse and round in flat regions.
2. **Fitting** — Adam on all 9 parameters of every Gaussian (position, two log-scales, rotation, colour, log-weight)
   against the image, with the exact gradients of the weighted-normalised Gaussian blend, 3000 steps by default.
3. **Resolution-independent rendering** — the model is a continuous function; the viewer evaluates it at any zoom and
   the exporter at any scale, tile by tile.
4. **Distance-field edges** — at magnification the Gaussian render is run through the same trick that keeps vector
   fonts sharp (Green, SIGGRAPH 2007): a locally normalised luminance field is re-thresholded with a transition that
   is a fixed number of *output* pixels wide, so edges stay crisp at 8× while gradients and shading are untouched.

## Quick start (users)

Requirements: a browser with WebGPU — Chrome 113+ / Edge 113+ on Windows, macOS, Linux (Chrome 121+) and
Android; Safari 26+ on macOS / iOS — and a GPU. Integrated GPUs work; a discrete GPU is much faster.

1. Download a release build (the `dist/` folder) or build it yourself (below).
2. Serve it over HTTP — browsers do not run WebGPU modules from `file://`:
   - Windows: double-click `serve.bat` (needs [Node.js](https://nodejs.org) 20+), or
   - any OS: `node tools/serve.mjs`, or `python -m http.server -d dist 8787`, or any static web server.
3. Open the printed address, drop an image onto the view, pick a quality (Fast / Standard / Best) and press
   **Fit & Enhance**. Drag the divider to compare with bicubic interpolation, zoom with the wheel, pan by dragging.
   The **Research** mode exposes every parameter and the analysis tools.
4. **Export PNG** at the chosen scale, or **Save .otsr** to keep the model (a few MB; re-renderable at any size
   later with **Load .otsr**).

See [docs/getting-started.md](docs/getting-started.md) for the details and
[docs/troubleshooting.md](docs/troubleshooting.md) if WebGPU does not start (dual-GPU laptops in particular).

## Quick start (developers)

```bash
npm install            # TypeScript, Vite, Vitest, Dawn's WebGPU for Node (tests), puppeteer-core (smoke test)
npm run dev            # live-reload dev server
npm run build          # type-check + production build into dist/ (+ docs)
npm test               # unit tests (CPU)
npm run test:gpu       # kernel tests on a real GPU in Node, against the Python reference
npm run smoke          # builds must exist: drives the real UI in headless Chrome
```

The Python reference (`reference/`) is the executable specification: it regenerates the golden test data with
`npm run golden` (needs PyTorch ≥ 2.0, CPU is enough). See [docs/development.md](docs/development.md) and
[docs/testing.md](docs/testing.md).

## Repository layout

```
index.html, src/app/      the web UI (viewer, controls, export)
src/engine/               the WebGPU engine: TypeScript orchestration + WGSL kernels (src/engine/shaders/)
src/core/                 CPU-side model code: constants, OT initialisation maths, the .otsr file format
reference/otsr_ref/       Python reference implementation (PyTorch, dense, float64-capable)
reference/upstream/       unmodified snapshot of the research code the reference was derived from
tests/unit, tests/gpu     Vitest suites; tests/golden holds the reference outputs they check against
tools/                    serve, smoke test, benchmark, WebGPU probes
docs/                     documentation (method, engine, file format, testing, performance, troubleshooting)
```

## Performance

Measured on an RTX 4090 (Windows, D3D12, Dawn in Node) with the default settings, 2048 × 2048 image,
0.1 Gaussians per pixel: initialisation 0.5 s, fit 5.3 s (565 steps/s, 52.7 dB), 4× export (8192 × 8192, with
edges) 1.3 s. Details and the memory model in [docs/performance.md](docs/performance.md).

## Documentation

[docs/index.md](docs/index.md) — getting started, the method, the engine, the `.otsr` format, testing,
development, performance, troubleshooting — and [docs/tutorial-porting.md](docs/tutorial-porting.md), the
step-by-step record of how the research code was turned into this engine (derivations, decisions, pitfalls).

## Acknowledgements

The WebGPU trainer of [SAD (Soft Anisotropic Diagrams)](https://github.com/luckyiyi/SAD) was a helpful reference for
structuring a browser-side fitting loop; the GPU tests run on [Dawn's Node bindings](https://github.com/dawn-gpu/node-webgpu).

## License

MIT — see [LICENSE](LICENSE).
