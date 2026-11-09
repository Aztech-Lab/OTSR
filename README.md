<h1 align="center">OTSR</h1>

<p align="center"><b>Optimal-transport Gaussian upscaling, in your browser, on your GPU.</b></p>

<p align="center">
  <a href="https://aztech-lab.github.io/OTSR/"><b>▶&nbsp; Open OTSR — aztech-lab.github.io/OTSR</b></a>
</p>

<p align="center">
  <a href="https://aztech-lab.github.io/OTSR/">Live app</a> ·
  <a href="https://aztech-lab.github.io/OTSR/report.html">Technical report</a> ·
  <a href="docs/getting-started.md">Getting started</a> ·
  <a href="docs/method.md">The method</a> ·
  <a href="CHANGELOG.md">Changelog</a> ·
  <a href="CITATION.md">Cite</a>
</p>

<p align="center">
  <img alt="version 0.1.4" src="https://img.shields.io/badge/version-0.1.4-c47a3a">
  <img alt="MIT" src="https://img.shields.io/badge/license-MIT-2b2b2b">
  <img alt="WebGPU" src="https://img.shields.io/badge/runs%20on-WebGPU-2b2b2b">
</p>

---

OTSR turns a raster image into a mesh of anisotropic 2-D Gaussians laid out by optimal transport, then renders that
vector model at **any** resolution. Nothing leaves your machine: the initialisation, the gradient fit, the viewer and
the export all run on your own GPU through [WebGPU](https://www.w3.org/TR/webgpu/). No server, no upload, no neural
network weights. A 2048 × 2048 anime frame becomes 419 k Gaussians in about 6 s on an RTX 4090 and can then be viewed
or exported at 2×, 4×, 8× … with edges as sharp as the original.

The method is from our workshop paper on placing Gaussians by optimal transport (see [CITATION.md](CITATION.md)); this
repository is its self-contained, browser-side implementation, by [Aztech Labs](https://github.com/Aztech-Lab).

## Try it

**[aztech-lab.github.io/OTSR](https://aztech-lab.github.io/OTSR/)** — open it in Chrome or Edge 113+ (Windows, macOS,
Linux, Android) or Safari 26+ (macOS, iOS), on a machine with a GPU. Integrated GPUs and phones work; a discrete GPU
is much faster. Then:

1. Press **Load sample**, or **Open image** / drop / paste an image of your own.
2. Pick a preset (Fast / Standard / Best) and press **Enhance**. While it runs the stage plays the real process: the
   optimal-transport mesh settling on the image through every Sinkhorn stage, the Gaussians growing in their id
   colours, a scan line revealing the result; the monitor on the right follows the three regions with the strongest
   gradients.
3. Drag the divider to compare against bicubic, zoom with the wheel, pan by dragging; the **View** menu switches the
   vector view, the OT mesh, the 1σ outlines and the error map.
4. **Export** as PNG, JPEG, WebP, AVIF or BMP at 1×–16× or any scale, or **Save .otsr** to keep the model (a few MB,
   re-renderable at any size later with **Load .otsr**).

On a phone the console is a sheet at the bottom (tap its title bar to open or fold it), two fingers zoom, and the
rest is the same. The **[technical report](https://aztech-lab.github.io/OTSR/report.html)** tells the method of the paper behind
the app the way a magazine would, with nine live demonstrations, what the app does, how the engine works, and a
reading list.

**Research** mode opens every parameter (Gaussians per pixel, steps, edge width, L1 / L2 loss), the split comparison
against any resampler, the region tool and the session information with the PSNR curve. Everything has a keyboard
shortcut (press **?**) and every label is in English and Traditional Chinese.

Details in [docs/getting-started.md](docs/getting-started.md); if WebGPU does not start, see
[docs/troubleshooting.md](docs/troubleshooting.md) (dual-GPU laptops in particular).

## How it works

1. **Optimal-transport initialisation** — an edge-importance density of the image is computed, an entropic OT map
   (Sinkhorn, on the GPU) moves a uniform mesh onto it, and every mesh cell, deformed by the map's Jacobian,
   becomes one Gaussian: dense and elongated along edges, sparse and round in flat regions.
2. **Fitting** — Adam on all 9 parameters of every Gaussian (position, two log-scales, rotation, colour, log-weight)
   against the image, with the exact gradients of the weighted-normalised Gaussian blend; 3000 steps by default.
3. **Resolution-independent rendering** — the model is a continuous function; the viewer evaluates it at any zoom and
   the exporter at any scale, tile by tile.
4. **Distance-field edges** — at magnification the Gaussian render goes through the trick that keeps vector fonts
   sharp (Green, SIGGRAPH 2007): a locally normalised luminance field is re-thresholded with a transition a fixed
   number of *output* pixels wide, so edges stay crisp at 8× while gradients and shading are untouched.

The full derivation is in [docs/method.md](docs/method.md) and the engine in [docs/engine.md](docs/engine.md).

## Performance

RTX 4090 (Windows, D3D12), default settings, 2048 × 2048 image, 0.1 Gaussians per pixel:

| Stage | Time |
|---|---|
| OT initialisation | 0.5 s |
| Fit, 3000 steps (565 steps/s, 52.7 dB) | 5.3 s |
| 4× export, 8192 × 8192 with edges | 1.3 s |

The memory model and more measurements are in [docs/performance.md](docs/performance.md).

## Run it yourself

Browsers do not run WebGPU modules from `file://`, so a release build has to be served over HTTP:

- Windows: double-click `serve.bat` (needs [Node.js](https://nodejs.org) 20+), or
- any OS: `node tools/serve.mjs`, or `python -m http.server -d dist 8787`, or any static web server.

To build from source:

```bash
npm install            # TypeScript, Vite, Vitest, Dawn's WebGPU for Node (tests), puppeteer-core (smoke test)
npm run dev            # live-reload dev server
npm run build          # type-check + production build into dist/ (+ docs)
npm test               # unit tests (CPU)
npm run test:gpu       # kernel tests on a real GPU in Node, against the Python reference
npm run smoke          # drives the real UI in headless Chrome (needs a build)
```

The Python reference (`reference/`) is the executable specification: it regenerates the golden test data with
`npm run golden` (PyTorch ≥ 2.0, CPU is enough). See [docs/development.md](docs/development.md) and
[docs/testing.md](docs/testing.md).

## Repository layout

```
index.html, src/app/      the web UI (viewer, panels, monitor, export); report.html, src/report.*: the technical report
src/engine/               the WebGPU engine: TypeScript orchestration + WGSL kernels (src/engine/shaders/)
src/core/                 CPU-side model code: constants, OT initialisation maths, the .otsr file format
reference/otsr_ref/       Python reference implementation (PyTorch, dense, float64-capable)
reference/upstream/       unmodified snapshot of the research code the reference was derived from
tests/unit, tests/gpu     Vitest suites; tests/golden holds the reference outputs they check against
tools/                    serve, smoke test, benchmark, WebGPU probes
public/samples/           the bundled sample image
docs/                     documentation (method, engine, file format, testing, performance, troubleshooting)
```

## Versions

| Version | What it brought |
|---|---|
| **0.1.4** | The **technical report**, rewritten as a magazine would tell it: the method of the paper in plain words, the pipeline run live under Figure A, nine live demonstrations, the results in prose, a reading list of 45 entries; one name, *mesh*, for the transported grid everywhere. |
| **0.1.3** | **Phones**: a bottom-sheet layout with pinch zoom, a column layout sideways; the **technical report** page; the mode switch in the console's title bar and a console that keeps its width; less work per frame. |
| **0.1.2** | The extras and the **vector view**: the model in Gaussian-id colours, the fit animation (OT mesh → Gaussians → scan reveal), the monitor panel, the View menu (vector / OT mesh / outlines / error map), export as PNG / JPEG / WebP / AVIF / BMP, a process log, notices in the status strip, bilingual labels, starlight and geometry on the stage, the way back to the start screen. |
| **0.1.1** | **Research mode**: custom budget and steps, edge width, L1 / L2 loss, view modes and split against any resampler, 1σ outlines, error map, region tool, session info with the PSNR curve. |
| **0.1.0** | First public version: the WebGPU engine and the UI in the lab-white / copper design, the `.otsr` format, the Python reference and the golden tests. |

Details per version are in [CHANGELOG.md](CHANGELOG.md).

## Documentation

[docs/index.md](docs/index.md) — getting started, the method, the engine, the `.otsr` format, testing,
development, performance, troubleshooting — and [docs/tutorial-porting.md](docs/tutorial-porting.md), the
step-by-step record of how the research code was turned into this engine (derivations, decisions, pitfalls).

## Acknowledgements

OTSR is built and maintained by [Aztech Labs](https://github.com/Aztech-Lab); the sample image is our own work.
The WebGPU trainer of [SAD (Soft Anisotropic Diagrams)](https://github.com/luckyiyi/SAD) was a helpful reference for
structuring a browser-side fitting loop; the GPU tests run on [Dawn's Node bindings](https://github.com/dawn-gpu/node-webgpu).

## License

MIT — see [LICENSE](LICENSE).
