# Development

## Setup

Node.js 20+ and npm. `npm install` brings TypeScript 5.9, Vite 8, Vitest 5, `@webgpu/types`, Dawn's `webgpu`
package (prebuilt binaries for Windows / macOS / Linux x64 and arm64) and `puppeteer-core` (uses an installed Chrome
or Edge; set `OTSR_BROWSER=<path>` if it is not found). Python with PyTorch is only needed to regenerate the golden
data.

| command | what |
|---|---|
| `npm run dev` | Vite dev server with live reload |
| `npm run build` | type-check, production build into `dist/`, copies `docs/` into `dist/docs/` |
| `npm run serve` | serves `dist/` (what `serve.bat` runs) |
| `npm test` / `npm run test:gpu` / `npm run test:all` | unit / GPU / both |
| `npm run smoke` | browser smoke test of `dist/` |
| `npm run smoke:mobile` | phone / tablet layout test of `dist/` (touch emulation, three viewports, the report page) |
| `npm run bench -- image.rgb.bin --adapter=4090` | end-to-end timing in Node (see performance.md) |
| `npm run golden` | regenerate `tests/golden/` from the Python reference |
| `npm run probe:dawn` / `npm run probe:webgpu` | adapter probes for Node (Dawn) / headless Chrome |

## Layout

```
src/core/spec.ts          every constant of the model (mirrors reference/otsr_ref/spec.py)
src/core/ot_cpu.ts        CPU parts of the OT initialisation (density, pooling, warp, eigen, sampling)
src/core/fileformat.ts    .otsr codec
src/engine/device.ts      adapter / device acquisition and error reporting
src/engine/pipelines.ts   compiles all WGSL kernels (common.wgsl is prepended to each)
src/engine/gpu.ts         buffer helpers, read-back, 2-D dispatch of 1-D work
src/engine/scan.ts        prefix sums
src/engine/raster.ts      the Rasterizer: tile binning + forward pass for one View
src/engine/fitter.ts      backward, gradient reduction, Adam, loss
src/engine/fit.ts         the fitting loop (runFit)
src/engine/sinkhorn.ts    GPU Sinkhorn (transportMap);  otinit.ts: the whole initialisation
src/engine/edges.ts       the edge pass;  present.ts: canvas presentation + classic resampling;  export.ts: tiled render
src/engine/shaders/*.wgsl the kernels
src/app/main.ts           bootstrap: device, viewer, panel, layout insets, shortcuts, header, drag & drop / paste, window.otsr
src/app/store.ts          the application state and persisted preferences (one store, one subscribe)
src/app/jobs.ts           the job runner: one GPU job at a time, progress, cancel, error reporting
src/app/actions.ts        the operations behind the buttons (open, fit, export, save / load, samples, region tool)
src/app/panel.ts          the side panel for the current mode (simple / research), kept in sync with the store
src/app/viewer.ts         WebGPU canvas + 2-D overlay: pan / zoom, split comparison, hold-to-peek, region tool
src/app/roi.ts            the shared region-of-interest model (drawing, editing, clamping)
src/app/presets.ts        quality presets and the time / memory estimates
src/app/shortcuts.ts      keyboard shortcuts;  toast.ts: notifications;  dom.ts: h(), bilingual labels, formats;  image.ts: decoding, PNG, downloads
src/tokens.css            design tokens and base elements, shared by the app (style.css) and the report (report.css)
report.html, src/report.ts the technical report page (second Vite entry)
```

The engine has no dependency on the DOM except `present.ts` (canvas format) and can run in Node (the tests and
`tools/bench.mjs` do exactly that, loading the TypeScript through Vite's module runner).

## Conventions

- All kernels share the parameter layout of `spec.ts` (9 floats per Gaussian, interleaved) and the `View` struct.
- Bind groups use `layout: 'auto'`; a bind group may only contain the bindings the entry point actually uses.
- A kernel change comes with a test; a maths change starts in the Python reference (CONTRIBUTING.md).
- Comments explain *why* and give the formula; the WGSL is the documentation of *what*.
- TypeScript is strict; `npm run typecheck` covers `src/` and `tests/`.

## The `window.otsr` hook

The app exposes the operations behind the buttons for scripts, demos and the smoke test:

```js
otsr.state()                              // { image, model, fitting, lastFit, adapter, zoom, mode, roi }
otsr.loadRgba(rgba8, W, H, name)          // load pixels (Uint8ClampedArray, RGBA)
await otsr.loadImage(blob, name)          // load a file / blob
await otsr.fit({ kPerPx: 0.1, steps: 3000 })   // custom settings (omit for the current preset) -> { psnr, seconds, steps, count, cancelled }
await otsr.export(4)                      // the same as the Export button
await otsr.render(4, true)                // { width, height, rgba } at 4x with edges, no download
otsr.modelBytes()                         // ArrayBuffer of the .otsr file, or null while fitting
otsr.setMode('research')                  // 'simple' | 'research'
otsr.setRoi({ x, y, w, h })               // the shared region (source pixels), or null
await otsr.snapshotPng(withOverlay)       // the current view (plus divider / region overlay) as a PNG data URL
otsr.viewer                               // the Viewer: mode, compare, edgePx, zoomAbout(z, cx, cy), fitToView(), snapshot()
otsr.store                                // the state store (state, setPrefs)
```

## Releasing a build for users

`npm run build` and ship the `dist/` folder together with `serve.bat` and `tools/serve.mjs` (both dependency-free),
or publish `dist/` on any static host. The build is relative-path only, so it works from a sub-directory.
