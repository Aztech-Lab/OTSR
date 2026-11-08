# Changelog

## 0.1.2 — 2026-11-08

The extras and the vector view.

- **Vector view**: the model drawn in Gaussian-id colours (every Gaussian in its own random colour times its luma —
  the figure of the paper), through a display-only kernel (`recolour.wgsl`); also the layer the fit animation and the
  monitor use while a fit runs.
- **Fit animation** from the real process: the optimal-transport lattice through every Sinkhorn ε-scaling stage,
  then the Gaussians rendered live by the engine in Gaussian-id colours, then a scan-line reveal of the true colours.
- **Monitor** panel during fits: the three regions of the image with the strongest gradient as live renders.
- **View menu** in the toolbar: vector view, OT lattice, 1σ outlines, error map (keys V / T / G / D).
- **Export** as PNG, JPEG, WebP, AVIF (where the browser encodes it) or BMP, with a quality setting for the lossy
  formats; the export button carries the format, the *Format* button next to it opens the menu.
- Process log with one updating line per running series (OT stages, steps, export tiles); notices in the middle of
  the status strip; errors in a dialog; the GPU memory estimate from the engine's own allocation formula before a fit
  and the allocated figure afterwards.
- Panels: the console on the left with the mode switch, collapsible by a copper bookmark; the monitor on the right;
  the action at the top centre with a live readout (step, PSNR, it/s, time left, PSNR curve).
- Bilingual (English / Traditional Chinese) labels on every control; a prominent light / dark button; black text on
  copper in the dark theme; pixel rulers and a cursor readout; a serif face in research mode.
- Stage: the dot grid with instrument-style circle geometry and copper starlight (switchable in the settings); an
  entering transition from the start screen and its reverse when the wordmark (or Home) is clicked; the start screen
  with Load sample / Open image as one pair and the wordmark centred.
- Second fits no longer trip a destroyed-buffer error from the monitor's crop renderer.

## 0.1.1 — 2026-11-08

Research mode.

- A second mode next to *Simple*: custom budget (Gaussians per pixel) and step count, the edge width of the
  distance-field pass, the pixel loss (L2 — the paper setting — or L1 / Charbonnier), view modes (OTSR, raw Gaussians,
  bicubic, bilinear, nearest) with the split comparison against any of them, the 1σ Gaussian outlines and the error
  map (|model − source|, inferno scale with automatic gain), a shared region-of-interest tool, and session
  information with the PSNR curve of the last fit.
- The engine's fit loop takes the loss option; the default path is unchanged (golden tests).

## 0.1.0 — 2026-11-07

First public version: the engine and the industrial sci-fi UI.

- WebGPU engine: tile-binned rasteriser with exact gradients, Adam fitting loop, GPU Sinkhorn for the
  optimal-transport initialisation, distance-field edge pass, tiled export, canvas presentation.
- Web UI: image → preset (Fast / Standard / Best) → Enhance → export at 1×–16× or any scale, with a split
  before/after comparison, drag and drop, clipboard paste, keyboard shortcuts, persisted settings, one job at a time
  with cancel, time and memory estimates, light / dark theme — in the lab-white / copper flat design.
- `.otsr` model file format 1.0 (`docs/file-format.md`), readable from Python (`reference/otsr_ref/fileformat.py`).
- Python reference implementation and golden tests; GPU kernel tests through Dawn in Node; browser smoke test.
