# Changelog

## 0.1.0 — unreleased

First public version.

- WebGPU engine: tile-binned rasteriser with exact gradients, Adam fitting loop, GPU Sinkhorn for the
  optimal-transport initialisation, distance-field edge pass, tiled export, canvas presentation.
- Web UI with two modes. *Simple*: image → quality preset (Fast / Standard / Best) → Fit & Enhance → export, with a
  split before/after comparison. *Research*: custom budget and steps, edge width, view modes, split against any
  resampler, a shared region-of-interest tool, session info. Drag and drop, clipboard paste, keyboard shortcuts,
  persisted settings, one job at a time with cancel, time and memory estimates, light / dark theme.
- `.otsr` model file format 1.0 (`docs/file-format.md`), readable from Python (`reference/otsr_ref/fileformat.py`).
- Python reference implementation and golden tests; GPU kernel tests through Dawn in Node; browser smoke test.
