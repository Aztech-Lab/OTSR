# Changelog

## 0.1.0 — unreleased

First public version.

- WebGPU engine: tile-binned rasteriser with exact gradients, Adam fitting loop, GPU Sinkhorn for the
  optimal-transport initialisation, distance-field edge pass, tiled export, canvas presentation.
- Web UI with two modes. *Simple*: image → preset (Fast / Standard / Best) → Enhance → export at 1×–16× or any
  scale, with a split before/after comparison. *Research*: custom budget and steps, edge width, L1 / L2 pixel loss,
  view modes, split against any resampler, Gaussian outline and error-map layers, a shared region-of-interest tool,
  session info with the PSNR curve. Drag and drop, clipboard paste, keyboard shortcuts, persisted settings, one job
  at a time with cancel, time and memory estimates, light / dark theme.
- Industrial sci-fi flat design: the action at the top centre with a live fit readout, console on the left (mode
  switch, process log, collapsible), monitor on the right during fits (the four most active regions as live renders),
  pixel rulers and cursor readout, vignettes under the floating UI, a slowly moving stage around the image, serif
  text in research mode, bilingual (English / Traditional Chinese) labels, a prominent theme button, error dialogs,
  motion toggle.
- Fit animation from the real process: the OT lattice through every Sinkhorn ε-scaling stage, the Gaussians growing
  into a random watercolour at their actual positions, and a scan-line reveal of the fitted image.
- `.otsr` model file format 1.0 (`docs/file-format.md`), readable from Python (`reference/otsr_ref/fileformat.py`).
- Python reference implementation and golden tests; GPU kernel tests through Dawn in Node; browser smoke test.
