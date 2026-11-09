# Changelog

## 0.1.4 — 2026-11-09

The technical report, rewritten as a magazine would tell it; the transport lattice is now called the mesh.

- **The method in plain words.** `report.html` tells the whole story of the workshop paper behind the app — the
  problem, the picture made of Gaussians, where they should go, the transport, the solver, the re-layout, the
  renderer, the fit, super-resolution, what the paper measured, editing through the mesh, the limits — in a magazine
  register, with the paper's results quoted in prose. No figures or tables are copied from the paper. Copper
  *In the app* tags mark what the app does now; *Paper* tags what the experiments show but the app does not do yet
  (the error-driven re-layout, the degradation-consistent refinement, the fold penalty).
- **Figure A runs the pipeline live.** Under each box of the pipeline diagram, its stage computed in the page on a
  112-pixel copy of the sample picture: the picture, its importance density on the transport grid, a real entropic
  transport and the mesh unfolding along it, the Gaussians of the mesh in the vector view, a real fit (the blend,
  exact gradients and Adam with the paper's schedule, 400 steps, about 30 dB) and a crop at 12× with the edge pass
  against its pixels. The first time the figure comes into view the strip shows the computation as it happens; then
  it replays the stages, the active box of the diagram lit. The numerical core is `src/report-fit.ts` (no DOM); its
  gradients are checked against finite differences in `tests/unit/report-fit.test.ts`.
- **Nine live demonstrations**, lettered in reading order: a handful of Gaussians at any scale next to their pixels,
  with the edge pass on (A); equal shares in one dimension (B) and in two, with a real transport solve (C); the mesh
  unfolding along a convex-potential map with the Gaussians of its cells (D); Sinkhorn iterations with ε-scaling (E);
  the error-driven re-layout (F); additive against weight-normalised blending (G); the edge pass across an edge, its
  legend now above the curves (H); editing through the mesh by dragging vertices, folded cells in red (I). Each has
  Play / Pause and sliders, runs only while on screen and follows the app's motion setting; the heavy ones compute
  when they first come into view.
- **Mesh, not lattice**, everywhere a reader sees it: the report, the View menu and the research panel (*OT mesh*),
  the shortcut list, the process log, the README and the docs.
- **A reading list** of 45 entries: the paper's 28 references and 17 more — Monge, Kantorovich, Brenier, Villani,
  Sinkhorn–Knopp, Lloyd, EWA splatting, Keys, Lanczos, the WebGPU specifications — each with a line on why it is here.
- The contents list scrolls on short windows; the report entry on the start screen sits under the pair.

Found on the way: the barycentric transport map is pulled half a cell inward at the border of the box by the entropic
blur, which left the outer band of a small picture without Gaussians; in the report's core the normal displacement now
fades to zero at the edge of the box, as a transport of the box onto itself keeps its boundary (the app's engine is
unchanged).

Verified: build and type check; unit 9/9 (three new: exact gradients against central differences, max deviation
1.1 × 10⁻¹⁰ against a gradient of 5.5 × 10⁻²; a fit from the transported mesh, 21.9 → 52.8 dB in 200 steps; the
transport of a uniform and of a left-heavy density); GPU golden 8/8; the browser and phone suites; desktop and phone
review shots of the report, the live strip run to the end, every demo mounted, no element outside the viewport.

Each entry records what the version brought, how it was verified before release, and what was found and fixed on
the way. Numbers are from the machines named; the engine (`src/engine`) is unchanged since 0.1.0 except where said.

## 0.1.3 — 2026-11-08

Phones, a technical report, a tighter console, less work per frame.

- **Phones and small tablets** (below 960 px) have a layout of their own: the console is a sheet at the bottom that
  folds to its title bar — it folds when an image arrives and while a fit runs, and opens on the result and the export
  controls — the zoom bar sits on the sheet and the action stays at the top; held sideways, the console is a column on
  the left that closes to a tab on the edge. Two fingers pinch-zoom the image; the page itself never zooms. No blur
  over the canvas, fewer stars and a slower animation cadence on phones; notices float above the zoom bar; the
  settings dialog scrolls; icon-only system buttons; "Choose image" instead of "Drop image" on touch screens.
  `npm run smoke:mobile` checks three touch viewports against these rules.
- **Technical report** (`report.html`): the method, the engine and the research process on one page, with diagrams
  and the validation figures, in the app's design and theme — from the start screen, the chip next to the version
  badge, and the settings.
- **Console**: the Simple / Research switch sits in the title bar (no mode row); the panel keeps its width whatever
  the file name (ellipsis; the Replace button stays put; no sideways scrolling); the result line shows the PSNR; the
  console is narrower on windows under 1280 px. The image is placed from the measured panels, and a fitted image
  stays fitted when a panel opens or closes or the window changes.
- **Less overhead**: the console no longer re-renders on every pan and zoom frame; the tile-entry check after a render
  is one read-back in flight at a time and at most every 120 ms while the view only moves; the monitor's crops are
  skipped when the panel is not shown; canvas resizes are coalesced; the export label is rebuilt only when the format
  changes; the unused icon set is gone. The design tokens live in `src/tokens.css`, shared by the app and the report.
- One sample image needs no selector; the settings show the GPU in use.

Found and fixed before release (a review of the change set, each finding traced in the code):

- The tile-entry read-back after a render was queued *before* the frame's own build (the read-back submits its
  own copy right away), so it measured the previous build; after a window resize or a single zoom step the bottom
  tiles of the image could stay empty until the next pan. The check now runs after the submit, and a check that
  cannot run yet is deferred, never dropped.
- The folded sheet's title bar showed on the start screen (the collapsed state is now only a layout class while an
  image is loaded); on 360–400 px phones the report chip overlapped the system buttons (hidden there — the start
  screen and the settings keep the link), the research zoom bar was wider than the screen and the view menu hung
  off it (both tightened); a two-finger pinch started in the region tool committed a stray 8 × 8 region (the region
  is restored); the Gaussian-outline note did not follow the zoom; in landscape the notices overlapped the console
  column; `viewport-fit=cover` without safe-area insets put buttons under a notch (removed).
- The report said a laptop GPU was slower "by the ratio of the cards"; it now gives the measured 5–20× for a
  discrete laptop GPU and a minute or two for an integrated one. The SAD comparison's time range is 38–60 % of
  SAD's time, as the figure's legend says (the porting record had 40–55 %).

Verified: build and type check; unit 6/6; GPU golden tests 8/8 (Dawn, RTX 5070 Ti); browser smoke test; the phone
suite on four touch viewports (390 × 844, 360 × 800, 844 × 390, 820 × 1180 — start screen, image, sheet or column,
pinch, a fit, menus, dialogs, dark theme, the report page; every floating element inside the viewport, no sideways
scrolling); desktop screenshot series with a synthetic image and with the 1024 × 815 sample (83,592 Gaussians,
1500 steps, 47.6 dB in 1.8 s on the 5070 Ti). Real phones were not in the loop; the layout was tested with Chrome's
touch emulation.

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

Found and fixed on the way: the first fit animation drew the Gaussians on the 2-D overlay and stalled at a few
frames per second — replaced by the GPU render of a recoloured copy of the live parameters; a colour-field version
of the vector view was replaced by the paper's per-Gaussian random colour × luma; the clip-path on the canvas swallowed
pointer events (the stage now receives them); a second fit submitted a destroyed crop buffer (renderers are retired
and destroyed after their render loop; a regression step covers it).

Verified: build; unit 6/6; GPU golden 8/8; browser smoke test; two screenshot series (synthetic 120 × 80 and the
1024 × 815 sample: 83,592 Gaussians, 1500 steps ≈ 2.5 s, 47.62 dB on the 5070 Ti; PNG / JPEG / WebP / BMP export
available, AVIF not in Chrome; the way back to the start screen leaves an empty state).

## 0.1.1 — 2026-11-08

Research mode.

- A second mode next to *Simple*: custom budget (Gaussians per pixel) and step count, the edge width of the
  distance-field pass, the pixel loss (L2 — the paper setting — or L1 / Charbonnier), view modes (OTSR, raw Gaussians,
  bicubic, bilinear, nearest) with the split comparison against any of them, the 1σ Gaussian outlines and the error
  map (|model − source|, inferno scale with automatic gain), a shared region-of-interest tool, and session
  information with the PSNR curve of the last fit.
- The engine's fit loop takes the loss option; the default path is unchanged (golden tests).

Engine additions for this version, all with the default path unchanged: `View.loss_kind` and a Charbonnier branch
in `backward.wgsl`; `transportMap(..., { onStage })` projecting every Sinkhorn stage's barycentric map inside the
single submit; `otInit({ onPhase, onStage })`; the display-only `recolour.wgsl` kernel and `Rasterizer.setParams()`.

Verified: build; unit 6/6; GPU golden 8/8 (the L2 path bit-for-bit as before); browser smoke test; L1 fits on the
synthetic image and the sample (the PSNR is reported against the source in both cases).

## 0.1.0 — 2026-11-07

First public version: the engine and the industrial sci-fi UI.

- WebGPU engine: tile-binned rasteriser with exact gradients, Adam fitting loop, GPU Sinkhorn for the
  optimal-transport initialisation, distance-field edge pass, tiled export, canvas presentation.
- Web UI: image → preset (Fast / Standard / Best) → Enhance → export at 1×–16× or any scale, with a split
  before/after comparison, drag and drop, clipboard paste, keyboard shortcuts, persisted settings, one job at a time
  with cancel, time and memory estimates, light / dark theme — in the lab-white / copper flat design.
- `.otsr` model file format 1.0 (`docs/file-format.md`), readable from Python (`reference/otsr_ref/fileformat.py`).
- Python reference implementation and golden tests; GPU kernel tests through Dawn in Node; browser smoke test.

Verified: the forward pass within 1.2 × 10⁻⁶ of the float64 dense reference, all nine gradient columns within
1.4 × 10⁻⁶ of float64 autograd, Adam within 10⁻⁷ of PyTorch, the 40-step golden fit identical to four decimals, the
edge pass within 8 × 10⁻⁷, tiled export equal to untiled (docs/testing.md); the research CUDA pipeline reproduced
within run-to-run noise (43.62 vs 43.55 dB, 46.49 vs 46.48 dB on two Kodak images) in 1/60 of the time; RTX 4090
timings in docs/performance.md.
