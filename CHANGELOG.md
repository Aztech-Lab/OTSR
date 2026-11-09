# Changelog

## 0.1.4 — 2026-11-09

Phones first, part two — a quicker tier, the tier in the run button, a system check on the start screen, a fit
animation that no longer slows the fit — and the technical report, rewritten as a magazine would tell it.

In the app:

- **A Turbo tier** (極速): the budget of Fast (1 Gaussian per 20 px) with 400 steps instead of 1500, and no fit
  animation, whose fixed timeline would otherwise be most of the wait. On the RTX 4090 (Dawn in Node) the sample
  (1024 × 815) fits in 0.11 s at 40.5 dB (Fast: 0.53 s, 43.4 dB) and a 2048 × 2048 anime frame in 0.53 s at 49.0 dB
  (Fast: 2.17 s, 50.1 dB). Halving the budget instead loses more and gains little, because a step costs mostly per
  pixel. The tiers are Turbo / Fast / Standard / Best on every layout; docs/performance.md has all four.
- **Phones start on Fast** until a tier is picked by hand (a phone: a touch screen whose short side is under 600 px;
  a custom setting of research mode is kept).
- **The run button is a split button on phones**: Enhance, and next to it the tier with its estimated time. Its menu
  lists the four tiers with what each costs for the open image (Gaussians, steps, and the time once this device has
  been measured); in research mode also the custom budget, the steps and the pixel loss. Under it, a strip of
  readouts in the style of the console's Source boxes: the source (name and size), its Gaussians, the GPU memory and
  the state (Idle, or Fitted with the PSNR). The sheet therefore opens on the output — scale, *Crisp edges*, export,
  save / load. *Crisp edges* (and the edge width of research mode) now belong to Output on every layout.
- **A system check on the start screen** (硬體檢測; bottom right on computers, under the instrument on phones, beside
  it when held sideways): whether WebGPU works here (*Ready*, *Limited*, *Slow*) and what that means — the largest
  image one fit can take at Fast and at Standard, and, once a fit has run, the measured speed and the time the
  sample takes — then the facts behind it: the browser and system, the GPU (model, vendor, architecture), its memory,
  the buffer limits, the optional features, the CPU threads and the screen. Browsers do not report the size of the GPU
  memory; the card says so and shows the system memory (which phones, Apple and Intel integrated GPUs share with the
  GPU), and the driver's memory heaps where Chrome's WebGPU developer flag exposes them. The boot log adds a SYSTEM
  CHECK line. When WebGPU does not start at all, the notice lists the browser, whether the page is secure, whether the
  API exists, what the adapter request answered and the GPU that WebGL sees.
- **The state of the work moved to the status strip** (■ Idle / Running / Fitted at the bottom left; in the strip
  under the action on phones), and the mode switch took its place at the right of the console's title bar, framed and
  larger.
- **Less work on phones**: the viewer draws at most 2 canvas pixels per CSS pixel (a 3× screen rendered 2.25 times
  the pixels for little visible gain, and the canvas-sized buffers shrink with it), the starlight holds still while a
  job runs, and a folded or closed console is not kept up to date until it opens.
- **A lighter phone console**: Output and Export only (the model file buttons and the process log stay on computers),
  so the image takes most of the screen; the sheet rises in when it opens, also by itself after a fit. The start
  screen keeps all its motion on phones, and the settings no longer scroll sideways on narrow screens.
  The three start actions are larger, and the report entry spans the width of the pair above it, on frosted glass.
- **The fit animation no longer slows the fit.** Its mesh was stroked two canvas pixels wide on high-density screens,
  a slow path of the 2-D canvas, and redrawn at the display rate: in headless Chrome the fit ran at about a third of
  its speed while the mesh was on screen. The mesh is now drawn with hairlines and one fill per brightness level
  instead of two per node, at most 30 times a second (on phones with 40 nodes per axis instead of 110), it fades out
  with the renders, and the overlay is left alone while it is blank. The fit keeps its own pace under the animation;
  on the phone layout the whole sample fit now finishes while the mesh is still settling.

The technical report:

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

Found on the way:

- The frosted panels were not frosted in current Chrome: the build kept only the `-webkit-` form of `backdrop-filter`,
  which Chrome no longer reads. The stylesheet now gives the standard property alone.
- After every fit the monitor's snapshot loop finished its 600 ms wait before the fit could end — up to 0.6 s added to
  each fit, and after a failed fit the loop kept running. It now stops at once, also on failure.
- docs/performance.md said that halving the budget roughly halves the fit time. On the RTX 4090 at 2048 × 2048,
  halving it from 1/10 to 1/20 per pixel cuts the step time by about a fifth (1.78 → 1.45 ms); the page is corrected.
- The barycentric transport map is pulled half a cell inward at the border of the box by the entropic blur, which
  left the outer band of a small picture without Gaussians; in the report's core the normal displacement now fades to
  zero at the edge of the box, as a transport of the box onto itself keeps its boundary (the app's engine is
  unchanged).
- The desktop smoke test now takes `--vendor=` like the phone test and stops on any other GPU.

Verified: build and type check; unit 9/9 (three new for the report's core: exact gradients against central
differences, max deviation 1.1 × 10⁻¹⁰ against a gradient of 5.5 × 10⁻²; a fit from the transported mesh,
21.9 → 52.8 dB in 200 steps; the transport of a uniform and of a left-heavy density); GPU golden 8/8 (the engine is
unchanged); the browser suite; the phone suite with four viewports, a 3× phone among them, and its new checks — the
first-visit tier, the viewer at 2× on a 3× screen, the system check clear of the instrument, the wordmark and the
navigation, the console opening on Output, the split button with its tier menu, a Turbo fit without the animation, the
strip with the result (167 checks); desktop checks at 1600 × 1000 and 1280 × 720 — the system check clear of the
instrument, the boot log and the status strip, the tiers, the state in the status strip, a Turbo fit, research mode,
back home with the measured speed; review shots of the mesh animation in both themes; the tier timings on the
RTX 4090; desktop and phone review shots of the report, the live strip run to the end, every demo mounted, no element
outside the viewport.

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
