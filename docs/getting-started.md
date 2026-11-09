# Getting started

## Requirements

- A browser with WebGPU: Chrome or Edge 113+ (Windows, macOS, ChromeOS, Android; Linux from Chrome 121),
  Safari 26+ (macOS, iOS), Firefox 141+ on Windows. If in doubt, open `chrome://gpu` and look for "WebGPU: Hardware
  accelerated".
- A GPU. Integrated GPUs (Intel, Apple M, AMD APU) work; discrete GPUs are 5–20× faster for the fit.
- The page must be served over `http://` or `https://`; opening `index.html` directly from the file system does not
  work (browsers block ES modules on `file://`).

## Running the app

**Prebuilt (`dist/` folder present)**

- Windows: double-click `serve.bat`. It needs [Node.js](https://nodejs.org) 20 or newer; it starts a local server
  on `http://127.0.0.1:8787/` and opens the browser. Closing the window stops the server.
- Any system with Node.js: `node tools/serve.mjs` (add `--port=9000` to change the port, `--no-open` to not open a
  browser).
- Any system with Python: `python -m http.server -d dist 8787`, then open `http://localhost:8787/`.
- Any static web host (GitHub Pages, nginx, …): publish the `dist/` folder as is; all paths are relative.

**From source**

```bash
npm install
npm run build        # writes dist/
npm run serve        # or: npm run dev for a live-reload development server
```

## Two modes

The switch in the console's title bar (Simple · Research) picks the mode; the choice and every setting are remembered
in the browser.

**Simple** — the three steps of the product and nothing else:

1. **Image**: open a file, drop it on the view, or paste it from the clipboard. PNG, JPEG, WebP, AVIF, BMP and GIF
   (first frame) are decoded by the browser. Until a model exists the view shows the image with bicubic interpolation.
2. **Quality**: *Fast* (1 Gaussian per 20 pixels, 1500 steps), *Standard* (1 per 10 pixels, 3000 steps — the
   paper setting) or *Best* (1 per 7 pixels, 4000 steps). Below the choice: the Gaussian count, the GPU memory the fit
   will take and, once one fit has run on this machine, the expected time. *Crisp edges at magnification* is the
   distance-field edge pass; leave it on unless you want the plain Gaussian render.
3. **Enhance** — the copper button at the top centre (`F` or Enter). While it runs, the button becomes a live
   readout: step, PSNR, iterations per second, time left and the PSNR curve; *Stop* (or Esc) keeps what has been
   fitted so far. Then **Export** at 1×, 2×, 4×, 8×, 16× or any factor (the output size is shown under each choice)
   as PNG, JPEG, WebP, AVIF or BMP — the ▾ next to the button opens the format menu with a quality slider for the
   lossy formats; a format this browser cannot encode is greyed out (AVIF in most browsers today) — or **Save model**
   (`.otsr`, 36 bytes per Gaussian) to re-open later with *Load model…* without a new fit. Clicking the **OTSR**
   wordmark (or pressing Home) returns to the start screen — the entering transition played backwards — and drops
   the image and the model.

While the fit runs the view plays the process itself (*Fit animation* in the settings): the uniform lattice, then the
real stages of the optimal-transport initialisation (the barycentric map after every ε-scaling stage of the Sinkhorn
loop) as a glowing mesh that settles on the image structure; then the Gaussians themselves, rendered live by the
engine from the parameters the optimiser is changing, growing from points, each with an independent random colour
(the Gaussian-id view of the research code, a small display-only kernel `recolour.wgsl`); finally a scan line from
left to right gives every Gaussian its true colour back. A **monitor** panel slides in from the right with the three regions of the image with the
strongest gradient (chosen when the fit starts), each as a live render — in Gaussian-id colours while the fit runs,
in true colours afterwards; it closes when the fit ends and reopens from the tab on the right edge. The console on the left logs the real steps (density, Sinkhorn stages, upload, optimisation, readback, edge
pass) — a running series such as the OT stages or the step counter keeps updating one line; the copper bookmark on a
panel's inner edge collapses it.

After the fit the view shows a **split comparison**: bicubic interpolation of the original on the left of the divider,
OTSR on the right. Drag the divider, zoom in (wheel, `+`/`−`, the toolbar) and pan (drag) to inspect; press `0` to fit
the image to the view and `1` for actual pixels. Hold `B` to see bicubic everywhere, `N` for nearest neighbour and `R`
for the raw Gaussians without the edge pass. The toolbar's *Compare* button (or `C`) toggles the divider; its *View* menu
switches the layers: the **vector view** (`V`, every Gaussian in a random colour times its luma — the Gaussian-id
figure of the paper), the **OT lattice** of the initialisation over the image (`T`), the **1σ outlines** (`G`) and
the **error map** (`D`). Pixel rulers
run along the top and left edge of the image and the status strip shows the source pixel under the cursor. The canvas
is clipped to the image, so the slowly moving stage shows around it when zoomed out. The *GPU mem* readout is an
estimate from the engine's allocations before the fit and the allocated figure afterwards. Errors open a dialog; other
notices show as a chip in the middle of the status strip, which also keeps the last status message. The light / dark
button sits at the top right; *System* follows the OS theme. The settings also switch the starlight over the stage, the
fit animation and all decorative motion off.

**Research** — the same workflow with every parameter exposed and the analysis tools:

- *Custom* quality with the budget (Gaussians per pixel) and the step count; the edge width slider (the transition width
  of the edge pass in output pixels, 1.4 by default: smaller is harder, larger softer); the pixel loss — *L2* (mean
  squared error, the paper setting) or *L1* (Charbonnier, ε = 10⁻³; PSNR is always reported against the source).
- *View*: OTSR, Gaussians only (no edge pass), and the classic resamplers of the source — bicubic, bilinear, nearest —
  plus the split comparison against any of them, and the four layers of the toolbar's *View* menu: **Vector view** (`V`),
  **OT lattice** (`T`), **Gaussians 1σ** (`G`) draws the 1-sigma outline
  of every Gaussian in view in its own colour from 2× zoom on; **Error map** (`D`) renders the model at 1× and shows
  |model − source| per pixel on an inferno scale whose gain follows the 99.5th percentile (the mean and maximum error
  and the white point are listed under the switches).
- *Region of interest*: one rectangle in source pixels shared by every analysis tool. Press `M` (or the *Region*
  button), drag on the view to draw it, drag the corners to resize, nudge it with the arrow keys (Shift: 10 px), hold
  Space to pan meanwhile. *Zoom to region* frames it.
- *Session*: the adapter, the model, the measured throughput of this GPU and the PSNR curve of the last fit.
- Research mode sets the text in a serif face; the controls and readouts are the same.

Research mode gains more tools over time (comparison figures of the region across methods, error maps, the Gaussian
structure view, fit curves); the CHANGELOG lists what is in.

## On a phone or a small tablet

Below 960 px the layout changes. The console is a sheet at the bottom of the screen: tap its title bar (the chevron)
to open or fold it. It folds by itself when an image arrives and while a fit runs — the picture is the point then —
and opens afterwards on the result and the export controls. The action stays at the top, the zoom bar sits on the
sheet. Pinch with two fingers to zoom, drag to pan, double-tap to fit the image. Held sideways, the console is a
column on the left that closes to a tab on the edge. The monitor panel, the pixel rulers' readout and the status strip
are not shown on phones; notices appear above the zoom bar. Fits run on the phone's own GPU and take correspondingly
longer: start with the *Fast* preset.

The **technical report** (`report.html`, from the start screen, the chip next to the version badge or the settings)
tells the method behind the app in plain words, with nine live demonstrations, what the app does and how the
engine works.

## Keyboard shortcuts

`?` shows the list in the app: `O` open · `F`/`Enter` fit · `E` export · `S` save model · `Esc` cancel / leave the
region tool · `C` compare · `B`/`N`/`R` hold to peek · `+`/`−`/`0`/`1`/`2`/`4`/`8` zoom · `M` region tool ·
arrow keys nudge the region.

## Limits and behaviour worth knowing

- Images above 8.5 megapixels ask for confirmation; above 33 megapixels they are refused. Memory grows with the
  pixel count (two float images) and with the Gaussian count (about 400 bytes per Gaussian during the fit); the
  estimate under the quality choice is for the current image.
- One job runs at a time (fit or export); *Cancel* stops a fit at the next progress tick and keeps the model fitted so
  far. Closing the tab while a job runs asks for confirmation.
- The browser's PNG encoder limits exports to 16384 pixels per side and about 120 megapixels.
- The model is fitted to the sRGB pixel values as displayed; there is no colour management. Alpha is ignored.
- Settings, the mode and the panel layout persist in the browser's local storage; nothing is sent anywhere.
