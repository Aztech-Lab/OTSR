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

The switch in the header picks the mode; the choice and every setting are remembered in the browser.

**Simple** — the three steps of the product and nothing else:

1. **Image**: open a file, drop it on the view, or paste it from the clipboard. PNG, JPEG, WebP, AVIF, BMP and GIF
   (first frame) are decoded by the browser. Until a model exists the view shows the image with bicubic interpolation.
2. **Quality**: *Fast* (1 Gaussian per 20 pixels, 1500 steps), *Standard* (1 per 10 pixels, 3000 steps — the
   paper setting) or *Best* (1 per 7 pixels, 4000 steps). Below the choice: the Gaussian count, the GPU memory the fit
   will take and, once one fit has run on this machine, the expected time. *Crisp edges at magnification* is the
   distance-field edge pass; leave it on unless you want the plain Gaussian render.
3. **Fit & Enhance**, then **Export** at 2×, 4×, 8× or any factor (the output size is shown), or **Save model**
   (`.otsr`, 36 bytes per Gaussian) to re-open later with *Load model…* without a new fit.

After the fit the view shows a **split comparison**: bicubic interpolation of the original on the left of the divider,
OTSR on the right. Drag the divider, zoom in (wheel, `+`/`−`, the toolbar) and pan (drag) to inspect; press `0` to fit
the image to the view and `1` for actual pixels. Hold `B` to see bicubic everywhere, `N` for nearest neighbour and `R`
for the raw Gaussians without the edge pass. The toolbar's *Compare* button (or `C`) toggles the divider.

**Research** — the same workflow with every parameter exposed and the analysis tools:

- *Custom* quality with the budget (Gaussians per pixel) and the step count; the edge width slider (the transition width
  of the edge pass in output pixels, 1.4 by default: smaller is harder, larger softer).
- *View*: OTSR, Gaussians only (no edge pass), and the classic resamplers of the source — bicubic, bilinear, nearest —
  plus the split comparison against any of them.
- *Region of interest*: one rectangle in source pixels shared by every analysis tool. Press `M` (or the *Region*
  button), drag on the view to draw it, drag the corners to resize, nudge it with the arrow keys (Shift: 10 px), hold
  Space to pan meanwhile. *Zoom to region* frames it.
- *Session*: the adapter, the model and the measured throughput of this GPU.

Research mode gains more tools over time (comparison figures of the region across methods, error maps, the Gaussian
structure view, fit curves); the CHANGELOG lists what is in.

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
