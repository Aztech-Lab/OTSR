# Performance

## Reference measurements

RTX 4090, Windows 11, driver 32.0.16.1692, Dawn (D3D12) in Node 24 — `node tools/bench.mjs` on
`anime-1_2k.png` (2048 × 2048 from the Image-GS anime set), default settings (0.1 Gaussians / px, 3000 steps),
2026-10-25:

| stage | result |
|---|---|
| OT initialisation | 419 256 Gaussians (647 × 648 mesh), 256 × 256 transport grid, 156 Sinkhorn iterations: **0.52 s** (density 0.36 s on the CPU, transport 0.04 s, parameters 0.11 s) |
| fit, 3000 steps | **5.3 s = 565 steps/s**, 52.75 dB (51.8 dB after 500 steps, 52.5 after 1500); 1.51 M tile entries = 3.6 per Gaussian |
| export 4× with edges | 8192 × 8192 in **1.27 s** (2048-px tiles), before PNG encoding |
| native render + full read-back | 83 ms per frame (dominated by reading 64 MB back to the CPU; the viewer does not read back) |

### The tiers

The same machine and engine, 2026-11-09: the fit time (the OT initialisation adds 0.1 s for the sample and 0.3–0.5 s
for the 2048 × 2048 frame) and the PSNR at the end.

| tier | budget, steps | sample, 1024 × 815 | anime-1_2k, 2048 × 2048 |
|---|---|---|---|
| Turbo | 1/20 px, 400 | 0.11 s, 40.54 dB | 0.53 s, 49.02 dB |
| Fast | 1/20 px, 1500 | 0.53 s, 43.43 dB | 2.17 s, 50.14 dB |
| Standard | 1/10 px, 3000 | 0.99 s, 48.60 dB | 5.33 s, 52.75 dB |
| Best | 1/7 px, 4000 | 1.68 s, 51.23 dB | 8.76 s, 54.52 dB |

Turbo keeps the budget of Fast and cuts the steps. Cutting the budget instead does less for the time and more harm to
the result: 1/40 px with 400 steps took 0.09 s for the sample at 36.38 dB and 0.45 s for the frame at 47.17 dB.

For scale: the same image on the Intel UHD 770 integrated GPU (the browser's default on this dual-GPU machine) fits a
120 × 80 test image at 1160 steps/s; a 2-megapixel image takes a minute or two there.

Browser timings are expected to match the Dawn-in-Node numbers closely — it is the same Dawn running the same WGSL
on D3D12 — minus whatever the page does in between: the view is re-rendered every 50 steps, and the fit animation
draws on a 2-D canvas over it. Since 0.1.4 that canvas is drawn with hairlines, batched fills and at most 30 frames a
second, and left alone while blank, so the animation no longer holds the fit back (before, a fit ran at about a
third of its speed while the mesh was on screen). Turbo skips the animation altogether; with it, the fixed timeline
of the show (about four seconds) is the floor of the wait for small images. On phones the viewer draws at most
2 canvas pixels per CSS pixel.

## Scaling

- Initialisation: CPU density O(pixels) — it dominates for large images (0.36 s for 4 MP) — plus Sinkhorn
  O(n_long² · n_long · iterations) on the GPU: 40 ms for a 256 × 256 grid.
- Fit: each step costs O(tile entries × 256) Gaussian–pixel evaluations in the forward and the backward pass; the
  entries are ≈ 3.6 N at the default budget — and per pixel: the render, the loss and the backward pass touch every
  pixel. At these budgets the pixels dominate: at 2048 × 2048, halving the budget from 1/10 to 1/20 per pixel cuts
  the step time by about a fifth (1.78 → 1.45 ms). The steps are linear, which is why Turbo cuts the steps.
- Export: linear in output pixels; the edge pass adds about 1 ms per megapixel.

## Memory

Fit: ≈ 400 B per Gaussian (parameters, gradient, Adam moments, projection, tile entries) + 32 B per pixel (target
and render). Viewer: the model (36 B per Gaussian) + the projection (72 B per Gaussian) + three output rasters of the
canvas size. Exporter: one 2048 × 2048 tile (64 MB) + the 8-bit result (4 B per output pixel).

## Running the benchmark

```bash
python reference/dump_image.py image.png image.rgb.bin          # raw float RGB with a 16-byte header
node tools/bench.mjs image.rgb.bin --adapter=4090 --kpp=0.1 --steps=3000 --scale=4 [--out=render.rgba]
```

`--adapter` selects the Dawn adapter by a part of its name (omit it for the system default). The script prints the
stage timings and a JSON summary line. Performance numbers for this document are taken on one fixed card; other
machines are reported with their card name.
