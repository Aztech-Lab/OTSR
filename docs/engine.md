# The WebGPU engine

`src/engine/` is a thin TypeScript layer that owns GPU buffers and encodes compute passes; all arithmetic is in the
WGSL kernels under `src/engine/shaders/`. `common.wgsl` (constants, the `View` struct, helpers) is prepended to every
kernel. Everything uses compute shaders and storage buffers; the only render pipeline is the one that puts the
final image on the canvas.

## Data layout

| buffer | size | contents |
|---|---|---|
| `params` | N × 9 f32 | the model, interleaved per Gaussian: uv(2) log_s(2) rot(1) color(3) log_w(1) |
| `splats` | N × 3 vec4 | per view: (mx, my, a, b), (d, w, –, –), (r, g, b, –) — centre and conic in output pixels |
| `rects`, `gcount`, `goffset` | N | tile rectangle (packed u16 pairs), number of tiles covered, exclusive prefix sum |
| `tileCount`, `tileStart`, `tileCursor` | T + 1 | Gaussians per tile, their prefix sum (`tileStart[T]` = total entries), fill cursors |
| `tileList`, `entrySlot` | cap | tile lists (Gaussian indices, grouped by tile) and, per Gaussian entry, its slot in `tileList` |
| `out` | W × H vec4 | the render: (r, g, b, D) with D = Σ w alpha + ε |
| `gt` | W × H vec4 | the target image (fit only) |
| `partial` | cap × 3 vec4 | per tile-list slot: the gradient contributions of that tile to that Gaussian |
| `grad`, `m`, `v` | N × 9 f32 | gradient and Adam moments |

A **View** (`raster.ts`) describes an output raster: source size, zoom (output pixels per source pixel), the offset of
the source origin in output pixels, the output size, and an optional isotropic prefilter `aa` (output pixels) used by
the viewer below native resolution. Output pixel (i, j) samples the image function at source position
((j + ½ − offset.x) / zoom, (i + ½ − offset.y) / zoom). The fit uses zoom 1 and offset 0; the viewer any zoom and
offset; the exporter zoom S and per-tile offsets.

## Passes

Per rendered frame (`Rasterizer.encodeBuild` + `encodeForward`):

1. **preprocess** — one thread per Gaussian: centre and conic in output pixels (with the optional `aa` added to the
   covariance and the weight rescaled by √(det Σ / det(Σ + aa² I)) so the Gaussian keeps its integral); the
   axis-aligned extent of the 4σ ellipse, K √Σₓₓ and K √Σᵧᵧ; the range of 16 × 16 tiles it touches; an
   `atomicAdd` into each tile's count. Non-finite or off-screen Gaussians get zero tiles.
2. **scan** — exclusive prefix sums of `tileCount` (→ `tileStart`) and `gcount` (→ `goffset`), blocks of 1024,
   recursive over block totals (`scan.ts`).
3. **fill** — one thread per Gaussian: appends its index to the list of every tile in its rectangle (counting sort via
   `tileCursor` atomics) and records the slot in its own CSR range of `entrySlot`. Slots beyond the capacity are
   dropped and recorded as `INVALID`.
4. **forward** — one workgroup (16 × 16 threads) per tile, one thread per pixel. The tile's Gaussians are streamed
   through workgroup memory in batches of 256 (centre, conic, weight, colour: 48 bytes each); every thread evaluates
   q for every Gaussian and accumulates Σ u c and Σ u where q < 8. Writes (rgb, D).

Per fit step (`Fitter`):

5. **loss** (every 25 steps) — per-workgroup partial sums of the squared error of the clamped and the raw output; the
   host adds the partials and reports PSNR.
6. **backward** — one workgroup per tile. It stages g = ∂loss/∂out / D and `out` for the tile's 256 pixels in
   workgroup memory, then assigns P threads to each Gaussian in the tile list (P is the largest power of two with
   n · P ≤ 256, so that all threads work even when a tile holds few Gaussians); each thread sums its share of the
   pixels into nine accumulators (∂m, ∂a, ∂b, ∂d, ∂c, ∂w), the P partial sums are reduced in workgroup memory and the
   result is written to `partial[slot]`. No atomics on floats.
7. **reduce_grad** — one thread per Gaussian: sums its entries' partials in CSR order (deterministic) and applies the
   chain rule to the nine parameters (method.md §3).
8. **adam** — one thread per parameter float: `torch.optim.Adam` semantics with bias correction, learning rate and
   clamp bounds per parameter type from a per-step uniform, then the clamps.

Other kernels: **sinkhorn** (`lse_x`, `lse_y`: the two one-dimensional log-sum-exps of the separable Sinkhorn update
and of the barycentric projection, with optional log-x / log-y weights; the ε-schedule and bind groups are built on
the host, the whole solve is one submit), **edges** (three passes: horizontal window min/max with colour indices,
vertical combination producing t and the range, final re-thresholding), **resample** (nearest / bilinear / Keys
bicubic of the source for the comparison views), **present** (full-screen triangle reading the storage image,
masking everything outside the image rectangle).

## Tile-entry capacity

The number of (tile, Gaussian) entries changes as Gaussians grow and shrink. `fill` never writes past the capacity;
the host reads `tileStart[T]` (one u32) to learn the true total. `runFit` sizes the buffers once from the first build
(×1.5 + 1024), checks every 10 steps asynchronously (this read-back also keeps the CPU from queueing hundreds of steps
ahead of the GPU) and grows them when needed; with `exact: true` (the tests) it checks synchronously before every
forward pass so no step ever runs with a truncated list. The viewer and the exporter check after each build.

## Numerics and limits

- Everything is float32. The forward pass agrees with a float64 dense reference to ~10⁻⁶ relative, the gradients to
  ~10⁻⁶, Adam to 10⁻⁷ (testing.md).
- Workgroup memory: the forward pass stages 12 KB, the backward pass 12 KB — under the 16 KB WebGPU default. Storage
  buffers per stage: at most 7 (the default limit is 8). The device requests the adapter's `maxBufferSize` and
  `maxStorageBufferBindingSize` (capped at 2 GB) and up to 10 storage buffers / 32 KB workgroup memory if available,
  but nothing above the defaults is required.
- Dispatches of more than 65535 workgroups are laid out as a 2-D grid and linearised in the kernel (`linear_id`).
- Non-finite gradients are treated as zero in `adam`; non-finite Gaussians are skipped in `preprocess`.
- Sums within a tile follow the (non-deterministic) order of the atomic fill, so repeated renders can differ in the
  last float bits; the per-Gaussian gradient reduction is deterministic given the lists.

## Memory

Per Gaussian during a fit: params, grad, m, v (36 B each), splats (48 B), rects, gcount, goffset (16 B) ≈ 208 B, plus
per tile entry 56 B (list, slot, partial) × ≈ 3.6 entries per Gaussian at the default budget ≈ 200 B; so about
400 B per Gaussian. Per pixel: `out` and `gt` (16 B each). A 2048 × 2048 image at 0.1 Gaussians per pixel therefore
needs about 170 MB + 134 MB ≈ 300 MB of GPU memory while fitting; viewing needs the model plus one output raster.
