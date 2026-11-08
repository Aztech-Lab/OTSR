# Tutorial: how the research code became a WebGPU engine

This is the working record of the port, written for people who have to do the same thing: take a
PyTorch/CUDA research pipeline and turn it into a verified, fast, dependency-free GPU implementation. It is
organised as the steps actually taken, with the derivations, the design decisions and the mistakes.

The whole port — reference, 13 kernels, UI, tests, docs — took one working day. Most of that time went into the
reference and the tests, not the kernels; that is the point of the tutorial.

## 0. Read the research code and write down the *semantics*, not the code

The research pipeline lived in a dozen files across experiments (`reference/upstream/` keeps them). The first task
was to extract the handful of definitions that actually determine the output and write them as formulas with every
constant pinned (`reference/otsr_ref/spec.py`):

- the parameterisation (9 floats per Gaussian, what units, what pixel-centre convention),
- the image function (the weighted-normalised blend, ε = 1e-4),
- the optimiser (Adam, per-group learning rates, cosine decay to 5 %, the three clamps),
- the initialisation (density, λ rule, transport-grid size, Sinkhorn schedule, σ = 0.6 h, 0.35 px floor),
- the edge pass (window 0.75 S, edge_px 1.4, gate 0.08/0.2).

Two rules helped: **take the constants from the final experiment scripts, not from library defaults** (the library
default of 2000 steps was overridden to 3000 everywhere that mattered), and **write each formula next to the file and
line it came from** so that later doubts can be settled by reading, not guessing.

## 1. Decide the one thing you are allowed to change

The research renderer culled Gaussians per 16 × 16 tile at ⌈3σ⌉ during fitting and ⌈5σ⌉ for S× renders. That is a
raster-dependent, tile-granular footprint: a Gaussian contributes its full `exp(−q)` to every pixel of every tile its
box touches, and the rendered *function* changes with the output resolution. For a product whose premise is
"resolution-independent", that had to go.

The replacement is the exact Mahalanobis cut-off `alpha = max(exp(−q) − exp(−8), 0)` (4σ), the same for fitting and
display, shifted so it reaches zero continuously. Everything else was kept bit-for-bit. One deliberate change,
documented in three places (method.md, upstream/README.md, the kernel comment), is manageable; several undocumented
"improvements" are how ports drift away from the thing they claim to implement.

## 2. Build the reference first, dense and in float64

`reference/otsr_ref/render.py` evaluates the image function with an O(N × pixels) loop over chunks of Gaussians —
no tiles, no culling, nothing clever, 40 lines. Because it is plain PyTorch, autograd gives the gradients for free
and the whole thing runs in float64. This is the oracle every kernel is measured against, and it was written *before*
any WGSL.

`reference/make_golden.py` then produces deterministic test cases: random Gaussians including degenerate ones
(0.2 px, off-screen, weight e⁶), a synthetic image with hard edges, an OT initialisation, 40 fitting steps, three
edge-pass runs. Each case is a JSON manifest + a binary blob of raw arrays (`goldenio.py`), 2.7 MB in total, so the
tests need no Python at run time.

## 3. Get a real GPU into the test runner

Headless Chrome was tried first and rejected for compute tests: it needs blocklist overrides, fails the first
`requestAdapter` intermittently, and picks the integrated GPU on this machine. Dawn's Node bindings
(`npm install webgpu`) give a standard `navigator.gpu` in Node with selectable adapters, so the kernels run under
Vitest exactly as they do in the browser (same Dawn, same D3D12 backend). Pitfalls met:

- Node 21+ has a read-only `globalThis.navigator` getter — replace it with `Object.defineProperty`.
- Dawn keeps the process alive while the GPU object is referenced; drop the reference in `afterAll`.
- A `.wgsl?raw` import works in Vitest because it goes through Vite's transform; the engine therefore never has to
  embed shader strings.

## 4. The rasteriser: forward pass

Structure (after gsplat / SAD): preprocess per Gaussian → prefix sums → counting sort into tile lists → one workgroup
per tile with the Gaussians streamed through workgroup memory in batches of 256. The `View` struct (zoom, offset,
output size, optional prefilter) makes the same kernels serve the fit (zoom 1), the viewer and the tiled export.

Details that mattered:

- The tile rectangle comes from the exact axis-aligned extent of the 4σ ellipse, K√Σₓₓ and K√Σᵧᵧ, so no pixel with
  alpha > 0 is ever missed; the forward pass then tests `q < 8` per pixel. Together they make the result *exactly*
  the dense reference, not an approximation of it.
- `tileCount` has T + 1 entries with a trailing zero, so the exclusive scan's last element is the total — one
  4-byte read-back tells the host whether the lists fit.
- Uniformity: loop bounds taken from a `storage, read` buffer at a uniform index are accepted by Tint's uniformity
  analysis, so `workgroupBarrier()` inside the batch loop compiles; the per-pixel `if (inside)` must stay barrier-free.

First golden comparison: render and D within 1.2 × 10⁻⁶ of float64. No debugging was needed, because the reference
and the convention (pixel centre at `uv·W − 0.5`) had been fixed in step 0.

## 5. The backward pass: derive, don't autograd

With u = w·alpha, D = Σu + ε, out = Σuc / D and G = ∂loss/∂out, the derivatives per Gaussian and pixel are

```
∂loss/∂c = G u / D
q_i      = Σ_ch G_ch (c_ch − out_ch) / D          (∂loss/∂u)
∂loss/∂w = q_i alpha
∂loss/∂q = −q_i w exp(−q)                        (inside the support; the shift has zero derivative)
∂q/∂m    = (a dx + b dy, b dx + d dy),  ∂q/∂(a,b,d) = (dx²/2, dx dy, dy²/2),  d = m − x
```

and the chain rule to the parameters is a per-Gaussian post-step:

```
i_k = 1/s_k²,  a = c² i₁ + s² i₂,  b = c s (i₂ − i₁),  d = s² i₁ + c² i₂
∂loss/∂i₁ = ∂a c² − ∂b c s + ∂d s²,   ∂loss/∂i₂ = ∂a s² + ∂b c s + ∂d c²,   ∂i_k/∂log_s_k = −2 i_k
∂loss/∂θ  = (i₂ − i₁)(2 c s ∂a + (c² − s²) ∂b − 2 c s ∂d)
∂loss/∂uv = ∂m · (W, H) · zoom,       ∂loss/∂log_w = ∂w · w
```

Design choice: **no float atomics**. SAD accumulates gradients with fixed-point `atomicAdd` on i32 (×10⁶); that is
fast but non-deterministic and quantised. Here the backward kernel works per tile: the tile's 256 values of g = G/D
and `out` are staged in workgroup memory, P threads share one Gaussian (P = largest power of two with n·P ≤ 256, so
the workgroup is busy even when a tile holds 10 Gaussians), each thread sums its share of the pixels into nine
registers, the P partials are tree-reduced in workgroup memory, and the result is written to that Gaussian's slot in
the tile list. A second kernel sums each Gaussian's slots in CSR order. Deterministic, 12 KB of workgroup memory,
no contention.

Budget the workgroup memory before writing: 9 floats × 256 threads for the reduction would have been 9 KB on top of
the 8 KB of pixel data — over the 16 KB default. Reducing three vec4 groups through one 4 KB array instead fits.

First golden comparison of all nine gradient columns: relative error ≤ 1.4 × 10⁻⁶. Again no debugging; the formulas
above were checked against float64 autograd in one shot.

## 6. Adam and the fit loop

`torch.optim.Adam` is reproduced literally: `p −= (lr / bc₁) · m / (√v / √bc₂ + ε)`, learning rate already multiplied
by the cosine factor of the *previous* iteration index (the research loop sets the rate at the top of iteration `it`
and steps at its end), then the clamps. The test feeds three steps of synthetic gradients, including values that push
parameters over every clamp, and compares parameters and both moments: 10⁻⁷.

**The one real bug of the port** showed up here: the Adam test wrote the gradients with `queue.writeBuffer` into a
buffer created without `COPY_DST`. WebGPU silently drops the write (a validation error goes to the error scope), so
the kernel ran on zeros and every parameter was off by exactly one learning-rate step. Lesson: wrap test bodies in
`pushErrorScope`/`popErrorScope` and **report the GPU error before the assertion that caught its consequence** — the
assertion said "0.005 too far", the error scope said "buffer usage does not include COPY_DST".

The fit loop batches build → forward → (loss) → backward → Adam into one submit per step and lets the CPU run ahead;
the periodic loss read-back is the throttle. The 40-step golden fit matches the reference's PSNR curve to four
decimals, which also shows that the tile capacity logic (grow when the read-back says the lists overflowed) is sound.

## 7. Sinkhorn on the GPU, and what "correct" means in float32

The transport map needs ~150 iterations of a separable log-domain Sinkhorn update; each half-update is two
one-dimensional log-sum-exps (`lse_x`, `lse_y`), each a thread per grid cell with an online max-rescaled sum. The
ε-schedule and the bind groups are prepared on the host and the entire solve is one command buffer: 40 ms on a
256 × 256 grid.

The golden comparison failed at first: the map differed from torch's by up to 1.9 × 10⁻³ box units. The cause is
physical, not a bug: at the final ε ≈ 1.4 × 10⁻⁴ the exponents are ~10⁴, where float32 has ~5 × 10⁻⁴ of absolute
precision, so any two float32 implementations differ at that level. The fix was in the *test*: the reference now
also solves in float64, and the GPU map must be at least as close to the float64 solution as torch's float32 map is
(it is: 1.60 × 10⁻³ vs 1.59 × 10⁻³). Downstream tolerances were then derived rather than tuned — positions inherit
the map noise, shapes inherit it amplified by the finite-difference stencil, and colours are bounded by the bilinear
interpolant's Lipschitz constant (largest neighbour difference of the image) times the position noise, with a
2 × 10⁻⁵ allowance for torch sampling coordinates in float32.

General lesson: when a float32 GPU result disagrees with a float32 CPU result, ask what the float64 answer is before
touching the kernel.

## 8. The edge pass and the export

The edge pass is three separable passes (horizontal window min/max carrying pixel indices, vertical combination,
re-thresholding); ties resolve to the first pixel in row-major order so that the colours D and B match
`torch.max_pool2d`'s index choice. It matched to 8 × 10⁻⁷ on the first run.

The exporter renders 2048-pixel tiles with an overlap of the edge window plus one pixel, so the tiled result equals
an untiled render — the test compares the two (≤ 1 in 8 bits, from rounding at .5 boundaries) instead of needing
another Python oracle.

## 9. The UI, and headless Chrome's three lies

The viewer re-renders the model at the exact zoom on every frame; the fit loop shares the model's GPU buffer, so the
view updates live. Nothing special — except testing it. Headless Chrome (1) ignores `powerPreference` on Windows and
runs on the integrated GPU, (2) fails the first `requestAdapter` after start-up sometimes (the app now retries), and
(3) does not composite WebGPU canvases into screenshots or `drawImage`, which made a working viewer look blank. The
diagnosis was to read the intermediate GPU buffers and to render the presenter into an offscreen texture: all
correct. The smoke test therefore checks `viewer.snapshot()` and saves it as PNG — look at the image, do not trust
the screenshot.

## 10. Measure against the original, then against the competitor

Two kinds of comparison closed the loop:

- **Same protocol, research CUDA pipeline vs WebGPU engine** on Kodak (OT init, 0.1/px, 3000 steps, same 4090):
  43.62 vs 43.55 dB and 46.49 vs 46.48 dB — equivalent within the noise of two independent fits — in 33 s vs 0.6 s.
  The 60× is not the kernels; it is the absence of Python, autograd and per-kernel synchronisation.
- **Same GPU and budget, SAD's official CUDA trainer vs the engine** (5070 Ti, SAD's initialisation not even
  counted): higher final PSNR on all four images (+1.7 to +6.6 dB) in 38–60 % of the time, with the time-PSNR curve
  above SAD's from 0.2 s onwards. The gain in quality is the representation and the OT initialisation; the gain in
  time is the pipeline (no sorting, one-pass exact gradients, no densify/prune machinery).

## Checklist for the next port

1. Extract the semantics into a spec file with sources; pin every constant from the final experiments.
2. Decide and document the one semantic change you make (if any). Keep everything else.
3. Write the dense float64 reference and the golden generator before the first kernel.
4. Get the kernels under a test runner on a real GPU (Dawn in Node), with error scopes that report GPU errors first.
5. Derive gradients by hand and test every column against autograd; prefer per-slot partials + deterministic
   reduction over float atomics.
6. Budget workgroup memory and storage bindings against the WebGPU *defaults* (16 KB, 8), not your card.
7. When float32 disagrees with float32, compute float64 and judge both against it; derive tolerances, don't tune them.
8. Test invariants where no oracle exists (tiled = untiled; resampler at zoom 1 = identity).
9. Never trust a headless screenshot of a GPU canvas; read buffers.
10. Finish with two comparisons: against the original pipeline (equivalence) and against the baseline (the claim).
