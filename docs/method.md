# The method

This is the mathematics the engine implements. `reference/otsr_ref/` is the same thing in PyTorch, and
`src/core/spec.ts` / `reference/otsr_ref/spec.py` hold every constant mentioned here.

## 1. The image model

An image of W × H pixels is represented by N Gaussians. Gaussian *i* has nine parameters:

| parameter | size | meaning |
|---|---|---|
| `uv` | 2 | centre in continuous image coordinates, (x, y) ∈ [0, 1]² ; pixel (row r, column c) is centred at ((c + ½)/W, (r + ½)/H) |
| `log_s` | 2 | log standard deviations along the two principal axes, in units of the long side L = max(W, H) |
| `rot` | 1 | rotation of the principal axes in radians |
| `color` | 3 | RGB |
| `log_w` | 1 | log weight |

At a point **x** (in source pixels) the Gaussian evaluates, with **m** = uv · (W, H), Σ = R diag(s²) Rᵀ and
R = [[cos θ, sin θ], [−sin θ, cos θ]]:

```
q(x)     = ½ (x − m)ᵀ Σ⁻¹ (x − m)
alpha(x) = max(exp(−q) − exp(−8), 0)          # support: the 4-sigma ellipse q < 8, continuous at its border
```

and the image is the weighted, normalised blend

```
out(x) = Σᵢ wᵢ alphaᵢ(x) cᵢ  /  (Σᵢ wᵢ alphaᵢ(x) + ε),      wᵢ = exp(log_wᵢ),  ε = 1e-4.
```

The blend is order-independent (no sorting, no depth), and because everything is in continuous coordinates the
same N Gaussians describe the image at every output resolution: rendering at scale S just samples `out` on a finer
grid. Because the fit is done at the native resolution, a Gaussian never has to be smaller than ~0.2 px; at 8× such
a Gaussian is a soft blob 1.6 output pixels wide, which is why the edge pass of section 4 exists.

**One deliberate difference from the research code.** The CUDA rasteriser used for the paper culls Gaussians per
16 × 16 tile at ⌈3σ⌉ (for fitting) or ⌈5σ⌉ (for the S× renders), so a Gaussian's footprint there depends on the
output resolution and jumps at tile borders. OTSR uses the exact 4σ cut-off above for fitting *and* display, which
makes the rendered function independent of the raster. The constant `exp(−8) = 3.4 × 10⁻⁴` shift keeps the function
continuous at the cut-off; at the default budget the fit reaches the same PSNR as the research renderer within the
run-to-run noise.

## 2. Optimal-transport initialisation

Good initial positions matter more than anything else for a 3000-step fit. OTSR starts from a regular mesh and
moves it with an optimal-transport map to an edge-importance density, so that Gaussians concentrate where the image
has detail while the mesh topology (no folds, no clumps) is preserved.

1. **Budget and mesh.** For a budget of K Gaussians, the mesh has gx = round(√(K W / H)) columns and
   gy = round(K / gx) rows (N = gy gx ≈ K), spacing h = √(W H / N) pixels.
2. **Density.** Luma Y = 0.299 R + 0.587 G + 0.114 B; g = |∂ₓY| + |∂ᵧY| by forward differences (zero on the last
   column / row). The target density is ρ = λ / (W H) + (1 − λ) g / Σg with the uniform share
   λ = min(0.3, max(0.1, 0.3 − 0.2 (log₂ h − 2))) (0.3 for dense meshes, 0.1 for sparse ones), averaged onto a
   transport grid of n_long = min(256, max(128, round(L / 4))) cells along the long side (aspect kept, ≥ 8 cells).
3. **Transport map.** Entropic OT from the uniform density to ρ on the box [0, W/L] × [0, H/L] with the quadratic
   cost, solved by log-domain Sinkhorn with ε-scaling: ε starts at the box size squared and shrinks by 0.6 per stage
   (6 iterations each) down to (1.5 cell)², where 30 final iterations run. The cost is separable, so each kernel
   application is two one-dimensional log-sum-exps. The map is the barycentric projection T(x) = E[y | x], an
   approximation of the Brenier map: the gradient of a convex potential, hence no shear and no folds.
4. **Gaussians from cells.** Each mesh vertex is moved by T (bilinear interpolation of the displacement field,
   clamped to the image). The Jacobian J = ∂T/∂x (central differences on the grid, interpolated likewise) deforms
   the cell; the Gaussian takes the covariance (0.6 h / L)² J Jᵀ, i.e. σ = 0.6 × the local spacing along each
   principal axis (eigendecomposition → scales and angle; scales clamped to ≥ 0.35 px), the bilinearly sampled image
   colour at its centre, and weight 1.

The Sinkhorn solver runs in float32 on the GPU. At the final ε the exponents are ~10⁴, where float32 keeps about
10⁻³ of absolute precision, so the map differs from a float64 solution by up to ~1.6 × 10⁻³ box units
(≈ 0.3 px on a 2 K image, rms 10× smaller) — the same as the float32 PyTorch solver used for the paper, where it was
measured to change the fitted PSNR by < 0.05 dB.

## 3. Fitting

Mean-squared-error loss over all 3 W H values, Adam (β = 0.9, 0.99, ε = 1e-8) with per-parameter learning rates

| uv | log_s | rot | color | log_w |
|---|---|---|---|---|
| 0.25 / L (a quarter pixel) | 0.02 | 0.02 | 0.01 | 0.02 |

each multiplied by a cosine schedule from 1 to 0.05 over the run (the factor for step *t* of *T* is
0.05 + 0.95 · ½ (1 + cos(π t / T)), evaluated before the step). After every step the parameters are clamped:
uv ∈ [−0.02, 1.02], log_s ∈ [log(0.2 / L), log 0.5], log_w ∈ [−8, 8].

The gradients are exact. With uᵢ = wᵢ alphaᵢ, D = Σ uᵢ + ε and G = ∂loss/∂out:

```
∂loss/∂cᵢ = G uᵢ / D
qᵢ        = Σ_ch G_ch (cᵢ,ch − out_ch) / D        (= ∂loss/∂uᵢ)
∂loss/∂wᵢ = Σₓ qᵢ alphaᵢ,     ∂loss/∂q = −qᵢ wᵢ exp(−q)   (inside the support)
∂q/∂m = (a dx + b dy, b dx + d dy),  ∂q/∂(a, b, d) = (dx²/2, dx dy, dy²/2)      with d = m − x, Σ⁻¹ = [[a, b], [b, d]]
```

then the chain rule to log_s and rot through a = c² i₁ + s² i₂, b = c s (i₂ − i₁), d = s² i₁ + c² i₂ with
iₖ = 1/sₖ², ∂iₖ/∂log_sₖ = −2 iₖ. These are the formulas the research code's PyTorch autograd produced; the WGSL
kernels reproduce them to ~10⁻⁶ relative (testing.md).

## 4. Distance-field edges

The Gaussian render at S× is correct but *soft* at edges: a 0.3-px-wide Gaussian becomes an S·0.3-px gradient. The
remedy is the one used to magnify signed-distance-field fonts (Green 2007), generalised from a binary coverage mask to
colour images:

1. In a window of radius r = round(0.75 S) around each output pixel, find the darkest and brightest pixels (by luma)
   and take their actual colours **D** and **B**.
2. t = (Y − Y_min) / (Y_max − Y_min) is a locally normalised field whose 0.5 level is the edge.
3. The field is re-thresholded with a smoothstep whose half-width w = ½ · edge_px · |∇t| (central differences) is
   the change of t over edge_px output pixels — so the transition is edge_px pixels wide at *every* magnification:
   alpha = smoothstep(½ − w, ½ + w, t), sharp = D + (B − D) alpha. All three channels share one curve, so no colour
   fringes appear.
4. The result is blended in only where the local contrast is real: gate = smoothstep(0.08, 0.2, Y_max − Y_min);
   out = img + gate (sharp − img). Flat regions and smooth gradients are untouched.

`edge_px = 1.4` by default (the UI slider), window 0.75, gate 0.08 / 0.2. The pass is a local post-process that
costs about a millisecond per megapixel, and it is applied to the exported image and, from 1.5× upwards, to the
viewer. It needs a clean continuous input (the Gaussian render); applied to a Lanczos upscale it would sharpen the
ringing too.
