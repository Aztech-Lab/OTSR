// The numerical core of the report's live strip (Figure A) and of the two-dimensional equal-shares demo: the importance
// density, entropic optimal transport (log-domain Sinkhorn with ε-scaling on a separable cost), the Gaussians of the
// mesh, the weight-normalised blend with its exact gradients, Adam with the paper's schedule, rendering at any scale
// and the distance-field edge pass. Plain TypeScript on the CPU without the DOM: the same steps the WebGPU engine
// runs, written small enough to read. tests/unit/report-fit.test.ts checks the gradients against finite differences.

/** exp(−8): every Gaussian is cut at its 4σ ellipse (q = 8) and shifted so that it reaches zero there. */
export const E8 = Math.exp(-8);

/** Lets the browser (or the test runner) breathe between chunks of work. */
export const tick = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

/** PSNR in dB of a mean squared error of values in 0..1. */
export const psnr = (mse: number): number => 10 * Math.log10(1 / Math.max(mse, 1e-12));

const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);

/** A deterministic hash of an integer to 0..1. */
export function hash01(i: number): number {
  let x = Math.imul(i + 0x9e3779b9, 0x85ebca6b);
  x ^= x >>> 13;
  x = Math.imul(x, 0xc2b2ae35);
  x ^= x >>> 16;
  return (x >>> 0) / 4294967296;
}

/** A transport grid: NX × NY cells with centres xs, ys (pixels) and the density mass of every cell. */
export interface Grid { NX: number; NY: number; xs: Float64Array; ys: Float64Array; rho: Float64Array }

/**
 * The importance density of an RGB image (W × H × 3 values in 0..1) pooled to a grid of cells about `cell` pixels wide:
 * ρ = λ / (W H) + (1 − λ) g / Σg with g = |∂x Y| + |∂y Y| of the luma Y (forward differences, zero on the last column
 * and row). Also returns g per pixel.
 */
export function densityGrid(rgb: Float64Array, W: number, H: number, cell: number, lambda: number): Grid & { g: Float64Array } {
  const n = W * H;
  const Y = new Float64Array(n);
  for (let i = 0; i < n; i++) Y[i] = 0.299 * rgb[3 * i] + 0.587 * rgb[3 * i + 1] + 0.114 * rgb[3 * i + 2];
  const g = new Float64Array(n);
  let gs = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x;
    const v = (x < W - 1 ? Math.abs(Y[i + 1] - Y[i]) : 0) + (y < H - 1 ? Math.abs(Y[i + W] - Y[i]) : 0);
    g[i] = v;
    gs += v;
  }
  const NX = Math.max(4, Math.round(W / cell)), NY = Math.max(4, Math.round(H / cell));
  const rho = new Float64Array(NX * NY);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const v = lambda / n + (1 - lambda) * (gs > 0 ? g[y * W + x] / gs : 1 / n);
    rho[Math.min(NY - 1, Math.floor((y * NY) / H)) * NX + Math.min(NX - 1, Math.floor((x * NX) / W))] += v;
  }
  const xs = new Float64Array(NX), ys = new Float64Array(NY);
  for (let i = 0; i < NX; i++) xs[i] = ((i + 0.5) * W) / NX;
  for (let j = 0; j < NY; j++) ys[j] = ((j + 0.5) * H) / NY;
  return { NX, NY, xs, ys, rho, g };
}

/**
 * Entropic optimal transport from the uniform density on the grid to its ρ with the squared-distance cost, by
 * log-domain Sinkhorn iterations with ε-scaling: ε from `eps0`, ×0.6 per stage with 6 iterations each, then 30
 * iterations at `epsEnd`. The cost is separable, so every half-iteration is two one-dimensional log-sum-exp passes.
 * Returns the barycentric map T(x) = E[y | x] at every cell centre; `onStage(k, n)` follows the stages, and the function
 * yields to the browser between them.
 */
export async function sinkhornMap(grid: Grid, eps0: number, epsEnd: number, onStage?: (k: number, n: number) => void): Promise<{ TX: Float64Array; TY: Float64Array }> {
  const { NX, NY, xs, ys, rho } = grid;
  const n = NX * NY;
  const f = new Float64Array(n), g = new Float64Array(n), tmp = new Float64Array(n), t2 = new Float64Array(n);
  const logmu = Math.log(1 / n);
  const logrho = new Float64Array(n);
  for (let k = 0; k < n; k++) logrho[k] = Math.log(Math.max(rho[k], 1e-300));
  const rowX = new Float64Array(NX), rowY = new Float64Array(NY);
  // dst_a = ε log Σ_b exp((src_b − |x_a − x_b|²) / ε), separably: along x within every row, then along y
  const lse2 = (src: Float64Array, dst: Float64Array, eps: number) => {
    for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) {
      let m = -Infinity;
      for (let i2 = 0; i2 < NX; i2++) { const d = xs[i] - xs[i2]; const v = (src[j * NX + i2] - d * d) / eps; rowX[i2] = v; if (v > m) m = v; }
      let s = 0;
      for (let i2 = 0; i2 < NX; i2++) s += Math.exp(rowX[i2] - m);
      t2[j * NX + i] = eps * (m + Math.log(s));
    }
    for (let i = 0; i < NX; i++) for (let j = 0; j < NY; j++) {
      let m = -Infinity;
      for (let j2 = 0; j2 < NY; j2++) { const d = ys[j] - ys[j2]; const v = (t2[j2 * NX + i] - d * d) / eps; rowY[j2] = v; if (v > m) m = v; }
      let s = 0;
      for (let j2 = 0; j2 < NY; j2++) s += Math.exp(rowY[j2] - m);
      dst[j * NX + i] = eps * (m + Math.log(s));
    }
  };
  const iterate = (eps: number, iters: number) => {
    for (let it = 0; it < iters; it++) {
      lse2(g, tmp, eps);
      for (let k = 0; k < n; k++) f[k] = eps * logmu - tmp[k];
      lse2(f, tmp, eps);
      for (let k = 0; k < n; k++) g[k] = eps * logrho[k] - tmp[k];
    }
  };
  const stages = Math.max(0, Math.ceil(Math.log(epsEnd / eps0) / Math.log(0.6)));
  let eps = eps0;
  for (let k = 0; k < stages; k++) {
    iterate(eps, 6);
    eps *= 0.6;
    onStage?.(k + 1, stages + 1);
    await tick();
  }
  iterate(epsEnd, 30);
  onStage?.(stages + 1, stages + 1);
  // the barycentric projection of the plan π_ab ∝ exp((f_a + g_b − |x_a − x_b|²) / ε)
  const TX = new Float64Array(n), TY = new Float64Array(n), vals = new Float64Array(n);
  for (let a = 0; a < n; a++) {
    const xa = xs[a % NX], ya = ys[(a / NX) | 0];
    let m = -Infinity;
    for (let b = 0; b < n; b++) {
      const dx = xa - xs[b % NX], dy = ya - ys[(b / NX) | 0];
      const v = (f[a] + g[b] - dx * dx - dy * dy) / epsEnd;
      vals[b] = v;
      if (v > m) m = v;
    }
    let s = 0, sx = 0, sy = 0;
    for (let b = 0; b < n; b++) { const p = Math.exp(vals[b] - m); s += p; sx += p * xs[b % NX]; sy += p * ys[(b / NX) | 0]; }
    TX[a] = sx / s;
    TY[a] = sy / s;
  }
  return { TX, TY };
}

/**
 * The map at a point (pixels): the displacement T(x) − x, bilinear between the cell centres. Between the outermost
 * centres and the edge of the box its normal component fades to zero, because a transport of the box onto itself keeps
 * the boundary (the barycentric map is pulled inward there by the entropic blur, which would leave the border empty).
 */
export function mapAt(grid: Grid, TX: Float64Array, TY: Float64Array, x: number, y: number, W: number, H: number): [number, number] {
  const { NX, NY, xs, ys } = grid;
  const u = clamp((x * NX) / W - 0.5, 0, NX - 1.000001), v = clamp((y * NY) / H - 0.5, 0, NY - 1.000001);
  const i = Math.floor(u), j = Math.floor(v), fu = u - i, fv = v - j;
  const k = j * NX + i;
  const at = (A: Float64Array, origin: (kk: number) => number) => {
    const d = (kk: number) => A[kk] - origin(kk);
    return (d(k) * (1 - fu) + d(k + 1) * fu) * (1 - fv) + (d(k + NX) * (1 - fu) + d(k + NX + 1) * fu) * fv;
  };
  let dx = at(TX, (kk) => xs[kk % NX]);
  let dy = at(TY, (kk) => ys[(kk / NX) | 0]);
  if (x < xs[0]) dx *= Math.max(0, x / xs[0]);
  else if (x > xs[NX - 1]) dx *= Math.max(0, (W - x) / (W - xs[NX - 1]));
  if (y < ys[0]) dy *= Math.max(0, y / ys[0]);
  else if (y > ys[NY - 1]) dy *= Math.max(0, (H - y) / (H - ys[NY - 1]));
  return [clamp(x + dx, 0, W), clamp(y + dy, 0, H)];
}

/** The uniform mesh and its vertices after the map, both as (x, y) pairs in pixels, row by row. */
export interface Mesh { gx: number; gy: number; N: number; h: number; uni: Float64Array; pos: Float64Array }

/**
 * The mesh of N = gx × gy ≈ K vertices (gx = round(√(K W / H))), its vertices moved by the map, and one Gaussian per
 * vertex with the covariance (0.6 h)² J Jᵀ of the map's Jacobian J (central differences one grid cell wide), scales
 * of at least 0.35 px, the colour of the image at its centre and weight one. Parameters, 9 per Gaussian: x, y
 * (pixels), log σ₁, log σ₂ (pixels), the angle of the first axis, r, g, b, log w.
 */
export function meshGaussians(grid: Grid, TX: Float64Array, TY: Float64Array, rgb: Float64Array, W: number, H: number, K: number): { mesh: Mesh; P: Float64Array } {
  const gx = Math.max(2, Math.round(Math.sqrt((K * W) / H))), gy = Math.max(2, Math.round(K / gx)), N = gx * gy;
  const h = Math.sqrt((W * H) / N);
  const uni = new Float64Array(2 * N), pos = new Float64Array(2 * N), P = new Float64Array(9 * N);
  const dX = W / grid.NX, dY = H / grid.NY;
  const at = (x: number, y: number) => mapAt(grid, TX, TY, x, y, W, H);
  const sample = (x: number, y: number, ch: number) => {
    const u = clamp(x - 0.5, 0, W - 1), v = clamp(y - 0.5, 0, H - 1);
    const i = Math.min(W - 2, Math.floor(u)), j = Math.min(H - 2, Math.floor(v)), fu = u - i, fv = v - j;
    const p = (ii: number, jj: number) => rgb[3 * (jj * W + ii) + ch];
    return (p(i, j) * (1 - fu) + p(i + 1, j) * fu) * (1 - fv) + (p(i, j + 1) * (1 - fu) + p(i + 1, j + 1) * fu) * fv;
  };
  for (let r = 0; r < gy; r++) for (let c = 0; c < gx; c++) {
    const k = r * gx + c;
    const x = ((c + 0.5) * W) / gx, y = ((r + 0.5) * H) / gy;
    const [mx, my] = at(x, y);
    uni[2 * k] = x; uni[2 * k + 1] = y;
    pos[2 * k] = mx; pos[2 * k + 1] = my;
    const [ax, ay] = at(x + dX, y), [bx, by] = at(x - dX, y), [cx, cy] = at(x, y + dY), [ex, ey] = at(x, y - dY);
    const j11 = (ax - bx) / (2 * dX), j21 = (ay - by) / (2 * dX), j12 = (cx - ex) / (2 * dY), j22 = (cy - ey) / (2 * dY);
    const s0 = 0.6 * h;
    const m11 = s0 * s0 * (j11 * j11 + j12 * j12), m12 = s0 * s0 * (j11 * j21 + j12 * j22), m22 = s0 * s0 * (j21 * j21 + j22 * j22);
    const tr = m11 + m22, det = m11 * m22 - m12 * m12;
    const l1 = tr / 2 + Math.sqrt(Math.max(0, (tr * tr) / 4 - det)), l2 = Math.max(1e-9, tr - l1);
    const b = 9 * k;
    P[b] = mx;
    P[b + 1] = my;
    P[b + 2] = Math.log(Math.max(0.35, Math.sqrt(l1)));
    P[b + 3] = Math.log(Math.max(0.35, Math.sqrt(l2)));
    P[b + 4] = Math.atan2(l1 - m11, m12);
    for (let ch = 0; ch < 3; ch++) P[b + 5 + ch] = sample(mx, my, ch);
    P[b + 8] = 0;
  }
  return { mesh: { gx, gy, N, h, uni, pos }, P };
}

/** Learning rates per parameter (pixels for the centre), as in the paper: a quarter pixel, 0.02, 0.02, 0.01, 0.02. */
const LR = [0.25, 0.25, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.02];

/**
 * Fits Gaussians (9 parameters each, in place) to an image with the weight-normalised blend
 * out = Σ wᵢ αᵢ cᵢ / (Σ wᵢ αᵢ + ε), its exact gradients and Adam (β = 0.9, 0.99) with per-parameter learning rates on a
 * cosine schedule from 1 to 0.05, and the clamps of the engine.
 */
export class Fitter {
  readonly N: number;
  /** Adam steps taken. */
  t = 0;
  readonly grad: Float64Array;
  private readonly m1: Float64Array;
  private readonly m2: Float64Array;
  private readonly den: Float64Array;
  private readonly num: Float64Array;
  private readonly out: Float64Array;
  private readonly G: Float64Array;
  private readonly con: Float64Array;
  private readonly box: Int32Array;

  constructor(readonly P: Float64Array, readonly rgb: Float64Array, readonly W: number, readonly H: number) {
    this.N = P.length / 9;
    const n = W * H;
    this.grad = new Float64Array(P.length);
    this.m1 = new Float64Array(P.length);
    this.m2 = new Float64Array(P.length);
    this.den = new Float64Array(n);
    this.num = new Float64Array(3 * n);
    this.out = new Float64Array(3 * n);
    this.G = new Float64Array(3 * n);
    this.con = new Float64Array(4 * this.N);
    this.box = new Int32Array(4 * this.N);
  }

  /** The conic, the weight and the pixel box of the 4σ ellipse of every Gaussian. */
  private prepare(): void {
    const { P, con, box, W, H } = this;
    for (let k = 0; k < this.N; k++) {
      const b = 9 * k;
      const c = Math.cos(P[b + 4]), s = Math.sin(P[b + 4]);
      const vx = Math.exp(2 * P[b + 2]), vy = Math.exp(2 * P[b + 3]);
      const i1 = 1 / vx, i2 = 1 / vy;
      con[4 * k] = c * c * i1 + s * s * i2;
      con[4 * k + 1] = c * s * (i1 - i2);
      con[4 * k + 2] = s * s * i1 + c * c * i2;
      con[4 * k + 3] = Math.exp(P[b + 8]);
      const ex = 4 * Math.sqrt(c * c * vx + s * s * vy), ey = 4 * Math.sqrt(s * s * vx + c * c * vy);
      box[4 * k] = Math.max(0, Math.ceil(P[b] - ex - 0.5));
      box[4 * k + 1] = Math.min(W - 1, Math.floor(P[b] + ex - 0.5));
      box[4 * k + 2] = Math.max(0, Math.ceil(P[b + 1] - ey - 0.5));
      box[4 * k + 3] = Math.min(H - 1, Math.floor(P[b + 1] + ey - 0.5));
    }
  }

  /** The blend and its mean squared error against the image; keeps ∂loss/∂out for the backward pass. */
  private forward(): number {
    const { P, con, box, den, num, out, G, rgb, W, H } = this;
    den.fill(1e-4);
    num.fill(0);
    for (let k = 0; k < this.N; k++) {
      const b = 9 * k, a = con[4 * k], bq = con[4 * k + 1], d = con[4 * k + 2], w = con[4 * k + 3];
      const mx = P[b], my = P[b + 1], cr = P[b + 5], cg = P[b + 6], cb = P[b + 7];
      for (let y = box[4 * k + 2]; y <= box[4 * k + 3]; y++) {
        const dy = y + 0.5 - my;
        for (let x = box[4 * k]; x <= box[4 * k + 1]; x++) {
          const dx = x + 0.5 - mx;
          const q = 0.5 * (a * dx * dx + 2 * bq * dx * dy + d * dy * dy);
          if (q >= 8) continue;
          const u = w * (Math.exp(-q) - E8);
          const p = y * W + x;
          den[p] += u;
          num[3 * p] += u * cr;
          num[3 * p + 1] += u * cg;
          num[3 * p + 2] += u * cb;
        }
      }
    }
    const n = W * H;
    let se = 0;
    for (let p = 0; p < n; p++) for (let ch = 0; ch < 3; ch++) {
      const o = num[3 * p + ch] / den[p];
      out[3 * p + ch] = o;
      const diff = o - rgb[3 * p + ch];
      se += diff * diff;
      G[3 * p + ch] = (2 * diff) / (3 * n);
    }
    return se / (3 * n);
  }

  /**
   * The exact gradient of the MSE. With uᵢ = wᵢ αᵢ, D = Σ uᵢ + ε and G = ∂loss/∂out: ∂/∂cᵢ = G uᵢ / D,
   * Qᵢ = G · (cᵢ − out) / D = ∂loss/∂uᵢ, ∂/∂log wᵢ = Σ Qᵢ uᵢ, ∂/∂q = −Qᵢ wᵢ e^{−q}, then the chain rule through the
   * conic (a, b, d) to the centre, the two log-scales and the angle.
   */
  private backward(): void {
    const { P, con, box, den, out, G, grad, W } = this;
    for (let k = 0; k < this.N; k++) {
      const b = 9 * k, a = con[4 * k], bq = con[4 * k + 1], d = con[4 * k + 2], w = con[4 * k + 3];
      const mx = P[b], my = P[b + 1], cr = P[b + 5], cg = P[b + 6], cb = P[b + 7];
      let gmx = 0, gmy = 0, ga = 0, gb = 0, gd = 0, gr = 0, gg = 0, gbl = 0, glw = 0;
      for (let y = box[4 * k + 2]; y <= box[4 * k + 3]; y++) {
        const dy = y + 0.5 - my;
        for (let x = box[4 * k]; x <= box[4 * k + 1]; x++) {
          const dx = x + 0.5 - mx;
          const q = 0.5 * (a * dx * dx + 2 * bq * dx * dy + d * dy * dy);
          if (q >= 8) continue;
          const e = Math.exp(-q);
          const u = w * (e - E8);
          const p = y * W + x;
          const inv = 1 / den[p];
          const g0 = G[3 * p], g1 = G[3 * p + 1], g2 = G[3 * p + 2];
          gr += g0 * u * inv;
          gg += g1 * u * inv;
          gbl += g2 * u * inv;
          const Q = (g0 * (cr - out[3 * p]) + g1 * (cg - out[3 * p + 1]) + g2 * (cb - out[3 * p + 2])) * inv;
          glw += Q * u;
          const gq = -Q * w * e;
          gmx -= gq * (a * dx + bq * dy);
          gmy -= gq * (bq * dx + d * dy);
          ga += gq * 0.5 * dx * dx;
          gb += gq * dx * dy;
          gd += gq * 0.5 * dy * dy;
        }
      }
      // a = c² i₁ + s² i₂, b = c s (i₁ − i₂), d = s² i₁ + c² i₂ with iₖ = e^{−2 log σₖ}
      const c = Math.cos(P[b + 4]), s = Math.sin(P[b + 4]);
      const i1 = Math.exp(-2 * P[b + 2]), i2 = Math.exp(-2 * P[b + 3]);
      grad[b] = gmx;
      grad[b + 1] = gmy;
      grad[b + 2] = (ga * c * c + gb * c * s + gd * s * s) * -2 * i1;
      grad[b + 3] = (ga * s * s - gb * c * s + gd * c * c) * -2 * i2;
      grad[b + 4] = ga * 2 * c * s * (i2 - i1) + gb * (c * c - s * s) * (i1 - i2) + gd * 2 * c * s * (i1 - i2);
      grad[b + 5] = gr;
      grad[b + 6] = gg;
      grad[b + 7] = gbl;
      grad[b + 8] = glw;
    }
  }

  private adam(total: number): void {
    const { P, grad, m1, m2, W, H } = this;
    const t = ++this.t;
    const sched = 0.05 + 0.95 * 0.5 * (1 + Math.cos((Math.PI * (t - 1)) / Math.max(1, total)));
    const bc1 = 1 - 0.9 ** t, bc2 = 1 - 0.99 ** t;
    for (let k = 0; k < P.length; k++) {
      const gk = Number.isFinite(grad[k]) ? grad[k] : 0;
      m1[k] = 0.9 * m1[k] + 0.1 * gk;
      m2[k] = 0.99 * m2[k] + 0.01 * gk * gk;
      P[k] -= (LR[k % 9] * sched * (m1[k] / bc1)) / (Math.sqrt(m2[k] / bc2) + 1e-8);
    }
    const lo = Math.log(0.2), hi = Math.log(0.5 * Math.max(W, H));
    for (let k = 0; k < this.N; k++) {
      const b = 9 * k;
      P[b] = clamp(P[b], -0.02 * W, 1.02 * W);
      P[b + 1] = clamp(P[b + 1], -0.02 * H, 1.02 * H);
      P[b + 2] = clamp(P[b + 2], lo, hi);
      P[b + 3] = clamp(P[b + 3], lo, hi);
      for (let ch = 5; ch < 8; ch++) P[b + ch] = clamp(P[b + ch], 0, 1);
      P[b + 8] = clamp(P[b + 8], -8, 8);
    }
  }

  /** One Adam step of a run of `total` steps; returns the MSE before the step. */
  step(total: number): number {
    this.prepare();
    const mse = this.forward();
    this.backward();
    this.adam(total);
    return mse;
  }

  /** The MSE of the current parameters. */
  evaluate(): number {
    this.prepare();
    return this.forward();
  }

  /** The MSE of the current parameters and their gradient in `grad` (no step). */
  gradient(): number {
    this.prepare();
    const mse = this.forward();
    this.backward();
    return mse;
  }
}

/**
 * Renders the Gaussians on an outW × outH raster whose pixel (i, j) samples the image at (x0 + (i + ½) / S,
 * y0 + (j + ½) / S) in source pixels; `colours` (3 per Gaussian) replaces their own, as in the vector view. RGBA bytes.
 */
export function renderGaussians(P: Float64Array, x0: number, y0: number, S: number, outW: number, outH: number, colours?: Float64Array): Uint8ClampedArray<ArrayBuffer> {
  const N = P.length / 9, n = outW * outH;
  const den = new Float64Array(n).fill(1e-4), num = new Float64Array(3 * n);
  for (let k = 0; k < N; k++) {
    const b = 9 * k, mx = P[b], my = P[b + 1];
    const c = Math.cos(P[b + 4]), s = Math.sin(P[b + 4]);
    const vx = Math.exp(2 * P[b + 2]), vy = Math.exp(2 * P[b + 3]);
    const A = (c * c) / vx + (s * s) / vy, B = c * s * (1 / vx - 1 / vy), D = (s * s) / vx + (c * c) / vy, w = Math.exp(P[b + 8]);
    const ex = 4 * Math.sqrt(c * c * vx + s * s * vy), ey = 4 * Math.sqrt(s * s * vx + c * c * vy);
    const i0 = Math.max(0, Math.ceil((mx - ex - x0) * S - 0.5)), i1 = Math.min(outW - 1, Math.floor((mx + ex - x0) * S - 0.5));
    const j0 = Math.max(0, Math.ceil((my - ey - y0) * S - 0.5)), j1 = Math.min(outH - 1, Math.floor((my + ey - y0) * S - 0.5));
    if (i0 > i1 || j0 > j1) continue;
    const cr = colours ? colours[3 * k] : P[b + 5], cg = colours ? colours[3 * k + 1] : P[b + 6], cb = colours ? colours[3 * k + 2] : P[b + 7];
    for (let j = j0; j <= j1; j++) {
      const dy = y0 + (j + 0.5) / S - my;
      for (let i = i0; i <= i1; i++) {
        const dx = x0 + (i + 0.5) / S - mx;
        const q = 0.5 * (A * dx * dx + 2 * B * dx * dy + D * dy * dy);
        if (q >= 8) continue;
        const u = w * (Math.exp(-q) - E8);
        const p = j * outW + i;
        den[p] += u;
        num[3 * p] += u * cr;
        num[3 * p + 1] += u * cg;
        num[3 * p + 2] += u * cb;
      }
    }
  }
  const rgba = new Uint8ClampedArray(new ArrayBuffer(4 * n));
  for (let p = 0; p < n; p++) {
    rgba[4 * p] = (num[3 * p] / den[p]) * 255;
    rgba[4 * p + 1] = (num[3 * p + 1] / den[p]) * 255;
    rgba[4 * p + 2] = (num[3 * p + 2] / den[p]) * 255;
    rgba[4 * p + 3] = 255;
  }
  return rgba;
}

/** The vector view's colours: every Gaussian a random colour times its own luma (× 1.6), as the app's recolour kernel. */
export function vectorColours(P: Float64Array, seed = 7): Float64Array {
  const N = P.length / 9;
  const out = new Float64Array(3 * N);
  for (let k = 0; k < N; k++) {
    const b = 9 * k;
    const luma = 0.299 * P[b + 5] + 0.587 * P[b + 6] + 0.114 * P[b + 7];
    for (let ch = 0; ch < 3; ch++) out[3 * k + ch] = Math.min(1, (0.15 + 0.85 * hash01(3 * k + ch + 7919 * seed)) * luma * 1.6);
  }
  return out;
}

/**
 * The distance-field edge pass of the app at magnification S, on RGBA bytes: the darkest and brightest pixels by luma
 * in a window of radius round(0.75 S), the locally normalised field t, a smoothstep edge_px output pixels wide, blended
 * in where the local contrast is real.
 */
export function edgePass(src: Uint8ClampedArray, w: number, h: number, S: number, edgePx = 1.4): Uint8ClampedArray<ArrayBuffer> {
  const n = w * h;
  const Y = new Float32Array(n);
  for (let i = 0; i < n; i++) Y[i] = (0.299 * src[4 * i] + 0.587 * src[4 * i + 1] + 0.114 * src[4 * i + 2]) / 255;
  const r = Math.max(1, Math.round(0.75 * S));
  const hmin = new Int32Array(n), hmax = new Int32Array(n), vmin = new Int32Array(n), vmax = new Int32Array(n);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let mi = y * w + x, ma = mi;
    for (let k = -r; k <= r; k++) { const xx = x + k; if (xx < 0 || xx >= w) continue; const q = y * w + xx; if (Y[q] < Y[mi]) mi = q; if (Y[q] > Y[ma]) ma = q; }
    hmin[y * w + x] = mi;
    hmax[y * w + x] = ma;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let mi = hmin[y * w + x], ma = hmax[y * w + x];
    for (let k = -r; k <= r; k++) { const yy = y + k; if (yy < 0 || yy >= h) continue; const a = hmin[yy * w + x], b = hmax[yy * w + x]; if (Y[a] < Y[mi]) mi = a; if (Y[b] > Y[ma]) ma = b; }
    vmin[y * w + x] = mi;
    vmax[y * w + x] = ma;
  }
  const t = new Float32Array(n), range = new Float32Array(n);
  for (let i = 0; i < n; i++) { const lo = Y[vmin[i]], hi = Y[vmax[i]]; range[i] = hi - lo; t[i] = range[i] > 1e-6 ? (Y[i] - lo) / range[i] : 0.5; }
  const sm = (a: number, b: number, x: number) => { const u = clamp((x - a) / Math.max(1e-6, b - a), 0, 1); return u * u * (3 - 2 * u); };
  const out = new Uint8ClampedArray(new ArrayBuffer(4 * n));
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    const gx = (t[y * w + Math.min(w - 1, x + 1)] - t[y * w + Math.max(0, x - 1)]) / 2, gy = (t[Math.min(h - 1, y + 1) * w + x] - t[Math.max(0, y - 1) * w + x]) / 2;
    const wd = 0.5 * edgePx * Math.hypot(gx, gy);
    const alpha = sm(0.5 - wd, 0.5 + wd, t[i]), gate = sm(0.08, 0.2, range[i]);
    const D = vmin[i], B = vmax[i];
    for (let c = 0; c < 3; c++) { const sharp = src[4 * D + c] + (src[4 * B + c] - src[4 * D + c]) * alpha; out[4 * i + c] = src[4 * i + c] + gate * (sharp - src[4 * i + c]); }
    out[4 * i + 3] = 255;
  }
  return out;
}
