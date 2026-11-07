// CPU parts of the optimal-transport initialisation (reference/otsr_ref/init.py): densities, the lattice, the
// warp of the lattice by the transport map, the eigendecomposition of the deformed cells and colour sampling.
// The Sinkhorn solver itself runs on the GPU (engine/sinkhorn.ts).

import { LUMA, MIN_SCALE_PX, SIGMA_REL, pyRound } from './spec';

export interface Grid {
  ny: number;
  nx: number;
}

/** Gy x Gx grid with Gy/Gx ~ H/W and Gy*Gx ~ N. */
export function latticeDims(N: number, H: number, W: number): { gy: number; gx: number } {
  const gx = Math.max(1, pyRound(Math.sqrt((N * W) / H)));
  const gy = Math.max(1, pyRound(N / gx));
  return { gy, gx };
}

/** Lattice cell centres, (x, y) in [0, 1]^2, row-major. */
export function latticeUv(gy: number, gx: number): Float64Array {
  const uv = new Float64Array(gy * gx * 2);
  for (let i = 0; i < gy; i++) {
    for (let j = 0; j < gx; j++) {
      uv[2 * (i * gx + j)] = (j + 0.5) / gx;
      uv[2 * (i * gx + j) + 1] = (i + 0.5) / gy;
    }
  }
  return uv;
}

/** Uniform share of the importance density by lattice spacing h (px): 0.3 at h <= 4, 0.1 at h >= 8. */
export function lamForSpacing(hPx: number): number {
  return Math.min(0.3, Math.max(0.1, 0.3 - 0.2 * (Math.log2(Math.max(hPx, 1e-6)) - 2)));
}

export function otGridShape(H: number, W: number, nLong: number): Grid {
  const L = Math.max(H, W);
  return { ny: Math.max(8, pyRound((nLong * H) / L)), nx: Math.max(8, pyRound((nLong * W) / L)) };
}

/** Image-GS importance per pixel (sums to 1): lam uniform + (1 - lam) (|dx| + |dy|) of luma, forward differences. */
export function pixelDensity(rgb: ArrayLike<number>, W: number, H: number, lam: number): Float64Array {
  const y = new Float64Array(W * H);
  for (let i = 0; i < W * H; i++) y[i] = LUMA[0] * rgb[3 * i] + LUMA[1] * rgb[3 * i + 1] + LUMA[2] * rgb[3 * i + 2];
  const g = new Float64Array(W * H);
  let sum = 0;
  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      const i = r * W + c;
      const gx = c + 1 < W ? Math.abs(y[i + 1] - y[i]) : 0;
      const gy = r + 1 < H ? Math.abs(y[i + W] - y[i]) : 0;
      g[i] = gx + gy;
      sum += g[i];
    }
  }
  const inv = (1 - lam) / Math.max(sum, 1e-12);
  const u = lam / (W * H);
  for (let i = 0; i < W * H; i++) g[i] = u + g[i] * inv;
  return g;
}

/** torch adaptive_avg_pool2d of a W x H map to ny x nx, normalised to sum 1. */
export function pooledDensity(p: Float64Array, W: number, H: number, grid: Grid): Float32Array {
  const { ny, nx } = grid;
  const out = new Float64Array(ny * nx);
  let total = 0;
  for (let i = 0; i < ny; i++) {
    const r0 = Math.floor((i * H) / ny);
    const r1 = Math.ceil(((i + 1) * H) / ny);
    for (let j = 0; j < nx; j++) {
      const c0 = Math.floor((j * W) / nx);
      const c1 = Math.ceil(((j + 1) * W) / nx);
      let s = 0;
      for (let r = r0; r < r1; r++) for (let c = c0; c < c1; c++) s += p[r * W + c];
      out[i * nx + j] = s / ((r1 - r0) * (c1 - c0));
      total += out[i * nx + j];
    }
  }
  const rho = new Float32Array(ny * nx);
  for (let i = 0; i < ny * nx; i++) rho[i] = out[i] / total;
  return rho;
}

/**
 * Bilinear sample of a C-channel ny x nx field at (x, y) with torch grid_sample semantics
 * (align_corners=False, padding_mode='border'): continuous coordinates u in [0, 1] map to u * n - 0.5.
 */
export function sampleBilinear(field: ArrayLike<number>, nx: number, ny: number, C: number, u: number, v: number, out: Float64Array, o: number): void {
  const x = Math.min(Math.max(u * nx - 0.5, 0), nx - 1);
  const y = Math.min(Math.max(v * ny - 0.5, 0), ny - 1);
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(x0 + 1, nx - 1);
  const y1 = Math.min(y0 + 1, ny - 1);
  const fx = x - x0;
  const fy = y - y0;
  for (let c = 0; c < C; c++) {
    const v00 = field[(y0 * nx + x0) * C + c];
    const v01 = field[(y0 * nx + x1) * C + c];
    const v10 = field[(y1 * nx + x0) * C + c];
    const v11 = field[(y1 * nx + x1) * C + c];
    out[o + c] = (1 - fy) * ((1 - fx) * v00 + fx * v01) + fy * ((1 - fx) * v10 + fx * v11);
  }
}

/**
 * Moves lattice points by the transport map T (ny x nx x 2, box units, at the cell centres) and returns the new
 * uv (n x 2) and the Jacobian dT/dx (n x 4: dTx/dx, dTx/dy, dTy/dx, dTy/dy) in box units, both bilinearly
 * interpolated from the grid (the Jacobian by central differences, one-sided at the border, as torch.gradient).
 */
export function warpPoints(uv: Float64Array, T: ArrayLike<number>, grid: Grid, ex: number, ey: number): { uv: Float64Array; J: Float64Array } {
  const { ny, nx } = grid;
  const D = new Float64Array(ny * nx * 2);
  const Jg = new Float64Array(ny * nx * 4);
  const hx = ex / nx;
  const hy = ey / ny;
  for (let i = 0; i < ny; i++) {
    for (let j = 0; j < nx; j++) {
      const k = i * nx + j;
      D[2 * k] = T[2 * k] - ((j + 0.5) / nx) * ex;
      D[2 * k + 1] = T[2 * k + 1] - ((i + 0.5) / ny) * ey;
      const jl = Math.max(j - 1, 0);
      const jr = Math.min(j + 1, nx - 1);
      const il = Math.max(i - 1, 0);
      const ir = Math.min(i + 1, ny - 1);
      const dxw = (jr - jl) * hx;
      const dyw = (ir - il) * hy;
      Jg[4 * k] = (T[2 * (i * nx + jr)] - T[2 * (i * nx + jl)]) / dxw;
      Jg[4 * k + 1] = (T[2 * (ir * nx + j)] - T[2 * (il * nx + j)]) / dyw;
      Jg[4 * k + 2] = (T[2 * (i * nx + jr) + 1] - T[2 * (i * nx + jl) + 1]) / dxw;
      Jg[4 * k + 3] = (T[2 * (ir * nx + j) + 1] - T[2 * (il * nx + j) + 1]) / dyw;
    }
  }
  const n = uv.length / 2;
  const out = new Float64Array(n * 2);
  const J = new Float64Array(n * 4);
  const d = new Float64Array(2);
  for (let p = 0; p < n; p++) {
    sampleBilinear(D, nx, ny, 2, uv[2 * p], uv[2 * p + 1], d, 0);
    sampleBilinear(Jg, nx, ny, 4, uv[2 * p], uv[2 * p + 1], J, 4 * p);
    out[2 * p] = Math.min(Math.max((uv[2 * p] * ex + d[0]) / ex, 0), 1);
    out[2 * p + 1] = Math.min(Math.max((uv[2 * p + 1] * ey + d[1]) / ey, 0), 1);
  }
  return { uv: out, J };
}

/**
 * Scales (larger first) and kernel angle of the SPD matrix [[p, q], [q, r]]: Sigma = R diag(s^2) R^T with
 * R = [[c, s], [-s, c]], first principal axis (c, -s). The angle is defined up to pi (same ellipse).
 */
export function covToScaleRot(p: number, q: number, r: number): { s1: number; s2: number; theta: number } {
  const mean = 0.5 * (p + r);
  const half = Math.sqrt(Math.max(0.25 * (p - r) * (p - r) + q * q, 0));
  const l1 = mean + half;
  const l2 = mean - half;
  // eigenvector of l1: (q, l1 - p) or (l1 - r, q), whichever is better conditioned
  let vx: number;
  let vy: number;
  if (Math.abs(l1 - p) > Math.abs(l1 - r)) {
    vx = q;
    vy = l1 - p;
  } else {
    vx = l1 - r;
    vy = q;
  }
  if (vx === 0 && vy === 0) {
    vx = p >= r ? 1 : 0;
    vy = p >= r ? 0 : 1;
  }
  return { s1: Math.sqrt(Math.max(l1, 1e-12)), s2: Math.sqrt(Math.max(l2, 1e-12)), theta: Math.atan2(-vy, vx) };
}

/**
 * Parameters (N x 9) of the lattice moved by the transport map: centre, the deformed cell as covariance
 * (SIGMA_REL h)^2 J J^T, the image colour at the centre, unit weight.
 */
export function paramsFromMap(rgb: ArrayLike<number>, W: number, H: number, gy: number, gx: number, T: ArrayLike<number>, grid: Grid): { params: Float32Array; uv: Float64Array; J: Float64Array } {
  const L = Math.max(H, W);
  const { uv, J } = warpPoints(latticeUv(gy, gx), T, grid, W / L, H / L);
  const h = Math.sqrt((H * W) / (gy * gx)) / L;
  const k = SIGMA_REL * h;
  const n = gy * gx;
  const params = new Float32Array(n * 9);
  const col = new Float64Array(3);
  const sMin = MIN_SCALE_PX / L;
  for (let i = 0; i < n; i++) {
    const a = J[4 * i];
    const b = J[4 * i + 1];
    const c = J[4 * i + 2];
    const d = J[4 * i + 3];
    // C = k^2 J J^T
    const { s1, s2, theta } = covToScaleRot(k * k * (a * a + b * b), k * k * (a * c + b * d), k * k * (c * c + d * d));
    sampleBilinear(rgb, W, H, 3, uv[2 * i], uv[2 * i + 1], col, 0);
    params[9 * i] = uv[2 * i];
    params[9 * i + 1] = uv[2 * i + 1];
    params[9 * i + 2] = Math.log(Math.max(s1, sMin));
    params[9 * i + 3] = Math.log(Math.max(s2, sMin));
    params[9 * i + 4] = theta;
    params[9 * i + 5] = col[0];
    params[9 * i + 6] = col[1];
    params[9 * i + 7] = col[2];
    params[9 * i + 8] = 0;
  }
  return { params, uv, J };
}

/** The eps-scaling schedule of the Sinkhorn solver: (eps, iterations) per stage. */
export function sinkhornSchedule(grid: Grid, ex: number, ey: number, shrink: number, itersPerStage: number, finalIters: number): { e: number; iters: number }[] {
  const h = Math.max(ex / grid.nx, ey / grid.ny);
  const eps = (1.5 * h) ** 2;
  const stages: { e: number; iters: number }[] = [];
  let e = Math.max(ex, ey) ** 2;
  for (;;) {
    stages.push({ e, iters: e <= eps ? finalIters : itersPerStage });
    if (e <= eps) break;
    e = Math.max(eps, e * shrink);
  }
  return stages;
}
