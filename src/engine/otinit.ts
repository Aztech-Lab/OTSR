// Optimal-transport initialisation of a Gaussian set for an image (reference/otsr_ref/init.py ot_init).

import { lamForSpacing, latticeDims, otGridShape, paramsFromMap, pixelDensity, pooledDensity } from '../core/ot_cpu';
import { K_PER_PX, nLongFor } from '../core/spec';
import type { GpuContext } from './device';
import type { Pipelines } from './pipelines';
import { transportMap } from './sinkhorn';

export interface OtInitOptions {
  /** Gaussian count (default: K_PER_PX x pixels); the lattice rounds it to gy x gx. */
  K?: number;
  /** Transport-grid cells on the long side (default: nLongFor(H, W)). */
  nLong?: number;
}

export interface OtInitResult {
  params: Float32Array;
  count: number;
  gy: number;
  gx: number;
  lam: number;
  nLong: number;
  grid: { ny: number; nx: number };
  rho: Float32Array;
  T: Float32Array;
  iters: number;
  eps: number;
  ms: { density: number; transport: number; params: number };
}

/** `rgb`: W x H x 3 floats in [0, 1]. */
export async function otInit(ctx: GpuContext, pipes: Pipelines, rgb: Float32Array, W: number, H: number, opts: OtInitOptions = {}): Promise<OtInitResult> {
  const L = Math.max(H, W);
  const K = opts.K ?? Math.round(K_PER_PX * H * W);
  const { gy, gx } = latticeDims(K, H, W);
  const hPx = Math.sqrt((H * W) / (gy * gx));
  const lam = lamForSpacing(hPx);
  const nLong = opts.nLong ?? nLongFor(H, W);
  const grid = otGridShape(H, W, nLong);
  let t = performance.now();
  const rho = pooledDensity(pixelDensity(rgb, W, H, lam), W, H, grid);
  const tDensity = performance.now() - t;
  t = performance.now();
  const { T, eps, iters } = await transportMap(ctx, pipes, rho, grid, W / L, H / L);
  const tTransport = performance.now() - t;
  t = performance.now();
  const { params } = paramsFromMap(rgb, W, H, gy, gx, T, grid);
  const tParams = performance.now() - t;
  return { params, count: gy * gx, gy, gx, lam, nLong, grid, rho, T, iters, eps, ms: { density: tDensity, transport: tTransport, params: tParams } };
}
