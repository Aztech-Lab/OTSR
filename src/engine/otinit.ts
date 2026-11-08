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
  /** Progress of the three phases (density on the CPU, transport on the GPU, lattice parameters on the CPU). */
  onPhase?: (phase: 'density' | 'transport' | 'params', info: { grid: { ny: number; nx: number }; gy: number; gx: number; lam: number; ms: number }) => void;
  /** Every Sinkhorn eps-scaling stage with its barycentric map (ny x nx x 2, box units; x pixels = map x long side). */
  onStage?: (T: Float32Array, stage: number, stages: number, eps: number, grid: { ny: number; nx: number }) => void;
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
  opts.onPhase?.('density', { grid, gy, gx, lam, ms: tDensity });
  t = performance.now();
  const onStage = opts.onStage;
  const { T, eps, iters } = await transportMap(ctx, pipes, rho, grid, W / L, H / L, onStage ? { onStage: (m, s, n, e) => onStage(m, s, n, e, grid) } : {});
  const tTransport = performance.now() - t;
  opts.onPhase?.('transport', { grid, gy, gx, lam, ms: tTransport });
  t = performance.now();
  const { params } = paramsFromMap(rgb, W, H, gy, gx, T, grid);
  const tParams = performance.now() - t;
  opts.onPhase?.('params', { grid, gy, gx, lam, ms: tParams });
  return { params, count: gy * gx, gy, gx, lam, nLong, grid, rho, T, iters, eps, ms: { density: tDensity, transport: tTransport, params: tParams } };
}
