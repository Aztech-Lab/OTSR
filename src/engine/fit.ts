// The fitting loop: build tile lists -> forward -> (loss) -> backward -> Adam, `steps` times, with progress reports,
// cancellation and automatic growth of the tile-entry capacity.

import type { GpuContext } from './device';
import type { Pipelines } from './pipelines';
import { FLOATS_PER_GAUSSIAN, STEPS } from '../core/spec';
import { Fitter } from './fitter';
import { readBuffer, uploadBuffer } from './gpu';
import { Rasterizer } from './raster';

export interface FitProgress {
  step: number;
  steps: number;
  psnr: number;
  mse: number;
  elapsedMs: number;
}

export interface FitOptions {
  steps?: number;
  /** Report PSNR every this many steps (0: only at the end). */
  logEvery?: number;
  onProgress?: (p: FitProgress) => void | Promise<void>;
  signal?: AbortSignal;
  /** Steps between capacity checks when not `exact` (also throttles how far the CPU runs ahead of the GPU). */
  checkEvery?: number;
  /** Check the tile-entry capacity synchronously before every forward pass (no step ever runs with truncated lists). */
  exact?: boolean;
  /** Pixel loss: 'l2' (mean squared error, the reference) or 'l1' (Charbonnier, eps 1e-3). Default 'l2'. */
  loss?: 'l2' | 'l1';
}

export interface FitResult {
  params: Float32Array;
  curve: FitProgress[];
  seconds: number;
  entries: number;
  entryCap: number;
  /** True if the fit stopped early through `signal`. */
  cancelled: boolean;
}

/** Initial parameters: a CPU array (uploaded and released by runFit) or a GPU buffer optimised in place. */
export type FitSource = Float32Array | { buffer: GPUBuffer; count: number };

/**
 * Fits the Gaussians to the W x H image `gtRgba` (W x H x 4 floats in [0, 1]) on the GPU. The parameters are
 * optimised in place on the GPU; a CPU copy of the result is returned.
 */
export async function runFit(ctx: GpuContext, pipes: Pipelines, source: FitSource, W: number, H: number, gtRgba: Float32Array, opts: FitOptions = {}): Promise<FitResult> {
  const device = ctx.device;
  const own = source instanceof Float32Array;
  const N = own ? source.length / FLOATS_PER_GAUSSIAN : source.count;
  if (!Number.isInteger(N) || N <= 0) throw new Error('invalid Gaussian count');
  const steps = opts.steps ?? STEPS;
  const logEvery = opts.logEvery ?? 25;
  const checkEvery = Math.max(1, opts.checkEvery ?? 10);
  const params = own ? uploadBuffer(device, source, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 'params') : source.buffer;
  const raster = new Rasterizer(ctx, pipes, params, N, W, H, Math.max(4096, N));
  const fitter = new Fitter(ctx, pipes, raster, gtRgba);
  raster.setView({ srcW: W, srcH: H, zoom: 1, offsetX: 0, offsetY: 0, aa: 0, outW: W, outH: H }, fitter.gradScale, opts.loss === 'l1' ? 1 : 0);
  const curve: FitProgress[] = [];
  const t0 = performance.now();
  let entries = 0;

  const build = async (exact: boolean): Promise<void> => {
    const enc = device.createCommandEncoder();
    raster.encodeBuild(enc);
    device.queue.submit([enc.finish()]);
    if (!exact) return;
    entries = await raster.readTotalEntries();
    if (entries > raster.entryCap) {
      raster.growEntries(Math.ceil(entries * 1.5));
      await build(false);
    }
  };

  await build(true);
  raster.growEntries(Math.ceil(entries * 1.5) + 1024);
  let cancelled = false;
  try {
    for (let it = 0; it <= steps; it++) {
      const log = (logEvery > 0 && it % logEvery === 0) || it === steps;
      if (opts.exact) await build(true);
      const enc = device.createCommandEncoder();
      if (!opts.exact) raster.encodeBuild(enc);
      raster.encodeForward(enc);
      if (log) fitter.encodeLoss(enc);
      if (it < steps) {
        fitter.encodeBackward(enc);
        fitter.encodeAdam(enc, it + 1, steps);
      }
      device.queue.submit([enc.finish()]);
      if (log) {
        const l = await fitter.readLoss();
        const p: FitProgress = { step: it, steps, psnr: l.psnr, mse: l.mseRaw, elapsedMs: performance.now() - t0 };
        curve.push(p);
        if (opts.onProgress) await opts.onProgress(p);
      } else if (!opts.exact && it % checkEvery === 0) {
        entries = await raster.readTotalEntries();
        if (entries > raster.entryCap) raster.growEntries(Math.ceil(entries * 1.5));
      }
      if (opts.signal?.aborted) {
        cancelled = true;
        break;
      }
    }
    const out = new Float32Array(await readBuffer(device, params, N * FLOATS_PER_GAUSSIAN * 4));
    return { params: out, curve, seconds: (performance.now() - t0) / 1000, entries, entryCap: raster.entryCap, cancelled };
  } finally {
    fitter.destroy();
    raster.destroy();
    if (own) params.destroy();
  }
}
