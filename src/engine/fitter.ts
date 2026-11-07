// Gradient, Adam and loss passes on top of a Rasterizer whose view is the fit raster (zoom 1, out = source size).

import type { GpuContext } from './device';
import type { Pipelines } from './pipelines';
import type { Rasterizer } from './raster';
import { ADAM_EPS, BETAS, FLOATS_PER_GAUSSIAN, LOG_S_MAX, LOG_S_MIN_PX, LOG_W_CLAMP, LR, UV_CLAMP, lrScale } from '../core/spec';
import { bindGroup, createBuffer, grid1d, readBuffer, runPass, uploadBuffer } from './gpu';

const ADAM_BYTES = 176;
const BIG = 3.0e38;

/** Adam uniform for optimiser step `t` (1-based) of a fit with `steps` steps on an image with long side L. */
export function packAdam(t: number, steps: number, L: number, n: number): ArrayBuffer {
  const ab = new ArrayBuffer(ADAM_BYTES);
  const f = new Float32Array(ab);
  const u = new Uint32Array(ab);
  const k = lrScale(t - 1, steps);
  const lr = [LR.uv / L, LR.uv / L, LR.log_s, LR.log_s, LR.rot, LR.color, LR.color, LR.color, LR.log_w];
  const lsMin = Math.log(LOG_S_MIN_PX / L);
  const lo = [UV_CLAMP[0], UV_CLAMP[0], lsMin, lsMin, -BIG, -BIG, -BIG, -BIG, LOG_W_CLAMP[0]];
  const hi = [UV_CLAMP[1], UV_CLAMP[1], LOG_S_MAX, LOG_S_MAX, BIG, BIG, BIG, BIG, LOG_W_CLAMP[1]];
  for (let i = 0; i < 9; i++) {
    f[i] = lr[i] * k;
    f[12 + i] = lo[i];
    f[24 + i] = hi[i];
  }
  f[36] = 1 - Math.pow(BETAS[0], t);
  f[37] = Math.sqrt(1 - Math.pow(BETAS[1], t));
  f[38] = BETAS[0];
  f[39] = BETAS[1];
  f[40] = ADAM_EPS;
  u[41] = n;
  return ab;
}

/** Converts an interleaved RGB float image to the RGBA layout of the gt buffer. */
export function rgbToRgba(rgb: Float32Array, pixels: number): Float32Array {
  const out = new Float32Array(pixels * 4);
  for (let i = 0; i < pixels; i++) {
    out[4 * i] = rgb[3 * i];
    out[4 * i + 1] = rgb[3 * i + 1];
    out[4 * i + 2] = rgb[3 * i + 2];
    out[4 * i + 3] = 1;
  }
  return out;
}

export class Fitter {
  readonly device: GPUDevice;
  readonly gt: GPUBuffer;
  readonly grad: GPUBuffer;
  readonly m: GPUBuffer;
  readonly v: GPUBuffer;
  readonly adamBuf: GPUBuffer;
  readonly lossPartial: GPUBuffer;
  readonly lossGroups: number;
  private partial!: GPUBuffer;
  private partialVersion = -1;
  private bgBack!: GPUBindGroup;
  private bgReduce!: GPUBindGroup;
  private readonly bgAdam: GPUBindGroup;
  private readonly bgLoss: GPUBindGroup;
  readonly nFloats: number;

  /** `gtRgba`: the target image, W x H x 4 floats in [0, 1] (alpha unused). */
  constructor(readonly ctx: GpuContext, readonly pipes: Pipelines, readonly raster: Rasterizer, gtRgba: Float32Array) {
    const d = (this.device = ctx.device);
    const pixels = raster.outW * raster.outH;
    if (gtRgba.length !== pixels * 4) throw new Error('gt size does not match the rasteriser');
    this.nFloats = raster.count * FLOATS_PER_GAUSSIAN;
    const S = GPUBufferUsage.STORAGE;
    this.gt = uploadBuffer(d, gtRgba, S | GPUBufferUsage.COPY_DST, 'gt');
    this.grad = createBuffer(d, this.nFloats * 4, S | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 'grad');
    this.m = createBuffer(d, this.nFloats * 4, S | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 'adam.m');
    this.v = createBuffer(d, this.nFloats * 4, S | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 'adam.v');
    this.adamBuf = d.createBuffer({ label: 'adam', size: ADAM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.lossGroups = Math.ceil(pixels / 256);
    this.lossPartial = createBuffer(d, this.lossGroups * 8, S | GPUBufferUsage.COPY_SRC, 'loss.partial');
    this.bgAdam = bindGroup(d, pipes.adam, { 0: this.adamBuf, 1: raster.params, 2: this.grad, 3: this.m, 4: this.v }, 'adam');
    this.bgLoss = bindGroup(d, pipes.loss, { 0: raster.viewBuf, 1: raster.out, 2: this.gt, 3: this.lossPartial }, 'loss');
    this.sync();
  }

  /** d(mean squared error)/d(sum of squared errors). */
  get gradScale(): number {
    return 1 / (3 * this.raster.outW * this.raster.outH);
  }

  /** Rebuilds the entry-sized buffers after the rasteriser grew its capacity. */
  private sync(): void {
    if (this.partialVersion === this.raster.entryVersion) return;
    const d = this.device;
    const r = this.raster;
    this.partial?.destroy();
    this.partial = createBuffer(d, r.entryCap * 48, GPUBufferUsage.STORAGE, 'grad.partial');
    this.bgBack = bindGroup(d, this.pipes.backward, { 0: r.viewBuf, 1: r.splats, 2: r.tileStart, 3: r.tileList, 4: r.out, 5: this.gt, 6: this.partial }, 'backward');
    this.bgReduce = bindGroup(d, this.pipes.reduceGrad, { 0: r.viewBuf, 1: r.params, 2: r.gcount, 3: r.goffset, 4: r.entrySlot, 5: this.partial, 6: this.grad }, 'reduce_grad');
    this.partialVersion = r.entryVersion;
  }

  /** Gradient of the MSE loss w.r.t. all parameters, into `grad` (requires a built view and a forward pass). */
  encodeBackward(enc: GPUCommandEncoder): void {
    this.sync();
    const r = this.raster;
    runPass(enc, this.pipes.backward, this.bgBack, [r.tilesX, r.tilesY], 'backward');
    runPass(enc, this.pipes.reduceGrad, this.bgReduce, grid1d(r.count), 'reduce_grad');
  }

  /** Optimiser step `t` (1-based) of `steps`: Adam on `grad`, then the clamps. */
  encodeAdam(enc: GPUCommandEncoder, t: number, steps: number): void {
    const L = Math.max(this.raster.outW, this.raster.outH);
    this.device.queue.writeBuffer(this.adamBuf, 0, packAdam(t, steps, L, this.nFloats));
    runPass(enc, this.pipes.adam, this.bgAdam, grid1d(this.nFloats), 'adam');
  }

  encodeLoss(enc: GPUCommandEncoder): void {
    runPass(enc, this.pipes.loss, this.bgLoss, grid1d(this.lossGroups, 1), 'loss');
  }

  /** Mean squared errors of the last encodeLoss: of the clamped output (what PSNR reports) and of the raw output. */
  async readLoss(): Promise<{ mseClamped: number; mseRaw: number; psnr: number }> {
    const p = new Float32Array(await readBuffer(this.device, this.lossPartial, this.lossGroups * 8));
    let c = 0;
    let r = 0;
    for (let i = 0; i < this.lossGroups; i++) {
      c += p[2 * i];
      r += p[2 * i + 1];
    }
    const n = 3 * this.raster.outW * this.raster.outH;
    const mseClamped = c / n;
    return { mseClamped, mseRaw: r / n, psnr: -10 * Math.log10(Math.max(mseClamped, 1e-12)) };
  }

  resetMoments(): void {
    const enc = this.device.createCommandEncoder();
    enc.clearBuffer(this.m);
    enc.clearBuffer(this.v);
    this.device.queue.submit([enc.finish()]);
  }

  destroy(): void {
    for (const b of [this.gt, this.grad, this.m, this.v, this.adamBuf, this.lossPartial, this.partial]) b.destroy();
  }
}
