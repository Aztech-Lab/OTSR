// Canvas presentation of storage images (with an optional split comparison), and the classic resampling of the
// source for comparison.

import type { GpuContext } from './device';
import type { Pipelines } from './pipelines';
import { bindGroup, runPass } from './gpu';
import { packView, type ViewParams } from './raster';
import presentSrc from './shaders/present.wgsl?raw';

export interface PresentOptions {
  /** Image rectangle in output pixels (x0, y0, x1, y1); outside it the background shows. Default: everything. */
  rect?: [number, number, number, number];
  /** Background colour (r, g, b in [0, 1]); default near-black. */
  background?: [number, number, number];
  /** Comparison image shown left of `split` (output pixels). */
  compare?: { img: GPUBuffer; split: number };
}

export class Presenter {
  readonly device: GPUDevice;
  private readonly pipeline: GPURenderPipeline;
  private readonly uni: GPUBuffer;

  constructor(readonly ctx: GpuContext, readonly format: GPUTextureFormat) {
    const d = (this.device = ctx.device);
    const module = d.createShaderModule({ label: 'present', code: presentSrc });
    this.pipeline = d.createRenderPipeline({
      label: 'present',
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    });
    this.uni = d.createBuffer({ label: 'present', size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  /** Draws `img` (width x height x vec4) onto `target` (same size). */
  encode(enc: GPUCommandEncoder, img: GPUBuffer, width: number, height: number, target: GPUTextureView, opts: PresentOptions = {}): void {
    const ab = new ArrayBuffer(48);
    const u = new Uint32Array(ab);
    const f = new Float32Array(ab);
    u[0] = width;
    u[1] = height;
    f[2] = opts.compare ? opts.compare.split : 0;
    const rect = opts.rect ?? [0, 0, width, height];
    const bgc = opts.background ?? [0.043, 0.047, 0.055];
    f.set(rect, 4);
    f.set([bgc[0], bgc[1], bgc[2], 1], 8);
    this.device.queue.writeBuffer(this.uni, 0, ab);
    const bg = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uni } },
        { binding: 1, resource: { buffer: img } },
        { binding: 2, resource: { buffer: opts.compare?.img ?? img } },
      ],
    });
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: target, loadOp: 'clear', clearValue: { r: bgc[0], g: bgc[1], b: bgc[2], a: 1 }, storeOp: 'store' }] });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bg);
    pass.draw(3);
    pass.end();
  }

  destroy(): void {
    this.uni.destroy();
  }
}

export type ResampleMode = 'nearest' | 'bilinear' | 'bicubic';
const MODES: Record<ResampleMode, number> = { nearest: 0, bilinear: 1, bicubic: 2 };

/** Resamples the source image (W x H x vec4 storage buffer) onto a View raster with a classic filter. */
export class Resampler {
  readonly device: GPUDevice;
  private readonly viewBuf: GPUBuffer;
  private readonly modeBuf: GPUBuffer;

  constructor(readonly ctx: GpuContext, readonly pipes: Pipelines) {
    const d = (this.device = ctx.device);
    this.viewBuf = d.createBuffer({ label: 'resample.view', size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.modeBuf = d.createBuffer({ label: 'resample.mode', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  encode(enc: GPUCommandEncoder, src: GPUBuffer, dst: GPUBuffer, view: ViewParams, mode: ResampleMode): void {
    this.device.queue.writeBuffer(this.viewBuf, 0, packView(view, 0, 0, 0));
    this.device.queue.writeBuffer(this.modeBuf, 0, new Uint32Array([MODES[mode], 0, 0, 0]));
    const bg = bindGroup(this.device, this.pipes.resample, { 0: this.viewBuf, 1: this.modeBuf, 2: src, 3: dst }, 'resample');
    runPass(enc, this.pipes.resample, bg, [Math.ceil(view.outW / 16), Math.ceil(view.outH / 16)], 'resample');
  }

  destroy(): void {
    this.viewBuf.destroy();
    this.modeBuf.destroy();
  }
}
