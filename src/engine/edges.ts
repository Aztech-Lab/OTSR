// The v3 distance-field edge pass on an S x render held in a (r, g, b, *) storage buffer.

import { EDGE_C0, EDGE_C1, EDGE_PX, edgeWindowRadius } from '../core/spec';
import type { GpuContext } from './device';
import type { Pipelines } from './pipelines';
import { bindGroup, createBuffer, grid1d, runPass } from './gpu';

export interface EdgeOptions {
  /** Transition width in output pixels (default 1.4; smaller = harder). */
  edgePx?: number;
  c0?: number;
  c1?: number;
}

export class EdgePass {
  readonly device: GPUDevice;
  private readonly uni: GPUBuffer;
  private readonly tmpH: GPUBuffer;
  private readonly tmpV: GPUBuffer;
  private readonly pixels: number;

  constructor(readonly ctx: GpuContext, readonly pipes: Pipelines, readonly width: number, readonly height: number) {
    const d = (this.device = ctx.device);
    this.pixels = width * height;
    this.uni = d.createBuffer({ label: 'edges', size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.tmpH = createBuffer(d, this.pixels * 16, GPUBufferUsage.STORAGE, 'edges.tmpH');
    this.tmpV = createBuffer(d, this.pixels * 16, GPUBufferUsage.STORAGE, 'edges.tmpV');
  }

  /** Sharpens `src` (an S x render) into `dst`; both width x height x vec4. */
  encode(enc: GPUCommandEncoder, src: GPUBuffer, dst: GPUBuffer, S: number, opts: EdgeOptions = {}): void {
    const ab = new ArrayBuffer(32);
    const u = new Uint32Array(ab);
    const f = new Float32Array(ab);
    u[0] = this.width;
    u[1] = this.height;
    u[2] = edgeWindowRadius(S);
    f[4] = opts.edgePx ?? EDGE_PX;
    f[5] = opts.c0 ?? EDGE_C0;
    f[6] = opts.c1 ?? EDGE_C1;
    this.device.queue.writeBuffer(this.uni, 0, ab);
    const d = this.device;
    const p = this.pipes;
    const n = grid1d(this.pixels);
    runPass(enc, p.edgeH, bindGroup(d, p.edgeH, { 0: this.uni, 1: src, 3: this.tmpH }), n, 'edge_h');
    runPass(enc, p.edgeV, bindGroup(d, p.edgeV, { 0: this.uni, 1: src, 2: this.tmpH, 3: this.tmpV }), n, 'edge_v');
    runPass(enc, p.edgeFinal, bindGroup(d, p.edgeFinal, { 0: this.uni, 1: src, 2: this.tmpV, 4: dst }), n, 'edge_final');
  }

  destroy(): void {
    this.uni.destroy();
    this.tmpH.destroy();
    this.tmpV.destroy();
  }
}
