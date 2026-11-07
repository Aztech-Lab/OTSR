// Tile-binned rasterisation of one Gaussian set for one output raster (a "view").
//
// build:   preprocess (per Gaussian: splat data, tile rectangle, counts) -> scan tile counts -> scan per-Gaussian
//          counts -> fill (tile lists + each Gaussian's slots)
// forward: one workgroup per tile -> out (r, g, b, D)

import type { GpuContext } from './device';
import type { Pipelines } from './pipelines';
import { TILE } from '../core/spec';
import { ScanPlan } from './scan';
import { bindGroup, createBuffer, grid1d, readBuffer, runPass } from './gpu';

export interface ViewParams {
  srcW: number;
  srcH: number;
  /** Output pixels per source pixel. */
  zoom: number;
  /** Output-pixel offset of the source origin. */
  offsetX: number;
  offsetY: number;
  /** Isotropic prefilter std in output pixels (display only; 0 for fitting). */
  aa: number;
  outW: number;
  outH: number;
}

export const VIEW_BYTES = 64;

export function packView(v: ViewParams, count: number, entryCap: number, gradScale: number): ArrayBuffer {
  const ab = new ArrayBuffer(VIEW_BYTES);
  const f = new Float32Array(ab);
  const u = new Uint32Array(ab);
  f[0] = v.srcW;
  f[1] = v.srcH;
  f[2] = v.zoom;
  f[3] = Math.max(v.srcW, v.srcH);
  f[4] = v.offsetX;
  f[5] = v.offsetY;
  f[6] = v.aa;
  f[7] = gradScale;
  u[8] = v.outW;
  u[9] = v.outH;
  u[10] = Math.ceil(v.outW / TILE);
  u[11] = Math.ceil(v.outH / TILE);
  u[12] = count;
  u[13] = entryCap;
  return ab;
}

export class Rasterizer {
  readonly device: GPUDevice;
  readonly tilesX: number;
  readonly tilesY: number;
  readonly nTiles: number;
  readonly viewBuf: GPUBuffer;
  readonly splats: GPUBuffer;
  readonly rects: GPUBuffer;
  readonly gcount: GPUBuffer;
  readonly goffset: GPUBuffer;
  readonly tileCount: GPUBuffer;
  readonly tileStart: GPUBuffer;
  readonly tileCursor: GPUBuffer;
  readonly out: GPUBuffer;
  tileList!: GPUBuffer;
  entrySlot!: GPUBuffer;
  /** Incremented whenever the entry buffers are reallocated; dependants rebuild their bind groups. */
  entryVersion = 0;
  private view: ViewParams | null = null;
  private gradScale = 0;
  private readonly scanTiles: ScanPlan;
  private readonly scanG: ScanPlan;
  private bgPre: GPUBindGroup;
  private bgFill!: GPUBindGroup;
  private bgFwd!: GPUBindGroup;

  constructor(
    readonly ctx: GpuContext,
    readonly pipes: Pipelines,
    readonly params: GPUBuffer,
    readonly count: number,
    readonly outW: number,
    readonly outH: number,
    public entryCap: number,
  ) {
    const d = (this.device = ctx.device);
    this.tilesX = Math.ceil(outW / TILE);
    this.tilesY = Math.ceil(outH / TILE);
    this.nTiles = this.tilesX * this.tilesY;
    const S = GPUBufferUsage.STORAGE;
    const SC = S | GPUBufferUsage.COPY_SRC;
    const SD = S | GPUBufferUsage.COPY_DST;
    this.viewBuf = d.createBuffer({ label: 'view', size: VIEW_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.splats = createBuffer(d, count * 48, SC, 'splats');
    this.rects = createBuffer(d, count * 8, S, 'rects');
    this.gcount = createBuffer(d, count * 4, SC, 'gcount');
    this.goffset = createBuffer(d, count * 4, SC, 'goffset');
    this.tileCount = createBuffer(d, (this.nTiles + 1) * 4, SD, 'tileCount');
    this.tileStart = createBuffer(d, (this.nTiles + 1) * 4, SC, 'tileStart');
    this.tileCursor = createBuffer(d, this.nTiles * 4, SD, 'tileCursor');
    this.out = createBuffer(d, outW * outH * 16, SC, 'out');
    this.scanTiles = new ScanPlan(d, pipes, this.tileCount, this.tileStart, this.nTiles + 1);
    this.scanG = new ScanPlan(d, pipes, this.gcount, this.goffset, count);
    this.bgPre = bindGroup(d, pipes.preprocess, { 0: this.viewBuf, 1: params, 2: this.splats, 3: this.rects, 4: this.gcount, 5: this.tileCount }, 'preprocess');
    this.allocEntries(entryCap);
  }

  private allocEntries(cap: number): void {
    const d = this.device;
    this.tileList?.destroy();
    this.entrySlot?.destroy();
    this.entryCap = cap;
    this.tileList = createBuffer(d, cap * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'tileList');
    this.entrySlot = createBuffer(d, cap * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'entrySlot');
    this.bgFill = bindGroup(d, this.pipes.fill, {
      0: this.viewBuf, 1: this.rects, 2: this.gcount, 3: this.goffset, 4: this.tileStart, 5: this.tileCursor, 6: this.tileList, 7: this.entrySlot,
    }, 'fill');
    this.bgFwd = bindGroup(d, this.pipes.forward, { 0: this.viewBuf, 1: this.splats, 2: this.tileStart, 3: this.tileList, 4: this.out }, 'forward');
    this.entryVersion++;
    if (this.view) this.writeView();
  }

  /** Grows the tile-entry capacity (keeps the larger of `cap` and the current capacity). */
  growEntries(cap: number): void {
    if (cap > this.entryCap) this.allocEntries(cap);
  }

  setView(view: ViewParams, gradScale = 0): void {
    if (view.outW !== this.outW || view.outH !== this.outH) throw new Error('view size does not match the rasteriser');
    this.view = view;
    this.gradScale = gradScale;
    this.writeView();
  }

  private writeView(): void {
    this.device.queue.writeBuffer(this.viewBuf, 0, packView(this.view!, this.count, this.entryCap, this.gradScale));
  }

  /** Projects the Gaussians and builds the tile lists for the current view. */
  encodeBuild(enc: GPUCommandEncoder): void {
    enc.clearBuffer(this.tileCount);
    enc.clearBuffer(this.tileCursor);
    runPass(enc, this.pipes.preprocess, this.bgPre, grid1d(this.count), 'preprocess');
    this.scanTiles.encode(enc);
    this.scanG.encode(enc);
    runPass(enc, this.pipes.fill, this.bgFill, grid1d(this.count), 'fill');
  }

  encodeForward(enc: GPUCommandEncoder): void {
    runPass(enc, this.pipes.forward, this.bgFwd, [this.tilesX, this.tilesY], 'forward');
  }

  /** Total tile entries of the last build (tileStart[nTiles]); more than `entryCap` means lists were truncated. */
  async readTotalEntries(): Promise<number> {
    const ab = await readBuffer(this.device, this.tileStart, 4, this.nTiles * 4);
    return new Uint32Array(ab)[0];
  }

  destroy(): void {
    this.scanTiles.destroy();
    this.scanG.destroy();
    for (const b of [this.viewBuf, this.splats, this.rects, this.gcount, this.goffset, this.tileCount, this.tileStart, this.tileCursor, this.out, this.tileList, this.entrySlot]) b.destroy();
  }
}
