// The monitor panel (right): the three regions of the image with the strongest gradient — fixed when the fit starts,
// since the image decides where the fit has the most to do — each shown as a live render of the model: in
// Gaussian-id colours (the vector view) while the fit runs, in true colours afterwards. The crops are rendered with
// the engine's rasteriser and edge pass.

import { EdgePass, Rasterizer, createBuffer, readBuffer, type GpuContext, type Pipelines } from '../engine';
import { bi, h } from './dom';
import type { ActiveRegion, AppModel, Store } from './store';
import type { Viewer } from './viewer';

const SLOTS = 3;
const CROP = 192; // crop render size in pixels
const LABELS = ['A', 'B', 'C'];

/** Renders a square window of the model at CROP x CROP pixels (with the edge pass at magnification). */
class CropRenderer {
  private readonly raster: Rasterizer;
  private readonly edge: EdgePass;
  private readonly dst: GPUBuffer;
  /** Set by retire(): a render in flight stops at its next await instead of submitting with buffers about to go. */
  private disposed = false;

  constructor(readonly ctx: GpuContext, readonly pipes: Pipelines, readonly params: GPUBuffer, readonly N: number, readonly W: number, readonly H: number) {
    this.raster = new Rasterizer(ctx, pipes, params, N, CROP, CROP, Math.max(65536, Math.min(N, 1 << 20)));
    this.edge = new EdgePass(ctx, pipes, CROP, CROP);
    this.dst = createBuffer(ctx.device, CROP * CROP * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'crop.dst');
  }

  /** Draws from another parameter buffer of the same count (null: the model's own). */
  setParams(buf: GPUBuffer | null): void {
    this.raster.setParams(buf ?? this.params);
  }

  async render(roi: { x: number; y: number; w: number; h: number }, edges: boolean, edgePx: number, pre: ((enc: GPUCommandEncoder) => void) | null = null): Promise<Uint8ClampedArray<ArrayBuffer>> {
    const zoom = CROP / Math.max(roi.w, roi.h);
    const view = { srcW: this.W, srcH: this.H, zoom, offsetX: -roi.x * zoom, offsetY: -roi.y * zoom, aa: zoom < 1 ? 0.5 * Math.sqrt(1 - zoom * zoom) : 0, outW: CROP, outH: CROP };
    const device = this.ctx.device;
    const useEdge = edges && zoom >= 1.5;
    for (let attempt = 0; attempt < 3; attempt++) {
      if (this.disposed) throw new Error('crop renderer retired');
      this.raster.setView(view);
      const enc = device.createCommandEncoder({ label: 'crop' });
      pre?.(enc);
      this.raster.encodeBuild(enc);
      this.raster.encodeForward(enc);
      if (useEdge) this.edge.encode(enc, this.raster.out, this.dst, zoom, { edgePx });
      device.queue.submit([enc.finish()]);
      const total = await this.raster.readTotalEntries();
      if (total <= this.raster.entryCap) break;
      this.raster.growEntries(Math.ceil(total * 1.5));
    }
    if (this.disposed) throw new Error('crop renderer retired');
    const f = new Float32Array(await readBuffer(device, useEdge ? this.dst : this.raster.out, CROP * CROP * 16));
    if (this.disposed) throw new Error('crop renderer retired');
    const out = new Uint8ClampedArray<ArrayBuffer>(new ArrayBuffer(CROP * CROP * 4));
    for (let i = 0; i < CROP * CROP; i++) {
      out[4 * i] = f[4 * i] * 255;
      out[4 * i + 1] = f[4 * i + 1] * 255;
      out[4 * i + 2] = f[4 * i + 2] * 255;
      out[4 * i + 3] = 255;
    }
    return out;
  }

  /** No further GPU work from this renderer; the buffers go with destroy() once no render is in flight. */
  retire(): void {
    this.disposed = true;
  }

  destroy(): void {
    this.disposed = true;
    this.raster.destroy();
    this.edge.destroy();
    this.dst.destroy();
  }
}

interface Card { root: HTMLElement; canvas: HTMLCanvasElement; meta: HTMLElement; bar: HTMLElement }

export class Monitor {
  private crop: CropRenderer | null = null;
  private model: AppModel | null = null;
  private cards: Card[];
  private rendering = false;
  private pendingRender = false;
  /** Renderers replaced while a render was in flight; destroyed once the render loop has finished. */
  private retired: CropRenderer[] = [];
  /** While the fit animation runs: the crops show its recoloured copy (the vector view) instead of the true colours. */
  private vector: { buf: GPUBuffer; pre: (enc: GPUCommandEncoder) => void } | null = null;

  constructor(readonly host: HTMLElement, readonly ctx: GpuContext, readonly pipes: Pipelines, readonly viewer: Viewer, readonly store: Store) {
    this.cards = LABELS.map((l) => {
      const canvas = h('canvas', { width: CROP, height: CROP });
      const meta = h('span', null, '—');
      const bar = h('s');
      const root = h('div', { class: 'roi-card' }, canvas, h('div', { class: 'cap' }, h('b', null, l), meta), h('i', { class: 'bar' }, bar));
      return { root, canvas, meta, bar };
    });
    host.replaceChildren(h('div', { class: 'roi-grid' }, ...this.cards.map((c) => c.root)),
      h('div', { class: 'note' }, bi('The three regions with the strongest image gradient, fixed for the fit · live renders, Gaussian-id colours while fitting', '圖像梯度最強的三個區域（擬合開始時決定）· 即時渲染，擬合中為矢量視角')));
    store.subscribe((s, changed) => {
      if (changed.has('model')) this.setModel(s.model);
      if (changed.has('monitor') && s.monitor && !s.job) void this.renderCrops(); // opened later: refresh from the final model
    });
  }

  /**
   * A new fit on a W x H image (`rgb`: W x H x 3 floats): the regions are the cells with the strongest gradient
   * energy (sum of |dL/dx| + |dL/dy| over the cell, luma), at least two cells apart.
   */
  reset(W: number, H: number, rgb: Float32Array): void {
    const cell = Math.max(40, Math.min(96, Math.round(Math.min(W, H) / 24)));
    const cols = Math.ceil(W / cell), rows = Math.ceil(H / cell);
    const energy = new Float32Array(cols * rows);
    const luma = (x: number, y: number) => { const p = 3 * (y * W + x); return 0.299 * rgb[p] + 0.587 * rgb[p + 1] + 0.114 * rgb[p + 2]; };
    for (let y = 1; y < H - 1; y += 2) {
      const cy = Math.min(rows - 1, (y / cell) | 0);
      for (let x = 1; x < W - 1; x += 2) {
        const e = Math.abs(luma(x + 1, y) - luma(x - 1, y)) + Math.abs(luma(x, y + 1) - luma(x, y - 1));
        energy[cy * cols + Math.min(cols - 1, (x / cell) | 0)] += e;
      }
    }
    const order = Array.from(energy.keys()).sort((a, b) => energy[b] - energy[a]);
    const taken: number[] = [];
    for (const k of order) {
      if (taken.length >= SLOTS) break;
      const far = taken.every((o) => Math.abs((o % cols) - (k % cols)) > 1 || Math.abs(Math.floor(o / cols) - Math.floor(k / cols)) > 1);
      if (far && energy[k] > 0) taken.push(k);
    }
    const best = Math.max(1e-9, ...taken.map((k) => energy[k]));
    const regions: ActiveRegion[] = taken.map((k, s) => {
      const x = Math.max(0, Math.min(W - cell, (k % cols) * cell));
      const y = Math.max(0, Math.min(H - cell, Math.floor(k / cols) * cell));
      return { x, y, w: Math.min(cell, W), h: Math.min(cell, H), score: energy[k] / best, label: LABELS[s] };
    });
    this.vector = null; // a new fit attaches its own recoloured copy
    this.crop?.setParams(null);
    this.store.set({ regions });
    for (const c of this.cards) { c.meta.textContent = '—'; c.bar.style.width = '0'; c.canvas.getContext('2d')?.clearRect(0, 0, CROP, CROP); }
  }

  private setModel(model: AppModel | null): void {
    if (model === this.model) return;
    const old = this.crop;
    this.crop = null;
    if (old) {
      old.retire();
      if (this.rendering) this.retired.push(old); else old.destroy();
    }
    this.model = model;
    if (model) {
      this.crop = new CropRenderer(this.ctx, this.pipes, model.params, model.N, model.W, model.H);
      if (this.vector) this.crop.setParams(this.vector.buf);
    }
  }

  /** Switches the crops to a recoloured copy of the parameters (`pre` refreshes it before each crop render) or back. */
  setVectorSource(buf: GPUBuffer | null, pre: ((enc: GPUCommandEncoder) => void) | null): void {
    this.vector = buf && pre ? { buf, pre } : null;
    this.crop?.setParams(this.vector?.buf ?? null);
    void this.renderCrops();
  }

  /** Re-renders the crop of every region (one at a time; a request during a render runs once more afterwards). */
  async renderCrops(): Promise<void> {
    if (this.rendering) { this.pendingRender = true; return; }
    this.rendering = true;
    try {
      const regions = this.store.state.regions;
      const { edges, edgePx } = this.store.state.prefs;
      for (let s = 0; s < SLOTS; s++) {
        const card = this.cards[s];
        const r = regions.find((q) => q.label === LABELS[s]);
        card.root.classList.toggle('empty', !r);
        if (!r) continue;
        card.meta.textContent = `${r.x}, ${r.y} · ${r.w} px`;
        card.bar.style.width = `${Math.round(r.score * 100)}%`;
        if (!this.crop) continue;
        const rgba = await this.crop.render(r, edges, edgePx, this.vector?.pre ?? null);
        if (this.store.state.regions !== regions) break; // a new fit started meanwhile
        card.canvas.getContext('2d')?.putImageData(new ImageData(rgba, CROP, CROP), 0, 0);
      }
    } catch {
      /* the model may have been replaced meanwhile */
    } finally {
      this.rendering = false;
      for (const c of this.retired.splice(0)) c.destroy();
      if (this.pendingRender) { this.pendingRender = false; void this.renderCrops(); }
    }
  }
}
