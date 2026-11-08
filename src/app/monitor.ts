// The monitor panel (right): the regions where the Gaussians changed most between two snapshots of the running fit,
// each shown as a small live render. Activity is measured from the parameters themselves (position and colour
// deltas per cell), so it follows the optimiser; the crops are rendered with the engine's rasteriser and edge pass.

import { OFFSET } from '../core/spec';
import { EdgePass, Rasterizer, createBuffer, readBuffer, type GpuContext, type Pipelines } from '../engine';
import { h } from './dom';
import type { ActiveRegion, AppModel, Store } from './store';
import type { Viewer } from './viewer';

const SLOTS = 4;
const CROP = 160; // crop render size in pixels
const LABELS = ['A', 'B', 'C', 'D'];

/** Renders a square window of the model at CROP x CROP pixels (with the edge pass at magnification). */
class CropRenderer {
  private readonly raster: Rasterizer;
  private readonly edge: EdgePass;
  private readonly dst: GPUBuffer;

  constructor(readonly ctx: GpuContext, readonly pipes: Pipelines, readonly params: GPUBuffer, readonly N: number, readonly W: number, readonly H: number) {
    this.raster = new Rasterizer(ctx, pipes, params, N, CROP, CROP, Math.max(65536, Math.min(N, 1 << 20)));
    this.edge = new EdgePass(ctx, pipes, CROP, CROP);
    this.dst = createBuffer(ctx.device, CROP * CROP * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'crop.dst');
  }

  async render(roi: { x: number; y: number; w: number; h: number }, edges: boolean, edgePx: number): Promise<Uint8ClampedArray<ArrayBuffer>> {
    const zoom = CROP / Math.max(roi.w, roi.h);
    const view = { srcW: this.W, srcH: this.H, zoom, offsetX: -roi.x * zoom, offsetY: -roi.y * zoom, aa: zoom < 1 ? 0.5 * Math.sqrt(1 - zoom * zoom) : 0, outW: CROP, outH: CROP };
    const device = this.ctx.device;
    const useEdge = edges && zoom >= 1.5;
    for (let attempt = 0; attempt < 3; attempt++) {
      this.raster.setView(view);
      const enc = device.createCommandEncoder({ label: 'crop' });
      this.raster.encodeBuild(enc);
      this.raster.encodeForward(enc);
      if (useEdge) this.edge.encode(enc, this.raster.out, this.dst, zoom, { edgePx });
      device.queue.submit([enc.finish()]);
      const total = await this.raster.readTotalEntries();
      if (total <= this.raster.entryCap) break;
      this.raster.growEntries(Math.ceil(total * 1.5));
    }
    const f = new Float32Array(await readBuffer(device, useEdge ? this.dst : this.raster.out, CROP * CROP * 16));
    const out = new Uint8ClampedArray<ArrayBuffer>(new ArrayBuffer(CROP * CROP * 4));
    for (let i = 0; i < CROP * CROP; i++) {
      out[4 * i] = f[4 * i] * 255;
      out[4 * i + 1] = f[4 * i + 1] * 255;
      out[4 * i + 2] = f[4 * i + 2] * 255;
      out[4 * i + 3] = 255;
    }
    return out;
  }

  destroy(): void {
    this.raster.destroy();
    this.edge.destroy();
    this.dst.destroy();
  }
}

interface Card { root: HTMLElement; canvas: HTMLCanvasElement; label: HTMLElement; meta: HTMLElement; bar: HTMLElement }

export class Monitor {
  private W = 0;
  private H = 0;
  private cell = 64;
  private cols = 0;
  private rows = 0;
  private prev: Float32Array | null = null;
  private score: Float32Array | null = null;
  private slots: (number | null)[] = Array.from({ length: SLOTS }, () => null); // cell index per slot
  private crop: CropRenderer | null = null;
  private model: AppModel | null = null;
  private cards: Card[];
  private note: HTMLElement;
  private rendering = false;
  private pendingRender = false;

  constructor(readonly host: HTMLElement, readonly ctx: GpuContext, readonly pipes: Pipelines, readonly viewer: Viewer, readonly store: Store) {
    this.cards = LABELS.map((l) => {
      const canvas = h('canvas', { width: CROP, height: CROP });
      const label = h('b', null, l);
      const meta = h('span', null, '—');
      const bar = h('s');
      const root = h('div', { class: 'roi-card' }, canvas, h('div', { class: 'meta' }, label, meta, h('i', { class: 'bar' }, bar)));
      return { root, canvas, label, meta, bar };
    });
    this.note = h('div', { class: 'note' }, 'Regions where the Gaussians moved most between two snapshots of the fit; live renders of the model.');
    host.replaceChildren(...this.cards.map((c) => c.root), this.note);
    store.subscribe((s, changed) => {
      if (changed.has('model')) this.setModel(s.model);
      if (changed.has('monitor') && s.monitor && !s.job) void this.renderCrops(); // opened later: refresh from the final model
    });
  }

  /** A new fit on a W x H image: the activity grid starts over. */
  reset(W: number, H: number): void {
    this.W = W;
    this.H = H;
    this.cell = Math.max(48, Math.min(256, Math.round(Math.min(W, H) / 10)));
    this.cols = Math.ceil(W / this.cell);
    this.rows = Math.ceil(H / this.cell);
    this.prev = null;
    this.score = new Float32Array(this.cols * this.rows);
    this.slots = this.slots.map(() => null);
    this.store.set({ regions: [] });
    for (const c of this.cards) { c.meta.textContent = '—'; c.bar.style.width = '0'; c.canvas.getContext('2d')?.clearRect(0, 0, CROP, CROP); }
  }

  private setModel(model: AppModel | null): void {
    if (model === this.model) return;
    this.crop?.destroy();
    this.crop = null;
    this.model = model;
    if (model) this.crop = new CropRenderer(this.ctx, this.pipes, model.params, model.N, model.W, model.H);
  }

  /** A snapshot of the parameters during the fit: accumulates per-cell activity, re-ranks, re-renders the crops. */
  update(params: Float32Array): void {
    if (!this.score || params.length !== (this.prev?.length ?? params.length)) this.prev = null;
    const score = this.score!;
    if (this.prev) {
      const acc = new Float32Array(score.length);
      const N = params.length / 9;
      const prev = this.prev;
      for (let i = 0; i < N; i++) {
        const b = 9 * i;
        const x = params[b + OFFSET.uv] * this.W, y = params[b + OFFSET.uv + 1] * this.H;
        const d = Math.abs(x - prev[b + OFFSET.uv] * this.W) + Math.abs(y - prev[b + OFFSET.uv + 1] * this.H)
          + 24 * (Math.abs(params[b + OFFSET.color] - prev[b + OFFSET.color]) + Math.abs(params[b + OFFSET.color + 1] - prev[b + OFFSET.color + 1]) + Math.abs(params[b + OFFSET.color + 2] - prev[b + OFFSET.color + 2]));
        const cx = Math.min(this.cols - 1, Math.max(0, (x / this.cell) | 0));
        const cy = Math.min(this.rows - 1, Math.max(0, (y / this.cell) | 0));
        acc[cy * this.cols + cx] += d;
      }
      for (let k = 0; k < score.length; k++) score[k] = 0.6 * score[k] + acc[k];
      this.rank();
    }
    this.prev = params;
    void this.renderCrops();
  }

  /** Top cells by score with hysteresis: a slot keeps its cell while it stays within the top 2 x SLOTS. */
  private rank(): void {
    const score = this.score!;
    const order = Array.from(score.keys()).sort((a, b) => score[b] - score[a]);
    const top = order.slice(0, 2 * SLOTS);
    const taken = new Set<number>();
    const far = (k: number) => [...taken].every((o) => Math.abs((o % this.cols) - (k % this.cols)) > 1 || Math.abs(Math.floor(o / this.cols) - Math.floor(k / this.cols)) > 1);
    // keep survivors first
    for (let s = 0; s < SLOTS; s++) {
      const k = this.slots[s];
      if (k !== null && top.includes(k) && far(k)) taken.add(k); else this.slots[s] = null;
    }
    for (let s = 0; s < SLOTS; s++) {
      if (this.slots[s] !== null) continue;
      const k = order.find((c) => score[c] > 0 && !taken.has(c) && far(c));
      if (k === undefined) break;
      this.slots[s] = k;
      taken.add(k);
    }
    const best = Math.max(1e-9, ...this.slots.map((k) => (k === null ? 0 : score[k])));
    const regions: ActiveRegion[] = [];
    this.slots.forEach((k, s) => {
      if (k === null) return;
      const size = this.cell;
      const x = Math.max(0, Math.min(this.W - size, (k % this.cols) * this.cell + (this.cell - size) / 2));
      const y = Math.max(0, Math.min(this.H - size, Math.floor(k / this.cols) * this.cell + (this.cell - size) / 2));
      regions.push({ x: Math.round(x), y: Math.round(y), w: Math.min(size, this.W), h: Math.min(size, this.H), score: score[k] / best, label: LABELS[s] });
    });
    this.store.set({ regions });
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
        card.meta.textContent = `x ${r.x} · y ${r.y} · ${r.w} px`;
        card.bar.style.width = `${Math.round(r.score * 100)}%`;
        if (!this.crop) continue;
        const rgba = await this.crop.render(r, edges, edgePx);
        if (this.store.state.regions !== regions && this.store.state.job) break; // a newer ranking exists
        card.canvas.getContext('2d')?.putImageData(new ImageData(rgba, CROP, CROP), 0, 0);
      }
    } catch {
      /* the model may have been replaced meanwhile */
    } finally {
      this.rendering = false;
      if (this.pendingRender) { this.pendingRender = false; void this.renderCrops(); }
    }
  }
}
