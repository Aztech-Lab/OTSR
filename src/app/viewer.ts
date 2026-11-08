// Interactive canvas viewer: pans and zooms over the model (rendered at the exact zoom, with the edge pass) and
// over the source image resampled with a classic filter; a split comparison between the two; a 2-D overlay for the
// split handle, labels and the region-of-interest tool.

import { EDGE_PX } from '../core/spec';
import { EdgePass, Presenter, Rasterizer, Resampler, createBuffer, type GpuContext, type Pipelines, type ResampleMode, type ViewParams } from '../engine';
import { encodePng } from './image';
import { RoiTool, type Roi, type RoiTransform } from './roi';

export type ViewMode = 'otsr' | 'raw' | ResampleMode;
export type Tool = 'pan' | 'roi';

export interface ViewerModel {
  N: number;
  W: number;
  H: number;
  params: GPUBuffer;
}

export interface ViewerSource {
  W: number;
  H: number;
  rgba: GPUBuffer;
}

export interface ViewerEvents {
  onViewChange?: (zoom: number) => void;
  onRoiChange?: (roi: Roi | null, final: boolean) => void;
  onToolChange?: (tool: Tool) => void;
}

const MIN_ZOOM = 1 / 32;
const MAX_ZOOM = 64;
const SPLIT_GRAB_PX = 10;

export class Viewer {
  mode: ViewMode = 'bicubic';
  edges = true;
  edgePx = EDGE_PX;
  /** Split comparison: the image left of the divider is `compareWith`, right of it the model. */
  compare = false;
  compareWith: ResampleMode = 'bicubic';
  /** Divider position as a fraction of the canvas width. */
  splitFrac = 0.5;
  /** Colour around the image (linear 0..1 RGB of the stage background); follows the page theme. */
  background: [number, number, number] = [0.043, 0.067, 0.078];
  /** Space kept clear of floating UI when fitting the image (CSS px). */
  insets = { left: 16, top: 16, right: 16, bottom: 16 };
  /** Output (device) pixels per source pixel. */
  zoom = 1;
  offsetX = 0;
  offsetY = 0;
  readonly roi = new RoiTool();
  events: ViewerEvents = {};
  private tool: Tool = 'pan';
  private model: ViewerModel | null = null;
  private source: ViewerSource | null = null;
  private readonly device: GPUDevice;
  private readonly canvasCtx: GPUCanvasContext;
  private readonly presenter: Presenter;
  private readonly resampler: Resampler;
  private readonly overlay: HTMLCanvasElement;
  private readonly g2d: CanvasRenderingContext2D;
  private raster: Rasterizer | null = null;
  private edge: EdgePass | null = null;
  private bufA: GPUBuffer | null = null;
  private bufB: GPUBuffer | null = null;
  private width = 0;
  private height = 0;
  private pending = false;
  private drag: 'pan' | 'split' | 'roi' | null = null;
  private lastX = 0;
  private lastY = 0;
  private holdMode: ViewMode | null = null;
  private readonly dpr = window.devicePixelRatio || 1;

  constructor(readonly canvas: HTMLCanvasElement, overlay: HTMLCanvasElement, readonly ctx: GpuContext, readonly pipes: Pipelines) {
    this.device = ctx.device;
    this.overlay = overlay;
    const cc = canvas.getContext('webgpu');
    if (!cc) throw new Error('The canvas has no WebGPU context.');
    this.canvasCtx = cc;
    const format = navigator.gpu.getPreferredCanvasFormat();
    cc.configure({ device: this.device, format, alphaMode: 'opaque' });
    this.presenter = new Presenter(ctx, format);
    this.resampler = new Resampler(ctx, pipes);
    const g = overlay.getContext('2d');
    if (!g) throw new Error('2D overlay unavailable');
    this.g2d = g;
    this.roi.onChange = (roi, final) => {
      this.drawOverlay();
      this.events.onRoiChange?.(roi, final);
    };
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
    this.bindPointer();
  }

  // ---- state ---------------------------------------------------------------------------------------------------

  get sourceSize(): { W: number; H: number } | null {
    return this.model ?? this.source;
  }

  get hasModel(): boolean {
    return this.model !== null;
  }

  get hasSource(): boolean {
    return this.source !== null;
  }

  get currentTool(): Tool {
    return this.tool;
  }

  setTool(tool: Tool): void {
    if (tool === this.tool) return;
    this.tool = tool;
    this.canvas.style.cursor = tool === 'roi' ? 'crosshair' : 'grab';
    this.events.onToolChange?.(tool);
    this.drawOverlay();
  }

  setModel(model: ViewerModel | null): void {
    this.model = model;
    this.rebuild();
    this.changed();
  }

  setSource(source: ViewerSource | null): void {
    this.source = source;
    this.changed();
  }

  /** The transform between source pixels and canvas device pixels. */
  transform(): RoiTransform | null {
    const s = this.sourceSize;
    return s ? { zoom: this.zoom, offsetX: this.offsetX, offsetY: this.offsetY, W: s.W, H: s.H } : null;
  }

  /** Temporarily shows another mode while a key is held (B / N / R). */
  hold(mode: ViewMode | null): void {
    if (mode) {
      if (this.holdMode !== null) return;
      this.holdMode = this.mode;
      this.mode = mode;
    } else {
      if (this.holdMode === null) return;
      this.mode = this.holdMode;
      this.holdMode = null;
    }
    this.requestRender();
  }

  // ---- navigation ----------------------------------------------------------------------------------------------

  /** The free area between the insets, in device pixels: [x0, y0, width, height]. */
  private freeArea(): [number, number, number, number] {
    const d = this.dpr;
    const i = this.insets;
    const w = Math.max(64 * d, this.width - (i.left + i.right) * d);
    const h = Math.max(64 * d, this.height - (i.top + i.bottom) * d);
    return [i.left * d, i.top * d, w, h];
  }

  fitToView(): void {
    const s = this.sourceSize;
    if (!s || !this.width) return;
    const [x0, y0, w, h] = this.freeArea();
    this.zoom = Math.max(MIN_ZOOM, Math.min(w / s.W, h / s.H));
    this.offsetX = x0 + (w - s.W * this.zoom) / 2;
    this.offsetY = y0 + (h - s.H * this.zoom) / 2;
    this.changed();
  }

  /** Sets the zoom, keeping the source point under canvas position (cx, cy) fixed (default: the centre). */
  zoomAbout(zoom: number, cx = this.width / 2, cy = this.height / 2): void {
    const z = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom));
    const sx = (cx - this.offsetX) / this.zoom;
    const sy = (cy - this.offsetY) / this.zoom;
    this.zoom = z;
    this.offsetX = cx - sx * z;
    this.offsetY = cy - sy * z;
    this.changed();
  }

  zoomBy(factor: number): void {
    this.zoomAbout(this.zoom * factor);
  }

  /** Zooms so that the region fills the view. */
  zoomToRoi(roi: Roi): void {
    if (!this.width) return;
    const [x0, y0, w, h] = this.freeArea();
    const z = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, Math.min(w / roi.w, h / roi.h)));
    this.zoom = z;
    this.offsetX = x0 + (w - roi.w * z) / 2 - roi.x * z;
    this.offsetY = y0 + (h - roi.h * z) / 2 - roi.y * z;
    this.changed();
  }

  /** Canvas device pixel -> source pixel. */
  toSource(cx: number, cy: number): [number, number] {
    return [(cx - this.offsetX) / this.zoom, (cy - this.offsetY) / this.zoom];
  }

  // ---- rendering -----------------------------------------------------------------------------------------------

  requestRender(): void {
    if (this.pending) return;
    this.pending = true;
    requestAnimationFrame(() => {
      this.pending = false;
      this.render();
    });
  }

  private changed(): void {
    this.events.onViewChange?.(this.zoom);
    this.requestRender();
  }

  private resize(): void {
    const w = Math.max(1, Math.round(this.canvas.clientWidth * this.dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * this.dpr));
    if (w === this.width && h === this.height) return;
    const first = this.width === 0;
    const cx = this.width ? (this.width / 2 - this.offsetX) / this.zoom : 0;
    const cy = this.height ? (this.height / 2 - this.offsetY) / this.zoom : 0;
    this.canvas.width = this.overlay.width = w;
    this.canvas.height = this.overlay.height = h;
    this.width = w;
    this.height = h;
    this.rebuild();
    if (first) this.fitToView();
    else {
      // keep the source point at the centre of the view where it was
      this.offsetX = w / 2 - cx * this.zoom;
      this.offsetY = h / 2 - cy * this.zoom;
    }
    this.requestRender();
  }

  private rebuild(): void {
    this.raster?.destroy();
    this.edge?.destroy();
    this.bufA?.destroy();
    this.bufB?.destroy();
    this.raster = null;
    this.edge = null;
    const bytes = this.width * this.height * 16;
    this.bufA = createBuffer(this.device, bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'viewer.A');
    this.bufB = createBuffer(this.device, bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'viewer.B');
    if (this.model) {
      this.raster = new Rasterizer(this.ctx, this.pipes, this.model.params, this.model.N, this.width, this.height, Math.max(4096, this.model.N));
      this.edge = new EdgePass(this.ctx, this.pipes, this.width, this.height);
    }
  }

  private imageRect(s: { W: number; H: number }): [number, number, number, number] {
    return [this.offsetX, this.offsetY, this.offsetX + s.W * this.zoom, this.offsetY + s.H * this.zoom];
  }

  /** Whether the split comparison is actually visible (needs a model and the source). */
  get splitVisible(): boolean {
    return this.compare && this.model !== null && this.source !== null && (this.mode === 'otsr' || this.mode === 'raw') && this.holdMode === null;
  }

  private render(): void {
    this.renderTo(this.canvasCtx.getCurrentTexture().createView());
    this.drawOverlay();
  }

  private renderTo(target: GPUTextureView): void {
    const s = this.sourceSize;
    if (!this.bufA || !this.bufB) return;
    if (!s) {
      // nothing loaded yet: clear to the stage colour
      const enc = this.device.createCommandEncoder({ label: 'viewer.clear' });
      this.presenter.encode(enc, this.bufA, 0, 0, target, { background: this.background });
      this.device.queue.submit([enc.finish()]);
      return;
    }
    const view: ViewParams = { srcW: s.W, srcH: s.H, zoom: this.zoom, offsetX: this.offsetX, offsetY: this.offsetY, aa: 0, outW: this.width, outH: this.height };
    const enc = this.device.createCommandEncoder({ label: 'viewer' });
    const modelMode = this.mode === 'otsr' || this.mode === 'raw';
    let main: GPUBuffer;
    if (modelMode && this.model && this.raster) {
      // below native resolution, prefilter with the pixel footprint (std 1/2 output px of the missing detail)
      view.aa = this.zoom < 1 ? 0.5 * Math.sqrt(1 - this.zoom * this.zoom) : 0;
      this.raster.setView(view);
      this.raster.encodeBuild(enc);
      this.raster.encodeForward(enc);
      main = this.raster.out;
      if (this.mode === 'otsr' && this.edges && this.zoom >= 1.5 && this.edge) {
        this.edge.encode(enc, this.raster.out, this.bufA, this.zoom, { edgePx: this.edgePx });
        main = this.bufA;
      }
      const raster = this.raster;
      raster.readTotalEntries().then((total) => {
        if (raster === this.raster && total > raster.entryCap) {
          raster.growEntries(Math.ceil(total * 1.5));
          this.requestRender();
        }
      }).catch(() => undefined);
    } else if (this.source) {
      const mode: ResampleMode = modelMode ? 'bicubic' : (this.mode as ResampleMode);
      view.aa = 0;
      this.resampler.encode(enc, this.source.rgba, this.bufA, view, mode);
      main = this.bufA;
    } else {
      this.presenter.encode(enc, this.bufA, 0, 0, target, { background: this.background });
      this.device.queue.submit([enc.finish()]);
      return;
    }
    let compare: { img: GPUBuffer; split: number } | undefined;
    if (this.splitVisible && this.source) {
      this.resampler.encode(enc, this.source.rgba, this.bufB, { ...view, aa: 0 }, this.compareWith);
      compare = { img: this.bufB, split: this.splitFrac * this.width };
    }
    this.presenter.encode(enc, main, this.width, this.height, target, { rect: this.imageRect(s), compare, background: this.background });
    this.device.queue.submit([enc.finish()]);
  }

  private drawOverlay(): void {
    const g = this.g2d;
    const dpr = this.dpr;
    g.clearRect(0, 0, this.width, this.height);
    const t = this.transform();
    if (!t) return;
    if (this.splitVisible) {
      const x = Math.round(this.splitFrac * this.width);
      g.save();
      g.shadowColor = 'rgba(0,0,0,0.25)';
      g.shadowBlur = 6 * dpr;
      g.lineWidth = 2 * dpr;
      g.strokeStyle = '#fff';
      g.beginPath(); g.moveTo(x, 0); g.lineTo(x, this.height); g.stroke();
      // handle: a square plate with two chevrons
      const r = 16 * dpr;
      const cy = this.height / 2;
      g.fillStyle = 'rgba(237,235,228,0.96)';
      g.fillRect(x - r, cy - r, 2 * r, 2 * r);
      g.shadowBlur = 0;
      g.strokeStyle = '#252820';
      g.lineWidth = 1.6 * dpr;
      g.lineCap = 'round';
      g.lineJoin = 'round';
      g.beginPath();
      g.moveTo(x - 3 * dpr, cy - 5 * dpr); g.lineTo(x - 8 * dpr, cy); g.lineTo(x - 3 * dpr, cy + 5 * dpr);
      g.moveTo(x + 3 * dpr, cy - 5 * dpr); g.lineTo(x + 8 * dpr, cy); g.lineTo(x + 3 * dpr, cy + 5 * dpr);
      g.stroke();
      // labels below the top bar
      const labelL = this.compareWith.charAt(0).toUpperCase() + this.compareWith.slice(1);
      const labelR = this.mode === 'raw' ? 'Gaussians' : 'OTSR';
      const ly = (this.insets.top + 4) * dpr;
      this.label(g, labelL, x - 10 * dpr, ly, 'right', dpr);
      this.label(g, labelR, x + 10 * dpr, ly, 'left', dpr);
      g.restore();
    }
    this.roi.draw(g, t, this.tool === 'roi', dpr);
  }

  private label(g: CanvasRenderingContext2D, text: string, x: number, y: number, align: 'left' | 'right', dpr: number): void {
    g.font = `600 ${10.5 * dpr}px system-ui, sans-serif`;
    if ('letterSpacing' in g) (g as CanvasRenderingContext2D & { letterSpacing: string }).letterSpacing = `${1.2 * dpr}px`;
    const label = text.toUpperCase();
    const w = g.measureText(label).width + 22 * dpr;
    const x0 = align === 'left' ? x : x - w;
    g.shadowColor = 'rgba(99,81,58,0.18)';
    g.shadowBlur = 10 * dpr;
    g.fillStyle = 'rgba(237,235,228,0.94)';
    g.fillRect(x0, y, w, 24 * dpr);
    g.shadowBlur = 0;
    g.fillStyle = '#080a08';
    g.fillText(label, x0 + 11 * dpr, y + 16 * dpr);
  }

  // ---- input ---------------------------------------------------------------------------------------------------

  private canvasPos(e: PointerEvent | WheelEvent): [number, number] {
    const r = this.canvas.getBoundingClientRect();
    return [(e.clientX - r.left) * this.dpr, (e.clientY - r.top) * this.dpr];
  }

  private bindPointer(): void {
    const c = this.canvas;
    c.style.cursor = 'grab';
    c.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      const [x, y] = this.canvasPos(e);
      const t = this.transform();
      c.setPointerCapture(e.pointerId);
      this.lastX = x;
      this.lastY = y;
      if (this.splitVisible && Math.abs(x - this.splitFrac * this.width) <= SPLIT_GRAB_PX * this.dpr) {
        this.drag = 'split';
      } else if (this.tool === 'roi' && t) {
        const [sx, sy] = this.toSource(x, y);
        this.roi.begin(sx, sy, t);
        this.drag = 'roi';
      } else {
        this.drag = 'pan';
        c.style.cursor = 'grabbing';
      }
    });
    c.addEventListener('pointermove', (e) => {
      const [x, y] = this.canvasPos(e);
      const t = this.transform();
      if (this.drag === 'pan') {
        this.offsetX += x - this.lastX;
        this.offsetY += y - this.lastY;
        this.lastX = x;
        this.lastY = y;
        this.changed();
      } else if (this.drag === 'split') {
        this.splitFrac = Math.max(0.02, Math.min(0.98, x / this.width));
        this.requestRender();
      } else if (this.drag === 'roi' && t) {
        const [sx, sy] = this.toSource(x, y);
        this.roi.move(sx, sy, t);
      } else {
        // hover cursor
        if (this.splitVisible && Math.abs(x - this.splitFrac * this.width) <= SPLIT_GRAB_PX * this.dpr) c.style.cursor = 'col-resize';
        else if (this.tool === 'roi' && t) {
          const [sx, sy] = this.toSource(x, y);
          c.style.cursor = this.roi.cursorFor(this.roi.hitTest(sx, sy, t));
        } else c.style.cursor = 'grab';
      }
    });
    const stop = () => {
      if (this.drag === 'roi') this.roi.end();
      this.drag = null;
      c.style.cursor = this.tool === 'roi' ? 'crosshair' : 'grab';
    };
    c.addEventListener('pointerup', stop);
    c.addEventListener('pointercancel', stop);
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      const [x, y] = this.canvasPos(e);
      const factor = Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015));
      this.zoomAbout(this.zoom * factor, x, y);
    }, { passive: false });
    c.addEventListener('dblclick', () => this.fitToView());
  }

  // ---- capture -------------------------------------------------------------------------------------------------

  /** The current view as 8-bit RGBA (rendered again into an offscreen texture; the canvas itself is not readable). */
  async snapshot(): Promise<{ width: number; height: number; rgba: Uint8ClampedArray<ArrayBuffer> }> {
    const { width, height } = this;
    const format = navigator.gpu.getPreferredCanvasFormat();
    const tex = this.device.createTexture({ size: [width, height], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const bpr = Math.ceil((width * 4) / 256) * 256;
    const staging = this.device.createBuffer({ size: bpr * height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.renderTo(tex.createView());
    const enc = this.device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: tex }, { buffer: staging, bytesPerRow: bpr }, [width, height]);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(staging.getMappedRange());
    const rgba = new Uint8ClampedArray(width * height * 4);
    const bgr = format === 'bgra8unorm';
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const s = y * bpr + 4 * x;
        const d = 4 * (y * width + x);
        rgba[d] = src[bgr ? s + 2 : s];
        rgba[d + 1] = src[s + 1];
        rgba[d + 2] = src[bgr ? s : s + 2];
        rgba[d + 3] = 255;
      }
    }
    staging.unmap();
    staging.destroy();
    tex.destroy();
    return { width, height, rgba };
  }

  /** Snapshot plus the overlay (split line, labels, region) as a PNG blob. */
  async snapshotPng(withOverlay = true): Promise<Blob> {
    const snap = await this.snapshot();
    if (withOverlay) {
      const c = new OffscreenCanvas(snap.width, snap.height);
      const g = c.getContext('2d')!;
      g.putImageData(new ImageData(snap.rgba, snap.width, snap.height), 0, 0);
      g.drawImage(this.overlay, 0, 0);
      return c.convertToBlob({ type: 'image/png' });
    }
    return encodePng(snap.rgba, snap.width, snap.height);
  }

  destroy(): void {
    this.raster?.destroy();
    this.edge?.destroy();
    this.bufA?.destroy();
    this.bufB?.destroy();
    this.presenter.destroy();
    this.resampler.destroy();
  }
}
