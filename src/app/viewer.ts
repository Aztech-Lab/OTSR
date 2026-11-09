// Interactive canvas viewer: pans and zooms over the model (rendered at the exact zoom, with the edge pass) and
// over the source image resampled with a classic filter; a split comparison between the two; a 2-D overlay for the
// split handle, labels and the region-of-interest tool.

import { EDGE_PX, OFFSET } from '../core/spec';
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
  /** CPU copy of the parameters when available (the Gaussian outline layer reads it). */
  cpu?: Float32Array | null;
  /** The transport map of the initialisation, for the OT mesh layer. */
  ot?: { T: Float32Array; grid: { ny: number; nx: number } } | null;
}

export interface ViewerLayers {
  /** 1-sigma outlines of the Gaussians in their own colour. */
  gaussians: boolean;
  /** The |error| map of the fit over the image. */
  error: boolean;
  /** Gaussian-id colours (handled by the recolour hook, not the overlay). */
  vector: boolean;
  /** The optimal-transport mesh of the initialisation. */
  ot: boolean;
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
  /** Pointer position in source pixels, or null when it leaves the image. */
  onHover?: (sx: number | null, sy: number | null) => void;
}

const MIN_ZOOM = 1 / 32;
const MAX_ZOOM = 64;
const SPLIT_GRAB_PX = 10;
/** The tile-entry check after a render (one small read-back) runs at most this often while the view moves. */
const ENTRY_CHECK_MS = 120;
/** Canvas resizes are coalesced over this long (phone browsers resize the viewport while their bars slide). */
const RESIZE_SETTLE_MS = 60;
/** Gaussian outlines are drawn from this zoom on (below it they are sub-pixel) and up to this many per frame. */
const OUTLINE_MIN_ZOOM = 2;
const OUTLINE_MAX = 60000;

function cssVar(name: string, fallback: string): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

/** Major ruler spacing in source pixels so that ticks are at least `minPx` device pixels apart. */
function rulerStep(zoom: number, minPx: number): number {
  const steps = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000];
  for (const s of steps) if (s * zoom >= minPx) return s;
  return 10000;
}

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
  /** Pixel rulers along the top and left edge of the image. */
  rulers = true;
  /** While set, the canvas shows only the background (the fit animation draws on the overlay instead). */
  curtain = false;
  /** Replaces the overlay layers while set (the fit animation); the region markers are drawn on top of it. */
  painter: ((g: CanvasRenderingContext2D, t: RoiTransform, dpr: number) => void) | null = null;
  /** True while the painter has nothing to draw: a blank overlay is then left untouched. */
  painterBlank: (() => boolean) | null = null;
  /** The overlay canvas holds nothing (set when it was cleared for a blank painter). */
  private overlayBlank = false;
  /** Labelled boxes in source pixels (the monitor's active regions). */
  markers: { x: number; y: number; w: number; h: number; label: string; score: number }[] = [];
  /** Extra compute work encoded before every model render (the fit animation refreshes its recoloured copy). */
  beforeRender: ((enc: GPUCommandEncoder) => void) | null = null;
  private paramsOverride: GPUBuffer | null = null;
  layers: ViewerLayers = { gaussians: false, error: false, vector: false, ot: false };
  /** Outline layer statistics of the last overlay draw: Gaussians inside the view and how many were drawn. */
  outlineInfo = { visible: 0, drawn: 0, tooFar: false };
  /** Output (device) pixels per source pixel. */
  zoom = 1;
  offsetX = 0;
  offsetY = 0;
  /** True while the view is the "fit to view" placement (cleared by any zoom or pan), so layout changes can keep it. */
  fitted = false;
  readonly roi = new RoiTool();
  events: ViewerEvents = {};
  private tool: Tool = 'pan';
  private resizeTimer = 0;
  private entryCheckAt = 0;
  private entryCheckZoom = 0;
  private entryCheckBusy = false;
  private entryCheckAgain = false;
  private entryCheckTimer = 0;
  /** Active pointers on the stage (touch: two of them pinch-zoom). */
  private readonly pointers = new Map<number, [number, number]>();
  private pinch: { dist: number; mx: number; my: number } | null = null;
  /** The region before a region drag began (restored when the drag turns into a pinch). */
  private roiBefore: Roi | null = null;
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
  private drag: 'pan' | 'split' | 'roi' | 'pinch' | null = null;
  private lastX = 0;
  private lastY = 0;
  private holdMode: ViewMode | null = null;
  private errorMap: ImageBitmap | null = null;
  private sweep: number | null = null;
  private hoverOn = false;
  /** Canvas pixels per CSS pixel: the screen's ratio, at most `maxDpr` (2 on phones, where a 3x screen would
   * triple the pixels of every render and of the canvas-sized buffers for little visible gain). */
  readonly dpr: number;

  /** Receives the pointer events (the whole stage, so panning and zooming work anywhere, not only over the image). */
  readonly input: HTMLElement;

  constructor(readonly canvas: HTMLCanvasElement, overlay: HTMLCanvasElement, readonly ctx: GpuContext, readonly pipes: Pipelines, input?: HTMLElement, opts: { maxDpr?: number } = {}) {
    this.dpr = Math.min(window.devicePixelRatio || 1, opts.maxDpr ?? Infinity);
    this.input = input ?? canvas;
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
    new ResizeObserver(() => {
      clearTimeout(this.resizeTimer);
      this.resizeTimer = window.setTimeout(() => this.resize(), this.width ? RESIZE_SETTLE_MS : 0);
    }).observe(canvas);
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
    this.input.style.cursor = tool === 'roi' ? 'crosshair' : 'grab';
    this.events.onToolChange?.(tool);
    this.drawOverlay();
  }

  setModel(model: ViewerModel | null): void {
    this.model = model;
    this.paramsOverride = null;
    this.rebuild();
    this.changed();
  }

  /** Draws the model from another parameter buffer of the same count (null: the model's own), e.g. a recoloured copy. */
  overrideParams(buf: GPUBuffer | null): void {
    this.paramsOverride = buf;
    if (this.raster && this.model) this.raster.setParams(buf ?? this.model.params);
    this.requestRender();
  }

  setSource(source: ViewerSource | null): void {
    this.source = source;
    this.changed();
  }

  /** The |error| image of the fit (source size), drawn by the error layer; null clears it. */
  setErrorMap(map: ImageBitmap | null): void {
    this.errorMap?.close();
    this.errorMap = map;
    this.drawOverlay();
  }

  get hasErrorMap(): boolean {
    return this.errorMap !== null;
  }

  /** Redraws the 2-D overlay only (layers, rulers, region). */
  refreshOverlay(): void {
    this.drawOverlay();
  }

  /** Runs the divider once across the visible image and back to its middle (after a fit; motion only). */
  sweepSplit(ms = 1500): void {
    const s = this.sourceSize;
    if (!this.splitVisible || !s) return;
    if (this.sweep !== null) cancelAnimationFrame(this.sweep);
    const d = this.dpr;
    const [ix0, , ix1] = this.imageRect(s);
    const a = Math.max(this.insets.left * d, ix0, 0) / this.width;
    const b = Math.min(this.width - this.insets.right * d, ix1, this.width) / this.width;
    if (b - a < 0.05) return;
    const from = this.splitFrac;
    const t0 = performance.now();
    const ease = (x: number) => 0.5 - 0.5 * Math.cos(Math.PI * x);
    const frame = () => {
      const u = Math.min(1, (performance.now() - t0) / ms);
      // from -> left edge -> right edge -> middle of the visible image
      const k = [from, a + 0.02, b - 0.02, (a + b) / 2];
      const seg = Math.min(2, Math.floor(u * 3));
      const f = ease(u * 3 - seg);
      this.splitFrac = k[seg] + (k[seg + 1] - k[seg]) * f;
      this.requestRender();
      this.sweep = u < 1 ? requestAnimationFrame(frame) : null;
    };
    this.sweep = requestAnimationFrame(frame);
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
    this.fitted = true;
    this.changed();
  }

  /** New insets (a panel opened, the sheet grew): a fitted image is fitted again, anything else stays where it is. */
  setInsets(insets: { left: number; top: number; right: number; bottom: number }): void {
    const i = this.insets;
    if (i.left === insets.left && i.top === insets.top && i.right === insets.right && i.bottom === insets.bottom) return;
    this.insets = insets;
    if (this.fitted) this.fitToView();
  }

  /** Sets the zoom, keeping the source point under canvas position (cx, cy) fixed (default: the centre). */
  zoomAbout(zoom: number, cx = this.width / 2, cy = this.height / 2): void {
    const z = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom));
    const sx = (cx - this.offsetX) / this.zoom;
    const sy = (cy - this.offsetY) / this.zoom;
    this.zoom = z;
    this.offsetX = cx - sx * z;
    this.offsetY = cy - sy * z;
    this.fitted = false;
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
    this.fitted = false;
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
    if (first || this.fitted) this.fitToView();
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
      if (this.paramsOverride) this.raster.setParams(this.paramsOverride);
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
    if (!s || this.curtain) {
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
    let built: Rasterizer | null = null;
    if (modelMode && this.model && this.raster) {
      // below native resolution, prefilter with the pixel footprint (std 1/2 output px of the missing detail)
      view.aa = this.zoom < 1 ? 0.5 * Math.sqrt(1 - this.zoom * this.zoom) : 0;
      this.beforeRender?.(enc);
      this.raster.setView(view);
      this.raster.encodeBuild(enc);
      this.raster.encodeForward(enc);
      main = this.raster.out;
      if (this.mode === 'otsr' && this.edges && this.zoom >= 1.5 && this.edge) {
        this.edge.encode(enc, this.raster.out, this.bufA, this.zoom, { edgePx: this.edgePx });
        main = this.bufA;
      }
      built = this.raster;
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
    if (built) this.checkEntries(built); // after the submit: the read-back is queued behind this frame's build
  }

  /**
   * Did the tile lists of the build just submitted fit? One 4-byte read-back, queued behind it; at most one in flight,
   * and while the view only moves (same zoom) no more than every ENTRY_CHECK_MS — a zoom step is checked right away.
   * A check that cannot run now is deferred, never dropped, so the last frame of a burst is checked too; a truncated
   * frame can therefore show for a few dozen milliseconds during a pan, then the grown lists re-render.
   */
  private checkEntries(raster: Rasterizer): void {
    if (this.entryCheckBusy) { this.entryCheckAgain = true; return; }
    const now = performance.now();
    const zoomJump = Math.abs(Math.log2(this.zoom / (this.entryCheckZoom || this.zoom))) > 0.25;
    const wait = zoomJump ? 0 : ENTRY_CHECK_MS - (now - this.entryCheckAt);
    if (wait > 0) {
      if (!this.entryCheckTimer) this.entryCheckTimer = window.setTimeout(() => { this.entryCheckTimer = 0; if (this.raster) this.checkEntries(this.raster); }, wait);
      return;
    }
    this.entryCheckBusy = true;
    this.entryCheckAgain = false;
    this.entryCheckAt = now;
    this.entryCheckZoom = this.zoom;
    raster.readTotalEntries().then((total) => {
      if (raster === this.raster && total > raster.entryCap) {
        raster.growEntries(Math.ceil(total * 1.5));
        this.requestRender();
      }
    }).catch(() => undefined).finally(() => {
      this.entryCheckBusy = false;
      if (this.entryCheckAgain && this.raster) { this.entryCheckAgain = false; this.checkEntries(this.raster); }
    });
  }

  private drawOverlay(): void {
    const g = this.g2d;
    const dpr = this.dpr;
    // every change of the 2-D overlay costs the browser a composited canvas frame; while the fit animation has nothing
    // on it, a blank overlay stays untouched (redrawing it on every render slowed the fit underneath about 2x)
    const blank = !!this.painter && !this.markers.length && (this.painterBlank?.() ?? false);
    if (blank && this.overlayBlank) return;
    this.overlayBlank = blank;
    g.clearRect(0, 0, this.width, this.height);
    const t = this.transform();
    if (!t || blank) return;
    if (this.painter) {
      this.painter(g, t, dpr);
      this.drawMarkers(g, dpr);
      return;
    }
    const [x0, y0, x1, y1] = this.imageRect(t);
    if (this.layers.error && this.errorMap) {
      g.save();
      g.imageSmoothingEnabled = this.zoom < 1;
      g.drawImage(this.errorMap, x0, y0, x1 - x0, y1 - y0);
      g.restore();
    }
    if (this.rulers) this.drawRulers(g, t, dpr);
    if (this.layers.ot) this.drawOtMesh(g, t, dpr);
    if (this.layers.gaussians) this.drawOutlines(g, t, dpr);
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
      // labels just below the command block
      const labelL = this.compareWith.charAt(0).toUpperCase() + this.compareWith.slice(1);
      const labelR = this.mode === 'raw' ? 'Gaussians' : 'OTSR';
      const ly = Math.max((this.insets.top + 12) * dpr, Math.min(y0 - 34 * dpr, this.height - 40 * dpr));
      this.label(g, labelL, x - 10 * dpr, ly, 'right', dpr);
      this.label(g, labelR, x + 10 * dpr, ly, 'left', dpr);
      g.restore();
    }
    this.roi.draw(g, t, this.tool === 'roi', dpr);
    this.drawMarkers(g, dpr);
  }

  /** The monitor's active regions: thin boxes with a letter, brighter for the more active ones. */
  private drawMarkers(g: CanvasRenderingContext2D, dpr: number): void {
    if (!this.markers.length) return;
    const z = this.zoom;
    g.save();
    g.font = `700 ${10 * dpr}px ui-monospace, Consolas, monospace`;
    g.textBaseline = 'top';
    const col = cssVar('--copper-2', '#d9a25f');
    for (const m of this.markers) {
      const x = this.offsetX + m.x * z, y = this.offsetY + m.y * z, w = m.w * z, hh = m.h * z;
      if (x + w < 0 || y + hh < 0 || x > this.width || y > this.height) continue;
      g.globalAlpha = 0.45 + 0.55 * m.score;
      g.strokeStyle = col;
      g.lineWidth = 1.5 * dpr;
      g.strokeRect(Math.round(x) + 0.5, Math.round(y) + 0.5, Math.round(w), Math.round(hh));
      g.fillStyle = col;
      g.fillRect(x, y - 15 * dpr, 16 * dpr, 15 * dpr);
      g.fillStyle = '#111312';
      g.fillText(m.label, x + 4 * dpr, y - 13 * dpr);
    }
    g.restore();
  }

  /** Tick rulers in source pixels along the top and the left edge of the image (outside it). */
  private drawRulers(g: CanvasRenderingContext2D, t: RoiTransform, dpr: number): void {
    const [x0, y0, x1, y1] = this.imageRect(t);
    const z = this.zoom;
    const major = rulerStep(z, 72 * dpr);
    const minor = major / (major % 5 === 0 && major >= 5 ? 5 : 2);
    const len = 7 * dpr;
    const gap = 8 * dpr; // distance from the image edge (the frame brackets sit there)
    const top = y0 - gap;
    const left = x0 - gap;
    const drawTop = top > this.insets.top * dpr * 0.8 && top < this.height - this.insets.bottom * dpr;
    const drawLeft = left > this.insets.left * dpr && left < this.width - this.insets.right * dpr;
    if (!drawTop && !drawLeft) return;
    g.save();
    g.strokeStyle = cssVar('--faint', '#a3a59d');
    g.fillStyle = cssVar('--muted', '#6f726a');
    g.lineWidth = 1 * dpr;
    g.font = `500 ${8.5 * dpr}px ui-monospace, Consolas, monospace`;
    g.textBaseline = 'alphabetic';
    g.beginPath();
    const sx0 = Math.max(0, Math.floor((0 - x0) / z / minor) * minor);
    const sx1 = Math.min(t.W, (this.width - x0) / z);
    const sy0 = Math.max(0, Math.floor((0 - y0) / z / minor) * minor);
    const sy1 = Math.min(t.H, (this.height - y0) / z);
    const isMajor = (s: number) => Math.abs(s / major - Math.round(s / major)) < 1e-6;
    if (drawTop) {
      g.moveTo(Math.max(0, x0), top + 0.5); g.lineTo(Math.min(this.width, x1), top + 0.5);
      for (let s = sx0; s <= sx1 + 1e-6; s += minor) {
        const x = Math.round(x0 + s * z) + 0.5;
        g.moveTo(x, top); g.lineTo(x, top - (isMajor(s) ? len : len * 0.5));
      }
    }
    if (drawLeft) {
      g.moveTo(left + 0.5, Math.max(0, y0)); g.lineTo(left + 0.5, Math.min(this.height, y1));
      for (let s = sy0; s <= sy1 + 1e-6; s += minor) {
        const y = Math.round(y0 + s * z) + 0.5;
        g.moveTo(left, y); g.lineTo(left - (isMajor(s) ? len : len * 0.5), y);
      }
    }
    g.stroke();
    if (drawTop) {
      g.textAlign = 'left';
      for (let s = Math.ceil(sx0 / major) * major; s <= sx1; s += major) g.fillText(String(s), x0 + s * z + 3 * dpr, top - len - 3 * dpr);
    }
    if (drawLeft) {
      for (let s = Math.ceil(sy0 / major) * major; s <= sy1; s += major) {
        g.save();
        g.translate(left - len - 4 * dpr, y0 + s * z - 3 * dpr);
        g.rotate(-Math.PI / 2);
        g.textAlign = 'left';
        g.fillText(String(s), 0, 0);
        g.restore();
      }
    }
    g.restore();
  }

  /** The transport lattice of the initialisation: the cell centres of the OT grid after the map, as a mesh. */
  private drawOtMesh(g: CanvasRenderingContext2D, t: RoiTransform, dpr: number): void {
    const ot = this.model?.ot;
    if (!ot) return;
    const { ny, nx } = ot.grid;
    const T = ot.T;
    const step = Math.max(1, Math.ceil(Math.max(nx, ny) / 128));
    const z = this.zoom * Math.max(t.W, t.H);
    const ox = this.offsetX, oy = this.offsetY;
    const path = new Path2D();
    for (let i = 0; i < ny; i += step) {
      for (let j = 0; j < nx; j += step) {
        const k = i * nx + j;
        const x = ox + T[2 * k] * z, y = oy + T[2 * k + 1] * z;
        if (j + step < nx) { const k2 = k + step; path.moveTo(x, y); path.lineTo(ox + T[2 * k2] * z, oy + T[2 * k2 + 1] * z); }
        if (i + step < ny) { const k2 = k + step * nx; path.moveTo(x, y); path.lineTo(ox + T[2 * k2] * z, oy + T[2 * k2 + 1] * z); }
      }
    }
    g.save();
    g.strokeStyle = cssVar('--copper', '#b8793c');
    g.lineWidth = 1 * dpr;
    g.globalAlpha = 0.45;
    g.stroke(path);
    g.fillStyle = cssVar('--copper-2', '#d9a25f');
    g.globalAlpha = 0.8;
    for (let i = 0; i < ny; i += step) for (let j = 0; j < nx; j += step) {
      const k = i * nx + j;
      g.fillRect(ox + T[2 * k] * z - 1 * dpr, oy + T[2 * k + 1] * z - 1 * dpr, 2 * dpr, 2 * dpr);
    }
    g.restore();
  }

  /** 1-sigma outlines of the Gaussians inside the view, each in its own colour. */
  private drawOutlines(g: CanvasRenderingContext2D, t: RoiTransform, dpr: number): void {
    const cpu = this.model?.cpu;
    const info = this.outlineInfo;
    info.visible = 0;
    info.drawn = 0;
    info.tooFar = this.zoom < OUTLINE_MIN_ZOOM;
    if (!cpu || info.tooFar) return;
    const z = this.zoom;
    const L = Math.max(t.W, t.H);
    const ox = this.offsetX;
    const oy = this.offsetY;
    // source-pixel window of the canvas, padded by the largest plausible outline
    const pad = 64;
    const wx0 = (0 - ox) / z - pad, wx1 = (this.width - ox) / z + pad;
    const wy0 = (0 - oy) / z - pad, wy1 = (this.height - oy) / z + pad;
    const N = cpu.length / 9;
    g.save();
    g.lineWidth = Math.max(1, 1 * dpr);
    g.globalAlpha = 0.9;
    for (let i = 0; i < N; i++) {
      const b = 9 * i;
      const cx = cpu[b + OFFSET.uv] * t.W;
      const cy = cpu[b + OFFSET.uv + 1] * t.H;
      if (cx < wx0 || cx > wx1 || cy < wy0 || cy > wy1) continue;
      info.visible++;
      if (info.drawn >= OUTLINE_MAX) continue;
      const sx = Math.exp(cpu[b + OFFSET.log_s]) * L * z;
      const sy = Math.exp(cpu[b + OFFSET.log_s + 1]) * L * z;
      const th = cpu[b + OFFSET.rot];
      const r = Math.max(0, Math.min(1, cpu[b + OFFSET.color])) * 255;
      const gg = Math.max(0, Math.min(1, cpu[b + OFFSET.color + 1])) * 255;
      const bb = Math.max(0, Math.min(1, cpu[b + OFFSET.color + 2])) * 255;
      g.strokeStyle = `rgb(${r | 0},${gg | 0},${bb | 0})`;
      g.beginPath();
      // the kernel's R = [[c, s], [-s, c]] maps the first axis to (cos th, -sin th): canvas rotation -th
      g.ellipse(ox + cx * z, oy + cy * z, Math.max(0.3, sx), Math.max(0.3, sy), -th, 0, 2 * Math.PI);
      g.stroke();
      info.drawn++;
    }
    g.restore();
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

  /** Distance and midpoint of the first two active pointers (device pixels). */
  private pinchState(): { dist: number; mx: number; my: number } | null {
    if (this.pointers.size < 2) return null;
    const [a, b] = [...this.pointers.values()];
    return { dist: Math.max(1, Math.hypot(b[0] - a[0], b[1] - a[1])), mx: (a[0] + b[0]) / 2, my: (a[1] + b[1]) / 2 };
  }

  private bindPointer(): void {
    const c = this.input;
    c.style.cursor = 'grab';
    c.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      const [x, y] = this.canvasPos(e);
      const t = this.transform();
      try { c.setPointerCapture(e.pointerId); } catch { /* synthetic pointers (tests) cannot be captured */ }
      this.pointers.set(e.pointerId, [x, y]);
      if (this.pointers.size >= 2) {
        // a second finger: whatever the first one was doing becomes a pinch (zoom about the midpoint + pan with it)
        if (this.drag === 'roi') this.roi.cancel(this.roiBefore);
        this.drag = 'pinch';
        this.pinch = this.pinchState();
        return;
      }
      this.lastX = x;
      this.lastY = y;
      if (this.splitVisible && Math.abs(x - this.splitFrac * this.width) <= SPLIT_GRAB_PX * this.dpr) {
        this.drag = 'split';
      } else if (this.tool === 'roi' && t) {
        const [sx, sy] = this.toSource(x, y);
        this.roiBefore = this.roi.roi;
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
      if (this.pointers.has(e.pointerId)) this.pointers.set(e.pointerId, [x, y]);
      if (this.drag === 'pinch') {
        const p = this.pinchState();
        if (p && this.pinch) {
          const z = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, this.zoom * (p.dist / this.pinch.dist)));
          // the source point under the old midpoint moves to the new midpoint, at the new zoom
          const sx = (this.pinch.mx - this.offsetX) / this.zoom;
          const sy = (this.pinch.my - this.offsetY) / this.zoom;
          this.zoom = z;
          this.offsetX = p.mx - sx * z;
          this.offsetY = p.my - sy * z;
          this.fitted = false;
          this.pinch = p;
          this.changed();
        }
        return;
      }
      if (this.drag === 'pan') {
        this.offsetX += x - this.lastX;
        this.offsetY += y - this.lastY;
        this.lastX = x;
        this.lastY = y;
        this.fitted = false;
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
      if (t && this.events.onHover) {
        const [sx, sy] = this.toSource(x, y);
        const inside = sx >= 0 && sy >= 0 && sx < t.W && sy < t.H;
        if (inside) { this.hoverOn = true; this.events.onHover(sx, sy); }
        else if (this.hoverOn) { this.hoverOn = false; this.events.onHover(null, null); }
      }
    });
    c.addEventListener('pointerleave', () => { if (this.hoverOn) { this.hoverOn = false; this.events.onHover?.(null, null); } });
    const stop = (e: PointerEvent) => {
      this.pointers.delete(e.pointerId);
      if (this.drag === 'pinch') {
        if (this.pointers.size >= 2) { this.pinch = this.pinchState(); return; }
        // one finger left: it continues as a pan from where it is
        const rest = [...this.pointers.values()][0];
        this.pinch = null;
        if (rest) { this.drag = 'pan'; [this.lastX, this.lastY] = rest; return; }
      }
      if (this.pointers.size) return; // another pointer is still down
      if (this.drag === 'roi') this.roi.end();
      this.drag = null;
      c.style.cursor = this.tool === 'roi' ? 'crosshair' : 'grab';
    };
    c.addEventListener('pointerup', stop);
    c.addEventListener('pointercancel', stop);
    // Safari's own page pinch would otherwise zoom the whole page
    document.addEventListener('gesturestart', (e) => e.preventDefault(), { passive: false });
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
    clearTimeout(this.resizeTimer);
    clearTimeout(this.entryCheckTimer);
    this.errorMap?.close();
    this.raster?.destroy();
    this.edge?.destroy();
    this.bufA?.destroy();
    this.bufB?.destroy();
    this.presenter.destroy();
    this.resampler.destroy();
  }
}
