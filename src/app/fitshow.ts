// The fit animation, drawn on the 2-D overlay while the WebGPU render waits behind a curtain:
//   1. the uniform lattice, then the real optimal-transport stages (the barycentric map after every eps-scaling
//      stage of the Sinkhorn loop) as a glowing, flickering mesh that settles on the image structure;
//   2. the Gaussians at their actual positions (read back from the GPU while the optimiser runs) growing from
//      nothing, grey at first, into random watercolour, while the mesh fades out;
//   3. a scan line from left to right that reveals the fitted image in its true colours.
// Everything here is cosmetic: positions and shapes come from the engine, the colours of phase 2 are random.

import { OFFSET } from '../core/spec';
import type { RoiTransform } from './roi';
import type { Viewer } from './viewer';

const STAGE_MS = 120; // playback of one OT stage
const HOLD_MS = 260; // pause on the final lattice before the paint phase
const PAINT_MIN_MS = 1500; // the watercolour is shown at least this long
const MESH_FADE_MS = 900;
const WARM_MS = 450; // the GPU render runs under the opaque overlay before the reveal (tile lists settle)
const REVEAL_MS = 1300;
const MAX_BLOBS = 7000;
const REPAINT_MIN_MS = 500; // the watercolour cache is rebuilt at most this often
const MAX_NODES = 110; // mesh nodes per axis (subsampled)
const FLICKER_MS = 90;

interface Grid { ny: number; nx: number }

/** Deterministic 0..1 hash of an integer. */
function hash(i: number): number {
  let x = Math.imul(i + 0x9e3779b9, 0x85ebca6b);
  x ^= x >>> 13;
  x = Math.imul(x, 0xc2b2ae35);
  x ^= x >>> 16;
  return (x >>> 0) / 4294967296;
}

function cssVar(name: string, fallback: string): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

function ease(u: number): number {
  return u <= 0 ? 0 : u >= 1 ? 1 : 0.5 - 0.5 * Math.cos(Math.PI * u);
}

export class FitShow {
  private phase: 'off' | 'lattice' | 'paint' | 'warm' | 'reveal' = 'off';
  private W = 0;
  private H = 0;
  private L = 1;
  private grid: Grid = { ny: 0, nx: 0 };
  /** Stage maps in box units (x pixels = value x L); the first entry is the uniform lattice. */
  private stages: Float32Array[] = [];
  private latticeDone = false;
  private playStart = 0;
  private holdStart = 0;
  private finishRequested = false;
  private paintStart = 0;
  private revealStart = 0;
  private resolveFinish: (() => void) | null = null;
  private snap: Float32Array | null = null;
  private snapVersion = 0;
  private painted = { version: -1, key: '', ramp: -1, at: 0 };
  private off: HTMLCanvasElement | null = null;
  private offG: CanvasRenderingContext2D | null = null;
  private raf = 0;
  private colour = '#b8793c';
  private colour2 = '#d9a25f';
  private paper = '#ebebe8';
  private dark = false;

  constructor(private readonly viewer: Viewer) {}

  get active(): boolean {
    return this.phase !== 'off';
  }

  get hasSnapshot(): boolean {
    return this.snap !== null;
  }

  get phaseName(): string {
    return this.phase;
  }

  /** Starts the show for a W x H image: curtain down, uniform lattice up. */
  start(W: number, H: number): void {
    this.cancel();
    this.W = W;
    this.H = H;
    this.L = Math.max(W, H);
    this.stages = [];
    this.grid = { ny: 0, nx: 0 };
    this.latticeDone = false;
    this.finishRequested = false;
    this.snap = null;
    this.snapVersion = 0;
    this.painted = { version: -1, key: '', ramp: -1, at: 0 };
    this.colour = cssVar('--copper', '#b8793c');
    this.paper = cssVar('--bg', '#ebebe8');
    this.colour2 = cssVar('--copper-2', '#d9a25f');
    this.dark = getComputedStyle(document.documentElement).colorScheme === 'dark';
    this.phase = 'lattice';
    this.playStart = performance.now();
    this.viewer.curtain = true;
    this.viewer.painter = (g, t, dpr) => this.paint(g, t, dpr);
    this.viewer.requestRender();
    document.body.classList.add('fitshow');
    this.loop();
  }

  /** The uniform lattice before any transport (cell centres of the OT grid). */
  private uniform(grid: Grid): Float32Array {
    const T = new Float32Array(grid.ny * grid.nx * 2);
    const ex = this.W / this.L, ey = this.H / this.L;
    for (let i = 0; i < grid.ny; i++) for (let j = 0; j < grid.nx; j++) {
      const k = i * grid.nx + j;
      T[2 * k] = ((j + 0.5) / grid.nx) * ex;
      T[2 * k + 1] = ((i + 0.5) / grid.ny) * ey;
    }
    return T;
  }

  /** Mean absolute deviation of a map from its centroid (box units): collapsed early stages have a small one. */
  private spread(T: Float32Array): number {
    const n = T.length / 2;
    let mx = 0, my = 0;
    for (let i = 0; i < n; i++) { mx += T[2 * i]; my += T[2 * i + 1]; }
    mx /= n; my /= n;
    let s = 0;
    for (let i = 0; i < n; i++) s += Math.abs(T[2 * i] - mx) + Math.abs(T[2 * i + 1] - my);
    return s / n;
  }

  /** A Sinkhorn stage from the engine. The nearly collapsed large-eps stages are skipped (they would implode the lattice). */
  otStage(T: Float32Array, grid: Grid): void {
    if (this.phase === 'off') return;
    if (!this.stages.length) {
      this.grid = grid;
      this.stages.push(this.uniform(grid));
      this.playStart = performance.now();
    }
    const u = this.spread(this.stages[0]);
    if (this.spread(T) < 0.55 * u) return;
    this.stages.push(Float32Array.from(T));
  }

  otDone(): void {
    this.latticeDone = true;
  }

  /** The current parameters (9 floats per Gaussian) read back from the GPU. */
  snapshot(params: Float32Array): void {
    if (this.phase === 'off') return;
    this.snap = params;
    this.snapVersion++;
  }

  /** Lets the lattice and paint phases finish, plays the reveal, then returns the view to normal. */
  finish(): Promise<void> {
    if (this.phase === 'off') return Promise.resolve();
    this.finishRequested = true;
    return new Promise((resolve) => { this.resolveFinish = resolve; });
  }

  /** Stops immediately (cancelled or failed fit). */
  cancel(): void {
    if (this.phase === 'off') return;
    this.phase = 'off';
    cancelAnimationFrame(this.raf);
    this.viewer.curtain = false;
    this.viewer.painter = null;
    this.viewer.requestRender();
    document.body.classList.remove('fitshow');
    const r = this.resolveFinish;
    this.resolveFinish = null;
    r?.();
  }

  // ---- timeline ----------------------------------------------------------------------------------------------

  private loop(): void {
    this.raf = requestAnimationFrame(() => {
      if (!this.active) return;
      this.advance(performance.now());
      if (this.active) {
        this.viewer.refreshOverlay();
        this.loop();
      }
    });
  }

  private latticeIndex(now: number): number {
    const last = this.stages.length - 1;
    return Math.max(0, Math.min(last, (now - this.playStart) / STAGE_MS));
  }

  private advance(now: number): void {
    if (this.phase === 'lattice') {
      const last = this.stages.length - 1;
      const atEnd = this.latticeDone && (last < 1 || this.latticeIndex(now) >= last);
      if (atEnd && !this.holdStart) this.holdStart = now;
      // the paint phase needs a snapshot; a model loaded without OT has no stages at all
      if ((atEnd && now - this.holdStart >= HOLD_MS && this.snap) || (this.latticeDone && !this.stages.length && this.snap)) {
        this.phase = 'paint';
        this.paintStart = now;
      } else if (this.finishRequested && !this.snap) {
        // nothing to paint (cancelled before the first snapshot): straight to the end
        this.cancel();
      }
    } else if (this.phase === 'paint') {
      if (this.finishRequested && now - this.paintStart >= PAINT_MIN_MS) {
        // the overlay stays opaque while the real render warms up underneath
        this.phase = 'warm';
        this.revealStart = now + WARM_MS;
        this.viewer.curtain = false;
        this.viewer.requestRender();
      }
    } else if (this.phase === 'warm') {
      this.viewer.requestRender();
      if (now >= this.revealStart) this.phase = 'reveal';
    } else if (this.phase === 'reveal') {
      if (now - this.revealStart >= REVEAL_MS) this.cancel();
    }
  }

  // ---- drawing -------------------------------------------------------------------------------------------------

  /** The blank "paper" over the image rectangle (the GPU render stays hidden), optionally only right of `clipX`. */
  private drawPaper(g: CanvasRenderingContext2D, t: RoiTransform, clipX: number | null): void {
    const x0 = t.offsetX, y0 = t.offsetY, w = this.W * t.zoom, hh = this.H * t.zoom;
    const left = clipX === null ? x0 : Math.max(x0, clipX);
    if (left >= x0 + w) return;
    g.fillStyle = this.paper;
    g.fillRect(left, y0, x0 + w - left, hh);
  }

  private paint(g: CanvasRenderingContext2D, t: RoiTransform, dpr: number): void {
    const now = performance.now();
    if (this.phase === 'lattice') {
      this.drawPaper(g, t, null);
      this.drawLattice(g, t, dpr, now, 1);
    } else if (this.phase === 'paint' || this.phase === 'warm') {
      const age = now - this.paintStart;
      const meshAlpha = 1 - ease(age / MESH_FADE_MS);
      this.drawPaper(g, t, null);
      this.drawWatercolour(g, t, age, null);
      if (meshAlpha > 0.01) this.drawLattice(g, t, dpr, now, meshAlpha);
    } else if (this.phase === 'reveal') {
      const u = ease((now - this.revealStart) / REVEAL_MS);
      const x0 = this.viewer.offsetX, x1 = this.viewer.offsetX + this.W * this.viewer.zoom;
      const sx = x0 + (x1 - x0) * u;
      this.drawPaper(g, t, sx);
      this.drawWatercolour(g, t, now - this.paintStart, sx);
      // the scan line: a trailing band and a bright core
      g.save();
      const band = 48 * dpr;
      const grad = g.createLinearGradient(sx, 0, sx + band, 0);
      grad.addColorStop(0, this.hexA(this.colour2, 0.55));
      grad.addColorStop(1, this.hexA(this.colour2, 0));
      g.fillStyle = grad;
      g.fillRect(sx, 0, band, g.canvas.height);
      g.shadowColor = this.colour2;
      g.shadowBlur = 16 * dpr;
      g.fillStyle = '#fff';
      g.fillRect(sx - 1 * dpr, 0, 2 * dpr, g.canvas.height);
      g.restore();
    }
  }

  private hexA(hex: string, a: number): string {
    const m = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex);
    return m ? `rgba(${parseInt(m[1], 16)},${parseInt(m[2], 16)},${parseInt(m[3], 16)},${a})` : hex;
  }

  /** The mesh at the current playback position, interpolated between two stages, with flickering nodes. */
  private drawLattice(g: CanvasRenderingContext2D, t: RoiTransform, dpr: number, now: number, alpha: number): void {
    if (!this.stages.length) return;
    const idx = this.latticeIndex(now);
    const a = Math.floor(idx), f = idx - a;
    const A = this.stages[a], B = this.stages[Math.min(this.stages.length - 1, a + 1)];
    const { ny, nx } = this.grid;
    const step = Math.max(1, Math.ceil(Math.max(nx, ny) / MAX_NODES));
    const z = t.zoom * this.L;
    const ox = t.offsetX, oy = t.offsetY;
    const px = (k: number) => ox + (A[2 * k] + (B[2 * k] - A[2 * k]) * f) * z;
    const py = (k: number) => oy + (A[2 * k + 1] + (B[2 * k + 1] - A[2 * k + 1]) * f) * z;
    g.save();
    g.globalAlpha = alpha;
    // mesh lines
    const path = new Path2D();
    for (let i = 0; i < ny; i += step) {
      for (let j = 0; j < nx; j += step) {
        const k = i * nx + j;
        const x = px(k), y = py(k);
        if (j + step < nx) { const k2 = k + step; path.moveTo(x, y); path.lineTo(px(k2), py(k2)); }
        if (i + step < ny) { const k2 = k + step * nx; path.moveTo(x, y); path.lineTo(px(k2), py(k2)); }
      }
    }
    g.strokeStyle = this.colour;
    g.lineWidth = 1 * dpr;
    g.globalAlpha = alpha * (this.dark ? 0.32 : 0.28);
    g.stroke(path);
    // nodes: a soft halo and a bright core, flickering
    const bucket = Math.floor(now / FLICKER_MS);
    const core = this.dark ? this.colour2 : this.colour;
    for (let i = 0; i < ny; i += step) {
      for (let j = 0; j < nx; j += step) {
        const k = i * nx + j;
        const x = px(k), y = py(k);
        const fl = 0.35 + 0.65 * hash(k * 31 + bucket * 7);
        g.globalAlpha = alpha * 0.18 * fl;
        g.fillStyle = this.colour2;
        g.fillRect(x - 3 * dpr, y - 3 * dpr, 6 * dpr, 6 * dpr);
        g.globalAlpha = alpha * fl;
        g.fillStyle = core;
        g.fillRect(x - 1 * dpr, y - 1 * dpr, 2 * dpr, 2 * dpr);
      }
    }
    g.restore();
  }

  /** The random watercolour of the latest snapshot (cached in an offscreen canvas), optionally only right of `clipX`. */
  private drawWatercolour(g: CanvasRenderingContext2D, t: RoiTransform, age: number, clipX: number | null): void {
    const snap = this.snap;
    if (!snap) return;
    const w = g.canvas.width, hgt = g.canvas.height;
    // ramps: size and opacity grow in the first second, the colour comes in after half a second
    const ramp = Math.min(1, age / 1400);
    const rampQ = Math.round(ramp * 12) / 12; // repaint the cache at most 12 times during the ramp
    const key = `${w}x${hgt}|${t.zoom}|${t.offsetX}|${t.offsetY}`;
    if (!this.off || this.off.width !== w || this.off.height !== hgt) {
      this.off = document.createElement('canvas');
      this.off.width = w;
      this.off.height = hgt;
      this.offG = this.off.getContext('2d');
      this.painted = { version: -1, key: '', ramp: -1, at: 0 };
    }
    const og = this.offG!;
    const now = performance.now();
    const stale = this.painted.version !== this.snapVersion && now - this.painted.at >= REPAINT_MIN_MS;
    if (stale || this.painted.key !== key || this.painted.ramp !== rampQ) {
      og.clearRect(0, 0, w, hgt);
      og.globalCompositeOperation = this.dark ? 'source-over' : 'multiply';
      const N = snap.length / 9;
      // a random subset (a regular stride would align with the lattice rows): each blob covers the area of its share
      const keep = Math.min(1, MAX_BLOBS / N);
      const spread = Math.sqrt(1 / keep);
      const z = t.zoom;
      const Lz = this.L * z;
      const size = (0.35 + 0.95 * ease(Math.min(1, age / 900))) * 1.35 * spread;
      const alpha = (this.dark ? 0.5 : 0.24) * ease(Math.min(1, age / 1100));
      const sat = ease(Math.max(0, (age - 450) / 1300));
      const grey = this.dark ? 150 : 150;
      og.globalAlpha = alpha;
      for (let i = 0; i < N; i++) {
        if (keep < 1 && hash(i * 7 + 3) >= keep) continue;
        const b = 9 * i;
        const cx = t.offsetX + snap[b + OFFSET.uv] * this.W * z;
        const cy = t.offsetY + snap[b + OFFSET.uv + 1] * this.H * z;
        if (cx < -40 || cy < -40 || cx > w + 40 || cy > hgt + 40) continue;
        const sx = Math.max(1.5, Math.exp(snap[b + OFFSET.log_s]) * Lz * size);
        const sy = Math.max(1.5, Math.exp(snap[b + OFFSET.log_s + 1]) * Lz * size);
        const h0 = hash(i) * 360, h1 = hash(i * 3 + 1);
        // random hue, mixed with grey while the colour "comes in"
        const r = Math.round(grey + sat * (this.hueR(h0) * (0.55 + 0.45 * h1) * 255 - grey));
        const gg = Math.round(grey + sat * (this.hueG(h0) * (0.55 + 0.45 * h1) * 255 - grey));
        const bb = Math.round(grey + sat * (this.hueB(h0) * (0.55 + 0.45 * h1) * 255 - grey));
        og.fillStyle = `rgb(${r},${gg},${bb})`;
        og.beginPath();
        og.ellipse(cx, cy, sx, sy, -snap[b + OFFSET.rot], 0, 2 * Math.PI);
        og.fill();
      }
      og.globalAlpha = 1;
      og.globalCompositeOperation = 'source-over';
      this.painted = { version: this.snapVersion, key, ramp: rampQ, at: now };
    }
    g.save();
    if (clipX !== null) {
      g.beginPath();
      g.rect(clipX, 0, w - clipX, hgt);
      g.clip();
    }
    g.drawImage(this.off, 0, 0);
    g.restore();
  }

  // pastel hue -> rgb components (0..1), light enough for multiply on paper and lighter on a dark stage
  private hueR(h: number): number { return 0.62 + 0.38 * Math.max(0, Math.cos((h * Math.PI) / 180)); }
  private hueG(h: number): number { return 0.62 + 0.38 * Math.max(0, Math.cos(((h - 120) * Math.PI) / 180)); }
  private hueB(h: number): number { return 0.62 + 0.38 * Math.max(0, Math.cos(((h - 240) * Math.PI) / 180)); }
}
