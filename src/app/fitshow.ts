// The fit animation:
//   1. the uniform lattice, then the real optimal-transport stages (the barycentric map after every eps-scaling
//      stage of the Sinkhorn loop) as a glowing, flickering mesh on the overlay, while the WebGPU render waits
//      behind a curtain;
//   2. the Gaussians themselves, rendered live by the engine from the parameters the optimiser is changing: a
//      display-only kernel copies them with each colour replaced by a random pastel colour times the Gaussian's
//      luma (grey at first, the colour comes in over a second) and the sizes growing from points, while the mesh
//      fades out;
//   3. a scan line from left to right that gives every Gaussian its true colour back.
// Positions, shapes and the pace are the real process; only the colours of phase 2 are random.

import { bindGroup, grid1d, runPass, type GpuContext, type Pipelines } from '../engine';
import type { RoiTransform } from './roi';
import type { AppModel } from './store';
import type { Viewer } from './viewer';

const STAGE_MS = 70; // playback of one OT stage (the stages arrive together after the single Sinkhorn submit)
const HOLD_MS = 160; // pause on the final lattice before the Gaussians appear
const PAINT_MIN_MS = 1800; // the recoloured Gaussians are shown at least this long
const MESH_FADE_MS = 900;
const REVEAL_MS = 1300;
const RENDER_INTERVAL_MS = 40; // live render cadence in phase 2 (the optimiser keeps running meanwhile)
const RENDER_INTERVAL_COMPACT_MS = 90; // on phones the fit itself needs the GPU more
const MAX_NODES = 110; // mesh nodes per axis (subsampled)
const MAX_NODES_COMPACT = 40; // on phones: an eighth of the nodes, still a dense mesh on a small screen
const FLICKER_MS = 90;
const LATTICE_FRAME_MS = 33; // the mesh is redrawn at most ~30 times a second: each overlay frame costs the fit time
const FLICKER_LEVELS = 4; // node brightness levels: one fill per level instead of two per node
const FX_BYTES = 32;

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
  private phase: 'off' | 'lattice' | 'paint' | 'reveal' = 'off';
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
  private lastRender = 0;
  private resolveFinish: (() => void) | null = null;
  private model: AppModel | null = null;
  private display: GPUBuffer | null = null;
  private uni: GPUBuffer | null = null;
  private bg: GPUBindGroup | null = null;
  private seed = 0;
  private raf = 0;
  private lastLatticeFrame = 0;
  private colour = '#b8793c';
  private colour2 = '#d9a25f';
  private paper = '#ebebe8';
  private dark = false;

  constructor(private readonly viewer: Viewer, private readonly ctx: GpuContext, private readonly pipes: Pipelines) {}

  get active(): boolean {
    return this.phase !== 'off';
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
    this.holdStart = 0;
    this.seed = (Math.random() * 1e9) >>> 0;
    this.colour = cssVar('--copper', '#b8793c');
    this.colour2 = cssVar('--copper-2', '#d9a25f');
    this.paper = cssVar('--bg', '#ebebe8');
    this.dark = getComputedStyle(document.documentElement).colorScheme === 'dark';
    this.phase = 'lattice';
    this.playStart = performance.now();
    this.viewer.curtain = true;
    this.viewer.painter = (g, t, dpr) => this.paint(g, t, dpr);
    // between the mesh fading out and the scan line, the overlay is blank
    this.viewer.painterBlank = () => this.phase === 'paint' && performance.now() - this.paintStart >= MESH_FADE_MS;
    this.viewer.requestRender();
    document.body.classList.add('fitshow');
    this.loop();
  }

  /** The model under optimisation: from now on the viewer draws its recoloured copy (refreshed before every render). */
  attachModel(model: AppModel): void {
    if (this.phase === 'off') return;
    this.releaseModel();
    const d = this.ctx.device;
    this.model = model;
    this.display = d.createBuffer({ label: 'fitshow.display', size: model.N * 36, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.uni = d.createBuffer({ label: 'fitshow.fx', size: FX_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bg = bindGroup(d, this.pipes.recolour, { 0: this.uni, 1: model.params, 2: this.display }, 'recolour');
    this.viewer.beforeRender = (enc) => this.encodeRecolour(enc);
    this.viewer.overrideParams(this.display);
    // a first copy right away, so other renders of the copy (the monitor's crops) never see an empty buffer
    const enc = d.createCommandEncoder({ label: 'fitshow.init' });
    this.encodeRecolour(enc);
    d.queue.submit([enc.finish()]);
  }

  /** The recoloured copy of the parameters (valid while the show runs with a model). */
  get displayBuffer(): GPUBuffer | null {
    return this.display;
  }

  private releaseModel(): void {
    this.viewer.beforeRender = null;
    if (this.model && this.viewer.hasModel) this.viewer.overrideParams(null);
    this.display?.destroy();
    this.uni?.destroy();
    this.display = null;
    this.uni = null;
    this.bg = null;
    this.model = null;
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

  /** Lets the lattice and the live phase finish, plays the reveal, then returns the view to normal. */
  finish(): Promise<void> {
    if (this.phase === 'off') return Promise.resolve();
    this.finishRequested = true;
    return new Promise((resolve) => { this.resolveFinish = resolve; });
  }

  /** Stops immediately (cancelled or failed fit) and gives the viewer its real parameters back. */
  cancel(): void {
    if (this.phase === 'off') return;
    this.phase = 'off';
    cancelAnimationFrame(this.raf);
    this.releaseModel();
    this.viewer.curtain = false;
    this.viewer.painter = null;
    this.viewer.painterBlank = null;
    this.viewer.requestRender();
    document.body.classList.remove('fitshow');
    const r = this.resolveFinish;
    this.resolveFinish = null;
    r?.();
  }

  // ---- the recoloured copy ------------------------------------------------------------------------------------

  /** Refreshes the recoloured copy from the live parameters (encoded before the render that uses it). */
  encodeRecolour(enc: GPUCommandEncoder): void {
    if (!this.model || !this.uni || !this.bg) return;
    const now = performance.now();
    let amount = 0, sizeLog = 0, scanU = 0;
    if (this.phase === 'paint') {
      const age = now - this.paintStart;
      amount = ease((age - 350) / 1500);
      sizeLog = Math.log(0.12 + 0.88 * ease(age / 900));
    } else if (this.phase === 'reveal') {
      amount = 1;
      scanU = ease((now - this.revealStart) / REVEAL_MS) * (this.W / this.L);
    } else {
      sizeLog = Math.log(0.12);
    }
    const ab = new ArrayBuffer(FX_BYTES);
    const u = new Uint32Array(ab);
    const f = new Float32Array(ab);
    u[0] = this.model.N;
    u[1] = this.seed;
    f[2] = amount;
    f[3] = sizeLog;
    f[4] = scanU;
    this.ctx.device.queue.writeBuffer(this.uni, 0, ab);
    runPass(enc, this.pipes.recolour, this.bg, grid1d(this.model.N), 'recolour');
  }

  // ---- timeline ----------------------------------------------------------------------------------------------

  private loop(): void {
    this.raf = requestAnimationFrame(() => {
      if (!this.active) return;
      const now = performance.now();
      this.advance(now);
      if (this.active) {
        const interval = matchMedia('(max-width: 960px)').matches ? RENDER_INTERVAL_COMPACT_MS : RENDER_INTERVAL_MS;
        if (this.phase === 'lattice') {
          if (now - this.lastLatticeFrame >= LATTICE_FRAME_MS) { this.lastLatticeFrame = now; this.viewer.refreshOverlay(); }
        } else if (now - this.lastRender >= interval) { this.lastRender = now; this.viewer.requestRender(); }
        // the mesh fades out with the renders (each redraws the overlay); only the scan line, after the fit, moves every
        // frame: a 2-D overlay changed at the display rate held the fit underneath at a third of its speed
        else if (this.phase === 'reveal') this.viewer.refreshOverlay();
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
      if (atEnd && now - this.holdStart >= HOLD_MS && this.display) {
        // the live render takes over: curtain up, the recoloured copy is drawn from here on
        this.phase = 'paint';
        this.paintStart = now;
        this.viewer.curtain = false;
        this.viewer.requestRender();
      } else if (this.finishRequested && !this.display) {
        this.cancel(); // nothing to show (cancelled before the model existed)
      }
    } else if (this.phase === 'paint') {
      if (this.finishRequested && now - this.paintStart >= PAINT_MIN_MS) {
        this.phase = 'reveal';
        this.revealStart = now;
      }
    } else if (this.phase === 'reveal') {
      if (now - this.revealStart >= REVEAL_MS + 60) this.cancel();
    }
  }

  // ---- overlay ------------------------------------------------------------------------------------------------

  /** The blank "paper" over the image rectangle while the GPU render is hidden. */
  private drawPaper(g: CanvasRenderingContext2D, t: RoiTransform): void {
    g.fillStyle = this.paper;
    g.fillRect(t.offsetX, t.offsetY, this.W * t.zoom, this.H * t.zoom);
  }

  private paint(g: CanvasRenderingContext2D, t: RoiTransform, dpr: number): void {
    const now = performance.now();
    if (this.phase === 'lattice') {
      this.drawPaper(g, t);
      this.drawLattice(g, t, dpr, now, 1);
    } else if (this.phase === 'paint') {
      const meshAlpha = 1 - ease((now - this.paintStart) / MESH_FADE_MS);
      if (meshAlpha > 0.01) this.drawLattice(g, t, dpr, now, meshAlpha);
    } else if (this.phase === 'reveal') {
      const u = ease((now - this.revealStart) / REVEAL_MS);
      const sx = this.viewer.offsetX + this.W * this.viewer.zoom * u;
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
    const compact = matchMedia('(max-width: 960px)').matches;
    const step = Math.max(1, Math.ceil(Math.max(nx, ny) / (compact ? MAX_NODES_COMPACT : MAX_NODES)));
    const z = t.zoom * this.L;
    const ox = t.offsetX, oy = t.offsetY;
    const px = (k: number) => ox + (A[2 * k] + (B[2 * k] - A[2 * k]) * f) * z;
    const py = (k: number) => oy + (A[2 * k + 1] + (B[2 * k + 1] - A[2 * k + 1]) * f) * z;
    g.save();
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
    // hairlines (one canvas pixel) take the fast path of the 2-D canvas; on phones' 2x canvases a little stronger
    g.lineWidth = 1;
    g.globalAlpha = alpha * (this.dark ? 0.32 : 0.28) * (dpr > 1.25 ? 1.5 : 1);
    g.stroke(path);
    // flickering nodes: every node falls in one of a few brightness levels (re-drawn every FLICKER_MS), and each level
    // is one path of small squares: two fills per level instead of two fillRect calls per node
    const bucket = Math.floor(now / FLICKER_MS);
    const core = this.dark ? this.colour2 : this.colour;
    const glows = Array.from({ length: FLICKER_LEVELS }, () => new Path2D());
    const cores = Array.from({ length: FLICKER_LEVELS }, () => new Path2D());
    for (let i = 0; i < ny; i += step) {
      for (let j = 0; j < nx; j += step) {
        const k = i * nx + j;
        const x = px(k), y = py(k);
        const lv = Math.min(FLICKER_LEVELS - 1, Math.floor(hash(k * 31 + bucket * 7) * FLICKER_LEVELS));
        glows[lv].rect(x - 3 * dpr, y - 3 * dpr, 6 * dpr, 6 * dpr);
        cores[lv].rect(x - 1 * dpr, y - 1 * dpr, 2 * dpr, 2 * dpr);
      }
    }
    for (let lv = 0; lv < FLICKER_LEVELS; lv++) {
      const fl = 0.35 + 0.65 * ((lv + 0.5) / FLICKER_LEVELS);
      g.globalAlpha = alpha * 0.18 * fl;
      g.fillStyle = this.colour2;
      g.fill(glows[lv]);
      g.globalAlpha = alpha * fl;
      g.fillStyle = core;
      g.fill(cores[lv]);
    }
    g.restore();
  }
}
