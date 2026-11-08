// The stage background: the dot grid, each dot a small fixed translucent ellipse. Far from the pointer the ellipses
// keep their own orientation; near it they turn to point at the pointer (orientation only, no stretching), like
// filings over a magnet. 2-D canvas at CSS resolution, every frame, a static frame when motion is off. The canvas is
// blurred by CSS (the frosted layer above it), so the ellipses are drawn large and strong.

const SPACING = 26;
const RX = 5.6;
const RY = 2.0;
const SIGMA = 330; // reach of the pointer (px)
const NEAR = 0.03; // influence below which a dot is drawn in the static batch

interface Dot { x: number; y: number; base: number }

function cssVar(name: string, fallback: string): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

/** Deterministic 0..1 hash of an integer. */
function hash(i: number): number {
  let x = Math.imul(i + 0x9e3779b9, 0x85ebca6b);
  x ^= x >>> 13;
  x = Math.imul(x, 0xc2b2ae35);
  x ^= x >>> 16;
  return (x >>> 0) / 4294967296;
}

export class BgField {
  private readonly g: CanvasRenderingContext2D;
  private w = 0;
  private h = 0;
  private raf = 0;
  private px = -1e6;
  private py = -1e6;
  private sx = -1e6;
  private sy = -1e6;
  private dots: Dot[] = [];
  private ink = '#111312';
  private alpha = 0.26;
  private frame = 0;
  private wasMoving = true;

  constructor(readonly canvas: HTMLCanvasElement, private readonly motion: () => boolean) {
    const g = canvas.getContext('2d');
    if (!g) throw new Error('2D background unavailable');
    this.g = g;
    this.resize();
    window.addEventListener('resize', () => this.resize());
    window.addEventListener('pointermove', (e) => { this.px = e.clientX; this.py = e.clientY; });
    window.addEventListener('pointerleave', () => { this.px = -1e6; this.py = -1e6; });
    document.addEventListener('visibilitychange', () => { if (!document.hidden) this.schedule(); });
    this.schedule();
  }

  private resize(): void {
    this.w = this.canvas.width = Math.max(1, window.innerWidth);
    this.h = this.canvas.height = Math.max(1, window.innerHeight);
    this.dots = [];
    let k = 0;
    for (let y = SPACING / 2; y < this.h; y += SPACING) {
      for (let x = SPACING / 2; x < this.w; x += SPACING) this.dots.push({ x, y, base: hash(k++) * Math.PI });
    }
    this.draw();
  }

  private schedule(): void {
    cancelAnimationFrame(this.raf);
    this.raf = requestAnimationFrame(() => this.tick());
  }

  private tick(): void {
    if (document.hidden) return;
    const moving = this.motion();
    if (moving) this.draw();
    else if (this.wasMoving) { this.sx = this.sy = -1e6; this.draw(); }
    this.wasMoving = moving;
    this.schedule();
  }

  private draw(): void {
    const g = this.g;
    if (this.frame++ % 60 === 0) {
      this.ink = cssVar('--ink', '#111312');
      this.alpha = getComputedStyle(document.documentElement).colorScheme === 'dark' ? 0.75 : 0.62;
    }
    const hasPointer = this.px > -1e5 && this.motion();
    if (hasPointer) {
      if (this.sx < -1e5) { this.sx = this.px; this.sy = this.py; }
      this.sx += (this.px - this.sx) * 0.45;
      this.sy += (this.py - this.sy) * 0.45;
    }
    const inv = -1 / (2 * SIGMA * SIGMA);
    g.clearRect(0, 0, this.w, this.h);
    g.fillStyle = this.ink;
    // the static batch: one path for every dot out of the pointer's reach
    g.globalAlpha = this.alpha;
    g.beginPath();
    const near: { d: Dot; s: number; ang: number }[] = [];
    for (const d of this.dots) {
      let s = 0;
      if (hasPointer) {
        const dx = this.sx - d.x, dy = this.sy - d.y;
        s = Math.exp((dx * dx + dy * dy) * inv);
      }
      if (s < NEAR) {
        g.moveTo(d.x + RX * Math.cos(d.base), d.y + RX * Math.sin(d.base));
        g.ellipse(d.x, d.y, RX, RY, d.base, 0, 2 * Math.PI);
      } else near.push({ d, s, ang: Math.atan2(this.sy - d.y, this.sx - d.x) });
    }
    g.fill();
    // the dots near the pointer turn towards it, brighter the closer they are
    for (const n of near) {
      let diff = n.ang - n.d.base;
      diff = Math.atan2(Math.sin(diff), Math.cos(diff)); // shortest turn (the ellipse is symmetric, so fold to +-90 deg)
      if (diff > Math.PI / 2) diff -= Math.PI; else if (diff < -Math.PI / 2) diff += Math.PI;
      const a = n.d.base + diff * n.s;
      g.globalAlpha = this.alpha * (1 + 1.8 * n.s);
      g.beginPath();
      g.ellipse(n.d.x, n.d.y, RX, RY, a, 0, 2 * Math.PI);
      g.fill();
    }
    g.globalAlpha = 1;
  }
}
