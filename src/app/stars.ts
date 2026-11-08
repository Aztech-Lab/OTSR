// Floating starlight over the stage: soft points of light in white and the copper of the theme, drifting slowly
// upwards and twinkling; a few larger ones carry a four-point flare. 2-D canvas at CSS resolution, 30 fps, a static
// frame when motion is off. On the light theme "white" becomes a soft ink glint, since white would vanish.

interface Star { x: number; y: number; r: number; vx: number; vy: number; phase: number; rate: number; copper: boolean; big: boolean }

const COUNT = 64;
const FPS = 30;
const SPRITE = 64;

function cssVar(name: string, fallback: string): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

function sprite(colour: string): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = c.height = SPRITE;
  const g = c.getContext('2d')!;
  const grad = g.createRadialGradient(SPRITE / 2, SPRITE / 2, 0, SPRITE / 2, SPRITE / 2, SPRITE / 2);
  grad.addColorStop(0, colour);
  grad.addColorStop(0.18, colour);
  grad.addColorStop(0.45, colour.replace(/[\d.]+\)$/, '0.22)'));
  grad.addColorStop(1, colour.replace(/[\d.]+\)$/, '0)'));
  g.fillStyle = grad;
  g.fillRect(0, 0, SPRITE, SPRITE);
  return c;
}

function rgba(hex: string, a: number): string {
  const m = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex);
  return m ? `rgba(${parseInt(m[1], 16)},${parseInt(m[2], 16)},${parseInt(m[3], 16)},${a})` : `rgba(255,255,255,${a})`;
}

export class Stars {
  private readonly g: CanvasRenderingContext2D;
  private w = 0;
  private h = 0;
  private raf = 0;
  private last = 0;
  private stars: Star[] = [];
  private light: HTMLCanvasElement | null = null;
  private copper: HTMLCanvasElement | null = null;
  private lightLine = '#fff';
  private copperLine = '#d9a25f';
  private key = '';
  private frame = 0;
  private wasMoving = true;

  constructor(readonly canvas: HTMLCanvasElement, private readonly motion: () => boolean) {
    const g = canvas.getContext('2d');
    if (!g) throw new Error('2D background unavailable');
    this.g = g;
    for (let k = 0; k < COUNT; k++) {
      const big = k % 9 === 0;
      this.stars.push({ x: Math.random(), y: Math.random(), r: big ? 2.2 + 1.2 * Math.random() : 0.8 + 1.2 * Math.random(), vx: (Math.random() - 0.5) * 4, vy: -(3 + 7 * Math.random()), phase: Math.random() * 6.28, rate: 0.5 + 0.9 * Math.random(), copper: Math.random() < 0.4, big });
    }
    this.resize();
    window.addEventListener('resize', () => this.resize());
    document.addEventListener('visibilitychange', () => { if (!document.hidden) this.schedule(); });
    this.schedule();
  }

  private resize(): void {
    this.w = this.canvas.width = Math.max(1, window.innerWidth);
    this.h = this.canvas.height = Math.max(1, window.innerHeight);
    this.draw(performance.now() / 1000, 0);
  }

  private schedule(): void {
    cancelAnimationFrame(this.raf);
    this.raf = requestAnimationFrame((t) => this.tick(t));
  }

  private tick(t: number): void {
    if (document.hidden) return;
    const moving = this.motion();
    if (moving) {
      if (t - this.last >= 1000 / FPS) { const dt = this.last ? Math.min(0.1, (t - this.last) / 1000) : 0; this.last = t; this.draw(t / 1000, dt); }
    } else if (this.wasMoving) {
      this.draw(0, 0);
    }
    this.wasMoving = moving;
    this.schedule();
  }

  private sprites(): void {
    const dark = getComputedStyle(document.documentElement).colorScheme === 'dark';
    const copper = cssVar('--copper-2', '#d9a25f');
    const key = `${dark}|${copper}`;
    if (key === this.key) return;
    this.key = key;
    // the "white" light: white on the dark stage, a soft ink glint on the light one
    this.lightLine = dark ? '#ffffff' : cssVar('--ink', '#111312');
    this.copperLine = copper;
    this.light = sprite(rgba(this.lightLine, dark ? 0.95 : 0.5));
    this.copper = sprite(rgba(copper, dark ? 0.9 : 0.75));
  }

  private draw(t: number, dt: number): void {
    const g = this.g;
    const w = this.w, h = this.h;
    if (this.frame++ % 60 === 0 || !this.light) this.sprites();
    g.clearRect(0, 0, w, h);
    for (const s of this.stars) {
      s.x = (s.x + (s.vx * dt) / w + 1) % 1;
      s.y = (s.y + (s.vy * dt) / h + 1) % 1;
      const x = s.x * w, y = s.y * h;
      const tw = 0.55 + 0.45 * Math.sin(t * s.rate + s.phase);
      const r = s.r * (s.big ? 5 : 4) * (0.85 + 0.3 * tw);
      g.globalAlpha = (s.big ? 0.9 : 0.7) * tw;
      g.drawImage((s.copper ? this.copper : this.light)!, x - r, y - r, 2 * r, 2 * r);
      if (s.big) {
        // a four-point flare that breathes with the twinkle
        const len = s.r * 7 * tw;
        g.strokeStyle = s.copper ? this.copperLine : this.lightLine;
        g.globalAlpha = 0.55 * tw;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(x - len, y); g.lineTo(x + len, y);
        g.moveTo(x, y - len); g.lineTo(x, y + len);
        g.stroke();
      }
    }
    g.globalAlpha = 1;
  }
}
