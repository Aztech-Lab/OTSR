// Starlight over the stage, deep-space style: a few huge, very soft glows (hundreds of pixels across) drifting
// slowly and breathing, a layer of medium soft stars that twinkle, some with long soft flares, and a sprinkling of
// tiny sharp points for depth. White and the copper of the theme; on the light theme "white" becomes a soft ink
// glow, since white would vanish. 2-D canvas at CSS resolution, 30 fps, a static frame when motion is off.

type Kind = 'glow' | 'star' | 'point';

interface Star { kind: Kind; x: number; y: number; r: number; vx: number; vy: number; phase: number; rate: number; copper: boolean; flare: boolean }

const FPS = 30;
const SPRITE = 128;

function cssVar(name: string, fallback: string): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

function rgba(hex: string, a: number): string {
  const m = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex);
  return m ? `rgba(${parseInt(m[1], 16)},${parseInt(m[2], 16)},${parseInt(m[3], 16)},${a})` : `rgba(255,255,255,${a})`;
}

/** A radial sprite: `core` is the radius fraction that stays at full strength (0 = a pure Gaussian-like glow). */
function sprite(hex: string, core: number): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = c.height = SPRITE;
  const g = c.getContext('2d')!;
  const grad = g.createRadialGradient(SPRITE / 2, SPRITE / 2, 0, SPRITE / 2, SPRITE / 2, SPRITE / 2);
  grad.addColorStop(0, rgba(hex, 1));
  grad.addColorStop(core, rgba(hex, 1));
  grad.addColorStop(core + (1 - core) * 0.35, rgba(hex, 0.32));
  grad.addColorStop(core + (1 - core) * 0.7, rgba(hex, 0.08));
  grad.addColorStop(1, rgba(hex, 0));
  g.fillStyle = grad;
  g.fillRect(0, 0, SPRITE, SPRITE);
  return c;
}

export class Stars {
  private readonly g: CanvasRenderingContext2D;
  private w = 0;
  private h = 0;
  private raf = 0;
  private last = 0;
  private stars: Star[] = [];
  private glowLight: HTMLCanvasElement | null = null;
  private glowCopper: HTMLCanvasElement | null = null;
  private starLight: HTMLCanvasElement | null = null;
  private starCopper: HTMLCanvasElement | null = null;
  private lightHex = '#ffffff';
  private copperHex = '#d9a25f';
  private dark = true;
  private key = '';
  private frame = 0;
  private wasMoving = true;

  constructor(readonly canvas: HTMLCanvasElement, private readonly motion: () => boolean) {
    const g = canvas.getContext('2d');
    if (!g) throw new Error('2D background unavailable');
    this.g = g;
    const add = (kind: Kind, n: number, make: () => Partial<Star>) => {
      for (let k = 0; k < n; k++) this.stars.push({ kind, x: Math.random(), y: Math.random(), r: 1, vx: 0, vy: 0, phase: Math.random() * 6.28, rate: 1, copper: Math.random() < 0.45, flare: false, ...make() });
    };
    // huge soft glows: a quarter of the screen across, barely moving, breathing over ~10 s
    add('glow', 8, () => ({ r: 110 + 170 * Math.random() ** 1.3, vx: (Math.random() - 0.5) * 2.4, vy: -(0.6 + 1.6 * Math.random()), rate: 0.25 + 0.25 * Math.random() }));
    // medium soft stars, some with long flares
    add('star', 16, () => ({ r: 14 + 36 * Math.random(), vx: (Math.random() - 0.5) * 3, vy: -(1.5 + 3 * Math.random()), rate: 0.4 + 0.6 * Math.random(), flare: Math.random() < 0.5 }));
    // tiny sharp points for depth
    add('point', 46, () => ({ r: 0.7 + 1.1 * Math.random(), vx: (Math.random() - 0.5) * 4, vy: -(2 + 5 * Math.random()), rate: 0.8 + 1.4 * Math.random() }));
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
    this.dark = dark;
    this.lightHex = dark ? '#ffffff' : cssVar('--ink', '#111312');
    this.copperHex = copper;
    this.glowLight = sprite(this.lightHex, 0);
    this.glowCopper = sprite(copper, 0);
    this.starLight = sprite(this.lightHex, 0.12);
    this.starCopper = sprite(copper, 0.12);
  }

  private draw(t: number, dt: number): void {
    const g = this.g;
    const w = this.w, h = this.h;
    if (this.frame++ % 60 === 0 || !this.glowLight) this.sprites();
    const dark = this.dark;
    g.clearRect(0, 0, w, h);
    // luminous blending on the dark stage; plain blending (soft tinted bokeh) on the light one
    g.globalCompositeOperation = dark ? 'lighter' : 'source-over';
    for (const s of this.stars) {
      s.x = (s.x + (s.vx * dt) / w + 1) % 1;
      s.y = (s.y + (s.vy * dt) / h + 1) % 1;
      const x = s.x * w, y = s.y * h;
      const tw = 0.5 + 0.5 * Math.sin(t * s.rate + s.phase);
      if (s.kind === 'glow') {
        const r = s.r * (0.9 + 0.2 * tw);
        g.globalAlpha = (dark ? 0.16 : 0.11) * (0.7 + 0.3 * tw);
        g.drawImage((s.copper ? this.glowCopper : this.glowLight)!, x - r, y - r, 2 * r, 2 * r);
      } else if (s.kind === 'star') {
        const r = s.r * (0.85 + 0.3 * tw);
        g.globalAlpha = (dark ? 0.55 : 0.32) * (0.45 + 0.55 * tw);
        g.drawImage((s.copper ? this.starCopper : this.starLight)!, x - r, y - r, 2 * r, 2 * r);
        if (s.flare) {
          // long soft flares that fade out from the core
          const len = r * 2.6 * (0.6 + 0.4 * tw);
          const hex = s.copper ? this.copperHex : this.lightHex;
          g.globalAlpha = (dark ? 0.5 : 0.3) * tw;
          for (const [dx, dy] of [[1, 0], [0, 1]] as const) {
            const grad = g.createLinearGradient(x - dx * len, y - dy * len, x + dx * len, y + dy * len);
            grad.addColorStop(0, rgba(hex, 0));
            grad.addColorStop(0.5, rgba(hex, 1));
            grad.addColorStop(1, rgba(hex, 0));
            g.strokeStyle = grad;
            g.lineWidth = 1.2;
            g.beginPath();
            g.moveTo(x - dx * len, y - dy * len);
            g.lineTo(x + dx * len, y + dy * len);
            g.stroke();
          }
        }
      } else {
        const r = s.r * (0.7 + 0.5 * tw);
        g.globalAlpha = (dark ? 0.9 : 0.45) * (0.3 + 0.7 * tw);
        g.fillStyle = s.copper ? this.copperHex : this.lightHex;
        g.beginPath();
        g.arc(x, y, r, 0, 2 * Math.PI);
        g.fill();
      }
    }
    g.globalAlpha = 1;
    g.globalCompositeOperation = 'source-over';
  }
}
