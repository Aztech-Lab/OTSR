// The small live demonstrations of the technical report, each a <figure class="fig demo" data-demo="…"> whose
// .demo-stage receives an SVG or a canvas and whose .demo-ctl receives its controls. Everything is computed in
// plain JavaScript on the page: the transport map of the unfold demo is the gradient of a convex potential, the
// one-dimensional transport is the quantile function, the Sinkhorn demo runs real iterations on a 48-bin problem,
// the resolution demo evaluates a handful of Gaussians per output pixel. A demo animates only while it is on screen
// and the motion setting is on; its sliders work in any case.

type Motion = () => boolean;

const NS = 'http://www.w3.org/2000/svg';

function el<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Record<string, string> = {}, ...kids: (Node | string)[]): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  e.append(...kids);
  return e;
}

function svgEl<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}): SVGElementTagNameMap[K] {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  return e;
}

function cssVar(name: string, fallback: string): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

/** A labelled range slider with a live readout. */
function slider(host: HTMLElement, label: string, min: number, max: number, step: number, value: number, fmt: (v: number) => string, onInput: (v: number) => void): HTMLInputElement {
  const input = el('input', { type: 'range', min: String(min), max: String(max), step: String(step), value: String(value) });
  const out = el('output', {}, fmt(value));
  input.addEventListener('input', () => { out.textContent = fmt(Number(input.value)); onInput(Number(input.value)); });
  host.append(el('label', { class: 'ctl' }, el('span', {}, label), input, out));
  return input;
}

function button(host: HTMLElement, label: string, onClick: () => void): HTMLButtonElement {
  const b = el('button', { type: 'button', class: 'textbtn' }, label);
  b.addEventListener('click', onClick);
  host.append(b);
  return b;
}

function toggle(host: HTMLElement, label: string, on: boolean, onChange: (v: boolean) => void): HTMLInputElement {
  const input = el('input', { type: 'checkbox' });
  input.checked = on;
  input.addEventListener('change', () => onChange(input.checked));
  host.append(el('label', { class: 'ctl check' }, input, el('span', {}, label)));
  return input;
}

/** Runs `frame(t, dt)` while the host is on screen and motion is on; `redraw()` draws the current state once. */
class Loop {
  private raf = 0;
  private last = 0;
  private visible = false;
  private button: HTMLButtonElement | null = null;
  playing = true;
  constructor(host: Element, private readonly motion: Motion, private readonly frame: (now: number, dt: number) => void) {
    new IntersectionObserver((entries) => { this.visible = entries.some((e) => e.isIntersecting); this.schedule(); }, { rootMargin: '80px' }).observe(host);
    document.addEventListener('visibilitychange', () => this.schedule());
  }
  schedule(): void {
    cancelAnimationFrame(this.raf);
    this.last = 0;
    if (this.visible && this.playing && this.motion() && !document.hidden) this.raf = requestAnimationFrame((t) => this.tick(t));
  }
  private tick(t: number): void {
    if (!this.playing) return;
    const dt = this.last ? Math.min(0.1, (t - this.last) / 1000) : 0;
    this.last = t;
    this.frame(t / 1000, dt);
    if (this.visible && this.playing && this.motion() && !document.hidden) this.raf = requestAnimationFrame((tt) => this.tick(tt));
  }
  /** Stops the animation (a slider took over); the button then offers Play. */
  pause(): void {
    this.playing = false;
    if (this.button) this.button.textContent = 'Play';
    this.schedule();
  }
  playButton(host: HTMLElement): void {
    this.button = button(host, this.playing ? 'Pause' : 'Play', () => { this.playing = !this.playing; this.button!.textContent = this.playing ? 'Pause' : 'Play'; this.schedule(); });
  }
}

const ease = (u: number) => (u <= 0 ? 0 : u >= 1 ? 1 : 0.5 - 0.5 * Math.cos(Math.PI * u));

// ---------------------------------------------------------------------------------------------------------------
// A. The lattice unfolding along the transport map T = ∇ψ, ψ convex, so no cell ever folds; Gaussians from J JT.
// ---------------------------------------------------------------------------------------------------------------
function demoUnfold(stage: HTMLElement, ctl: HTMLElement, motion: Motion): void {
  const W = 640, H = 424, COLS = 17, ROWS = 11, X0 = 40, Y0 = 36, PW = 560, PH = 330;
  // T(ξ) = ξ + ∇G for a bump G along a ridge and a blob: the gradient of a convex potential (pull < 1), so no folds
  const BX = 468, BY = 112, SB = 46;
  const { T, ridge } = makeMap(0.72, [{ x: BX, y: BY, s: SB, a: 0.5 }]);
  const pts: [number, number][] = [];
  for (let i = 0; i < ROWS; i++) for (let j = 0; j < COLS; j++) pts.push([X0 + (j * PW) / (COLS - 1), Y0 + (i * PH) / (ROWS - 1)]);
  const target = pts.map(([x, y]) => T(x, y));
  const h = PW / (COLS - 1);
  // ellipse axes from the Jacobian (central differences), Σ = (0.6 h)² J Jᵀ
  const ellipses = pts.map(([x, y]) => {
    const e = 0.5;
    const [ax, ay] = T(x + e, y), [bx, by] = T(x - e, y), [cx, cy] = T(x, y + e), [dx, dy] = T(x, y - e);
    return ellipseOf((ax - bx) / (2 * e), (cx - dx) / (2 * e), (ay - by) / (2 * e), (cy - dy) / (2 * e), h);
  });
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'A lattice unfolding onto a ridge and a blob along a transport map' });
  const ridgePath = svgEl('path', { d: Array.from({ length: 41 }, (_, k) => `${k ? 'L' : 'M'}${(X0 + (k * PW) / 40).toFixed(1)} ${ridge(X0 + (k * PW) / 40).toFixed(1)}`).join(' '), class: 'cu2', 'stroke-width': 2, 'stroke-dasharray': '6 5', opacity: 0.8, fill: 'none' });
  const blob = svgEl('circle', { cx: BX, cy: BY, r: SB, class: 'cu2', 'stroke-dasharray': '4 5', opacity: 0.6, fill: 'none' });
  svg.append(ridgePath, blob);
  const lines: SVGPolylineElement[] = [];
  for (let i = 0; i < ROWS; i++) lines.push(svgEl('polyline', { class: 'faint', 'stroke-width': 0.9, fill: 'none' }));
  for (let j = 0; j < COLS; j++) lines.push(svgEl('polyline', { class: 'faint', 'stroke-width': 0.9, fill: 'none' }));
  const ellGroup = svgEl('g', { class: 'cu', 'stroke-width': 1.1, fill: 'none', opacity: 0 });
  const ells = ellipses.map(() => { const e = svgEl('ellipse', {}); ellGroup.append(e); return e; });
  const dots = pts.map(() => svgEl('circle', { r: 2.2, class: 'cuf', stroke: 'none' }));
  svg.append(...lines, ellGroup, ...dots);
  const label = svgEl('text', { x: X0, y: H - 8, class: 'mu' });
  svg.append(label);
  stage.append(svg);
  let t = 0, showG = false;
  const draw = () => {
    const cur = pts.map(([x, y], k) => [x + (target[k][0] - x) * t, y + (target[k][1] - y) * t]);
    for (let i = 0; i < ROWS; i++) lines[i].setAttribute('points', cur.slice(i * COLS, (i + 1) * COLS).map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' '));
    for (let j = 0; j < COLS; j++) lines[ROWS + j].setAttribute('points', Array.from({ length: ROWS }, (_, i) => cur[i * COLS + j]).map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' '));
    cur.forEach((p, k) => { dots[k].setAttribute('cx', p[0].toFixed(1)); dots[k].setAttribute('cy', p[1].toFixed(1)); });
    const ga = showG ? Math.max(0, (t - 0.6) / 0.4) : 0;
    ellGroup.setAttribute('opacity', ga.toFixed(2));
    if (ga > 0) cur.forEach((p, k) => { const e = ellipses[k]; ells[k].setAttribute('cx', p[0].toFixed(1)); ells[k].setAttribute('cy', p[1].toFixed(1)); ells[k].setAttribute('rx', (e.rx * t).toFixed(1)); ells[k].setAttribute('ry', (e.ry * t).toFixed(1)); ells[k].setAttribute('transform', `rotate(${e.ang.toFixed(1)} ${p[0].toFixed(1)} ${p[1].toFixed(1)})`); });
    label.textContent = `t = ${t.toFixed(2)} · (1 − t) ξ + t T(ξ) · dashed: the density ridge and the blob`;
  };
  let phase = 0; // seconds into the loop
  const loop = new Loop(stage, motion, (_now, dt) => {
    phase = (phase + dt) % 7.4;
    t = phase < 3 ? ease(phase / 3) : phase < 4.4 ? 1 : phase < 6.2 ? 1 - ease((phase - 4.4) / 1.8) : 0;
    tSlider.value = t.toFixed(2);
    draw();
  });
  loop.playButton(ctl);
  const tSlider = slider(ctl, 't', 0, 1, 0.01, 0, (v) => v.toFixed(2), (v) => { t = v; loop.pause(); draw(); });
  toggle(ctl, 'Gaussians from the cells', false, (v) => { showG = v; draw(); });
  draw();
}

// ---------------------------------------------------------------------------------------------------------------
// B. Equal shares in one dimension: quantiles against samples of the same density.
// ---------------------------------------------------------------------------------------------------------------
function demoShares(stage: HTMLElement, ctl: HTMLElement, motion: Motion): void {
  const W = 640, H = 322, X0 = 40, X1 = 600;
  const N = 400;
  const dens = (x: number) => 0.25 + 3.2 * Math.exp(-((x - 0.22) ** 2) / 0.0018) + 2.2 * Math.exp(-((x - 0.55) ** 2) / 0.008) + 4 * Math.exp(-((x - 0.8) ** 2) / 0.0006);
  const cdf: number[] = [0];
  for (let i = 1; i <= N; i++) cdf.push(cdf[i - 1] + dens((i - 0.5) / N) / N);
  const total = cdf[N];
  const F = (x: number) => { const u = Math.max(0, Math.min(N - 1e-9, x * N)); const i = Math.floor(u); return (cdf[i] + (cdf[i + 1] - cdf[i]) * (u - i)) / total; };
  const Finv = (p: number) => { let lo = 0, hi = 1; for (let k = 0; k < 40; k++) { const m = (lo + hi) / 2; if (F(m) < p) lo = m; else hi = m; } return (lo + hi) / 2; };
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'A density with its quantile positions and random samples, and the mass of each cell' });
  const px = (x: number) => X0 + x * (X1 - X0);
  const densPath = svgEl('path', { class: 'cu', 'stroke-width': 1.6, fill: 'none', d: Array.from({ length: 161 }, (_, k) => { const x = k / 160; return `${k ? 'L' : 'M'}${px(x).toFixed(1)} ${(96 - (dens(x) / 4.4) * 86).toFixed(1)}`; }).join(' ') });
  svg.append(densPath);
  const axis = (y: number, name: string) => { svg.append(svgEl('line', { x1: X0, y1: y, x2: X1, y2: y, class: 'faint' })); const tx = svgEl('text', { x: X0, y: y - 18, class: 'cap' }); tx.textContent = name; svg.append(tx); };
  axis(134, 'OT: the quantiles');
  axis(230, 'samples of the same density');
  const gOT = svgEl('g'), gS = svgEl('g');
  svg.append(gOT, gS);
  const readOT = svgEl('text', { x: X1, y: 116, 'text-anchor': 'end', class: 'mu cut' }), readS = svgEl('text', { x: X1, y: 212, 'text-anchor': 'end', class: 'mu' });
  svg.append(readOT, readS);
  const note = svgEl('text', { x: X0, y: H - 6, class: 'mu' });
  note.textContent = 'ticks: positions · bars: the density mass of each cell × K (1 = its fair share)';
  svg.append(note);
  stage.append(svg);
  let K = 24;
  const cells = (xs: number[]) => xs.map((x, i) => { const a = i ? (xs[i - 1] + x) / 2 : 0, b = i < xs.length - 1 ? (x + xs[i + 1]) / 2 : 1; return (F(b) - F(a)) * xs.length; });
  const cv = (m: number[]) => { const mean = m.reduce((s, v) => s + v, 0) / m.length; return Math.sqrt(m.reduce((s, v) => s + (v - mean) ** 2, 0) / m.length) / mean; };
  const drawRow = (g: SVGGElement, xs: number[], y: number, cls: string) => {
    g.replaceChildren();
    const m = cells(xs);
    xs.forEach((x, i) => {
      g.append(svgEl('line', { x1: px(x), y1: y - 10, x2: px(x), y2: y + 2, class: cls, 'stroke-width': 1.5 }));
      const hgt = Math.min(54, m[i] * 20);
      g.append(svgEl('rect', { x: px(x) - 2.2, y: y + 4, width: 4.4, height: hgt, class: cls === 'cu' ? 'cuf' : 'mutedf', stroke: 'none', opacity: 0.85 }));
    });
    g.append(svgEl('line', { x1: X0, y1: y + 24, x2: X1, y2: y + 24, class: 'faint', 'stroke-dasharray': '3 4' }));
    return cv(m);
  };
  let samples: number[] = [];
  const redraw = (resample: boolean) => {
    const ot = Array.from({ length: K }, (_, i) => Finv((i + 0.5) / K));
    if (resample || samples.length !== K) samples = Array.from({ length: K }, () => Finv(Math.random())).sort((a, b) => a - b);
    readOT.textContent = `CV ${drawRow(gOT, ot, 134, 'cu').toFixed(2)}`;
    readS.textContent = `CV ${drawRow(gS, samples, 230, 'ink').toFixed(2)}`;
  };
  let clock = 0;
  const loop = new Loop(stage, motion, (_now, dt) => { clock += dt; if (clock > 2.4) { clock = 0; redraw(true); } });
  loop.playButton(ctl);
  slider(ctl, 'K', 8, 48, 1, K, (v) => String(v), (v) => { K = v; redraw(true); });
  button(ctl, 'Draw again', () => redraw(true));
  redraw(true);
}

// ---------------------------------------------------------------------------------------------------------------
// C. Additive against weight-normalised blending of two overlapping Gaussians.
// ---------------------------------------------------------------------------------------------------------------
function demoBlend(stage: HTMLElement, ctl: HTMLElement, motion: Motion): void {
  const W = 640, H = 270, X0 = 40, X1 = 600;
  const canvas = el('canvas', { width: '1280', height: '540', 'aria-label': 'Two overlapping Gaussians blended additively and with weight normalisation' });
  canvas.style.width = '100%';
  canvas.style.aspectRatio = `${W} / ${H}`;
  stage.append(canvas);
  const g = canvas.getContext('2d')!;
  let gap = 0.34, sigma = 0.09;
  const c1 = [217, 162, 95], c2 = [58, 96, 140];
  const draw = () => {
    const sc = 2;
    g.setTransform(sc, 0, 0, sc, 0, 0);
    g.clearRect(0, 0, W, H);
    const ink = cssVar('--text', '#24261f'), faint = cssVar('--faint', '#a3a59d'), muted = cssVar('--muted', '#6f726a');
    const strip = (y: number, mode: 'add' | 'norm', title: string) => {
      for (let i = 0; i < 560; i++) {
        const x = i / 560;
        const a1 = Math.exp(-((x - 0.5 + gap / 2) ** 2) / (2 * sigma * sigma)), a2 = Math.exp(-((x - 0.5 - gap / 2) ** 2) / (2 * sigma * sigma));
        let r, gg, b;
        if (mode === 'add') { r = Math.min(255, a1 * c1[0] + a2 * c2[0]); gg = Math.min(255, a1 * c1[1] + a2 * c2[1]); b = Math.min(255, a1 * c1[2] + a2 * c2[2]); const bg = 1 - Math.min(1, a1 + a2); r += bg * 235; gg += bg * 235; b += bg * 232; }
        else { const d = a1 + a2 + 1e-4; const cov = Math.min(1, a1 + a2); r = (a1 * c1[0] + a2 * c2[0]) / d; gg = (a1 * c1[1] + a2 * c2[1]) / d; b = (a1 * c1[2] + a2 * c2[2]) / d; r = r * cov + 235 * (1 - cov); gg = gg * cov + 235 * (1 - cov); b = b * cov + 232 * (1 - cov); }
        g.fillStyle = `rgb(${r | 0},${gg | 0},${b | 0})`;
        g.fillRect(X0 + i, y, 1.2, 34);
      }
      g.strokeStyle = faint; g.lineWidth = 1; g.strokeRect(X0 + 0.5, y + 0.5, X1 - X0, 34);
      g.fillStyle = ink; g.font = '700 11px ui-monospace, Consolas, monospace'; g.fillText(title, X0, y - 6);
    };
    strip(26, 'add', 'ADDITIVE SUM, CLAMPED (GAUSSIANIMAGE)');
    strip(96, 'norm', 'WEIGHT-NORMALISED BLEND (OTSR, EQ. 1)');
    // the weights and their sum
    const y0 = 250, yh = 80;
    g.strokeStyle = faint; g.beginPath(); g.moveTo(X0, y0 + 0.5); g.lineTo(X1, y0 + 0.5); g.stroke();
    g.setLineDash([3, 4]); g.beginPath(); g.moveTo(X0, y0 - yh + 0.5); g.lineTo(X1, y0 - yh + 0.5); g.stroke(); g.setLineDash([]);
    g.fillStyle = muted; g.font = '500 10px ui-monospace, Consolas, monospace'; g.fillText('1', X0 - 10, y0 - yh + 4); g.fillText('0', X0 - 10, y0 + 4);
    const curve = (f: (x: number) => number, color: string, width: number) => { g.strokeStyle = color; g.lineWidth = width; g.beginPath(); for (let i = 0; i <= 280; i++) { const x = i / 280; const v = Math.min(2, f(x)); const px = X0 + x * (X1 - X0), py = y0 - v * yh; if (i) g.lineTo(px, py); else g.moveTo(px, py); } g.stroke(); };
    const a1 = (x: number) => Math.exp(-((x - 0.5 + gap / 2) ** 2) / (2 * sigma * sigma)), a2 = (x: number) => Math.exp(-((x - 0.5 - gap / 2) ** 2) / (2 * sigma * sigma));
    curve((x) => a1(x) + a2(x), ink, 1.3); curve(a1, `rgb(${c1.join(',')})`, 2.2); curve(a2, `rgb(${c2.join(',')})`, 2.2);
    g.fillStyle = muted; g.fillText('α₁, α₂ and their sum: above 1 the additive sum saturates; the normalised blend is a convex combination everywhere', X0, 266);
  };
  let clock = 0;
  const loop = new Loop(stage, motion, (_now, dt) => { clock += dt; gap = 0.34 + 0.22 * Math.sin(clock * 0.9); gapS.value = gap.toFixed(2); draw(); });
  loop.playButton(ctl);
  const gapS = slider(ctl, 'distance', 0.05, 0.6, 0.01, gap, (v) => v.toFixed(2), (v) => { gap = v; loop.pause(); draw(); });
  slider(ctl, 'σ', 0.03, 0.18, 0.005, sigma, (v) => v.toFixed(3), (v) => { sigma = v; draw(); });
  draw();
}

// ---------------------------------------------------------------------------------------------------------------
// D. The edge pass: a soft ramp at S× re-thresholded to a transition of edge_px output pixels.
// ---------------------------------------------------------------------------------------------------------------
function demoEdge(stage: HTMLElement, ctl: HTMLElement, motion: Motion): void {
  const W = 640, H = 250, X0 = 56, X1 = 620, Y0 = 196, Y1 = 36;
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'The soft ramp of the Gaussian render across an edge and the transition after the edge pass' });
  const px = (u: number) => X0 + ((u + 12) / 24) * (X1 - X0), py = (v: number) => Y0 - v * (Y0 - Y1);
  svg.append(svgEl('line', { x1: X0, y1: Y0, x2: X1, y2: Y0, class: 'faint' }), svgEl('line', { x1: X0, y1: Y0, x2: X0, y2: Y1, class: 'faint' }), svgEl('line', { x1: X0, y1: py(0.5), x2: X1, y2: py(0.5), class: 'faint', 'stroke-dasharray': '3 5' }));
  for (const u of [-8, -4, 0, 4, 8]) { const tx = svgEl('text', { x: px(u), y: Y0 + 18, 'text-anchor': 'middle', class: 'mu' }); tx.textContent = `${u > 0 ? '+' : ''}${u} px`; svg.append(tx); }
  const soft = svgEl('polyline', { class: 'faint', 'stroke-width': 2, fill: 'none' }), sharp = svgEl('polyline', { class: 'cu', 'stroke-width': 2.2, fill: 'none' });
  const bracket = svgEl('path', { class: 'cu', 'stroke-width': 1, fill: 'none' });
  const text = svgEl('text', { x: X1, y: Y1 + 4, 'text-anchor': 'end', class: 'mu cut' });
  svg.append(soft, sharp, bracket, text);
  stage.append(svg);
  let S = 8, edgePx = 1.4;
  const draw = () => {
    const scale = 0.2 * S;
    soft.setAttribute('points', Array.from({ length: 241 }, (_, i) => { const u = -12 + i / 10; return `${px(u).toFixed(1)},${py(1 / (1 + Math.exp(-u / scale))).toFixed(1)}`; }).join(' '));
    const sm = (t: number) => { t = Math.max(0, Math.min(1, t)); return t * t * (3 - 2 * t); };
    sharp.setAttribute('points', Array.from({ length: 481 }, (_, i) => { const u = -12 + i / 20; return `${px(u).toFixed(1)},${py(sm((u + edgePx / 2) / edgePx)).toFixed(1)}`; }).join(' '));
    const yb = py(0.5) + 26;
    bracket.setAttribute('d', `M${px(-edgePx / 2).toFixed(1)} ${yb - 5} v5 H${px(edgePx / 2).toFixed(1)} v-5`);
    text.textContent = `S = ${S}×: the 0.3-px Gaussian becomes a ${(0.3 * S).toFixed(1)}-px ramp; after the pass ${edgePx.toFixed(1)} px`;
  };
  let clock = 0;
  const loop = new Loop(stage, motion, (_now, dt) => { clock += dt; S = Math.round(2 + 7 * (1 - Math.cos(clock * 0.7)) ); sS.value = String(S); draw(); });
  loop.playButton(ctl);
  const sS = slider(ctl, 'S', 2, 16, 1, S, (v) => `${v}×`, (v) => { S = v; loop.pause(); draw(); });
  slider(ctl, 'edge_px', 0.4, 3, 0.1, edgePx, (v) => `${v.toFixed(1)} px`, (v) => { edgePx = v; draw(); });
  draw();
}

// ---------------------------------------------------------------------------------------------------------------
// E. A handful of Gaussians rendered at any scale, next to the pixels they were fitted to.
// ---------------------------------------------------------------------------------------------------------------
function demoAnyRes(stage: HTMLElement, ctl: HTMLElement, motion: Motion): void {
  const BW = 40, BH = 26;
  // the model: centre (px), scales (px), rotation, colour; a ring of elongated Gaussians, a stroke and a dot
  type G = { x: number; y: number; sx: number; sy: number; th: number; c: [number, number, number]; w: number };
  const gs: G[] = [];
  for (let k = 0; k < 10; k++) { const a = (k / 10) * 2 * Math.PI; gs.push({ x: 14 + 7.2 * Math.cos(a), y: 13 + 7.2 * Math.sin(a), sx: 2.6, sy: 0.55, th: a + Math.PI / 2, c: [24, 26, 24], w: 1 }); }
  gs.push({ x: 14, y: 13, sx: 1.1, sy: 1.1, th: 0, c: [184, 121, 60], w: 1 });
  for (let k = 0; k < 4; k++) gs.push({ x: 26 + k * 3.4, y: 20 - k * 3.6, sx: 2.4, sy: 0.5, th: -0.82, c: [184, 121, 60], w: 1 });
  gs.push({ x: 34, y: 8, sx: 1.3, sy: 1.3, th: 0, c: [58, 96, 140], w: 1 });
  gs.push({ x: 8, y: 5, sx: 3, sy: 0.45, th: 0.1, c: [58, 96, 140], w: 1 });
  const bg = [235, 235, 232];
  const render = (S: number): ImageData => {
    const w = BW * S, hh = BH * S;
    const out = new ImageData(w, hh);
    const d = out.data;
    for (let j = 0; j < hh; j++) for (let i = 0; i < w; i++) {
      const x = (i + 0.5) / S, y = (j + 0.5) / S;
      let r = 0, g2 = 0, b = 0, den = 1e-4;
      for (const q of gs) {
        const dx = x - q.x, dy = y - q.y, c = Math.cos(q.th), s = Math.sin(q.th);
        const u = (c * dx + s * dy) / q.sx, v = (-s * dx + c * dy) / q.sy;
        const qq = 0.5 * (u * u + v * v);
        if (qq > 8) continue;
        const a = (Math.exp(-qq) - Math.exp(-8)) * q.w;
        r += a * q.c[0]; g2 += a * q.c[1]; b += a * q.c[2]; den += a;
      }
      const cov = Math.min(1, den);
      const k = 4 * (j * w + i);
      d[k] = (r / den) * cov + bg[0] * (1 - cov); d[k + 1] = (g2 / den) * cov + bg[1] * (1 - cov); d[k + 2] = (b / den) * cov + bg[2] * (1 - cov); d[k + 3] = 255;
    }
    return out;
  };
  const base = render(1);
  // the distance-field edge pass of the app, on the CPU: window min / max by luma, a locally normalised field, a
  // smoothstep of edge_px output pixels, blended in where there is contrast
  const edgePass = (src: ImageData, S: number, edgePx = 1.4): ImageData => {
    const w = src.width, hh = src.height, d = src.data, n = w * hh;
    const Y = new Float32Array(n);
    for (let i = 0; i < n; i++) Y[i] = (0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2]) / 255;
    const r = Math.max(1, Math.round(0.75 * S));
    const hmin = new Int32Array(n), hmax = new Int32Array(n), vmin = new Int32Array(n), vmax = new Int32Array(n);
    for (let y = 0; y < hh; y++) for (let x = 0; x < w; x++) { let mi = y * w + x, ma = mi; for (let k = -r; k <= r; k++) { const xx = x + k; if (xx < 0 || xx >= w) continue; const q = y * w + xx; if (Y[q] < Y[mi]) mi = q; if (Y[q] > Y[ma]) ma = q; } hmin[y * w + x] = mi; hmax[y * w + x] = ma; }
    for (let y = 0; y < hh; y++) for (let x = 0; x < w; x++) { let mi = hmin[y * w + x], ma = hmax[y * w + x]; for (let k = -r; k <= r; k++) { const yy = y + k; if (yy < 0 || yy >= hh) continue; const a = hmin[yy * w + x], b = hmax[yy * w + x]; if (Y[a] < Y[mi]) mi = a; if (Y[b] > Y[ma]) ma = b; } vmin[y * w + x] = mi; vmax[y * w + x] = ma; }
    const t = new Float32Array(n), range = new Float32Array(n);
    for (let i = 0; i < n; i++) { const lo = Y[vmin[i]], hi = Y[vmax[i]]; range[i] = hi - lo; t[i] = range[i] > 1e-6 ? (Y[i] - lo) / range[i] : 0.5; }
    const sm = (a: number, b: number, x: number) => { const u = Math.max(0, Math.min(1, (x - a) / Math.max(1e-6, b - a))); return u * u * (3 - 2 * u); };
    const out = new ImageData(w, hh), o = out.data;
    for (let y = 0; y < hh; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const gx = (t[y * w + Math.min(w - 1, x + 1)] - t[y * w + Math.max(0, x - 1)]) / 2, gy = (t[Math.min(hh - 1, y + 1) * w + x] - t[Math.max(0, y - 1) * w + x]) / 2;
      const wd = 0.5 * edgePx * Math.hypot(gx, gy);
      const alpha = sm(0.5 - wd, 0.5 + wd, t[i]), gate = sm(0.08, 0.2, range[i]);
      const D = vmin[i], B = vmax[i];
      for (let c = 0; c < 3; c++) { const sharp = d[4 * D + c] + (d[4 * B + c] - d[4 * D + c]) * alpha; o[4 * i + c] = d[4 * i + c] + gate * (sharp - d[4 * i + c]); }
      o[4 * i + 3] = 255;
    }
    return out;
  };
  let edge = false;
  const left = el('canvas', { 'aria-label': 'The pixels, magnified' }), right = el('canvas', { 'aria-label': 'The Gaussians rendered at the magnification' });
  const wrap = el('div', { class: 'twoup' }, el('div', {}, left, el('div', { class: 'mu' }, 'THE 40 × 26 PIXELS, MAGNIFIED')), el('div', {}, right, el('div', { class: 'mu' }, 'THE SAME GAUSSIANS RENDERED AT S×')));
  stage.append(wrap);
  let S = 4;
  const draw = () => {
    const w = BW * S, hh = BH * S;
    left.width = right.width = w; left.height = right.height = hh;
    const gl = left.getContext('2d')!, gr = right.getContext('2d')!;
    gl.imageSmoothingEnabled = false;
    const tmp = el('canvas', { width: String(BW), height: String(BH) });
    tmp.getContext('2d')!.putImageData(base, 0, 0);
    gl.drawImage(tmp, 0, 0, w, hh);
    const img = render(S);
    gr.putImageData(edge && S >= 2 ? edgePass(img, S) : img, 0, 0);
    sLabel.textContent = `S = ${S}×${edge ? (S >= 2 ? ' · edge pass on' : ' · edge pass from 2×') : ''}`;
  };
  let clock = 0, last = 0;
  const loop = new Loop(stage, motion, (_now, dt) => { clock += dt; const s = Math.round(1 + 7 * (0.5 - 0.5 * Math.cos(clock * 0.5))); if (s !== last) { last = s; S = s; sS.value = String(S); draw(); } });
  loop.playButton(ctl);
  const sS = slider(ctl, 'S', 1, 8, 1, S, (v) => `${v}×`, (v) => { S = v; loop.pause(); draw(); });
  toggle(ctl, 'edge pass (1.4 px)', false, (v) => { edge = v; draw(); });
  const sLabel = el('span', { class: 'mu' });
  ctl.append(sLabel);
  draw();
}

// ---------------------------------------------------------------------------------------------------------------
// F. Sinkhorn with ε-scaling on a 48-bin problem: the transport plan sharpens into the map as ε shrinks.
// ---------------------------------------------------------------------------------------------------------------
function demoSinkhorn(stage: HTMLElement, ctl: HTMLElement, motion: Motion): void {
  const n = 48;
  const mu = new Float64Array(n).fill(1 / n);
  const nu = new Float64Array(n);
  for (let i = 0; i < n; i++) { const x = (i + 0.5) / n; nu[i] = 0.2 + 3 * Math.exp(-((x - 0.25) ** 2) / 0.003) + 2 * Math.exp(-((x - 0.62) ** 2) / 0.012) + 3.5 * Math.exp(-((x - 0.86) ** 2) / 0.0012); }
  const tot = nu.reduce((s, v) => s + v, 0);
  for (let i = 0; i < n; i++) nu[i] /= tot;
  const C = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) C[i * n + j] = ((i - j) / n) ** 2;
  // log-domain Sinkhorn at a given ε, warm-started from the previous potentials
  const f = new Float64Array(n), g = new Float64Array(n);
  const lse = (row: (j: number) => number) => { let m = -Infinity; for (let j = 0; j < n; j++) m = Math.max(m, row(j)); let s = 0; for (let j = 0; j < n; j++) s += Math.exp(row(j) - m); return m + Math.log(s); };
  const solve = (eps: number, iters: number) => {
    for (let it = 0; it < iters; it++) {
      for (let i = 0; i < n; i++) f[i] = eps * Math.log(mu[i]) - eps * lse((j) => (g[j] - C[i * n + j]) / eps);
      for (let j = 0; j < n; j++) g[j] = eps * Math.log(nu[j]) - eps * lse((i) => (f[i] - C[i * n + j]) / eps);
    }
  };
  const plan = (eps: number) => { const P = new Float64Array(n * n); for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) P[i * n + j] = Math.exp((f[i] + g[j] - C[i * n + j]) / eps); return P; };
  const W = 640, H = 330, X0 = 70, Y0 = 20, SZ = 230;
  const canvas = el('canvas', { width: '1280', height: '660', 'aria-label': 'The entropic transport plan between a uniform and a peaked density, sharpening as the regularisation shrinks' });
  canvas.style.width = '100%';
  canvas.style.aspectRatio = `${W} / ${H}`;
  stage.append(canvas);
  const ctx = canvas.getContext('2d')!;
  const stages = Array.from({ length: 14 }, (_, k) => 0.5 * 0.6 ** k);
  let stageIdx = 0, epsNow = stages[0];
  let lastSolved = -1;
  const draw = () => {
    if (lastSolved !== stageIdx) { for (let k = Math.max(0, lastSolved + 1); k <= stageIdx; k++) solve(stages[k], k === stageIdx ? 30 : 6); lastSolved = stageIdx; epsNow = stages[stageIdx]; }
    const P = plan(epsNow);
    const sc = 2;
    ctx.setTransform(sc, 0, 0, sc, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const cell = SZ / n;
    let pmax = 0; for (let k = 0; k < n * n; k++) pmax = Math.max(pmax, P[k]);
    const cu = cssVar('--copper', '#b8793c');
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const v = P[i * n + j] / pmax; if (v < 0.02) continue; ctx.globalAlpha = Math.min(1, v ** 0.6); ctx.fillStyle = cu; ctx.fillRect(X0 + i * cell, Y0 + j * cell, cell + 0.3, cell + 0.3); }
    ctx.globalAlpha = 1;
    const faint = cssVar('--faint', '#a3a59d'), muted = cssVar('--muted', '#6f726a'), ink = cssVar('--text', '#24261f');
    ctx.strokeStyle = faint; ctx.lineWidth = 1; ctx.strokeRect(X0 + 0.5, Y0 + 0.5, SZ, SZ);
    // marginals: uniform (top, along x) and the target (left, along y)
    ctx.fillStyle = muted;
    for (let j = 0; j < n; j++) { const w = (nu[j] / Math.max(...nu)) * 40; ctx.fillRect(X0 - 8 - w, Y0 + j * cell, w, cell - 0.4); }
    for (let i = 0; i < n; i++) ctx.fillRect(X0 + i * cell, Y0 + SZ + 8, cell - 0.4, (mu[i] * n) * 10);
    // the barycentric map
    ctx.strokeStyle = ink; ctx.lineWidth = 1.6; ctx.beginPath();
    for (let i = 0; i < n; i++) { let s = 0, m = 0; for (let j = 0; j < n; j++) { s += P[i * n + j] * (j + 0.5); m += P[i * n + j]; } const y = Y0 + (s / m) * cell, x = X0 + (i + 0.5) * cell; if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
    ctx.stroke();
    ctx.fillStyle = muted; ctx.font = '500 10px ui-monospace, Consolas, monospace';
    ctx.fillText('x: the uniform lattice →', X0, Y0 + SZ + 36);
    ctx.save(); ctx.translate(14, Y0 + SZ); ctx.rotate(-Math.PI / 2); ctx.fillText('y: the target density ρ →', 0, 0); ctx.restore();
    ctx.fillStyle = ink; ctx.font = '700 11px ui-monospace, Consolas, monospace';
    ctx.fillText(`STAGE ${stageIdx + 1} / ${stages.length} · ε = ${epsNow.toExponential(1)}`, X0 + SZ + 24, Y0 + 16);
    ctx.fillStyle = muted; ctx.font = '500 10.5px ui-monospace, Consolas, monospace';
    const lines = ['copper: the plan π(x, y),', 'brighter = more mass', '', 'line: the barycentric map', 'T(x) = E[y | x]', '', 'as ε shrinks the plan', 'concentrates on the map,', 'monotone: no crossings'];
    lines.forEach((l, k) => ctx.fillText(l, X0 + SZ + 24, Y0 + 40 + k * 15));
  };
  let clock = 0;
  const loop = new Loop(stage, motion, (_now, dt) => { clock += dt; if (clock > 0.55) { clock = 0; stageIdx = (stageIdx + 1) % stages.length; if (stageIdx === 0) { f.fill(0); g.fill(0); lastSolved = -1; } sStage.value = String(stageIdx); draw(); } });
  loop.playButton(ctl);
  const sStage = slider(ctl, 'stage', 0, stages.length - 1, 1, 0, (v) => `${v + 1}`, (v) => { if (v < stageIdx) { f.fill(0); g.fill(0); lastSolved = -1; } stageIdx = v; loop.pause(); draw(); });
  draw();
}

// ---------------------------------------------------------------------------------------------------------------
// Shared: the synthetic transport map of the lattice demos (the gradient of a convex potential: a ridge and blobs).
// ---------------------------------------------------------------------------------------------------------------
interface Bump { x: number; y: number; s: number; a: number }

function makeMap(ridgeA: number, blobs: Bump[]) {
  const X0 = 40, PW = 560, S = 38;
  const ridge = (x: number) => 215 + 62 * Math.sin(((x - X0) / PW) * 1.25 * Math.PI - 0.35);
  const dridge = (x: number) => 62 * Math.cos(((x - X0) / PW) * 1.25 * Math.PI - 0.35) * (1.25 * Math.PI / PW);
  const T = (x: number, y: number): [number, number] => {
    const fp = dridge(x);
    const den = Math.sqrt(1 + fp * fp);
    const d = (y - ridge(x)) / den;
    const g = Math.exp(-(d * d) / (2 * S * S));
    let ox = x - ridgeA * d * g * (-fp / den), oy = y - ridgeA * d * g * (1 / den);
    for (const b of blobs) { const rx = x - b.x, ry = y - b.y; const gb = Math.exp(-(rx * rx + ry * ry) / (2 * b.s * b.s)); ox -= b.a * rx * gb; oy -= b.a * ry * gb; }
    return [ox, oy];
  };
  return { T, ridge };
}

/** Ellipse (rx, ry, angle in degrees) of a Gaussian with covariance (0.6 h)² J Jᵀ for a 2 × 2 Jacobian. */
function ellipseOf(j11: number, j12: number, j21: number, j22: number, h: number): { rx: number; ry: number; ang: number } {
  const s = 0.6 * h;
  const m11 = s * s * (j11 * j11 + j12 * j12), m12 = s * s * (j11 * j21 + j12 * j22), m22 = s * s * (j21 * j21 + j22 * j22);
  const tr = m11 + m22, det = m11 * m22 - m12 * m12;
  const l1 = tr / 2 + Math.sqrt(Math.max(0, (tr * tr) / 4 - det)), l2 = Math.max(1e-6, tr - l1);
  const ang = Math.atan2(l1 - m11, m12 || 1e-9);
  return { rx: Math.sqrt(l1), ry: Math.sqrt(l2), ang: (ang * 180) / Math.PI };
}

// ---------------------------------------------------------------------------------------------------------------
// G. Error-driven re-layout: the lattice of the first map moves on to a second density that has more mass where the
// error of a short fit remains.
// ---------------------------------------------------------------------------------------------------------------
function demoRelayout(stage: HTMLElement, ctl: HTMLElement, motion: Motion): void {
  const W = 640, H = 424, COLS = 17, ROWS = 11, X0 = 40, Y0 = 36, PW = 560, PH = 330;
  const blob1: Bump = { x: 468, y: 112, s: 46, a: 0.5 };
  const err: Bump = { x: 190, y: 120, s: 52, a: 0.48 };
  const m1 = makeMap(0.72, [blob1]);
  const m2 = makeMap(0.72, [blob1, err]);
  const pts: [number, number][] = [];
  for (let i = 0; i < ROWS; i++) for (let j = 0; j < COLS; j++) pts.push([X0 + (j * PW) / (COLS - 1), Y0 + (i * PH) / (ROWS - 1)]);
  const A = pts.map(([x, y]) => m1.T(x, y)), B = pts.map(([x, y]) => m2.T(x, y));
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'A lattice moving from the layout of the importance density to the layout of the error-driven density' });
  svg.append(svgEl('path', { d: Array.from({ length: 41 }, (_, k) => `${k ? 'L' : 'M'}${(X0 + (k * PW) / 40).toFixed(1)} ${m1.ridge(X0 + (k * PW) / 40).toFixed(1)}`).join(' '), class: 'cu2', 'stroke-width': 2, 'stroke-dasharray': '6 5', opacity: 0.8, fill: 'none' }));
  svg.append(svgEl('circle', { cx: blob1.x, cy: blob1.y, r: blob1.s, class: 'cu2', 'stroke-dasharray': '4 5', opacity: 0.6, fill: 'none' }));
  const errGlow = svgEl('circle', { cx: err.x, cy: err.y, r: err.s * 1.3, class: 'cuf', stroke: 'none', opacity: 0 });
  const errLabel = svgEl('text', { x: err.x, y: err.y + 4, 'text-anchor': 'middle', class: 'cap cut', opacity: 0 });
  errLabel.textContent = 'error after 300 steps';
  svg.append(errGlow, errLabel);
  const lines: SVGPolylineElement[] = [];
  for (let k = 0; k < ROWS + COLS; k++) lines.push(svgEl('polyline', { class: 'faint', 'stroke-width': 0.9, fill: 'none' }));
  const dots = pts.map(() => svgEl('circle', { r: 2.2, class: 'cuf', stroke: 'none' }));
  svg.append(...lines, ...dots);
  const label = svgEl('text', { x: X0, y: H - 8, class: 'mu' });
  svg.append(label);
  stage.append(svg);
  let u = 0, glow = 0;
  const draw = () => {
    const cur = A.map((p, k) => [p[0] + (B[k][0] - p[0]) * u, p[1] + (B[k][1] - p[1]) * u]);
    for (let i = 0; i < ROWS; i++) lines[i].setAttribute('points', cur.slice(i * COLS, (i + 1) * COLS).map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' '));
    for (let j = 0; j < COLS; j++) lines[ROWS + j].setAttribute('points', Array.from({ length: ROWS }, (_, i) => cur[i * COLS + j]).map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' '));
    cur.forEach((p, k) => { dots[k].setAttribute('cx', p[0].toFixed(1)); dots[k].setAttribute('cy', p[1].toFixed(1)); });
    errGlow.setAttribute('opacity', (0.22 * glow).toFixed(2));
    errLabel.setAttribute('opacity', glow.toFixed(2));
    label.textContent = u < 0.02 ? 'the first layout: the lattice of the importance density ρ' : u > 0.98 ? 'the second layout: more cells where the error remained, fewer in the flat regions' : `moving to the second transport · ${Math.round(u * 100)} %`;
  };
  let phase = 0;
  const loop = new Loop(stage, motion, (_now, dt) => {
    phase = (phase + dt) % 8;
    glow = phase < 1.2 ? 0 : phase < 2.2 ? ease(phase - 1.2) : phase < 6.2 ? 1 : 1 - ease((phase - 6.2) / 0.8);
    u = phase < 2.4 ? 0 : phase < 4.2 ? ease((phase - 2.4) / 1.8) : phase < 6.2 ? 1 : 1 - ease((phase - 6.2) / 1.2);
    uSlider.value = u.toFixed(2);
    draw();
  });
  loop.playButton(ctl);
  const uSlider = slider(ctl, 'layout 1 → 2', 0, 1, 0.01, 0, (v) => v.toFixed(2), (v) => { u = v; glow = v > 0 ? 1 : 0; loop.pause(); draw(); });
  draw();
}

// ---------------------------------------------------------------------------------------------------------------
// H. Equal shares in two dimensions: a real entropic transport of a uniform lattice onto a density, against samples
// of the same density; every point's nearest-centre cell coloured by the share of the density it carries.
// ---------------------------------------------------------------------------------------------------------------
function demoShares2d(stage: HTMLElement, ctl: HTMLElement, motion: Motion): void {
  const NX = 36, NY = 24; // transport grid
  const LX = 12, LY = 8; // lattice
  const K = LX * LY;
  // the density on the grid (box [0,1.5] x [0,1])
  const rho = new Float64Array(NX * NY);
  const dens = (x: number, y: number) => {
    const ry = 0.55 + 0.17 * Math.sin(x * 2.6 - 0.4);
    const d = y - ry;
    return 0.18 + 2.6 * Math.exp(-(d * d) / (2 * 0.045 * 0.045)) + 1.6 * Math.exp(-((x - 1.17) ** 2 + (y - 0.27) ** 2) / (2 * 0.09 * 0.09));
  };
  let tot = 0;
  for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) { const v = dens(((i + 0.5) / NX) * 1.5, (j + 0.5) / NY); rho[j * NX + i] = v; tot += v; }
  for (let k = 0; k < NX * NY; k++) rho[k] /= tot;
  // separable log-domain Sinkhorn: uniform -> rho, eps-scaling
  const xs = Array.from({ length: NX }, (_, i) => ((i + 0.5) / NX) * 1.5), ys = Array.from({ length: NY }, (_, j) => (j + 0.5) / NY);
  const f = new Float64Array(NX * NY), g = new Float64Array(NX * NY);
  const logmu = Math.log(1 / (NX * NY));
  const tmp = new Float64Array(NX * NY);
  const lseRow = (src: Float64Array, dst: Float64Array, eps: number, sign: number) => {
    // dst[j*NX+i] = eps * log sum_{i'} exp((src[j*NX+i'] - (x_i - x_i')²)/eps) then along y
    const t2 = new Float64Array(NX * NY);
    for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) {
      let m = -Infinity; for (let i2 = 0; i2 < NX; i2++) m = Math.max(m, (src[j * NX + i2] - (xs[i] - xs[i2]) ** 2) / eps);
      let s = 0; for (let i2 = 0; i2 < NX; i2++) s += Math.exp((src[j * NX + i2] - (xs[i] - xs[i2]) ** 2) / eps - m);
      t2[j * NX + i] = eps * (m + Math.log(s));
    }
    for (let i = 0; i < NX; i++) for (let j = 0; j < NY; j++) {
      let m = -Infinity; for (let j2 = 0; j2 < NY; j2++) m = Math.max(m, (t2[j2 * NX + i] - (ys[j] - ys[j2]) ** 2) / eps);
      let s = 0; for (let j2 = 0; j2 < NY; j2++) s += Math.exp((t2[j2 * NX + i] - (ys[j] - ys[j2]) ** 2) / eps - m);
      dst[j * NX + i] = sign * eps * (m + Math.log(s));
    }
  };
  let eps = 1.5 * 1.5;
  const epsEnd = (1.5 / NX * 1.5) ** 2;
  while (eps > epsEnd) {
    for (let it = 0; it < 6; it++) {
      lseRow(g, tmp, eps, 1); for (let k = 0; k < NX * NY; k++) f[k] = eps * logmu - tmp[k];
      lseRow(f, tmp, eps, 1); for (let k = 0; k < NX * NY; k++) g[k] = eps * Math.log(rho[k]) - tmp[k];
    }
    eps *= 0.6;
  }
  eps = epsEnd;
  for (let it = 0; it < 30; it++) {
    lseRow(g, tmp, eps, 1); for (let k = 0; k < NX * NY; k++) f[k] = eps * logmu - tmp[k];
    lseRow(f, tmp, eps, 1); for (let k = 0; k < NX * NY; k++) g[k] = eps * Math.log(rho[k]) - tmp[k];
  }
  // barycentric projection on the grid (dense, once)
  const TX = new Float64Array(NX * NY), TY = new Float64Array(NX * NY);
  for (let a = 0; a < NX * NY; a++) {
    const xa = xs[a % NX], ya = ys[(a / NX) | 0];
    let m = -Infinity; const vals = new Float64Array(NX * NY);
    for (let b = 0; b < NX * NY; b++) { const xb = xs[b % NX], yb = ys[(b / NX) | 0]; vals[b] = (f[a] + g[b] - (xa - xb) ** 2 - (ya - yb) ** 2) / eps; if (vals[b] > m) m = vals[b]; }
    let s = 0, sx = 0, sy = 0;
    for (let b = 0; b < NX * NY; b++) { const p = Math.exp(vals[b] - m); s += p; sx += p * xs[b % NX]; sy += p * ys[(b / NX) | 0]; }
    TX[a] = sx / s; TY[a] = sy / s;
  }
  const mapAt = (x: number, y: number): [number, number] => {
    const u = Math.max(0, Math.min(NX - 1.001, (x / 1.5) * NX - 0.5)), v = Math.max(0, Math.min(NY - 1.001, y * NY - 0.5));
    const i = Math.floor(u), j = Math.floor(v), fu = u - i, fv = v - j;
    const at = (arr: Float64Array, ii: number, jj: number) => arr[jj * NX + ii];
    const lerp = (arr: Float64Array) => (at(arr, i, j) * (1 - fu) + at(arr, i + 1, j) * fu) * (1 - fv) + (at(arr, i, j + 1) * (1 - fu) + at(arr, i + 1, j + 1) * fu) * fv;
    return [lerp(TX), lerp(TY)];
  };
  const lattice: [number, number][] = [];
  for (let j = 0; j < LY; j++) for (let i = 0; i < LX; i++) lattice.push(mapAt(((i + 0.5) / LX) * 1.5, (j + 0.5) / LY));
  // sampling rho: inverse CDF over the cells
  const cdf = new Float64Array(NX * NY); let acc = 0; for (let k = 0; k < NX * NY; k++) { acc += rho[k]; cdf[k] = acc; }
  const sample = (): [number, number] => { const r = Math.random(); let lo = 0, hi = NX * NY - 1; while (lo < hi) { const m = (lo + hi) >> 1; if (cdf[m] < r) lo = m + 1; else hi = m; } return [((lo % NX) + Math.random()) / NX * 1.5, (((lo / NX) | 0) + Math.random()) / NY]; };
  // cell masses on the transport grid: nearest centre
  const masses = (pts: [number, number][]) => {
    const m = new Float64Array(pts.length); const owner = new Int16Array(NX * NY);
    for (let k = 0; k < NX * NY; k++) {
      const x = xs[k % NX], y = ys[(k / NX) | 0];
      let best = 0, bd = Infinity;
      for (let p = 0; p < pts.length; p++) { const d = (pts[p][0] - x) ** 2 + (pts[p][1] - y) ** 2; if (d < bd) { bd = d; best = p; } }
      owner[k] = best; m[best] += rho[k] * pts.length;
    }
    return { m, owner };
  };
  const W = 640, H = 300, PW = 300, PH = 200, Y0 = 34;
  const canvas = el('canvas', { width: '1280', height: '600', 'aria-label': 'Two point sets on the same density, the lattice of the transport map and random samples, with their cells coloured by the share they carry' });
  canvas.style.width = '100%'; canvas.style.aspectRatio = `${W} / ${H}`;
  stage.append(canvas);
  const ctx = canvas.getContext('2d')!;
  let samples: [number, number][] = [];
  const stats = (m: Float64Array) => { const mean = m.reduce((s, v) => s + v, 0) / m.length; const cv = Math.sqrt(m.reduce((s, v) => s + (v - mean) ** 2, 0) / m.length) / mean; const under = m.reduce((s, v) => s + (v > 2 ? v : 0), 0) / m.length; const red = m.filter((v) => v < 0.25).length / m.length; return { cv, under, red }; };
  const panel = (px: number, pts: [number, number][], title: string) => {
    const { m, owner } = masses(pts);
    const cw = PW / NX, ch = PH / NY;
    for (let k = 0; k < NX * NY; k++) {
      const v = Math.log2(Math.max(0.05, m[owner[k]]));
      const t = Math.max(-1, Math.min(1, v / 1.6));
      // blue: redundant (below the share), white: fair share, red: under-served
      const r = t > 0 ? 200 : 200 - 110 * -t, gg = 200 - 120 * Math.abs(t), b = t < 0 ? 215 : 215 - 140 * t;
      ctx.fillStyle = `rgba(${r | 0},${gg | 0},${b | 0},0.9)`;
      ctx.fillRect(px + (k % NX) * cw, Y0 + ((k / NX) | 0) * ch, cw + 0.5, ch + 0.5);
    }
    ctx.strokeStyle = cssVar('--faint', '#a3a59d'); ctx.lineWidth = 1; ctx.strokeRect(px + 0.5, Y0 + 0.5, PW, PH);
    ctx.fillStyle = cssVar('--ink', '#111312');
    for (const [x, y] of pts) { ctx.beginPath(); ctx.arc(px + (x / 1.5) * PW, Y0 + y * PH, 2.1, 0, 2 * Math.PI); ctx.fill(); }
    const s = stats(m);
    ctx.font = '700 11px ui-monospace, Consolas, monospace'; ctx.fillText(title, px, Y0 - 10);
    ctx.font = '500 10px ui-monospace, Consolas, monospace'; ctx.fillStyle = cssVar('--muted', '#6f726a');
    ctx.fillText(`CV ${s.cv.toFixed(2)} · under-served mass ${(s.under * 100).toFixed(0)} % · redundant ${(s.red * 100).toFixed(0)} %`, px, Y0 + PH + 18);
  };
  const draw = (resample: boolean) => {
    if (resample || !samples.length) samples = Array.from({ length: K }, sample);
    ctx.setTransform(2, 0, 0, 2, 0, 0); ctx.clearRect(0, 0, W, H);
    panel(14, lattice, `OT LATTICE, K = ${K}`);
    panel(326, samples, `SAMPLES OF THE SAME DENSITY, K = ${K}`);
    ctx.fillStyle = cssVar('--muted', '#6f726a'); ctx.font = '500 10px ui-monospace, Consolas, monospace';
    ctx.fillText('cells: nearest centre · white = the fair share 1/K · red = carries more than its share · blue = less (redundant)', 14, H - 8);
  };
  let clock = 0;
  const loop = new Loop(stage, motion, (_now, dt) => { clock += dt; if (clock > 2.6) { clock = 0; draw(true); } });
  loop.playButton(ctl);
  button(ctl, 'Draw again', () => draw(true));
  draw(true);
}

// ---------------------------------------------------------------------------------------------------------------
// I. Editing through the lattice: drag a vertex, its neighbours follow with a smooth fall-off, every Gaussian's shape
// follows the local affine map of the mesh; folded cells turn red.
// ---------------------------------------------------------------------------------------------------------------
function demoEdit(stage: HTMLElement, ctl: HTMLElement, _motion: Motion): void {
  const W = 640, H = 424, COLS = 17, ROWS = 11, X0 = 40, Y0 = 36, PW = 560, PH = 330, R = 72;
  const m = makeMap(0.72, [{ x: 468, y: 112, s: 46, a: 0.5 }]);
  const h = PW / (COLS - 1);
  const uni: [number, number][] = [];
  for (let i = 0; i < ROWS; i++) for (let j = 0; j < COLS; j++) uni.push([X0 + (j * PW) / (COLS - 1), Y0 + (i * PH) / (ROWS - 1)]);
  const base = uni.map(([x, y]) => m.T(x, y));
  let cur = base.map((p) => [p[0], p[1]] as [number, number]);
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'A lattice of Gaussians that can be dragged; the Gaussians follow the deformation of their cells' });
  svg.style.touchAction = 'none';
  svg.style.cursor = 'grab';
  const cells = svgEl('g', { stroke: 'none' });
  const cellRects: SVGPolygonElement[] = [];
  for (let i = 0; i < ROWS - 1; i++) for (let j = 0; j < COLS - 1; j++) { const p = svgEl('polygon', { fill: 'rgba(200,60,40,0.35)', opacity: 0 }); cells.append(p); cellRects.push(p); }
  const lines: SVGPolylineElement[] = [];
  for (let k = 0; k < ROWS + COLS; k++) lines.push(svgEl('polyline', { class: 'faint', 'stroke-width': 0.9, fill: 'none' }));
  const ellGroup = svgEl('g', { class: 'cu', 'stroke-width': 1, fill: 'none', opacity: 0.45 });
  const ells = base.map(() => { const e = svgEl('ellipse', {}); ellGroup.append(e); return e; });
  const dots = base.map(() => svgEl('circle', { r: 2.4, class: 'cuf', stroke: 'none' }));
  svg.append(cells, ...lines, ellGroup, ...dots);
  const label = svgEl('text', { x: X0, y: H - 8, class: 'mu' });
  svg.append(label);
  stage.append(svg);
  const idx = (i: number, j: number) => i * COLS + j;
  const draw = () => {
    for (let i = 0; i < ROWS; i++) lines[i].setAttribute('points', cur.slice(i * COLS, (i + 1) * COLS).map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' '));
    for (let j = 0; j < COLS; j++) lines[ROWS + j].setAttribute('points', Array.from({ length: ROWS }, (_, i) => cur[idx(i, j)]).map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' '));
    let folded = 0;
    for (let i = 0; i < ROWS - 1; i++) for (let j = 0; j < COLS - 1; j++) {
      const a = cur[idx(i, j)], b = cur[idx(i, j + 1)], c = cur[idx(i + 1, j + 1)], d = cur[idx(i + 1, j)];
      const area = (p: number[], q: number[], r: number[]) => (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
      const fold = area(a, b, c) <= 0 || area(a, c, d) <= 0;
      const poly = cellRects[i * (COLS - 1) + j];
      poly.setAttribute('points', `${a[0].toFixed(1)},${a[1].toFixed(1)} ${b[0].toFixed(1)},${b[1].toFixed(1)} ${c[0].toFixed(1)},${c[1].toFixed(1)} ${d[0].toFixed(1)},${d[1].toFixed(1)}`);
      poly.setAttribute('opacity', fold ? '1' : '0');
      if (fold) folded++;
    }
    cur.forEach((p, k) => {
      dots[k].setAttribute('cx', p[0].toFixed(1)); dots[k].setAttribute('cy', p[1].toFixed(1));
      const i = (k / COLS) | 0, j = k % COLS;
      const l = cur[idx(i, Math.max(0, j - 1))], r = cur[idx(i, Math.min(COLS - 1, j + 1))], u = cur[idx(Math.max(0, i - 1), j)], d = cur[idx(Math.min(ROWS - 1, i + 1), j)];
      const dx = (Math.min(COLS - 1, j + 1) - Math.max(0, j - 1)) * h, dy = (Math.min(ROWS - 1, i + 1) - Math.max(0, i - 1)) * (PH / (ROWS - 1));
      const e = ellipseOf((r[0] - l[0]) / dx, (d[0] - u[0]) / dy, (r[1] - l[1]) / dx, (d[1] - u[1]) / dy, h);
      ells[k].setAttribute('cx', p[0].toFixed(1)); ells[k].setAttribute('cy', p[1].toFixed(1)); ells[k].setAttribute('rx', e.rx.toFixed(1)); ells[k].setAttribute('ry', e.ry.toFixed(1)); ells[k].setAttribute('transform', `rotate(${e.ang.toFixed(1)} ${p[0].toFixed(1)} ${p[1].toFixed(1)})`);
    });
    label.textContent = folded ? `${folded} folded cell${folded > 1 ? 's' : ''} (red): the mesh is no longer valid there` : 'drag any vertex · neighbours within 72 px follow with a Gaussian fall-off · ellipses: Σ′ = J Σ Jᵀ';
  };
  let drag: { k: number; start: [number, number][]; x0: number; y0: number } | null = null;
  const pos = (e: PointerEvent): [number, number] => { const r = svg.getBoundingClientRect(); return [((e.clientX - r.left) / r.width) * W, ((e.clientY - r.top) / r.height) * H]; };
  svg.addEventListener('pointerdown', (e) => {
    const [x, y] = pos(e);
    let best = -1, bd = 400;
    cur.forEach((p, k) => { const d = (p[0] - x) ** 2 + (p[1] - y) ** 2; if (d < bd) { bd = d; best = k; } });
    if (best < 0) return;
    drag = { k: best, start: cur.map((p) => [p[0], p[1]]), x0: x, y0: y };
    try { svg.setPointerCapture(e.pointerId); } catch { /* synthetic */ }
    svg.style.cursor = 'grabbing';
    e.preventDefault();
  });
  svg.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const [x, y] = pos(e);
    const dx = x - drag.x0, dy = y - drag.y0;
    const c = drag.start[drag.k];
    cur = drag.start.map((p) => { const w = Math.exp(-((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2) / (2 * R * R)); return [p[0] + dx * w, p[1] + dy * w] as [number, number]; });
    draw();
  });
  const end = () => { drag = null; svg.style.cursor = 'grab'; };
  svg.addEventListener('pointerup', end);
  svg.addEventListener('pointercancel', end);
  button(ctl, 'Reset', () => { cur = base.map((p) => [p[0], p[1]] as [number, number]); draw(); });
  const hint = el('span', { class: 'mu' }, 'DRAG A VERTEX');
  ctl.append(hint);
  draw();
}

const DEMOS: Record<string, (stage: HTMLElement, ctl: HTMLElement, motion: Motion) => void> = {
  unfold: demoUnfold, shares: demoShares, blend: demoBlend, edge: demoEdge, anyres: demoAnyRes, sinkhorn: demoSinkhorn,
  relayout: demoRelayout, shares2d: demoShares2d, edit: demoEdit,
};

/** Mounts every demo figure on the page. */
export function mountDemos(motion: Motion): void {
  for (const fig of document.querySelectorAll<HTMLElement>('figure[data-demo]')) {
    const name = fig.dataset.demo ?? '';
    const stage = fig.querySelector<HTMLElement>('.demo-stage');
    const ctl = fig.querySelector<HTMLElement>('.demo-ctl');
    const run = DEMOS[name];
    if (!stage || !ctl || !run) continue;
    try {
      run(stage, ctl, motion);
    } catch (e) {
      stage.textContent = `This demonstration could not start: ${(e as Error).message}`;
    }
  }
}
