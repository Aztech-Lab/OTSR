// Tiny DOM helpers: element creation, the bilingual label, number formats, the PSNR sparkline.

type Child = Node | string | null | undefined | false;

export function h<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Record<string, unknown> | null = null, ...children: Child[]): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v === undefined || v === null || v === false) continue;
      if (k === 'class') el.className = String(v);
      else if (k === 'html') el.innerHTML = String(v);
      else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v as EventListener);
      else if (k === 'dataset') Object.assign(el.dataset, v as Record<string, string>);
      else if (v === true) el.setAttribute(k, '');
      else el.setAttribute(k, String(v));
    }
  }
  for (const c of children) {
    if (c === null || c === undefined || c === false) continue;
    el.append(typeof c === 'string' ? document.createTextNode(c) : c);
  }
  return el;
}

/** A bilingual label: the English term with its Traditional Chinese counterpart underneath (every control has one). */
export function bi(en: string, zh: string): HTMLElement {
  return h('span', { class: 'bi' }, h('b', null, en), h('small', null, zh));
}

export function fmtSeconds(s: number): string {
  if (!Number.isFinite(s)) return '—';
  if (s < 1) return `${(s * 1000).toFixed(0)} ms`;
  if (s < 10) return `${s.toFixed(1)} s`;
  if (s < 90) return `${Math.round(s)} s`;
  return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
}

export function fmtBytes(b: number): string {
  return b < 1e6 ? `${(b / 1e3).toFixed(0)} kB` : b < 1e9 ? `${(b / 1e6).toFixed(b < 1e7 ? 1 : 0)} MB` : `${(b / 1e9).toFixed(2)} GB`;
}

export function fmtInt(n: number): string {
  return Math.round(n).toLocaleString('en-US');
}

/** Counts as the compact readouts show them: 950, 41.7k, 1.25M. */
export function fmtCount(n: number): string {
  return n >= 1e6 ? `${+(n / 1e6).toFixed(2)}M` : n >= 1e4 ? `${+(n / 1e3).toFixed(1)}k` : fmtInt(n);
}

/** Writes text only when it differs (a store change of any kind runs every subscriber; unchanged text stays put). */
export function setText(el: HTMLElement, text: string): void {
  if (el.textContent !== text) el.textContent = text;
}

/** Draws a PSNR curve into an SVG polyline (viewBox 0 0 240 36). */
export function sparkline(svg: SVGSVGElement, pts: { step: number; psnr: number }[], steps: number): void {
  const line = svg.querySelector('polyline');
  if (!line) return;
  if (pts.length < 2 || !(steps > 0)) { line.setAttribute('points', ''); return; }
  let lo = Infinity, hi = -Infinity;
  for (const p of pts) { lo = Math.min(lo, p.psnr); hi = Math.max(hi, p.psnr); }
  const span = Math.max(0.5, hi - lo);
  line.setAttribute('points', pts.map((p) => `${((p.step / steps) * 240).toFixed(1)},${(34 - ((p.psnr - lo) / span) * 32).toFixed(1)}`).join(' '));
}
