// The technical report page: the theme button (shared preference with the app), the table of contents following
// the scroll position, and the motion switch of the app respected.

import './report.css';

const PREFS_KEY = 'otsr.prefs.v1';

type Theme = 'auto' | 'light' | 'dark';

function readPrefs(): Record<string, unknown> {
  try {
    return (JSON.parse(localStorage.getItem(PREFS_KEY) ?? '{}') as Record<string, unknown>) || {};
  } catch {
    return {};
  }
}

function effectiveTheme(theme: Theme): 'light' | 'dark' {
  return theme === 'auto' ? (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light') : theme;
}

function apply(theme: Theme): void {
  if (theme === 'auto') document.documentElement.removeAttribute('data-theme');
  else document.documentElement.setAttribute('data-theme', theme);
  const dark = effectiveTheme(theme) === 'dark';
  const ico = document.getElementById('theme-ico');
  const label = document.getElementById('theme-label');
  if (ico) ico.textContent = dark ? '☼' : '☾';
  if (label) label.textContent = dark ? 'Light · 淺色' : 'Dark · 深色';
}

const prefs = readPrefs();
let theme: Theme = (prefs.theme as Theme) ?? 'light';
apply(theme);
if (prefs.motion === false) document.body.classList.add('no-motion');
document.getElementById('theme-btn')?.addEventListener('click', () => {
  theme = effectiveTheme(theme) === 'dark' ? 'light' : 'dark';
  apply(theme);
  try {
    localStorage.setItem(PREFS_KEY, JSON.stringify({ ...readPrefs(), theme }));
  } catch {
    /* no persistence in private mode */
  }
});

// the table of contents follows the section in view
const links = [...document.querySelectorAll<HTMLAnchorElement>('.toc a[href^="#"]')];
const sections = links.map((a) => document.getElementById(a.getAttribute('href')!.slice(1))).filter((s): s is HTMLElement => !!s);
if (sections.length && 'IntersectionObserver' in window) {
  const visible = new Map<Element, number>();
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) visible.set(e.target, e.isIntersecting ? e.boundingClientRect.top : Infinity);
    let best: Element | null = null;
    let bestTop = Infinity;
    for (const [el, top] of visible) if (top < bestTop && top > -el.clientHeight * 0.8) { best = el; bestTop = top; }
    if (!best) return;
    for (const a of links) a.classList.toggle('on', a.getAttribute('href') === `#${best.id}`);
  }, { rootMargin: '-80px 0px -60% 0px', threshold: [0, 0.1] });
  for (const s of sections) io.observe(s);
}
