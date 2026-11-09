// OTSR web UI entry point: starts WebGPU, builds the viewer, the console, the monitor, the command block, the view
// menu and the shortcuts, exposes window.otsr.

import { acquireDevice, createPipelines, renderImage, type GpuContext, type Pipelines } from '../engine';
import { encodeOtsr } from '../core/fileformat';
import { Actions } from './actions';
import { Stars } from './stars';
import { bi, fmtBytes, fmtInt, fmtSeconds, h, sparkline } from './dom';
import { FitShow } from './fitshow';
import { JobRunner } from './jobs';
import { Monitor } from './monitor';
import { Panel } from './panel';
import { effectiveSettings, fitMemoryBytes, fitSecondsEstimate, gaussianCount } from './presets';
import { SHORTCUTS, installShortcuts } from './shortcuts';
import { Store, type Layers, type Mode, type Theme } from './store';
import { toast } from './toast';
import { VectorView } from './vectorview';
import { Viewer } from './viewer';
import '../style.css';

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`#${id} missing`);
  return el as T;
};

const ZOOM_STOPS = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32];
const PROGRESS_CELLS = 24;
/** The phone / small-tablet layout of style.css: the console is a bottom sheet, the zoom bar sits on it. */
const COMPACT = matchMedia('(max-width: 960px)');
/** Phones held sideways: the console is a column on the left instead of a sheet. */
const LANDSCAPE = matchMedia('(max-width: 960px) and (max-height: 520px) and (orientation: landscape)');
/** Portrait phones and small tablets, where the sheet folds away while the image is the point. */
const sheetLayout = () => COMPACT.matches && !LANDSCAPE.matches;
/** Layout geometry of a fixed element without CSS transforms (the entering animation slides panels in). */
const box = (el: HTMLElement) => ({ left: el.offsetLeft, top: el.offsetTop, right: el.offsetLeft + el.offsetWidth, bottom: el.offsetTop + el.offsetHeight, width: el.offsetWidth, height: el.offsetHeight });
const LAYER_ROWS: [keyof Layers, string, string, string][] = [
  ['vector', 'Vector view', '矢量視角', 'every Gaussian in a random colour × its luma (V)'],
  ['ot', 'OT mesh', '傳輸網格', 'the transport mesh of the initialisation over the image (T)'],
  ['gaussians', 'Gaussians 1σ', '高斯輪廓', 'outline of every Gaussian in its own colour, from 2× zoom (G)'],
  ['error', 'Error map', '誤差圖', '|model − source| per pixel, inferno scale (D)'],
];

function applyLook(theme: Theme, motion: boolean): void {
  if (theme === 'auto') document.documentElement.removeAttribute('data-theme');
  else document.documentElement.setAttribute('data-theme', theme);
  document.body.classList.toggle('no-motion', !motion);
}

/** The theme in effect ('auto' resolved through the system preference). */
function effectiveTheme(theme: Theme): 'light' | 'dark' {
  return theme === 'auto' ? (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light') : theme;
}

/** The stage colour of the current theme (CSS --bg, #rrggbb) as 0..1 RGB for the canvas background. */
function stageColor(): [number, number, number] {
  const hex = getComputedStyle(document.documentElement).getPropertyValue('--bg').trim();
  const m = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex);
  return m ? [parseInt(m[1], 16) / 255, parseInt(m[2], 16) / 255, parseInt(m[3], 16) / 255] : [0.91, 0.9, 0.88];
}

/** A radio group in the tab style, bound to a value. */
function radioTabs<T extends string>(host: HTMLElement, name: string, options: [T, string][], value: () => T, onChange: (v: T) => void): () => void {
  const inputs = options.map(([v, label]) => {
    const input = h('input', { type: 'radio', name, value: v, id: `${name}-${v}`, onChange: () => onChange(v) });
    host.append(input, h('label', { for: `${name}-${v}` }, label));
    return input;
  });
  const sync = () => inputs.forEach((el) => { el.checked = el.value === value(); });
  sync();
  return sync;
}

/** Boot log lines in the empty state, at least 140 ms apart so the start-up sequence is readable. */
function bootLog(): (line: string) => void {
  const host = $('boot');
  let last = 0;
  let queue = Promise.resolve();
  return (line: string) => {
    queue = queue.then(() => new Promise<void>((resolve) => {
      const wait = Math.max(0, last + 140 - performance.now());
      setTimeout(() => { host.append(h('div', null, line)); last = performance.now(); resolve(); }, wait);
    }));
  };
}

async function boot(): Promise<void> {
  const store = new Store();
  applyLook(store.state.prefs.theme, store.state.prefs.motion !== false);
  const log = bootLog();
  log('SYS.INIT // OTSR 0.1.4');
  const status = $('status');
  const statusDot = $('status-dot');
  const readoutValue = $('readout-value');
  const readoutSub = $('readout-sub');
  const consoleState = $('console-state');
  const segCount = $('seg-count');
  // the mode switch in the console's title bar
  const modeHead = $('mode-head');
  const modeButtons = (['simple', 'research'] as Mode[]).map((m) => {
    const b = h('button', { role: 'radio', onClick: () => store.setPrefs({ mode: m }) }, h('b', null, m === 'simple' ? 'Simple' : 'Research'), h('small', null, m === 'simple' ? '簡單' : '科研'));
    modeHead.append(b);
    return [m, b] as const;
  });
  // on touch screens there is nothing to drop or paste from a keyboard
  if (matchMedia('(pointer: coarse)').matches) { $('empty-title').textContent = 'Choose image'; $('empty-zh').textContent = '選擇 · 貼上'; }
  $('report-chip').addEventListener('click', (e) => e.stopPropagation()); // the wordmark around it goes home
  const segCells = Array.from({ length: 12 }, () => h('i'));
  segCount.replaceChildren(...segCells);
  // entering the canvas: the instrument collapses, the image irises open, brackets and panels fly in (CSS, body.entering)
  let wasEmpty = true;
  let enterTimer = 0;
  const enter = () => {
    document.body.classList.remove('entering');
    void document.body.offsetWidth; // restart the animations
    document.body.classList.add('entering');
    clearTimeout(enterTimer);
    enterTimer = window.setTimeout(() => document.body.classList.remove('entering'), 1600);
  };
  store.subscribe((s, changed) => {
    if (changed.has('status')) {
      status.textContent = s.status.text;
      status.classList.toggle('error', s.status.error);
    }
    statusDot.className = `sq ${s.job ? 'blink' : s.status.error ? 'err' : 'ok'}`;
    const empty = !s.image && !s.model;
    if (wasEmpty && !empty) {
      enter();
      if (sheetLayout() && s.prefs.panelLeft !== false) store.setPrefs({ panelLeft: false }); // the image first; the sheet is a tap away
    }
    wasEmpty = empty;
    document.body.classList.toggle('empty', empty);
    document.body.classList.toggle('has-fit', !!s.lastFit && !s.job);
    readoutValue.textContent = s.lastFit ? s.lastFit.psnr.toFixed(2) : '—';
    readoutSub.textContent = s.lastFit ? `${s.lastFit.count.toLocaleString('en-US')} GAUSSIANS // ${s.lastFit.steps.toLocaleString('en-US')} STEPS // ${(s.lastFit.seconds).toFixed(1)} S` : '';
    consoleState.replaceChildren(h('span', { class: `sq ${s.job ? 'blink' : 'ok'}` }), ` ${s.job ? 'running' : s.model ? 'fitted' : 'idle'}`);
    for (const [m, b] of modeButtons) { b.classList.toggle('on', s.prefs.mode === m); b.setAttribute('aria-checked', String(s.prefs.mode === m)); }
    const frac = s.job ? Math.max(0, s.job.fraction) : 0;
    segCells.forEach((c, i) => c.classList.toggle('on', s.job ? i < Math.round(frac * 12) : false));
  });

  // clock in the footer (text only, once a second)
  const clock = $('clock');
  const tick = () => { clock.textContent = new Date().toLocaleTimeString('en-GB', { hour12: false }); };
  tick();
  setInterval(tick, 1000);
  $('reload-btn').addEventListener('click', () => location.reload());
  $('alert-close').addEventListener('click', () => $<HTMLDialogElement>('alert').close());

  const fail = (msg: string) => {
    store.setStatus(msg, true);
    $('nogpu').hidden = false;
    $('nogpu-msg').textContent = msg;
  };
  if (!('gpu' in navigator)) return fail('WebGPU is not available in this browser.');
  let ctx: GpuContext;
  let pipes: Pipelines;
  try {
    ctx = await acquireDevice(navigator.gpu, { onError: (m) => { store.setStatus(m, true); toast(m, 'error'); } });
    log(`WEBGPU ADAPTER // ${(ctx.info.description || ctx.info.device || ctx.info.vendor || 'GPU').toUpperCase()}`);
    pipes = await createPipelines(ctx.device);
    log(`KERNELS // ${Object.keys(pipes).length} PIPELINES COMPILED`);
  } catch (e) {
    return fail(`WebGPU failed to start: ${(e as Error).message}`);
  }
  store.set({ adapter: ctx.info });
  const info = ctx.info;
  const gpuName = $('adapter');
  const gpuLabel = info.description || info.device || [info.vendor, info.architecture].filter(Boolean).join(' ') || 'GPU';
  gpuName.textContent = `GPU ${gpuLabel}`;
  $('corner-gpu').textContent = `GPU: ${gpuLabel}`;
  $('settings-gpu').textContent = gpuLabel;
  gpuName.title = `${info.vendor} ${info.architecture} ${info.device}${info.isFallback ? ' — software fallback!' : ''}`;
  if (info.isFallback) toast('WebGPU is running on a software fallback adapter; fits will be very slow.', 'error');

  const viewer = new Viewer($('view'), $('overlay'), ctx, pipes, $('stage')); // pointer events on the whole stage
  const view = $('view');
  new Stars($<HTMLCanvasElement>('stars'), () => store.state.prefs.motion !== false && !matchMedia('(prefers-reduced-motion: reduce)').matches, () => store.state.prefs.stars !== false);
  const zoombar = $('toolbar');
  const inspector = $('inspector');
  const monitorEl = $('monitor');
  const command = $('command');
  // keep the fitted image clear of the console (left), the monitor (right), the command block (top) and the bottom
  // bars, measured from the elements themselves; on the compact layout the sheet's height also docks the zoom bar
  const setInsets = () => {
    const vw = window.innerWidth;
    const open = store.state.prefs.panelLeft !== false;
    const loaded = !!(store.state.image || store.state.model);
    if (viewer.rulers === COMPACT.matches) { viewer.rulers = !COMPACT.matches; viewer.refreshOverlay(); }
    if (COMPACT.matches) {
      const cmd = box(command);
      const top = Math.round(cmd.height ? cmd.bottom + 10 : 60);
      if (LANDSCAPE.matches) {
        document.documentElement.style.setProperty('--sheet-h', '0px');
        const left = open && loaded ? box(inspector).right + 10 : 10;
        viewer.setInsets({ left, top, right: 10, bottom: box(zoombar).height + 20 });
      } else {
        const sheetH = loaded ? box(inspector).height : 0;
        document.documentElement.style.setProperty('--sheet-h', `${sheetH}px`);
        viewer.setInsets({ left: 10, top, right: 10, bottom: sheetH + 58 });
      }
      zoombar.style.left = '';
      return;
    }
    const left = open ? box(inspector).right + 40 : 64;
    const right = store.state.monitor ? vw - box(monitorEl).left + 24 : 48;
    viewer.setInsets({ left, top: 124, right, bottom: 230 });
    zoombar.style.left = `${(left + (vw - right)) / 2}px`;
  };
  setInsets();
  window.addEventListener('resize', setInsets);
  COMPACT.addEventListener('change', setInsets);
  LANDSCAPE.addEventListener('change', setInsets);
  new ResizeObserver(() => setInsets()).observe(inspector);
  new ResizeObserver(() => setInsets()).observe(command);
  const syncStage = () => {
    viewer.background = stageColor();
    viewer.requestRender();
  };
  syncStage();
  matchMedia('(prefers-color-scheme: light)').addEventListener('change', () => { syncStage(); syncThemeButton(); });
  viewer.edges = store.state.prefs.edges;
  viewer.edgePx = store.state.prefs.edgePx;
  viewer.compare = store.state.prefs.compare === 'split';
  viewer.compareWith = store.state.prefs.compareWith;
  const jobs = new JobRunner(store);
  const actions = new Actions(store, ctx, pipes, viewer, jobs);
  const show = new FitShow(viewer, ctx, pipes);
  const monitor = new Monitor($('monitor-body'), ctx, pipes, viewer, store);
  const vectorView = new VectorView(viewer, ctx, pipes, store);
  actions.show = show;
  actions.monitor = monitor;
  actions.vectorView = vectorView;
  new Panel($('panel'), store, actions, viewer);

  // zoom navigation: tick marks for the zoom stops, the nearest one raised
  const ticks = $('zoom-ticks');
  const tickButtons = ZOOM_STOPS.map((z) => {
    const b = h('button', { title: z >= 1 ? `${z}×` : `${z * 100}%`, 'aria-label': `Zoom ${z}`, 'data-l': z >= 1 ? String(z) : `1/${1 / z}`, onClick: () => viewer.zoomAbout(z) });
    ticks.append(b);
    return b;
  });
  const zoomOut = $('zoom-readout');
  const frame = $('frame');
  const frameTag = $('frame-tag');
  // the frame decoration follows the image; the canvas is clipped to the image so the animated stage shows around it
  const placeFrame = () => {
    const s = viewer.sourceSize;
    if (!s) { frame.hidden = true; view.style.clipPath = ''; return; }
    const dpr = window.devicePixelRatio || 1;
    const x = viewer.offsetX / dpr, y = viewer.offsetY / dpr, w = (s.W * viewer.zoom) / dpr, hh = (s.H * viewer.zoom) / dpr;
    frame.hidden = false;
    frame.style.cssText = `left:${x}px;top:${y}px;width:${w}px;height:${hh}px;--h:${hh}px`;
    frameTag.textContent = `${store.state.image?.name ?? 'MODEL'} // ${s.W}×${s.H}`.toUpperCase();
    const cw = view.clientWidth, ch = view.clientHeight;
    view.style.clipPath = `inset(${Math.max(0, y)}px ${Math.max(0, cw - x - w)}px ${Math.max(0, ch - y - hh)}px ${Math.max(0, x)}px)`;
  };
  const cursor = $('cursor');
  viewer.events = {
    onViewChange: (z) => {
      placeFrame();
      zoomOut.textContent = z >= 1 ? `${z >= 10 ? z.toFixed(0) : z.toFixed(2)}×` : `${Math.round(z * 100)}%`;
      let best = 0;
      ZOOM_STOPS.forEach((v, i) => { if (Math.abs(Math.log2(z / v)) < Math.abs(Math.log2(z / ZOOM_STOPS[best]))) best = i; });
      tickButtons.forEach((b, i) => b.classList.toggle('sel', i === best));
      store.set({ zoom: z });
    },
    onRoiChange: (roi) => store.set({ roi }),
    onToolChange: (tool) => store.set({ tool }),
    onHover: (sx, sy) => { cursor.textContent = sx === null || sy === null ? 'X — Y —' : `X ${Math.floor(sx)} Y ${Math.floor(sy)}`; },
  };
  viewer.events.onViewChange?.(viewer.zoom);
  $('tb-zoom-out').addEventListener('click', () => viewer.zoomBy(0.8));
  $('tb-zoom-in').addEventListener('click', () => viewer.zoomBy(1.25));
  $('tb-fit').addEventListener('click', () => viewer.fitToView());
  $('tb-100').addEventListener('click', () => viewer.zoomAbout(1));
  const tbSplit = $('tb-split');
  tbSplit.addEventListener('click', () => {
    const on = store.state.prefs.compare !== 'split';
    store.setPrefs({ compare: on ? 'split' : 'off' });
    viewer.compare = on;
    viewer.requestRender();
  });
  const tbRoi = $('tb-roi');
  tbRoi.addEventListener('click', () => actions.setTool(store.state.tool === 'roi' ? 'pan' : 'roi'));

  // the view menu: the four layers, from the toolbar
  const tbView = $('tb-view');
  const viewMenu = $('viewmenu');
  const layerInputs = LAYER_ROWS.map(([key, en, zh, sub]) => {
    const input = h('input', { type: 'checkbox', onChange: () => actions.setLayer(key, input.checked) });
    viewMenu.append(h('label', { class: 'switch' }, h('span', { class: 'lab' }, en, h('i', { class: 'zh' }, zh), h('small', null, sub)), input, h('span', { class: 'tg' })));
    return [key, input] as const;
  });
  const closeMenu = () => { viewMenu.hidden = true; };
  tbView.addEventListener('click', () => { viewMenu.hidden = !viewMenu.hidden; });
  document.addEventListener('pointerdown', (e) => { if (!viewMenu.hidden && !viewMenu.contains(e.target as Node) && e.target !== tbView && !tbView.contains(e.target as Node)) closeMenu(); });
  window.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !viewMenu.hidden) { closeMenu(); e.stopImmediatePropagation(); } }, true);

  // the command block at the top centre: the one action when idle, the live readout while a job runs
  const cmdIdle = $('cmd-idle');
  const cmdLive = $('cmd-live');
  const cmdFit = $<HTMLButtonElement>('cmd-fit');
  const cmdFitLabel = $('cmd-fit-label');
  const cmdSub = $('cmd-sub');
  const liveTitle = $('live-title');
  const liveStep = $('live-step');
  const liveNums = $('live-nums');
  const livePsnr = $('live-psnr');
  const liveRate = $('live-rate');
  const liveLeft = $('live-left');
  const liveCurve = $<HTMLElement>('live-curve') as unknown as SVGSVGElement;
  const liveText = $('live-text');
  const liveCells = Array.from({ length: PROGRESS_CELLS }, () => h('i'));
  const liveProgress = $('live-progress');
  liveProgress.replaceChildren(...liveCells);
  cmdFit.addEventListener('click', () => void actions.fit());
  $('cmd-stop').addEventListener('click', () => jobs.cancel());
  let fitLabelMode: Mode | null = null;
  store.subscribe((s) => {
    const busy = !!s.job;
    cmdIdle.hidden = busy;
    cmdLive.hidden = !busy;
    cmdFit.disabled = !s.image;
    if (fitLabelMode !== s.prefs.mode) {
      fitLabelMode = s.prefs.mode;
      cmdFitLabel.replaceChildren(...(s.prefs.mode === 'research' ? bi('Fit', '擬合') : bi('Enhance', '增強')).childNodes);
    }
    if (s.image) {
      const { kPerPx, steps } = effectiveSettings(s.prefs);
      const N = gaussianCount(kPerPx, s.image.W, s.image.H);
      const est = fitSecondsEstimate(s.prefs, N, steps);
      cmdSub.textContent = `×${s.prefs.scale} · ${s.prefs.preset} · ${fmtInt(N)} G · ≈${fmtBytes(fitMemoryBytes(N, s.image.W, s.image.H))}${est !== null ? ` · ~${fmtSeconds(est)}` : ''}${s.model ? ' · refit' : ''}`;
    } else cmdSub.textContent = '';
    if (busy) {
      const job = s.job!;
      const live = s.live;
      liveTitle.textContent = job.name === 'Fit' ? (s.prefs.mode === 'research' ? 'Fitting · 擬合中' : 'Enhancing · 增強中') : job.name;
      liveStep.textContent = live ? `STEP ${fmtInt(live.step)} / ${fmtInt(live.steps)}` : job.fraction >= 0 ? `${Math.round(job.fraction * 100)} %` : '';
      liveNums.hidden = !live;
      if (live) {
        livePsnr.textContent = live.psnr.toFixed(2);
        liveRate.textContent = `${fmtInt(live.rate)} IT/S`;
        liveLeft.textContent = Number.isFinite(live.left) ? `~${fmtSeconds(live.left)} LEFT` : '';
      }
      sparkline(liveCurve, s.curve, live?.steps ?? 0);
      liveCurve.style.display = s.curve.length > 1 ? '' : 'none';
      const frac = Math.max(0, job.fraction);
      liveProgress.classList.toggle('indet', job.fraction < 0);
      liveCells.forEach((c, i) => c.classList.toggle('on', i < Math.round(frac * PROGRESS_CELLS)));
      liveText.textContent = live ? '' : job.text;
    }
  });

  // the wordmark: back to the start screen with the entering transition played backwards (body.leaving, then .arriving)
  let leaving = false;
  const goHome = async () => {
    if (leaving || (!store.state.image && !store.state.model)) return;
    leaving = true;
    try {
      if (store.state.job) {
        jobs.cancel();
        await new Promise<void>((resolve) => { const off = store.subscribe((s) => { if (!s.job) { off(); resolve(); } }); });
      }
      const motionOn = store.state.prefs.motion !== false && !matchMedia('(prefers-reduced-motion: reduce)').matches;
      document.body.classList.remove('entering');
      document.body.classList.add('leaving');
      if (motionOn) await new Promise((r) => setTimeout(r, 720));
      actions.reset();
      document.body.classList.remove('leaving');
      if (motionOn) {
        document.body.classList.add('arriving');
        setTimeout(() => document.body.classList.remove('arriving'), 900);
      }
    } finally {
      leaving = false;
    }
  };
  goHomeRef = goHome;
  $('brand').addEventListener('click', () => void goHome());
  window.addEventListener('keydown', (e) => {
    const t = e.target as HTMLElement | null;
    if (e.key === 'Home' && !(t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA'))) { e.preventDefault(); void goHome(); }
  });

  // panels: the console collapses to a tab on the left (on phones: to its title bar); the monitor opens during fits
  // and from a tab on the right
  $('console-bookmark').addEventListener('click', () => store.setPrefs({ panelLeft: false }));
  $('left-tab').addEventListener('click', () => store.setPrefs({ panelLeft: true }));
  $('console-toggle').addEventListener('click', () => store.setPrefs({ panelLeft: store.state.prefs.panelLeft === false }));
  $('monitor-bookmark').addEventListener('click', () => store.set({ monitor: false }));
  $('right-tab').addEventListener('click', () => store.set({ monitor: true }));
  // on portrait phones the sheet folds away while a fit runs, so the process is visible, and opens afterwards on
  // the result and the export controls
  let foldedForFit = false;
  store.subscribe((s, changed) => {
    if (!changed.has('job')) return;
    if (s.job?.name === 'Fit' && sheetLayout() && !foldedForFit) {
      foldedForFit = true;
      if (s.prefs.panelLeft !== false) store.setPrefs({ panelLeft: false });
    } else if (!s.job && foldedForFit) {
      foldedForFit = false;
      if (s.prefs.panelLeft === false) store.setPrefs({ panelLeft: true });
    }
  });

  // system navigation, the theme button and the settings
  const themeBtn = $('theme-btn');
  const syncThemeButton = () => {
    const dark = effectiveTheme(store.state.prefs.theme) === 'dark';
    $('theme-ico').textContent = dark ? '☼' : '☾';
    $('theme-en').textContent = dark ? 'Light mode' : 'Dark mode';
    $('theme-zh').textContent = dark ? '淺色模式' : '深色模式';
    themeBtn.setAttribute('aria-pressed', String(dark));
  };
  const setTheme = (t: Theme) => {
    store.setPrefs({ theme: t });
    applyLook(t, store.state.prefs.motion !== false);
    syncStage();
    syncThemeButton();
  };
  themeBtn.addEventListener('click', () => setTheme(effectiveTheme(store.state.prefs.theme) === 'dark' ? 'light' : 'dark'));
  const settings = $<HTMLDialogElement>('settings');
  $('settings-btn').addEventListener('click', () => settings.showModal());
  $('settings-close').addEventListener('click', () => settings.close());
  const syncTheme = radioTabs<Theme>($('theme-seg'), 'set-theme', [['light', 'Light · 淺色'], ['dark', 'Dark · 深色'], ['auto', 'System · 系統']], () => store.state.prefs.theme, setTheme);
  const motion = $<HTMLInputElement>('motion-toggle');
  motion.checked = store.state.prefs.motion !== false;
  motion.addEventListener('change', () => { store.setPrefs({ motion: motion.checked }); applyLook(store.state.prefs.theme, motion.checked); });
  const anim = $<HTMLInputElement>('anim-toggle');
  anim.checked = store.state.prefs.fitAnim !== false;
  anim.addEventListener('change', () => store.setPrefs({ fitAnim: anim.checked }));
  const starsToggle = $<HTMLInputElement>('stars-toggle');
  const applyStars = () => { starsToggle.checked = store.state.prefs.stars !== false; document.body.classList.toggle('no-stars', store.state.prefs.stars === false); };
  applyStars();
  starsToggle.addEventListener('change', () => { store.setPrefs({ stars: starsToggle.checked }); applyStars(); });
  const syncMode = radioTabs<Mode>($('mode-seg'), 'set-mode', [['simple', 'Simple · 簡單'], ['research', 'Research · 科研']], () => store.state.prefs.mode, (m) => store.setPrefs({ mode: m }));
  const help = $<HTMLDialogElement>('help');
  $('help-table').replaceChildren(...SHORTCUTS.map(([k, d]) => h('tr', null, h('td', null, ...k.split(' / ').flatMap((p, i) => [i ? ' / ' : '', h('span', { class: 'key' }, p)])), h('td', null, d))));
  const toggleHelp = () => (help.open ? help.close() : help.showModal());
  $('help-btn').addEventListener('click', toggleHelp);
  $('help-close').addEventListener('click', () => help.close());
  syncThemeButton();

  store.subscribe((s, changed) => {
    syncMode();
    syncTheme();
    if (changed.has('prefs')) syncThemeButton();
    tbSplit.classList.toggle('active', s.prefs.compare === 'split');
    tbSplit.toggleAttribute('disabled', !s.model || !s.sourceBuf);
    tbRoi.classList.toggle('active', s.tool === 'roi');
    tbRoi.hidden = s.prefs.mode !== 'research';
    for (const [key, input] of layerInputs) input.checked = s.layers[key];
    tbView.classList.toggle('active', Object.values(s.layers).some(Boolean));
    document.body.dataset.mode = s.prefs.mode;
    document.body.classList.toggle('busy', !!s.job);
    document.body.classList.toggle('left-closed', s.prefs.panelLeft === false && !!(s.image || s.model));
    document.body.classList.toggle('monitor-open', s.monitor);
    document.body.classList.toggle('has-regions', s.regions.length > 0);
    if (changed.has('prefs') || changed.has('monitor')) setInsets();
    if (changed.has('regions') || changed.has('monitor') || changed.has('job')) {
      viewer.markers = s.monitor || s.job ? s.regions : [];
      viewer.refreshOverlay();
    }
    if (changed.has('zoom') || changed.has('image') || changed.has('model')) placeFrame();
  });

  // drag & drop, paste, file picker
  const stage = $('stage');
  const dropHint = $('drop-hint');
  let dragDepth = 0;
  window.addEventListener('dragenter', (e) => { e.preventDefault(); if (++dragDepth === 1) dropHint.hidden = false; });
  window.addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; dropHint.hidden = true; } });
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => {
    e.preventDefault();
    dragDepth = 0;
    dropHint.hidden = true;
    const f = e.dataTransfer?.files?.[0];
    if (f) void actions.openAny(f);
  });
  window.addEventListener('paste', (e) => {
    const item = [...(e.clipboardData?.items ?? [])].find((i) => i.type.startsWith('image/'));
    const f = item?.getAsFile();
    if (f) void actions.openImage(f, f.name || `pasted.${f.type.split('/')[1] ?? 'png'}`);
  });
  const picker = h('input', { type: 'file', accept: 'image/*,.otsr', hidden: true, onChange: () => { const f = picker.files?.[0]; if (f) void actions.openAny(f); picker.value = ''; } });
  document.body.appendChild(picker);
  stage.addEventListener('dblclick', (e) => { if (!viewer.sourceSize) { e.preventDefault(); picker.click(); } });
  $('nav-open').addEventListener('click', () => picker.click());
  $('empty-open').addEventListener('click', () => picker.click());
  installShortcuts(store, actions, viewer, () => picker.click(), toggleHelp);

  // sample images, when the build ships them: the Load-sample button loads the one chosen in the select (no select
  // when there is only one)
  void actions.sampleList().then((list) => {
    if (!list.length) return;
    const box = $('empty-samples');
    const sel = h('select', { 'aria-label': 'Sample image' }, ...list.map((e) => h('option', { value: e.file }, `${e.file} (${e.w} × ${e.h})`)));
    if (list.length > 1) {
      box.replaceChildren(h('span', { class: 'micro' }, 'Sample · 樣本'), sel);
      box.hidden = false;
    }
    const btn = $('empty-sample');
    btn.hidden = false;
    btn.title = `${list[0].file} · ${list[0].w} × ${list[0].h}`;
    btn.addEventListener('click', () => void actions.loadSample(sel.value));
  });

  // device loss
  ctx.device.lost.then((l) => {
    if (l.reason === 'destroyed') return;
    $('nogpu').hidden = false;
    $('nogpu-msg').textContent = `The GPU device was lost (${l.message || l.reason}). Reload the page to continue.`;
  });

  store.setStatus('');
  log('READY // DROP, PASTE OR OPEN AN IMAGE');
  (window as unknown as { otsr: unknown }).otsr = api(store, actions, viewer, ctx, pipes, show);
}

/** window.otsr: the same operations as the buttons, for scripts and the browser smoke test (docs/development.md). */
function api(store: Store, actions: Actions, viewer: Viewer, ctx: GpuContext, pipes: Pipelines, show: FitShow) {
  return {
    state: () => {
      const s = store.state;
      return { image: s.image && { name: s.image.name, W: s.image.W, H: s.image.H }, model: s.model && { N: s.model.N, W: s.model.W, H: s.model.H }, fitting: !!s.job, lastFit: s.lastFit, live: s.live, adapter: s.adapter, zoom: s.zoom, mode: s.prefs.mode, roi: s.roi, layers: s.layers, errorStats: s.errorStats, regions: s.regions, monitor: s.monitor, phase: show.phaseName };
    },
    loadRgba: (rgba8: Uint8ClampedArray | number[], W: number, H: number, name?: string) => actions.loadRgba(rgba8, W, H, name),
    loadImage: (file: Blob, name: string) => actions.openImage(file, name),
    fit: async (opts: { kPerPx?: number; steps?: number; loss?: 'l2' | 'l1' } = {}) => {
      if (opts.kPerPx !== undefined || opts.steps !== undefined) {
        store.setPrefs({ preset: 'custom', kPerPx: opts.kPerPx ?? store.state.prefs.kPerPx, steps: opts.steps ?? store.state.prefs.steps });
      }
      if (opts.loss) store.setPrefs({ loss: opts.loss });
      await actions.fit();
      return store.state.lastFit;
    },
    export: (scale?: number, format?: 'png' | 'jpeg' | 'webp' | 'avif' | 'bmp') => actions.exportImage(format, scale),
    home: () => goHomeRef?.(),
    render: (S: number, edges = true) => {
      const m = store.state.model;
      return m ? renderImage(ctx, pipes, m.params, m.N, m.W, m.H, S, { edges }) : Promise.reject(new Error('no model'));
    },
    modelBytes: () => {
      const m = store.state.model;
      return m?.cpu ? encodeOtsr({ W: m.W, H: m.H, N: m.N, params: m.cpu, meta: m.meta }) : null;
    },
    setMode: (mode: Mode) => store.setPrefs({ mode }),
    setLayer: (name: keyof Layers, on: boolean) => actions.setLayer(name, on),
    setRoi: (roi: { x: number; y: number; w: number; h: number } | null) => { viewer.roi.set(roi, viewer.transform() ?? undefined); },
    /** The viewer canvas as a PNG data URL (headless browsers cannot read the canvas itself). */
    snapshotPng: async (withOverlay = true) => {
      const blob = await viewer.snapshotPng(withOverlay);
      return new Promise<string>((resolve) => {
        const r = new FileReader();
        r.onload = () => resolve(r.result as string);
        r.readAsDataURL(blob);
      });
    },
    viewer,
    store,
  };
}

let goHomeRef: (() => Promise<void>) | null = null;

void boot();
