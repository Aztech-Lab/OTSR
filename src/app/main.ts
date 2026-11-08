// OTSR web UI entry point: starts WebGPU, builds the viewer, the sheet and the shortcuts, exposes window.otsr.

import { acquireDevice, createPipelines, renderImage, type GpuContext, type Pipelines } from '../engine';
import { encodeOtsr } from '../core/fileformat';
import { Actions } from './actions';
import { h } from './dom';
import { JobRunner } from './jobs';
import { Panel } from './panel';
import { SHORTCUTS, installShortcuts } from './shortcuts';
import { Store, type Mode, type Theme } from './store';
import { toast } from './toast';
import { Viewer } from './viewer';
import '../style.css';

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`#${id} missing`);
  return el as T;
};

const ZOOM_STOPS = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32];

function applyLook(theme: Theme, motion: boolean): void {
  if (theme === 'auto') document.documentElement.removeAttribute('data-theme');
  else document.documentElement.setAttribute('data-theme', theme);
  document.body.classList.toggle('no-motion', !motion);
}

/** The stage colour of the current theme (CSS --stage, #rrggbb) as 0..1 RGB for the canvas background. */
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

async function boot(): Promise<void> {
  const store = new Store();
  applyLook(store.state.prefs.theme, store.state.prefs.motion !== false);
  const status = $('status');
  const statusDot = $('status-dot');
  const readoutValue = $('readout-value');
  const readoutSub = $('readout-sub');
  const consoleState = $('console-state');
  const consoleTitle = $('console-title');
  const segCount = $('seg-count');
  const segCells = Array.from({ length: 12 }, () => h('i'));
  segCount.replaceChildren(...segCells);
  store.subscribe((s, changed) => {
    if (changed.has('status')) {
      status.textContent = s.status.text;
      status.classList.toggle('error', s.status.error);
    }
    statusDot.className = `sq ${s.job ? 'blink' : s.status.error ? 'err' : 'ok'}`;
    document.body.classList.toggle('empty', !s.image && !s.model);
    document.body.classList.toggle('has-fit', !!s.lastFit && !s.job);
    readoutValue.textContent = s.lastFit ? s.lastFit.psnr.toFixed(2) : '—';
    readoutSub.textContent = s.lastFit ? `${s.lastFit.count.toLocaleString('en-US')} GAUSSIANS // ${s.lastFit.steps.toLocaleString('en-US')} STEPS // ${(s.lastFit.seconds).toFixed(1)} S` : '';
    consoleState.replaceChildren(h('span', { class: `sq ${s.job ? 'blink' : 'ok'}` }), ` ${s.job ? 'running' : s.model ? 'fitted' : 'idle'}`);
    consoleTitle.textContent = s.prefs.mode === 'research' ? 'Control // research' : 'Control // 01';
    const frac = s.job ? Math.max(0, s.job.fraction) : 0;
    segCells.forEach((c, i) => c.classList.toggle('on', s.job ? i < Math.round(frac * 12) : false));
  });

  // clock in the footer (text only, once a second)
  const clock = $('clock');
  const tick = () => { clock.textContent = new Date().toLocaleTimeString('en-GB', { hour12: false }); };
  tick();
  setInterval(tick, 1000);
  $('reload-btn').addEventListener('click', () => location.reload());

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
    pipes = await createPipelines(ctx.device);
  } catch (e) {
    return fail(`WebGPU failed to start: ${(e as Error).message}`);
  }
  store.set({ adapter: ctx.info });
  const info = ctx.info;
  const gpuName = $('adapter');
  const gpuLabel = info.description || info.device || [info.vendor, info.architecture].filter(Boolean).join(' ') || 'GPU';
  gpuName.textContent = `GPU ${gpuLabel}`;
  $('corner-gpu').textContent = `GPU: ${gpuLabel}`;
  gpuName.title = `${info.vendor} ${info.architecture} ${info.device}${info.isFallback ? ' — software fallback!' : ''}`;
  if (info.isFallback) toast('WebGPU is running on a software fallback adapter; fits will be very slow.', 'error');

  const viewer = new Viewer($('view'), $('overlay'), ctx, pipes);
  // keep the fitted image clear of the brand, the sheet and the bottom navigation (CSS px)
  const setInsets = () => {
    viewer.insets = window.innerWidth <= 960
      ? { left: 10, top: 160, right: 10, bottom: Math.round(window.innerHeight * 0.52) + 20 }
      : { left: 48, top: 150, right: 480, bottom: 230 };
  };
  setInsets();
  window.addEventListener('resize', setInsets);
  const syncStage = () => {
    viewer.background = stageColor();
    viewer.requestRender();
  };
  syncStage();
  matchMedia('(prefers-color-scheme: light)').addEventListener('change', syncStage);
  viewer.edges = store.state.prefs.edges;
  viewer.edgePx = store.state.prefs.edgePx;
  viewer.compare = store.state.prefs.compare === 'split';
  viewer.compareWith = store.state.prefs.compareWith;
  const jobs = new JobRunner(store);
  const actions = new Actions(store, ctx, pipes, viewer, jobs);
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
  const placeFrame = () => {
    const s = viewer.sourceSize;
    if (!s) { frame.hidden = true; return; }
    const dpr = window.devicePixelRatio || 1;
    const x = viewer.offsetX / dpr, y = viewer.offsetY / dpr, w = (s.W * viewer.zoom) / dpr, hh = (s.H * viewer.zoom) / dpr;
    frame.hidden = false;
    frame.style.cssText = `left:${x}px;top:${y}px;width:${w}px;height:${hh}px;--h:${hh}px`;
    frameTag.textContent = `${store.state.image?.name ?? 'MODEL'} // ${s.W}×${s.H}`.toUpperCase();
  };
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

  // system navigation and settings
  const modeInputs = document.querySelectorAll<HTMLInputElement>('input[name=mode]');
  for (const el of modeInputs) el.addEventListener('change', () => store.setPrefs({ mode: el.value as Mode }));
  const settings = $<HTMLDialogElement>('settings');
  $('settings-btn').addEventListener('click', () => settings.showModal());
  $('settings-close').addEventListener('click', () => settings.close());
  const syncTheme = radioTabs<Theme>($('theme-seg'), 'set-theme', [['light', 'Light'], ['dark', 'Dark'], ['auto', 'System']], () => store.state.prefs.theme, (t) => {
    store.setPrefs({ theme: t });
    applyLook(t, store.state.prefs.motion !== false);
    syncStage();
  });
  const motion = $<HTMLInputElement>('motion-toggle');
  motion.checked = store.state.prefs.motion !== false;
  motion.addEventListener('change', () => { store.setPrefs({ motion: motion.checked }); applyLook(store.state.prefs.theme, motion.checked); });
  const syncMode = radioTabs<Mode>($('mode-seg'), 'set-mode', [['simple', 'Simple'], ['research', 'Research']], () => store.state.prefs.mode, (m) => store.setPrefs({ mode: m }));
  const help = $<HTMLDialogElement>('help');
  $('help-table').replaceChildren(...SHORTCUTS.map(([k, d]) => h('tr', null, h('td', null, ...k.split(' / ').flatMap((p, i) => [i ? ' / ' : '', h('span', { class: 'key' }, p)])), h('td', null, d))));
  const toggleHelp = () => (help.open ? help.close() : help.showModal());
  $('help-btn').addEventListener('click', toggleHelp);
  $('help-close').addEventListener('click', () => help.close());

  store.subscribe((s) => {
    for (const el of modeInputs) el.checked = el.value === s.prefs.mode;
    syncMode();
    syncTheme();
    tbSplit.classList.toggle('active', s.prefs.compare === 'split');
    tbSplit.toggleAttribute('disabled', !s.model || !s.sourceBuf);
    tbRoi.classList.toggle('active', s.tool === 'roi');
    tbRoi.hidden = s.prefs.mode !== 'research';
    document.body.dataset.mode = s.prefs.mode;
    document.body.classList.toggle('busy', !!s.job);
    placeFrame();
  });
  for (const el of modeInputs) el.checked = el.value === store.state.prefs.mode;

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

  // sample images, when the build ships them
  void actions.sampleList().then((list) => {
    if (!list.length) return;
    const box = $('empty-samples');
    const sel = h('select', { 'aria-label': 'Sample image' }, ...list.map((e) => h('option', { value: e.file }, `${e.file} (${e.w} × ${e.h})`)));
    box.replaceChildren(sel, h('button', { class: 'textbtn', onClick: () => void actions.loadSample(sel.value) }, 'Load sample →'));
    box.hidden = false;
  });

  // device loss
  ctx.device.lost.then((l) => {
    if (l.reason === 'destroyed') return;
    $('nogpu').hidden = false;
    $('nogpu-msg').textContent = `The GPU device was lost (${l.message || l.reason}). Reload the page to continue.`;
  });

  store.setStatus('');
  (window as unknown as { otsr: unknown }).otsr = api(store, actions, viewer, ctx, pipes);
}

/** window.otsr: the same operations as the buttons, for scripts and the browser smoke test (docs/development.md). */
function api(store: Store, actions: Actions, viewer: Viewer, ctx: GpuContext, pipes: Pipelines) {
  return {
    state: () => {
      const s = store.state;
      return { image: s.image && { name: s.image.name, W: s.image.W, H: s.image.H }, model: s.model && { N: s.model.N, W: s.model.W, H: s.model.H }, fitting: !!s.job, lastFit: s.lastFit, adapter: s.adapter, zoom: s.zoom, mode: s.prefs.mode, roi: s.roi };
    },
    loadRgba: (rgba8: Uint8ClampedArray | number[], W: number, H: number, name?: string) => actions.loadRgba(rgba8, W, H, name),
    loadImage: (file: Blob, name: string) => actions.openImage(file, name),
    fit: async (opts: { kPerPx?: number; steps?: number } = {}) => {
      if (opts.kPerPx !== undefined || opts.steps !== undefined) {
        store.setPrefs({ preset: 'custom', kPerPx: opts.kPerPx ?? store.state.prefs.kPerPx, steps: opts.steps ?? store.state.prefs.steps });
      }
      await actions.fit();
      return store.state.lastFit;
    },
    export: (scale?: number) => actions.exportPng(scale),
    render: (S: number, edges = true) => {
      const m = store.state.model;
      return m ? renderImage(ctx, pipes, m.params, m.N, m.W, m.H, S, { edges }) : Promise.reject(new Error('no model'));
    },
    modelBytes: () => {
      const m = store.state.model;
      return m?.cpu ? encodeOtsr({ W: m.W, H: m.H, N: m.N, params: m.cpu, meta: m.meta }) : null;
    },
    setMode: (mode: Mode) => store.setPrefs({ mode }),
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

void boot();
