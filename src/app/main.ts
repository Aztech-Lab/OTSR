// OTSR web UI entry point: starts WebGPU, builds the viewer, the panel and the shortcuts, exposes window.otsr.

import { acquireDevice, createPipelines, renderImage, type GpuContext, type Pipelines } from '../engine';
import { encodeOtsr } from '../core/fileformat';
import { Actions } from './actions';
import { h, icon } from './dom';
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

function applyTheme(theme: Theme): void {
  if (theme === 'auto') document.documentElement.removeAttribute('data-theme');
  else document.documentElement.setAttribute('data-theme', theme);
}

async function boot(): Promise<void> {
  const store = new Store();
  applyTheme(store.state.prefs.theme);
  const status = $('status');
  store.subscribe((s, changed) => {
    if (changed.has('status')) {
      status.textContent = s.status.text;
      status.classList.toggle('error', s.status.error);
    }
  });
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
  const gpuBadge = $('adapter');
  gpuBadge.replaceChildren(icon('chip', 14), ` ${info.description || info.device || [info.vendor, info.architecture].filter(Boolean).join(' ') || 'GPU'}`);
  gpuBadge.title = `${info.vendor} ${info.architecture} ${info.device}${info.isFallback ? ' — software fallback!' : ''}`;
  if (info.isFallback) toast('WebGPU is running on a software fallback adapter; fits will be very slow.', 'error');

  const viewer = new Viewer($('view'), $('overlay'), ctx, pipes);
  viewer.edges = store.state.prefs.edges;
  viewer.edgePx = store.state.prefs.edgePx;
  viewer.compare = store.state.prefs.compare === 'split';
  viewer.compareWith = store.state.prefs.compareWith;
  const jobs = new JobRunner(store);
  const actions = new Actions(store, ctx, pipes, viewer, jobs);
  new Panel($('panel'), store, actions, viewer);

  // viewer -> store
  const zoomOut = $('zoom-readout');
  viewer.events = {
    onViewChange: (z) => { zoomOut.textContent = z >= 10 ? `${z.toFixed(0)}×` : `${z.toFixed(2)}×`; store.set({ zoom: z }); },
    onRoiChange: (roi) => store.set({ roi }),
    onToolChange: (tool) => store.set({ tool }),
  };
  viewer.events.onViewChange?.(viewer.zoom);

  // header: mode switch, theme, help
  const modeInputs = document.querySelectorAll<HTMLInputElement>('input[name=mode]');
  for (const el of modeInputs) {
    el.checked = el.value === store.state.prefs.mode;
    el.addEventListener('change', () => store.setPrefs({ mode: el.value as Mode }));
  }
  const themeBtn = $('theme');
  const themeIcon = () => themeBtn.replaceChildren(icon(store.state.prefs.theme === 'dark' ? 'moon' : store.state.prefs.theme === 'light' ? 'sun' : 'sun', 16));
  themeBtn.addEventListener('click', () => {
    const next: Theme = store.state.prefs.theme === 'auto' ? 'dark' : store.state.prefs.theme === 'dark' ? 'light' : 'auto';
    store.setPrefs({ theme: next });
    applyTheme(next);
    themeIcon();
    themeBtn.title = `Theme: ${next}`;
  });
  themeIcon();
  themeBtn.title = `Theme: ${store.state.prefs.theme}`;
  const help = $<HTMLDialogElement>('help');
  $('help-table').replaceChildren(...SHORTCUTS.map(([k, d]) => h('tr', null, h('td', null, ...k.split(' / ').flatMap((p, i) => [i ? ' / ' : '', h('kbd', null, p)])), h('td', null, d))));
  const toggleHelp = () => (help.open ? help.close() : help.showModal());
  $('help-btn').addEventListener('click', toggleHelp);
  $('help-close').addEventListener('click', () => help.close());

  // stage toolbar
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
  store.subscribe((s) => {
    tbSplit.classList.toggle('active', s.prefs.compare === 'split');
    tbSplit.toggleAttribute('disabled', !s.model || !s.sourceBuf);
    tbRoi.classList.toggle('active', s.tool === 'roi');
    tbRoi.hidden = s.prefs.mode !== 'research';
    document.body.dataset.mode = s.prefs.mode;
    document.body.classList.toggle('busy', !!s.job);
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
  installShortcuts(store, actions, viewer, () => picker.click(), toggleHelp);

  // device loss
  ctx.device.lost.then((l) => {
    if (l.reason === 'destroyed') return;
    $('nogpu').hidden = false;
    $('nogpu-msg').textContent = `The GPU device was lost (${l.message || l.reason}). Reload the page to continue.`;
  });

  store.setStatus(store.state.image ? '' : 'Ready. Open or drop an image to start.');
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
