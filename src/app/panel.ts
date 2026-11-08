// The console (left): the mode switch, numbered sections — source, preset, output, and in research mode view /
// region / session — and the process log of the running / last job. The fit button floats over the image (main.ts).

import { EDGE_PX } from '../core/spec';
import type { Actions, SampleEntry } from './actions';
import { fmtBytes, fmtInt, fmtSeconds, h, sparkline } from './dom';
import { effectiveSettings, fitMemoryBytes, fitSecondsEstimate, gaussianCount, viewMemoryBytes } from './presets';
import type { AppState, Loss, Mode, Preset, Store, ViewMode } from './store';
import type { Viewer } from './viewer';

type Refresh = (s: AppState) => void;

const PRESET_TEXT: Record<Preset, [string, string]> = {
  fast: ['Fast', '1/20 px · 1.5k'],
  standard: ['Std', '1/10 px · 3k'],
  best: ['Best', '1/7 px · 4k'],
  custom: ['Custom', 'manual'],
};
const VIEW_TEXT: [ViewMode, string, string][] = [
  ['otsr', 'OTSR', 'enhanced'],
  ['raw', 'Raw', 'gaussians'],
  ['bicubic', 'Bicubic', 'source'],
  ['bilinear', 'Bilinear', 'source'],
  ['nearest', 'Nearest', 'source'],
];
const LOSS_TEXT: [Loss, string, string][] = [
  ['l2', 'L2', 'mse'],
  ['l1', 'L1', 'charbonnier'],
];
const SCALES = [1, 2, 4, 8, 16];
const PROGRESS_CELLS = 24;
const LOG_LINES = 6;
/** Pixel counts for the scale buttons: 2048, 8192, 16.4k. */
const px = (n: number) => (n >= 10000 ? `${(n / 1000).toFixed(1)}k` : String(Math.round(n)));

export class Panel {
  private refreshers: Refresh[] = [];
  private builtMode: Mode | null = null;
  private samples: SampleEntry[] = [];

  constructor(readonly root: HTMLElement, readonly store: Store, readonly actions: Actions, readonly viewer: Viewer) {
    store.subscribe((s, changed) => {
      if (s.prefs.mode !== this.builtMode) this.build(s.prefs.mode);
      else if (changed.size) for (const r of this.refreshers) r(s);
    });
    void actions.sampleList().then((list) => {
      this.samples = list;
      if (list.length && this.builtMode === 'research') this.build('research');
    });
    this.build(store.state.prefs.mode);
  }

  private build(mode: Mode): void {
    this.builtMode = mode;
    this.refreshers = [];
    const a = this.actions;
    const s = this.store;
    const research = mode === 'research';
    let idx = 0;
    const section = (name: string, zh: string, ...children: (Node | null)[]) => h('section', { class: 'sec' },
      h('div', { class: 'sec-head' }, h('span', { class: 'idx' }, String(++idx).padStart(2, '0')), h('span', { class: 'name' }, name), h('span', { class: 'zh' }, zh), h('span', { class: 'fill' })),
      ...children.filter((c): c is Node => !!c));
    const readout = (k: string) => {
      const key = h('div', { class: 'k' }, k);
      const v = h('div', { class: 'v' }, '—');
      return [h('div', { class: 'readout' }, key, v), v, key] as const;
    };

    // ---- mode ---------------------------------------------------------------------------------------------------
    const modeBtn = (m: Mode, en: string, zh: string) => h('button', { class: `on-${m}`, onClick: () => s.setPrefs({ mode: m }) }, h('b', null, en), h('small', null, zh));
    const modebar = h('div', { class: 'modebar' }, h('span', { class: 'micro' }, 'Mode · 模式'),
      h('div', { class: 'seg solidseg', role: 'radiogroup' }, modeBtn('simple', 'Simple', '簡單'), modeBtn('research', 'Research', '科研')));
    this.refreshers.push((st) => { modebar.querySelectorAll('button').forEach((b) => b.classList.toggle('on', b.classList.contains(`on-${st.prefs.mode}`))); });

    // ---- 01 source --------------------------------------------------------------------------------------------
    const fileInput = h('input', { type: 'file', accept: 'image/*,.otsr', hidden: true, onChange: () => { const f = fileInput.files?.[0]; if (f) void a.openAny(f); fileInput.value = ''; } });
    const fileName = h('div', { class: 'name' }, '—');
    const fileMeta = h('div', { class: 'meta' });
    const [rSize, vSize] = readout('Size');
    const [rCount, vCount] = readout('Gaussians');
    const [rMem, vMem, kMem] = readout('GPU mem · est');
    const source = section('Source', '輸入',
      h('div', { class: 'file' }, h('div', null, fileName, fileMeta), h('button', { class: 'textbtn', onClick: () => fileInput.click() }, 'Replace', h('span', { class: 'key' }, 'O')), fileInput),
      h('div', { class: 'readouts' }, rSize, rCount, rMem),
    );
    this.refreshers.push((st) => {
      const base = st.image ?? st.model;
      const name = st.image?.name ?? (st.model ? String(st.model.meta.source ?? 'model.otsr') : '');
      fileName.textContent = name || '—';
      fileName.title = name;
      const mp = base ? (base.W * base.H) / 1e6 : 0;
      fileMeta.textContent = base ? `${st.image ? 'IMAGE' : 'MODEL'} // ${mp < 1 ? mp.toFixed(2) : mp.toFixed(1)} MP${st.model ? ' // FITTED' : ''}` : 'NO INPUT';
      vSize.textContent = base ? `${base.W}×${base.H}` : '—';
      const { kPerPx } = effectiveSettings(st.prefs);
      const N = st.model ? st.model.N : st.image ? gaussianCount(kPerPx, st.image.W, st.image.H) : 0;
      vCount.textContent = N ? fmtInt(N) : '—';
      if (base) {
        const dpr = window.devicePixelRatio || 1;
        const canvasPx = window.innerWidth * window.innerHeight * dpr * dpr;
        const fit = fitMemoryBytes(N, base.W, base.H, st.fitEntries || undefined);
        const view = viewMemoryBytes(N, base.W, base.H, canvasPx);
        vMem.textContent = `${st.fitEntries ? '' : '≈'}${fmtBytes(fit + view)}`;
        kMem.textContent = st.fitEntries ? 'GPU mem · fit' : 'GPU mem · est';
        rMem.title = `fit buffers ${fmtBytes(fit)} (parameters, splats, gradient, Adam moments, ${st.fitEntries ? fmtInt(st.fitEntries) : '≈4 per Gaussian'} tile entries) + viewer ${fmtBytes(view)}`;
      } else { vMem.textContent = '—'; kMem.textContent = 'GPU mem · est'; rMem.title = ''; }
    });

    // ---- 02 preset ----------------------------------------------------------------------------------------------
    const presetKeys: Preset[] = research ? ['fast', 'standard', 'best', 'custom'] : ['fast', 'standard', 'best'];
    const presets = this.choices(presetKeys.map((k) => [k, PRESET_TEXT[k][0], PRESET_TEXT[k][1]] as const), (st) => st.prefs.preset, (k) => s.setPrefs({ preset: k }));
    const custom = h('div', null,
      this.numberField('Gaussians / px', 'kPerPx', 0.005, 1, 0.01, (v) => s.setPrefs({ kPerPx: v, preset: 'custom' })),
      this.numberField('Steps', 'steps', 10, 20000, 100, (v) => s.setPrefs({ steps: Math.round(v), preset: 'custom' })),
    );
    const note = h('div', { class: 'note' });
    const preset = section('Preset', '檔位', presets, custom, note,
      this.switchRow('Crisp edges', 'distance-field edge pass at magnification', (st) => st.prefs.edges, (v) => { s.setPrefs({ edges: v }); this.viewer.edges = v; this.viewer.requestRender(); }),
      research ? this.rangeField('Edge width', 'edgePx', 0.4, 3, 0.1, 'px', (v) => { s.setPrefs({ edgePx: v }); this.viewer.edgePx = v; this.viewer.requestRender(); }) : null,
      research ? h('div', { class: 'field', style: 'grid-template-columns: 1fr 150px' }, h('span', null, 'Pixel loss'),
        this.choices(LOSS_TEXT, (st) => st.prefs.loss ?? 'l2', (k) => s.setPrefs({ loss: k }), 'small')) : null,
    );
    this.refreshers.push((st) => {
      custom.hidden = !(research && st.prefs.preset === 'custom');
      const { kPerPx, steps } = effectiveSettings(st.prefs);
      if (!research && st.prefs.preset === 'custom') note.textContent = `CUSTOM (research): ${st.prefs.kPerPx}/px · ${fmtInt(st.prefs.steps)} steps`;
      else if (st.image) {
        const N = gaussianCount(kPerPx, st.image.W, st.image.H);
        const sec = fitSecondsEstimate(st.prefs, N, steps);
        note.textContent = `${fmtInt(N)} gaussians · ${fmtInt(steps)} steps${sec !== null ? ` · est. ${fmtSeconds(sec)}` : ''}${research && st.prefs.loss === 'l1' ? ' · L1' : ''}`;
      } else note.textContent = 'budget and step count of the fit';
    });

    // ---- 03 output ----------------------------------------------------------------------------------------------
    const customScale = this.numberField('Custom scale', 'scale', 0.5, 32, 0.5, (v) => s.setPrefs({ scale: v }));
    const scaleItems = SCALES.map((v) => [v, `×${v}`, ''] as const);
    const scales = this.choices([...scaleItems, [0, '···', 'custom'] as const], (st) => (SCALES.includes(st.prefs.scale) ? st.prefs.scale : 0), (v) => {
      if (v === 0) customScale.hidden = false;
      else { s.setPrefs({ scale: v }); customScale.hidden = true; }
    }, 'tight');
    customScale.hidden = SCALES.includes(s.state.prefs.scale);
    const exportBtn = h('button', { class: 'btn solid big', style: 'margin-top:12px', onClick: () => void a.exportPng() }, h('span', null, 'Export PNG'), h('span', { class: 'arrow' }, '↓'));
    const saveBtn = h('button', { class: 'textbtn', onClick: () => a.saveModel() }, 'Save model', h('span', { class: 'key' }, 'S'));
    const loadInput = h('input', { type: 'file', accept: '.otsr', hidden: true, onChange: () => { const f = loadInput.files?.[0]; if (f) void a.openModelFile(f); loadInput.value = ''; } });
    const output = section('Output', '輸出', scales, customScale, exportBtn,
      h('div', { class: 'links' }, saveBtn, h('button', { class: 'textbtn', onClick: () => loadInput.click() }, 'Load model ↗'), loadInput));
    this.refreshers.push((st) => {
      const base = st.model ?? st.image;
      scales.querySelectorAll('button').forEach((b, i) => {
        if (i < SCALES.length) b.querySelector('small')!.textContent = base ? `${px(base.W * SCALES[i])}×${px(base.H * SCALES[i])}` : '';
      });
      exportBtn.disabled = !st.model || !!st.job;
      saveBtn.disabled = !st.model?.cpu || !!st.job;
    });

    const sections: HTMLElement[] = [source, preset, output];

    // ---- research: view, region, session ----------------------------------------------------------------------
    if (research) {
      const views = this.choices(VIEW_TEXT, (st) => st.viewMode, (k) => { this.viewer.mode = k; s.set({ viewMode: k }); this.viewer.requestRender(); });
      const cmpWith = h('select', { onChange: () => { const v = cmpWith.value as 'bicubic' | 'bilinear' | 'nearest'; s.setPrefs({ compareWith: v }); this.viewer.compareWith = v; this.viewer.requestRender(); } },
        ...(['bicubic', 'bilinear', 'nearest'] as const).map((k) => h('option', { value: k, selected: s.state.prefs.compareWith === k }, k.charAt(0).toUpperCase() + k.slice(1))));
      const layerNote = h('div', { class: 'note', style: 'white-space:pre-wrap' });
      sections.push(section('View', '視圖', views,
        this.switchRow('Split compare', 'divider between the source and the result (C)', (st) => st.prefs.compare === 'split', (v) => { s.setPrefs({ compare: v ? 'split' : 'off' }); this.viewer.compare = v; this.viewer.requestRender(); }),
        h('label', { class: 'field' }, h('span', null, 'Compare with'), cmpWith),
        this.switchRow('Gaussians 1σ', 'outline of every Gaussian in its own colour, from 2× zoom (G)', (st) => st.layers.gaussians, (v) => a.setLayer('gaussians', v)),
        this.switchRow('Error map', '|model − source| per pixel, inferno scale, auto gain (D)', (st) => st.layers.error, (v) => a.setLayer('error', v)),
        layerNote));
      this.refreshers.push((st) => {
        views.querySelectorAll('button').forEach((b, i) => { const k = VIEW_TEXT[i][0]; b.disabled = k === 'otsr' || k === 'raw' ? !st.model : !st.sourceBuf; });
        const lines: string[] = [];
        if (st.layers.error) lines.push(st.errorStats ? `ERROR  mae ${st.errorStats.mae.toFixed(2)}/255 · max ${st.errorStats.max.toFixed(0)}/255 · white ≥ ${(255 / st.errorStats.gain).toFixed(1)}/255` : st.model && st.image ? 'ERROR  computing …' : 'ERROR  needs a fitted model and its image');
        layerNote.textContent = lines.join('\n');
        layerNote.hidden = !st.layers.error && !st.layers.gaussians;
        if (st.layers.gaussians) {
          // the overlay is drawn in the next animation frame: read its numbers then
          requestAnimationFrame(() => {
            const o = this.viewer.outlineInfo;
            const txt = !st.model?.cpu ? 'GAUSS  available after the fit' : o.tooFar ? 'GAUSS  zoom in to 2× or more to see the outlines' : `GAUSS  ${fmtInt(o.drawn)} of ${fmtInt(o.visible)} in view${o.drawn < o.visible ? ' (limit)' : ''}`;
            layerNote.textContent = [txt, ...lines].join('\n');
          });
        }
      });

      const roiInfo = h('div', { class: 'note' });
      const zoomRoi = h('button', { class: 'textbtn', onClick: () => { const r = this.viewer.roi.roi; if (r) this.viewer.zoomToRoi(r); } }, 'Zoom to region →');
      const clearRoi = h('button', { class: 'textbtn', onClick: () => { this.viewer.roi.set(null); a.setTool('pan'); } }, 'Clear ×');
      sections.push(section('Region', '區域',
        this.switchRow('Region tool', 'drag on the image to draw; arrow keys nudge (M)', (st) => st.tool === 'roi', (v) => a.setTool(v ? 'roi' : 'pan')),
        this.switchRow('Square', 'keep the region square', () => this.viewer.roi.opts.square, (v) => { this.viewer.roi.opts.square = v; }),
        roiInfo, h('div', { class: 'links' }, zoomRoi, clearRoi)));
      this.refreshers.push((st) => {
        roiInfo.textContent = st.roi ? `x ${st.roi.x} · y ${st.roi.y} · ${st.roi.w}×${st.roi.h} px` : 'no region';
        zoomRoi.disabled = !st.roi;
        clearRoi.disabled = !st.roi;
      });

      const info = h('div', { class: 'note', style: 'white-space:pre-wrap' });
      const curve = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      curve.setAttribute('viewBox', '0 0 240 36');
      curve.setAttribute('preserveAspectRatio', 'none');
      curve.setAttribute('class', 'spark big');
      curve.appendChild(document.createElementNS('http://www.w3.org/2000/svg', 'polyline'));
      const curveLabel = h('div', { class: 'note', style: 'margin-top:4px' });
      const session = section('Session', '會話', info, curve, curveLabel);
      if (this.samples.length) {
        const sel = h('select', { 'aria-label': 'Sample image' }, ...this.samples.map((e) => h('option', { value: e.file }, `${e.file} (${e.w}×${e.h})`)));
        session.append(h('label', { class: 'field' }, h('span', null, 'Sample'), sel), h('div', { class: 'links' }, h('button', { class: 'textbtn', onClick: () => void a.loadSample(sel.value) }, 'Load sample →')));
      }
      sections.push(session);
      this.refreshers.push((st) => {
        const lines = [`GPU   ${st.adapter?.description || st.adapter?.device || '—'}`];
        if (st.model) lines.push(`MODEL ${fmtInt(st.model.N)} g · ${(st.model.N / (st.model.W * st.model.H)).toFixed(3)}/px · ${fmtBytes(st.model.N * 36)}${st.model.meta.loss === 'l1' ? ' · L1' : ''}`);
        if (st.prefs.throughput) lines.push(`RATE  ${(st.prefs.throughput / 1e6).toFixed(0)} M gaussian-steps/s`);
        info.textContent = lines.join('\n');
        const steps = st.live?.steps ?? st.curve[st.curve.length - 1]?.step ?? 0;
        sparkline(curve, st.curve, steps);
        curve.style.display = st.curve.length > 1 ? '' : 'none';
        const first = st.curve[0], last = st.curve[st.curve.length - 1];
        curveLabel.textContent = st.curve.length > 1 ? `PSNR  ${first.psnr.toFixed(2)} → ${last.psnr.toFixed(2)} dB over ${fmtInt(last.step)} steps` : '';
      });
    }

    // ---- process log: progress cells while a job runs, the last lines of the log, the result tag ---------------
    const cells = Array.from({ length: PROGRESS_CELLS }, () => h('i'));
    const progress = h('div', { class: 'progress' }, ...cells);
    const logHead = h('div', { class: 'sec-head', style: 'margin-top:14px' }, h('span', { class: 'idx' }, 'LOG'), h('span', { class: 'name' }, 'Process'), h('span', { class: 'zh' }, '過程'), h('span', { class: 'fill' }));
    const log = h('div', { class: 'log' });
    const done = h('div', { class: 'done' });
    const run = h('div', { class: 'run' }, logHead, progress, log, done);
    this.refreshers.push((st) => {
      const busy = !!st.job;
      progress.hidden = !busy;
      progress.classList.toggle('indet', busy && st.job!.fraction < 0);
      const frac = busy ? Math.max(0, st.job!.fraction) : 0;
      cells.forEach((c, i) => c.classList.toggle('on', i < Math.round(frac * PROGRESS_CELLS)));
      const lines = st.log.slice(-LOG_LINES);
      log.hidden = lines.length === 0;
      logHead.hidden = lines.length === 0;
      log.replaceChildren(...lines.map((l) => h('div', { title: l }, l)));
      if (st.lastFit && !busy) {
        const f = st.lastFit;
        done.replaceChildren(h('span', { class: 'note', style: 'margin:0' }, `${fmtInt(f.count)} g · ${fmtInt(f.steps)} steps · ${fmtSeconds(f.seconds)}`), h('span', { class: 'tag' }, f.cancelled ? 'Stopped' : 'Complete'));
        done.hidden = false;
      } else done.hidden = true;
    });

    this.root.replaceChildren(modebar, ...sections, run);
    for (const r of this.refreshers) r(s.state);
  }

  // ---- widgets ---------------------------------------------------------------------------------------------------

  /** A row of exclusive choices (label + small subtitle), bound to a value. */
  private choices<T extends string | number>(items: readonly (readonly [T, string, string])[], get: (st: AppState) => T, set: (v: T) => void, cls = ''): HTMLElement {
    const host = h('div', { class: `choices ${cls}`.trim(), role: 'radiogroup' });
    const buttons = items.map(([k, label, sub]) => {
      const b = h('button', { role: 'radio', onClick: () => set(k) }, h('b', null, label), h('small', null, sub));
      host.append(b);
      return [k, b] as const;
    });
    this.refreshers.push((st) => { const v = get(st); for (const [k, b] of buttons) b.classList.toggle('on', v === k); });
    return host;
  }

  private switchRow(label: string, sub: string, get: (st: AppState) => boolean, onChange: (v: boolean) => void): HTMLElement {
    const input = h('input', { type: 'checkbox', onChange: () => onChange(input.checked) });
    this.refreshers.push((st) => { input.checked = get(st); });
    return h('label', { class: 'switch' }, h('span', { class: 'lab' }, label, h('small', null, sub)), input, h('span', { class: 'tg' }));
  }

  private numberInput(key: 'kPerPx' | 'steps' | 'scale', min: number, max: number, step: number, onChange: (v: number) => void): HTMLInputElement {
    const input = h('input', { type: 'number', min, max, step, value: String(this.store.state.prefs[key]) });
    input.addEventListener('change', () => {
      const v = Number(input.value);
      if (Number.isFinite(v) && v >= min && v <= max) onChange(v);
      else input.value = String(this.store.state.prefs[key]);
    });
    this.refreshers.push((st) => { if (document.activeElement !== input) input.value = String(st.prefs[key]); });
    return input;
  }

  private numberField(label: string, key: 'kPerPx' | 'steps' | 'scale', min: number, max: number, step: number, onChange: (v: number) => void): HTMLElement {
    return h('label', { class: 'field' }, h('span', null, label), this.numberInput(key, min, max, step, onChange));
  }

  private rangeField(label: string, key: 'edgePx', min: number, max: number, step: number, unit: string, onChange: (v: number) => void): HTMLElement {
    const input = h('input', { type: 'range', min, max, step, value: String(this.store.state.prefs[key] ?? EDGE_PX) });
    const out = h('output', null, `${Number(input.value).toFixed(1)} ${unit}`);
    input.addEventListener('input', () => { out.textContent = `${Number(input.value).toFixed(1)} ${unit}`; onChange(Number(input.value)); });
    return h('label', { class: 'range' }, h('span', null, label), out, input);
  }
}
