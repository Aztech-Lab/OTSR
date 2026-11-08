// The console: numbered sections (source, preset, output, and in research mode view / region / session), one solid
// action, a segmented progress bar with a running log, and the result readout. Kept in sync with the store.

import { EDGE_PX } from '../core/spec';
import type { Actions, SampleEntry } from './actions';
import { fmtBytes, fmtInt, fmtSeconds, h } from './dom';
import { effectiveSettings, fitMemoryBytes, fitSecondsEstimate, gaussianCount } from './presets';
import type { AppState, Mode, Preset, Store, ViewMode } from './store';
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
const PROGRESS_CELLS = 24;

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
      const v = h('div', { class: 'v' }, '—');
      return [h('div', { class: 'readout' }, h('div', { class: 'k' }, k), v), v] as const;
    };

    // ---- 01 source --------------------------------------------------------------------------------------------
    const fileInput = h('input', { type: 'file', accept: 'image/*,.otsr', hidden: true, onChange: () => { const f = fileInput.files?.[0]; if (f) void a.openAny(f); fileInput.value = ''; } });
    const fileName = h('div', { class: 'name' }, '—');
    const fileMeta = h('div', { class: 'meta' });
    const [rSize, vSize] = readout('Size');
    const [rCount, vCount] = readout('Gaussians');
    const [rMem, vMem] = readout('GPU mem');
    const source = section('Source', '输入',
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
      vMem.textContent = base ? `≈${fmtBytes(fitMemoryBytes(N, base.W, base.H))}` : '—';
    });

    // ---- 02 preset ----------------------------------------------------------------------------------------------
    const presetKeys: Preset[] = research ? ['fast', 'standard', 'best', 'custom'] : ['fast', 'standard', 'best'];
    const choices = h('div', { class: 'choices', role: 'radiogroup' });
    const presetButtons = presetKeys.map((k) => {
      const b = h('button', { role: 'radio', onClick: () => s.setPrefs({ preset: k }) }, h('b', null, PRESET_TEXT[k][0]), h('small', null, PRESET_TEXT[k][1]));
      choices.append(b);
      return [k, b] as const;
    });
    const custom = h('div', null,
      this.numberField('Gaussians / px', 'kPerPx', 0.005, 1, 0.01, (v) => s.setPrefs({ kPerPx: v, preset: 'custom' })),
      this.numberField('Steps', 'steps', 10, 20000, 100, (v) => s.setPrefs({ steps: Math.round(v), preset: 'custom' })),
    );
    const note = h('div', { class: 'note' });
    const preset = section('Preset', '档位', choices, custom, note,
      this.switchRow('Crisp edges', 'distance-field edge pass at magnification', (st) => st.prefs.edges, (v) => { s.setPrefs({ edges: v }); this.viewer.edges = v; this.viewer.requestRender(); }),
      research ? this.rangeField('Edge width', 'edgePx', 0.4, 3, 0.1, 'px', (v) => { s.setPrefs({ edgePx: v }); this.viewer.edgePx = v; this.viewer.requestRender(); }) : null,
    );
    this.refreshers.push((st) => {
      for (const [k, b] of presetButtons) b.classList.toggle('on', st.prefs.preset === k);
      custom.hidden = !(research && st.prefs.preset === 'custom');
      const { kPerPx, steps } = effectiveSettings(st.prefs);
      if (!research && st.prefs.preset === 'custom') note.textContent = `CUSTOM (research): ${st.prefs.kPerPx}/px · ${fmtInt(st.prefs.steps)} steps`;
      else if (st.image) {
        const N = gaussianCount(kPerPx, st.image.W, st.image.H);
        const sec = fitSecondsEstimate(st.prefs, N, steps);
        note.textContent = `${fmtInt(N)} gaussians · ${fmtInt(steps)} steps${sec !== null ? ` · est. ${fmtSeconds(sec)}` : ''}`;
      } else note.textContent = 'budget and step count of the fit';
    });

    // ---- 03 output ----------------------------------------------------------------------------------------------
    const scales = [2, 4, 8, 16];
    const scaleChoices = h('div', { class: 'choices', role: 'radiogroup' });
    const scaleButtons = scales.map((v) => {
      const b = h('button', { role: 'radio', onClick: () => { s.setPrefs({ scale: v }); customScale.hidden = true; } }, h('b', null, `×${v}`), h('small', null, ''));
      scaleChoices.append(b);
      return [v, b] as const;
    });
    const customBtn = h('button', { role: 'radio', onClick: () => { customScale.hidden = false; } }, h('b', null, '···'), h('small', null, 'custom'));
    scaleChoices.append(customBtn);
    const customScale = this.numberField('Custom scale', 'scale', 0.5, 32, 0.5, (v) => s.setPrefs({ scale: v }));
    customScale.hidden = scales.includes(s.state.prefs.scale);
    const saveBtn = h('button', { class: 'textbtn', onClick: () => a.saveModel() }, 'Save model', h('span', { class: 'key' }, 'S'));
    const loadInput = h('input', { type: 'file', accept: '.otsr', hidden: true, onChange: () => { const f = loadInput.files?.[0]; if (f) void a.openModelFile(f); loadInput.value = ''; } });
    const output = section('Output', '输出', scaleChoices, customScale,
      h('div', { class: 'links' }, saveBtn, h('button', { class: 'textbtn', onClick: () => loadInput.click() }, 'Load model ↗'), loadInput));
    this.refreshers.push((st) => {
      for (const [v, b] of scaleButtons) {
        b.classList.toggle('on', st.prefs.scale === v);
        const base = st.model ?? st.image;
        b.querySelector('small')!.textContent = base ? `${Math.round(base.W * v)}×${Math.round(base.H * v)}` : '';
      }
      customBtn.classList.toggle('on', !scales.includes(st.prefs.scale));
      saveBtn.disabled = !st.model?.cpu || !!st.job;
    });

    const sections: HTMLElement[] = [source, preset, output];

    // ---- research: view, region, session ----------------------------------------------------------------------
    if (research) {
      const viewChoices = h('div', { class: 'choices', role: 'radiogroup' });
      const viewButtons = VIEW_TEXT.map(([k, en, sub]) => {
        const b = h('button', { role: 'radio', onClick: () => { this.viewer.mode = k; s.set({ viewMode: k }); this.viewer.requestRender(); } }, h('b', null, en), h('small', null, sub));
        viewChoices.append(b);
        return [k, b] as const;
      });
      const cmpWith = h('select', { onChange: () => { const v = cmpWith.value as 'bicubic' | 'bilinear' | 'nearest'; s.setPrefs({ compareWith: v }); this.viewer.compareWith = v; this.viewer.requestRender(); } },
        ...(['bicubic', 'bilinear', 'nearest'] as const).map((k) => h('option', { value: k, selected: s.state.prefs.compareWith === k }, k.charAt(0).toUpperCase() + k.slice(1))));
      sections.push(section('View', '视图', viewChoices,
        this.switchRow('Split compare', 'divider between the source and the result', (st) => st.prefs.compare === 'split', (v) => { s.setPrefs({ compare: v ? 'split' : 'off' }); this.viewer.compare = v; this.viewer.requestRender(); }),
        h('label', { class: 'field' }, h('span', null, 'Compare with'), cmpWith)));
      this.refreshers.push((st) => {
        for (const [k, b] of viewButtons) {
          b.classList.toggle('on', st.viewMode === k);
          b.disabled = k === 'otsr' || k === 'raw' ? !st.model : !st.sourceBuf;
        }
      });

      const roiInfo = h('div', { class: 'note' });
      const zoomRoi = h('button', { class: 'textbtn', onClick: () => { const r = this.viewer.roi.roi; if (r) this.viewer.zoomToRoi(r); } }, 'Zoom to region →');
      const clearRoi = h('button', { class: 'textbtn', onClick: () => { this.viewer.roi.set(null); a.setTool('pan'); } }, 'Clear ×');
      sections.push(section('Region', '区域',
        this.switchRow('Region tool', 'drag on the image to draw; arrow keys nudge (M)', (st) => st.tool === 'roi', (v) => a.setTool(v ? 'roi' : 'pan')),
        this.switchRow('Square', 'keep the region square', () => this.viewer.roi.opts.square, (v) => { this.viewer.roi.opts.square = v; }),
        roiInfo, h('div', { class: 'links' }, zoomRoi, clearRoi)));
      this.refreshers.push((st) => {
        roiInfo.textContent = st.roi ? `x ${st.roi.x} · y ${st.roi.y} · ${st.roi.w}×${st.roi.h} px` : 'no region';
        zoomRoi.disabled = !st.roi;
        clearRoi.disabled = !st.roi;
      });

      const info = h('div', { class: 'note', style: 'white-space:pre-wrap' });
      const session = section('Session', '会话', info);
      if (this.samples.length) {
        const sel = h('select', { 'aria-label': 'Sample image' }, ...this.samples.map((e) => h('option', { value: e.file }, `${e.file} (${e.w}×${e.h})`)));
        session.append(h('label', { class: 'field' }, h('span', null, 'Sample'), sel), h('div', { class: 'links' }, h('button', { class: 'textbtn', onClick: () => void a.loadSample(sel.value) }, 'Load sample →')));
      }
      sections.push(session);
      this.refreshers.push((st) => {
        const lines = [`GPU   ${st.adapter?.description || st.adapter?.device || '—'}`];
        if (st.model) lines.push(`MODEL ${fmtInt(st.model.N)} g · ${(st.model.N / (st.model.W * st.model.H)).toFixed(3)}/px · ${fmtBytes(st.model.N * 36)}`);
        if (st.prefs.throughput) lines.push(`RATE  ${(st.prefs.throughput / 1e6).toFixed(0)} M gaussian-steps/s`);
        info.textContent = lines.join('\n');
      });
    }

    // ---- run: action, progress, log, result -------------------------------------------------------------------
    const fitBtn = h('button', { class: 'btn copper big', onClick: () => void a.fit() }, h('span', null, research ? 'Fit' : 'Enhance'), h('span', { class: 'arrow' }, '▶'));
    const exportBtn = h('button', { class: 'btn solid big', onClick: () => void a.exportPng() }, h('span', null, 'Export PNG'), h('span', { class: 'arrow' }, '↓'));
    const cancelBtn = h('button', { class: 'textbtn', onClick: () => a.jobs.cancel() }, 'Abort ×');
    const cells = Array.from({ length: PROGRESS_CELLS }, () => h('i'));
    const progress = h('div', { class: 'progress' }, ...cells);
    const log = h('div', { class: 'log' });
    const done = h('div', { class: 'done' });
    const run = h('div', { class: 'run' }, h('div', { class: 'links', style: 'margin:0; gap:10px' }, fitBtn, exportBtn), progress, log, done, h('div', { class: 'links', style: 'justify-content:flex-end' }, cancelBtn));
    this.refreshers.push((st) => {
      const busy = !!st.job;
      fitBtn.disabled = busy || !st.image;
      exportBtn.disabled = busy || !st.model;
      exportBtn.hidden = !st.model;
      cancelBtn.hidden = !busy;
      progress.hidden = !busy;
      progress.classList.toggle('indet', busy && st.job!.fraction < 0);
      const frac = busy ? Math.max(0, st.job!.fraction) : st.lastFit ? 1 : 0;
      cells.forEach((c, i) => c.classList.toggle('on', i < Math.round(frac * PROGRESS_CELLS)));
      log.hidden = !busy || st.log.length === 0;
      log.replaceChildren(...st.log.slice(-3).map((l) => h('div', null, l)));
      if (st.lastFit && !busy) {
        const f = st.lastFit;
        done.replaceChildren(h('span', { class: 'note', style: 'margin:0' }, `${fmtInt(f.count)} g · ${fmtInt(f.steps)} steps · ${fmtSeconds(f.seconds)}`), h('span', { class: 'tag' }, f.cancelled ? 'Stopped' : 'Complete'));
        done.hidden = false;
      } else done.hidden = true;
    });

    this.root.replaceChildren(...sections, run);
    for (const r of this.refreshers) r(s.state);
  }

  // ---- widgets ---------------------------------------------------------------------------------------------------

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
