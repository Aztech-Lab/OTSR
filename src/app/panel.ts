// The side panel: built from the current mode, kept in sync with the store. Simple mode shows the three steps
// (image, quality, scale) and one button; research mode exposes every parameter plus the analysis tools.

import { EDGE_PX } from '../core/spec';
import type { Actions, SampleEntry } from './actions';
import { fmtBytes, fmtInt, fmtSeconds, h, icon } from './dom';
import { PRESETS, effectiveSettings, fitMemoryBytes, fitSecondsEstimate, gaussianCount } from './presets';
import type { AppState, Mode, Preset, Store } from './store';
import type { Viewer } from './viewer';

type Refresh = (s: AppState) => void;

export class Panel {
  private refreshers: Refresh[] = [];
  private builtMode: Mode | null = null;
  private samples: SampleEntry[] | null = null;

  constructor(readonly root: HTMLElement, readonly store: Store, readonly actions: Actions, readonly viewer: Viewer) {
    store.subscribe((s, changed) => {
      if (s.prefs.mode !== this.builtMode) this.build(s.prefs.mode);
      else if (changed.size) for (const r of this.refreshers) r(s);
    });
    void actions.sampleList().then((list) => {
      this.samples = list;
      if (list.length) this.build(this.store.state.prefs.mode);
    });
    this.build(store.state.prefs.mode);
  }

  private build(mode: Mode): void {
    this.builtMode = mode;
    this.refreshers = [];
    this.root.replaceChildren();
    const a = this.actions;
    const s = this.store;
    const research = mode === 'research';

    // ---- 1. image --------------------------------------------------------------------------------------------
    const fileInput = h('input', { type: 'file', accept: 'image/*,.otsr', hidden: true, onChange: () => { const f = fileInput.files?.[0]; if (f) void a.openAny(f); fileInput.value = ''; } });
    const imageInfo = h('div', { class: 'info' });
    const imageSec = this.section('image', research ? 'Image' : '1 · Image', true, [
      h('div', { class: 'row' },
        h('button', { class: 'btn', onClick: () => fileInput.click() }, icon('folder'), 'Open image…'),
        fileInput,
      ),
      h('div', { class: 'hint' }, 'Drop an image or .otsr file onto the view, or paste from the clipboard.'),
      imageInfo,
    ]);
    if (this.samples?.length) {
      const sel = h('select', { class: 'wide', 'aria-label': 'Sample image' }, ...this.samples.map((e) => h('option', { value: e.file }, `${e.file} (${e.w} × ${e.h})`)));
      imageSec.append(h('div', { class: 'row' }, sel, h('button', { class: 'btn', onClick: () => void a.loadSample(sel.value) }, 'Load sample')));
    }
    this.refreshers.push((st) => {
      const mp = st.image ? (st.image.W * st.image.H) / 1e6 : 0;
      imageInfo.textContent = st.image ? `${st.image.name} — ${st.image.W} × ${st.image.H} (${mp < 1 ? mp.toFixed(2) : mp.toFixed(1)} MP)` : st.model ? 'Model loaded without its image (comparison views unavailable).' : 'No image.';
    });

    // ---- 2. quality ------------------------------------------------------------------------------------------
    const presetKeys: Preset[] = research ? ['fast', 'standard', 'best', 'custom'] : ['fast', 'standard', 'best'];
    const seg = this.segmented(presetKeys.map((k) => [k, k === 'custom' ? 'Custom' : PRESETS[k].label]), s.state.prefs.preset, (v) => {
      s.setPrefs({ preset: v as Preset });
    });
    const presetHint = h('div', { class: 'hint' });
    const custom = h('div', { class: 'custom', hidden: !research },
      this.numberField('Gaussians / px', 'kPerPx', 0.005, 1, 0.01, (v) => s.setPrefs({ kPerPx: v, preset: 'custom' })),
      this.numberField('Steps', 'steps', 10, 20000, 100, (v) => s.setPrefs({ steps: Math.round(v), preset: 'custom' })),
    );
    const estimate = h('div', { class: 'estimate' });
    const edgesToggle = this.checkbox('Crisp edges at magnification', 'edges', (v) => { s.setPrefs({ edges: v }); this.viewer.edges = v; this.viewer.requestRender(); });
    const qualitySec = this.section('quality', research ? 'Fit' : '2 · Quality', true, [seg, presetHint, custom, estimate, edgesToggle]);
    if (research) {
      qualitySec.append(this.rangeField('Edge width', 'edgePx', 0.4, 3, 0.1, 'px', (v) => { s.setPrefs({ edgePx: v }); this.viewer.edgePx = v; this.viewer.requestRender(); }));
    }
    this.refreshers.push((st) => {
      const p = st.prefs;
      presetHint.textContent = p.preset === 'custom' ? (research ? 'Your own budget and step count.' : `Custom settings from Research mode: ${p.kPerPx} Gaussians / px, ${fmtInt(p.steps)} steps. Pick a preset to go back.`) : PRESETS[p.preset].hint;
      for (const el of seg.querySelectorAll<HTMLInputElement>('input')) el.checked = el.value === p.preset;
      custom.hidden = !(research && p.preset === 'custom');
      const { kPerPx, steps } = effectiveSettings(p);
      if (st.image) {
        const N = gaussianCount(kPerPx, st.image.W, st.image.H);
        const sec = fitSecondsEstimate(p, N, steps);
        estimate.textContent = `≈ ${fmtInt(N)} Gaussians · ${fmtInt(steps)} steps · ≈ ${fmtBytes(fitMemoryBytes(N, st.image.W, st.image.H))} GPU memory${sec !== null ? ` · ≈ ${fmtSeconds(sec)} on this GPU` : ''}`;
      } else estimate.textContent = '';
    });

    // ---- action ----------------------------------------------------------------------------------------------
    const fitBtn = h('button', { class: 'btn primary big', onClick: () => void a.fit() }, icon('sparkles', 18), research ? 'Fit' : 'Fit & Enhance');
    const cancelBtn = h('button', { class: 'btn', onClick: () => a.jobs.cancel() }, icon('stop', 14), 'Cancel');
    const progress = h('progress', { max: 1, value: 0 }) as HTMLProgressElement;
    const jobText = h('div', { class: 'info job' });
    const result = h('div', { class: 'result' });
    const actionSec = h('section', { class: 'action' }, h('div', { class: 'row' }, fitBtn, cancelBtn), progress, jobText, result);
    this.refreshers.push((st) => {
      const busy = !!st.job;
      fitBtn.disabled = busy || !st.image;
      cancelBtn.hidden = !busy;
      progress.hidden = !busy;
      if (st.job) {
        if (st.job.fraction < 0) progress.removeAttribute('value'); else progress.value = st.job.fraction;
      }
      jobText.textContent = st.job?.text ?? '';
      if (st.lastFit && !busy) {
        const f = st.lastFit;
        result.replaceChildren(h('b', null, `${f.psnr.toFixed(2)} dB`), ` · ${fmtInt(f.count)} Gaussians · ${fmtInt(f.steps)} steps · ${fmtSeconds(f.seconds)}${f.cancelled ? ' (stopped)' : ''}`);
      } else result.replaceChildren();
    });

    // ---- 3. export -------------------------------------------------------------------------------------------
    const scaleSeg = this.segmented([['2', '2×'], ['4', '4×'], ['8', '8×'], ['custom', '…']], String(s.state.prefs.scale), (v) => {
      if (v !== 'custom') s.setPrefs({ scale: Number(v) });
      else customScale.hidden = false;
    });
    const customScale = h('div', { class: 'row', hidden: ![2, 4, 8].every((x) => x !== s.state.prefs.scale) },
      h('label', null, 'Scale ', this.numberInput('scale', 0.5, 32, 0.5, (v) => s.setPrefs({ scale: v })), ' ×'));
    const outSize = h('div', { class: 'hint' });
    const exportBtn = h('button', { class: 'btn primary', onClick: () => void a.exportPng() }, icon('download'), 'Export PNG');
    const saveBtn = h('button', { class: 'btn', onClick: () => a.saveModel() }, icon('save'), 'Save model');
    const loadInput = h('input', { type: 'file', accept: '.otsr', hidden: true, onChange: () => { const f = loadInput.files?.[0]; if (f) void a.openModelFile(f); loadInput.value = ''; } });
    const exportSec = this.section('export', research ? 'Export' : '3 · Export', true, [
      scaleSeg, customScale, outSize,
      h('div', { class: 'row' }, exportBtn, saveBtn),
      h('div', { class: 'row' }, h('button', { class: 'btn ghost', onClick: () => loadInput.click() }, icon('folder'), 'Load model…'), loadInput),
    ]);
    this.refreshers.push((st) => {
      const m = st.model;
      exportBtn.disabled = !m || !!st.job;
      saveBtn.disabled = !m?.cpu || !!st.job;
      const base = m ?? st.image;
      outSize.textContent = base ? `${Math.round(base.W * st.prefs.scale)} × ${Math.round(base.H * st.prefs.scale)} px` : '';
    });

    this.root.append(imageSec, qualitySec, actionSec, exportSec);

    // ---- research-only sections ------------------------------------------------------------------------------
    if (research) {
      const modes: [string, string][] = [['otsr', 'OTSR'], ['raw', 'Gaussians only'], ['bicubic', 'Bicubic'], ['bilinear', 'Bilinear'], ['nearest', 'Nearest']];
      const modeSeg = this.segmented(modes, s.state.viewMode, (v) => { this.viewer.mode = v as never; s.set({ viewMode: v as never }); this.viewer.requestRender(); }, 'view-modes');
      const compare = this.checkbox('Split comparison', 'compare', (v) => { s.setPrefs({ compare: v ? 'split' : 'off' }); this.viewer.compare = v; this.viewer.requestRender(); }, s.state.prefs.compare === 'split');
      const cmpWith = h('select', { onChange: () => { const v = cmpWith.value as 'bicubic' | 'bilinear' | 'nearest'; s.setPrefs({ compareWith: v }); this.viewer.compareWith = v; this.viewer.requestRender(); } },
        ...(['bicubic', 'bilinear', 'nearest'] as const).map((k) => h('option', { value: k, selected: s.state.prefs.compareWith === k }, k.charAt(0).toUpperCase() + k.slice(1))));
      const viewSec = this.section('view', 'View', true, [
        modeSeg, compare, h('label', { class: 'inline' }, 'Compare with ', cmpWith),
        h('div', { class: 'hint' }, 'Hold B / N / R to peek at bicubic / nearest / the raw Gaussians.'),
      ]);
      this.refreshers.push((st) => {
        for (const el of modeSeg.querySelectorAll<HTMLInputElement>('input')) {
          const needsSource = !['otsr', 'raw'].includes(el.value);
          el.disabled = needsSource ? !st.sourceBuf : !st.model;
          el.checked = el.value === st.viewMode;
        }
      });

      const roiToggle = h('button', { class: 'btn', onClick: () => a.setTool(s.state.tool === 'roi' ? 'pan' : 'roi') }, icon('crop'), 'Region tool');
      const square = this.checkbox('Square region', 'square', (v) => { this.viewer.roi.opts.square = v; });
      const roiInfo = h('div', { class: 'info' });
      const zoomRoi = h('button', { class: 'btn', onClick: () => { const r = this.viewer.roi.roi; if (r) this.viewer.zoomToRoi(r); } }, icon('zoomIn'), 'Zoom to region');
      const clearRoi = h('button', { class: 'btn ghost', onClick: () => { this.viewer.roi.set(null); a.setTool('pan'); } }, 'Clear');
      const roiSec = this.section('region', 'Region of interest', false, [
        h('div', { class: 'row' }, roiToggle, square), roiInfo, h('div', { class: 'row' }, zoomRoi, clearRoi),
        h('div', { class: 'hint' }, 'Drag on the view to draw, drag corners to resize, arrow keys to nudge (Shift: 10 px). The region is shared by every analysis tool.'),
      ]);
      this.refreshers.push((st) => {
        roiToggle.classList.toggle('active', st.tool === 'roi');
        roiInfo.textContent = st.roi ? `x ${st.roi.x}, y ${st.roi.y}, ${st.roi.w} × ${st.roi.h} px` : 'No region.';
        zoomRoi.disabled = !st.roi;
        clearRoi.disabled = !st.roi;
      });

      const info = h('div', { class: 'info mono' });
      const infoSec = this.section('info', 'Session', false, [info]);
      this.refreshers.push((st) => {
        const lines = [`GPU: ${st.adapter?.description || st.adapter?.device || '—'}`];
        if (st.model) lines.push(`Model: ${fmtInt(st.model.N)} Gaussians, ${st.model.W} × ${st.model.H} (${(st.model.N / (st.model.W * st.model.H)).toFixed(3)} / px), ${fmtBytes(st.model.N * 36)}`);
        if (st.prefs.throughput) lines.push(`Throughput: ${(st.prefs.throughput / 1e6).toFixed(0)} M Gaussian-steps / s`);
        info.textContent = lines.join('\n');
      });
      this.root.append(viewSec, roiSec, infoSec);
    }

    for (const r of this.refreshers) r(s.state);
  }

  // ---- widgets ---------------------------------------------------------------------------------------------------

  private section(key: string, title: string, openByDefault: boolean, children: Node[]): HTMLElement {
    const open = this.store.state.prefs.sections[key] ?? openByDefault;
    const details = h('details', { class: 'section', open }, h('summary', null, title), ...children);
    details.addEventListener('toggle', () => this.store.setPrefs({ sections: { ...this.store.state.prefs.sections, [key]: details.open } }));
    return details;
  }

  private segmented(options: [string, string][], value: string, onChange: (v: string) => void, name = `seg-${Math.random().toString(36).slice(2, 7)}`): HTMLElement {
    const seg = h('div', { class: 'seg', role: 'radiogroup' });
    for (const [v, label] of options) {
      const input = h('input', { type: 'radio', name, value: v, id: `${name}-${v}`, checked: v === value, onChange: () => onChange(v) });
      seg.append(input, h('label', { for: `${name}-${v}` }, label));
    }
    return seg;
  }

  private checkbox(label: string, key: 'edges' | 'compare' | 'square', onChange: (v: boolean) => void, initial?: boolean): HTMLElement {
    const checked = initial ?? (key === 'edges' ? this.store.state.prefs.edges : key === 'square' ? this.viewer.roi.opts.square : false);
    const input = h('input', { type: 'checkbox', checked, onChange: () => onChange(input.checked) });
    return h('label', { class: 'check' }, input, label);
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

  private numberField(label: string, key: 'kPerPx' | 'steps', min: number, max: number, step: number, onChange: (v: number) => void): HTMLElement {
    return h('label', { class: 'field' }, h('span', null, label), this.numberInput(key, min, max, step, onChange));
  }

  private rangeField(label: string, key: 'edgePx', min: number, max: number, step: number, unit: string, onChange: (v: number) => void): HTMLElement {
    const input = h('input', { type: 'range', min, max, step, value: String(this.store.state.prefs[key] ?? EDGE_PX) });
    const out = h('span', { class: 'value' }, `${Number(input.value).toFixed(1)} ${unit}`);
    input.addEventListener('input', () => { out.textContent = `${Number(input.value).toFixed(1)} ${unit}`; onChange(Number(input.value)); });
    return h('label', { class: 'field' }, h('span', null, label), input, out);
  }
}
