// The operations behind the buttons: open an image, fit, export, save / load a model. All GPU work goes through
// the job runner, so only one thing runs at a time and every failure is reported the same way.

import { decodeOtsr, encodeOtsr } from '../core/fileformat';
import { otInit, renderImage, runFit, uploadBuffer, type GpuContext, type Pipelines } from '../engine';
import { fmtInt, fmtSeconds } from './dom';
import { MAX_PIXELS, WARN_PIXELS, decodeImageFile, downloadBlob, encodePng, imageFromRgba8, stem, type LoadedImage } from './image';
import type { JobRunner } from './jobs';
import { effectiveSettings, gaussianCount } from './presets';
import { defaultRoi } from './roi';
import type { AppModel, FitSummary, Store } from './store';
import { toast } from './toast';
import type { Viewer } from './viewer';

export interface SampleEntry {
  file: string;
  w: number;
  h: number;
}

export class Actions {
  constructor(readonly store: Store, readonly ctx: GpuContext, readonly pipes: Pipelines, readonly viewer: Viewer, readonly jobs: JobRunner) {}

  // ---- images --------------------------------------------------------------------------------------------------

  async openImage(file: Blob, name: string): Promise<void> {
    if (this.store.busy) return void toast('Wait for the current job to finish.', 'info', 3000);
    try {
      this.store.setStatus(`Decoding ${name} …`);
      const img = await decodeImageFile(file, name);
      const mp = (img.W * img.H) / 1e6;
      if (img.W * img.H > WARN_PIXELS && !confirm(`${img.W} × ${img.H} = ${mp.toFixed(1)} megapixels.\n\nFitting needs several hundred MB of GPU memory and, on a laptop GPU, minutes at this size. Continue?`)) {
        this.store.setStatus('Cancelled.');
        return;
      }
      this.setImage(img);
    } catch (e) {
      const msg = (e as Error).message;
      this.store.setStatus(`Could not open ${name}: ${msg}`, true);
      toast(`Could not open ${name}: ${msg}`, 'error');
    }
  }

  setImage(img: LoadedImage): void {
    const s = this.store;
    s.state.sourceBuf?.destroy();
    const sourceBuf = uploadBuffer(this.ctx.device, img.rgba, GPUBufferUsage.STORAGE, 'source');
    if (s.state.model && (s.state.model.W !== img.W || s.state.model.H !== img.H)) this.dropModel();
    s.set({ image: img, sourceBuf, roi: null, lastFit: s.state.model ? s.state.lastFit : null });
    this.viewer.roi.set(null);
    this.viewer.setSource({ W: img.W, H: img.H, rgba: sourceBuf });
    if (!s.state.model) {
      this.viewer.mode = 'bicubic';
      s.set({ viewMode: 'bicubic' });
    }
    this.viewer.fitToView();
    s.setStatus(`${img.name} — ${img.W} × ${img.H}. ${s.state.model ? '' : 'Choose a quality and press Fit & Enhance.'}`);
  }

  dropModel(): void {
    const s = this.store;
    this.viewer.setModel(null);
    s.state.model?.params.destroy();
    s.set({ model: null, lastFit: null });
  }

  setModel(params: Float32Array, W: number, H: number, meta: Record<string, unknown>, cpu = true): AppModel {
    this.dropModel();
    const N = params.length / 9;
    const buf = uploadBuffer(this.ctx.device, params, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 'model');
    const model: AppModel = { N, W, H, params: buf, cpu: cpu ? params : null, meta, saved: false };
    this.store.set({ model, viewMode: 'otsr' });
    this.viewer.setModel({ N, W, H, params: buf });
    this.viewer.mode = 'otsr';
    return model;
  }

  // ---- fitting -------------------------------------------------------------------------------------------------

  async fit(): Promise<FitSummary | undefined> {
    const s = this.store;
    const img = s.state.image;
    if (!img) return void toast('Open an image first.', 'info', 3000);
    const { kPerPx, steps } = effectiveSettings(s.state.prefs);
    if (!(kPerPx > 0) || !(steps >= 1)) return void toast('Budget and steps must be positive.', 'error');
    const K = gaussianCount(kPerPx, img.W, img.H);
    return this.jobs.run('Fit', async ({ signal, progress }) => {
      const t0 = performance.now();
      progress(-1, `Optimal-transport initialisation of ${fmtInt(K)} Gaussians …`);
      const init = await otInit(this.ctx, this.pipes, img.rgb, img.W, img.H, { K });
      if (signal.aborted) return { psnr: 0, seconds: 0, steps: 0, count: init.count, cancelled: true };
      const model = this.setModel(init.params, img.W, img.H, {}, false);
      s.setStatus(`Initialised ${fmtInt(init.count)} Gaussians in ${fmtSeconds((performance.now() - t0) / 1000)}; fitting …`);
      this.viewer.compare = s.state.prefs.compare === 'split';
      this.viewer.requestRender();
      const res = await runFit(this.ctx, this.pipes, { buffer: model.params, count: model.N }, img.W, img.H, img.rgba, {
        steps,
        logEvery: 25,
        signal,
        onProgress: (p) => {
          const rate = p.step / Math.max(1e-3, p.elapsedMs / 1000);
          const left = p.step > 0 ? (p.steps - p.step) / rate : NaN;
          progress(p.step / p.steps, `Step ${fmtInt(p.step)} / ${fmtInt(p.steps)} · ${p.psnr.toFixed(2)} dB · ${rate.toFixed(0)} it/s${p.step > 0 ? ` · ~${fmtSeconds(left)} left` : ''}`);
          this.viewer.requestRender();
        },
      });
      if (s.state.model !== model) return { psnr: 0, seconds: res.seconds, steps: 0, count: model.N, cancelled: true }; // replaced meanwhile
      const last = res.curve[res.curve.length - 1];
      model.cpu = res.params;
      model.meta = {
        generator: 'otsr-web 0.1.0', source: img.name, width: img.W, height: img.H, count: model.N, preset: s.state.prefs.preset,
        k_per_px: kPerPx, steps: last?.step ?? 0, psnr: last?.psnr, seconds: res.seconds, adapter: this.ctx.info.description,
      };
      const summary: FitSummary = { psnr: last?.psnr ?? 0, seconds: res.seconds, steps: last?.step ?? 0, count: model.N, cancelled: res.cancelled };
      if (!res.cancelled && res.seconds > 0.2) s.setPrefs({ throughput: (model.N * steps) / res.seconds });
      s.set({ lastFit: summary, model });
      s.setStatus(`${res.cancelled ? 'Fit stopped' : 'Fit finished'}: ${summary.psnr.toFixed(2)} dB after ${fmtInt(summary.steps)} steps in ${fmtSeconds(res.seconds)}. Drag the divider to compare; zoom in to inspect.`);
      if (!res.cancelled) toast(`Fitted ${fmtInt(model.N)} Gaussians — ${summary.psnr.toFixed(2)} dB in ${fmtSeconds(res.seconds)}`, 'success');
      this.viewer.requestRender();
      return summary;
    });
  }

  // ---- export --------------------------------------------------------------------------------------------------

  async exportPng(scale = this.store.state.prefs.scale): Promise<void> {
    const s = this.store;
    const m = s.state.model;
    if (!m) return void toast('Fit a model first.', 'info', 3000);
    if (!(scale > 0)) return void toast('The export scale must be positive.', 'error');
    const W = Math.round(m.W * scale);
    const H = Math.round(m.H * scale);
    if (W > 16384 || H > 16384 || W * H > 120e6) return void toast(`${W} × ${H} is too large to encode in the browser (max 16384 per side, 120 MP). Choose a smaller scale.`, 'error');
    const { edges, edgePx } = s.state.prefs;
    await this.jobs.run('Export', async ({ signal, progress }) => {
      const t0 = performance.now();
      const img = await renderImage(this.ctx, this.pipes, m.params, m.N, m.W, m.H, scale, {
        edges, edge: { edgePx }, signal,
        onProgress: (d, t) => progress(d / (t + 1), `Rendering ${W} × ${H} — tile ${d} / ${t}`),
      });
      progress(0.95, `Encoding PNG (${W} × ${H}) …`);
      const blob = await encodePng(img.rgba, img.width, img.height);
      const name = `${stem(s.state.image?.name ?? String(m.meta.source ?? 'model'))}_otsr_x${scale}${edges ? '' : '_raw'}.png`;
      downloadBlob(blob, name);
      m.saved = true;
      s.setStatus(`Exported ${name} (${(blob.size / 1e6).toFixed(1)} MB) in ${fmtSeconds((performance.now() - t0) / 1000)}.`);
      toast(`Exported ${name}`, 'success');
    });
  }

  saveModel(): void {
    const s = this.store;
    const m = s.state.model;
    if (!m?.cpu) return void toast(s.busy ? 'Wait for the fit to finish.' : 'Fit a model first.', 'info', 3000);
    const buf = encodeOtsr({ W: m.W, H: m.H, N: m.N, params: m.cpu, meta: m.meta });
    const name = `${stem(s.state.image?.name ?? String(m.meta.source ?? 'model'))}.otsr`;
    downloadBlob(new Blob([buf], { type: 'application/octet-stream' }), name);
    m.saved = true;
    s.setStatus(`Saved ${name} (${(buf.byteLength / 1e6).toFixed(1)} MB).`);
  }

  async openModelFile(file: File): Promise<void> {
    const s = this.store;
    if (s.busy) return void toast('Wait for the current job to finish.', 'info', 3000);
    try {
      const m = decodeOtsr(await file.arrayBuffer());
      if (s.state.image && (s.state.image.W !== m.W || s.state.image.H !== m.H)) {
        s.state.sourceBuf?.destroy();
        s.set({ image: null, sourceBuf: null });
        this.viewer.setSource(null);
        toast('The model has another size than the open image; comparison views need the model\'s own image.', 'info', 6000);
      }
      const model = this.setModel(m.params, m.W, m.H, { ...m.meta, source: m.meta.source ?? stem(file.name) });
      model.saved = true;
      const psnr = typeof m.meta.psnr === 'number' ? m.meta.psnr : NaN;
      s.set({ lastFit: Number.isFinite(psnr) ? { psnr, seconds: Number(m.meta.seconds ?? 0), steps: Number(m.meta.steps ?? 0), count: m.N, cancelled: false } : null });
      this.viewer.fitToView();
      s.setStatus(`Loaded ${file.name}: ${fmtInt(m.N)} Gaussians for ${m.W} × ${m.H}.`);
    } catch (e) {
      const msg = (e as Error).message;
      s.setStatus(`Could not open ${file.name}: ${msg}`, true);
      toast(`Could not open ${file.name}: ${msg}`, 'error');
    }
  }

  /** Accepts a dropped / pasted / picked file of either kind. */
  async openAny(file: File): Promise<void> {
    if (file.name.toLowerCase().endsWith('.otsr')) return this.openModelFile(file);
    if (!file.type.startsWith('image/') && !/\.(png|jpe?g|webp|avif|bmp|gif|tiff?)$/i.test(file.name)) {
      return void toast(`${file.name} is not an image or .otsr file.`, 'error');
    }
    return this.openImage(file, file.name);
  }

  // ---- samples -------------------------------------------------------------------------------------------------

  async sampleList(): Promise<SampleEntry[]> {
    try {
      const res = await fetch('samples/index.json');
      return res.ok ? ((await res.json()) as SampleEntry[]) : [];
    } catch {
      return [];
    }
  }

  async loadSample(file: string): Promise<void> {
    this.store.setStatus(`Fetching ${file} …`);
    const r = await fetch(`samples/${file}`);
    if (!r.ok) return void toast(`Could not fetch ${file} (${r.status})`, 'error');
    await this.openImage(await r.blob(), file);
  }

  // ---- region --------------------------------------------------------------------------------------------------

  /** Switches the region tool on (creating a default region) or off. */
  setTool(tool: 'pan' | 'roi'): void {
    const s = this.store;
    const size = this.viewer.sourceSize;
    if (tool === 'roi' && !size) return void toast('Open an image first.', 'info', 3000);
    if (tool === 'roi' && !this.viewer.roi.roi && size) this.viewer.roi.set(defaultRoi(size.W, size.H));
    this.viewer.setTool(tool);
    s.set({ tool, roi: this.viewer.roi.roi });
  }

  /** Loads raw pixels (scripting / tests). */
  loadRgba(rgba8: Uint8ClampedArray | number[], W: number, H: number, name = 'script.png'): void {
    if (W * H > MAX_PIXELS) throw new Error('too large');
    this.setImage(imageFromRgba8(rgba8 instanceof Uint8ClampedArray ? rgba8 : new Uint8ClampedArray(rgba8), W, H, name));
  }
}
