// The operations behind the buttons: open an image, fit, export, save / load a model. All GPU work goes through
// the job runner, so only one thing runs at a time and every failure is reported the same way.

import { decodeOtsr, encodeOtsr } from '../core/fileformat';
import { otInit, renderImage, runFit, uploadBuffer, type GpuContext, type Pipelines } from '../engine';
import { fmtBytes, fmtInt, fmtSeconds } from './dom';
import type { FitShow } from './fitshow';
import type { Monitor } from './monitor';
import type { VectorView } from './vectorview';
import { EXPORT_FORMATS, MAX_PIXELS, WARN_PIXELS, decodeImageFile, downloadBlob, encodeImage, imageFromRgba8, stem, type ExportFormat, type LoadedImage } from './image';
import type { JobRunner } from './jobs';
import { effectiveSettings, gaussianCount } from './presets';
import { defaultRoi } from './roi';
import type { AppModel, FitSummary, Layers, Store } from './store';
import { toast } from './toast';
import type { Viewer } from './viewer';

export interface SampleEntry {
  file: string;
  w: number;
  h: number;
}

/** Display gain of the error map: 1 / (99.5th percentile of |error|), kept between 4 and 128 (16 = white from 16/255). */
const ERROR_GAIN_MIN = 4;
const ERROR_GAIN_MAX = 128;
/** Inferno-like colour scale, 256 RGB entries interpolated from 8 anchors. */
const ERROR_LUT: Uint8ClampedArray = (() => {
  const k = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]];
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const x = (i / 255) * (k.length - 1);
    const j = Math.min(k.length - 2, Math.floor(x));
    const f = x - j;
    for (let c = 0; c < 3; c++) lut[3 * i + c] = k[j][c] + (k[j + 1][c] - k[j][c]) * f;
  }
  return lut;
})();

/** Interval of the parameter snapshots during a fit (animation + monitor). */
const SNAPSHOT_MS = 600;

export class Actions {
  /** The fit animation and the monitor panel, attached by main.ts once the DOM exists. */
  show: FitShow | null = null;
  monitor: Monitor | null = null;
  vectorView: VectorView | null = null;

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
    // re-checked here, after any decode await: a job started meanwhile still holds the model's buffers
    if (s.busy) return void toast('Wait for the current job to finish.', 'info', 3000);
    s.state.sourceBuf?.destroy();
    const sourceBuf = uploadBuffer(this.ctx.device, img.rgba, GPUBufferUsage.STORAGE, 'source');
    if (s.state.model && (s.state.model.W !== img.W || s.state.model.H !== img.H)) this.dropModel();
    s.set({ image: img, sourceBuf, roi: null, lastFit: s.state.model ? s.state.lastFit : null, errorStats: null });
    this.viewer.setErrorMap(null);
    this.viewer.roi.set(null);
    this.viewer.setSource({ W: img.W, H: img.H, rgba: sourceBuf });
    if (!s.state.model) {
      this.viewer.mode = 'bicubic';
      s.set({ viewMode: 'bicubic' });
    }
    this.viewer.fitToView();
    s.setStatus(`${img.name} — ${img.W} × ${img.H}. ${s.state.model ? '' : 'Choose a preset and press Enhance.'}`);
  }

  dropModel(): void {
    const s = this.store;
    this.vectorView?.release(); // its recoloured copy binds the parameters about to be destroyed
    this.viewer.setModel(null);
    this.viewer.setErrorMap(null);
    s.state.model?.params.destroy();
    s.set({ model: null, lastFit: null, errorStats: null });
  }

  setModel(params: Float32Array, W: number, H: number, meta: Record<string, unknown>, cpu = true): AppModel {
    this.dropModel();
    const N = params.length / 9;
    const buf = uploadBuffer(this.ctx.device, params, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 'model');
    const model: AppModel = { N, W, H, params: buf, cpu: cpu ? params : null, meta, saved: false };
    this.store.set({ model, viewMode: 'otsr' });
    this.viewer.setModel(model); // the same object: the outline layer sees `cpu` once the fit has filled it
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
    const loss = s.state.prefs.loss ?? 'l2';
    const prefs = s.state.prefs;
    const show = prefs.fitAnim !== false && prefs.motion !== false && !matchMedia('(prefers-reduced-motion: reduce)').matches ? this.show : null;
    this.vectorView?.release();
    s.set({ live: null, curve: [], errorStats: null, monitor: true, fitEntries: 0, layers: { ...s.state.layers, vector: false } });
    this.viewer.layers = { ...this.viewer.layers, vector: false };
    this.viewer.setErrorMap(null);
    this.monitor?.reset(img.W, img.H, img.rgb);
    const summary = await this.jobs.run('Fit', async ({ signal, progress }) => {
      const t0 = performance.now();
      show?.start(img.W, img.H);
      try {
        progress(-1, `Optimal transport for ${fmtInt(K)} Gaussians · pixel density on the CPU …`);
        const init = await otInit(this.ctx, this.pipes, img.rgb, img.W, img.H, {
          K,
          onPhase: (ph, i) => {
            if (ph === 'density') progress(-1, `Density ρ (λ = ${i.lam.toFixed(3)}) pooled to ${i.grid.nx} × ${i.grid.ny} cells · ${i.ms.toFixed(0)} ms`);
            else if (ph === 'transport') progress(-1, `Sinkhorn transport map on the GPU · ${i.ms.toFixed(0)} ms`);
            else progress(-1, `Lattice ${i.gx} × ${i.gy} warped by the map, colours sampled · ${i.ms.toFixed(0)} ms`);
          },
          onStage: (T, k, n, eps, grid) => {
            show?.otStage(T, grid);
            progress(-1, `OT ε-scaling stage ${k + 1} / ${n} · ε = ${eps.toExponential(1)}`, 'ot');
          },
        });
        show?.otDone();
        if (signal.aborted) return { psnr: 0, seconds: 0, steps: 0, count: init.count, cancelled: true };
        const model = this.setModel(init.params, img.W, img.H, {}, false);
        model.ot = { T: init.T, grid: init.grid };
        show?.attachModel(model);
        if (show?.displayBuffer) this.monitor?.setVectorSource(show.displayBuffer, (enc) => show.encodeRecolour(enc));
        s.setStatus(`Initialised ${fmtInt(init.count)} Gaussians in ${fmtSeconds((performance.now() - t0) / 1000)}; fitting …`);
        progress(-1, `Uploaded ${fmtInt(model.N)} × 9 parameters (${fmtBytes(model.N * 36)}) · Adam moments cleared · ${loss.toUpperCase()} loss`);
        progress(-1, `Optimising · ${fmtInt(steps)} steps · progress every 50 steps`);
        this.viewer.compare = s.state.prefs.compare === 'split';
        this.viewer.requestRender();
        // the monitor's crops follow the optimiser at a fixed cadence
        let fitting = true;
        const snapLoop = (async () => {
          while (fitting) {
            await new Promise((r) => setTimeout(r, SNAPSHOT_MS));
            if (!fitting || s.state.model !== model) break;
            await this.monitor?.renderCrops();
          }
        })();
        const res = await runFit(this.ctx, this.pipes, { buffer: model.params, count: model.N }, img.W, img.H, img.rgba, {
          steps,
          logEvery: 50,
          checkEvery: 40, // fewer CPU-GPU round trips: the overlay animation runs on the same thread
          signal,
          loss,
          onProgress: (p) => {
            const rate = p.step / Math.max(1e-3, p.elapsedMs / 1000);
            const left = p.step > 0 ? (p.steps - p.step) / rate : NaN;
            progress(p.step / p.steps, `Step ${fmtInt(p.step)} / ${fmtInt(p.steps)} · ${p.psnr.toFixed(2)} dB · ${rate.toFixed(0)} it/s${p.step > 0 ? ` · ~${fmtSeconds(left)} left` : ''}`, 'step');
            s.set({ live: { step: p.step, steps: p.steps, psnr: p.psnr, rate, left }, curve: [...s.state.curve, { step: p.step, psnr: p.psnr }] });
            this.viewer.requestRender();
          },
        });
        fitting = false;
        await snapLoop;
        if (s.state.model !== model) return { psnr: 0, seconds: res.seconds, steps: 0, count: model.N, cancelled: true }; // replaced meanwhile
        const last = res.curve[res.curve.length - 1];
        model.cpu = res.params;
        model.meta = {
          generator: 'otsr-web 0.1.0', source: img.name, width: img.W, height: img.H, count: model.N, preset: s.state.prefs.preset,
          k_per_px: kPerPx, steps: last?.step ?? 0, psnr: last?.psnr, seconds: res.seconds, adapter: this.ctx.info.description, loss,
        };
        const summary: FitSummary = { psnr: last?.psnr ?? 0, seconds: res.seconds, steps: last?.step ?? 0, count: model.N, cancelled: res.cancelled };
        if (!res.cancelled && res.seconds > 0.2) s.setPrefs({ throughput: (model.N * steps) / res.seconds });
        progress(1, `Read back ${fmtInt(model.N)} Gaussians (${fmtBytes(model.N * 36)}) · tile entries ${fmtInt(res.entries)}, capacity ${fmtInt(res.entryCap)}`);
        progress(1, `Edge pass ready · distance field ${(s.state.prefs.edgePx ?? 1.4).toFixed(1)} px from 1.5× (view and export)`);
        s.set({ lastFit: summary, model, live: null, fitEntries: res.entries });
        s.setStatus(`${res.cancelled ? 'Fit stopped' : 'Fit finished'}: ${summary.psnr.toFixed(2)} dB after ${fmtInt(summary.steps)} steps in ${fmtSeconds(res.seconds)}. Drag the divider to compare; zoom in to inspect.`);
        if (!res.cancelled) toast(`Fitted ${fmtInt(model.N)} Gaussians — ${summary.psnr.toFixed(2)} dB in ${fmtSeconds(res.seconds)}`, 'success');
        this.viewer.requestRender();
        if (show?.active) {
          if (res.cancelled) show.cancel();
          else {
            progress(1, `Scan reveal · ${summary.psnr.toFixed(2)} dB`);
            await show.finish();
          }
        } else if (!res.cancelled && s.state.prefs.motion !== false) this.viewer.sweepSplit();
        return summary;
      } finally {
        this.monitor?.setVectorSource(null, null); // crops back on the model's own buffer before the copy is destroyed
        show?.cancel();
      }
    });
    s.set({ live: null, monitor: false });
    if (summary && s.state.layers.error) void this.refreshErrorMap();
    return summary;
  }

  // ---- research layers -----------------------------------------------------------------------------------------

  /** Switches a view layer on or off (the error map is computed on first use; the vector view needs a model). */
  setLayer(name: keyof Layers, on: boolean): void {
    if (name === 'vector') {
      if (on && (!this.store.state.model || this.store.busy)) return void toast(this.store.busy ? 'The vector view is shown by the fit itself; toggle it afterwards.' : 'Fit a model first.', 'info', 3000);
      on = this.vectorView?.set(on) ?? false;
    }
    const layers = { ...this.store.state.layers, [name]: on };
    this.store.set({ layers });
    this.viewer.layers = layers;
    if (name === 'error' && on && !this.viewer.hasErrorMap) void this.refreshErrorMap();
    else if (name === 'vector') this.viewer.requestRender();
    else this.viewer.refreshOverlay();
  }

  /** Renders the model at 1x and colours |model - source| per pixel (mean over RGB) with an inferno scale. */
  async refreshErrorMap(): Promise<void> {
    const s = this.store;
    const m = s.state.model;
    const img = s.state.image;
    if (!m || !img || m.W !== img.W || m.H !== img.H) {
      this.viewer.setErrorMap(null);
      s.set({ errorStats: null });
      if (m && !img) toast('The error map needs the model\'s source image.', 'info', 4000);
      return;
    }
    if (s.busy) return;
    await this.jobs.run('Error map', async ({ signal, progress }) => {
      progress(-1, `Error map · rendering ${m.W} × ${m.H} at 1× without the edge pass …`);
      const r = await renderImage(this.ctx, this.pipes, m.params, m.N, m.W, m.H, 1, { edges: false, signal });
      const n = m.W * m.H;
      const out = new Uint8ClampedArray(n * 4);
      const gt = img.rgba;
      const px = r.rgba;
      const err = new Float32Array(n);
      const hist = new Uint32Array(1024); // |error| in [0, 0.25) at 1/4096 resolution
      let sum = 0;
      let max = 0;
      for (let i = 0; i < n; i++) {
        const e = (Math.abs(px[4 * i] / 255 - gt[4 * i]) + Math.abs(px[4 * i + 1] / 255 - gt[4 * i + 1]) + Math.abs(px[4 * i + 2] / 255 - gt[4 * i + 2])) / 3;
        err[i] = e;
        sum += e;
        if (e > max) max = e;
        hist[Math.min(1023, (e * 4096) | 0)]++;
      }
      let acc = 0;
      let bin = 0;
      for (; bin < 1024 && acc < n * 0.995; bin++) acc += hist[bin];
      const gain = Math.max(ERROR_GAIN_MIN, Math.min(ERROR_GAIN_MAX, 1 / Math.max(1e-4, (bin + 1) / 4096)));
      for (let i = 0; i < n; i++) {
        const k = 3 * Math.min(255, Math.round(err[i] * gain * 255));
        out[4 * i] = ERROR_LUT[k];
        out[4 * i + 1] = ERROR_LUT[k + 1];
        out[4 * i + 2] = ERROR_LUT[k + 2];
        out[4 * i + 3] = 255;
      }
      const bmp = await createImageBitmap(new ImageData(out, m.W, m.H));
      if (s.state.model !== m) { bmp.close(); return; }
      this.viewer.setErrorMap(bmp);
      s.set({ errorStats: { mae: (sum / n) * 255, max: max * 255, gain } });
    });
  }

  // ---- export --------------------------------------------------------------------------------------------------

  /** Renders the model at `scale` and downloads it as `format` (the chosen one by default; PNG when the browser cannot encode it). */
  async exportImage(format: ExportFormat = this.store.state.prefs.exportFormat ?? 'png', scale = this.store.state.prefs.scale): Promise<void> {
    const s = this.store;
    const m = s.state.model;
    if (!m) return void toast('Fit a model first.', 'info', 3000);
    if (!(scale > 0)) return void toast('The export scale must be positive.', 'error');
    const W = Math.round(m.W * scale);
    const H = Math.round(m.H * scale);
    if (W > 16384 || H > 16384 || W * H > 120e6) return void toast(`${W} × ${H} is too large to encode in the browser (max 16384 per side, 120 MP). Choose a smaller scale.`, 'error');
    const { edges, edgePx, exportQuality } = s.state.prefs;
    const info = EXPORT_FORMATS.find((f) => f.key === format) ?? EXPORT_FORMATS[0];
    await this.jobs.run('Export', async ({ signal, progress }) => {
      const t0 = performance.now();
      const img = await renderImage(this.ctx, this.pipes, m.params, m.N, m.W, m.H, scale, {
        edges, edge: { edgePx }, signal,
        onProgress: (d, t) => progress(d / (t + 1), `Rendering ${W} × ${H} — tile ${d} / ${t}`, 'tiles'),
      });
      progress(0.95, `Encoding ${info.label} (${W} × ${H})${info.lossy ? ` · quality ${Math.round((exportQuality ?? 0.92) * 100)}` : ''} …`);
      const enc = await encodeImage(img.rgba, img.width, img.height, info.key, exportQuality ?? 0.92);
      const name = `${stem(s.state.image?.name ?? String(m.meta.source ?? 'model'))}_otsr_x${scale}${edges ? '' : '_raw'}.${enc.ext}`;
      downloadBlob(enc.blob, name);
      m.saved = true;
      s.setStatus(`Exported ${name} (${(enc.blob.size / 1e6).toFixed(1)} MB) in ${fmtSeconds((performance.now() - t0) / 1000)}.`);
      if (enc.fallback) toast(`${info.label} is not supported by this browser — saved as PNG instead.`, 'info', 6000);
      else toast(`Exported ${name}`, 'success');
    });
  }

  /** Back to the start screen: drops the model and the image and clears everything that belonged to them. */
  reset(): void {
    const s = this.store;
    this.vectorView?.release();
    this.dropModel();
    s.state.sourceBuf?.destroy();
    this.viewer.setSource(null);
    this.viewer.roi.set(null);
    this.viewer.layers = { gaussians: false, error: false, vector: false, ot: false };
    this.viewer.markers = [];
    this.viewer.mode = 'bicubic';
    s.set({ image: null, sourceBuf: null, roi: null, lastFit: null, curve: [], live: null, regions: [], monitor: false, log: [], layers: { gaussians: false, error: false, vector: false, ot: false }, viewMode: 'bicubic', errorStats: null, fitEntries: 0, tool: 'pan' });
    s.setStatus('');
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
      if (s.busy) return void toast('Wait for the current job to finish.', 'info', 3000); // a job may have started during the read
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
    if (this.store.busy) throw new Error('Wait for the current job to finish.');
    if (W * H > MAX_PIXELS) throw new Error('too large');
    this.setImage(imageFromRgba8(rgba8 instanceof Uint8ClampedArray ? rgba8 : new Uint8ClampedArray(rgba8), W, H, name));
  }
}
