// Quality tiers and the estimates shown before a fit.

import { MAX_PIXELS } from './image';
import type { AppState, Preset, Prefs } from './store';

export type Tier = Exclude<Preset, 'custom'>;

export interface PresetValues {
  kPerPx: number;
  steps: number;
  /** English and Traditional Chinese name. */
  label: string;
  zh: string;
  hint: string;
}

/** The tiers from the quickest to the finest: the order of every tier list in the UI. */
export const TIERS: readonly Tier[] = ['turbo', 'fast', 'standard', 'best'];

/**
 * Turbo keeps the budget of Fast and cuts the steps: on the RTX 4090 the sample (1024 x 815) fits in 0.11 s at 40.5 dB
 * (Fast: 0.53 s, 43.4 dB) and a 2048 x 2048 anime frame in 0.53 s at 49.0 dB (Fast: 2.17 s, 50.1 dB); halving the
 * budget instead loses more and gains little, because a step costs mostly per pixel. Turbo also skips the fit
 * animation, whose fixed timeline (about four seconds) would otherwise be most of the wait.
 */
export const PRESETS: Record<Tier, PresetValues> = {
  turbo: { kPerPx: 0.05, steps: 400, label: 'Turbo', zh: '極速', hint: '1 Gaussian per 20 px, 400 steps, no fit animation — about 4× quicker than Fast' },
  fast: { kPerPx: 0.05, steps: 1500, label: 'Fast', zh: '快速', hint: '1 Gaussian per 20 px, 1500 steps — quick look' },
  standard: { kPerPx: 0.1, steps: 3000, label: 'Standard', zh: '標準', hint: '1 Gaussian per 10 px, 3000 steps — the paper setting' },
  best: { kPerPx: 0.15, steps: 4000, label: 'Best', zh: '最佳', hint: '1 Gaussian per 7 px, 4000 steps — fine textures' },
};

/** Gaussian count for an image at a budget (the mesh rounds it slightly). */
export function gaussianCount(kPerPx: number, W: number, H: number): number {
  return Math.round(kPerPx * W * H);
}

/**
 * GPU memory of a fit, from the engine's allocations: parameters, splats, rects, counts (raster), gradient and the two
 * Adam moments (fitter), the W x H float output and target, and the tile-entry arrays (list, slot and the 48-byte
 * gradient partials) at 1.5 x the entry count + 1024. `entries` defaults to 4 per Gaussian (typical at 1/10 px).
 */
export function fitMemoryBytes(N: number, W: number, H: number, entries = 4 * N): number {
  const cap = Math.ceil(entries * 1.5) + 1024;
  const tiles = Math.ceil(W / 16) * Math.ceil(H / 16);
  return N * (36 + 48 + 8 + 4 + 4 + 3 * 36) + W * H * (16 + 16) + cap * (4 + 4 + 48) + tiles * 12;
}

/** The largest single GPU buffer of a fit of P pixels at a budget: the gradient partials (48 bytes per tile entry,
 * 1.5 x 4 entries per Gaussian) or the float output / target image (16 bytes per pixel). */
export function largestFitBuffer(kPerPx: number, P: number): number {
  return Math.max(48 * (Math.ceil(1.5 * 4 * kPerPx * P) + 1024), 16 * P);
}

/** The largest image (pixels) one fit at a budget can take within the device's buffer limits (and the app's cap). */
export function maxFitPixels(kPerPx: number, maxBinding: number, maxBuffer: number): number {
  const limit = Math.min(maxBinding, maxBuffer);
  let lo = 0, hi = MAX_PIXELS;
  if (largestFitBuffer(kPerPx, hi) <= limit) return hi;
  while (hi - lo > 1000) { const mid = (lo + hi) / 2; if (largestFitBuffer(kPerPx, mid) <= limit) lo = mid; else hi = mid; }
  return lo;
}

/** GPU memory of the viewer: the source image, the display rasteriser and its two canvas-sized float buffers. */
export function viewMemoryBytes(N: number, W: number, H: number, canvasPixels: number): number {
  return 16 * W * H + N * (48 + 8 + 4 + 4) + canvasPixels * 16 * 5;
}

/** The GPU memory of the open image or model: the fit at the current tier (exact once a fit has counted its tile
 * entries) plus the viewer at `canvasPixels`. */
export function sessionMemory(st: AppState, canvasPixels: number): { bytes: number; exact: boolean; fit: number; view: number; N: number } | null {
  const base = st.image ?? st.model;
  if (!base) return null;
  const { kPerPx } = effectiveSettings(st.prefs);
  const N = st.model ? st.model.N : gaussianCount(kPerPx, base.W, base.H);
  const fit = fitMemoryBytes(N, base.W, base.H, st.fitEntries || undefined);
  const view = viewMemoryBytes(N, base.W, base.H, canvasPixels);
  return { bytes: fit + view, exact: !!st.fitEntries, fit, view, N };
}

/** Seconds a fit will take on this machine, from the measured throughput (Gaussian-steps per second); null if unknown. */
export function fitSecondsEstimate(prefs: Prefs, N: number, steps: number): number | null {
  if (!(prefs.throughput > 0)) return null;
  return (N * steps) / prefs.throughput + 0.5;
}

/** Resolves the effective budget and steps for the current preset. */
export function effectiveSettings(prefs: Prefs): { kPerPx: number; steps: number } {
  if (prefs.preset === 'custom') return { kPerPx: prefs.kPerPx, steps: prefs.steps };
  const p = PRESETS[prefs.preset] ?? PRESETS.standard;
  return { kPerPx: p.kPerPx, steps: p.steps };
}

/** Whether a fit with these preferences plays the fit animation (Turbo never does: it is about the wait). */
export function fitAnimates(prefs: Prefs): boolean {
  return prefs.preset !== 'turbo' && prefs.fitAnim !== false && prefs.motion !== false && !matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/** Steps as the tier lists show them: 400, 1.5k, 3k. */
export function fmtSteps(n: number): string {
  return n >= 1000 ? `${+(n / 1000).toFixed(1)}k` : String(n);
}
