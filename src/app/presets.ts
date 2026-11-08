// Quality presets and the estimates shown before a fit.

import type { Preset, Prefs } from './store';

export interface PresetValues {
  kPerPx: number;
  steps: number;
  label: string;
  hint: string;
}

export const PRESETS: Record<Exclude<Preset, 'custom'>, PresetValues> = {
  fast: { kPerPx: 0.05, steps: 1500, label: 'Fast', hint: '1 Gaussian per 20 px, 1500 steps — quick look' },
  standard: { kPerPx: 0.1, steps: 3000, label: 'Standard', hint: '1 Gaussian per 10 px, 3000 steps — the paper setting' },
  best: { kPerPx: 0.15, steps: 4000, label: 'Best', hint: '1 Gaussian per 7 px, 4000 steps — fine textures' },
};

/** Gaussian count for an image at a budget (the lattice rounds it slightly). */
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

/** GPU memory of the viewer: the source image, the display rasteriser and its two canvas-sized float buffers. */
export function viewMemoryBytes(N: number, W: number, H: number, canvasPixels: number): number {
  return 16 * W * H + N * (48 + 8 + 4 + 4) + canvasPixels * 16 * 5;
}

/** Seconds a fit will take on this machine, from the measured throughput (Gaussian-steps per second); null if unknown. */
export function fitSecondsEstimate(prefs: Prefs, N: number, steps: number): number | null {
  if (!(prefs.throughput > 0)) return null;
  return (N * steps) / prefs.throughput + 0.5;
}

/** Resolves the effective budget and steps for the current preset. */
export function effectiveSettings(prefs: Prefs): { kPerPx: number; steps: number } {
  if (prefs.preset === 'custom') return { kPerPx: prefs.kPerPx, steps: prefs.steps };
  const p = PRESETS[prefs.preset];
  return { kPerPx: p.kPerPx, steps: p.steps };
}
