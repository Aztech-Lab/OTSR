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

/** Rough GPU memory of a fit: ~400 B per Gaussian + 32 B per pixel. */
export function fitMemoryBytes(N: number, W: number, H: number): number {
  return 400 * N + 32 * W * H;
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
