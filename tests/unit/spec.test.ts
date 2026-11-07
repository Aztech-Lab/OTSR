// src/core/spec.ts must agree with reference/otsr_ref/spec.py (via tests/golden/spec.json).

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import * as spec from '../../src/core/spec';
import { GOLDEN_DIR } from '../helpers/golden';

const ref = JSON.parse(readFileSync(join(GOLDEN_DIR, 'spec.json'), 'utf8'));

describe('spec constants', () => {
  it('match the Python reference', () => {
    const scalars = ['K_SIGMA', 'Q_CUT', 'ALPHA_SHIFT', 'EPS', 'TILE', 'FLOATS_PER_GAUSSIAN', 'STEPS', 'FINAL_LR_FRAC', 'ADAM_EPS',
      'LOG_S_MIN_PX', 'LOG_S_MAX', 'K_PER_PX', 'SIGMA_REL', 'MIN_SCALE_PX', 'SINKHORN_SHRINK', 'SINKHORN_ITERS_PER_STAGE',
      'SINKHORN_FINAL_ITERS', 'EDGE_PX', 'EDGE_WINDOW', 'EDGE_C0', 'EDGE_C1'] as const;
    for (const k of scalars) expect((spec as any)[k], k).toBeCloseTo(ref[k], 12);
    expect(spec.LR).toEqual(ref.LR);
    expect([...spec.BETAS]).toEqual(ref.BETAS);
    expect([...spec.UV_CLAMP]).toEqual(ref.UV_CLAMP);
    expect([...spec.LOG_W_CLAMP]).toEqual(ref.LOG_W_CLAMP);
    expect([...spec.LUMA]).toEqual(ref.LUMA);
    expect(spec.PARAM_LAYOUT.map(([k, n]) => [k, n])).toEqual(ref.PARAM_LAYOUT);
  });

  it('lrScale and nLongFor match', () => {
    for (const [it, steps, v] of ref.LR_SCALE_SAMPLES as [number, number, number][]) expect(spec.lrScale(it, steps)).toBeCloseTo(v, 12);
    for (const [h, w, v] of ref.N_LONG_SAMPLES as [number, number, number][]) expect(spec.nLongFor(h, w)).toBe(v);
  });

  it('pyRound rounds halves to even like Python', () => {
    expect(spec.pyRound(0.5)).toBe(0);
    expect(spec.pyRound(1.5)).toBe(2);
    expect(spec.pyRound(2.5)).toBe(2);
    expect(spec.pyRound(-0.5)).toBe(0);
    expect(spec.pyRound(-1.5)).toBe(-2);
    expect(spec.pyRound(2.4999)).toBe(2);
    expect(spec.pyRound(2.5001)).toBe(3);
  });
});
