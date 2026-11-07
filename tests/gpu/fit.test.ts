// The whole fitting loop against the Python reference (tests/golden/fit.*): 40 steps from the OT initialisation
// of a synthetic image, comparing the PSNR curve and the final parameters.

import { describe, expect, it } from 'vitest';
import { runFit } from '../../src/engine/fit';
import { channel, expectClose } from '../helpers/compare';
import { f32, loadGolden } from '../helpers/golden';
import { checked, testGpu } from '../helpers/gpu';

const COLS = ['uv.x', 'uv.y', 'log_s.0', 'log_s.1', 'rot', 'color.r', 'color.g', 'color.b', 'log_w'];

describe('fit loop vs Python reference', () => {
  it('follows the reference PSNR curve and parameters over 40 steps', async () => {
    const g = loadGolden('fit');
    const { H, W, N, steps, log_steps } = g.meta as { H: number; W: number; N: number; steps: number; log_steps: number[] };
    const { device, pipes, ctx, errors } = await testGpu();
    const gt = f32(g, 'gt');
    const rgba = new Float32Array(W * H * 4);
    for (let i = 0; i < W * H; i++) rgba.set([gt[3 * i], gt[3 * i + 1], gt[3 * i + 2], 1], 4 * i);

    const res = await checked(device, () => runFit(ctx, pipes, f32(g, 'params0'), W, H, rgba, { steps, logEvery: 10, exact: true }));
    expect(res.params.length).toBe(N * 9);
    const refPsnr = f32(g, 'psnr');
    const refLoss = f32(g, 'loss');
    expect(res.curve.map((c) => c.step)).toEqual(log_steps);
    console.log(`  ${N} Gaussians, ${W}x${H}, ${steps} steps in ${(res.seconds * 1000).toFixed(0)} ms, ${res.entries} entries (cap ${res.entryCap})`);
    res.curve.forEach((c, i) => console.log(`  step ${String(c.step).padStart(3)}  PSNR ${c.psnr.toFixed(4)} dB (ref ${refPsnr[i].toFixed(4)})  loss ${c.mse.toExponential(4)} (ref ${refLoss[i].toExponential(4)})`));
    for (let i = 0; i < res.curve.length; i++) expect(Math.abs(res.curve[i].psnr - refPsnr[i])).toBeLessThan(0.02);
    const refP = f32(g, 'params');
    for (let k = 0; k < 9; k++) expectClose(`params.${COLS[k]}`, channel(res.params, 9, k), channel(refP, 9, k), 0, 2e-4);
    expect(errors).toEqual([]);
  });
});
