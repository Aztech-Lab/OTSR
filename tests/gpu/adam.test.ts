// The Adam kernel against torch.optim.Adam (tests/golden/adam.*): three steps with given gradients, including the
// cosine learning-rate schedule, bias correction and the parameter clamps.

import { describe, expect, it } from 'vitest';
import { Fitter, Rasterizer, readBuffer, uploadBuffer } from '../../src/engine';
import { channel, expectClose } from '../helpers/compare';
import { f32, loadGolden } from '../helpers/golden';
import { checked, testGpu } from '../helpers/gpu';

const COLS = ['uv.x', 'uv.y', 'log_s.0', 'log_s.1', 'rot', 'color.r', 'color.g', 'color.b', 'log_w'];

describe('adam kernel vs torch.optim.Adam', () => {
  it('reproduces three steps with clamps', async () => {
    const g = loadGolden('adam');
    const { H, W, N, steps_total, iters } = g.meta as { H: number; W: number; N: number; steps_total: number; iters: number };
    const { device, pipes, ctx, errors } = await testGpu();
    const nf = N * 9;
    const params = uploadBuffer(device, f32(g, 'params0'), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 'params');
    await checked(device, async () => {
      const raster = new Rasterizer(ctx, pipes, params, N, W, H, 16);
      const fitter = new Fitter(ctx, pipes, raster, new Float32Array(W * H * 4));
      const grads = f32(g, 'grads');
      const refP = f32(g, 'params');
      const refM = f32(g, 'm');
      const refV = f32(g, 'v');
      for (let it = 0; it < iters; it++) {
        device.queue.writeBuffer(fitter.grad, 0, grads.subarray(it * nf, (it + 1) * nf));
        const enc = device.createCommandEncoder();
        fitter.encodeAdam(enc, it + 1, steps_total);
        device.queue.submit([enc.finish()]);
        const p = new Float32Array(await readBuffer(device, params, nf * 4));
        const m = new Float32Array(await readBuffer(device, fitter.m, nf * 4));
        const v = new Float32Array(await readBuffer(device, fitter.v, nf * 4));
        console.log(`  step ${it + 1}`);
        for (let k = 0; k < 9; k++) {
          expectClose(`params.${COLS[k]}`, channel(p, 9, k), channel(refP.subarray(it * nf, (it + 1) * nf), 9, k), 2e-6, 1e-6);
        }
        expectClose('m', m, refM.subarray(it * nf, (it + 1) * nf), 0, 1e-5);
        expectClose('v', v, refV.subarray(it * nf, (it + 1) * nf), 0, 1e-5);
      }
      raster.destroy();
      fitter.destroy();
    });
    params.destroy();
    expect(errors).toEqual([]);
  });
});
