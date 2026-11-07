// The v3 edge pass against reference/otsr_ref/edges.py (tests/golden/edges.*): three magnifications / widths.

import { describe, expect, it } from 'vitest';
import { EdgePass, createBuffer, readBuffer, uploadBuffer } from '../../src/engine';
import { channel, expectClose } from '../helpers/compare';
import { f32, loadGolden } from '../helpers/golden';
import { checked, testGpu } from '../helpers/gpu';

describe('edge pass vs Python reference', () => {
  it('reproduces edge_field for every golden run', async () => {
    const g = loadGolden('edges');
    const { H, W, runs } = g.meta as { H: number; W: number; runs: { S: number; edge_px: number }[] };
    const { ctx, pipes, device, errors } = await testGpu();
    const img = f32(g, 'img');
    const rgba = new Float32Array(W * H * 4);
    for (let i = 0; i < W * H; i++) rgba.set([img[3 * i], img[3 * i + 1], img[3 * i + 2], 1], 4 * i);
    await checked(device, async () => {
      const src = uploadBuffer(device, rgba, GPUBufferUsage.STORAGE, 'img');
      const dst = createBuffer(device, W * H * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'out');
      const pass = new EdgePass(ctx, pipes, W, H);
      for (let k = 0; k < runs.length; k++) {
        const enc = device.createCommandEncoder();
        pass.encode(enc, src, dst, runs[k].S, { edgePx: runs[k].edge_px });
        const t0 = performance.now();
        device.queue.submit([enc.finish()]);
        await device.queue.onSubmittedWorkDone();
        console.log(`  run ${k}: S=${runs[k].S} edge_px=${runs[k].edge_px} (${W}x${H}) ${(performance.now() - t0).toFixed(1)} ms`);
        const out = new Float32Array(await readBuffer(device, dst, W * H * 16));
        const ref = f32(g, `out${k}`);
        for (let c = 0; c < 3; c++) expectClose(`out${k}.${'rgb'[c]}`, channel(out, 4, c), channel(ref, 3, c), 2e-6, 0);
      }
      pass.destroy();
      src.destroy();
      dst.destroy();
    });
    expect(errors).toEqual([]);
  });
});
