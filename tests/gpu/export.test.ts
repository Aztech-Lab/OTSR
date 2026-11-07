// Tiled export must equal an untiled render (the tiles overlap by the edge window), and the classic resamplers
// must reproduce exact values at integer positions.

import { describe, expect, it } from 'vitest';
import { EdgePass, Rasterizer, Resampler, createBuffer, readBuffer, renderImage, uploadBuffer } from '../../src/engine';
import { f32, loadGolden } from '../helpers/golden';
import { checked, testGpu } from '../helpers/gpu';

describe('tiled export', () => {
  it('matches an untiled render with edges', async () => {
    const g = loadGolden('fit');
    const { H, W, N } = g.meta as { H: number; W: number; N: number };
    const { ctx, pipes, device, errors } = await testGpu();
    const S = 3;
    const params = uploadBuffer(device, f32(g, 'params'), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'params');
    await checked(device, async () => {
      const tiled = await renderImage(ctx, pipes, params, N, W, H, S, { tile: 256, edges: true });
      expect([tiled.width, tiled.height]).toEqual([W * S, H * S]);
      const raster = new Rasterizer(ctx, pipes, params, N, W * S, H * S, 1 << 16);
      raster.setView({ srcW: W, srcH: H, zoom: S, offsetX: 0, offsetY: 0, aa: 0, outW: W * S, outH: H * S });
      const dst = createBuffer(device, W * S * H * S * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const edge = new EdgePass(ctx, pipes, W * S, H * S);
      const enc = device.createCommandEncoder();
      raster.encodeBuild(enc);
      raster.encodeForward(enc);
      edge.encode(enc, raster.out, dst, S);
      device.queue.submit([enc.finish()]);
      expect(await raster.readTotalEntries()).toBeLessThanOrEqual(raster.entryCap);
      const full = new Float32Array(await readBuffer(device, dst, W * S * H * S * 16));
      let maxDiff = 0;
      for (let i = 0; i < W * S * H * S; i++) {
        for (let c = 0; c < 3; c++) {
          const ref = Math.min(255, Math.max(0, Math.round(full[4 * i + c] * 255)));
          maxDiff = Math.max(maxDiff, Math.abs(tiled.rgba[4 * i + c] - ref));
        }
      }
      console.log(`  ${tiled.width}x${tiled.height} export in 256-px tiles vs untiled: max 8-bit difference ${maxDiff}`);
      expect(maxDiff).toBeLessThanOrEqual(1); // float rounding at the .5 boundaries only
      raster.destroy();
      edge.destroy();
      dst.destroy();
    });
    params.destroy();
    expect(errors).toEqual([]);
  });
});

describe('classic resamplers', () => {
  it('reproduce the source at zoom 1 and interpolate at zoom 2', async () => {
    const g = loadGolden('fit');
    const { H, W } = g.meta as { H: number; W: number };
    const { ctx, pipes, device, errors } = await testGpu();
    const gt = f32(g, 'gt');
    const rgba = new Float32Array(W * H * 4);
    for (let i = 0; i < W * H; i++) rgba.set([gt[3 * i], gt[3 * i + 1], gt[3 * i + 2], 1], 4 * i);
    await checked(device, async () => {
      const src = uploadBuffer(device, rgba, GPUBufferUsage.STORAGE);
      const rs = new Resampler(ctx, pipes);
      for (const mode of ['nearest', 'bilinear', 'bicubic'] as const) {
        const dst = createBuffer(device, W * H * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        const enc = device.createCommandEncoder();
        rs.encode(enc, src, dst, { srcW: W, srcH: H, zoom: 1, offsetX: 0, offsetY: 0, aa: 0, outW: W, outH: H }, mode);
        device.queue.submit([enc.finish()]);
        const out = new Float32Array(await readBuffer(device, dst, W * H * 16));
        let maxDiff = 0;
        for (let i = 0; i < W * H; i++) for (let c = 0; c < 3; c++) maxDiff = Math.max(maxDiff, Math.abs(out[4 * i + c] - rgba[4 * i + c]));
        console.log(`  ${mode} at zoom 1: max |diff| ${maxDiff.toExponential(2)}`);
        expect(maxDiff).toBeLessThan(1e-6);
        dst.destroy();
      }
      // zoom 2 bilinear: output pixel (1, 1) sits at source (0.25, 0.25) -> a known blend of the 2x2 corner block
      const dst = createBuffer(device, W * 2 * H * 2 * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const enc = device.createCommandEncoder();
      rs.encode(enc, src, dst, { srcW: W, srcH: H, zoom: 2, offsetX: 0, offsetY: 0, aa: 0, outW: 2 * W, outH: 2 * H }, 'bilinear');
      device.queue.submit([enc.finish()]);
      const out = new Float32Array(await readBuffer(device, dst, W * 2 * H * 2 * 16));
      const at = (x: number, y: number, c: number) => rgba[4 * (y * W + x) + c];
      for (let c = 0; c < 3; c++) {
        const expected = 0.75 * (0.75 * at(0, 0, c) + 0.25 * at(1, 0, c)) + 0.25 * (0.75 * at(0, 1, c) + 0.25 * at(1, 1, c));
        expect(out[4 * (1 * 2 * W + 1) + c]).toBeCloseTo(expected, 5);
      }
      dst.destroy();
      rs.destroy();
      src.destroy();
    });
    expect(errors).toEqual([]);
  });
});
