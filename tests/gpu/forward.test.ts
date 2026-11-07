// The rasteriser against the dense float64 Python reference (tests/golden/render.*):
// projection terms, the native render, a zoomed / offset view, the MSE gradient of every parameter.

import { describe, expect, it } from 'vitest';
import { Fitter, Rasterizer, readBuffer, uploadBuffer } from '../../src/engine';
import { FLOATS_PER_GAUSSIAN } from '../../src/core/spec';
import { channel, expectClose } from '../helpers/compare';
import { f32, loadGolden } from '../helpers/golden';
import { checked, testGpu } from '../helpers/gpu';

const PARAM_NAMES = ['uv.x', 'uv.y', 'log_s.0', 'log_s.1', 'rot', 'color.r', 'color.g', 'color.b', 'log_w'];

describe('rasteriser vs Python reference', () => {
  it('reproduces projection, render, view and gradients', async () => {
    const g = loadGolden('render');
    const { H, W, N } = g.meta as { H: number; W: number; N: number };
    const { device, pipes, ctx, errors } = await testGpu();
    const params = uploadBuffer(device, f32(g, 'params'), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 'params');

    await checked(device, async () => {
      // --- native raster ---------------------------------------------------------------------------------------
      const raster = new Rasterizer(ctx, pipes, params, N, W, H, 4096);
      const fitter = new Fitter(ctx, pipes, raster, rgbaOf(f32(g, 'gt'), W * H));
      raster.setView({ srcW: W, srcH: H, zoom: 1, offsetX: 0, offsetY: 0, aa: 0, outW: W, outH: H }, fitter.gradScale);
      let enc = device.createCommandEncoder();
      raster.encodeBuild(enc);
      device.queue.submit([enc.finish()]);
      const total = await raster.readTotalEntries();
      console.log(`  ${N} Gaussians, ${raster.nTiles} tiles, ${total} tile entries`);
      if (total > raster.entryCap) raster.growEntries(Math.ceil(total * 1.5));
      enc = device.createCommandEncoder();
      raster.encodeBuild(enc);
      raster.encodeForward(enc);
      fitter.encodeBackward(enc);
      fitter.encodeLoss(enc);
      device.queue.submit([enc.finish()]);

      // projection terms: (mx, my, a, b, d, w) per Gaussian
      const splats = new Float32Array(await readBuffer(device, raster.splats, N * 48));
      const terms = f32(g, 'terms');
      const gotTerms = new Float64Array(N * 6);
      for (let i = 0; i < N; i++) {
        gotTerms.set([splats[12 * i], splats[12 * i + 1], splats[12 * i + 2], splats[12 * i + 3], splats[12 * i + 4], splats[12 * i + 5]], 6 * i);
      }
      for (let c = 0; c < 6; c++) expectClose(`terms.${['mx', 'my', 'a', 'b', 'd', 'w'][c]}`, channel(gotTerms, 6, c), channel(terms, 6, c), 0, 2e-6);

      // tile coverage must contain every pixel of the 4-sigma box (support)
      const gcount = new Uint32Array(await readBuffer(device, raster.gcount, N * 4));
      let covered = 0;
      for (let i = 0; i < N; i++) covered += gcount[i];
      expect(covered).toBe(total);

      // render: rgb and D
      const out = new Float32Array(await readBuffer(device, raster.out, W * H * 16));
      const refOut = f32(g, 'out');
      for (let c = 0; c < 3; c++) expectClose(`out.${'rgb'[c]}`, channel(out, 4, c), channel(refOut, 3, c), 2e-5, 2e-6);
      expectClose('D', channel(out, 4, 3), f32(g, 'D'), 0, 2e-6);

      // loss
      const loss = await fitter.readLoss();
      const refLoss = f32(g, 'loss')[0];
      console.log(`  loss ${loss.mseRaw.toExponential(6)} (ref ${refLoss.toExponential(6)}), PSNR of clamped render ${loss.psnr.toFixed(3)} dB`);
      expect(Math.abs(loss.mseRaw - refLoss)).toBeLessThan(1e-6 * refLoss + 1e-9);

      // gradients, per parameter column: max |diff| <= 1e-4 * max |ref| of that column
      const grad = new Float32Array(await readBuffer(device, fitter.grad, N * FLOATS_PER_GAUSSIAN * 4));
      const refGrad = f32(g, 'grad');
      for (let k = 0; k < FLOATS_PER_GAUSSIAN; k++) {
        expectClose(`grad.${PARAM_NAMES[k]}`, channel(grad, 9, k), channel(refGrad, 9, k), 0, 1e-4);
      }

      // --- zoomed, offset view -------------------------------------------------------------------------------
      const v = g.meta.view as { zoom: number; tx: number; ty: number; out_w: number; out_h: number };
      const rv = new Rasterizer(ctx, pipes, params, N, v.out_w, v.out_h, 1 << 15);
      rv.setView({ srcW: W, srcH: H, zoom: v.zoom, offsetX: v.tx, offsetY: v.ty, aa: 0, outW: v.out_w, outH: v.out_h });
      enc = device.createCommandEncoder();
      rv.encodeBuild(enc);
      rv.encodeForward(enc);
      device.queue.submit([enc.finish()]);
      expect(await rv.readTotalEntries()).toBeLessThanOrEqual(rv.entryCap);
      const outV = new Float32Array(await readBuffer(device, rv.out, v.out_w * v.out_h * 16));
      const refV = f32(g, 'out_view');
      for (let c = 0; c < 3; c++) expectClose(`view.${'rgb'[c]}`, channel(outV, 4, c), channel(refV, 3, c), 2e-5, 2e-6);

      raster.destroy();
      fitter.destroy();
      rv.destroy();
    });
    params.destroy();
    expect(errors).toEqual([]);
  });
});

function rgbaOf(rgb: Float32Array, pixels: number): Float32Array {
  const out = new Float32Array(pixels * 4);
  for (let i = 0; i < pixels; i++) {
    out[4 * i] = rgb[3 * i];
    out[4 * i + 1] = rgb[3 * i + 1];
    out[4 * i + 2] = rgb[3 * i + 2];
    out[4 * i + 3] = 1;
  }
  return out;
}
