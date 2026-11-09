// The numerical core of the technical report's live strip: exact gradients against central finite differences, a
// short fit from the transported mesh, and the transport map on two simple densities.

import { describe, expect, it } from 'vitest';
import { Fitter, densityGrid, meshGaussians, psnr, sinkhornMap, type Grid } from '../../src/report-fit';

/** A small synthetic picture: a colour gradient, a bright disc, a dark bar and a thin light line. */
function picture(W: number, H: number): Float64Array {
  const rgb = new Float64Array(W * H * 3);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let r = 0.15 + (0.7 * x) / W, g = 0.35 + (0.45 * y) / H, b = 0.8 - (0.6 * x) / W;
    if ((x - 0.3 * W) ** 2 + (y - 0.4 * H) ** 2 < (0.18 * H) ** 2) { r = 0.95; g = 0.8; b = 0.25; }
    if (x > 0.55 * W && x < 0.75 * W && y > 0.2 * H && y < 0.45 * H) { r = 0.08; g = 0.08; b = 0.12; }
    if (Math.abs(y - 0.7 * H - 0.25 * (x - 0.5 * W)) < 0.8) { r = g = b = 0.97; }
    const p = 3 * (y * W + x);
    rgb[p] = r; rgb[p + 1] = g; rgb[p + 2] = b;
  }
  return rgb;
}

describe('report-fit (the live pipeline of the technical report)', () => {
  it('computes the exact gradient of the blend for all nine parameters', () => {
    const W = 14, H = 11;
    const rgb = picture(W, H);
    let seed = 7;
    const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    const N = 7;
    const P = new Float64Array(9 * N);
    for (let k = 0; k < N; k++) {
      const b = 9 * k;
      P[b] = 2 + rnd() * (W - 4);
      P[b + 1] = 2 + rnd() * (H - 4);
      P[b + 2] = Math.log(0.8 + 2 * rnd());
      P[b + 3] = Math.log(0.6 + 1.4 * rnd());
      P[b + 4] = rnd() * 3;
      P[b + 5] = rnd(); P[b + 6] = rnd(); P[b + 7] = rnd();
      P[b + 8] = rnd() - 0.5;
    }
    const f = new Fitter(P, rgb, W, H);
    f.gradient();
    const grad = Float64Array.from(f.grad);
    const scale = Math.max(...grad.map(Math.abs));
    let worst = 0;
    for (let k = 0; k < P.length; k++) {
      const h = 1e-6, p0 = P[k];
      P[k] = p0 + h; const up = f.evaluate();
      P[k] = p0 - h; const down = f.evaluate();
      P[k] = p0;
      worst = Math.max(worst, Math.abs((up - down) / (2 * h) - grad[k]));
    }
    console.log(`gradient: max |analytic − finite difference| = ${worst.toExponential(2)}, max |gradient| = ${scale.toExponential(2)}`);
    expect(scale).toBeGreaterThan(0);
    expect(worst / scale).toBeLessThan(1e-4);
  });

  it('fits a picture from the transported mesh', async () => {
    const W = 40, H = 30;
    const rgb = picture(W, H);
    const grid = densityGrid(rgb, W, H, 4, 0.3);
    const { TX, TY } = await sinkhornMap(grid, Math.max(W, H) ** 2, 36);
    const { mesh, P } = meshGaussians(grid, TX, TY, rgb, W, H, 0.25 * W * H);
    expect(mesh.N).toBe(mesh.gx * mesh.gy);
    expect(P.every(Number.isFinite)).toBe(true);
    const f = new Fitter(P, rgb, W, H);
    const before = psnr(f.evaluate());
    const steps = 200;
    for (let s = 0; s < steps; s++) f.step(steps);
    const after = psnr(f.evaluate());
    console.log(`fit: ${mesh.N} Gaussians, ${before.toFixed(2)} dB → ${after.toFixed(2)} dB after ${steps} steps`);
    expect(before).toBeGreaterThan(14);
    expect(after).toBeGreaterThan(before + 8);
  });

  it('transports a uniform density by the identity and a left-heavy one to the left', async () => {
    const NX = 16, NY = 12, W = 64, H = 48;
    const xs = new Float64Array(NX).map((_, i) => ((i + 0.5) * W) / NX);
    const ys = new Float64Array(NY).map((_, j) => ((j + 0.5) * H) / NY);
    const uniform: Grid = { NX, NY, xs, ys, rho: new Float64Array(NX * NY).fill(1 / (NX * NY)) };
    const id = await sinkhornMap(uniform, W * W, 36);
    let dev = 0;
    for (let k = 0; k < NX * NY; k++) dev = Math.max(dev, Math.abs(id.TX[k] - xs[k % NX]), Math.abs(id.TY[k] - ys[(k / NX) | 0]));
    expect(dev).toBeLessThan(0.5 * (W / NX));
    const rho = new Float64Array(NX * NY);
    for (let k = 0; k < NX * NY; k++) rho[k] = (k % NX) < NX / 2 ? 3 : 1;
    const s = rho.reduce((a, v) => a + v, 0);
    const left = await sinkhornMap({ NX, NY, xs, ys, rho: rho.map((v) => v / s) }, W * W, 36);
    const meanX = left.TX.reduce((a, v) => a + v, 0) / (NX * NY);
    expect(meanX).toBeLessThan(0.45 * W);
    // monotone along x: the map keeps the order of a row
    for (let j = 0; j < NY; j++) for (let i = 1; i < NX; i++) expect(left.TX[j * NX + i]).toBeGreaterThan(left.TX[j * NX + i - 1]);
  });
});
