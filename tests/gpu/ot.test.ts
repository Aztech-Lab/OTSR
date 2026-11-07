// The OT initialisation against the Python reference (tests/golden/ot.* and ot_tall.*): the pooled density
// (CPU), the transport map (GPU Sinkhorn) and the resulting Gaussian parameters.
//
// Both the torch float32 solver and the WGSL one lose ~1e-3 of absolute precision at the final eps (the exponents
// are ~1e4), so the map is judged against the float64 solution: the GPU map must be at least as close to it as the
// float32 reference is, and the parameters derived from it are compared with matching tolerances.

import { describe, expect, it } from 'vitest';
import { otInit } from '../../src/engine';
import { channel, expectClose, report, stats } from '../helpers/compare';
import { f32, loadGolden } from '../helpers/golden';
import { checked, testGpu } from '../helpers/gpu';

for (const name of ['ot', 'ot_tall']) {
  describe(`ot init vs Python reference (${name})`, () => {
    it('reproduces density, transport map and parameters', async () => {
      const g = loadGolden(name);
      const m = g.meta as { H: number; W: number; K: number | null; n_long: number; gy: number; gx: number; lam: number; eps: number; iters: number };
      const { ctx, pipes, device, errors } = await testGpu();
      const res = await checked(device, () => otInit(ctx, pipes, f32(g, 'gt'), m.W, m.H, { K: m.K ?? undefined, nLong: m.n_long }));
      console.log(`  ${name}: ${m.W}x${m.H}, grid ${res.grid.ny}x${res.grid.nx}, lattice ${res.gy}x${res.gx}, lam ${res.lam.toFixed(4)}, ${res.iters} iterations, eps ${res.eps.toExponential(3)}; density ${res.ms.density.toFixed(1)} ms, transport ${res.ms.transport.toFixed(1)} ms, params ${res.ms.params.toFixed(1)} ms`);
      expect([res.gy, res.gx]).toEqual([m.gy, m.gx]);
      expect(res.lam).toBeCloseTo(m.lam, 12);
      expect(res.iters).toBe(m.iters);
      expect(res.eps).toBeCloseTo(m.eps, 9);
      expectClose('rho', res.rho, f32(g, 'rho'), 1e-9, 1e-6);

      // transport map: GPU float32 and torch float32, each against float64
      const T64 = f32(g, 'T64');
      const T32 = f32(g, 'T');
      const gpu = report('T gpu  vs float64', stats(res.T, T64));
      const ref = report('T ref  vs float64', stats(T32, T64));
      report('T gpu  vs ref', stats(res.T, T32));
      expect(gpu.maxAbs).toBeLessThanOrEqual(1.5 * ref.maxAbs + 1e-5);
      expect(gpu.rms).toBeLessThanOrEqual(1.5 * ref.rms + 1e-6);

      // parameters: tolerances follow from the map noise (uv ~ T; shapes ~ finite differences of T over 2 cells)
      const refP = f32(g, 'params');
      const tolUv = 3 * Math.max(gpu.maxAbs, ref.maxAbs);
      const cells = Math.max(res.grid.nx, res.grid.ny);
      expectClose('uv.x', channel(res.params, 9, 0), channel(refP, 9, 0), tolUv, 0);
      expectClose('uv.y', channel(res.params, 9, 1), channel(refP, 9, 1), tolUv, 0);
      expectClose('log_s.0', channel(res.params, 9, 2), channel(refP, 9, 2), tolUv * cells, 0);
      expectClose('log_s.1', channel(res.params, 9, 3), channel(refP, 9, 3), tolUv * cells, 0);
      const refCov = f32(g, 'cov');
      const cov = new Float64Array(res.count * 3);
      for (let i = 0; i < res.count; i++) {
        const s1 = Math.exp(2 * res.params[9 * i + 2]);
        const s2 = Math.exp(2 * res.params[9 * i + 3]);
        const c = Math.cos(res.params[9 * i + 4]);
        const s = Math.sin(res.params[9 * i + 4]);
        cov[3 * i] = c * c * s1 + s * s * s2;
        cov[3 * i + 1] = c * s * (s2 - s1);
        cov[3 * i + 2] = s * s * s1 + c * c * s2;
      }
      for (let k = 0; k < 3; k++) expectClose(`cov.${['xx', 'xy', 'yy'][k]}`, channel(cov, 3, k), channel(refCov, 3, k), 0, 2 * tolUv * cells);
      // colours are bilinear samples at the centres: |dc| <= |du| W Gx + |dv| H Gy with Gx, Gy the largest
      // neighbour differences of the image (the Lipschitz bound of the interpolant)
      const gt = f32(g, 'gt');
      let Gx = 0;
      let Gy = 0;
      for (let y = 0; y < m.H; y++) {
        for (let x = 0; x < m.W; x++) {
          for (let c = 0; c < 3; c++) {
            const i = 3 * (y * m.W + x) + c;
            if (x + 1 < m.W) Gx = Math.max(Gx, Math.abs(gt[i + 3] - gt[i]));
            if (y + 1 < m.H) Gy = Math.max(Gy, Math.abs(gt[i + 3 * m.W] - gt[i]));
          }
        }
      }
      let worst = 0;
      let worstAt = '';
      let maxDc = 0;
      for (let i = 0; i < res.count; i++) {
        const du = (res.params[9 * i] - refP[9 * i]) * m.W;
        const dv = (res.params[9 * i + 1] - refP[9 * i + 1]) * m.H;
        const bound = Math.abs(du) * Gx + Math.abs(dv) * Gy + 2e-5; // slack: torch samples with float32 coordinates (~1e-5 px)
        for (let c = 0; c < 3; c++) {
          const dc = Math.abs(res.params[9 * i + 5 + c] - refP[9 * i + 5 + c]);
          maxDc = Math.max(maxDc, dc);
          if (dc / bound > worst) {
            worst = dc / bound;
            worstAt = `#${i} ch${c}: uv_px gpu (${(res.params[9 * i] * m.W).toFixed(4)}, ${(res.params[9 * i + 1] * m.H).toFixed(4)}) ref (${(refP[9 * i] * m.W).toFixed(4)}, ${(refP[9 * i + 1] * m.H).toFixed(4)}), dc ${dc.toExponential(3)}, bound ${bound.toExponential(3)}`;
          }
        }
      }
      console.log(`  color: max |dc| ${maxDc.toExponential(2)}, worst |dc| / Lipschitz bound ${worst.toFixed(3)} (Gx ${Gx.toFixed(3)}, Gy ${Gy.toFixed(3)}) at ${worstAt}`);
      expect(worst).toBeLessThanOrEqual(1);
      expectClose('log_w', channel(res.params, 9, 8), channel(refP, 9, 8), 0, 0);
      expect(errors).toEqual([]);
    });
  });
}
