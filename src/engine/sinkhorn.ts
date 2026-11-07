// GPU entropic optimal transport from the uniform density to rho (reference/otsr_ref/init.py transport_map).

import type { Grid } from '../core/ot_cpu';
import { sinkhornSchedule } from '../core/ot_cpu';
import { SINKHORN_FINAL_ITERS, SINKHORN_ITERS_PER_STAGE, SINKHORN_SHRINK } from '../core/spec';
import type { GpuContext } from './device';
import type { Pipelines } from './pipelines';
import { bindGroup, createBuffer, grid1d, readBuffer, runPass, uploadBuffer } from './gpu';

const OT_BYTES = 48;

function packOt(grid: Grid, ex: number, ey: number, e: number, wx: number, wy: number, A: number, B: number): ArrayBuffer {
  const ab = new ArrayBuffer(OT_BYTES);
  const u = new Uint32Array(ab);
  const f = new Float32Array(ab);
  u[0] = grid.nx;
  u[1] = grid.ny;
  u[2] = wx;
  u[3] = wy;
  f[4] = ex;
  f[5] = ey;
  f[6] = e;
  f[7] = A;
  f[8] = B;
  return ab;
}

export interface TransportResult {
  /** (ny, nx, 2): (x, y) in box units at the grid cell centres. */
  T: Float32Array;
  eps: number;
  iters: number;
}

/**
 * Barycentric map of the entropic OT plan between the uniform density and `rho` (ny x nx, sums to 1) on the box
 * [0, ex] x [0, ey]. Float32 log-domain Sinkhorn with eps-scaling; the loop runs entirely on the GPU in one submit.
 */
export async function transportMap(ctx: GpuContext, pipes: Pipelines, rho: Float32Array, grid: Grid, ex: number, ey: number): Promise<TransportResult> {
  const device = ctx.device;
  const n = grid.nx * grid.ny;
  if (rho.length !== n) throw new Error('rho does not match the grid');
  const lnu = new Float32Array(n);
  const lmu = new Float32Array(n).fill(-Math.log(n));
  let sum = 0;
  for (let i = 0; i < n; i++) sum += Math.max(rho[i], 0) + 1e-12;
  for (let i = 0; i < n; i++) lnu[i] = Math.log((Math.max(rho[i], 0) + 1e-12) / sum);
  const S = GPUBufferUsage.STORAGE;
  const bLnu = uploadBuffer(device, lnu, S, 'ot.lnu');
  const bLmu = uploadBuffer(device, lmu, S, 'ot.lmu');
  const f = createBuffer(device, n * 4, S, 'ot.f');
  const g = createBuffer(device, n * 4, S, 'ot.g');
  const inner = createBuffer(device, n * 4, S, 'ot.inner');
  const innerX = createBuffer(device, n * 4, S, 'ot.innerX');
  const outLse = createBuffer(device, n * 4, S | GPUBufferUsage.COPY_SRC, 'ot.lse');
  const outX = createBuffer(device, n * 4, S | GPUBufferUsage.COPY_SRC, 'ot.lseX');
  const outY = createBuffer(device, n * 4, S | GPUBufferUsage.COPY_SRC, 'ot.lseY');
  const uniforms: GPUBuffer[] = [];
  const uniform = (e: number, wx: number, wy: number, A: number, B: number): GPUBuffer => {
    const b = device.createBuffer({ size: OT_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, packOt(grid, ex, ey, e, wx, wy, A, B));
    uniforms.push(b);
    return b;
  };
  const dispatch = grid1d(n);
  const stages = sinkhornSchedule(grid, ex, ey, SINKHORN_SHRINK, SINKHORN_ITERS_PER_STAGE, SINKHORN_FINAL_ITERS);
  const enc = device.createCommandEncoder({ label: 'sinkhorn' });
  let iters = 0;
  for (const st of stages) {
    const uX = uniform(st.e, 0, 0, 0, 0);
    const uF = uniform(st.e, 0, 0, -st.e, st.e);
    const bgXg = bindGroup(device, pipes.lseX, { 0: uX, 1: g, 2: inner }, 'lse_x(g)');
    const bgYf = bindGroup(device, pipes.lseY, { 0: uF, 1: inner, 2: f, 3: bLmu }, 'f = e lmu - e lse');
    const bgXf = bindGroup(device, pipes.lseX, { 0: uX, 1: f, 2: inner }, 'lse_x(f)');
    const bgYg = bindGroup(device, pipes.lseY, { 0: uF, 1: inner, 2: g, 3: bLnu }, 'g = e lnu - e lse');
    for (let k = 0; k < st.iters; k++) {
      runPass(enc, pipes.lseX, bgXg, dispatch);
      runPass(enc, pipes.lseY, bgYf, dispatch);
      runPass(enc, pipes.lseX, bgXf, dispatch);
      runPass(enc, pipes.lseY, bgYg, dispatch);
      iters++;
    }
  }
  // barycentric projection at the final eps
  const e = stages[stages.length - 1].e;
  const uX0 = uniform(e, 0, 0, 0, 0);
  const uX1 = uniform(e, 1, 0, 0, 0);
  const uY0 = uniform(e, 0, 0, 1, 0);
  const uY1 = uniform(e, 0, 1, 1, 0);
  runPass(enc, pipes.lseX, bindGroup(device, pipes.lseX, { 0: uX0, 1: g, 2: inner }), dispatch);
  runPass(enc, pipes.lseX, bindGroup(device, pipes.lseX, { 0: uX1, 1: g, 2: innerX }), dispatch);
  runPass(enc, pipes.lseY, bindGroup(device, pipes.lseY, { 0: uY0, 1: inner, 2: outLse, 3: bLmu }), dispatch);
  runPass(enc, pipes.lseY, bindGroup(device, pipes.lseY, { 0: uY0, 1: innerX, 2: outX, 3: bLmu }), dispatch);
  runPass(enc, pipes.lseY, bindGroup(device, pipes.lseY, { 0: uY1, 1: inner, 2: outY, 3: bLmu }), dispatch);
  device.queue.submit([enc.finish()]);
  const [lse, lx, ly] = await Promise.all([readBuffer(device, outLse, n * 4), readBuffer(device, outX, n * 4), readBuffer(device, outY, n * 4)]);
  const L = new Float32Array(lse);
  const X = new Float32Array(lx);
  const Y = new Float32Array(ly);
  const T = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    T[2 * i] = Math.exp(X[i] - L[i]);
    T[2 * i + 1] = Math.exp(Y[i] - L[i]);
  }
  for (const b of [bLnu, bLmu, f, g, inner, innerX, outLse, outX, outY, ...uniforms]) b.destroy();
  return { T, eps: e, iters };
}
