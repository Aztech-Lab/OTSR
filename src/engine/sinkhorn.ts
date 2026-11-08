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

export interface TransportOptions {
  /**
   * Called for every eps-scaling stage (in order, after the loop has finished) with the barycentric map at that
   * stage, same layout as `T`. The maps are projected into extra buffers inside the single submit, so the loop itself
   * runs exactly as without the hook; the final map is unchanged.
   */
  onStage?: (T: Float32Array, stage: number, stages: number, eps: number) => void;
}

/**
 * Barycentric map of the entropic OT plan between the uniform density and `rho` (ny x nx, sums to 1) on the box
 * [0, ex] x [0, ey]. Float32 log-domain Sinkhorn with eps-scaling; the loop runs entirely on the GPU in one submit.
 */
export async function transportMap(ctx: GpuContext, pipes: Pipelines, rho: Float32Array, grid: Grid, ex: number, ey: number, opts: TransportOptions = {}): Promise<TransportResult> {
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
  // barycentric projection at eps `e` into three output buffers (lse, x, y)
  type Outs = { lse: GPUBuffer; x: GPUBuffer; y: GPUBuffer };
  const project = (enc: GPUCommandEncoder, e: number, o: Outs): void => {
    const uX0 = uniform(e, 0, 0, 0, 0);
    const uX1 = uniform(e, 1, 0, 0, 0);
    const uY0 = uniform(e, 0, 0, 1, 0);
    const uY1 = uniform(e, 0, 1, 1, 0);
    runPass(enc, pipes.lseX, bindGroup(device, pipes.lseX, { 0: uX0, 1: g, 2: inner }), dispatch);
    runPass(enc, pipes.lseX, bindGroup(device, pipes.lseX, { 0: uX1, 1: g, 2: innerX }), dispatch);
    runPass(enc, pipes.lseY, bindGroup(device, pipes.lseY, { 0: uY0, 1: inner, 2: o.lse, 3: bLmu }), dispatch);
    runPass(enc, pipes.lseY, bindGroup(device, pipes.lseY, { 0: uY0, 1: innerX, 2: o.x, 3: bLmu }), dispatch);
    runPass(enc, pipes.lseY, bindGroup(device, pipes.lseY, { 0: uY1, 1: inner, 2: o.y, 3: bLmu }), dispatch);
  };
  const readMap = async (o: Outs): Promise<Float32Array> => {
    const [lse, lx, ly] = await Promise.all([readBuffer(device, o.lse, n * 4), readBuffer(device, o.x, n * 4), readBuffer(device, o.y, n * 4)]);
    const L = new Float32Array(lse);
    const X = new Float32Array(lx);
    const Y = new Float32Array(ly);
    const T = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      T[2 * i] = Math.exp(X[i] - L[i]);
      T[2 * i + 1] = Math.exp(Y[i] - L[i]);
    }
    return T;
  };
  const finalOuts: Outs = { lse: outLse, x: outX, y: outY };
  // with the stage hook, every intermediate stage gets its own output buffers inside the same submit
  const stageOuts: Outs[] = opts.onStage
    ? stages.slice(0, -1).map((_, i) => ({ lse: createBuffer(device, n * 4, S | GPUBufferUsage.COPY_SRC, `ot.stage${i}.lse`), x: createBuffer(device, n * 4, S | GPUBufferUsage.COPY_SRC, `ot.stage${i}.x`), y: createBuffer(device, n * 4, S | GPUBufferUsage.COPY_SRC, `ot.stage${i}.y`) }))
    : [];
  const enc = device.createCommandEncoder({ label: 'sinkhorn' });
  let iters = 0;
  for (let si = 0; si < stages.length; si++) {
    const st = stages[si];
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
    if (si < stageOuts.length) project(enc, st.e, stageOuts[si]);
  }
  const e = stages[stages.length - 1].e;
  project(enc, e, finalOuts);
  device.queue.submit([enc.finish()]);
  const [T, ...stageMaps] = await Promise.all([readMap(finalOuts), ...stageOuts.map((o) => readMap(o))]);
  if (opts.onStage) {
    stageMaps.forEach((m, i) => opts.onStage!(m, i, stages.length, stages[i].e));
    opts.onStage(T, stages.length - 1, stages.length, e);
  }
  for (const b of [bLnu, bLmu, f, g, inner, innerX, outLse, outX, outY, ...uniforms, ...stageOuts.flatMap((o) => [o.lse, o.x, o.y])]) b.destroy();
  return { T, eps: e, iters };
}
