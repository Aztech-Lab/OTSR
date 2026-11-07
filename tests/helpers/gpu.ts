// One device + compiled pipelines per test file (tests/gpu/setup.ts installs navigator.gpu from Dawn).

import { acquireDevice, createPipelines, type GpuContext, type Pipelines } from '../../src/engine';

export interface TestGpu {
  ctx: GpuContext;
  device: GPUDevice;
  pipes: Pipelines;
  errors: string[];
}

let shared: Promise<TestGpu> | null = null;

export function testGpu(): Promise<TestGpu> {
  shared ??= (async () => {
    const errors: string[] = [];
    const gpu = (globalThis as any).navigator?.gpu as GPU | undefined;
    const ctx = await acquireDevice(gpu, { onError: (m) => errors.push(m) });
    const i = ctx.info;
    console.log(`  GPU: ${i.description || i.device} (${i.vendor} ${i.architecture})`);
    const pipes = await createPipelines(ctx.device);
    return { ctx, device: ctx.device, pipes, errors };
  })();
  return shared;
}

/** Runs `fn` inside validation + out-of-memory error scopes and fails on any captured error. */
export async function checked<T>(device: GPUDevice, fn: () => Promise<T>): Promise<T> {
  device.pushErrorScope('out-of-memory');
  device.pushErrorScope('validation');
  let result: T | undefined;
  let failure: unknown;
  try {
    result = await fn();
  } catch (e) {
    failure = e;
  }
  await device.queue.onSubmittedWorkDone();
  const v = await device.popErrorScope();
  const o = await device.popErrorScope();
  // A GPU error explains a wrong result better than the assertion that caught it, so report it first.
  if (v) throw new Error(`validation error: ${v.message}${failure ? `\n(then: ${(failure as Error).message})` : ''}`);
  if (o) throw new Error(`out of memory: ${o.message}`);
  if (failure) throw failure;
  return result as T;
}
