// Small WebGPU helpers shared by the engine.

export const WG = 256; // threads per 1-D workgroup (matches @workgroup_size(256) in the kernels)
export const MAX_WG_DIM = 65535;

function roundUp(n: number, to: number): number {
  return Math.ceil(n / to) * to;
}

export function createBuffer(device: GPUDevice, byteLength: number, usage: GPUBufferUsageFlags, label?: string): GPUBuffer {
  return device.createBuffer({ label, size: Math.max(16, roundUp(byteLength, 16)), usage });
}

export function uploadBuffer(device: GPUDevice, data: ArrayBufferView, usage: GPUBufferUsageFlags, label?: string): GPUBuffer {
  const size = Math.max(16, roundUp(data.byteLength, 16));
  const buf = device.createBuffer({ label, size, usage, mappedAtCreation: true });
  new Uint8Array(buf.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  buf.unmap();
  return buf;
}

/** Copies `byteLength` bytes of `src` into a staging buffer and returns them (waits for the GPU). */
export async function readBuffer(device: GPUDevice, src: GPUBuffer, byteLength: number, srcOffset = 0): Promise<ArrayBuffer> {
  const size = roundUp(byteLength, 4);
  const staging = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, srcOffset, staging, 0, size);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const copy = staging.getMappedRange().slice(0, byteLength);
  staging.unmap();
  staging.destroy();
  return copy;
}

/** Workgroup grid (x, y) for `n` items at `perGroup` items per workgroup; kernels linearise it with linear_id. */
export function grid1d(n: number, perGroup = WG): [number, number] {
  const total = Math.max(1, Math.ceil(n / perGroup));
  const x = Math.min(total, MAX_WG_DIM);
  return [x, Math.ceil(total / x)];
}

export function bindGroup(device: GPUDevice, pipeline: GPUComputePipeline, buffers: Record<number, GPUBuffer>, label?: string): GPUBindGroup {
  return device.createBindGroup({
    label,
    layout: pipeline.getBindGroupLayout(0),
    entries: Object.entries(buffers).map(([binding, buffer]) => ({ binding: Number(binding), resource: { buffer } })),
  });
}

export function runPass(enc: GPUCommandEncoder, pipeline: GPUComputePipeline, bg: GPUBindGroup, [x, y]: [number, number], label?: string): void {
  const pass = enc.beginComputePass({ label });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(x, y);
  pass.end();
}
