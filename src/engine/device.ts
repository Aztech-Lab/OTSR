// Adapter and device acquisition. The same code serves the browser (navigator.gpu) and Node (Dawn's `webgpu`).

export interface AdapterInfo {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  isFallback: boolean;
}

export interface GpuContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  info: AdapterInfo;
  /** Called with every uncaptured validation / out-of-memory error and on device loss. */
  onError?: (message: string) => void;
}

export interface AcquireOptions {
  powerPreference?: GPUPowerPreference;
  onError?: (message: string) => void;
}

export function describeAdapter(adapter: GPUAdapter): AdapterInfo {
  const i = adapter.info ?? ({} as Partial<GPUAdapterInfo>);
  return {
    vendor: i.vendor ?? '',
    architecture: i.architecture ?? '',
    device: i.device ?? '',
    description: i.description ?? '',
    isFallback: (adapter as { isFallbackAdapter?: boolean }).isFallbackAdapter ?? false,
  };
}

const REQUEST_TIMEOUT_MS = 15_000;

/**
 * A hung graphics driver can leave requestAdapter / requestDevice pending forever (seen with headless Chrome 156 on
 * Windows); fail with a clear message instead of an endless start-up.
 */
function withTimeout<T>(p: Promise<T>, what: 'adapter' | 'device'): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const t = setTimeout(() => reject(new Error(`The browser did not answer the WebGPU ${what} request within ${REQUEST_TIMEOUT_MS / 1000} s. The graphics driver may be stuck: restart the browser (or the computer) and try again.`)), REQUEST_TIMEOUT_MS);
    p.then((v) => { clearTimeout(t); resolve(v); }, (e) => { clearTimeout(t); reject(e); });
  });
}

/**
 * Requests a high-performance adapter and a device with the largest buffer limits the adapter offers.
 * The adapter request is retried a few times: Chrome occasionally fails the first request right after start-up.
 * Note that Chrome on Windows ignores powerPreference and uses the GPU that drives the display (crbug.com/369219127).
 */
export async function acquireDevice(gpu: GPU | undefined, opts: AcquireOptions = {}): Promise<GpuContext> {
  if (!gpu) throw new Error('WebGPU is not available in this environment.');
  let adapter: GPUAdapter | null = null;
  for (let attempt = 0; attempt < 4 && !adapter; attempt++) {
    if (attempt) await new Promise((r) => setTimeout(r, 300 * attempt));
    adapter = await withTimeout(gpu.requestAdapter({ powerPreference: opts.powerPreference ?? 'high-performance' }), 'adapter');
  }
  if (!adapter) throw new Error('No WebGPU adapter found.');
  const L = adapter.limits;
  const requiredLimits: Record<string, number> = {
    maxBufferSize: Math.min(L.maxBufferSize, 2 ** 31),
    maxStorageBufferBindingSize: Math.min(L.maxStorageBufferBindingSize, 2 ** 31),
    maxStorageBuffersPerShaderStage: Math.max(8, Math.min(L.maxStorageBuffersPerShaderStage, 10)),
    maxComputeWorkgroupStorageSize: Math.max(16384, Math.min(L.maxComputeWorkgroupStorageSize, 32768)),
  };
  const device = await withTimeout(adapter.requestDevice({ requiredLimits }), 'device');
  const ctx: GpuContext = { adapter, device, info: describeAdapter(adapter), onError: opts.onError };
  const report = (msg: string) => (ctx.onError ?? console.error)(msg);
  device.lost.then((info) => {
    if (info.reason !== 'destroyed') report(`WebGPU device lost (${info.reason}): ${info.message}`);
  });
  if (typeof device.addEventListener === 'function') {
    device.addEventListener('uncapturederror', (ev) => report(`WebGPU error: ${(ev as GPUUncapturedErrorEvent).error.message}`));
  }
  return ctx;
}
