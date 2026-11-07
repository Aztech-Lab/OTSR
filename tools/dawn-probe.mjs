// Lists the GPU adapters Dawn (npm "webgpu") can use in Node, i.e. what the GPU tests run on.
// Usage: node tools/dawn-probe.mjs [adapter=<name part>] [backend=d3d12|vulkan|metal]
import { create, globals } from 'webgpu';

Object.assign(globalThis, globals);
const flags = process.argv.slice(2);
const gpu = create(flags);
const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
if (!adapter) {
  console.log('no adapter' + (flags.length ? ` for ${flags.join(' ')}` : ''));
} else {
  const i = adapter.info;
  const l = adapter.limits;
  console.log(`adapter: ${i.device || i.description} (${i.vendor} ${i.architecture}) ${i.description}`);
  console.log(`features: ${[...adapter.features].sort().join(', ')}`);
  console.log(`limits: maxBufferSize ${l.maxBufferSize}, maxStorageBufferBindingSize ${l.maxStorageBufferBindingSize}, ` +
    `maxStorageBuffersPerShaderStage ${l.maxStorageBuffersPerShaderStage}, maxComputeWorkgroupStorageSize ${l.maxComputeWorkgroupStorageSize}, ` +
    `maxComputeWorkgroupsPerDimension ${l.maxComputeWorkgroupsPerDimension}`);
  console.log('To list every adapter, pass a name that does not exist: node tools/dawn-probe.mjs adapter=list');
}
