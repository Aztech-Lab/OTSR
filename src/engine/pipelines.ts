// Compiles every WGSL kernel once per device. common.wgsl is prepended to each module.

import common from './shaders/common.wgsl?raw';
import preprocess from './shaders/preprocess.wgsl?raw';
import scan from './shaders/scan.wgsl?raw';
import fill from './shaders/fill.wgsl?raw';
import forward from './shaders/forward.wgsl?raw';
import backward from './shaders/backward.wgsl?raw';
import reduceGrad from './shaders/reduce_grad.wgsl?raw';
import adam from './shaders/adam.wgsl?raw';
import loss from './shaders/loss.wgsl?raw';
import sinkhorn from './shaders/sinkhorn.wgsl?raw';
import edges from './shaders/edges.wgsl?raw';
import resample from './shaders/resample.wgsl?raw';

export interface Pipelines {
  resample: GPUComputePipeline;
  lseX: GPUComputePipeline;
  lseY: GPUComputePipeline;
  edgeH: GPUComputePipeline;
  edgeV: GPUComputePipeline;
  edgeFinal: GPUComputePipeline;
  preprocess: GPUComputePipeline;
  scanBlocks: GPUComputePipeline;
  addOffsets: GPUComputePipeline;
  fill: GPUComputePipeline;
  forward: GPUComputePipeline;
  backward: GPUComputePipeline;
  reduceGrad: GPUComputePipeline;
  adam: GPUComputePipeline;
  loss: GPUComputePipeline;
}

export const SHADER_SOURCES = { common, preprocess, scan, fill, forward, backward, reduceGrad, adam, loss, sinkhorn, edges, resample };

async function compile(device: GPUDevice, name: string, source: string, entryPoint: string): Promise<GPUComputePipeline> {
  const module = device.createShaderModule({ label: name, code: `${common}\n${source}` });
  try {
    return await device.createComputePipelineAsync({ label: `${name}:${entryPoint}`, layout: 'auto', compute: { module, entryPoint } });
  } catch (e) {
    const info = await module.getCompilationInfo();
    const msgs = info.messages.map((m) => `  ${m.type} ${name}.wgsl:${m.lineNum}:${m.linePos}: ${m.message}`).join('\n');
    throw new Error(`Failed to compile ${name}:${entryPoint}: ${(e as Error).message}\n${msgs}`);
  }
}

export async function createPipelines(device: GPUDevice): Promise<Pipelines> {
  const [p, sb, ao, fi, fw, bw, rg, ad, lo, lx, ly, eh, ev, ef, rs] = await Promise.all([
    compile(device, 'preprocess', preprocess, 'preprocess'),
    compile(device, 'scan', scan, 'scan_blocks'),
    compile(device, 'scan', scan, 'add_offsets'),
    compile(device, 'fill', fill, 'fill'),
    compile(device, 'forward', forward, 'forward'),
    compile(device, 'backward', backward, 'backward'),
    compile(device, 'reduce_grad', reduceGrad, 'reduce_grad'),
    compile(device, 'adam', adam, 'adam'),
    compile(device, 'loss', loss, 'loss'),
    compile(device, 'sinkhorn', sinkhorn, 'lse_x'),
    compile(device, 'sinkhorn', sinkhorn, 'lse_y'),
    compile(device, 'edges', edges, 'edge_h'),
    compile(device, 'edges', edges, 'edge_v'),
    compile(device, 'edges', edges, 'edge_final'),
    compile(device, 'resample', resample, 'resample'),
  ]);
  return {
    preprocess: p, scanBlocks: sb, addOffsets: ao, fill: fi, forward: fw, backward: bw, reduceGrad: rg, adam: ad, loss: lo,
    lseX: lx, lseY: ly, edgeH: eh, edgeV: ev, edgeFinal: ef, resample: rs,
  };
}
