// Exclusive prefix sum of a u32 buffer (scan.wgsl): blocks of 1024, recursive over block totals.

import type { Pipelines } from './pipelines';
import { bindGroup, createBuffer, grid1d, runPass } from './gpu';

const BLOCK = 1024;

interface Level {
  n: number;
  blocks: number;
  uni: GPUBuffer;
  dst: GPUBuffer;
  sums: GPUBuffer;
  bgScan: GPUBindGroup;
  bgAdd?: GPUBindGroup;
}

export class ScanPlan {
  private readonly levels: Level[] = [];
  /** The grand total lives in this buffer at byte offset 0 after `encode` has run. */
  readonly total: GPUBuffer;

  constructor(device: GPUDevice, private readonly pipes: Pipelines, src: GPUBuffer, dst: GPUBuffer, readonly n: number) {
    let curSrc = src;
    let curDst = dst;
    let curN = n;
    for (;;) {
      const blocks = Math.ceil(curN / BLOCK);
      const uni = device.createBuffer({ label: 'scan.n', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(uni, 0, new Uint32Array([curN, 0, 0, 0]));
      const sums = createBuffer(device, blocks * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'scan.sums');
      const bgScan = bindGroup(device, pipes.scanBlocks, { 0: uni, 1: curSrc, 2: curDst, 3: sums }, 'scan.blocks');
      this.levels.push({ n: curN, blocks, uni, dst: curDst, sums, bgScan });
      if (blocks === 1) break;
      curSrc = sums;
      curDst = createBuffer(device, blocks * 4, GPUBufferUsage.STORAGE, 'scan.sumsScanned');
      curN = blocks;
    }
    for (let k = 0; k + 1 < this.levels.length; k++) {
      const lv = this.levels[k];
      lv.bgAdd = bindGroup(device, pipes.addOffsets, { 0: lv.uni, 2: lv.dst, 4: this.levels[k + 1].dst }, 'scan.add');
    }
    this.total = this.levels[this.levels.length - 1].sums;
  }

  encode(enc: GPUCommandEncoder): void {
    for (const lv of this.levels) runPass(enc, this.pipes.scanBlocks, lv.bgScan, grid1d(lv.blocks, 1), 'scan');
    for (let k = this.levels.length - 2; k >= 0; k--) {
      const lv = this.levels[k];
      runPass(enc, this.pipes.addOffsets, lv.bgAdd!, grid1d(lv.blocks, 1), 'scan.add');
    }
  }

  destroy(): void {
    for (let k = 0; k < this.levels.length; k++) {
      const lv = this.levels[k];
      lv.uni.destroy();
      lv.sums.destroy();
      if (k > 0) lv.dst.destroy(); // level 0's dst belongs to the caller
    }
  }
}
