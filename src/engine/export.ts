// Full-image rendering at S x in tiles (with the edge pass), returning 8-bit RGBA for saving.

import { edgeWindowRadius } from '../core/spec';
import type { GpuContext } from './device';
import { EdgePass, type EdgeOptions } from './edges';
import { createBuffer, readBuffer } from './gpu';
import type { Pipelines } from './pipelines';
import { Rasterizer } from './raster';

export interface RenderImageOptions {
  /** Apply the distance-field edge pass (default true). */
  edges?: boolean;
  edge?: EdgeOptions;
  /** Tile edge in output pixels (default 2048). */
  tile?: number;
  onProgress?: (done: number, total: number) => void | Promise<void>;
  signal?: AbortSignal;
}

export interface RenderedImage {
  width: number;
  height: number;
  rgba: Uint8ClampedArray<ArrayBuffer>;
}

/**
 * Renders the model (`params` on the GPU, N Gaussians fitted to a W x H image) at magnification `S` into an
 * 8-bit RGBA image of round(W S) x round(H S) pixels. Tiles overlap by the edge window so the result is identical to
 * an untiled render.
 */
export async function renderImage(ctx: GpuContext, pipes: Pipelines, params: GPUBuffer, N: number, W: number, H: number, S: number, opts: RenderImageOptions = {}): Promise<RenderedImage> {
  const device = ctx.device;
  const width = Math.round(W * S);
  const height = Math.round(H * S);
  const useEdges = opts.edges ?? true;
  const margin = useEdges ? edgeWindowRadius(S) + 1 : 0;
  const tile = Math.max(256, opts.tile ?? 2048);
  const rgba = new Uint8ClampedArray(width * height * 4);
  const tilesX = Math.ceil(width / tile);
  const tilesY = Math.ceil(height / tile);
  const total = tilesX * tilesY;
  const cache = new Map<string, { raster: Rasterizer; edge?: EdgePass; dst?: GPUBuffer }>();
  let done = 0;
  try {
    for (let ty = 0; ty < tilesY; ty++) {
      for (let tx = 0; tx < tilesX; tx++) {
        if (opts.signal?.aborted) throw new Error('cancelled');
        const x0 = tx * tile;
        const y0 = ty * tile;
        const tw = Math.min(tile, width - x0);
        const th = Math.min(tile, height - y0);
        // the rendered window includes the margin, clipped to the image
        const wx0 = Math.max(0, x0 - margin);
        const wy0 = Math.max(0, y0 - margin);
        const wx1 = Math.min(width, x0 + tw + margin);
        const wy1 = Math.min(height, y0 + th + margin);
        const ww = wx1 - wx0;
        const wh = wy1 - wy0;
        const key = `${ww}x${wh}`;
        let entry = cache.get(key);
        if (!entry) {
          const raster = new Rasterizer(ctx, pipes, params, N, ww, wh, Math.max(4096, N));
          entry = { raster };
          if (useEdges) {
            entry.edge = new EdgePass(ctx, pipes, ww, wh);
            entry.dst = createBuffer(device, ww * wh * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'export.dst');
          }
          cache.set(key, entry);
        }
        const { raster } = entry;
        raster.setView({ srcW: W, srcH: H, zoom: S, offsetX: -wx0, offsetY: -wy0, aa: 0, outW: ww, outH: wh });
        let enc = device.createCommandEncoder();
        raster.encodeBuild(enc);
        device.queue.submit([enc.finish()]);
        const entries = await raster.readTotalEntries();
        if (entries > raster.entryCap) {
          raster.growEntries(Math.ceil(entries * 1.5));
          enc = device.createCommandEncoder();
          raster.encodeBuild(enc);
          device.queue.submit([enc.finish()]);
        }
        enc = device.createCommandEncoder();
        raster.encodeForward(enc);
        if (useEdges) entry.edge!.encode(enc, raster.out, entry.dst!, S, opts.edge);
        device.queue.submit([enc.finish()]);
        const px = new Float32Array(await readBuffer(device, useEdges ? entry.dst! : raster.out, ww * wh * 16));
        for (let y = y0; y < y0 + th; y++) {
          const row = (y - wy0) * ww;
          for (let x = x0; x < x0 + tw; x++) {
            const s = 4 * (row + (x - wx0));
            const d = 4 * (y * width + x);
            rgba[d] = px[s] * 255 + 0.5;
            rgba[d + 1] = px[s + 1] * 255 + 0.5;
            rgba[d + 2] = px[s + 2] * 255 + 0.5;
            rgba[d + 3] = 255;
          }
        }
        done++;
        if (opts.onProgress) await opts.onProgress(done, total);
      }
    }
  } finally {
    for (const e of cache.values()) {
      e.raster.destroy();
      e.edge?.destroy();
      e.dst?.destroy();
    }
  }
  return { width, height, rgba };
}
