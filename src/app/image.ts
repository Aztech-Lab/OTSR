// Image decoding for the browser: any format the browser can decode -> float RGB in [0, 1] (sRGB values, no
// colour management: the model is fitted to the pixel values as displayed).

export interface LoadedImage {
  name: string;
  W: number;
  H: number;
  /** W x H x 3 floats. */
  rgb: Float32Array;
  /** W x H x 4 floats (alpha = 1), the GPU layout. */
  rgba: Float32Array;
}

/** Pixel count above which the UI asks for confirmation (memory and time grow linearly with it). */
export const WARN_PIXELS = 8_500_000;
/** Pixel count the UI refuses (buffers would exceed what most WebGPU implementations allow). */
export const MAX_PIXELS = 33_000_000;

export function imageFromRgba8(rgba8: Uint8ClampedArray | Uint8Array, W: number, H: number, name: string): LoadedImage {
  const n = W * H;
  const rgb = new Float32Array(n * 3);
  const rgba = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    const r = rgba8[4 * i] / 255;
    const g = rgba8[4 * i + 1] / 255;
    const b = rgba8[4 * i + 2] / 255;
    rgb[3 * i] = r;
    rgb[3 * i + 1] = g;
    rgb[3 * i + 2] = b;
    rgba[4 * i] = r;
    rgba[4 * i + 1] = g;
    rgba[4 * i + 2] = b;
    rgba[4 * i + 3] = 1;
  }
  return { name, W, H, rgb, rgba };
}

export async function decodeImageFile(file: Blob, name: string): Promise<LoadedImage> {
  const bitmap = await createImageBitmap(file, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  try {
    const { width: W, height: H } = bitmap;
    if (W * H > MAX_PIXELS) throw new Error(`${W} x ${H} is too large (${(W * H / 1e6).toFixed(1)} MP; the limit is ${MAX_PIXELS / 1e6} MP)`);
    const canvas = new OffscreenCanvas(W, H);
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('2D canvas unavailable');
    ctx.drawImage(bitmap, 0, 0);
    const data = ctx.getImageData(0, 0, W, H).data;
    return imageFromRgba8(data, W, H, name);
  } finally {
    bitmap.close();
  }
}

/** 8-bit RGBA -> PNG blob (via a canvas; the browser's encoder). */
export async function encodePng(rgba: Uint8ClampedArray<ArrayBuffer>, W: number, H: number): Promise<Blob> {
  const canvas = new OffscreenCanvas(W, H);
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D canvas unavailable');
  ctx.putImageData(new ImageData(rgba, W, H), 0, 0);
  return canvas.convertToBlob({ type: 'image/png' });
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

export function stem(name: string): string {
  return name.replace(/\.[^.]+$/, '') || 'image';
}
