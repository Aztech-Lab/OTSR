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

// ---- export formats --------------------------------------------------------------------------------------------

export type ExportFormat = 'png' | 'jpeg' | 'webp' | 'avif' | 'bmp';

export interface ExportFormatInfo {
  key: ExportFormat;
  label: string;
  ext: string;
  mime: string;
  /** Takes a quality setting. */
  lossy: boolean;
}

/** The formats offered by the export menu; the browser's encoder does all but BMP (written by hand). */
export const EXPORT_FORMATS: readonly ExportFormatInfo[] = [
  { key: 'png', label: 'PNG', ext: 'png', mime: 'image/png', lossy: false },
  { key: 'jpeg', label: 'JPEG', ext: 'jpg', mime: 'image/jpeg', lossy: true },
  { key: 'webp', label: 'WebP', ext: 'webp', mime: 'image/webp', lossy: true },
  { key: 'avif', label: 'AVIF', ext: 'avif', mime: 'image/avif', lossy: true },
  { key: 'bmp', label: 'BMP', ext: 'bmp', mime: 'image/bmp', lossy: false },
];

const support = new Map<ExportFormat, Promise<boolean>>();

/** Whether this browser can encode the format (a 1 x 1 probe; the result is cached). */
export function canEncode(format: ExportFormat): Promise<boolean> {
  let p = support.get(format);
  if (!p) {
    p = (async () => {
      if (format === 'png' || format === 'bmp') return true;
      try {
        const info = EXPORT_FORMATS.find((f) => f.key === format)!;
        const c = new OffscreenCanvas(1, 1);
        c.getContext('2d')!.fillRect(0, 0, 1, 1);
        const b = await c.convertToBlob({ type: info.mime, quality: 0.9 });
        return b.type === info.mime;
      } catch {
        return false;
      }
    })();
    support.set(format, p);
  }
  return p;
}

/** Uncompressed 24-bit BMP (bottom-up rows padded to 4 bytes; alpha dropped). */
export function encodeBmp(rgba: Uint8ClampedArray, W: number, H: number): Blob {
  const rowBytes = (W * 3 + 3) & ~3;
  const size = 54 + rowBytes * H;
  const buf = new ArrayBuffer(size);
  const v = new DataView(buf);
  const u8 = new Uint8Array(buf);
  v.setUint8(0, 0x42); v.setUint8(1, 0x4d); // BM
  v.setUint32(2, size, true); v.setUint32(10, 54, true);
  v.setUint32(14, 40, true); v.setInt32(18, W, true); v.setInt32(22, H, true);
  v.setUint16(26, 1, true); v.setUint16(28, 24, true); v.setUint32(34, rowBytes * H, true);
  v.setInt32(38, 2835, true); v.setInt32(42, 2835, true); // 72 dpi
  for (let y = 0; y < H; y++) {
    const dst = 54 + (H - 1 - y) * rowBytes;
    const src = y * W * 4;
    for (let x = 0; x < W; x++) {
      u8[dst + 3 * x] = rgba[src + 4 * x + 2];
      u8[dst + 3 * x + 1] = rgba[src + 4 * x + 1];
      u8[dst + 3 * x + 2] = rgba[src + 4 * x];
    }
  }
  return new Blob([buf], { type: 'image/bmp' });
}

/**
 * Encodes 8-bit RGBA in the chosen format (quality 0..1 for the lossy ones). If the browser cannot encode the format
 * the result is a PNG and `fallback` is true.
 */
export async function encodeImage(rgba: Uint8ClampedArray<ArrayBuffer>, W: number, H: number, format: ExportFormat, quality = 0.92): Promise<{ blob: Blob; ext: string; fallback: boolean }> {
  const info = EXPORT_FORMATS.find((f) => f.key === format) ?? EXPORT_FORMATS[0];
  if (info.key === 'bmp') return { blob: encodeBmp(rgba, W, H), ext: 'bmp', fallback: false };
  if (info.key === 'png') return { blob: await encodePng(rgba, W, H), ext: 'png', fallback: false };
  const canvas = new OffscreenCanvas(W, H);
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D canvas unavailable');
  ctx.putImageData(new ImageData(rgba, W, H), 0, 0);
  const blob = await canvas.convertToBlob({ type: info.mime, quality: Math.max(0.05, Math.min(1, quality)) });
  if (blob.type !== info.mime) return { blob: await canvas.convertToBlob({ type: 'image/png' }), ext: 'png', fallback: true };
  return { blob, ext: info.ext, fallback: false };
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
