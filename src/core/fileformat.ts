// The .otsr model file, format 1.0 (docs/file-format.md; reference/otsr_ref/fileformat.py is the Python twin).
//
//   offset  size   field
//   0       4      magic "OTSR"
//   4       2      version major (1)        6   2   version minor (0)
//   8       4      width W (u32)            12  4   height H (u32)
//   16      4      Gaussian count N (u32)   20  4   flags (u32, 0)
//   24      4      metadata length M (u32)  28  4   reserved (u32, 0)
//   32      M      metadata: UTF-8 JSON object (M = 0: none), then zero padding to a multiple of 4
//   ...     36 N   float32 payload, one planar block per parameter: uv (N x 2), log_s (N x 2), rot (N), color (N x 3), log_w (N)
// All integers and floats are little-endian.

import { FLOATS_PER_GAUSSIAN, PARAM_LAYOUT } from './spec';

export const OTSR_MAGIC = 'OTSR';
export const OTSR_VERSION = [1, 0] as const;
const HEADER_BYTES = 32;

export interface OtsrModel {
  W: number;
  H: number;
  N: number;
  /** Interleaved N x 9 floats (the engine layout). */
  params: Float32Array;
  meta: Record<string, unknown>;
}

export function encodeOtsr(model: OtsrModel): ArrayBuffer {
  const { W, H, N, params } = model;
  if (params.length !== N * FLOATS_PER_GAUSSIAN) throw new Error('params length does not match N');
  const metaBytes = Object.keys(model.meta).length ? new TextEncoder().encode(JSON.stringify(model.meta)) : new Uint8Array(0);
  const pad = (4 - ((HEADER_BYTES + metaBytes.length) % 4)) % 4;
  const payloadOffset = HEADER_BYTES + metaBytes.length + pad;
  const buf = new ArrayBuffer(payloadOffset + N * FLOATS_PER_GAUSSIAN * 4);
  const dv = new DataView(buf);
  new Uint8Array(buf, 0, 4).set([0x4f, 0x54, 0x53, 0x52]); // "OTSR"
  dv.setUint16(4, OTSR_VERSION[0], true);
  dv.setUint16(6, OTSR_VERSION[1], true);
  dv.setUint32(8, W, true);
  dv.setUint32(12, H, true);
  dv.setUint32(16, N, true);
  dv.setUint32(20, 0, true);
  dv.setUint32(24, metaBytes.length, true);
  dv.setUint32(28, 0, true);
  new Uint8Array(buf, HEADER_BYTES, metaBytes.length).set(metaBytes);
  const out = new Float32Array(buf, payloadOffset, N * FLOATS_PER_GAUSSIAN);
  let o = 0;
  let col = 0;
  for (const [, n] of PARAM_LAYOUT) {
    for (let i = 0; i < N; i++) for (let k = 0; k < n; k++) out[o++] = params[i * FLOATS_PER_GAUSSIAN + col + k];
    col += n;
  }
  return buf;
}

export function decodeOtsr(buf: ArrayBuffer): OtsrModel {
  const dv = new DataView(buf);
  if (buf.byteLength < HEADER_BYTES || String.fromCharCode(...new Uint8Array(buf, 0, 4)) !== OTSR_MAGIC) throw new Error('not an .otsr file');
  const major = dv.getUint16(4, true);
  const minor = dv.getUint16(6, true);
  if (major !== OTSR_VERSION[0]) throw new Error(`unsupported .otsr version ${major}.${minor}`);
  const W = dv.getUint32(8, true);
  const H = dv.getUint32(12, true);
  const N = dv.getUint32(16, true);
  const metaLen = dv.getUint32(24, true);
  const meta = metaLen ? (JSON.parse(new TextDecoder().decode(new Uint8Array(buf, HEADER_BYTES, metaLen))) as Record<string, unknown>) : {};
  const payloadOffset = HEADER_BYTES + metaLen + ((4 - ((HEADER_BYTES + metaLen) % 4)) % 4);
  if (buf.byteLength !== payloadOffset + N * FLOATS_PER_GAUSSIAN * 4) throw new Error(`.otsr size mismatch: expected ${payloadOffset + N * 36} bytes, got ${buf.byteLength}`);
  const planar = new Float32Array(buf.slice(payloadOffset), 0, N * FLOATS_PER_GAUSSIAN);
  const params = new Float32Array(N * FLOATS_PER_GAUSSIAN);
  let o = 0;
  let col = 0;
  for (const [, n] of PARAM_LAYOUT) {
    for (let i = 0; i < N; i++) for (let k = 0; k < n; k++) params[i * FLOATS_PER_GAUSSIAN + col + k] = planar[o++];
    col += n;
  }
  return { W, H, N, params, meta };
}
