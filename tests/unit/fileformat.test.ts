// .otsr encoding/decoding against the sample the Python reference wrote (tests/golden/sample.otsr).

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { decodeOtsr, encodeOtsr } from '../../src/core/fileformat';
import { GOLDEN_DIR, f32, loadGolden } from '../helpers/golden';

function fileBuffer(name: string): ArrayBuffer {
  const b = readFileSync(join(GOLDEN_DIR, name));
  const ab = new ArrayBuffer(b.byteLength);
  new Uint8Array(ab).set(b);
  return ab;
}

describe('.otsr file format', () => {
  it('decodes the Python-written sample', () => {
    const g = loadGolden('sample');
    const m = decodeOtsr(fileBuffer('sample.otsr'));
    expect([m.W, m.H, m.N]).toEqual([g.meta.W, g.meta.H, g.meta.N]);
    expect(m.meta).toEqual(g.meta.meta);
    expect(Array.from(m.params)).toEqual(Array.from(f32(g, 'params')));
  });

  it('re-encodes byte for byte', () => {
    const bytes = fileBuffer('sample.otsr');
    const again = encodeOtsr(decodeOtsr(bytes));
    expect(new Uint8Array(again)).toEqual(new Uint8Array(bytes));
  });

  it('round-trips without metadata and rejects bad input', () => {
    const params = new Float32Array(2 * 9).map((_, i) => i / 7);
    const buf = encodeOtsr({ W: 3, H: 2, N: 2, params, meta: {} });
    expect(buf.byteLength).toBe(32 + 2 * 36);
    const m = decodeOtsr(buf);
    expect(m.meta).toEqual({});
    expect(Array.from(m.params)).toEqual(Array.from(params));
    expect(() => decodeOtsr(new ArrayBuffer(8))).toThrow(/not an .otsr/);
    expect(() => decodeOtsr(buf.slice(0, buf.byteLength - 4))).toThrow(/size mismatch/);
  });
});
