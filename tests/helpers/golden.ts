// Loads a golden case written by reference/make_golden.py: <case>.json (manifest) + <case>.bin (arrays).

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

export const GOLDEN_DIR = join(dirname(fileURLToPath(import.meta.url)), '..', 'golden');

export type GoldenArray = Float32Array<ArrayBuffer> | Int32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>;

interface Entry {
  dtype: 'float32' | 'int32' | 'uint32';
  shape: number[];
  offset: number;
  count: number;
}

export interface Golden {
  meta: Record<string, any>;
  arrays: Record<string, GoldenArray>;
  shape: (name: string) => number[];
}

export function loadGolden(name: string): Golden {
  const manifest = JSON.parse(readFileSync(join(GOLDEN_DIR, `${name}.json`), 'utf8')) as { meta: Record<string, any>; arrays: Record<string, Entry> };
  const bin = readFileSync(join(GOLDEN_DIR, `${name}.bin`));
  const buf = new ArrayBuffer(bin.byteLength); // a plain ArrayBuffer (not a Buffer pool slice): WebGPU uploads need one
  new Uint8Array(buf).set(bin);
  const arrays: Record<string, GoldenArray> = {};
  for (const [key, e] of Object.entries(manifest.arrays)) {
    const C = e.dtype === 'float32' ? Float32Array : e.dtype === 'int32' ? Int32Array : Uint32Array;
    arrays[key] = new C(buf, e.offset, e.count);
  }
  return { meta: manifest.meta, arrays, shape: (n) => manifest.arrays[n].shape };
}

export function f32(g: Golden, name: string): Float32Array<ArrayBuffer> {
  const a = g.arrays[name];
  if (!(a instanceof Float32Array)) throw new Error(`${name} is not float32`);
  return a;
}
