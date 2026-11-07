// Numeric comparison helpers that also print a one-line summary, so a passing run documents its accuracy.

export interface Stats {
  n: number;
  maxAbs: number;
  rms: number;
  maxRef: number;
  /** max |diff| / max |ref| */
  maxRel: number;
  argmax: number;
}

export function stats(got: ArrayLike<number>, ref: ArrayLike<number>, stride = 1, offset = 0): Stats {
  if (got.length !== ref.length) throw new Error(`length mismatch: got ${got.length}, ref ${ref.length}`);
  let maxAbs = 0;
  let sum = 0;
  let maxRef = 0;
  let argmax = -1;
  let n = 0;
  for (let i = offset; i < ref.length; i += stride) {
    const d = Math.abs(got[i] - ref[i]);
    if (Number.isNaN(d)) return { n: 1, maxAbs: NaN, rms: NaN, maxRef: 0, maxRel: NaN, argmax: i };
    if (d > maxAbs) {
      maxAbs = d;
      argmax = i;
    }
    sum += d * d;
    maxRef = Math.max(maxRef, Math.abs(ref[i]));
    n++;
  }
  return { n, maxAbs, rms: Math.sqrt(sum / Math.max(1, n)), maxRef, maxRel: maxRef > 0 ? maxAbs / maxRef : maxAbs, argmax };
}

export function report(label: string, s: Stats): Stats {
  console.log(`  ${label.padEnd(26)} n=${String(s.n).padStart(8)}  max|d|=${s.maxAbs.toExponential(2)}  rms=${s.rms.toExponential(2)}  max|ref|=${s.maxRef.toExponential(2)}  rel=${s.maxRel.toExponential(2)}`);
  return s;
}

/** Asserts max |got - ref| <= atol + rtol * max |ref| and prints the statistics. */
export function expectClose(label: string, got: ArrayLike<number>, ref: ArrayLike<number>, atol: number, rtol = 0, stride = 1, offset = 0): Stats {
  const s = report(label, stats(got, ref, stride, offset));
  const tol = atol + rtol * s.maxRef;
  if (!(s.maxAbs <= tol)) {
    throw new Error(`${label}: max |diff| ${s.maxAbs} exceeds ${tol} at index ${s.argmax} (got ${got[s.argmax]}, ref ${ref[s.argmax]})`);
  }
  return s;
}

/** Picks channel `c` of an interleaved array with `channels` per element. */
export function channel(a: ArrayLike<number>, channels: number, c: number): Float64Array {
  const n = Math.floor(a.length / channels);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = a[i * channels + c];
  return out;
}
