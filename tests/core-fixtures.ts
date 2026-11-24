/** Loaders for the Python reference fixtures written by `reference/make_fixtures.py`. */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

export const FIXTURES = join(dirname(fileURLToPath(import.meta.url)), "fixtures", "core");

export interface ArrFixture {
  shape: number[];
  dtype: "<f8" | "<f4" | "u1";
  b64: string;
}
export interface BitsFixture {
  shape: number[];
  count: number;
  b64: string;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function loadJson(name: string): any {
  return JSON.parse(readFileSync(join(FIXTURES, name), "utf8"));
}

/** Decode a base64 array fixture to Float64Array (floats) or Uint8Array (u1). */
export function decode(a: ArrFixture): Float64Array | Uint8Array {
  const buf = Buffer.from(a.b64, "base64");
  const bytes = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength).slice();
  if (a.dtype === "u1") return bytes;
  if (a.dtype === "<f8") return new Float64Array(bytes.buffer);
  return Float64Array.from(new Float32Array(bytes.buffer));
}

export function decodeF64(a: ArrFixture): Float64Array {
  const v = decode(a);
  return v instanceof Float64Array ? v : Float64Array.from(v);
}

/** Unpack an `np.packbits(..., bitorder="little")` mask. */
export function decodeBits(b: BitsFixture): Uint8Array {
  const packed = Buffer.from(b.b64, "base64");
  const n = b.shape.reduce((p, q) => p * q, 1);
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = (packed[i >> 3]! >> (i & 7)) & 1;
  return out;
}

/** Same fingerprint as `checksum()` in make_fixtures.py. */
export function checksum(data: ArrayLike<number>, rows: number, cols: number) {
  let nan = 0, sum = 0, sumsq = 0, wsum = 0, min = Infinity, max = -Infinity, first = -1, firstVal: number | null = null;
  for (let i = 0; i < data.length; i++) {
    const v = data[i]!;
    if (!Number.isFinite(v)) {
      nan++;
      continue;
    }
    if (first < 0) {
      first = i;
      firstVal = v;
    }
    sum += v;
    sumsq += v * v;
    wsum += v * ((i % 97) + 1);
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return { shape: [rows, cols], nan, sum, sumsq, wsum, min, max, first_finite_index: first, first_finite: firstVal };
}

/** |a - b| <= atol + rtol * |b| */
export function close(a: number, b: number, rtol = 1e-9, atol = 0): boolean {
  return Math.abs(a - b) <= atol + rtol * Math.abs(b);
}
