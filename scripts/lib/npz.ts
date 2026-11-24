/**
 * Minimal NumPy `.npy` / `.npz` reader and writer (C order, little endian).
 *
 * The revision runs store label grids and synthetic heads as `.npz` so the same files open in
 * Python with `np.load(path)["labels"]`, and runs made by the Python scripts open here.
 * Supported dtypes: `|u1`, `|b1`, `<i4`, `<i8` (as float64), `<f4`, `<f8`.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { unzipSync, zipSync } from "fflate";

export type NpyDtype = "|u1" | "|b1" | "<i4" | "<f4" | "<f8";
export interface NpyArray {
  dtype: NpyDtype;
  shape: number[];
  data: Uint8Array | Int32Array | Float32Array | Float64Array;
}

/** Encode one array as a version-1.0 `.npy` file. */
export function encodeNpy(arr: NpyArray): Uint8Array {
  const shape = arr.shape.length === 1 ? `(${arr.shape[0]},)` : `(${arr.shape.join(", ")})`;
  let header = `{'descr': '${arr.dtype}', 'fortran_order': False, 'shape': ${shape}, }`;
  const total = 10 + header.length + 1;
  header += " ".repeat((64 - (total % 64)) % 64) + "\n";
  const bytes = new Uint8Array(arr.data.buffer, arr.data.byteOffset, arr.data.byteLength);
  const out = new Uint8Array(10 + header.length + bytes.length);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0], 0); // \x93NUMPY v1.0
  out[8] = header.length & 0xff;
  out[9] = header.length >> 8;
  for (let i = 0; i < header.length; i++) out[10 + i] = header.charCodeAt(i);
  out.set(bytes, 10 + header.length);
  return out;
}

/** Decode a `.npy` file (versions 1-3). */
export function decodeNpy(buf: Uint8Array): NpyArray {
  if (buf[0] !== 0x93 || String.fromCharCode(...buf.subarray(1, 6)) !== "NUMPY") throw new Error("not a .npy file");
  const major = buf[6]!;
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const hlen = major === 1 ? dv.getUint16(8, true) : dv.getUint32(8, true);
  const start = major === 1 ? 10 : 12;
  const header = new TextDecoder().decode(buf.subarray(start, start + hlen));
  const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  if (/'fortran_order':\s*True/.test(header)) throw new Error("Fortran-order arrays are not supported");
  const shapeText = /'shape':\s*\(([^)]*)\)/.exec(header)?.[1] ?? "";
  const shape = shapeText.split(",").map((s) => s.trim()).filter(Boolean).map(Number);
  const n = shape.reduce((a, b) => a * b, 1);
  const off = start + hlen;
  const raw = buf.slice(off); // copy so typed arrays are aligned
  switch (descr) {
    case "|u1":
    case "|b1":
      return { dtype: descr, shape, data: raw.subarray(0, n) };
    case "<i4":
      return { dtype: "<i4", shape, data: new Int32Array(raw.buffer, 0, n) };
    case "<i8": {
      const big = new BigInt64Array(raw.buffer, 0, n);
      return { dtype: "<f8", shape, data: Float64Array.from(big, (v) => Number(v)) };
    }
    case "<f4":
      return { dtype: "<f4", shape, data: new Float32Array(raw.buffer, 0, n) };
    case "<f8":
      return { dtype: "<f8", shape, data: new Float64Array(raw.buffer, 0, n) };
    default:
      throw new Error(`unsupported .npy dtype ${descr}`);
  }
}

/** Write `np.savez_compressed`-style archive: one `<name>.npy` member per entry. */
export function saveNpz(path: string, arrays: Record<string, NpyArray>): void {
  const files: Record<string, [Uint8Array, { level: 6 }]> = {};
  for (const [k, v] of Object.entries(arrays)) files[`${k}.npy`] = [encodeNpy(v), { level: 6 }];
  writeFileSync(path, zipSync(files));
}

/** Read every array of an `.npz` archive. */
export function loadNpz(path: string): Record<string, NpyArray> {
  const out: Record<string, NpyArray> = {};
  for (const [name, bytes] of Object.entries(unzipSync(readFileSync(path)))) {
    out[name.replace(/\.npy$/, "")] = decodeNpy(bytes);
  }
  return out;
}

/** A 2-D uint8 grid as an npy array (`|u1`, or `|b1` for masks). */
export function gridNpy(g: { data: Uint8Array; rows: number; cols: number }, dtype: "|u1" | "|b1" = "|u1"): NpyArray {
  return { dtype, shape: [g.rows, g.cols], data: g.data };
}

/** A float64 scalar as a 0-d npy array. */
export function scalarNpy(x: number): NpyArray {
  return { dtype: "<f8", shape: [], data: Float64Array.of(x) };
}

/** A 2-D npy array as a grid (rows, cols from the shape). */
export function npyGrid<T extends NpyArray["data"]>(a: NpyArray): { data: T; rows: number; cols: number } {
  if (a.shape.length !== 2) throw new Error(`expected a 2-D array, got shape (${a.shape.join(", ")})`);
  return { data: a.data as T, rows: a.shape[0]!, cols: a.shape[1]! };
}
