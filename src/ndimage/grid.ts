/**
 * Row-major 2-D grids and the NumPy basics the pipeline needs (percentile, median, MAD).
 *
 * A 1-D signal is a grid with `rows = 1`.
 */

export type Num = Float64Array | Float32Array;

export interface Grid<T extends ArrayLike<number> = Float64Array> {
  data: T;
  rows: number;
  cols: number;
}

export type Mask = Grid<Uint8Array>; // 0/1
export type Labels = Grid<Int32Array>; // 0 = background, 1..n

/** Any numeric typed array a grid may hold. */
export type AnyArray =
  | Float64Array
  | Float32Array
  | Int32Array
  | Uint32Array
  | Int16Array
  | Uint16Array
  | Int8Array
  | Uint8Array
  | Uint8ClampedArray;

type ArrayCtor<T> = { new (n: number): T };

/** Wrap `data` (or a new zero-filled Float64Array) as a grid. */
export function grid(rows: number, cols: number): Grid<Float64Array>;
export function grid<T extends ArrayLike<number>>(rows: number, cols: number, data: T): Grid<T>;
export function grid(rows: number, cols: number, data?: ArrayLike<number>): Grid<ArrayLike<number>> {
  const d = data ?? new Float64Array(rows * cols);
  if (d.length !== rows * cols) throw new Error(`grid: data length ${d.length} != ${rows}x${cols}`);
  return { data: d, rows, cols };
}

/** Zero grid; Float64Array unless another typed-array constructor is given. */
export function zeros(rows: number, cols: number): Grid<Float64Array>;
export function zeros<T extends AnyArray>(rows: number, cols: number, ctor: ArrayCtor<T>): Grid<T>;
export function zeros(rows: number, cols: number, ctor?: ArrayCtor<AnyArray>): Grid<AnyArray> {
  const C = ctor ?? Float64Array;
  return { data: new C(rows * cols), rows, cols };
}

export function full(rows: number, cols: number, value: number): Grid<Float64Array> {
  const data = new Float64Array(rows * cols);
  data.fill(value);
  return { data, rows, cols };
}

export function clone<T extends AnyArray>(g: Grid<T>): Grid<T> {
  return { data: g.data.slice() as T, rows: g.rows, cols: g.cols };
}

/** Element-wise map into a new Float64 grid. `fn(value, flatIndex)`. */
export function map(g: Grid<ArrayLike<number>>, fn: (v: number, i: number) => number): Grid<Float64Array> {
  const n = g.rows * g.cols;
  const out = new Float64Array(n);
  const d = g.data;
  for (let i = 0; i < n; i++) out[i] = fn(d[i], i);
  return { data: out, rows: g.rows, cols: g.cols };
}

export function at(g: Grid<ArrayLike<number>>, r: number, c: number): number {
  return g.data[r * g.cols + c];
}

/** `np.isfinite(g)` as a 0/1 mask. */
export function isFiniteMask(g: Grid<ArrayLike<number>>): Mask {
  const n = g.rows * g.cols;
  const out = new Uint8Array(n);
  const d = g.data;
  for (let i = 0; i < n; i++) out[i] = Number.isFinite(d[i]) ? 1 : 0;
  return { data: out, rows: g.rows, cols: g.cols };
}

/** `np.where(mask, a, b)`; `a`/`b` may be grids or scalars. */
export function where(
  mask: Grid<ArrayLike<number>>,
  a: Grid<ArrayLike<number>> | number,
  b: Grid<ArrayLike<number>> | number,
): Grid<Float64Array> {
  const n = mask.rows * mask.cols;
  const out = new Float64Array(n);
  const m = mask.data;
  const ad = typeof a === "number" ? null : a.data;
  const bd = typeof b === "number" ? null : b.data;
  const av = typeof a === "number" ? a : 0;
  const bv = typeof b === "number" ? b : 0;
  for (let i = 0; i < n; i++) out[i] = m[i] ? (ad ? ad[i] : av) : bd ? bd[i] : bv;
  return { data: out, rows: mask.rows, cols: mask.cols };
}

/** Values of `g` where `mask` is set (raster order), like `g[mask]`. */
export function select(g: Grid<ArrayLike<number>>, mask: Grid<ArrayLike<number>>): Float64Array {
  const n = g.rows * g.cols;
  const m = mask.data;
  let k = 0;
  for (let i = 0; i < n; i++) if (m[i]) k++;
  const out = new Float64Array(k);
  k = 0;
  const d = g.data;
  for (let i = 0; i < n; i++) if (m[i]) out[k++] = d[i];
  return out;
}

// ----------------------------------------------------------------------------------------- selection

/** In-place selection (introselect-style quickselect): after the call a[k] holds the k-th smallest. */
function nthElement(a: Float64Array, k: number, lo = 0, hi = a.length - 1): void {
  while (hi > lo) {
    if (hi - lo < 16) {
      // insertion sort for short ranges
      for (let i = lo + 1; i <= hi; i++) {
        const v = a[i];
        let j = i - 1;
        while (j >= lo && a[j] > v) {
          a[j + 1] = a[j];
          j--;
        }
        a[j + 1] = v;
      }
      return;
    }
    // median of three pivot
    const mid = (lo + hi) >>> 1;
    let x = a[lo], y = a[mid], z = a[hi];
    if (x > y) { const t = x; x = y; y = t; }
    if (y > z) { y = z; }
    if (x > y) { y = x; }
    const pivot = y;
    // Hoare partition
    let i = lo, j = hi;
    while (i <= j) {
      while (a[i] < pivot) i++;
      while (a[j] > pivot) j--;
      if (i <= j) {
        const t = a[i]; a[i] = a[j]; a[j] = t;
        i++;
        j--;
      }
    }
    if (k <= j) hi = j;
    else if (k >= i) lo = i;
    else return; // a[j+1..i-1] all equal the pivot
  }
}

function toFloat64Copy(values: ArrayLike<number>): Float64Array {
  const out = new Float64Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = values[i];
  return out;
}

function hasNaN(a: Float64Array): boolean {
  for (let i = 0; i < a.length; i++) if (a[i] !== a[i]) return true;
  return false;
}

/** NumPy's `_lerp`: a + (b-a)*t, computed from the b side when t >= 0.5. */
function lerp(a: number, b: number, t: number): number {
  const diff = b - a;
  return t >= 0.5 ? b - diff * (1 - t) : a + diff * t;
}

function percentileOfCopy(a: Float64Array, q: number): number {
  const n = a.length;
  const vi = (n - 1) * (q / 100);
  const lo = Math.floor(vi);
  const gamma = vi - lo;
  const loIdx = Math.min(Math.max(lo, 0), n - 1);
  const hiIdx = Math.min(loIdx + 1, n - 1);
  nthElement(a, loIdx);
  const vlo = a[loIdx];
  if (hiIdx === loIdx) return lerp(vlo, vlo, gamma);
  let vhi = Infinity;
  for (let i = loIdx + 1; i < n; i++) if (a[i] < vhi) vhi = a[i];
  return lerp(vlo, vhi, gamma);
}

/**
 * `np.percentile(values, q)` with the default 'linear' method (q in [0, 100]).
 * Pass an array of q to get several percentiles at once. NaN in the input gives NaN (like NumPy).
 */
export function percentile(values: ArrayLike<number>, q: number): number;
export function percentile(values: ArrayLike<number>, q: readonly number[]): number[];
export function percentile(values: ArrayLike<number>, q: number | readonly number[]): number | number[] {
  if (values.length === 0) throw new Error("percentile of empty array");
  const a = toFloat64Copy(values);
  const nan = hasNaN(a);
  if (typeof q === "number") return nan ? NaN : percentileOfCopy(a, q);
  return q.map((qq) => (nan ? NaN : percentileOfCopy(a, qq)));
}

/** `np.median(values)` (mean of the two middle values for even n). */
export function median(values: ArrayLike<number>): number {
  const n = values.length;
  if (n === 0) return NaN;
  const a = toFloat64Copy(values);
  if (hasNaN(a)) return NaN;
  const h = n >> 1;
  nthElement(a, h);
  if (n & 1) return a[h];
  let lo = -Infinity;
  for (let i = 0; i < h; i++) if (a[i] > lo) lo = a[i];
  return (lo + a[h]) / 2;
}

/** Median of the finite values only (`np.nanmedian`, also ignores +-inf). */
export function nanmedian(values: ArrayLike<number>): number {
  let k = 0;
  for (let i = 0; i < values.length; i++) if (Number.isFinite(values[i])) k++;
  const a = new Float64Array(k);
  k = 0;
  for (let i = 0; i < values.length; i++) if (Number.isFinite(values[i])) a[k++] = values[i];
  return median(a);
}

/**
 * Raw median absolute deviation from the median: `median(|v - median(v)|)`.
 * Not scaled; multiply by 1.4826 for a Gaussian sigma estimate.
 */
export function mad(values: ArrayLike<number>): number {
  const m = median(values);
  const d = new Float64Array(values.length);
  for (let i = 0; i < values.length; i++) d[i] = Math.abs(values[i] - m);
  return median(d);
}

/**
 * NumPy's pairwise summation (`np.add.reduce` on a contiguous float64 array), bit-for-bit.
 */
export function pairwiseSum(a: ArrayLike<number>, start = 0, n = a.length - start): number {
  if (n < 8) {
    let res = 0;
    for (let i = 0; i < n; i++) res += a[start + i];
    return res;
  }
  if (n <= 128) {
    const r = [a[start], a[start + 1], a[start + 2], a[start + 3], a[start + 4], a[start + 5], a[start + 6], a[start + 7]];
    let i = 8;
    for (; i < n - (n % 8); i += 8) {
      for (let k = 0; k < 8; k++) r[k] += a[start + i + k];
    }
    let res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += a[start + i];
    return res;
  }
  let n2 = n >> 1;
  n2 -= n2 % 8;
  return pairwiseSum(a, start, n2) + pairwiseSum(a, start + n2, n - n2);
}
