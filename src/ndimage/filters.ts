/**
 * Separable linear and rank filters with `scipy.ndimage` semantics (mode 'reflect', origin 0).
 *
 * Axis order, kernel construction and the per-pixel accumulation order follow SciPy's C code
 * (`NI_Correlate1D`, `NI_UniformFilter1D`, `NI_MinOrMaxFilter1D`), so results agree with SciPy to a
 * few ulps.
 */
import { type Grid, pairwiseSum } from "./grid.js";

type G = Grid<ArrayLike<number>>;

/** scipy 'reflect' (half-sample symmetric: d c b a | a b c d | d c b a), periodic with period 2n. */
export function reflectIndex(i: number, n: number): number {
  if (n === 1) return 0;
  if (i >= 0 && i < n) return i;
  const p = 2 * n;
  let m = i % p;
  if (m < 0) m += p;
  return m < n ? m : p - 1 - m;
}

/** Copy `src[off + k*stride]`, k = 0..n-1, into `buf[pad0..pad0+n)` and reflect-extend both ends. */
function fillLine(
  buf: Float64Array,
  src: ArrayLike<number>,
  off: number,
  stride: number,
  n: number,
  pad0: number,
  pad1: number,
): void {
  for (let k = 0; k < n; k++) buf[pad0 + k] = src[off + k * stride];
  for (let k = 1; k <= pad0; k++) buf[pad0 - k] = buf[pad0 + reflectIndex(-k, n)];
  for (let k = 0; k < pad1; k++) buf[pad0 + n + k] = buf[pad0 + reflectIndex(n + k, n)];
}

/** SciPy keeps a float32 input's dtype between axis passes; emulate by rounding in place. */
function roundToFloat32(a: Float64Array): void {
  for (let i = 0; i < a.length; i++) a[i] = Math.fround(a[i]);
}

// ------------------------------------------------------------------------------------- gaussian

/** `scipy.ndimage._filters._gaussian_kernel1d` (order 0 or 1), not yet reversed. */
export function gaussianKernel1d(sigma: number, order: number, radius: number): Float64Array {
  const n = 2 * radius + 1;
  const sigma2 = sigma * sigma;
  const c = -0.5 / sigma2;
  const phi = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const x = i - radius;
    phi[i] = Math.exp(c * (x * x));
  }
  const s = pairwiseSum(phi);
  for (let i = 0; i < n; i++) phi[i] = phi[i] / s;
  if (order === 0) return phi;
  if (order !== 1) throw new Error("gaussianKernel1d: only order 0 and 1 are supported");
  const q1 = 1 / -sigma2;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = (i - radius) * q1 * phi[i];
  return out;
}

/**
 * `ndi.correlate1d(input, weights, axis, mode='reflect')` on a 2-D grid (axis 0 = along columns,
 * axis 1 = along rows). Weights must have odd length for the symmetric fast paths (as in SciPy).
 */
export function correlate1d(g: G, weights: ArrayLike<number>, axis: 0 | 1): Grid<Float64Array> {
  const { rows, cols } = g;
  const src = g.data;
  const out = new Float64Array(rows * cols);
  const fs = weights.length;
  const size1 = fs >> 1;
  const size2 = fs - size1 - 1;
  // symmetry detection exactly as NI_Correlate1D
  let symmetric = 0;
  if (fs & 1) {
    symmetric = 1;
    for (let i = 1; i <= size1; i++) {
      if (Math.abs(weights[size1 + i] - weights[size1 - i]) > Number.EPSILON) {
        symmetric = 0;
        break;
      }
    }
    if (symmetric === 0) {
      symmetric = -1;
      for (let i = 1; i <= size1; i++) {
        if (Math.abs(weights[size1 + i] + weights[size1 - i]) > Number.EPSILON) {
          symmetric = 0;
          break;
        }
      }
    }
  }
  const w = Float64Array.from(weights);
  const n = axis === 0 ? rows : cols;
  if (n === 0 || rows * cols === 0) return { data: out, rows, cols };

  if (axis === 1) {
    const buf = new Float64Array(cols + size1 + size2);
    for (let r = 0; r < rows; r++) {
      fillLine(buf, src, r * cols, 1, cols, size1, size2);
      const o = r * cols;
      if (symmetric > 0) {
        const w0 = w[size1];
        for (let c = 0; c < cols; c++) {
          const ci = c + size1;
          let acc = buf[ci] * w0;
          for (let k = size1; k >= 1; k--) acc += (buf[ci - k] + buf[ci + k]) * w[size1 - k];
          out[o + c] = acc;
        }
      } else if (symmetric < 0) {
        const w0 = w[size1];
        for (let c = 0; c < cols; c++) {
          const ci = c + size1;
          let acc = buf[ci] * w0;
          for (let k = size1; k >= 1; k--) acc += (buf[ci - k] - buf[ci + k]) * w[size1 - k];
          out[o + c] = acc;
        }
      } else {
        for (let c = 0; c < cols; c++) {
          const ci = c + size1;
          let acc = buf[ci + size2] * w[size1 + size2];
          for (let j = -size1; j < size2; j++) acc += buf[ci + j] * w[size1 + j];
          out[o + c] = acc;
        }
      }
    }
    return { data: out, rows, cols };
  }

  // axis 0: whole-row operations (cache friendly), same per-element accumulation order
  const rowIdx = (r: number) => reflectIndex(r, rows) * cols;
  const s = src instanceof Float64Array ? src : Float64Array.from(src);
  for (let r = 0; r < rows; r++) {
    const o = r * cols;
    if (symmetric !== 0) {
      const w0 = w[size1];
      const ic = r * cols;
      for (let c = 0; c < cols; c++) out[o + c] = s[ic + c] * w0;
      for (let k = size1; k >= 1; k--) {
        const a = rowIdx(r - k);
        const b = rowIdx(r + k);
        const wk = w[size1 - k];
        if (symmetric > 0) for (let c = 0; c < cols; c++) out[o + c] += (s[a + c] + s[b + c]) * wk;
        else for (let c = 0; c < cols; c++) out[o + c] += (s[a + c] - s[b + c]) * wk;
      }
    } else {
      const last = rowIdx(r + size2);
      const wl = w[size1 + size2];
      for (let c = 0; c < cols; c++) out[o + c] = s[last + c] * wl;
      for (let j = -size1; j < size2; j++) {
        const a = rowIdx(r + j);
        const wj = w[size1 + j];
        for (let c = 0; c < cols; c++) out[o + c] += s[a + c] * wj;
      }
    }
  }
  return { data: out, rows, cols };
}

export interface GaussianOptions {
  /** Derivative order per axis `[dy, dx]` (0 or 1). */
  order?: [number, number] | number;
  /** Kernel half-width in sigmas (radius = int(truncate * sigma + 0.5)). Default 4. */
  truncate?: number;
}

/**
 * `ndi.gaussian_filter(g, sigma, order, mode='reflect', truncate=4.0)`.
 * `sigma` may be a scalar or `[sigmaY, sigmaX]`; axes with sigma <= 1e-15 are skipped (as SciPy).
 * The result is Float64; for a Float32Array input every pass is rounded to float32, reproducing
 * SciPy's float32 output values.
 */
export function gaussianFilter(g: G, sigma: number | [number, number], opts: GaussianOptions = {}): Grid<Float64Array> {
  const truncate = opts.truncate ?? 4.0;
  const sig = typeof sigma === "number" ? [sigma, sigma] : sigma;
  const ord = opts.order === undefined ? [0, 0] : typeof opts.order === "number" ? [opts.order, opts.order] : opts.order;
  const f32 = g.data instanceof Float32Array;
  let cur: G = g;
  let touched = false;
  for (const axis of [0, 1] as const) {
    const s = sig[axis];
    if (!(s > 1e-15)) continue;
    const radius = Math.trunc(truncate * s + 0.5);
    const k = gaussianKernel1d(s, ord[axis], radius);
    k.reverse(); // gaussian_filter1d correlates with weights[::-1]
    cur = correlate1d(cur, k, axis);
    if (f32) roundToFloat32(cur.data as Float64Array);
    touched = true;
  }
  if (!touched) return { data: Float64Array.from(g.data), rows: g.rows, cols: g.cols };
  return cur as Grid<Float64Array>;
}

/** `ndi.gaussian_filter1d` on a 1-D signal. */
export function gaussianFilter1d(values: ArrayLike<number>, sigma: number, order = 0, truncate = 4.0): Float64Array {
  return gaussianFilter({ data: values, rows: 1, cols: values.length }, [0, sigma], { order: [0, order], truncate }).data;
}

// ------------------------------------------------------------------------------------- uniform

function uniform1d(g: G, size: number, axis: 0 | 1): Grid<Float64Array> {
  const { rows, cols } = g;
  const out = new Float64Array(rows * cols);
  const n = axis === 0 ? rows : cols;
  const m = axis === 0 ? cols : rows;
  const stride = axis === 0 ? cols : 1;
  const lineStep = axis === 0 ? 1 : cols;
  const size1 = size >> 1;
  const size2 = size - size1 - 1;
  const buf = new Float64Array(n + size1 + size2);
  for (let l = 0; l < m; l++) {
    const off = l * lineStep;
    fillLine(buf, g.data, off, stride, n, size1, size2);
    let tmp = 0;
    for (let k = 0; k < size; k++) tmp += buf[k];
    tmp /= size;
    out[off] = tmp;
    for (let i = 1; i < n; i++) {
      tmp += (buf[i + size - 1] - buf[i - 1]) / size;
      out[off + i * stride] = tmp;
    }
  }
  return { data: out, rows, cols };
}

/** `ndi.uniform_filter(g, size)` (mode reflect); size scalar or `[sy, sx]`. */
export function uniformFilter(g: G, size: number | [number, number]): Grid<Float64Array> {
  const sz = typeof size === "number" ? [size, size] : size;
  const f32 = g.data instanceof Float32Array;
  let cur: G = g;
  for (const axis of [0, 1] as const) {
    if (sz[axis] <= 1) continue;
    cur = uniform1d(cur, sz[axis], axis);
    if (f32) roundToFloat32(cur.data as Float64Array);
  }
  return cur === g ? { data: Float64Array.from(g.data), rows: g.rows, cols: g.cols } : (cur as Grid<Float64Array>);
}

// ------------------------------------------------------------------------------------- maximum

function max1d(g: G, size: number, axis: 0 | 1): Grid<Float64Array> {
  const { rows, cols } = g;
  const out = new Float64Array(rows * cols);
  const n = axis === 0 ? rows : cols;
  const m = axis === 0 ? cols : rows;
  const stride = axis === 0 ? cols : 1;
  const lineStep = axis === 0 ? 1 : cols;
  const size1 = size >> 1;
  const size2 = size - size1 - 1;
  const L = n + size1 + size2;
  const buf = new Float64Array(L);
  const dq = new Int32Array(L);
  for (let l = 0; l < m; l++) {
    const off = l * lineStep;
    fillLine(buf, g.data, off, stride, n, size1, size2);
    // window for output i is buf[i .. i+size-1]
    let head = 0, tail = 0;
    for (let j = 0; j < L; j++) {
      const v = buf[j];
      while (tail > head && buf[dq[tail - 1]] <= v) tail--;
      dq[tail++] = j;
      const i = j - size + 1;
      if (i >= 0) {
        while (dq[head] < i) head++;
        out[off + i * stride] = buf[dq[head]];
      }
    }
  }
  return { data: out, rows, cols };
}

/** `ndi.maximum_filter(g, size)` (separable, mode reflect); size scalar or `[sy, sx]`. */
export function maximumFilter(g: G, size: number | [number, number]): Grid<Float64Array> {
  const sz = typeof size === "number" ? [size, size] : size;
  let cur: G = g;
  for (const axis of [0, 1] as const) if (sz[axis] > 1) cur = max1d(cur, sz[axis], axis);
  return cur === g ? { data: Float64Array.from(g.data), rows: g.rows, cols: g.cols } : (cur as Grid<Float64Array>);
}

// ------------------------------------------------------------------------------------- median

/** k-th smallest of a[0..n) (destructive). */
function kth(a: Float64Array, n: number, k: number): number {
  let lo = 0, hi = n - 1;
  while (hi > lo) {
    const pivot = a[(lo + hi) >>> 1];
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
    else break;
  }
  return a[k];
}

/**
 * `ndi.median_filter(g, size)` with a full size x size footprint, mode reflect. As in SciPy the
 * result is the element of rank `floor(size_total / 2)` (upper median for even windows).
 */
export function medianFilter(g: G, size: number | [number, number]): Grid<Float64Array> {
  const [sy, sx] = typeof size === "number" ? [size, size] : size;
  const { rows, cols } = g;
  const out = new Float64Array(rows * cols);
  const oy = sy >> 1, ox = sx >> 1;
  const ri = new Int32Array(sy), ci = new Int32Array(cols * sx);
  for (let c = 0; c < cols; c++) for (let k = 0; k < sx; k++) ci[c * sx + k] = reflectIndex(c - ox + k, cols);
  const win = new Float64Array(sy * sx);
  const rank = (sy * sx) >> 1;
  const d = g.data;
  for (let r = 0; r < rows; r++) {
    for (let k = 0; k < sy; k++) ri[k] = reflectIndex(r - oy + k, rows) * cols;
    for (let c = 0; c < cols; c++) {
      let n = 0;
      for (let a = 0; a < sy; a++) for (let b = 0; b < sx; b++) win[n++] = d[ri[a] + ci[c * sx + b]];
      out[r * cols + c] = kth(win, n, rank);
    }
  }
  return { data: out, rows, cols };
}

/** `ndi.median_filter(arr, size)` on a 1-D array (mode reflect). */
export function medianFilter1d(arr: ArrayLike<number>, size: number): Float64Array {
  return medianFilter({ data: arr, rows: 1, cols: arr.length }, [1, size]).data;
}

// ------------------------------------------------------------------------------------- gradient

/**
 * `np.gradient(g, spacing)` (edge_order=1): central differences inside, one-sided at the edges.
 * Returns `{ gy, gx }` (axis 0, axis 1). An axis of length < 2 yields zeros (NumPy would raise).
 */
export function gradient(g: G, spacing: number | [number, number] = 1): { gy: Grid<Float64Array>; gx: Grid<Float64Array> } {
  const [dy, dx] = typeof spacing === "number" ? [spacing, spacing] : spacing;
  const { rows, cols } = g;
  const d = g.data;
  const gy = new Float64Array(rows * cols);
  const gx = new Float64Array(rows * cols);
  if (rows >= 2) {
    const two = 2 * dy;
    for (let r = 0; r < rows; r++) {
      const o = r * cols;
      if (r === 0) for (let c = 0; c < cols; c++) gy[c] = (d[cols + c] - d[c]) / dy;
      else if (r === rows - 1) for (let c = 0; c < cols; c++) gy[o + c] = (d[o + c] - d[o - cols + c]) / dy;
      else for (let c = 0; c < cols; c++) gy[o + c] = (d[o + cols + c] - d[o - cols + c]) / two;
    }
  }
  if (cols >= 2) {
    const two = 2 * dx;
    for (let r = 0; r < rows; r++) {
      const o = r * cols;
      gx[o] = (d[o + 1] - d[o]) / dx;
      for (let c = 1; c < cols - 1; c++) gx[o + c] = (d[o + c + 1] - d[o + c - 1]) / two;
      gx[o + cols - 1] = (d[o + cols - 1] - d[o + cols - 2]) / dx;
    }
  }
  return { gy: { data: gy, rows, cols }, gx: { data: gx, rows, cols } };
}
