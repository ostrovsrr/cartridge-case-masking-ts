/**
 * NumPy / Python semantics the pipeline relies on for parity with `cartmask`:
 * round-half-even, floor modulo, `np.arange` / `np.linspace` value construction,
 * first-occurrence argmax, and Python's `format()` for warning messages.
 *
 * Masks are `Grid<Uint8Array>` with 0/1 values; helpers here allocate new grids.
 */
import { type Grid, type Mask, pairwiseSum } from "../ndimage/index.js";

type G = Grid<ArrayLike<number>>;

// ------------------------------------------------------------------------------------ scalars

/** `np.round(x)` / Python `round(x)`: round half to even. */
export function roundHalfEven(x: number): number {
  const f = Math.floor(x);
  const d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/** Python `int(round(x))`. */
export const pyIntRound = roundHalfEven;

/**
 * Python `round(x, n)` for floats: correctly rounded on the exact binary value, ties to even.
 * Non-finite values are returned unchanged.
 */
export function pyRoundN(x: number, n: number): number {
  if (!Number.isFinite(x)) return x;
  if (Math.abs(x) >= 1e15) return x;
  const s = Math.abs(x).toFixed(n + 1 + 24); // exact decimal digits of the double (JS spec)
  const cut = s.indexOf(".") + n + 1;
  const tieDigit = s[cut];
  const rest = s.slice(cut + 1);
  let r = Number(Math.abs(x).toFixed(n)); // ties go up (away from zero)
  if (tieDigit === "5" && /^0*$/.test(rest)) {
    // exact tie: choose the even last digit
    const down = Number(s.slice(0, cut));
    const lastDigit = Number(s[cut - 1] === "." ? s[cut - 2] : s[cut - 1]);
    r = lastDigit % 2 === 0 ? down : r;
  }
  return x < 0 ? -r : r;
}

/** Python `a % b` for floats (result has the sign of `b`), also `np.remainder`. */
export function pyMod(a: number, b: number): number {
  const m = a % b;
  if (m !== 0 && m < 0 !== b < 0) return m + b;
  return m === 0 ? (b < 0 ? -0 : 0) : m;
}

/** `np.hypot`; plain sqrt of the sum of squares (no overflow in this pipeline). */
export function hypot(x: number, y: number): number {
  return Math.sqrt(x * x + y * y);
}

/** `np.deg2rad`. */
export function deg2rad(d: number): number {
  return d * (Math.PI / 180);
}

/** `np.degrees` / `np.rad2deg`. */
export function rad2deg(r: number): number {
  return r * (180 / Math.PI);
}

/** `np.angle(np.exp(1j * x))`: wrap an angle to (-pi, pi]. */
export function wrapAngle(x: number): number {
  return Math.atan2(Math.sin(x), Math.cos(x));
}

// ------------------------------------------------------------------------------------ ranges

/** `np.arange(start, stop, step)` for floats (NumPy fills `start + i * (a[1] - a[0])`). */
export function arange(start: number, stop: number, step = 1): Float64Array {
  const len = Math.max(0, Math.ceil((stop - start) / step));
  const out = new Float64Array(len);
  if (len === 0) return out;
  out[0] = start;
  if (len === 1) return out;
  const next = start + step;
  out[1] = next;
  const delta = next - start;
  for (let i = 2; i < len; i++) out[i] = start + i * delta;
  return out;
}

/** `np.linspace(start, stop, num, endpoint=False)`. */
export function linspaceOpen(start: number, stop: number, num: number): Float64Array {
  const step = (stop - start) / num;
  const out = new Float64Array(num);
  for (let i = 0; i < num; i++) out[i] = i * step + start;
  return out;
}

// ------------------------------------------------------------------------------------ reductions

/** First index of the maximum (`np.argmax`; NaN wins like NumPy). */
export function argmax(a: ArrayLike<number>): number {
  let best = 0;
  let v = a[0];
  if (v !== v) return 0;
  for (let i = 1; i < a.length; i++) {
    const x = a[i];
    if (x !== x) return i;
    if (x > v) {
      v = x;
      best = i;
    }
  }
  return best;
}

/** First index of the minimum (`np.argmin`; NaN wins like NumPy). */
export function argmin(a: ArrayLike<number>): number {
  let best = 0;
  let v = a[0];
  if (v !== v) return 0;
  for (let i = 1; i < a.length; i++) {
    const x = a[i];
    if (x !== x) return i;
    if (x < v) {
      v = x;
      best = i;
    }
  }
  return best;
}

/** `np.mean` of a float64 array (pairwise summation like NumPy). */
export function mean(a: ArrayLike<number>): number {
  return a.length ? pairwiseSum(a) / a.length : NaN;
}

/** `np.sum` of a float64 array (pairwise). */
export function sum(a: ArrayLike<number>): number {
  return pairwiseSum(a);
}

/** Number of set pixels of a mask. */
export function count(m: G): number {
  const d = m.data;
  let s = 0;
  for (let i = 0; i < d.length; i++) if (d[i]) s++;
  return s;
}

export function any(m: G): boolean {
  const d = m.data;
  for (let i = 0; i < d.length; i++) if (d[i]) return true;
  return false;
}

/** Row and column of the set pixels in raster order (`np.nonzero`). */
export function nonzero(m: G): { ys: Int32Array; xs: Int32Array } {
  const n = count(m);
  const ys = new Int32Array(n), xs = new Int32Array(n);
  const { rows, cols, data } = m;
  let k = 0;
  for (let r = 0; r < rows; r++) {
    const o = r * cols;
    for (let c = 0; c < cols; c++) {
      if (data[o + c]) {
        ys[k] = r;
        xs[k] = c;
        k++;
      }
    }
  }
  return { ys, xs };
}

/** Mean of integer coordinates (exact sum, as NumPy's float64 sum of small ints). */
export function intMean(a: Int32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s / a.length;
}

// ------------------------------------------------------------------------------------ masks

export function emptyMask(rows: number, cols: number): Mask {
  return { data: new Uint8Array(rows * cols), rows, cols };
}

/** AND of any number of masks (all the same shape). */
export function and(...ms: G[]): Mask {
  const { rows, cols } = ms[0];
  const n = rows * cols;
  const out = new Uint8Array(n);
  const ds = ms.map((m) => m.data);
  outer: for (let i = 0; i < n; i++) {
    for (const d of ds) if (!d[i]) continue outer;
    out[i] = 1;
  }
  return { data: out, rows, cols };
}

export function or(...ms: G[]): Mask {
  const { rows, cols } = ms[0];
  const n = rows * cols;
  const out = new Uint8Array(n);
  const ds = ms.map((m) => m.data);
  for (let i = 0; i < n; i++) {
    for (const d of ds) {
      if (d[i]) {
        out[i] = 1;
        break;
      }
    }
  }
  return { data: out, rows, cols };
}

export function not(m: G): Mask {
  const n = m.rows * m.cols;
  const out = new Uint8Array(n);
  const d = m.data;
  for (let i = 0; i < n; i++) out[i] = d[i] ? 0 : 1;
  return { data: out, rows: m.rows, cols: m.cols };
}

/** `a & ~b`. */
export function andNot(a: G, b: G): Mask {
  const n = a.rows * a.cols;
  const out = new Uint8Array(n);
  const x = a.data, y = b.data;
  for (let i = 0; i < n; i++) out[i] = x[i] && !y[i] ? 1 : 0;
  return { data: out, rows: a.rows, cols: a.cols };
}

/** Mask of `pred(value, index)`. */
export function maskOf(g: G, pred: (v: number, i: number) => boolean): Mask {
  const n = g.rows * g.cols;
  const out = new Uint8Array(n);
  const d = g.data;
  for (let i = 0; i < n; i++) out[i] = pred(d[i], i) ? 1 : 0;
  return { data: out, rows: g.rows, cols: g.cols };
}

/** `np.where(valid, a, nan)` as a new Float64 grid. */
export function whereValid(a: G, valid: G): Grid<Float64Array> {
  const n = a.rows * a.cols;
  const out = new Float64Array(n);
  const d = a.data, v = valid.data;
  for (let i = 0; i < n; i++) out[i] = v[i] ? d[i] : NaN;
  return { data: out, rows: a.rows, cols: a.cols };
}

/** Per-pixel distance to `(cx, cy)` (`np.hypot(xx - cx, yy - cy)`). */
export function radiusGrid(rows: number, cols: number, cx: number, cy: number): Grid<Float64Array> {
  const out = new Float64Array(rows * cols);
  for (let y = 0; y < rows; y++) {
    const dy = y - cy;
    const o = y * cols;
    for (let x = 0; x < cols; x++) {
      const dx = x - cx;
      out[o + x] = Math.sqrt(dx * dx + dy * dy);
    }
  }
  return { data: out, rows, cols };
}

// ------------------------------------------------------------------------------------ formatting

/** Python `format(x, ".Nf")`. */
export function fmtF(x: number, digits: number): string {
  if (!Number.isFinite(x)) return Number.isNaN(x) ? "nan" : x > 0 ? "inf" : "-inf";
  const r = pyRoundN(x, digits);
  return (Object.is(r, -0) || (r === 0 && x < 0) ? "-" : "") + Math.abs(r).toFixed(digits);
}

/** Python `format(x, ".N%")`. */
export function fmtPct(x: number, digits = 0): string {
  return fmtF(x * 100, digits) + "%";
}

/** Python `format(x, ".Ng")` (general format, trailing zeros removed). */
export function fmtG(x: number, p: number): string {
  if (!Number.isFinite(x)) return Number.isNaN(x) ? "nan" : x > 0 ? "inf" : "-inf";
  if (x === 0) return Object.is(x, -0) ? "-0" : "0";
  const e = Number(x.toExponential(p - 1).split("e")[1]);
  if (e < -4 || e >= p) {
    const [m, ex] = x.toExponential(p - 1).split("e");
    const mm = m.includes(".") ? m.replace(/0+$/, "").replace(/\.$/, "") : m;
    const n = Number(ex);
    return `${mm}e${n < 0 ? "-" : "+"}${String(Math.abs(n)).padStart(2, "0")}`;
  }
  const s = x.toFixed(Math.max(0, p - 1 - e));
  return s.includes(".") ? s.replace(/0+$/, "").replace(/\.$/, "") : s;
}

/** Python `str(float)` for messages such as `f"{p.aperture_max_ratio} x"`. */
export function pyStr(x: number): string {
  if (Number.isInteger(x) && Math.abs(x) < 1e16) return x.toFixed(1);
  return String(x);
}

/** `x ** 2` as NumPy computes it (`x * x`). */
export function sq(x: number): number {
  return x * x;
}
