/**
 * Resampling: `ndi.map_coordinates` (order 1, mode nearest) and the OpenCV geometric transforms
 * the pipeline uses (`cv2.resize` INTER_AREA / INTER_NEAREST / INTER_LINEAR, `cv2.warpAffine`,
 * `cv2.getRotationMatrix2D`).
 *
 * The OpenCV ports follow `imgproc/src/resize.cpp` and `imgwarp.cpp` (4.x): float32 interpolation
 * coefficients, the INTER_AREA decimation tables, the integer-factor "area fast" path, and
 * warpAffine's fixed-point coordinates (1/32 pixel bilinear table, 1/1024 pixel nearest).
 * Float64 inputs are computed in double like OpenCV's CV_64F path; for Float32Array inputs the
 * float arithmetic is emulated with Math.fround.
 */
import type { AnyArray, Grid } from "./grid.js";

type G = Grid<ArrayLike<number>>;
const f32 = Math.fround;
const id = (x: number): number => x;

/** cvRound / saturate_cast<int>(double): round half to even. */
function cvRound(x: number): number {
  const r = Math.round(x);
  return r - x === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

// ------------------------------------------------------------------------------- map_coordinates

/**
 * `ndi.map_coordinates(g, [ys, xs], order=1, mode='nearest')`: bilinear samples at (ys[i], xs[i]);
 * coordinates outside the grid are clamped to the edge.
 */
export function mapCoordinates(g: G, ys: ArrayLike<number>, xs: ArrayLike<number>): Float64Array {
  const { rows, cols } = g;
  const d = g.data;
  const n = ys.length;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let y = ys[i], x = xs[i];
    if (y < 0) y = 0; else if (y > rows - 1) y = rows - 1;
    if (x < 0) x = 0; else if (x > cols - 1) x = cols - 1;
    const y0 = Math.floor(y), x0 = Math.floor(x);
    const ty = y - y0, tx = x - x0;
    const y1 = y0 + 1 < rows ? y0 + 1 : rows - 1;
    const x1 = x0 + 1 < cols ? x0 + 1 : cols - 1;
    const wy0 = 1 - ty, wx0 = 1 - tx;
    const r0 = y0 * cols, r1 = y1 * cols;
    let t = 0;
    t += d[r0 + x0] * wy0 * wx0;
    t += d[r0 + x1] * wy0 * tx;
    t += d[r1 + x0] * ty * wx0;
    t += d[r1 + x1] * ty * tx;
    out[i] = t;
  }
  return out;
}

// ------------------------------------------------------------------------------------- resize

function newLike(src: ArrayLike<number>, n: number): AnyArray {
  const C = (src as AnyArray).constructor as { new (n: number): AnyArray } | undefined;
  if (C && ArrayBuffer.isView(src)) return new C(n);
  return new Float64Array(n);
}

/** `cv2.resize(g, (cols, rows), interpolation=cv2.INTER_NEAREST)`; keeps the array type. */
export function resizeNearest<T extends AnyArray>(g: Grid<T>, rows: number, cols: number): Grid<T> {
  const sw = g.cols, sh = g.rows;
  const out = newLike(g.data, rows * cols) as T;
  if (rows === sh && cols === sw) {
    out.set(g.data as ArrayLike<number>);
    return { data: out, rows, cols };
  }
  const ifx = 1 / (cols / sw), ify = 1 / (rows / sh);
  const xo = new Int32Array(cols);
  for (let x = 0; x < cols; x++) xo[x] = Math.min(Math.floor(x * ifx), sw - 1);
  for (let y = 0; y < rows; y++) {
    const sy = Math.min(Math.floor(y * ify), sh - 1);
    const sr = sy * sw, dr = y * cols;
    for (let x = 0; x < cols; x++) out[dr + x] = g.data[sr + xo[x]];
  }
  return { data: out, rows, cols };
}

interface Decimate { di: Int32Array; si: Int32Array; alpha: Float64Array; n: number }

/** OpenCV `computeResizeAreaTab` (alpha rounded to float32 like DecimateAlpha::alpha). */
function areaTab(ssize: number, dsize: number, scale: number): Decimate {
  const di: number[] = [], si: number[] = [], al: number[] = [];
  for (let dx = 0; dx < dsize; dx++) {
    const fsx1 = dx * scale;
    const fsx2 = fsx1 + scale;
    const cellWidth = Math.min(scale, ssize - fsx1);
    let sx1 = Math.ceil(fsx1), sx2 = Math.floor(fsx2);
    sx2 = Math.min(sx2, ssize - 1);
    sx1 = Math.min(sx1, sx2);
    if (sx1 - fsx1 > 1e-3) { di.push(dx); si.push(sx1 - 1); al.push(f32((sx1 - fsx1) / cellWidth)); }
    for (let sx = sx1; sx < sx2; sx++) { di.push(dx); si.push(sx); al.push(f32(1.0 / cellWidth)); }
    if (fsx2 - sx2 > 1e-3) {
      di.push(dx); si.push(sx2);
      al.push(f32(Math.min(Math.min(fsx2 - sx2, 1), cellWidth) / cellWidth));
    }
  }
  return { di: Int32Array.from(di), si: Int32Array.from(si), alpha: Float64Array.from(al), n: di.length };
}

function resizeAreaGeneric(g: G, rows: number, cols: number, scaleX: number, scaleY: number, R: (x: number) => number): Float64Array {
  const src = g.data, sw = g.cols;
  const xt = areaTab(g.cols, cols, scaleX);
  const yt = areaTab(g.rows, rows, scaleY);
  const out = new Float64Array(rows * cols);
  const buf = new Float64Array(cols), sum = new Float64Array(cols);
  let prevDy = yt.n > 0 ? yt.di[0] : 0;
  for (let j = 0; j < yt.n; j++) {
    const beta = yt.alpha[j];
    const dy = yt.di[j];
    const so = yt.si[j] * sw;
    buf.fill(0);
    for (let k = 0; k < xt.n; k++) {
      const dxn = xt.di[k];
      buf[dxn] = R(buf[dxn] + R(src[so + xt.si[k]] * xt.alpha[k]));
    }
    if (dy !== prevDy) {
      const o = prevDy * cols;
      for (let dx = 0; dx < cols; dx++) {
        out[o + dx] = sum[dx];
        sum[dx] = R(beta * buf[dx]);
      }
      prevDy = dy;
    } else {
      for (let dx = 0; dx < cols; dx++) sum[dx] = R(sum[dx] + R(beta * buf[dx]));
    }
  }
  const o = prevDy * cols;
  for (let dx = 0; dx < cols; dx++) out[o + dx] = sum[dx];
  return out;
}

function resizeAreaFast(g: G, rows: number, cols: number, isx: number, isy: number, R: (x: number) => number): Float64Array {
  const src = g.data, sw = g.cols, sh = g.rows;
  const area = isx * isy;
  const scale = f32(1 / area);
  const ofs = new Int32Array(area);
  for (let sy = 0, k = 0; sy < isy; sy++) for (let sx = 0; sx < isx; sx++) ofs[k++] = sy * sw + sx;
  const out = new Float64Array(rows * cols);
  const dwidth1 = Math.trunc(sw / isx);
  for (let dy = 0; dy < rows; dy++) {
    const sy0 = dy * isy;
    const o = dy * cols;
    if (sy0 >= sh) continue;
    const w = sy0 + isy <= sh ? dwidth1 : 0;
    let dx = 0;
    for (; dx < w; dx++) {
      const base = sy0 * sw + dx * isx;
      let sum = 0;
      let k = 0;
      for (; k <= area - 4; k += 4) {
        sum = R(sum + R(R(R(src[base + ofs[k]] + src[base + ofs[k + 1]]) + src[base + ofs[k + 2]]) + src[base + ofs[k + 3]]));
      }
      for (; k < area; k++) sum = R(sum + src[base + ofs[k]]);
      out[o + dx] = R(sum * scale);
    }
    for (; dx < cols; dx++) {
      let sum = 0, count = 0;
      const sx0 = dx * isx;
      if (sx0 >= sw) { out[o + dx] = 0; continue; }
      for (let sy = 0; sy < isy; sy++) {
        if (sy0 + sy >= sh) break;
        for (let sx = 0; sx < isx; sx++) {
          if (sx0 + sx >= sw) break;
          sum = R(sum + src[(sy0 + sy) * sw + sx0 + sx]);
          count++;
        }
      }
      out[o + dx] = f32(f32(sum) / count);
    }
  }
  return out;
}

/** Linear (and INTER_AREA up-scaling) via OpenCV's resizeGeneric_ with float32 coefficients. */
function resizeLinearGeneric(g: G, rows: number, cols: number, area: boolean, R: (x: number) => number): Float64Array {
  const src = g.data, sw = g.cols, sh = g.rows;
  const invX = cols / sw, invY = rows / sh;
  const scaleX = 1 / invX, scaleY = 1 / invY;
  const coeffs = (dsize: number, ssize: number, scale: number, inv: number) => {
    const ofs = new Int32Array(dsize);
    const a0 = new Float64Array(dsize), a1 = new Float64Array(dsize);
    let xmax = dsize;
    for (let dx = 0; dx < dsize; dx++) {
      let sx: number, fx: number;
      if (!area) {
        // OpenCV 4.11's CV_64F INTER_LINEAR uses double coefficients (verified against cv2)
        fx = (dx + 0.5) * (ssize / dsize) - 0.5;
        sx = Math.floor(fx);
        fx = fx - sx;
      } else {
        sx = Math.floor(dx * scale);
        fx = f32((dx + 1) - (sx + 1) * inv);
        fx = fx <= 0 ? 0 : f32(fx - Math.floor(fx));
      }
      if (sx < 0) { fx = 0; sx = 0; }
      if (sx + 1 >= ssize) {
        xmax = Math.min(xmax, dx);
        if (sx >= ssize - 1) { fx = 0; sx = ssize - 1; }
      }
      ofs[dx] = sx;
      a0[dx] = area ? f32(1 - fx) : 1 - fx;
      a1[dx] = fx;
    }
    return { ofs, a0, a1, xmax };
  };
  const X = coeffs(cols, sw, scaleX, invX);
  const Y = coeffs(rows, sh, scaleY, invY);
  // horizontal pass per source row (cached), then vertical blend
  const hrow = new Map<number, Float64Array>();
  const hres = (sy: number): Float64Array => {
    let h = hrow.get(sy);
    if (h) return h;
    h = new Float64Array(cols);
    const o = sy * sw;
    let dx = 0;
    for (; dx < X.xmax; dx++) {
      const sx = X.ofs[dx];
      h[dx] = R(R(src[o + sx] * X.a0[dx]) + R(src[o + sx + 1] * X.a1[dx]));
    }
    for (; dx < cols; dx++) h[dx] = src[o + X.ofs[dx]];
    hrow.set(sy, h);
    return h;
  };
  const out = new Float64Array(rows * cols);
  for (let dy = 0; dy < rows; dy++) {
    const sy = Y.ofs[dy];
    const s0 = hres(Math.min(Math.max(sy, 0), sh - 1));
    const s1 = hres(Math.min(Math.max(sy + 1, 0), sh - 1));
    const b0 = Y.a0[dy], b1 = Y.a1[dy];
    const o = dy * cols;
    for (let dx = 0; dx < cols; dx++) out[o + dx] = R(R(s0[dx] * b0) + R(s1[dx] * b1));
    for (const k of hrow.keys()) if (k < sy) hrow.delete(k);
  }
  return out;
}

function finish(g: G, data: Float64Array, rows: number, cols: number): Grid<Float64Array> {
  if (g.data instanceof Float32Array) for (let i = 0; i < data.length; i++) data[i] = f32(data[i]);
  return { data, rows, cols };
}

function resizeImpl(g: G, rows: number, cols: number, mode: "area" | "linear"): Grid<Float64Array> {
  const R = g.data instanceof Float32Array ? f32 : id;
  if (rows === g.rows && cols === g.cols) return { data: Float64Array.from(g.data), rows, cols };
  const invX = cols / g.cols, invY = rows / g.rows;
  const scaleX = 1 / invX, scaleY = 1 / invY;
  const isx = cvRound(scaleX), isy = cvRound(scaleY);
  const fast = Math.abs(scaleX - isx) < Number.EPSILON && Math.abs(scaleY - isy) < Number.EPSILON;
  let m = mode;
  if (m === "linear" && fast && isx === 2 && isy === 2) m = "area";
  if (m === "area" && scaleX >= 1 && scaleY >= 1) {
    const data = fast ? resizeAreaFast(g, rows, cols, isx, isy, R) : resizeAreaGeneric(g, rows, cols, scaleX, scaleY, R);
    return finish(g, data, rows, cols);
  }
  return finish(g, resizeLinearGeneric(g, rows, cols, m === "area", R), rows, cols);
}

/** `cv2.resize(g, (cols, rows), interpolation=cv2.INTER_AREA)` (Float64 result). */
export function resizeArea(g: G, rows: number, cols: number): Grid<Float64Array> {
  return resizeImpl(g, rows, cols, "area");
}

/** `cv2.resize(g, (cols, rows), interpolation=cv2.INTER_LINEAR)` (Float64 result). */
export function resizeLinear(g: G, rows: number, cols: number): Grid<Float64Array> {
  return resizeImpl(g, rows, cols, "linear");
}

// ------------------------------------------------------------------------------------- warp

/** `cv2.getRotationMatrix2D((cx, cy), angleDeg, scale)` as a row-major 2x3 matrix. */
export function rotationMatrix(cx: number, cy: number, angleDeg: number, scale = 1): Float64Array {
  const x = f32(cx), y = f32(cy); // the C++ API takes Point2f
  const angle = angleDeg * (Math.PI / 180);
  const alpha = Math.cos(angle) * scale;
  const beta = Math.sin(angle) * scale;
  return Float64Array.from([alpha, beta, (1 - alpha) * x - beta * y, -beta, alpha, beta * x + (1 - alpha) * y]);
}

/** Invert a 2x3 affine matrix exactly as cv::warpAffine does (without WARP_INVERSE_MAP). */
export function invertAffine(M: ArrayLike<number>): Float64Array {
  const m = Float64Array.from(M);
  let D = m[0] * m[4] - m[1] * m[3];
  D = D !== 0 ? 1 / D : 0;
  const A11 = m[4] * D, A22 = m[0] * D;
  m[0] = A11; m[1] *= -D;
  m[3] *= -D; m[4] = A22;
  const b1 = -m[0] * m[2] - m[1] * m[5];
  const b2 = -m[3] * m[2] - m[4] * m[5];
  m[2] = b1; m[5] = b2;
  return m;
}

export interface WarpOptions {
  /** INTER_NEAREST instead of INTER_LINEAR. */
  nearest?: boolean;
  /** Constant border value (default 0). */
  border?: number;
  /** M already maps destination to source (cv2.WARP_INVERSE_MAP). */
  inverse?: boolean;
}

/**
 * `cv2.warpAffine(g, M, (cols, rows), flags, borderMode=BORDER_CONSTANT, borderValue=border)`.
 * Nearest keeps the input array type; bilinear returns Float64 (float32-rounded for Float32 input).
 */
export function warpAffine<T extends AnyArray>(g: Grid<T>, M: ArrayLike<number>, rows: number, cols: number, opts: WarpOptions & { nearest: true }): Grid<T>;
export function warpAffine(g: Grid<AnyArray>, M: ArrayLike<number>, rows: number, cols: number, opts?: WarpOptions & { nearest?: false }): Grid<Float64Array>;
export function warpAffine(g: Grid<AnyArray>, M: ArrayLike<number>, rows: number, cols: number, opts?: WarpOptions): Grid<AnyArray>;
export function warpAffine(g: Grid<AnyArray>, M: ArrayLike<number>, rows: number, cols: number, opts: WarpOptions = {}): Grid<AnyArray> {
  const nearest = !!opts.nearest;
  const cval = opts.border ?? 0;
  const m = opts.inverse ? Float64Array.from(M) : invertAffine(M);
  const AB_BITS = 10, AB_SCALE = 1 << AB_BITS, INTER_BITS = 5, TAB = 1 << INTER_BITS;
  const roundDelta = nearest ? AB_SCALE / 2 : AB_SCALE / TAB / 2;
  const adelta = new Int32Array(cols), bdelta = new Int32Array(cols);
  for (let x = 0; x < cols; x++) {
    adelta[x] = cvRound(m[0] * x * AB_SCALE);
    bdelta[x] = cvRound(m[3] * x * AB_SCALE);
  }
  const sw = g.cols, sh = g.rows;
  const src = g.data;
  const sat16 = (v: number) => (v > 32767 ? 32767 : v < -32768 ? -32768 : v);
  if (nearest) {
    const out = newLike(src, rows * cols);
    for (let y = 0; y < rows; y++) {
      const X0 = cvRound((m[1] * y + m[2]) * AB_SCALE) + roundDelta;
      const Y0 = cvRound((m[4] * y + m[5]) * AB_SCALE) + roundDelta;
      const o = y * cols;
      for (let x = 0; x < cols; x++) {
        const sx = sat16((X0 + adelta[x]) >> AB_BITS);
        const sy = sat16((Y0 + bdelta[x]) >> AB_BITS);
        out[o + x] = sx >= 0 && sx < sw && sy >= 0 && sy < sh ? src[sy * sw + sx] : cval;
      }
    }
    return { data: out, rows, cols };
  }
  const R = src instanceof Float32Array ? f32 : id;
  const out = new Float64Array(rows * cols);
  const shift = AB_BITS - INTER_BITS;
  const inv = 1 / TAB;
  for (let y = 0; y < rows; y++) {
    const X0 = cvRound((m[1] * y + m[2]) * AB_SCALE) + roundDelta;
    const Y0 = cvRound((m[4] * y + m[5]) * AB_SCALE) + roundDelta;
    const o = y * cols;
    for (let x = 0; x < cols; x++) {
      const X = (X0 + adelta[x]) >> shift;
      const Y = (Y0 + bdelta[x]) >> shift;
      const sx = sat16(X >> INTER_BITS), sy = sat16(Y >> INTER_BITS);
      // OpenCV's float table: 1-x and x at 1/32 steps, products rounded to float32 (exact here)
      const tx = f32((X & (TAB - 1)) * inv), ty = f32((Y & (TAB - 1)) * inv);
      const w0 = f32(f32(1 - ty) * f32(1 - tx)), w1 = f32(f32(1 - ty) * tx);
      const w2 = f32(ty * f32(1 - tx)), w3 = f32(ty * tx);
      let v0: number, v1: number, v2: number, v3: number;
      if (sx >= 0 && sx < sw - 1 && sy >= 0 && sy < sh - 1) {
        const p = sy * sw + sx;
        v0 = src[p]; v1 = src[p + 1]; v2 = src[p + sw]; v3 = src[p + sw + 1];
      } else if (sx >= sw || sx + 1 < 0 || sy >= sh || sy + 1 < 0) {
        out[o + x] = cval;
        continue;
      } else {
        const x0ok = sx >= 0 && sx < sw, x1ok = sx + 1 >= 0 && sx + 1 < sw;
        const y0ok = sy >= 0 && sy < sh, y1ok = sy + 1 >= 0 && sy + 1 < sh;
        v0 = x0ok && y0ok ? src[sy * sw + sx] : cval;
        v1 = x1ok && y0ok ? src[sy * sw + sx + 1] : cval;
        v2 = x0ok && y1ok ? src[(sy + 1) * sw + sx] : cval;
        v3 = x1ok && y1ok ? src[(sy + 1) * sw + sx + 1] : cval;
      }
      out[o + x] = R(R(R(R(v0 * w0) + R(v1 * w1)) + R(v2 * w2)) + R(v3 * w3));
    }
  }
  return { data: out, rows, cols };
}
