/**
 * Small geometric helpers shared by the pipeline stages (port of `cartmask/steps/geometry.py`).
 *
 * All coordinates are in pixels of the working grid, `x` = column and
 * `y` = row (image convention, y grows downward).
 */
import { type Mask, lstsq, mapCoordinates, median } from "../ndimage/index.js";
import type { Grid } from "../ndimage/index.js";

/**
 * Disk mask `(x - cx)^2 + (y - cy)^2 <= r^2` (boundary included).
 *
 * @param shape `[rows, cols]` of the output mask.
 * @param cx Centre column [px].
 * @param cy Centre row [px].
 * @param r Radius [px].
 */
export function disk(shape: readonly [number, number], cx: number, cy: number, r: number): Mask {
  const [rows, cols] = shape;
  const out = new Uint8Array(rows * cols);
  const r2 = r * r;
  for (let y = 0; y < rows; y++) {
    const dy = y - cy;
    const dy2 = dy * dy;
    const o = y * cols;
    for (let x = 0; x < cols; x++) {
      const dx = x - cx;
      out[o + x] = dx * dx + dy2 <= r2 ? 1 : 0;
    }
  }
  return { data: out, rows, cols };
}

/**
 * Algebraic (Kasa) least-squares circle fit.
 *
 * @returns `[cx, cy, r]` in pixels.
 */
export function fitCircle(x: ArrayLike<number>, y: ArrayLike<number>): [number, number, number] {
  const n = x.length;
  const A = new Float64Array(n * 3);
  const b = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    A[3 * i] = 2 * x[i];
    A[3 * i + 1] = 2 * y[i];
    A[3 * i + 2] = 1;
    b[i] = x[i] * x[i] + y[i] * y[i];
  }
  const [cx, cy, c] = lstsq({ data: A, rows: n, cols: 3 }, b);
  return [cx, cy, Math.sqrt(Math.max(c + cx * cx + cy * cy, 1e-12))];
}

/**
 * Circle fit with iterative MAD-based outlier rejection.
 *
 * @param iters Maximum number of refit / re-select rounds.
 * @param k Points farther than `k` x (MAD-estimated sigma) from the circle are dropped.
 * @param minTol Lower bound on the inlier tolerance [px], so a near-perfect fit does not reject everything.
 * @returns `[cx, cy, r, keep]`; `keep` is the 0/1 inlier mask of the last fit. The circle is NaN
 *   when fewer than 8 points were available.
 */
export function robustCircle(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  iters = 6,
  k = 2.5,
  minTol = 1.0,
): [number, number, number, Uint8Array] {
  const n = x.length;
  let keep = new Uint8Array(n).fill(1);
  let cx = NaN, cy = NaN, r = NaN;
  const res = new Float64Array(n);
  for (let it = 0; it < iters; it++) {
    let nk = 0;
    for (let i = 0; i < n; i++) nk += keep[i];
    if (nk < 8) break;
    const xs = new Float64Array(nk), ys = new Float64Array(nk);
    for (let i = 0, j = 0; i < n; i++) {
      if (keep[i]) {
        xs[j] = x[i];
        ys[j] = y[i];
        j++;
      }
    }
    [cx, cy, r] = fitCircle(xs, ys);
    const kept = new Float64Array(nk);
    for (let i = 0, j = 0; i < n; i++) {
      const dx = x[i] - cx, dy = y[i] - cy;
      res[i] = Math.abs(Math.sqrt(dx * dx + dy * dy) - r);
      if (keep[i]) kept[j++] = res[i];
    }
    const mad = median(kept) * 1.4826;
    const tol = Math.max(k * mad, minTol);
    const next = new Uint8Array(n);
    let same = true;
    for (let i = 0; i < n; i++) {
      next[i] = res[i] <= tol ? 1 : 0;
      if (next[i] !== keep[i]) same = false;
    }
    if (same) break;
    keep = next;
  }
  return [cx, cy, r, keep];
}

/**
 * Bilinear interpolation of `img` at sub-pixel points (nearest at the border),
 * `ndi.map_coordinates(img, [ys, xs], order=1, mode="nearest")`.
 */
export function sample(img: Grid<ArrayLike<number>>, xs: ArrayLike<number>, ys: ArrayLike<number>): Float64Array {
  return mapCoordinates(img, ys, xs);
}
