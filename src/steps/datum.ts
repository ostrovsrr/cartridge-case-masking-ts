/** Stage 2 - datum plane, polarity check and surface roughness (port of `cartmask/steps/datum.py`). */
import { binaryFillHoles, gaussianFilter, lstsq, median, percentile, type Grid, type Mask } from "../ndimage/index.js";
import { fillNearest } from "../preprocess.js";
import { disk } from "./geometry.js";
import { whereValid } from "./np.js";

/**
 * Plane `z = a*x + b*y + c` fitted with iterative 3-sigma (MAD) rejection.
 *
 * @param z Height map [m].
 * @param mask Pixels used for the fit.
 * @param iters Number of refit / re-select rounds.
 * @returns `[coef, sigma]`: `coef = [a, b, c]` (m/px, m/px, m) and the robust standard deviation
 *   of the residual of the inliers [m].
 */
export function robustPlane(z: Grid<Float64Array>, mask: Mask, iters = 5): [Float64Array, number] {
  const { cols } = z;
  let n = 0;
  for (let i = 0; i < mask.data.length; i++) if (mask.data[i]) n++;
  const xx = new Float64Array(n), yy = new Float64Array(n), zz = new Float64Array(n);
  for (let i = 0, k = 0; i < mask.data.length; i++) {
    if (!mask.data[i]) continue;
    const r = (i / cols) | 0;
    yy[k] = r;
    xx[k] = i - r * cols;
    zz[k] = z.data[i];
    k++;
  }
  let keep = new Uint8Array(n).fill(1);
  let coef: Float64Array = new Float64Array(3);
  const res = new Float64Array(n);
  const resid = () => {
    for (let i = 0; i < n; i++) res[i] = zz[i] - (coef[0] * xx[i] + coef[1] * yy[i] + coef[2]);
  };
  const madKept = (): number => {
    let nk = 0;
    for (let i = 0; i < n; i++) nk += keep[i];
    const a = new Float64Array(nk);
    for (let i = 0, j = 0; i < n; i++) if (keep[i]) a[j++] = Math.abs(res[i]);
    return 1.4826 * median(a);
  };
  for (let it = 0; it < iters; it++) {
    let nk = 0;
    for (let i = 0; i < n; i++) nk += keep[i];
    const A = new Float64Array(nk * 3), b = new Float64Array(nk);
    for (let i = 0, j = 0; i < n; i++) {
      if (!keep[i]) continue;
      A[3 * j] = xx[i];
      A[3 * j + 1] = yy[i];
      A[3 * j + 2] = 1;
      b[j] = zz[i];
      j++;
    }
    coef = lstsq({ data: A, rows: nk, cols: 3 }, b);
    resid();
    const s = madKept();
    const lim = 3 * Math.max(s, 1e-9);
    const next = new Uint8Array(n);
    for (let i = 0; i < n; i++) next[i] = Math.abs(res[i]) < lim ? 1 : 0;
    keep = next;
  }
  resid();
  return [coef, madKept()];
}

/**
 * Polarity check: is the middle of the primer dominated by a bump?
 *
 * The firing-pin impression is a pit. If the dominant deviation in the middle of the primer is a
 * bump, the scan is a cast or an inverted export. An unmeasured hole in the middle is itself
 * evidence of a (steep) pit: then raised flow-back is the only visible relief and must not
 * trigger a flip.
 *
 * @returns True when the heights should be negated.
 */
export function isInverted(z: Grid<Float64Array>, valid: Mask, cx: number, cy: number, rp: number): boolean {
  const shape: [number, number] = [z.rows, z.cols];
  const midDisk = disk(shape, cx, cy, 0.6 * rp);
  const filled = binaryFillHoles(valid);
  const d95 = disk(shape, cx, cy, 0.95 * rp);
  const n = z.data.length;
  let nMid = 0, nHoles = 0, nMidDisk = 0, nRing = 0;
  for (let i = 0; i < n; i++) {
    const v = valid.data[i];
    if (midDisk.data[i]) {
      nMidDisk++;
      if (v) nMid++;
      else if (filled.data[i]) nHoles++;
    } else if (v && d95.data[i]) nRing++;
  }
  if (nMid > 100 && nHoles < 0.02 * nMidDisk) {
    const ring = new Float64Array(nRing), dm = new Float64Array(nMid);
    for (let i = 0, a = 0, b = 0; i < n; i++) {
      if (!valid.data[i]) continue;
      if (midDisk.data[i]) dm[b++] = z.data[i];
      else if (d95.data[i]) ring[a++] = z.data[i];
    }
    const ref = median(ring);
    for (let i = 0; i < nMid; i++) dm[i] = dm[i] - ref;
    return percentile(dm, 99.5) > 3 * Math.max(-percentile(dm, 0.5), 1e-9);
  }
  return false;
}

/**
 * Depth below the primer face and the face's roughness.
 *
 * A robust plane is fitted to the outer annulus of the primer face (the middle holds the crater
 * and the flow-back ring). Roughness is the MAD of the high-pass residual on that annulus; it is
 * the noise floor that the firing-pin and aperture thresholds scale with.
 *
 * @returns `[d, roughUm]`: `d = z - plane` [m] (negative = below the primer surface; NaN where
 *   `z` is NaN) and the roughness [um].
 */
export function fitDatum(
  z: Grid<Float64Array>,
  valid: Mask,
  face: Mask,
  cx: number,
  cy: number,
  rp: number,
): [Grid<Float64Array>, number] {
  const { rows, cols } = z;
  const n = rows * cols;
  const inner = disk([rows, cols], cx, cy, 0.45 * rp);
  let ann: Mask = { data: new Uint8Array(n), rows, cols };
  let na = 0;
  for (let i = 0; i < n; i++) {
    if (face.data[i] && valid.data[i] && !inner.data[i]) {
      ann.data[i] = 1;
      na++;
    }
  }
  if (na < 200) {
    ann = { data: new Uint8Array(n), rows, cols };
    na = 0;
    for (let i = 0; i < n; i++) {
      if (face.data[i] && valid.data[i]) {
        ann.data[i] = 1;
        na++;
      }
    }
  }
  const [coef] = robustPlane(z, ann);
  const d = new Float64Array(n);
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const i = y * cols + x;
      d[i] = z.data[i] - (coef[0] * x + coef[1] * y + coef[2]);
    }
  }
  const dg: Grid<Float64Array> = { data: d, rows, cols };
  const df = fillNearest(whereValid(dg, valid));
  const sm = gaussianFilter(df, 4.0);
  const hp = new Float64Array(na);
  for (let i = 0, k = 0; i < n; i++) if (ann.data[i]) hp[k++] = df.data[i] - sm.data[i];
  const m = median(hp);
  for (let i = 0; i < na; i++) hp[i] = Math.abs(hp[i] - m);
  const roughUm = 1.4826 * median(hp) * 1e6;
  return [dg, roughUm];
}
