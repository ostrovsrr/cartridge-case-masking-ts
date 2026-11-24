/**
 * Exact Euclidean distance / feature transform, ported from SciPy's
 * `NI_EuclideanFeatureTransform` (`_ComputeFT` + `_VoronoiFT`, Maurer et al. 2003).
 *
 * Because this is the same algorithm with the same scan order and the same tie rules, both the
 * distances and the nearest-pixel indices (`return_indices=True`) match SciPy exactly, ties included.
 */
import type { Grid } from "./grid.js";

type M = Grid<ArrayLike<number>>;

/** Scratch buffers for one line. */
interface Scratch { f0: Float64Array; f1: Float64Array; g: Int32Array; o0: Float64Array; o1: Float64Array }

/**
 * One `_VoronoiFT` line over positions `base + k * stride`, k < len. `d` is the axis along the
 * line (0 = rows vary, 1 = cols vary) and `coor` the fixed coordinate of the other axis.
 */
function voronoiLine(fr: Int32Array, fc: Int32Array, base: number, stride: number, len: number, d: 0 | 1, coor: number, S: Scratch): void {
  const { f0, f1, g, o0, o1 } = S;
  // f0 = feature component along d, f1 = component of the other axis
  const A = d === 0 ? fr : fc, B = d === 0 ? fc : fr;
  for (let i = 0, p = base; i < len; i++, p += stride) {
    f0[i] = A[p];
    f1[i] = B[p];
  }
  let l = -1;
  for (let i = 0, p = base; i < len; i++, p += stride) {
    if (fr[p] < 0) continue; // SciPy tests the first (row) component
    const fd = f0[i];
    const tw = f1[i] - coor;
    const wR = tw * tw;
    while (l >= 1) {
      const idx1 = g[l], idx2 = g[l - 1];
      const fv1 = f0[idx1];
      const a = fv1 - f0[idx2];
      const b = fd - fv1;
      const c = a + b;
      // uR: middle candidate (g[l]), vR: previous candidate (g[l-1]); Maurer's RemoveFT test
      const cR = f1[idx1] - coor, dR = f1[idx2] - coor;
      const uR = cR * cR, vR = dR * dR;
      if (c * uR - b * vR - a * wR - a * b * c <= 0.0) break;
      --l;
    }
    ++l;
    g[l] = i;
  }
  const maxl = l;
  if (maxl < 0) return;
  l = 0;
  for (let i = 0; i < len; i++) {
    // squared distance; the two-term sum is order independent, so this equals SciPy's rank loop
    let k = g[l];
    let t0 = f0[k] - i, t1 = f1[k] - coor;
    let delta1 = t0 * t0 + t1 * t1;
    while (l < maxl) {
      k = g[l + 1];
      t0 = f0[k] - i; t1 = f1[k] - coor;
      const delta2 = t0 * t0 + t1 * t1;
      if (delta1 <= delta2) break;
      delta1 = delta2;
      ++l;
    }
    o0[i] = f0[g[l]];
    o1[i] = f1[g[l]];
  }
  for (let i = 0, p = base; i < len; i++, p += stride) {
    A[p] = o0[i];
    B[p] = o1[i];
  }
}

/**
 * Feature transform: for every pixel the (row, col) of the nearest pixel where `m` is 0,
 * `ndi.distance_transform_edt(m, return_indices=True)[1]`. If `m` has no zero pixel at all the
 * features are -1 (as SciPy).
 */
export function featureTransform(m: M): { iy: Int32Array; ix: Int32Array } {
  const { rows, cols } = m;
  const n = rows * cols;
  const fr = new Int32Array(n), fc = new Int32Array(n);
  const src = m.data;
  const L = Math.max(rows, cols);
  const S: Scratch = { f0: new Float64Array(L), f1: new Float64Array(L), g: new Int32Array(L), o0: new Float64Array(L), o1: new Float64Array(L) };
  for (let p = 0, r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++, p++) {
      if (src[p]) { fr[p] = -1; fc[p] = -1; } else { fr[p] = r; fc[p] = c; }
    }
  }
  // pass 1: along axis 0 for every column; pass 2: along axis 1 for every row
  for (let c = 0; c < cols; c++) voronoiLine(fr, fc, c, cols, rows, 0, c, S);
  for (let r = 0; r < rows; r++) voronoiLine(fr, fc, r * cols, 1, cols, 1, r, S);
  return { iy: fr, ix: fc };
}

/** `ndi.distance_transform_edt(m)`: Euclidean distance of every nonzero pixel to the nearest zero. */
export function edt(m: M): Grid<Float64Array> {
  const { rows, cols } = m;
  const { iy, ix } = featureTransform(m);
  const out = new Float64Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const p = r * cols + c;
      const dy = iy[p] - r, dx = ix[p] - c;
      out[p] = Math.sqrt(dy * dy + dx * dx);
    }
  }
  return { data: out, rows, cols };
}

/**
 * For every pixel, the flat index (row * cols + col) of the nearest pixel where `invalid` is 0
 * (valid pixels map to themselves): `ndi.distance_transform_edt(invalid, return_indices=True)`.
 * -1 everywhere if there is no valid pixel.
 */
export function edtIndices(invalid: M): Int32Array {
  const { iy, ix } = featureTransform(invalid);
  const n = invalid.rows * invalid.cols;
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) out[i] = iy[i] < 0 ? -1 : iy[i] * invalid.cols + ix[i];
  return out;
}
