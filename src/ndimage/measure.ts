/**
 * Connected-component labelling and per-label reductions (`ndi.label`, `ndi.sum`).
 */
import type { Grid, Labels } from "./grid.js";

type M = Grid<ArrayLike<number>>;

/**
 * `ndi.label(m)` (connectivity 4 = SciPy default cross) or `ndi.label(m, np.ones((3,3)))` (8).
 * Labels are numbered 1..n in raster order of each component's first pixel, exactly like SciPy.
 */
export function label(m: M, connectivity: 4 | 8 = 4): { labels: Labels; n: number } {
  const { rows, cols } = m;
  const total = rows * cols;
  const src = m.data;
  const lab = new Int32Array(total);
  const queue = new Int32Array(total);
  const eight = connectivity === 8;
  let n = 0;
  for (let start = 0; start < total; start++) {
    if (!src[start] || lab[start]) continue;
    const id = ++n;
    lab[start] = id;
    let head = 0, tail = 0;
    queue[tail++] = start;
    while (head < tail) {
      const i = queue[head++];
      const r = (i / cols) | 0, c = i - r * cols;
      const r0 = r > 0 ? r - 1 : r, r1 = r < rows - 1 ? r + 1 : r;
      const c0 = c > 0 ? c - 1 : c, c1 = c < cols - 1 ? c + 1 : c;
      for (let rr = r0; rr <= r1; rr++) {
        for (let cc = c0; cc <= c1; cc++) {
          if (!eight && rr !== r && cc !== c) continue;
          const p = rr * cols + cc;
          if (src[p] && !lab[p]) {
            lab[p] = id;
            queue[tail++] = p;
          }
        }
      }
    }
  }
  return { labels: { data: lab, rows, cols }, n };
}

/**
 * `ndi.sum(values, labels, index=range(n+1))` as a Float64Array of length n+1 (index 0 =
 * background). Summed in raster order like `np.bincount(labels, weights)`.
 */
export function labelSums(values: M, labels: Labels, n: number): Float64Array {
  const out = new Float64Array(n + 1);
  const total = labels.rows * labels.cols;
  const lab = labels.data, v = values.data;
  for (let i = 0; i < total; i++) {
    const l = lab[i];
    if (l >= 0 && l <= n) out[l] += v[i];
  }
  return out;
}

/** Pixel count per label, length n+1 (index 0 = background). */
export function labelSizes(labels: Labels, n: number): Int32Array {
  const out = new Int32Array(n + 1);
  const total = labels.rows * labels.cols;
  const lab = labels.data;
  for (let i = 0; i < total; i++) {
    const l = lab[i];
    if (l >= 0 && l <= n) out[l]++;
  }
  return out;
}

/** 0/1 mask of the pixels whose label is in `keep` (a set of label ids). */
export function labelMask(labels: Labels, keep: Iterable<number>): Grid<Uint8Array> {
  const total = labels.rows * labels.cols;
  const lab = labels.data;
  let max = 0;
  for (let i = 0; i < total; i++) if (lab[i] > max) max = lab[i];
  const lut = new Uint8Array(max + 1);
  for (const k of keep) if (k >= 0 && k <= max) lut[k] = 1;
  const out = new Uint8Array(total);
  for (let i = 0; i < total; i++) out[i] = lab[i] >= 0 ? lut[lab[i]] : 0;
  return { data: out, rows: labels.rows, cols: labels.cols };
}
