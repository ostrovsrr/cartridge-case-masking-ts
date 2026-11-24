/** Stage 6 - firing-pin drag: the impression's tail outside the strike circle (port of `drag.py`). */
import { binaryOpening, label, labelSizes, pairwiseSum, type Mask } from "../ndimage/index.js";
import type { Params } from "../segment.js";
import { and, any, argmax, rad2deg } from "./np.js";

/** Detected firing-pin drag (returned by {@link findDrag}). */
export interface Drag {
  /** Mask of the drag channel (firing-pin shear), clipped to the primer face. */
  mask: Mask;
  /** Tail tip `[x, y]` [px]. */
  tip: [number, number];
  /** Drag direction [deg], image coordinates (0 = +x, 90 = down). */
  angle: number;
  /** Strike centre to tail tip [mm] (includes the pin radius). */
  lengthMm: number;
  /** 0..1; 1 = the tail points one way, 0 = a symmetric ring. */
  coherence: number;
}

/**
 * Firing-pin drag: the part of the impression outside the strike circle.
 *
 * A hemispherical pin leaves a round crater; when the pin is still protruding while the barrel
 * unlocks it is dragged across the primer and the crater acquires a tail (teardrop). The tail is
 * accepted as drag when it reaches beyond `p.dragMinReach` x core radius, is big enough, and
 * points in one direction (coherence of the tail's direction vectors).
 *
 * @param fpi Firing-pin-impression mask.
 * @param cx Pin-strike centre column [px].
 * @param cy Pin-strike centre row [px].
 * @param rc Pin-strike radius [px].
 * @param face Primer-face mask.
 * @param pxMm Pixel size [mm].
 * @returns The drag, or null when the tail fails any acceptance test.
 */
export function findDrag(fpi: Mask, cx: number, cy: number, rc: number, face: Mask, pxMm: number, p: Params): Drag | null {
  const { rows, cols } = fpi;
  const n = rows * cols;
  const cand = new Uint8Array(n);
  const lim = rc * 1.1;
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const i = y * cols + x;
      if (!fpi.data[i]) continue;
      const dx = x - cx, dy = y - cy;
      if (Math.sqrt(dx * dx + dy * dy) > lim) cand[i] = 1;
    }
  }
  let tail = binaryOpening({ data: cand, rows, cols }, 2);
  if (!any(tail)) return null;
  const { labels, n: nl } = label(tail);
  const sizes = labelSizes(labels, nl).subarray(1);
  const id = argmax(sizes) + 1;
  const t = new Uint8Array(n);
  const vx: number[] = [], vy: number[] = [];
  for (let i = 0; i < n; i++) {
    if (labels.data[i] === id) {
      t[i] = 1;
      const y = Math.floor(i / cols);
      vx.push(i - y * cols - cx);
      vy.push(y - cy);
    }
  }
  tail = { data: t, rows, cols };
  const m = vx.length;
  const dist = new Float64Array(m);
  let dmax = -Infinity;
  for (let i = 0; i < m; i++) {
    dist[i] = Math.sqrt(vx[i] * vx[i] + vy[i] * vy[i]);
    if (dist[i] > dmax) dmax = dist[i];
  }
  if (dmax < p.dragMinReach * rc || m < p.dragMinAreaFrac * Math.PI * rc * rc) return null;
  const w = new Float64Array(m), ax = new Float64Array(m), ay = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    w[i] = Math.max(dist[i] - rc, 0);
    ax[i] = (vx[i] / dist[i]) * w[i];
    ay[i] = (vy[i] / dist[i]) * w[i];
  }
  let ux = pairwiseSum(ax), uy = pairwiseSum(ay);
  const norm = Math.sqrt(ux * ux + uy * uy);
  const coherence = norm / Math.max(pairwiseSum(w), 1e-9); // 1 = one direction, 0 = ring
  if (coherence < p.dragMinCoherence) return null;
  ux = ux / norm;
  uy = uy / norm;
  let length = -Infinity;
  for (let i = 0; i < m; i++) {
    const v = vx[i] * ux + vy[i] * uy;
    if (v > length) length = v;
  }
  return {
    mask: and(tail, face),
    tip: [cx + ux * length, cy + uy * length],
    angle: rad2deg(Math.atan2(uy, ux)),
    lengthMm: length * pxMm,
    coherence,
  };
}
