/**
 * External contours of a binary image, reproducing
 * `cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)`.
 *
 * Foreground is 8-connected, background 4-connected (Suzuki-Abe). Each contour is traced with
 * OpenCV's `icvFetchContour` (same start pixel, direction codes and search order), so the point
 * sequence matches OpenCV exactly, and contours are returned in OpenCV's order (reverse of the
 * raster order in which their first pixels are met). Components lying inside a hole of another
 * component are skipped, as RETR_EXTERNAL does.
 */
import type { Grid } from "./grid.js";
import { label } from "./measure.js";

export type Point = [x: number, y: number];

const CODE_DX = [1, 1, 0, -1, -1, -1, 0, 1];
const CODE_DY = [0, -1, -1, -1, 0, 1, 1, 1];

/** All external contours, in the order cv2.findContours returns them. Points are [x, y]. */
export function findExternalContours(m: Grid<ArrayLike<number>>): Point[][] {
  const { rows, cols } = m;
  const W = cols + 2, H = rows + 2;
  const pad = new Uint8Array(W * H);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) if (m.data[r * cols + c]) pad[(r + 1) * W + c + 1] = 1;

  // background 4-connected to the frame
  const outer = new Uint8Array(W * H);
  const queue = new Int32Array(W * H);
  let head = 0, tail = 0;
  outer[0] = 1;
  queue[tail++] = 0;
  while (head < tail) {
    const i = queue[head++];
    const y = (i / W) | 0, x = i - y * W;
    const nb = [x > 0 ? i - 1 : -1, x < W - 1 ? i + 1 : -1, y > 0 ? i - W : -1, y < H - 1 ? i + W : -1];
    for (const p of nb) {
      if (p < 0 || outer[p] || pad[p]) continue;
      outer[p] = 1;
      queue[tail++] = p;
    }
  }

  const { labels, n } = label(m, 8);
  const seen = new Uint8Array(n + 1);
  const deltas = [1, -W + 1, -W, -W - 1, -1, W - 1, W, W + 1];
  const contours: Point[][] = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const l = labels.data[r * cols + c];
      if (!l || seen[l]) continue;
      seen[l] = 1; // first raster pixel of this component
      const i0 = (r + 1) * W + c + 1;
      if (!outer[i0 - 1]) continue; // inside a hole of another component
      contours.push(trace(pad, i0, c, r, deltas));
    }
  }
  return contours.reverse();
}

function trace(pad: Uint8Array, i0: number, x0: number, y0: number, deltas: number[]): Point[] {
  const pts: Point[] = [];
  let s = 4;
  const sEnd0 = 4;
  let i1: number;
  do {
    s = (s - 1) & 7;
    i1 = i0 + deltas[s];
  } while (pad[i1] === 0 && s !== sEnd0);
  if (s === sEnd0 && pad[i1] === 0) return [[x0, y0]]; // single pixel
  let x = x0, y = y0;
  let i3 = i0;
  for (;;) {
    let i4 = i3;
    while (s < 15) {
      ++s;
      i4 = i3 + deltas[s & 7];
      if (pad[i4] !== 0) break;
    }
    s &= 7;
    pts.push([x, y]);
    x += CODE_DX[s];
    y += CODE_DY[s];
    if (i4 === i0 && i3 === i1) break;
    i3 = i4;
    s = (s + 4) & 7;
  }
  return pts;
}

/**
 * Outline of the largest external contour (most points, first in OpenCV order on ties), i.e.
 * `max(cv2.findContours(...)[0], key=len)[:, 0, :]`. Empty if the mask is empty.
 */
export function externalContour(m: Grid<ArrayLike<number>>): Point[] {
  const all = findExternalContours(m);
  let best: Point[] = [];
  for (const c of all) if (c.length > best.length) best = c;
  return best;
}
