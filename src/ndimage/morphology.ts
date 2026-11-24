/**
 * Binary morphology with `scipy.ndimage` semantics.
 *
 * Structuring elements are masks with centre `(floor(h/2), floor(w/2))`, so for a structure S:
 *   erosion(p)  = AND_{s in S} in[p + (s - c)]   (outside the image counts as 0, border_value=0)
 *   dilation(p) = OR_{s in S}  in[p - (s - c)]   (SciPy mirrors the structure for dilation)
 * which reproduces SciPy for odd and even sized structures with origin 0.
 */
import type { Grid, Mask } from "./grid.js";

type M = Grid<ArrayLike<number>>;

function mk(rows: number, cols: number, bits: number[]): Mask {
  return { data: Uint8Array.from(bits), rows, cols };
}

/** `generate_binary_structure(2, 1)`: 4-connected cross (SciPy's default). */
export const CROSS3: Mask = mk(3, 3, [0, 1, 0, 1, 1, 1, 0, 1, 0]);
/** `np.ones((3, 3))` / `generate_binary_structure(2, 2)`: 8-connected square. */
export const SQUARE3: Mask = mk(3, 3, [1, 1, 1, 1, 1, 1, 1, 1, 1]);

/** All-ones structure of the given size (e.g. `ones(1, 2*gap+1)` for a 1-D line closing). */
export function ones(rows: number, cols: number): Mask {
  return { data: new Uint8Array(rows * cols).fill(1), rows, cols };
}

/** `(xx - cx)^2 + (yy - cy)^2 <= r^2` on a rows x cols grid (cartmask `disk`; boundary included). */
export function disk(rows: number, cols: number, cx: number, cy: number, r: number): Mask {
  const out = new Uint8Array(rows * cols);
  const r2 = r * r;
  for (let y = 0; y < rows; y++) {
    const dy = (y - cy) ** 2;
    for (let x = 0; x < cols; x++) out[y * cols + x] = (x - cx) ** 2 + dy <= r2 ? 1 : 0;
  }
  return { data: out, rows, cols };
}

/** Disk structuring element of radius k on a (2k+1)^2 grid: `disk((2k+1,)*2, k, k, k)`. */
export function ellipseKernel(k: number): Mask {
  return disk(2 * k + 1, 2 * k + 1, k, k, k);
}

function toMask(m: M): Mask {
  if (m.data instanceof Uint8Array) {
    let ok = true;
    const d = m.data;
    for (let i = 0; i < d.length; i++) if (d[i] > 1) { ok = false; break; }
    if (ok) return m as Mask;
  }
  const n = m.rows * m.cols;
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = m.data[i] ? 1 : 0;
  return { data: out, rows: m.rows, cols: m.cols };
}

function sameStruct(a: Mask, b: Mask): boolean {
  if (a.rows !== b.rows || a.cols !== b.cols) return false;
  for (let i = 0; i < a.data.length; i++) if ((a.data[i] ? 1 : 0) !== b.data[i]) return false;
  return true;
}

interface Offsets { dy: Int32Array; dx: Int32Array; centerTrue: boolean; py0: number; py1: number; px0: number; px1: number }

/** Offsets s - c of the structure's set elements. */
function offsets(st: Mask): Offsets {
  const cy = st.rows >> 1, cx = st.cols >> 1;
  const dy: number[] = [], dx: number[] = [];
  for (let y = 0; y < st.rows; y++) for (let x = 0; x < st.cols; x++) if (st.data[y * st.cols + x]) { dy.push(y - cy); dx.push(x - cx); }
  return {
    dy: Int32Array.from(dy), dx: Int32Array.from(dx),
    centerTrue: !!st.data[cy * st.cols + cx],
    py0: cy, py1: st.rows - 1 - cy, px0: cx, px1: st.cols - 1 - cx,
  };
}

/** One erosion (invert=false) or dilation (invert=true) pass on a 0/1 mask. */
function pass(m: Mask, o: Offsets, dilate: boolean): Mask {
  const { rows, cols } = m;
  const src = m.data;
  // pad by the structure extent with zeros (border_value 0; ignored by dilation)
  const P = Math.max(o.py0, o.py1), Q = Math.max(o.px0, o.px1);
  const W = cols + 2 * Q;
  const pad = new Uint8Array((rows + 2 * P) * W);
  for (let r = 0; r < rows; r++) pad.set(src.subarray(r * cols, (r + 1) * cols), (r + P) * W + Q);
  const k = o.dy.length;
  const off = new Int32Array(k);
  const sgn = dilate ? -1 : 1;
  for (let i = 0; i < k; i++) off[i] = sgn * (o.dy[i] * W + o.dx[i]);
  const out = new Uint8Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    const base = (r + P) * W + Q;
    const ob = r * cols;
    for (let c = 0; c < cols; c++) {
      const p = base + c;
      const v = pad[p];
      if (dilate) {
        if (o.centerTrue && v) { out[ob + c] = 1; continue; }
        let res = 0;
        for (let i = 0; i < k; i++) if (pad[p + off[i]]) { res = 1; break; }
        out[ob + c] = res;
      } else {
        if (o.centerTrue && !v) continue;
        let res = 1;
        for (let i = 0; i < k; i++) if (!pad[p + off[i]]) { res = 0; break; }
        out[ob + c] = res;
      }
    }
  }
  return { data: out, rows, cols };
}

/**
 * Iterated cross / square morphology via an exact two-pass city-block / chessboard distance
 * transform (k iterations of the 3x3 cross = L1 ball of radius k; square = L-inf ball).
 */
function distanceMorph(m: Mask, iterations: number, chess: boolean, dilate: boolean): Mask {
  const { rows, cols } = m;
  const n = rows * cols;
  const INF = 0x3fffffff;
  const d = new Int32Array(n);
  const src = m.data;
  // distance to the nearest source: 1-pixels for dilation, 0-pixels (and the outside) for erosion
  for (let i = 0; i < n; i++) d[i] = (dilate ? src[i] : !src[i]) ? 0 : INF;
  if (!dilate) {
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const b = Math.min(r + 1, c + 1, rows - r, cols - c);
        const i = r * cols + c;
        if (b < d[i]) d[i] = b;
      }
    }
  }
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      let v = d[i];
      if (c > 0 && d[i - 1] + 1 < v) v = d[i - 1] + 1;
      if (r > 0) {
        if (d[i - cols] + 1 < v) v = d[i - cols] + 1;
        if (chess) {
          if (c > 0 && d[i - cols - 1] + 1 < v) v = d[i - cols - 1] + 1;
          if (c < cols - 1 && d[i - cols + 1] + 1 < v) v = d[i - cols + 1] + 1;
        }
      }
      d[i] = v;
    }
  }
  for (let r = rows - 1; r >= 0; r--) {
    for (let c = cols - 1; c >= 0; c--) {
      const i = r * cols + c;
      let v = d[i];
      if (c < cols - 1 && d[i + 1] + 1 < v) v = d[i + 1] + 1;
      if (r < rows - 1) {
        if (d[i + cols] + 1 < v) v = d[i + cols] + 1;
        if (chess) {
          if (c < cols - 1 && d[i + cols + 1] + 1 < v) v = d[i + cols + 1] + 1;
          if (c > 0 && d[i + cols - 1] + 1 < v) v = d[i + cols - 1] + 1;
        }
      }
      d[i] = v;
    }
  }
  const out = new Uint8Array(n);
  if (dilate) for (let i = 0; i < n; i++) out[i] = d[i] <= iterations ? 1 : 0;
  else for (let i = 0; i < n; i++) out[i] = d[i] > iterations ? 1 : 0;
  return { data: out, rows, cols };
}

function equalMasks(a: Mask, b: Mask): boolean {
  const x = a.data, y = b.data;
  for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false;
  return true;
}

function morph(m: M, iterations: number, structure: Mask, dilate: boolean): Mask {
  let cur = toMask(m);
  if (cur === m) cur = { data: cur.data.slice(), rows: cur.rows, cols: cur.cols };
  const st = toMask(structure);
  if (iterations >= 2) {
    const cross = sameStruct(st, CROSS3), square = sameStruct(st, SQUARE3);
    if (cross || square) return distanceMorph(cur, iterations, square, dilate);
  }
  const o = offsets(st);
  if (iterations < 1) {
    for (;;) {
      const next = pass(cur, o, dilate);
      if (equalMasks(next, cur)) return next;
      cur = next;
    }
  }
  for (let i = 0; i < iterations; i++) cur = pass(cur, o, dilate);
  return cur;
}

/** `ndi.binary_erosion(m, structure, iterations)` (border 0; iterations < 1 = until stable). */
export function binaryErosion(m: M, iterations = 1, structure: Mask = CROSS3): Mask {
  return morph(m, iterations, structure, false);
}

/** `ndi.binary_dilation(m, structure, iterations)`. */
export function binaryDilation(m: M, iterations = 1, structure: Mask = CROSS3): Mask {
  return morph(m, iterations, structure, true);
}

/** `ndi.binary_opening`: erosion then dilation, each `iterations` times. */
export function binaryOpening(m: M, iterations = 1, structure: Mask = CROSS3): Mask {
  return binaryDilation(binaryErosion(m, iterations, structure), iterations, structure);
}

/**
 * `ndi.binary_closing`: dilation then erosion. Like SciPy there is no padding, so the erosion
 * treats the outside as 0 and can remove pixels near the image border.
 */
export function binaryClosing(m: M, iterations = 1, structure: Mask = CROSS3): Mask {
  return binaryErosion(binaryDilation(m, iterations, structure), iterations, structure);
}

/**
 * Grow `seed` through `mask` pixels connected by `structure` (`ndi.binary_propagation`).
 * Seed pixels outside the mask are kept (SciPy leaves masked-out pixels unchanged).
 * Without a mask every pixel connected to a seed is set.
 */
export function binaryPropagation(seed: M, mask?: M, structure: Mask = CROSS3): Mask {
  const { rows, cols } = seed;
  const n = rows * cols;
  const out = new Uint8Array(n);
  const md = mask ? mask.data : null;
  const o = offsets(toMask(structure));
  const k = o.dy.length;
  const queue = new Int32Array(n);
  let head = 0, tail = 0;
  for (let i = 0; i < n; i++) if (seed.data[i]) { out[i] = 1; queue[tail++] = i; }
  while (head < tail) {
    const i = queue[head++];
    const r = (i / cols) | 0, c = i - r * cols;
    for (let j = 0; j < k; j++) {
      // q is set => p = q + (s - c) may be set (dilation reads in[p - (s - c)])
      const rr = r + o.dy[j], cc = c + o.dx[j];
      if (rr < 0 || rr >= rows || cc < 0 || cc >= cols) continue;
      const p = rr * cols + cc;
      if (out[p] || (md && !md[p])) continue;
      out[p] = 1;
      queue[tail++] = p;
    }
  }
  return { data: out, rows, cols };
}

/**
 * `ndi.binary_fill_holes(m, structure)`: background pixels not connected (via `structure`,
 * default 4-connectivity) to the image border become foreground.
 */
export function binaryFillHoles(m: M, structure: Mask = CROSS3): Mask {
  const { rows, cols } = m;
  const n = rows * cols;
  const src = m.data;
  const outside = new Uint8Array(n);
  const o = offsets(toMask(structure));
  const k = o.dy.length;
  const queue = new Int32Array(n);
  let head = 0, tail = 0;
  // seeds: background pixels that read an out-of-image neighbour (border_value = 1)
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      if (src[i]) continue;
      for (let j = 0; j < k; j++) {
        const rr = r - o.dy[j], cc = c - o.dx[j];
        if (rr < 0 || rr >= rows || cc < 0 || cc >= cols) { outside[i] = 1; queue[tail++] = i; break; }
      }
    }
  }
  while (head < tail) {
    const i = queue[head++];
    const r = (i / cols) | 0, c = i - r * cols;
    for (let j = 0; j < k; j++) {
      const rr = r + o.dy[j], cc = c + o.dx[j];
      if (rr < 0 || rr >= rows || cc < 0 || cc >= cols) continue;
      const p = rr * cols + cc;
      if (outside[p] || src[p]) continue;
      outside[p] = 1;
      queue[tail++] = p;
    }
  }
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = outside[i] ? 0 : 1;
  return { data: out, rows, cols };
}

// ------------------------------------------------------------------------------ small mask helpers

export function maskAnd(a: M, b: M): Mask {
  const n = a.rows * a.cols;
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = a.data[i] && b.data[i] ? 1 : 0;
  return { data: out, rows: a.rows, cols: a.cols };
}

export function maskOr(a: M, b: M): Mask {
  const n = a.rows * a.cols;
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = a.data[i] || b.data[i] ? 1 : 0;
  return { data: out, rows: a.rows, cols: a.cols };
}

export function maskNot(a: M): Mask {
  const n = a.rows * a.cols;
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = a.data[i] ? 0 : 1;
  return { data: out, rows: a.rows, cols: a.cols };
}

export function maskCount(a: M): number {
  let s = 0;
  const n = a.rows * a.cols;
  for (let i = 0; i < n; i++) if (a.data[i]) s++;
  return s;
}
