/** Resampling, gap filling and shaded rendering of height maps (port of `cartmask/preprocess.py`). */
import { edtIndices, gaussianFilter, percentile, resizeArea } from "./ndimage/index.js";
import type { Grid, Mask } from "./ndimage/index.js";
import type { Surface } from "./x3p.js";

/** Python's built-in `round()` (round half to even). */
function pyRound(v: number): number {
  const f = Math.floor(v);
  const d = v - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/**
 * Area-average the height map to a coarser isotropic grid.
 *
 * Invalid (NaN) samples are excluded from the average; an output pixel is
 * invalid when fewer than half of its source samples were valid.
 */
export function resample(surf: Surface, umPerPx: number): Surface {
  let sx = (surf.dx * 1e6) / umPerPx; // output/input size ratio per axis
  let sy = (surf.dy * 1e6) / umPerPx;
  if (sx >= 1.0 && sy >= 1.0 && Math.abs(surf.dx - surf.dy) <= 1e-9 * surf.dx) {
    return surf; // already coarse enough and square: never upsample
  }
  if (sx > 1.0 || sy > 1.0) {
    // anisotropic and coarser than target on one axis:
    umPerPx = Math.max(surf.dx, surf.dy) * 1e6; // square the grid at the coarser spacing
    sx = (surf.dx * 1e6) / umPerPx;
    sy = (surf.dy * 1e6) / umPerPx;
  }
  const { rows: h, cols: w, data } = surf.z;
  const outW = Math.max(1, pyRound(w * sx));
  const outH = Math.max(1, pyRound(h * sy));
  const z0 = new Float64Array(data.length);
  const valid = new Float64Array(data.length);
  for (let i = 0; i < data.length; i++) {
    const v = data[i]!;
    if (Number.isFinite(v)) {
      z0[i] = v;
      valid[i] = 1;
    }
  }
  const num = resizeArea({ data: z0, rows: h, cols: w }, outH, outW);
  const den = resizeArea({ data: valid, rows: h, cols: w }, outH, outW);
  const z = new Float64Array(outH * outW);
  for (let i = 0; i < z.length; i++) {
    const d = den.data[i]!;
    z[i] = d >= 0.5 ? num.data[i]! / Math.max(d, 1e-9) : NaN;
  }
  return { z: { data: z, rows: outH, cols: outW }, dx: umPerPx * 1e-6, dy: umPerPx * 1e-6, name: surf.name };
}

/** Replace NaNs by the nearest valid value (for filtering/gradients only). */
export function fillNearest(z: Grid<Float64Array>): Grid<Float64Array> {
  const { rows, cols, data } = z;
  const invalid = new Uint8Array(data.length);
  let any = false;
  for (let i = 0; i < data.length; i++) {
    if (!Number.isFinite(data[i]!)) {
      invalid[i] = 1;
      any = true;
    }
  }
  if (!any) return { data: data.slice(), rows, cols };
  const idx = edtIndices({ data: invalid, rows, cols });
  const out = new Float64Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = data[idx[i]!]!;
  return { data: out, rows, cols };
}

/** `np.gradient(f, h)` for a 2-D array: central differences inside, one-sided first order at the edges. */
function gradient(f: Grid<Float64Array>, h: number): [Float64Array, Float64Array] {
  const { rows, cols, data } = f;
  const gy = new Float64Array(data.length);
  const gx = new Float64Array(data.length);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      if (rows > 1) {
        if (r === 0) gy[i] = (data[i + cols]! - data[i]!) / h;
        else if (r === rows - 1) gy[i] = (data[i]! - data[i - cols]!) / h;
        else gy[i] = (data[i + cols]! - data[i - cols]!) / (2 * h);
      }
      if (cols > 1) {
        if (c === 0) gx[i] = (data[i + 1]! - data[i]!) / h;
        else if (c === cols - 1) gx[i] = (data[i]! - data[i - 1]!) / h;
        else gx[i] = (data[i + 1]! - data[i - 1]!) / (2 * h);
      }
    }
  }
  return [gy, gx];
}

/** Options of `hillshade` (Python keyword arguments). */
export interface HillshadeOptions {
  valid?: Mask | null;
  azimuthDeg?: number;
  altitudeDeg?: number;
  exaggeration?: number;
}

/**
 * Render a height map (metres) as an 8-bit shaded relief image.
 *
 * This mimics the grey "virtual light" renderings produced by comparison
 * microscopes (Evofinder, IBIS, TopMatch) that the brief's figures use.
 */
export function hillshade(z: Grid<Float64Array>, pxM: number, opts: HillshadeOptions = {}): Grid<Uint8Array> {
  const azimuthDeg = opts.azimuthDeg ?? 315.0;
  const altitudeDeg = opts.altitudeDeg ?? 35.0;
  const exaggeration = opts.exaggeration ?? 3.0;
  const { rows, cols } = z;
  const n = rows * cols;
  let valid: Uint8Array;
  if (opts.valid == null) {
    valid = new Uint8Array(n);
    for (let i = 0; i < n; i++) valid[i] = Number.isFinite(z.data[i]!) ? 1 : 0;
  } else {
    valid = opts.valid.data;
  }
  let allValid = true;
  let anyValid = false;
  for (let i = 0; i < n; i++) {
    if (valid[i]) anyValid = true;
    else allValid = false;
  }
  let zf = !allValid ? fillNearest(z) : z;
  zf = gaussianFilter(zf, 0.7);
  const scaled = new Float64Array(n);
  for (let i = 0; i < n; i++) scaled[i] = zf.data[i]! * exaggeration;
  const [gy, gx] = gradient({ data: scaled, rows, cols }, pxM);
  const az = (azimuthDeg * Math.PI) / 180;
  const alt = (altitudeDeg * Math.PI) / 180;
  // light direction: compass azimuth (0 = up, 315 = upper left); image rows grow downward
  const lx = Math.cos(alt) * Math.sin(az);
  const ly = -Math.cos(alt) * Math.cos(az);
  const lz = Math.sin(alt);
  const shade = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const a = gx[i]!;
    const b = gy[i]!;
    shade[i] = (-a * lx - b * ly + lz) / Math.sqrt(1.0 + a ** 2 + b ** 2);
  }
  let v: Float64Array;
  if (anyValid) {
    v = new Float64Array(n);
    let k = 0;
    for (let i = 0; i < n; i++) if (valid[i]) v[k++] = shade[i]!;
    v = v.subarray(0, k);
  } else {
    v = shade;
  }
  const lo = percentile(v, 0.5);
  const hi = percentile(v, 99.5);
  const span = Math.max(hi - lo, 1e-9);
  const img = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (!valid[i]) continue; // img[~valid] = 0
    const t = Math.min(Math.max((shade[i]! - lo) / span, 0), 1);
    img[i] = Math.trunc(t * 235 + 10); // astype(np.uint8) truncates
  }
  return { data: img, rows, cols };
}
