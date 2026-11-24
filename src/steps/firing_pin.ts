/** Stages 3-4 - firing-pin impression (hysteresis on depth) and pin-strike circle (port of `firing_pin.py`). */
import {
  SQUARE3,
  binaryClosing,
  binaryFillHoles,
  binaryOpening,
  binaryPropagation,
  edt,
  ellipseKernel,
  externalContour,
  label,
  labelSums,
  percentile,
  type Grid,
  type Mask,
} from "../ndimage/index.js";
import type { Params } from "../segment.js";
import { disk, robustCircle } from "./geometry.js";
import { and, any, argmax, count, emptyMask, pyIntRound } from "./np.js";

type G = Grid<ArrayLike<number>>;

/**
 * Connected component (4-connected) of `cand` with the largest summed seed weight.
 *
 * @param cand Candidate mask.
 * @param seeds Seed mask (0/1).
 * @param weight Per-pixel weight of a seed.
 * @returns Mask of the winning component; empty when no component contains a positively weighted seed.
 */
export function largestSeededComponent(cand: G, seeds: G, weight: G): Mask {
  const { labels, n } = label(cand);
  if (n === 0) return emptyMask(cand.rows, cand.cols);
  const nPix = cand.rows * cand.cols;
  const ws = new Float64Array(nPix);
  for (let i = 0; i < nPix; i++) ws[i] = weight.data[i] * (seeds.data[i] ? 1 : 0);
  const score = labelSums({ data: ws, rows: cand.rows, cols: cand.cols }, labels, n).subarray(1);
  const best = argmax(score);
  if (!(score[best] > 0)) return emptyMask(cand.rows, cand.cols);
  const out = new Uint8Array(nPix);
  const id = best + 1;
  for (let i = 0; i < nPix; i++) out[i] = labels.data[i] === id ? 1 : 0;
  return { data: out, rows: cand.rows, cols: cand.cols };
}

/**
 * Hysteresis segmentation of the firing-pin impression inside the primer.
 *
 * Seeds are pixels deeper than `max(2 x edge threshold, fpiSeedFrac x crater depth)`; they are
 * grown to the edge threshold `max(fpiMinDepthUm, fpiNoiseK x roughness, fpiRelDepth x crater
 * depth)`. The relative term matters because primers dish around the strike. Unmeasured holes in
 * the middle of the primer (steep craters a confocal scanner cannot see) count as crater.
 *
 * @param d Depth below the primer datum plane [m] (negative = deeper).
 * @param pc Primer centre `[cx, cy]` [px].
 * @param rp Primer radius [px].
 * @param sigmaUm Surface roughness of the primer face [um].
 * @returns `[fpi, thresholdUm, maxDepthUm]` (fpi empty when none was found).
 */
export function segmentFiringPin(
  d: Grid<Float64Array>,
  valid: Mask,
  primerFace: Mask,
  pc: [number, number],
  rp: number,
  p: Params,
  sigmaUm: number,
  warnings: string[],
): [Mask, number, number] {
  const { rows, cols } = d;
  const n = rows * cols;
  const inner = and(primerFace, disk([rows, cols], pc[0], pc[1], 0.85 * rp));
  // Unmeasured steep crater (confocal scans). Opening by reconstruction drops separate thin
  // no-data lines along steep walls but keeps the full outline of any hole with a thick core.
  const holes0 = new Uint8Array(n);
  for (let i = 0; i < n; i++) holes0[i] = inner.data[i] && !valid.data[i] ? 1 : 0;
  const holesM: Mask = { data: holes0, rows, cols };
  const rOpen = Math.max(1, pyIntRound(40.0 / p.workUmPerPx));
  const core = binaryOpening(holesM, 1, ellipseKernel(rOpen));
  const holes = binaryPropagation(core, holesM);
  const ddz = new Float64Array(n); // np.nan_to_num(dd, nan=0.0) with dd = where(valid, d_um, nan)
  let maxDepth = 0.0;
  let anyInner = false;
  let minInner = Infinity;
  for (let i = 0; i < n; i++) {
    const v = valid.data[i] ? d.data[i] * 1e6 : NaN;
    ddz[i] = v === v ? v : 0.0;
    if (inner.data[i] && valid.data[i]) {
      anyInner = true;
      if (v < minInner) minInner = v;
    }
  }
  if (anyInner) maxDepth = -minInner;
  // Edge of the crater: a fixed fraction of its depth, never below the noise floor.
  const thr = Math.max(p.fpiMinDepthUm, p.fpiNoiseK * sigmaUm, p.fpiRelDepth * Math.max(maxDepth, 0));
  const seedThr = Math.min(Math.max(thr * 2, p.fpiSeedFrac * maxDepth), 0.6 * maxDepth);
  const cand = new Uint8Array(n), seeds = new Uint8Array(n), weight = new Float64Array(n);
  const holeW = 0.5 * Math.max(maxDepth, seedThr);
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const i = y * cols + x;
      const h = holes.data[i];
      cand[i] = primerFace.data[i] && (ddz[i] < -thr || h) ? 1 : 0;
      seeds[i] = inner.data[i] && (ddz[i] < -seedThr || h) ? 1 : 0;
      // deeper seeds and seeds close to the primer centre count more
      const dx = x - pc[0], dy = y - pc[1];
      const central = Math.min(Math.max(1.0 - Math.sqrt(dx * dx + dy * dy) / rp, 0), 1);
      // unmeasured holes weigh like a typical crater pixel (half the max depth)
      weight[i] = central * (h ? holeW : Math.max(-ddz[i], 0));
    }
  }
  let fpi = largestSeededComponent({ data: cand, rows, cols }, { data: seeds, rows, cols }, { data: weight, rows, cols });
  if (!any(fpi)) {
    warnings.push("no firing-pin impression found");
    return [fpi, thr, maxDepth];
  }
  fpi = binaryFillHoles(fpi);
  fpi = binaryOpening(fpi, 2, SQUARE3);
  fpi = largestSeededComponent(fpi, fpi, { data: new Float64Array(n).fill(1), rows, cols });
  fpi = binaryFillHoles(binaryClosing(fpi, 2, SQUARE3));
  if (any(holes) && count(and(fpi, holes)) > 0.3 * count(fpi)) {
    warnings.push("firing-pin crater partly unmeasured (no data); outline from data gap");
  }
  return [and(fpi, primerFace), thr, maxDepth];
}

/**
 * Circle of the pin strike: robust fit to the round part of the impression outline.
 *
 * Starts from the deepest point of the distance transform, then refits a robust circle to the
 * outline points within 1.15 x the median radius, so a drag tail does not pull the fit.
 *
 * @param fpi Firing-pin-impression mask (must be non-empty).
 * @returns `[cx, cy, r]` [px].
 */
export function fitCore(fpi: Mask): [number, number, number] {
  const dt = edt(fpi);
  const i0 = argmax(dt.data);
  const iy = Math.floor(i0 / fpi.cols), ix = i0 - iy * fpi.cols;
  let cx = ix, cy = iy, r = dt.data[i0];
  const pts = externalContour(fpi);
  const n = pts.length;
  const px = new Float64Array(n), py = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    px[i] = pts[i][0];
    py[i] = pts[i][1];
  }
  const rad = new Float64Array(n);
  for (let it = 0; it < 3; it++) {
    for (let i = 0; i < n; i++) {
      const dx = px[i] - cx, dy = py[i] - cy;
      rad[i] = Math.sqrt(dx * dx + dy * dy);
    }
    const lim = percentile(rad, 50) * 1.15;
    const xs: number[] = [], ys: number[] = [];
    for (let i = 0; i < n; i++) {
      if (rad[i] <= lim) {
        xs.push(px[i]);
        ys.push(py[i]);
      }
    }
    if (xs.length < 10) break;
    [cx, cy, r] = robustCircle(xs, ys, 6, 2.5, 1.5);
  }
  return [cx, cy, r];
}
