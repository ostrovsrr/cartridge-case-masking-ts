/** Stage 1 - primer face: signed circular Hough transform, plausibility re-ranking, ray refinement. */
import {
  binaryFillHoles,
  gaussianFilter,
  gradient,
  label,
  maximumFilter,
  median,
  percentile,
  type Grid,
  type Mask,
} from "../ndimage/index.js";
import { fillNearest } from "../preprocess.js";
import type { Params } from "../segment.js";
import { disk, robustCircle, sample } from "./geometry.js";
import { argmax, arange, deg2rad, fmtF, fmtPct, linspaceOpen, roundHalfEven, sq } from "./np.js";

/** Hough peaks (beyond the strongest) that are refined and re-ranked. */
export const N_CANDIDATES = 8;
/** A valid region at least this wide [mm] is treated as a visible case head. */
export const MIN_HEAD_DIAMETER_MM = 8.0;

/** Plausibility details of a primer circle (Python dict keys). */
export interface PrimerInfo {
  diameter_mm?: number;
  data_edge?: boolean;
  diameter_in_prior?: boolean;
  head_offset_mm?: number;
  crater_inside?: boolean;
  reranked?: boolean;
}

/** Steepest outward drop along 360 rays around `(hx, hy, r0)`, then a robust circle fit. */
function refine(zs: Grid<Float64Array>, hx: number, hy: number, r0: number, pxMm: number): [number, number, number, number] {
  const nA = 360;
  const band = 0.3 / pxMm;
  const rs = arange(Math.max(r0 - band, 0.5 * r0), r0 + band, 0.5);
  const nR = rs.length;
  const X = new Float64Array(nA * nR), Y = new Float64Array(nA * nR);
  for (let a = 0; a < nA; a++) {
    const t = deg2rad(a);
    const c = Math.cos(t), s = Math.sin(t);
    for (let j = 0; j < nR; j++) {
      X[a * nR + j] = hx + c * rs[j];
      Y[a * nR + j] = hy + s * rs[j];
    }
  }
  const prof = sample(zs, X, Y);
  const ks = new Int32Array(nA);
  const strength = new Float64Array(nA);
  const der = new Float64Array(nR);
  for (let a = 0; a < nA; a++) {
    const o = a * nR;
    // np.gradient along the ray (edge_order 1)
    if (nR >= 2) {
      der[0] = prof[o + 1] - prof[o];
      for (let j = 1; j < nR - 1; j++) der[j] = (prof[o + j + 1] - prof[o + j - 1]) / 2;
      der[nR - 1] = prof[o + nR - 1] - prof[o + nR - 2];
    } else der[0] = 0;
    let k = 0;
    for (let j = 1; j < nR; j++) if (der[j] < der[k]) k = j;
    ks[a] = k;
    strength[a] = -der[k];
  }
  const thr = 0.25 * median(strength);
  const ex: number[] = [], ey: number[] = [];
  for (let a = 0; a < nA; a++) {
    if (strength[a] > thr) {
      ex.push(X[a * nR + ks[a]]);
      ey.push(Y[a * nR + ks[a]]);
    }
  }
  const [fcx, fcy, fr, keep] = robustCircle(ex, ey, 6, 2.5, 2.0);
  let nk = 0;
  for (let i = 0; i < keep.length; i++) nk += keep[i];
  return [fcx, fcy, fr, nk / nA];
}

/**
 * Centre and equivalent diameter of a visible case head, or null for primer-only scans.
 *
 * Full-head scans (e.g. NIST E3) show the whole case head as one disc of valid data; its
 * centroid is the head axis, and the primer pocket is machined concentric with it. Scans trimmed
 * to the primer are smaller than {@link MIN_HEAD_DIAMETER_MM} and return null.
 *
 * @returns `[cx, cy, diameterMm]` or null.
 */
export function caseHead(valid: Mask, pxMm: number, filled?: Mask): [number, number, number] | null {
  const f = filled ?? binaryFillHoles(valid);
  const { rows, cols, data } = f;
  let area = 0, sx = 0, sy = 0;
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      if (data[y * cols + x]) {
        area++;
        sx += x;
        sy += y;
      }
    }
  }
  const diameterMm = 2.0 * Math.sqrt(area / Math.PI) * pxMm;
  if (diameterMm < MIN_HEAD_DIAMETER_MM) return null;
  return [sx / area, sy / area, diameterMm];
}

/** True when measured surface surrounds the circle (it is a real edge, not the data edge). */
function headAround(valid: Mask, cx: number, cy: number, r: number): boolean {
  const { rows: h, cols: w } = valid;
  const t = linspaceOpen(0, 2 * Math.PI, 180);
  const hits: number[] = [];
  for (const f of [1.2, 1.35, 1.5]) {
    let ok = 0;
    for (let i = 0; i < t.length; i++) {
      const x = cx + f * r * Math.cos(t[i]);
      const y = cy + f * r * Math.sin(t[i]);
      if (x >= 0 && x < w && y >= 0 && y < h && valid.data[Math.trunc(y) * w + Math.trunc(x)]) ok++;
    }
    hits.push(ok / t.length);
  }
  return (hits[0] + hits[1] + hits[2]) / 3 > 0.5;
}

/**
 * Is there a crater-like feature (pit, bump or unmeasured hole) in the middle of the circle?
 *
 * Polarity-agnostic, because the polarity check runs after the primer search: the 99.5th
 * percentile of |height - face median| inside 0.6 r must exceed both 20 um and twice the same
 * statistic on the 0.65-0.9 r annulus.
 */
function centralRelief(z: Grid<Float64Array>, valid: Mask, cx: number, cy: number, r: number, filled: Mask): boolean {
  const shape: [number, number] = [z.rows, z.cols];
  const midDisk = disk(shape, cx, cy, 0.6 * r);
  const d90 = disk(shape, cx, cy, 0.9 * r);
  const d65 = disk(shape, cx, cy, 0.65 * r);
  const n = z.rows * z.cols;
  const v = valid.data;
  let nAnn = 0, nMid = 0, nHoles = 0, nMidDisk = 0;
  for (let i = 0; i < n; i++) {
    if (v[i] && d90.data[i] && !d65.data[i]) nAnn++;
    if (midDisk.data[i]) {
      nMidDisk++;
      if (v[i]) nMid++;
      else if (filled.data[i]) nHoles++;
    }
  }
  if (nAnn < 50 || nMid < 50) return false;
  if (nHoles > 0.01 * nMidDisk) return true;
  const ann = new Float64Array(nAnn), mid = new Float64Array(nMid);
  for (let i = 0, a = 0, m = 0; i < n; i++) {
    if (!v[i]) continue;
    if (d90.data[i] && !d65.data[i]) ann[a++] = z.data[i];
    if (midDisk.data[i]) mid[m++] = z.data[i];
  }
  const ref = median(ann);
  for (let i = 0; i < nMid; i++) mid[i] = Math.abs(mid[i] - ref);
  for (let i = 0; i < nAnn; i++) ann[i] = Math.abs(ann[i] - ref);
  const aMid = percentile(mid, 99.5);
  const aAnn = percentile(ann, 99.5);
  return aMid > Math.max(20e-6, 2.0 * aAnn);
}

/**
 * Physical plausibility of a primer circle as a factor in `(0, 1]` plus its reasons.
 *
 * - `diameter`: a circle surrounded by measured case head must match the visible primer-face
 *   diameter prior `p.primerDiameterPriorMm`; a circle at the data edge (scan trimmed to the
 *   primer) is exempt below the range.
 * - `concentric`: on full-head scans the primer must share the head axis.
 * - `crater`: a crater-like feature lies in the middle of the circle.
 *
 * @param filled Optional precomputed `binaryFillHoles(valid)` (a cache; same result).
 */
export function plausibility(
  z: Grid<Float64Array>,
  valid: Mask,
  cx: number,
  cy: number,
  r: number,
  pxMm: number,
  p: Params,
  head: [number, number, number] | null,
  filled?: Mask,
): [number, PrimerInfo] {
  const [lo, hi] = p.primerDiameterPriorMm;
  const dia = 2.0 * r * pxMm;
  const info: PrimerInfo = { diameter_mm: dia, data_edge: !headAround(valid, cx, cy, r) };
  let f = 1.0;
  const out = Math.max(lo - dia, dia - hi, 0.0);
  info.diameter_in_prior = out === 0.0 || (info.data_edge! && dia <= hi);
  if (!info.diameter_in_prior) f *= Math.exp(-0.5 * sq(out / 0.2));
  if (head !== null) {
    const dx = cx - head[0], dy = cy - head[1];
    const offMm = Math.sqrt(dx * dx + dy * dy) * pxMm;
    info.head_offset_mm = offMm;
    const excess = Math.max(offMm - 0.75, 0.0);
    f *= Math.exp(-0.5 * sq(excess / 0.5));
  }
  info.crater_inside = centralRelief(z, valid, cx, cy, r, filled ?? binaryFillHoles(valid));
  if (!info.crater_inside) f *= 0.6;
  return [f, info];
}

type Cand = [score: number, x: number, y: number, r: number];

/**
 * Locate the primer face (the visible primer-cup disc).
 *
 * The primer-cup edge drops outward into the primer-pocket gap, so the height gradient on that
 * edge points towards the centre. A *signed* circular Hough transform over the physical radius
 * range `p.primerRadiusMm` gives candidate circles: the strongest peak, the next
 * {@link N_CANDIDATES} separated peaks and, on full-head scans, the best peak concentric with the
 * case head. Each candidate is refined by casting 360 rays from its centre, taking the steepest
 * outward drop on each and fitting a robust circle. Candidates are then re-ranked by Hough
 * strength x {@link plausibility} (diameter prior, concentricity with the case head, crater
 * inside), so an embossed headstamp letter cannot outvote a weaker but physically plausible
 * primer edge.
 *
 * @param z Height map [m] on the working grid; NaN where invalid.
 * @param valid Mask of measured pixels.
 * @param pxMm Pixel size [mm].
 * @param warnings Human-readable warnings are appended here.
 * @param info Optional object filled with the chosen candidate's plausibility details.
 * @returns `[cx, cy, r, inlierFraction, houghScore]`.
 */
export function findPrimer(
  z: Grid<Float64Array>,
  valid: Mask,
  pxMm: number,
  p: Params,
  warnings: string[],
  info?: PrimerInfo,
): [number, number, number, number, number] {
  const { rows: h, cols: w } = z;
  const n = h * w;
  const rmin = p.primerRadiusMm[0] / pxMm, rmax = p.primerRadiusMm[1] / pxMm;
  const v = valid.data;

  // Missing data connected to the scan border is treated as "low": scans trimmed to the primer
  // then have their primer boundary at the data edge. Interior dropouts are filled from their
  // neighbours instead, otherwise every hole would become a steep fake pit.
  let nv = 0;
  for (let i = 0; i < n; i++) if (v[i]) nv++;
  const zv = new Float64Array(nv);
  for (let i = 0, k = 0; i < n; i++) if (v[i]) zv[k++] = z.data[i];
  const [lo, hi] = percentile(zv, [1, 99]);
  const inv = new Uint8Array(n);
  for (let i = 0; i < n; i++) inv[i] = v[i] ? 0 : 1;
  const { labels: lab } = label({ data: inv, rows: h, cols: w });
  const border = new Set<number>();
  for (let c = 0; c < w; c++) {
    border.add(lab.data[c]);
    border.add(lab.data[(h - 1) * w + c]);
  }
  for (let r = 0; r < h; r++) {
    border.add(lab.data[r * w]);
    border.add(lab.data[r * w + w - 1]);
  }
  border.delete(0);
  const filledZ = fillNearest(whereValidF(z, valid));
  const low = lo - 0.25 * (hi - lo);
  const zl = new Float64Array(n);
  for (let i = 0; i < n; i++) zl[i] = border.has(lab.data[i]) ? low : filledZ.data[i];
  const zs = gaussianFilter({ data: zl, rows: h, cols: w }, 2.0);
  const { gy, gx } = gradient(zs);
  const mag = new Float64Array(n);
  for (let i = 0; i < n; i++) mag[i] = Math.sqrt(gx.data[i] * gx.data[i] + gy.data[i] * gy.data[i]);
  const magThr = percentile(mag, 90);
  let ne = 0;
  for (let i = 0; i < n; i++) if (mag[i] > magThr) ne++;
  const ex = new Float64Array(ne), ey = new Float64Array(ne), ux = new Float64Array(ne), uy = new Float64Array(ne);
  for (let i = 0, k = 0; i < n; i++) {
    if (mag[i] > magThr) {
      const r = (i / w) | 0;
      ey[k] = r;
      ex[k] = i - r * w;
      ux[k] = gx.data[i] / mag[i];
      uy[k] = gy.data[i] / mag[i];
      k++;
    }
  }

  const filled = binaryFillHoles(valid);
  const head = caseHead(valid, pxMm, filled);
  const [dlo, dhi] = p.primerDiameterPriorMm;

  // Signed circular Hough transform at 1/2 resolution.
  const s = 2;
  const ah = Math.floor(h / s) + 1, aw = Math.floor(w / s) + 1;
  const na = ah * aw;
  const radii = arange(rmin, rmax, 2.0);
  let best: Cand = [-1.0, 0, 0, 0.0];
  const diag = 0.5 * Math.sqrt(h * h + w * w);
  const centrePrior = new Float64Array(na);
  for (let yy = 0; yy < ah; yy++) {
    for (let xx = 0; xx < aw; xx++) {
      const dx = xx * s - w / 2, dy = yy * s - h / 2;
      centrePrior[yy * aw + xx] = 1.0 - (0.3 * Math.sqrt(dx * dx + dy * dy)) / diag;
    }
  }
  const peak = new Float64Array(na);
  const peakR = new Float64Array(na);
  let nearHead: Uint8Array | null = null;
  if (head !== null) {
    nearHead = new Uint8Array(na);
    for (let yy = 0; yy < ah; yy++) {
      for (let xx = 0; xx < aw; xx++) {
        const dx = xx * s - head[0], dy = yy * s - head[1];
        nearHead[yy * aw + xx] = Math.sqrt(dx * dx + dy * dy) * pxMm < 1.0 ? 1 : 0;
      }
    }
  }
  let bestHead: Cand = [-1.0, 0, 0, 0.0];
  const accRaw = new Float64Array(na);
  const accGrid = { data: accRaw, rows: ah, cols: aw };
  for (const r of radii) {
    accRaw.fill(0);
    for (let k = 0; k < ne; k++) {
      const cx = roundHalfEven((ex[k] + r * ux[k]) / s);
      const cy = roundHalfEven((ey[k] + r * uy[k]) / s);
      if (cx >= 0 && cx < aw && cy >= 0 && cy < ah) accRaw[cy * aw + cx] += 1;
    }
    const acc = gaussianFilter(accGrid, 1.0).data;
    const norm = 2 * Math.PI * r;
    for (let i = 0; i < na; i++) acc[i] = (acc[i] / norm) * centrePrior[i];
    const i = argmax(acc);
    if (acc[i] > best[0]) best = [acc[i], (i % aw) * s, Math.floor(i / aw) * s, r];
    for (let j = 0; j < na; j++) {
      if (acc[j] > peak[j]) {
        peak[j] = acc[j];
        peakR[j] = r;
      }
    }
    if (nearHead !== null && dlo <= 2 * r * pxMm && 2 * r * pxMm <= dhi) {
      let j = 0, bv = -Infinity;
      for (let q = 0; q < na; q++) {
        const val = nearHead[q] ? acc[q] : -1.0;
        if (val > bv) {
          bv = val;
          j = q;
        }
      }
      if (acc[j] > bestHead[0]) bestHead = [acc[j], (j % aw) * s, Math.floor(j / aw) * s, r];
    }
  }

  // Candidate list: global best first (the historical choice), then separated peaks.
  const cands: Cand[] = [best];
  const nms = maximumFilter({ data: peak, rows: ah, cols: aw }, Math.max(3, Math.trunc(0.5 / pxMm / s))).data;
  const pk: number[] = [];
  for (let i = 0; i < na; i++) if (peak[i] === nms[i] && peak[i] > 0.3 * best[0]) pk.push(i);
  // np.argsort(-peak) (ties: stable order; NumPy's quicksort order on exact ties may differ)
  const order = pk.map((_, k) => k).sort((a, b) => peak[pk[b]] - peak[pk[a]] || a - b).slice(0, N_CANDIDATES);
  for (const k of order) {
    const i = pk[k];
    cands.push([peak[i], (i % aw) * s, Math.floor(i / aw) * s, peakR[i]]);
  }
  if (bestHead[0] > 0) cands.push(bestHead);

  type Chosen = { rank: number; k: number; circle: [number, number, number]; inlier: number; reliable: boolean; score: number; why: PrimerInfo };
  let chosen: Chosen | null = null;
  for (let k = 0; k < cands.length; k++) {
    const [score, hx, hy, r0] = cands[k];
    let dup = false;
    if (k > 0) {
      for (let q = 0; q < k; q++) {
        const c = cands[q];
        if (Math.sqrt(sq(hx - c[1]) + sq(hy - c[2])) < 4 && Math.abs(r0 - c[3]) < 4) {
          dup = true;
          break;
        }
      }
    }
    if (dup) continue;
    const [fcx, fcy, fr, inlier] = refine(zs, hx, hy, r0, pxMm);
    const reliable = Number.isFinite(fr) && 0.8 * rmin <= fr && fr <= 1.2 * rmax && inlier >= 0.3;
    const circle: [number, number, number] = reliable ? [fcx, fcy, fr] : [hx, hy, r0];
    const [fac, why] = plausibility(z, valid, circle[0], circle[1], circle[2], pxMm, p, head, filled);
    const rank = best[0] > 0 ? (score / best[0]) * fac : fac;
    if (chosen === null || rank > chosen.rank) chosen = { rank, k, circle, inlier, reliable, score, why };
    if (k === 0 && fac === 1.0) break; // the strongest edge circle is already physically plausible
  }
  if (chosen === null) throw new Error("findPrimer: no candidate");
  const { k, circle, inlier, reliable, score, why } = chosen;
  const [fcx, fcy, fr] = circle;
  if (k > 0) {
    const top = cands[0];
    warnings.push(
      `primer re-ranked: the strongest edge circle (diameter ${fmtF(2 * top[3] * pxMm, 2)} mm) ` +
        `was physically implausible; chose diameter ${fmtF(2 * fr * pxMm, 2)} mm`,
    );
  }
  if (!reliable) warnings.push("primer refinement unreliable; using Hough estimate");
  if (inlier < 0.6) warnings.push(`primer edge only found on ${fmtPct(inlier)} of rays`);
  if (info !== undefined) {
    Object.assign(info, why);
    info.reranked = k > 0;
  }
  return [fcx, fcy, fr, inlier, score];
}

function whereValidF(z: Grid<Float64Array>, valid: Mask): Grid<Float64Array> {
  const out = new Float64Array(z.data.length);
  for (let i = 0; i < out.length; i++) out[i] = valid.data[i] ? z.data[i] : NaN;
  return { data: out, rows: z.rows, cols: z.cols };
}

/** Fraction of the primer circle (sampled every 1 deg) that lies inside the image. */
export function primerInsideFraction(shape: readonly [number, number], cx: number, cy: number, r: number): number {
  const t = linspaceOpen(0, 2 * Math.PI, 360);
  let k = 0;
  for (let i = 0; i < 360; i++) {
    const x = cx + r * Math.cos(t[i]), y = cy + r * Math.sin(t[i]);
    if (x >= 0 && x < shape[1] && y >= 0 && y < shape[0]) k++;
  }
  return k / 360;
}
