/** Stage 5 - firing-pin aperture: flow-back ring (round) or slot (rectangular). Port of `aperture.py`. */
import {
  binaryClosing,
  binaryDilation,
  binaryFillHoles,
  gaussianFilter,
  gradient,
  label,
  median,
  medianFilter1d,
  nanmedian,
  ones,
  percentile,
  type Grid,
  type Mask,
} from "../ndimage/index.js";
import { fillNearest } from "../preprocess.js";
import type { Params } from "../segment.js";
import { disk, robustCircle, sample } from "./geometry.js";
import {
  and,
  andNot,
  arange,
  argmax,
  count,
  deg2rad,
  linspaceOpen,
  pyIntRound,
  pyMod,
  rad2deg,
  radiusGrid,
  roundHalfEven,
  whereValid,
} from "./np.js";

/** Evidence dictionaries keep the Python (snake_case) keys; they are written to JSON as-is. */
export type Evidence = Record<string, unknown>;

/**
 * Per-radius median profile around `(cx, cy)` over `sel`: unique rounded radii (ascending),
 * their medians of `vals`, and per-pixel radius bin and value (in raster order of `sel`).
 */
function radialProfile(d: Grid<Float64Array>, sel: Uint8Array, rr: Float64Array): { radii: number[]; m: Float64Array; rb: Int32Array; vals: Float64Array; idx: Int32Array } {
  let ns = 0;
  for (let i = 0; i < sel.length; i++) if (sel[i]) ns++;
  const rb = new Int32Array(ns), vals = new Float64Array(ns), idx = new Int32Array(ns);
  let rmax = 0;
  for (let i = 0, k = 0; i < sel.length; i++) {
    if (!sel[i]) continue;
    rb[k] = roundHalfEven(rr[i]);
    vals[k] = d.data[i] * 1e6;
    idx[k] = i;
    if (rb[k] > rmax) rmax = rb[k];
    k++;
  }
  const cnt = new Int32Array(rmax + 1);
  for (let k = 0; k < ns; k++) cnt[rb[k]]++;
  const start = new Int32Array(rmax + 2);
  for (let r = 0; r <= rmax; r++) start[r + 1] = start[r] + cnt[r];
  const fill = start.slice(0, rmax + 1);
  const grouped = new Float64Array(ns);
  for (let k = 0; k < ns; k++) grouped[fill[rb[k]]++] = vals[k];
  const radii: number[] = [];
  const med: number[] = [];
  for (let r = 0; r <= rmax; r++) {
    if (!cnt[r]) continue;
    radii.push(r);
    med.push(median(grouped.subarray(start[r], start[r + 1])));
  }
  return { radii, m: medianFilter1d(Float64Array.from(med), 5), rb, vals, idx };
}

/**
 * Firing-pin aperture radius from the primer flow-back ring.
 *
 * When the primer is struck, metal flows back into the firing-pin hole of the breech face and
 * forms a raised ring around the crater. On the per-radius median profile m(r) around the pin
 * centre this is a peak just outside the crater; the aperture edge is where m(r) has fallen back
 * to `p.apertureFall` of that peak. A primer that simply rises towards its edge has no such fall.
 *
 * @param d Depth below the primer datum plane [m].
 * @param rc Pin-strike radius [px]; `cx, cy` its centre.
 * @param rp Primer radius [px].
 * @param face Primer-face mask; the profile is restricted to it.
 * @param roughUm Primer-face roughness [um]; the ring must exceed 4x this.
 * @returns Aperture radius [px], or null when no flow-back ring is found.
 */
export function findAperture(
  d: Grid<Float64Array>,
  valid: Mask,
  cx: number,
  cy: number,
  rc: number,
  rp: number,
  face: Mask,
  roughUm: number,
  p: Params,
): number | null {
  const n = d.data.length;
  const rr = radiusGrid(d.rows, d.cols, cx, cy).data;
  const rHi = Math.min(0.85 * rp, p.apertureMaxRatio * rc);
  const sel = new Uint8Array(n);
  let ns = 0;
  for (let i = 0; i < n; i++) {
    if (valid.data[i] && face.data[i] && rr[i] >= rc && rr[i] <= rHi) {
      sel[i] = 1;
      ns++;
    }
  }
  if (ns < 100) return null;
  const { radii, m } = radialProfile(d, sel, rr);
  if (!radii.some((r) => r <= 1.8 * rc)) return null;
  const iPk = argmax(radii.map((r, i) => (r <= 1.8 * rc ? m[i] : -Infinity)));
  const peak = m[iPk];
  if (peak < Math.max(p.fpiMinDepthUm, 4 * roughUm)) return null;
  for (let i = 0; i < radii.length; i++) if (radii[i] > radii[iPk] && m[i] <= p.apertureFall * peak) return radii[i];
  return null;
}

/** Flow-back ring whose fall-off lies beyond `apertureMaxRatio`, confirmed sector by sector. */
function wideRing(
  d: Grid<Float64Array>,
  valid: Mask,
  cx: number,
  cy: number,
  rc: number,
  rp: number,
  face: Mask,
  roughUm: number,
  p: Params,
  ev: Evidence,
): number | null {
  const { rows, cols } = d;
  const n = rows * cols;
  const rr = radiusGrid(rows, cols, cx, cy).data;
  const rHi = Math.min(p.apertureRingMaxPrimer * rp, p.apertureEdgeMaxRatio * rc);
  const sel = new Uint8Array(n);
  let ns = 0;
  for (let i = 0; i < n; i++) {
    if (valid.data[i] && face.data[i] && rr[i] >= rc && rr[i] <= rHi) {
      sel[i] = 1;
      ns++;
    }
  }
  if (ns < 100) return null;
  const { radii, m, rb, vals, idx } = radialProfile(d, sel, rr);
  if (!radii.some((r) => r <= p.apertureMaxRatio * rc)) return null;
  const iPk = argmax(radii.map((r, i) => (r <= p.apertureMaxRatio * rc ? m[i] : -Infinity)));
  const peak = m[iPk];
  ev.ring_peak_um = peak;
  if (peak < Math.max(p.fpiMinDepthUm, 4 * roughUm)) return null;
  let rEdge = NaN;
  for (let i = 0; i < radii.length; i++) {
    if (radii[i] > radii[iPk] && m[i] <= p.apertureFall * peak) {
      rEdge = radii[i];
      break;
    }
  }
  if (rEdge !== rEdge) return null;
  // the fall must happen all around the crater, not in one direction only
  const sec = new Int32Array(rb.length);
  for (let k = 0; k < rb.length; k++) {
    const i = idx[k];
    const y = Math.floor(i / cols), x = i - y * cols;
    const ang = Math.atan2(y - cy, x - cx);
    sec[k] = pyMod(Math.floor(((ang + Math.PI) / (2 * Math.PI)) * 12), 12);
  }
  let agree = 0;
  const rpk = radii[iPk];
  for (let s = 0; s < 12; s++) {
    const inner: number[] = [], outer: number[] = [];
    for (let k = 0; k < rb.length; k++) {
      if (sec[k] !== s) continue;
      if (rb[k] >= rpk - 0.1 * rc && rb[k] <= rpk + 0.1 * rc) inner.push(vals[k]);
      if (rb[k] >= rEdge && rb[k] <= rEdge + 0.15 * rc) outer.push(vals[k]);
    }
    if (inner.length < 10 || outer.length < 10) continue;
    const hi = median(inner), lo = median(outer);
    if (hi - lo >= 0.5 * (1 - p.apertureFall) * peak) agree++;
  }
  const cover = agree / 12.0;
  ev.ring_radius_ratio = rEdge / rc;
  ev.ring_sector_cover = cover;
  if (cover < p.apertureRingMinCover) return null;
  return rEdge;
}

/**
 * Firing-pin aperture radius from its edge, when {@link findAperture} found no ring.
 *
 * Two cues, each accepted only with evidence all around the crater:
 *
 * 1. **Wide ring** - the flow-back profile whose fall-off lies beyond `p.apertureMaxRatio` (up to
 *    `p.apertureEdgeMaxRatio`) x the pin radius, within `p.apertureRingMaxPrimer` x the primer
 *    radius and seen in at least `p.apertureRingMinCover` of twelve 30 deg sectors.
 * 2. **Edge step** - inside the aperture the primer flowed back and stands higher than the
 *    breech-face contact outside it, so the aperture edge is a step *down* going outward. Along
 *    `p.apertureEdgeRays` rays from the pin centre the strongest outward down-step between
 *    `p.apertureEdgeMinRatio` x and `p.apertureMaxRatio` x the pin radius is located; a robust
 *    circle is fitted to the steps that exceed `p.apertureEdgeK` x the gradient noise of the
 *    primer face. Accepted only when supported by at least `p.apertureEdgeMinCover` of the rays,
 *    in the search range and centred within 25 % of the pin radius of the strike centre.
 *
 * @param evidence Optional object filled with `method`, the measurements of both cues and
 *   `reason` when rejected (Python keys).
 * @returns Radius [px] of the smallest disk around the strike centre that contains the edge
 *   circle, or null without clear evidence.
 */
export function findApertureEdge(
  d: Grid<Float64Array>,
  valid: Mask,
  cx: number,
  cy: number,
  rc: number,
  rp: number,
  face: Mask,
  pxMm: number,
  roughUm: number,
  p: Params,
  evidence?: Evidence,
): number | null {
  const ev: Evidence = evidence ?? {};
  const wide = wideRing(d, valid, cx, cy, rc, rp, face, roughUm, p, ev);
  if (wide !== null) {
    ev.method = "wide ring";
    ev.accepted = true;
    return wide;
  }
  const rLo = p.apertureEdgeMinRatio * rc;
  const rHi = Math.min(0.85 * rp, p.apertureMaxRatio * rc);
  if (rHi - rLo < 4) {
    ev.reason = "no room between crater and primer edge";
    return null;
  }
  const { rows, cols } = d;
  const n = rows * cols;
  const um = pxMm * 1000.0;
  const f = fillNearest(whereValid(d, valid));
  for (let i = 0; i < n; i++) f.data[i] *= 1e6;
  const df = gaussianFilter(f, Math.max(1.0, p.rectSmoothUm / um));
  const { gy, gx } = gradient(df);
  const gr = new Float64Array(n);
  const rrA = new Float64Array(n);
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const i = y * cols + x;
      const dx = x - cx, dy = y - cy;
      const rr = Math.max(Math.sqrt(dx * dx + dy * dy), 1e-6);
      rrA[i] = rr;
      gr[i] = (gx.data[i] * dx + gy.data[i] * dy) / rr; // um per px, outward
    }
  }
  const pick = (lim: number): number[] => {
    const out: number[] = [];
    for (let i = 0; i < n; i++) if (face.data[i] && valid.data[i] && rrA[i] > lim) out.push(gr[i]);
    return out;
  };
  let ref = pick(rHi);
  if (ref.length < 200) ref = pick(rLo);
  const mref = median(ref);
  const noise = 1.4826 * median(ref.map((v) => Math.abs(v - mref))) + 1e-9;
  const nr = p.apertureEdgeRays;
  const angs = linspaceOpen(0, 2 * Math.PI, nr);
  const radii = arange(rLo, rHi, 1.0);
  const nR = radii.length;
  const X = new Float64Array(nr * nR), Y = new Float64Array(nr * nR);
  for (let a = 0; a < nr; a++) {
    const c = Math.cos(angs[a]), s = Math.sin(angs[a]);
    for (let j = 0; j < nR; j++) {
      X[a * nR + j] = cx + c * radii[j];
      Y[a * nR + j] = cy + s * radii[j];
    }
  }
  const grG = { data: gr, rows, cols };
  const g = sample(grG, X, Y);
  const vf = new Float64Array(n);
  for (let i = 0; i < n; i++) vf[i] = valid.data[i] && face.data[i] ? 1 : 0;
  const okS = sample({ data: vf, rows, cols }, X, Y);
  for (let k = 0; k < g.length; k++) if (!(okS[k] > 0.5)) g[k] = Infinity;
  const gmin = new Float64Array(nr);
  const jArr = new Int32Array(nr);
  const strong = new Uint8Array(nr);
  let nStrong = 0;
  for (let a = 0; a < nr; a++) {
    let j = 0;
    for (let q = 1; q < nR; q++) if (g[a * nR + q] < g[a * nR + j]) j = q;
    jArr[a] = j;
    gmin[a] = g[a * nR + j];
    strong[a] = Number.isFinite(gmin[a]) && -gmin[a] > p.apertureEdgeK * noise ? 1 : 0;
    nStrong += strong[a];
  }
  if (nStrong < 8) {
    ev.coverage = nStrong / nr;
    ev.reason = "too few strong outward steps";
    return null;
  }
  const px: number[] = [], py: number[] = [], gs: number[] = [];
  for (let a = 0; a < nr; a++) {
    if (!strong[a]) continue;
    px.push(X[a * nR + jArr[a]]);
    py.push(Y[a * nR + jArr[a]]);
    gs.push(-gmin[a]);
  }
  const [ecx, ecy, er, keep] = robustCircle(px, py, 6, 2.5, Math.max(1.5, 0.02 / pxMm));
  const off = Math.sqrt((ecx - cx) * (ecx - cx) + (ecy - cy) * (ecy - cy));
  let nk = 0;
  const kept: number[] = [];
  for (let i = 0; i < keep.length; i++) {
    if (keep[i]) {
      nk++;
      kept.push(gs[i]);
    }
  }
  const cover = nk / nr;
  ev.coverage = cover;
  ev.radius_ratio = er / rc;
  ev.centre_offset_ratio = off / rc;
  ev.step_snr = nk > 0 ? median(kept) / noise : 0.0;
  const fails: string[] = [];
  if (cover < p.apertureEdgeMinCover) fails.push("edge not closed around the crater");
  if (!(rLo <= er && er <= rHi)) fails.push("edge radius out of range");
  if (off > 0.25 * rc) fails.push("edge not centred on the pin");
  if (fails.length) {
    ev.reason = fails.join(", ");
    return null;
  }
  ev.method = "edge step";
  ev.accepted = true;
  return er + off;
}

/**
 * Region excluded from the breech face around the firing pin.
 *
 * The aperture disk (or `region`, e.g. a rectangular slot) united with the impression grown by a
 * 40 um safety band, holes filled, clipped to the primer face.
 *
 * @param ra Aperture radius [px] around the pin-strike centre `(cx, cy)`.
 * @param region Optional aperture region used instead of the disk.
 */
export function apertureZone(
  fpi: Mask,
  cx: number,
  cy: number,
  ra: number,
  face: Mask,
  pxMm: number,
  region: Mask | null = null,
): Mask {
  const margin = Math.max(2, pyIntRound(0.04 / pxMm)); // 40 um safety band around the FPI
  const ap = region ?? disk([fpi.rows, fpi.cols], cx, cy, ra);
  const dil = binaryDilation(fpi, margin);
  const zone = new Uint8Array(fpi.data.length);
  for (let i = 0; i < zone.length; i++) zone[i] = ap.data[i] || dil.data[i] ? 1 : 0;
  return and(binaryFillHoles({ data: zone, rows: fpi.rows, cols: fpi.cols }), face);
}

/** Rectangular firing-pin aperture (returned by {@link findRectAperture}). */
export interface RectAperture {
  /** Aperture-zone mask (the slot between the step edges, clipped to the primer face). */
  mask: Mask;
  /** Mask of strips (+-`rectEdgeGuardUm`) along the two step edges. */
  edges: Mask;
  /** Direction of the slot's long axis [deg], image coordinates, 0..180. */
  angleDeg: number;
  /** Distance between the two step edges [mm]. */
  widthMm: number;
  /** Extent of the zone along the slot inside the primer face [mm]. */
  lengthMm: number;
  /** Slot centre `[x, y]` [px] (mid-line, middle of the edge support). */
  centerPx: [number, number];
}

/**
 * Rectangular (slotted) firing-pin aperture from two long, straight, parallel step edges.
 *
 * Pistols with a rectangular firing-pin hole (e.g. Glock) leave a raised, flat flow-back plateau
 * bounded by two straight parallel steps on either side of an elongated impression. For every
 * direction `theta` the height gradient component along `theta` is averaged over lines
 * perpendicular to it (a signed Radon projection); a raised slot gives a positive peak on one
 * side of the pin and a negative one on the other. The strongest pair over all directions is a
 * slot only when both peaks stand out `p.rectMinScore` x above the typical projection magnitude,
 * the pair is direction-selective (`p.rectMinAnisotropy` x the median over all directions) and the
 * edges are `p.rectWidthMm` apart. Along the slot, the zone spans the flow-back plateau (bins
 * where the strip between the edges stands above the face just outside them by at least
 * `p.rectPlateauFrac` of the typical step, gaps up to `p.rectEdgeGapUm` bridged), joined to the
 * crater.
 *
 * @param evidence Optional object filled with `score`, `anisotropy`, `width_mm`, `angle_deg`,
 *   `plateau_step_um` and `reason` when rejected.
 * @returns The slot, or null when the evidence does not support one.
 */
export function findRectAperture(
  d: Grid<Float64Array>,
  valid: Mask,
  face: Mask,
  fpi: Mask,
  cx: number,
  cy: number,
  rc: number,
  rp: number,
  pxMm: number,
  p: Params,
  evidence?: Evidence,
): RectAperture | null {
  const ev: Evidence = evidence ?? {};
  const { rows, cols } = d;
  const n = rows * cols;
  let nf = 0, sx = 0, sy = 0;
  for (let i = 0; i < n; i++) {
    if (face.data[i]) {
      const y = Math.floor(i / cols);
      nf++;
      sy += y;
      sx += i - y * cols;
    }
  }
  if (nf === 0) {
    ev.reason = "empty primer face";
    return null;
  }
  const pcx = sx / nf, pcy = sy / nf;
  const um = pxMm * 1000.0;
  const f = fillNearest(whereValid(d, valid));
  for (let i = 0; i < n; i++) f.data[i] *= 1e6;
  const df = gaussianFilter(f, Math.max(1.0, p.rectSmoothUm / um));
  const { gy, gx } = gradient(df);
  const margin = Math.max(2, pyIntRound(0.04 / pxMm));
  // exclude the crater walls; impression pixels far from the strike circle are often the lower
  // primer face beside a raised slot and must stay visible
  const crater = and(fpi, disk([rows, cols], cx, cy, 1.3 * rc));
  const craterDil = binaryDilation(crater, margin);
  const region = andNot(disk([rows, cols], pcx, pcy, p.rectSearchFrac * rp), craterDil);
  const nRegion = count(region);
  if (nRegion < 500) {
    ev.reason = "too little primer face outside the impression";
    return null;
  }
  const rx = new Float64Array(nRegion), ry = new Float64Array(nRegion);
  const gxr = new Float64Array(nRegion), gyr = new Float64Array(nRegion);
  for (let i = 0, k = 0; i < n; i++) {
    if (!region.data[i]) continue;
    const y = Math.floor(i / cols);
    ry[k] = y - cy;
    rx[k] = i - y * cols - cx;
    gxr[k] = gx.data[i];
    gyr[k] = gy.data[i];
    k++;
  }
  const minCount = Math.max(10, pyIntRound(0.15 / pxMm)); // >= 150 um of line inside the region
  let best: [number, number, number, number, number] | null = null;
  const scores: number[] = [];
  const sc = new Int32Array(nRegion);
  for (let deg = 0; deg < 180; deg += 2) {
    const th = deg2rad(deg);
    const c = Math.cos(th), s = Math.sin(th);
    let lo = Infinity, hi = -Infinity;
    for (let k = 0; k < nRegion; k++) {
      const v = roundHalfEven(rx[k] * c + ry[k] * s);
      sc[k] = v;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const L = hi - lo + 1;
    const cnt = new Int32Array(L);
    const prof = new Float64Array(L);
    for (let k = 0; k < nRegion; k++) {
      const b = sc[k] - lo;
      cnt[b]++;
      prof[b] += gxr[k] * c + gyr[k] * s;
    }
    let nOk = 0;
    for (let b = 0; b < L; b++) {
      prof[b] = prof[b] / Math.max(cnt[b], 1);
      if (cnt[b] >= minCount) nOk++;
    }
    if (nOk < 10) {
      scores.push(0.0);
      continue;
    }
    const absOk = new Float64Array(nOk);
    for (let b = 0, q = 0; b < L; b++) if (cnt[b] >= minCount) absOk[q++] = Math.abs(prof[b]);
    const noise = median(absOk) + 1e-12;
    // rising edge before the pin centre, falling edge after it
    let i1 = 0, a1 = -Infinity, i2 = 0, b2 = -Infinity;
    for (let b = 0; b < L; b++) {
      const pos = b + lo;
      const ok = cnt[b] >= minCount;
      const av = ok && pos < 0 ? prof[b] : -Infinity;
      const bv = ok && pos > 0 ? -prof[b] : -Infinity;
      if (av > a1) {
        a1 = av;
        i1 = b;
      }
      if (bv > b2) {
        b2 = bv;
        i2 = b;
      }
    }
    const mn = Math.min(a1, b2);
    const score = mn / noise;
    scores.push(Math.max(score, 0.0));
    if (best === null || score > best[0]) best = [score, th, i1 + lo, i2 + lo, mn];
  }
  if (best === null) {
    ev.reason = "no projection";
    return null;
  }
  const [score, th, s1, s2] = best;
  const aniso = score / Math.max(median(scores), 1.0); // scores are in noise units
  const widthMm = (s2 - s1) * pxMm;
  ev.score = score;
  ev.anisotropy = aniso;
  ev.width_mm = widthMm;
  ev.angle_deg = pyMod(rad2deg(th) + 90.0, 180.0);
  const fails: string[] = [];
  if (score < p.rectMinScore) fails.push("edges weak");
  if (aniso < p.rectMinAnisotropy) fails.push("not direction-selective (ring-like)");
  if (!(p.rectWidthMm[0] <= widthMm && widthMm <= p.rectWidthMm[1])) fails.push("implausible width");
  if (fails.length) {
    ev.reason = fails.join(", ");
    return null;
  }
  // Extent along the slot: where the strip between the edges stands above the primer face just
  // outside them (the flow-back plateau), plus the crater.
  const c = Math.cos(th), s = Math.sin(th);
  const sco = new Float64Array(n), tco = new Float64Array(n);
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const i = y * cols + x;
      const dx = x - cx, dy = y - cy;
      sco[i] = dx * c + dy * s;
      tco[i] = -dx * s + dy * c;
    }
  }
  const w = s2 - s1;
  const sm = 0.5 * (s1 + s2);
  const bw = Math.max(2, pyIntRound(0.04 / pxMm)); // 40 um bins along the slot
  const tb = new Int32Array(n);
  let tMin = Infinity, tMax = -Infinity;
  for (let i = 0; i < n; i++) {
    tb[i] = Math.floor(tco[i] / bw);
    if (face.data[i]) {
      if (tb[i] < tMin) tMin = tb[i];
      if (tb[i] > tMax) tMax = tb[i];
    }
  }
  const nb = tMax - tMin + 1;
  const o1a = s1 - margin, o1b = s1 - margin - 0.15 / pxMm;
  const o2a = s2 + margin, o2b = s2 + margin + 0.15 / pxMm;
  const inVals: number[][] = Array.from({ length: nb }, () => []);
  const inCount = new Int32Array(nb);
  const outVals: number[][] = Array.from({ length: nb }, () => []);
  for (let i = 0; i < n; i++) {
    if (!face.data[i] || craterDil.data[i]) continue; // clear
    const k = tb[i] - tMin;
    if (Math.abs(sco[i] - sm) <= 0.35 * w) {
      inCount[k]++;
      if (valid.data[i]) inVals[k].push(d.data[i] * 1e6);
    }
    if (valid.data[i] && ((sco[i] < o1a && sco[i] > o1b) || (sco[i] > o2a && sco[i] < o2b))) {
      outVals[k].push(d.data[i] * 1e6);
    }
  }
  const medIn = new Float64Array(nb).fill(NaN);
  let medOut = new Float64Array(nb).fill(NaN);
  const unmeasured = new Uint8Array(nb);
  for (let k = 0; k < nb; k++) {
    const a = inVals[k], b = outVals[k];
    if (inCount[k] >= 5 && a.length < 0.5 * inCount[k]) unmeasured[k] = 1; // dropout inside the slot
    else if (a.length >= 5) medIn[k] = median(a);
    if (b.length >= 5) medOut[k] = median(b);
  }
  if (medOut.some(Number.isFinite)) {
    // the outer strips leave the face first
    const fillV = nanmedian(medOut);
    medOut = medOut.map((v) => (Number.isFinite(v) ? v : fillV));
  }
  const step = new Float64Array(nb);
  const have = new Uint8Array(nb);
  for (let k = 0; k < nb; k++) {
    step[k] = medIn[k] - medOut[k];
    have[k] = Number.isFinite(step[k]) ? 1 : 0;
  }
  const craterBins = new Set<number>();
  for (let i = 0; i < n; i++) if (crater.data[i]) craterBins.add(tb[i] - tMin);
  const plateau = unmeasured.slice();
  if (have.some((v) => v === 1)) {
    const hs: number[] = [];
    for (let k = 0; k < nb; k++) if (have[k]) hs.push(step[k]);
    const ref = percentile(hs, 90); // plateau height; the slot may be short
    ev.plateau_step_um = ref;
    for (let k = 0; k < nb; k++) {
      const add = ref > 0 ? have[k] && step[k] > p.rectPlateauFrac * Math.abs(ref) : have[k];
      if (add) plateau[k] = 1;
    }
  }
  let cbMin = Infinity, cbMax = -Infinity;
  for (const b of craterBins) {
    plateau[b] = 1;
    if (b < cbMin) cbMin = b;
    if (b > cbMax) cbMax = b;
  }
  if (craterBins.size) for (let b = cbMin; b <= cbMax; b++) plateau[b] = 1;
  const gap = Math.max(1, pyIntRound(p.rectEdgeGapUm / (bw * um)));
  const closed = binaryClosing({ data: plateau, rows: 1, cols: nb }, 1, ones(1, 2 * gap + 1));
  const fb = new Uint8Array(nb);
  for (let k = 0; k < nb; k++) fb[k] = closed.data[k] || plateau[k] ? 1 : 0;
  const { labels: lab, n: nl } = label({ data: fb, rows: 1, cols: nb });
  const run = new Uint8Array(nb);
  if (craterBins.size) {
    const ids = new Set<number>();
    for (const b of craterBins) ids.add(lab.data[b]);
    for (let k = 0; k < nb; k++) run[k] = ids.has(lab.data[k]) ? 1 : 0;
  } else {
    if (nl === 0) throw new Error("attempt to get argmax of an empty sequence");
    const bc = new Int32Array(nl + 1);
    for (let k = 0; k < nb; k++) bc[lab.data[k]]++;
    const id = argmax(bc.subarray(1)) + 1;
    for (let k = 0; k < nb; k++) run[k] = lab.data[k] === id ? 1 : 0;
  }
  let first = -1, last = -1;
  for (let k = 0; k < nb; k++) {
    if (run[k]) {
      if (first < 0) first = k;
      last = k;
    }
  }
  const t0 = (first + tMin) * bw - margin, t1 = (last + tMin + 1) * bw + margin;
  const guard = p.rectEdgeGuardUm / um;
  const mask = new Uint8Array(n), edges = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (!face.data[i]) continue;
    const inT = tco[i] >= t0 && tco[i] <= t1;
    if (!inT) continue;
    if (sco[i] >= s1 - margin && sco[i] <= s2 + margin) mask[i] = 1;
    if (Math.abs(sco[i] - s1) <= guard || Math.abs(sco[i] - s2) <= guard) edges[i] = 1;
  }
  ev.accepted = true;
  const tm = 0.5 * (t0 + t1);
  return {
    mask: { data: mask, rows, cols },
    edges: { data: edges, rows, cols },
    angleDeg: ev.angle_deg as number,
    widthMm,
    lengthMm: (t1 - t0) * pxMm,
    centerPx: [cx + sm * c - tm * s, cy + sm * s + tm * c],
  };
}
