/**
 * Stage 7 - aperture shear: striated primer metal at the aperture edge (port of `shear.py`).
 *
 * Two cues are combined:
 *
 * - **texture energy** - local RMS of the high-pass residual; sheared metal is rougher than the
 *   breech face;
 * - **striation orientation** - a structure tensor on the high-pass residual gives, per pixel,
 *   the direction of the striations and how one-directional they are (coherence). Aperture shear
 *   is cut by the aperture edge while the barrel unlocks, i.e. by the same relative motion that
 *   drags the pin, so its striations run *parallel to the drag axis* and, seen from the pin
 *   centre, *radially* through the sheared sector. Breech-face filing marks, crater-rim steps
 *   (tangential) and isotropic roughness do not share that geometry.
 *
 * A sector of the aperture band is accepted only when it is rougher than the breech face *and*
 * its striations are more coherently aligned with the shear axis than the breech face is with
 * the same axis.
 */
import {
  binaryClosing,
  binaryDilation,
  binaryOpening,
  binaryPropagation,
  gaussianFilter,
  gradient,
  label,
  labelSums,
  median,
  pairwiseSum,
  percentile,
  uniformFilter,
  type Grid,
  type Mask,
} from "../ndimage/index.js";
import { fillNearest } from "../preprocess.js";
import type { Params } from "../segment.js";
import type { Evidence } from "./aperture.js";
import { and, any, andNot, count, deg2rad, pyIntRound, pyMod, rad2deg, whereValid } from "./np.js";

type F = Grid<Float64Array>;

/**
 * Local RMS of the high-pass residual: high on striated / sheared metal.
 *
 * @param d Height or depth map (any unit); NaN-free where `valid`.
 * @param valid Measured pixels; the rest is filled from the nearest valid neighbour.
 * @returns Non-negative map in the unit of `d` (Gaussian sigma 3 px high-pass, 9 x 9 px RMS window).
 */
export function textureEnergy(d: F, valid: Mask): F {
  const df = fillNearest(whereValid(d, valid));
  const g = gaussianFilter(df, 3.0);
  const n = df.data.length;
  const hp2 = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const h = df.data[i] - g.data[i];
    hp2[i] = h * h;
  }
  const u = uniformFilter({ data: hp2, rows: d.rows, cols: d.cols }, 9);
  for (let i = 0; i < n; i++) u.data[i] = Math.sqrt(Math.max(u.data[i], 0));
  return u;
}

/**
 * Per-pixel striation orientation and coherence from a structure tensor.
 *
 * The depth map is band-passed (`p.shearStriationUm` high-pass, 1 px smoothing), gradient
 * magnitudes are capped at their 95th percentile so a single step edge cannot dominate a window,
 * and the tensor is averaged over a Gaussian window of `p.shearTensorUm`.
 *
 * @returns `[coherence, orientation]`: coherence in `[0, 1]` and the striation direction [rad],
 *   image coordinates, modulo pi.
 */
export function striationField(d: F, valid: Mask, pxMm: number, p: Params): [F, F] {
  const { rows, cols } = d;
  const n = rows * cols;
  const um = pxMm * 1000.0;
  const df = fillNearest(whereValid(d, valid));
  const a = gaussianFilter(df, 1.0), b = gaussianFilter(df, Math.max(1.5, p.shearStriationUm / um));
  const hp = new Float64Array(n);
  for (let i = 0; i < n; i++) hp[i] = a.data[i] - b.data[i];
  const { gy, gx } = gradient({ data: hp, rows, cols });
  const gxd = gx.data, gyd = gy.data;
  const mag = new Float64Array(n);
  let nv = 0;
  for (let i = 0; i < n; i++) {
    mag[i] = Math.sqrt(gxd[i] * gxd[i] + gyd[i] * gyd[i]);
    if (valid.data[i]) nv++;
  }
  let cap = 1.0;
  if (nv > 0) {
    const mv = new Float64Array(nv);
    for (let i = 0, k = 0; i < n; i++) if (valid.data[i]) mv[k++] = mag[i];
    cap = percentile(mv, 95);
  }
  const xx = new Float64Array(n), yy = new Float64Array(n), xy = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const k = Math.min(1.0, cap / Math.max(mag[i], 1e-30));
    const x = gxd[i] * k, y = gyd[i] * k;
    xx[i] = x * x;
    yy[i] = y * y;
    xy[i] = x * y;
  }
  const s = Math.max(1.0, p.shearTensorUm / um);
  const jxx = gaussianFilter({ data: xx, rows, cols }, s).data;
  const jyy = gaussianFilter({ data: yy, rows, cols }, s).data;
  const jxy = gaussianFilter({ data: xy, rows, cols }, s).data;
  const coh = new Float64Array(n), orient = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const dd = jxx[i] - jyy[i];
    const c = Math.sqrt(dd * dd + 4 * (jxy[i] * jxy[i])) / Math.max(jxx[i] + jyy[i], 1e-30);
    coh[i] = Math.min(Math.max(c, 0.0), 1.0);
    // dominant *gradient* direction; striations run perpendicular to it
    orient[i] = pyMod(0.5 * Math.atan2(2 * jxy[i], dd) + Math.PI / 2, Math.PI);
  }
  return [{ data: coh, rows, cols }, { data: orient, rows, cols }];
}

/**
 * Texture energy carried by striations parallel to `axis`:
 * `energy x coherence x max(cos(2 (orientation - axis)), 0)`.
 *
 * @param axis Reference direction(s) [rad]: a scalar or a per-pixel grid.
 */
export function orientedEnergy(energy: F, coh: F, orient: F, axis: F | number): F {
  const n = energy.data.length;
  const out = new Float64Array(n);
  const ax = typeof axis === "number" ? null : axis.data;
  const a0 = typeof axis === "number" ? axis : 0;
  for (let i = 0; i < n; i++) {
    const t = Math.max(Math.cos(2.0 * (orient.data[i] - (ax ? ax[i] : a0))), 0.0);
    out[i] = energy.data[i] * coh.data[i] * t;
  }
  return { data: out, rows: energy.rows, cols: energy.cols };
}

/** Coherence-weighted mean striation direction [rad, mod pi] and its consistency [0..1]. */
function orientationStats(coh: F, orient: F, m: Mask): [number, number] {
  const re: number[] = [], im: number[] = [];
  for (let i = 0; i < m.data.length; i++) {
    if (!m.data[i]) continue;
    const c = coh.data[i], t = 2 * orient.data[i];
    re.push(c * Math.cos(t));
    im.push(c * Math.sin(t));
  }
  // NumPy complex division by (k + 0j): (a + b*0) * (1 / k)
  const scl = 1.0 / Math.max(re.length, 1);
  const zr = pairwiseSum(re) * scl, zi = pairwiseSum(im) * scl;
  return [pyMod(0.5 * Math.atan2(zi, zr), Math.PI), Math.sqrt(zr * zr + zi * zi)];
}

function maskedMean(v: Float64Array, m: Mask): number {
  const out: number[] = [];
  for (let i = 0; i < m.data.length; i++) if (m.data[i]) out.push(v[i]);
  return pairwiseSum(out) / out.length;
}

/** `np.angle(np.exp(1j * x))`. */
function wrap(x: number): number {
  return Math.atan2(Math.sin(x), Math.cos(x));
}

/**
 * Sheared, striated primer metal at the aperture edge.
 *
 * As the barrel unlocks, the primer metal that flowed into the firing-pin aperture is sheared
 * against the aperture edge. The striations this leaves run along the unlocking motion: parallel
 * to the drag axis when a drag is known, and through the pin centre (radially) otherwise. Per
 * pixel, the *oriented texture energy* ({@link orientedEnergy}) is divided by its breech-face
 * mean, so breech-face filing marks that share the direction raise the reference rather than
 * producing shear.
 *
 * Candidates are connected regions of the aperture band where that ratio exceeds
 * `p.shearOrientedK`; with a drag only the sector opposite the drag is searched. The strongest
 * candidate (by summed excess) is accepted when it covers at least `p.shearMinAreaMm2`, is
 * rougher than the breech face (`p.shearTextureK` with a drag, `p.shearNodragK` without), and its
 * dominant striation direction is consistent (`p.shearMinConsistency`) and within
 * `p.shearAxisTolDeg` of the shear axis. Further candidates within `p.shearSectorDeg` of it are
 * kept as well, and the kept candidates are grown into neighbouring pixels above `p.shearGrowK`.
 *
 * @param fpi Firing-pin-impression mask (excluded from the search).
 * @param apertureZone Aperture-zone mask; the search band is the zone and its rim.
 * @param cx Pin-strike centre column [px] (the sector apex); `cy` its row.
 * @param rp Primer radius [px] (sets the rim width, 6 % of it).
 * @param dragAngle Drag direction [deg] (image coordinates), or null without drag.
 * @param evidence Optional object filled with the measurements behind the decision (Python keys).
 * @param exclude Optional mask of pixels never searched (e.g. straight slot edges).
 * @returns Shear mask, or null when nothing qualifies.
 */
export function findApertureShear(
  d: F,
  valid: Mask,
  fpi: Mask,
  apertureZone: Mask,
  face: Mask,
  cx: number,
  cy: number,
  rp: number,
  pxMm: number,
  dragAngle: number | null,
  p: Params,
  evidence?: Evidence,
  exclude: Mask | null = null,
): Mask | null {
  const ev: Evidence = evidence ?? {};
  ev.accepted = false;
  const { rows, cols } = d;
  const n = rows * cols;
  const bandOut = Math.max(1, pyIntRound(0.06 * rp));
  const dil = binaryDilation(apertureZone, bandOut);
  const band = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    band[i] = dil.data[i] && face.data[i] && !fpi.data[i] && valid.data[i] && !(exclude && exclude.data[i]) ? 1 : 0;
  }
  const bandM: Mask = { data: band, rows, cols };
  if (!any(bandM)) {
    ev.reason = "empty search band";
    return null;
  }
  const energy = gaussianFilter(textureEnergy(d, valid), 3);
  let bfRef = andNot(and(face, valid), apertureZone);
  if (count(bfRef) < 100) bfRef = and(face, valid);
  if (!any(bfRef)) {
    ev.reason = "no breech-face reference";
    return null;
  }
  const eRef: number[] = [];
  for (let i = 0; i < n; i++) if (bfRef.data[i]) eRef.push(energy.data[i]);
  const ref = median(eRef);
  const [coh, orient] = striationField(d, valid, pxMm, p);
  const ang = new Float64Array(n);
  for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) ang[y * cols + x] = Math.atan2(y - cy, x - cx);
  let axis: F | number;
  const half = deg2rad(p.shearSectorDeg);
  let minEnergy: number;
  if (dragAngle !== null) {
    axis = deg2rad(dragAngle);
    const centre = deg2rad(dragAngle + 180.0);
    for (let i = 0; i < n; i++) if (band[i] && !(Math.abs(wrap(ang[i] - centre)) <= half)) band[i] = 0;
    minEnergy = p.shearTextureK;
  } else {
    axis = { data: ang, rows, cols }; // radial through the pin centre
    minEnergy = p.shearNodragK;
  }
  const sig = Math.max(1.0, p.shearTensorUm / (pxMm * 1000.0));
  const oe = gaussianFilter(orientedEnergy(energy, coh, orient, axis), sig);
  const qd = oe.data;
  const qden = Math.max(maskedMean(qd, bfRef), 1e-30);
  const q = new Float64Array(n);
  for (let i = 0; i < n; i++) q[i] = qd[i] / qden;
  const c0 = new Uint8Array(n);
  for (let i = 0; i < n; i++) c0[i] = band[i] && q[i] > p.shearOrientedK ? 1 : 0;
  const cand = binaryOpening({ data: c0, rows, cols }, 1);
  const { labels: lab, n: nl } = label(cand);
  if (nl === 0) {
    ev.reason = "no striations along the shear axis stand out from the breech face";
    return null;
  }
  const excess = new Float64Array(n);
  for (let i = 0; i < n; i++) excess[i] = q[i] - p.shearOrientedK;
  const mass = labelSums({ data: excess, rows, cols }, lab, nl).subarray(1);
  // np.argsort(mass)[::-1] (descending; exact ties in reverse index order)
  const order = Array.from({ length: nl }, (_, i) => i).sort((a, b) => mass[a] - mass[b] || a - b).reverse();
  const minPx = p.shearMinAreaMm2 / (pxMm * pxMm);
  // per-label pixel lists
  const sizes = new Int32Array(nl + 1);
  const sy = new Float64Array(nl + 1), sx = new Float64Array(nl + 1);
  for (let i = 0; i < n; i++) {
    const l = lab.data[i];
    if (!l) continue;
    sizes[l]++;
    const y = Math.floor(i / cols);
    sy[l] += y;
    sx[l] += i - y * cols;
  }
  const best = order[0] + 1;
  const reg: Mask = { data: new Uint8Array(n), rows, cols };
  for (let i = 0; i < n; i++) reg.data[i] = lab.data[i] === best ? 1 : 0;
  const [dom, cons] = orientationStats(coh, orient, reg);
  const sec = Math.atan2(sy[best] / sizes[best] - cy, sx[best] / sizes[best] - cx);
  const axRef = dragAngle !== null ? deg2rad(dragAngle) : sec;
  const err = rad2deg(Math.abs(wrap(2 * (dom - axRef))) / 2);
  const regPx = sizes[best];
  ev.area_mm2 = regPx * (pxMm * pxMm);
  ev.oriented_ratio = maskedMean(q, reg);
  ev.energy_ratio = maskedMean(energy.data, reg) / Math.max(ref, 1e-30);
  ev.consistency = cons;
  ev.axis_deg = pyMod(rad2deg(dom), 180);
  ev.axis_error_deg = err;
  ev.sector_deg = pyMod(rad2deg(sec), 360);
  const fails: string[] = [];
  if (regPx < minPx) fails.push("too small");
  if ((ev.energy_ratio as number) < minEnergy) fails.push("not rougher than the breech face");
  if (cons < p.shearMinConsistency) fails.push("striation direction not consistent");
  if (err > p.shearAxisTolDeg) fails.push("striations not along the shear axis");
  if (fails.length) {
    ev.reason = "strongest candidate " + fails.join(", ");
    return null;
  }
  const keep = new Uint8Array(nl + 1); // further candidates in the same sector
  keep[best] = 1;
  for (const o of order.slice(1)) {
    const j = o + 1;
    if (sizes[j] < 0.25 * minPx) continue;
    const a = Math.atan2(sy[j] / sizes[j] - cy, sx[j] / sizes[j] - cx);
    if (Math.abs(wrap(a - sec)) <= half) keep[j] = 1;
  }
  const seeds = new Uint8Array(n), gmask = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    seeds[i] = keep[lab.data[i]] && lab.data[i] ? 1 : 0;
    gmask[i] = band[i] && q[i] > p.shearGrowK ? 1 : 0;
  }
  const grown = binaryPropagation({ data: seeds, rows, cols }, { data: gmask, rows, cols });
  const shear = and(binaryClosing(grown, 3), { data: band, rows, cols });
  ev.accepted = true;
  ev.area_mm2 = count(shear) * (pxMm * pxMm);
  return shear;
}

