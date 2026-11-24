/**
 * Automatic masking of cartridge-case head scans (port of `cartmask/segment.py`).
 *
 * The pipeline works on the *height map* (topography), not on a rendered image, so every
 * threshold has a physical meaning (millimetres, micrometres):
 *
 * 1. Primer   - circular Hough transform on height-gradient edges, restricted to physically
 *               plausible primer radii, then refined by casting rays from the centre and fitting
 *               a circle to the steepest outward drop (the primer-cup edge falling into the
 *               primer-pocket gap).
 * 2. Datum    - a robust plane is fitted to the primer face; the residual `d = z - plane` is depth
 *               below the primer surface. A crater that appears as a bump means a cast /
 *               inverted scan: flip.
 * 3. Firing-pin impression (FPI) - hysteresis threshold on depth (seeded by the deepest region,
 *               grown to max(noise floor, 15 % of crater depth)); unmeasured holes inside the
 *               primer (steep craters on confocal scans) are treated as part of the impression.
 * 4. Strike circle - a circle robustly fitted to the round part of the FPI outline gives the pin
 *               centre and radius.
 * 5. Aperture - primer metal flows back into the breech-face firing-pin hole and forms a raised
 *               ring around the crater; on the per-radius median depth profile its outer
 *               fall-off gives the aperture radius.
 *    Drag     - the part of the FPI outside the strike circle, accepted when it is elongated and
 *               points one way; the drag vector runs from the strike centre to the tail tip.
 * 6. Aperture shear - striated texture along the aperture zone on the side opposite the drag;
 *               without a drag, the most textured sector if it is clearly rougher.
 * 7. Breech-face impression - primer face minus the aperture zone, minus the rounded primer-cup
 *               rim and invalid data.
 *
 * This module holds the parameters, the result type, the label values and the orchestrating
 * {@link segment}; each stage lives in `./steps/`.
 */
import { binaryOpening, type Grid, type Mask } from "./ndimage/index.js";
import { fillNearest, resample } from "./preprocess.js";
import {
  apertureZone,
  findAperture,
  findApertureEdge,
  findRectAperture,
  type Evidence,
  type RectAperture,
} from "./steps/aperture.js";
import { fitDatum, isInverted } from "./steps/datum.js";
import { findDrag } from "./steps/drag.js";
import { fitCore, largestSeededComponent, segmentFiringPin } from "./steps/firing_pin.js";
import { disk } from "./steps/geometry.js";
import { any, count, fmtF, fmtG, fmtPct, pyRoundN, pyStr } from "./steps/np.js";
import { findPrimer, primerInsideFraction, type PrimerInfo } from "./steps/primer.js";
import { MAX_IMPRESSION_FRACTION, qualityControl } from "./steps/qc.js";
import { findApertureShear } from "./steps/shear.js";
import type { Surface } from "./x3p.js";

export { fillNearest, resample };

// label values in the output label image (drawing order = brief's masking order)
export const BREECH_FACE = 1;
export const APERTURE_SHEAR = 2;
export const FIRING_PIN_SHEAR = 3;
export const FIRING_PIN_IMPRESSION = 4;
export const LABEL_NAMES: Readonly<Record<number, string>> = Object.freeze({
  [BREECH_FACE]: "breech_face_impression",
  [APERTURE_SHEAR]: "aperture_shear",
  [FIRING_PIN_SHEAR]: "firing_pin_shear",
  [FIRING_PIN_IMPRESSION]: "firing_pin_impression",
});

/**
 * Tunable parameters of {@link segment}, all in physical or relative units.
 *
 * One set of defaults serves every scan and calibre; resolution invariance at 6, 8 and 12 um/px
 * is tested in `tests/robustness.test.ts`. Field names are the camelCase forms of the Python
 * `Params` fields ({@link paramsFromPython} converts a snake_case dict).
 */
export interface Params {
  /** Working resolution [um/px]. Scans are area-averaged to this square grid. Pixel-based sizes
   * elsewhere are derived from millimetres, so results do not depend on it; 8 um keeps a full E3
   * head (3.57 um/px native) at ~1200^2 px. */
  workUmPerPx: number;
  /** Search range of the visible primer-face radius [mm]. Covers small pistol primers (4.45 mm
   * nominal diameter, minus the rounded cup) through large pistol primers (.45 ACP, ~2.65 mm). */
  primerRadiusMm: [number, number];
  /** Plausible visible primer-face diameter [mm] when the case head around it is measured. Small
   * pistol primers are 4.45 mm and large pistol primers 5.33 mm nominal; the rounded cup edge makes
   * the visible face slightly smaller (4.05-4.50 mm on the E3 heads). Hough candidates outside the
   * range are down-weighted, and a chosen primer outside it is routed to review. Circles at the
   * data edge of scans trimmed to the primer are exempt below the range. */
  primerDiameterPriorMm: [number, number];
  /** Minimum scan extent [mm] (shorter side). Anything smaller cannot hold a whole primer (e.g.
   * firing-pin-only scans) and is skipped with a reason. */
  minFieldOfViewMm: number;
  /** Minimum fraction [-] of the primer circle that must lie inside the image; below it the primer
   * is too clipped to segment and the scan is skipped. */
  minPrimerInside: number;
  /** Rim trimmed off the primer face [fraction of primer radius]. The rounded primer-cup edge
   * never touches the breech face, so it is not breech face. */
  rimMarginFrac: number;
  /** Lower bound on the firing-pin-impression edge threshold [um]; also the minimum height of a
   * flow-back ring. Keeps smooth, very quiet scans from growing the crater into tool marks. */
  fpiMinDepthUm: number;
  /** Edge threshold >= `fpiNoiseK` x primer-face roughness [-] (robust sigma), so surface noise
   * neither seeds nor grows the crater. */
  fpiNoiseK: number;
  /** Impression edge at this fraction of the crater depth [-]. Primers dish around the strike; a
   * pure noise threshold swallowed the whole dish. */
  fpiRelDepth: number;
  /** Hysteresis seeds are deeper than this fraction of the crater depth [-] (and than 2 x the edge
   * threshold), so only the crater core seeds. */
  fpiSeedFrac: number;
  /** Aperture radius cap [x pin radius]; the flow-back profile is not searched beyond it (nor
   * beyond 0.85 x primer radius). */
  apertureMaxRatio: number;
  /** Aperture edge where the flow-back profile has fallen to this fraction of its peak [-]; a
   * primer that only rises towards its edge never falls. */
  apertureFall: number;
  /** Drag tail tip must reach beyond this multiple of the pin radius [-]. */
  dragMinReach: number;
  /** Minimum drag tail area [fraction of the pin-strike disk area]. */
  dragMinAreaFrac: number;
  /** Minimum directional coherence of the tail [-]: 1 = points one way, 0 = symmetric ring.
   * Rejects a moat all around the crater. */
  dragMinCoherence: number;
  /** Aperture radius [x pin radius] used when no flow-back ring is found (the QC score then drops). */
  apertureDefaultRatio: number;
  /** Half-width of the aperture-shear search sector [deg], centred opposite the drag (Lightstone
   * 2010: a strong but not exclusive prior). */
  shearSectorDeg: number;
  /** With a drag, the accepted aperture-shear candidate's mean texture energy must exceed this
   * multiple of the breech-face median [-] (sheared metal is rougher than the breech face). */
  shearTextureK: number;
  /** QC score [0..1] below which a scan is flagged `needsReview`. */
  qcReviewBelow: number;
  /** Without a drag, the accepted candidate's mean texture energy must exceed this multiple of the
   * breech-face median [-]; higher than `shearTextureK` because there is no drag to fix sector and
   * axis. */
  shearNodragK: number;
  /** Aperture-shear candidates: oriented texture energy (texture energy x striation coherence x
   * alignment with the shear axis) above this multiple of its breech-face mean [-]. Relative to the
   * breech face, so filing marks that share the direction raise the bar instead of producing shear. */
  shearOrientedK: number;
  /** Smallest accepted aperture-shear candidate [mm^2] (a 1 mm x 50 um band); isolated rough spots
   * are smaller. */
  shearMinAreaMm2: number;
  /** Accepted candidates are grown (hysteresis) into neighbouring band pixels whose
   * oriented-energy ratio exceeds this value [-], so the mask follows the whole striated band. */
  shearGrowK: number;
  /** Minimum orientation consistency `|mean(coherence x exp(2i orientation))|` [0..1] of the
   * accepted candidate: its striations share one direction. */
  shearMinConsistency: number;
  /** The candidate's dominant striation direction must lie within this angle [deg] of the shear
   * axis: the drag axis, or without a drag the radial direction through the candidate. */
  shearAxisTolDeg: number;
  /** High-pass scale [um] (Gaussian sigma) of the striation structure tensor; keeps striations of
   * roughly 20-150 um spacing, drops primer form. */
  shearStriationUm: number;
  /** Averaging window [um] (Gaussian sigma) of the structure tensor: a few striation spacings, so
   * coherence measures *runs* of parallel lines. */
  shearTensorUm: number;
  /** Look for a rectangular (slotted) firing-pin aperture before falling back to the circular
   * model. Round-pin scans are unaffected unless two long, straight, parallel step edges are found. */
  rectDetect: boolean;
  /** Both slot edges must stand out this many times [-] above the median magnitude of the signed
   * gradient projection at their direction. */
  rectMinScore: number;
  /** The slot score must be at least this multiple [-] of its median over all directions; a
   * circular flow-back ring scores alike in every direction (round-pin NIST scans reach at most 2.8). */
  rectMinAnisotropy: number;
  /** Plausible distance between the slot edges [mm]; brackets the Glock rectangular firing-pin
   * hole (~1.1-1.6 mm across the plateau) with margin. */
  rectWidthMm: [number, number];
  /** Along the slot, the zone extends while the strip between the edges stands above the face just
   * outside them by at least this fraction [-] of the plateau height (90th percentile of the step). */
  rectPlateauFrac: number;
  /** Gaps [um] bridged when following the plateau along the slot (unmeasured or damaged spots). */
  rectEdgeGapUm: number;
  /** Half-width [um] of the strips along the slot edges that the aperture-shear search ignores: a
   * straight step along the drag axis would otherwise read as one long striation. */
  rectEdgeGuardUm: number;
  /** Gaussian smoothing [um] of the depth map before the edge projection. */
  rectSmoothUm: number;
  /** Slot edges are searched within this fraction [-] of the primer radius from the primer
   * centre, away from the rounded primer-cup rim. */
  rectSearchFrac: number;
  /** With a slotted aperture, drop the *measured* part of the firing-pin impression (and drag)
   * outside the slot: measured depth below the datum outside the raised slot is the lower primer
   * face, not crater. Unmeasured crater (a data gap) is kept. */
  rectClipImpression: boolean;
  /** When no flow-back ring is found, look for the aperture edge (a wider ring confirmed all
   * around, or a closed outward down-step around the crater) before using the default ratio. Off
   * by default: not yet checked against independent labels. */
  apertureEdge: boolean;
  /** Inner limit of the aperture-edge search [x pin radius], clear of the crater rim (outer limit:
   * `apertureMaxRatio`). */
  apertureEdgeMinRatio: number;
  /** Outer limit [x pin radius] of the wide-ring cue: a flow-back fall-off between
   * `apertureMaxRatio` and this, confirmed in most sectors. */
  apertureEdgeMaxRatio: number;
  /** The wide-ring fall-off must lie within this fraction [-] of the primer radius; further out, a
   * fall-off is the primer dish or cup rounding, which is equally symmetric. */
  apertureRingMaxPrimer: number;
  /** Fraction [-] of twelve 30 deg sectors in which a wide ring's fall-off must be seen (each
   * sector drops by half the global fall). */
  apertureRingMinCover: number;
  /** An aperture-edge step must exceed this multiple [-] of the robust radial-gradient noise of
   * the primer face outside the search range. */
  apertureEdgeK: number;
  /** Fraction [-] of the rays whose strongest step must lie on the fitted edge circle: a closed
   * contour around the crater, not a single scratch. */
  apertureEdgeMinCover: number;
  /** Number of rays [-] cast from the pin centre for the aperture edge. */
  apertureEdgeRays: number;
}

/** Default parameters (identical to Python `Params()`). */
export const DEFAULT_PARAMS: Readonly<Params> = Object.freeze({
  workUmPerPx: 8.0,
  primerRadiusMm: [1.5, 3.0],
  primerDiameterPriorMm: [3.7, 5.6],
  minFieldOfViewMm: 3.4,
  minPrimerInside: 0.75,
  rimMarginFrac: 0.05,
  fpiMinDepthUm: 10.0,
  fpiNoiseK: 5.0,
  fpiRelDepth: 0.15,
  fpiSeedFrac: 0.25,
  apertureMaxRatio: 2.1,
  apertureFall: 0.3,
  dragMinReach: 1.15,
  dragMinAreaFrac: 0.01,
  dragMinCoherence: 0.75,
  apertureDefaultRatio: 1.5,
  shearSectorDeg: 75.0,
  shearTextureK: 1.4,
  qcReviewBelow: 0.7,
  shearNodragK: 1.6,
  shearOrientedK: 2.5,
  shearMinAreaMm2: 0.05,
  shearGrowK: 1.5,
  shearMinConsistency: 0.3,
  shearAxisTolDeg: 20.0,
  shearStriationUm: 24.0,
  shearTensorUm: 48.0,
  rectDetect: true,
  rectMinScore: 10.0,
  rectMinAnisotropy: 3.0,
  rectWidthMm: [0.8, 2.2],
  rectPlateauFrac: 0.3,
  rectEdgeGapUm: 120.0,
  rectEdgeGuardUm: 200.0,
  rectSmoothUm: 16.0,
  rectSearchFrac: 0.8,
  rectClipImpression: true,
  apertureEdge: false,
  apertureEdgeMinRatio: 1.2,
  apertureEdgeMaxRatio: 2.6,
  apertureRingMaxPrimer: 0.6,
  apertureRingMinCover: 0.75,
  apertureEdgeK: 4.0,
  apertureEdgeMinCover: 0.5,
  apertureEdgeRays: 120,
}) as Readonly<Params>;

/** `Params(**overrides)`: defaults with the given fields replaced. */
export function makeParams(overrides: Partial<Params> = {}): Params {
  const p = { ...DEFAULT_PARAMS, ...overrides } as Params;
  p.primerRadiusMm = [...p.primerRadiusMm] as [number, number];
  p.primerDiameterPriorMm = [...p.primerDiameterPriorMm] as [number, number];
  p.rectWidthMm = [...p.rectWidthMm] as [number, number];
  return p;
}

const camel = (k: string): string => k.replace(/_([a-z0-9])/g, (_, c: string) => c.toUpperCase());

/** Params from a Python-style snake_case dict (e.g. `{"work_um_per_px": 10, "aperture_edge": true}`). */
export function paramsFromPython(d: Record<string, unknown> = {}): Params {
  const o: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(d)) {
    const ck = camel(k);
    if (!(ck in DEFAULT_PARAMS)) throw new Error(`unknown parameter ${k}`);
    o[ck] = v;
  }
  return makeParams(o as Partial<Params>);
}

export type Status = "ok" | "partial" | "skipped" | "failed";

/**
 * Output of {@link segment}: geometry, masks, warnings and QC.
 *
 * Pixel quantities refer to the working grid (`umPerPx`), arrays are in specimen view (rows top
 * to bottom as seen), angles in image coordinates. Dictionary-valued fields (`areasMm2`,
 * `qcFactors`, `qcObservations`, `apertureEvidence`, `shearEvidence`) keep the Python keys.
 */
export class Result {
  /** Scan name (file stem). */
  name: string;
  /** Working-grid pixel size [um]. */
  umPerPx: number;
  /** `ok`, `partial` (no firing pin), `skipped` (not a usable head scan) or `failed`. */
  status: Status = "ok";
  /** Human-readable notes on every fallback that was taken. */
  warnings: string[] = [];
  /** Primer-face centre `[x, y]` [px]. */
  primerCenterPx: [number, number] | null = null;
  /** Primer-face radius [px]. */
  primerRadiusPx: number | null = null;
  /** Fraction of the 360 refinement rays whose edge point fits the primer circle. */
  primerEdgeInlierFrac: number | null = null;
  /** Pin-strike centre `[x, y]` [px]. */
  fpCoreCenterPx: [number, number] | null = null;
  /** Pin-strike radius [px]. */
  fpCoreRadiusPx: number | null = null;
  /** Deepest measured point of the crater below the primer datum [um]. */
  fpMaxDepthUm: number | null = null;
  /** Robust high-pass roughness of the primer face [um] (the noise floor). */
  roughnessUm: number | null = null;
  /** Firing-pin aperture radius [px] (measured or default). */
  apertureRadiusPx: number | null = null;
  /** True when the aperture came from evidence (a flow-back ring or a slot) rather than the default ratio. */
  apertureDetected = false;
  /** True when a firing-pin drag was accepted. */
  dragDetected = false;
  /** Drag tail tip `[x, y]` [px]. */
  dragTipPx: [number, number] | null = null;
  /** Drag direction [deg], image coordinates: 0 = +x (3 o'clock), 90 = down. */
  dragAngleDeg: number | null = null;
  /** Strike centre to tail tip [mm] (includes the pin radius). */
  dragLengthMm: number | null = null;
  /** Area of every label plus `primer_face` [mm^2]. */
  areasMm2: Record<string, number> = {};
  /** Confidence in the masks [0..1]: product of `qcFactors`. */
  qcScore = 0.0;
  /** Route to an examiner when true. */
  needsReview = true;
  /** Named factors of `qcScore` (see {@link qualityControl}). */
  qcFactors: Record<string, number> = {};
  /** Height map on the working grid [m]; NaN = invalid (negated if the polarity was flipped). */
  z: Grid<Float64Array> | null = null;
  /** Mask of measured pixels. */
  valid: Mask | null = null;
  /** Depth below the primer datum plane [m] (negative = below the primer face). */
  depth: Grid<Float64Array> | null = null;
  /** Label image: 0 background, then the values in {@link LABEL_NAMES}. */
  labels: Grid<Uint8Array> | null = null;
  /** Measured fractions and shape agreement that explain review triggers. */
  qcObservations: Record<string, number> = {};
  /** Short machine-readable reasons why `needsReview` is true (empty when it is false). */
  reviewReasons: string[] = [];
  /** `circle` (disk of `apertureRadiusPx`) or `rectangle` (slotted aperture; geometry in `apertureEvidence`). */
  apertureShape: "circle" | "rectangle" = "circle";
  /** Measurements behind the aperture decision (slot score, anisotropy, width, `source`, `edge`). */
  apertureEvidence: Evidence = {};
  /** Texture and striation-orientation evidence of the aperture-shear sector. */
  shearEvidence: Evidence = {};

  constructor(name: string, umPerPx: number, init: Partial<Result> = {}) {
    this.name = name;
    this.umPerPx = umPerPx;
    Object.assign(this, init);
  }

  /**
   * JSON-serialisable summary with the keys of Python's `masks.json` (snake_case, same order):
   * every non-array field, plus `primer_diameter_mm`, `fp_core_diameter_mm` and
   * `aperture_diameter_mm` when known; floats rounded to 4 decimals (diameters to 3).
   */
  summary(): Record<string, unknown> {
    const d: Record<string, unknown> = {
      name: this.name,
      um_per_px: this.umPerPx,
      status: this.status,
      warnings: [...this.warnings],
      primer_center_px: this.primerCenterPx,
      primer_radius_px: this.primerRadiusPx,
      primer_edge_inlier_frac: this.primerEdgeInlierFrac,
      fp_core_center_px: this.fpCoreCenterPx,
      fp_core_radius_px: this.fpCoreRadiusPx,
      fp_max_depth_um: this.fpMaxDepthUm,
      roughness_um: this.roughnessUm,
      aperture_radius_px: this.apertureRadiusPx,
      aperture_detected: this.apertureDetected,
      drag_detected: this.dragDetected,
      drag_tip_px: this.dragTipPx,
      drag_angle_deg: this.dragAngleDeg,
      drag_length_mm: this.dragLengthMm,
      areas_mm2: this.areasMm2,
      qc_score: this.qcScore,
      needs_review: this.needsReview,
      qc_factors: this.qcFactors,
      qc_observations: this.qcObservations,
      review_reasons: [...this.reviewReasons],
      aperture_shape: this.apertureShape,
      aperture_evidence: this.apertureEvidence,
      shear_evidence: this.shearEvidence,
    };
    const mm = this.umPerPx / 1000.0;
    // Python truthiness: None and 0.0 are skipped
    if (this.primerRadiusPx) d.primer_diameter_mm = pyRoundN(2 * this.primerRadiusPx * mm, 3);
    if (this.fpCoreRadiusPx) d.fp_core_diameter_mm = pyRoundN(2 * this.fpCoreRadiusPx * mm, 3);
    if (this.apertureRadiusPx) d.aperture_diameter_mm = pyRoundN(2 * this.apertureRadiusPx * mm, 3);
    return roundDeep(d) as Record<string, unknown>;
  }
}

/** Recursively round floats to 4 decimals (Python `_round`). Integers are left alone. */
function roundDeep(o: unknown): unknown {
  if (typeof o === "number") return Number.isInteger(o) ? o : pyRoundN(o, 4);
  if (Array.isArray(o)) return o.map(roundDeep);
  if (o !== null && typeof o === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(o)) out[k] = roundDeep(v);
    return out;
  }
  return o;
}

/**
 * Reasons *against* flipping the heights; an empty list means flip.
 *
 * A central bump alone is weak evidence: strong flow-back into the firing-pin aperture also
 * raises the middle of a correctly oriented primer. The flip is accepted only when, with negated
 * heights, (1) the signed primer-edge evidence is stronger than as measured and (2) the crater
 * segments as a plausible firing-pin impression: smaller than `MAX_IMPRESSION_FRACTION` of the
 * primer face and a strike circle at most half the primer diameter.
 */
function flipEvidence(
  zf: Grid<Float64Array>,
  valid: Mask,
  flipped: [number, number, number, number, number],
  score: number,
  pxMm: number,
  p: Params,
): string[] {
  const [fcx, fcy, frp, , fscore] = flipped;
  const reasons: string[] = [];
  if (!(fscore > score)) reasons.push(`primer edge evidence ${fmtG(fscore, 3)} flipped vs ${fmtG(score, 3)} as measured`);
  const face = disk([zf.rows, zf.cols], fcx, fcy, frp * (1 - p.rimMarginFrac));
  let nfv = 0;
  for (let i = 0; i < face.data.length; i++) if (face.data[i] && valid.data[i]) nfv++;
  if (any(face) && nfv > 100) {
    const [d, rough] = fitDatum(zf, valid, face, fcx, fcy, frp);
    const [fpi] = segmentFiringPin(d, valid, face, [fcx, fcy], frp, p, rough, []);
    if (!any(fpi)) reasons.push("no crater when flipped");
    else {
      const frac = count(fpi) / count(face);
      const [, , rc] = fitCore(fpi);
      if (frac > MAX_IMPRESSION_FRACTION || rc > 0.5 * frp) {
        reasons.push(
          `flipped crater implausible: ${fmtPct(frac)} of the primer face, strike diameter ${fmtF(2 * rc * pxMm, 2)} mm`,
        );
      }
    }
  } else reasons.push("no primer face when flipped");
  return reasons;
}

function negate(z: Grid<Float64Array>): Grid<Float64Array> {
  const out = new Float64Array(z.data.length);
  for (let i = 0; i < out.length; i++) out[i] = -z.data[i];
  return { data: out, rows: z.rows, cols: z.cols };
}

/**
 * Mask the primer regions of one cartridge-case head scan.
 *
 * @param surf Height map (metres, NaN = invalid) with its pixel spacing, in specimen view (as
 *   returned by `readX3P`).
 * @param params Parameters; defaults to {@link DEFAULT_PARAMS}. A partial object is merged with
 *   the defaults.
 * @returns Geometry, label image and QC. `status` is `skipped` for scans that cannot contain a
 *   whole primer, `failed` for unusable data and `partial` when no firing-pin impression was
 *   found; each fallback adds a line to `warnings`.
 */
export function segment(surf: Surface, params: Partial<Params> = {}): Result {
  const p = makeParams(params);
  const s = resample(surf, p.workUmPerPx);
  const pxMm = s.dx * 1e3;
  let z = s.z;
  const { rows, cols } = z;
  const n = rows * cols;
  const valid: Mask = { data: new Uint8Array(n), rows, cols };
  let nValid = 0;
  for (let i = 0; i < n; i++) {
    if (Number.isFinite(z.data[i])) {
      valid.data[i] = 1;
      nValid++;
    }
  }
  const res = new Result(surf.name, s.dx * 1e6, { z, valid });
  const labels: Grid<Uint8Array> = { data: new Uint8Array(n), rows, cols };
  res.labels = labels;
  const shape: [number, number] = [rows, cols];

  const extentMm = Math.min(rows, cols) * pxMm;
  if (extentMm < p.minFieldOfViewMm) {
    res.status = "skipped";
    res.warnings.push(
      `field of view ${fmtF(extentMm, 2)} mm is smaller than a primer; scan probably covers the firing-pin impression only`,
    );
    res.depth = { data: new Float64Array(n), rows, cols };
    return res;
  }
  if (nValid / n < 0.2) {
    res.status = "failed";
    res.warnings.push("less than 20% valid data");
    return res;
  }

  // 1. primer -------------------------------------------------------------------------------
  let pinfo: PrimerInfo = {};
  let [cx, cy, rp, inl, score] = findPrimer(z, valid, pxMm, p, res.warnings, pinfo);
  // 2a. polarity: a crater that looks like a bump can mean a cast or an inverted export, but a
  // primer with strong flow-back looks the same. Flip only when the flipped scan is better on
  // every count.
  if (isInverted(z, valid, cx, cy, rp)) {
    const fw: string[] = [];
    const finfo: PrimerInfo = {};
    const zn = negate(z);
    const flipped = findPrimer(zn, valid, pxMm, p, fw, finfo);
    const reasons = flipEvidence(zn, valid, flipped, score, pxMm, p);
    if (reasons.length === 0) {
      z = zn;
      res.z = z;
      res.warnings.push("height polarity inverted (cast?) - flipped");
      res.warnings.push(...fw);
      [cx, cy, rp, inl, score] = flipped;
      pinfo = finfo;
    } else {
      res.qcObservations.polarity_ambiguous = 1.0;
      res.warnings.push(
        "The primer centre is dominated by a raised feature, but flipping the heights " +
          `was not supported (${reasons.join("; ")}). Heights kept as measured.`,
      );
    }
  }
  res.primerCenterPx = [cx, cy];
  res.primerRadiusPx = rp;
  res.primerEdgeInlierFrac = inl;
  res.qcObservations.primer_diameter_mm = 2 * rp * pxMm;
  res.qcObservations.primer_in_prior = (pinfo.diameter_in_prior ?? true) ? 1.0 : 0.0;
  res.qcObservations.primer_at_data_edge = (pinfo.data_edge ?? false) ? 1.0 : 0.0;
  const inside = primerInsideFraction(shape, cx, cy, rp);
  if (inside < p.minPrimerInside) {
    res.status = "skipped";
    res.warnings.push(`only ${fmtPct(inside)} of the primer circle lies inside the scan`);
    res.depth = { data: new Float64Array(n), rows, cols };
    return res;
  }
  const face = disk(shape, cx, cy, rp * (1 - p.rimMarginFrac));
  if (!any(face)) {
    res.status = "failed";
    return res;
  }

  // 2b. datum plane on the primer face, roughness ---------------------------------------------
  const [d, roughUm] = fitDatum(z, valid, face, cx, cy, rp);
  res.depth = d;
  res.roughnessUm = roughUm;

  // 3. firing-pin impression ------------------------------------------------------------------
  let [fpi, , maxDepth] = segmentFiringPin(d, valid, face, [cx, cy], rp, p, roughUm, res.warnings);
  res.fpMaxDepthUm = maxDepth;

  let tail: Mask = { data: new Uint8Array(n), rows, cols };
  let zone: Mask = { data: new Uint8Array(n), rows, cols };
  if (any(fpi)) {
    // 4. pin-strike circle ---------------------------------------------------------------------
    let [fcx, fcy, rc] = fitCore(fpi);

    // 5a. rectangular (slotted) aperture ---------------------------------------------------------
    let rect: RectAperture | null = null;
    if (p.rectDetect) rect = findRectAperture(d, valid, face, fpi, fcx, fcy, rc, rp, pxMm, p, res.apertureEvidence);
    if (rect !== null) {
      res.apertureShape = "rectangle";
      res.apertureEvidence.center_px = rect.centerPx;
      res.apertureEvidence.length_mm = rect.lengthMm;
      res.warnings.push(`rectangular aperture: ${fmtF(rect.widthMm, 2)} mm slot at ${fmtF(rect.angleDeg, 0)} deg`);
      if (p.rectClipImpression) {
        // unmeasured crater beyond the slot stays crater
        const keep: Mask = { data: new Uint8Array(n), rows, cols };
        for (let i = 0; i < n; i++) keep.data[i] = fpi.data[i] && (rect.mask.data[i] || !valid.data[i]) ? 1 : 0;
        const dk = disk(shape, fcx, fcy, rc);
        const w = new Float64Array(n);
        for (let i = 0; i < n; i++) w[i] = dk.data[i];
        const inSlot = largestSeededComponent(keep, keep, { data: w, rows, cols });
        const nIn = count(inSlot), nF = count(fpi);
        if (nIn > 0 && nIn < nF) {
          res.warnings.push(`impression clipped to the slot (${fmtPct(1 - nIn / nF)} outside it)`);
          fpi = inSlot;
          [fcx, fcy, rc] = fitCore(fpi);
        }
      }
    }
    res.fpCoreCenterPx = [fcx, fcy];
    res.fpCoreRadiusPx = rc;

    // 5b. round aperture (flow-back plateau) ------------------------------------------------------
    let ra: number;
    if (rect !== null) {
      ra = (0.5 * rect.widthMm) / pxMm;
      res.apertureDetected = true;
    } else {
      let ring = findAperture(d, valid, fcx, fcy, rc, rp, face, roughUm, p);
      if (ring !== null) res.apertureEvidence.source = "flow-back ring";
      else if (p.apertureEdge) {
        const edgeEv: Evidence = {};
        ring = findApertureEdge(d, valid, fcx, fcy, rc, rp, face, pxMm, roughUm, p, edgeEv);
        res.apertureEvidence.edge = edgeEv;
        if (ring !== null) {
          res.apertureEvidence.source = edgeEv.method;
          res.warnings.push(
            `no flow-back ring within ${pyStr(p.apertureMaxRatio)} x pin radius; aperture from the ${String(edgeEv.method)} cue`,
          );
        }
      }
      res.apertureDetected = ring !== null;
      ra = ring !== null ? ring : rc * p.apertureDefaultRatio;
      if (ring === null) {
        res.warnings.push(`no flow-back ring; aperture taken as ${pyStr(p.apertureDefaultRatio)} x firing-pin radius`);
      }
    }
    res.apertureRadiusPx = ra;

    // 6. drag ----------------------------------------------------------------------------------
    const drag = findDrag(fpi, fcx, fcy, rc, face, pxMm, p);
    if (drag !== null) {
      tail = drag.mask;
      res.dragDetected = true;
      res.dragTipPx = drag.tip;
      res.dragAngleDeg = drag.angle;
      res.dragLengthMm = drag.lengthMm;
    } else res.warnings.push("no firing-pin drag detected");

    zone = apertureZone(fpi, fcx, fcy, ra, face, pxMm, rect !== null ? rect.mask : null);

    // 7. aperture shear ------------------------------------------------------------------------
    const shear = findApertureShear(
      d, valid, fpi, zone, face, fcx, fcy, rp, pxMm, res.dragAngleDeg, p, res.shearEvidence,
      rect !== null ? rect.edges : null,
    );
    if (shear === null) res.warnings.push("no aperture shear found");
    else for (let i = 0; i < n; i++) if (shear.data[i]) labels.data[i] = APERTURE_SHEAR;
  } else res.status = "partial";

  // 8. breech face --------------------------------------------------------------------------------
  const bf0 = new Uint8Array(n);
  for (let i = 0; i < n; i++) bf0[i] = face.data[i] && valid.data[i] && !zone.data[i] && labels.data[i] === 0 ? 1 : 0;
  const bf = binaryOpening({ data: bf0, rows, cols }, 1);
  for (let i = 0; i < n; i++) {
    if (bf.data[i]) labels.data[i] = BREECH_FACE;
    if (fpi.data[i]) labels.data[i] = FIRING_PIN_IMPRESSION;
    if (tail.data[i]) labels.data[i] = FIRING_PIN_SHEAR;
  }

  const a = pxMm * pxMm;
  const counts = new Int32Array(256);
  for (let i = 0; i < n; i++) counts[labels.data[i]]++;
  for (const [v, name] of Object.entries(LABEL_NAMES)) res.areasMm2[name] = counts[Number(v)] * a;
  res.areasMm2.primer_face = count(face) * a;

  // 9. quality control -----------------------------------------------------------------------------
  qualityControl(res, face, valid, fpi, p);
  return res;
}
