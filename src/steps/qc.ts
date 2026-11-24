/** Stage 9 - quality control: a transparent, factorised confidence score (port of `qc.py`). */
import type { Mask } from "../ndimage/index.js";
import type { Params, Result } from "../segment.js";
import { disk } from "./geometry.js";
import { any, fmtF, fmtPct, pyRoundN, pyStr } from "./np.js";

export const MIN_CRATER_MEASURED = 0.7;
export const MIN_CIRCLE_AGREEMENT = 0.85;
/** Breech face must cover at least this fraction of the primer face. */
export const MIN_BREECH_FACE_FRACTION = 0.2;
/** Firing-pin impression (with drag) must cover less than this fraction of the primer face. */
export const MAX_IMPRESSION_FRACTION = 0.5;

const clip01 = (v: number): number => Math.min(Math.max(v, 0), 1);

/**
 * Fill `res.qcFactors`, `res.qcScore` and `res.needsReview` in place.
 *
 * Automatic masks in casework should be reviewed when the evidence for them is weak; the score
 * is a product of interpretable factors in `[0, 1]` (stored in `qcFactors`) so it says *why*:
 *
 * - `primer_edge`      fraction of refinement rays that agree with the primer circle
 * - `primer_data`      measured fraction of the primer face
 * - `firing_pin`       a firing-pin impression was found
 * - `crater_measured`  the crater was measured rather than a data gap
 * - `aperture`         the aperture came from a flow-back ring, not the default
 *
 * Hard sanity gates route a scan to review regardless of the score, each with a short entry in
 * `res.reviewReasons`: crater coverage and circle agreement, a breech face smaller than
 * {@link MIN_BREECH_FACE_FRACTION} of the primer face, an impression larger than
 * {@link MAX_IMPRESSION_FRACTION} of it, a primer diameter outside `p.primerDiameterPriorMm` and a
 * polarity check that saw a central bump but declined to flip.
 *
 * @param res Result being completed (`primerEdgeInlierFrac`, `apertureDetected` and `status`
 *   must already be set).
 */
export function qualityControl(res: Result, face: Mask, valid: Mask, fpi: Mask, p: Params): void {
  const n = face.data.length;
  let nFace = 0, nFaceValid = 0, nFpi = 0, nFpiValid = 0;
  for (let i = 0; i < n; i++) {
    if (face.data[i]) {
      nFace++;
      if (valid.data[i]) nFaceValid++;
    }
    if (fpi.data[i]) {
      nFpi++;
      if (valid.data[i]) nFpiValid++;
    }
  }
  const hasFpi = any(fpi);
  const f: Record<string, number> = {};
  f.primer_edge = clip01(((res.primerEdgeInlierFrac ?? 0) - 0.4) / 0.5);
  f.primer_data = clip01((nFaceValid / Math.max(nFace, 1) - 0.5) / 0.45);
  f.firing_pin = hasFpi ? 1.0 : 0.3;
  const measured = hasFpi ? nFpiValid / Math.max(nFpi, 1) : 0.0;
  f.crater_measured = measured > MIN_CRATER_MEASURED ? 1.0 : 0.85;
  f.aperture = res.apertureDetected ? 1.0 : 0.8;
  res.qcFactors = {};
  for (const [k, v] of Object.entries(f)) res.qcFactors[k] = pyRoundN(v, 3);
  let prod = 1;
  let first = true;
  for (const v of Object.values(f)) {
    prod = first ? v : prod * v;
    first = false;
  }
  res.qcScore = prod;
  const reasons = res.reviewReasons;
  if (res.qcScore < p.qcReviewBelow) reasons.push("low_qc_score");
  if (res.status !== "ok") reasons.push(`status_${res.status}`);
  if (hasFpi) {
    res.qcObservations.crater_measured_fraction = measured;
    if (measured <= MIN_CRATER_MEASURED) {
      reasons.push("crater_unmeasured");
      res.warnings.push(`Only ${fmtPct(measured)} of the impression has measured height. Review crater geometry.`);
    }
    if (res.fpCoreCenterPx !== null && res.fpCoreRadiusPx !== null) {
      const [cx, cy] = res.fpCoreCenterPx;
      const circle = disk([fpi.rows, fpi.cols], cx, cy, res.fpCoreRadiusPx);
      let inter = 0, uni = 0;
      for (let i = 0; i < n; i++) {
        const a = circle.data[i], b = fpi.data[i];
        if (a && b) inter++;
        if (a || b) uni++;
      }
      const agreement = inter / uni;
      res.qcObservations.impression_circle_iou = agreement;
      if (agreement < MIN_CIRCLE_AGREEMENT) {
        reasons.push("impression_not_circular");
        res.warnings.push(
          `Impression agrees with the circular model at IoU ${fmtF(agreement, 2)}. Review the aperture and drag boundaries.`,
        );
      }
    }
    const imp = nFpi / Math.max(nFace, 1);
    res.qcObservations.impression_fraction = imp;
    if (imp >= MAX_IMPRESSION_FRACTION) {
      reasons.push("impression_too_large");
      res.warnings.push(
        `The impression covers ${fmtPct(imp)} of the primer face. The crater was probably not separated from the primer.`,
      );
    }
  }
  const faceMm2 = res.areasMm2.primer_face;
  if (faceMm2) {
    const bf = (res.areasMm2.breech_face_impression ?? 0.0) / faceMm2;
    res.qcObservations.breech_face_fraction = bf;
    if (bf < MIN_BREECH_FACE_FRACTION) {
      reasons.push("breech_face_too_small");
      res.warnings.push(`The breech face covers only ${fmtPct(bf)} of the primer face.`);
    }
  }
  if ((res.qcObservations.primer_in_prior ?? 1.0) < 1.0) {
    const [lo, hi] = p.primerDiameterPriorMm;
    reasons.push("primer_diameter_implausible");
    res.warnings.push(
      `Primer diameter ${fmtF(res.qcObservations.primer_diameter_mm ?? 0, 2)} mm is outside ` +
        `the ${pyStr(lo)}-${pyStr(hi)} mm prior. Check that the primer, not a headstamp feature, was found.`,
    );
  }
  if (res.qcObservations.polarity_ambiguous) reasons.push("polarity_ambiguous");
  res.needsReview = reasons.length > 0;
}
