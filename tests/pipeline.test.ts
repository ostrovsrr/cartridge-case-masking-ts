/**
 * End-to-end tests on synthetic topography with known ground truth (port of the `segment` parts
 * of Python `tests/test_pipeline.py`). The TypeScript `makeHead` noise differs from NumPy's, so
 * these check behaviour against truth, not bit parity (that is `parity.test.ts`).
 */
import { beforeAll, describe, expect, it } from "vitest";
import { BREECH_FACE, FIRING_PIN_IMPRESSION, FIRING_PIN_SHEAR, Result, makeParams, segment } from "../src/segment.js";
import { makeHead, type HeadTruth } from "../src/synthetic.js";

const dist = (a: readonly number[], b: readonly number[]): number => Math.hypot(a[0] - b[0], a[1] - b[1]);
const countLabel = (res: Result, v: number): number => res.labels!.data.reduce((s, x) => s + (x === v ? 1 : 0), 0);

describe("dragged head", () => {
  let res: Result;
  let t: HeadTruth;
  beforeAll(() => {
    const [surf, truth] = makeHead({ dragAngleDeg: 30.0 });
    res = segment(surf, makeParams({ workUmPerPx: 8.0 }));
    t = truth;
  });

  it("finds the primer", () => {
    expect(res.status).toBe("ok");
    expect(dist(res.primerCenterPx!, t.primerCenterPx)).toBeLessThan(4); // < 32 um
    // the detected edge sits on the rounded cup edge, slightly inside
    expect(Math.abs(res.primerRadiusPx! - t.primerRadiusPx)).toBeLessThan(0.06 * t.primerRadiusPx);
  });

  it("finds the firing-pin impression", () => {
    expect(dist(res.fpCoreCenterPx!, t.pinCenterPx)).toBeLessThan(0.15 * t.pinRadiusPx);
    expect(Math.abs(res.fpCoreRadiusPx! - t.pinRadiusPx)).toBeLessThan(0.2 * t.pinRadiusPx);
    // crater depth 400 um minus the 60 um flow-back it sits in
    expect(Math.abs(res.fpMaxDepthUm! - 340)).toBeLessThan(34);
  });

  it("finds the drag direction", () => {
    expect(res.dragDetected).toBe(true);
    const err = ((((res.dragAngleDeg! - t.dragAngleDeg! + 180) % 360) + 360) % 360) - 180;
    expect(Math.abs(err)).toBeLessThan(15);
    let inter = 0, tm = 0;
    for (let i = 0; i < t.tailMask.data.length; i++) {
      if (t.tailMask.data[i]) {
        tm++;
        if (res.labels!.data[i] === FIRING_PIN_SHEAR) inter++;
      }
    }
    expect(inter / Math.max(tm, 1)).toBeGreaterThan(0.4);
  });

  it("finds the aperture from the flow-back ring", () => {
    expect(res.apertureDetected).toBe(true);
    expect(Math.abs(res.apertureRadiusPx! - t.apertureRadiusPx)).toBeLessThan(0.2 * t.apertureRadiusPx);
  });

  it("keeps the breech face out of the aperture", () => {
    const { rows, cols, data } = res.labels!;
    const [cx, cy] = t.pinCenterPx;
    let bad = 0;
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        if (data[y * cols + x] === BREECH_FACE && Math.hypot(x - cx, y - cy) < 0.9 * t.apertureRadiusPx) bad++;
      }
    }
    expect(bad).toBe(0);
    expect(countLabel(res, BREECH_FACE)).toBeGreaterThan(0);
    expect(countLabel(res, FIRING_PIN_IMPRESSION)).toBeGreaterThan(0);
  });
});

it("finds no drag when the crater is round", () => {
  const [surf] = makeHead({ dragAngleDeg: null, seed: 1 });
  const res = segment(surf);
  expect(res.dragDetected).toBe(false);
  expect(countLabel(res, FIRING_PIN_SHEAR)).toBe(0);
  expect(res.fpCoreRadiusPx).not.toBeNull();
});

it("flips an inverted scan", () => {
  const [surf, t] = makeHead({ seed: 2 });
  for (let i = 0; i < surf.z.data.length; i++) surf.z.data[i] = -surf.z.data[i];
  const res = segment(surf);
  expect(res.warnings.some((w) => w.includes("polarity"))).toBe(true);
  expect(dist(res.fpCoreCenterPx!, t.pinCenterPx)).toBeLessThan(0.2 * t.pinRadiusPx);
});

it("skips a small field of view", () => {
  const [surf] = makeHead({ sizeMm: 1.0 });
  expect(segment(surf).status).toBe("skipped");
});

it("scores QC and passes a clean head", () => {
  const [surf] = makeHead({ dragAngleDeg: 20.0 });
  const res = segment(surf);
  expect(res.qcScore).toBeGreaterThanOrEqual(0);
  expect(res.qcScore).toBeLessThanOrEqual(1);
  expect(res.needsReview).toBe(false);
});

it("summary has the masks.json keys", () => {
  const [surf] = makeHead({ dragAngleDeg: 20.0, sizeMm: 6.0, umPerPx: 10.0 });
  const s = segment(surf, { workUmPerPx: 10.0 }).summary();
  expect(Object.keys(s)).toEqual([
    "name", "um_per_px", "status", "warnings", "primer_center_px", "primer_radius_px", "primer_edge_inlier_frac",
    "fp_core_center_px", "fp_core_radius_px", "fp_max_depth_um", "roughness_um", "aperture_radius_px",
    "aperture_detected", "drag_detected", "drag_tip_px", "drag_angle_deg", "drag_length_mm", "areas_mm2", "qc_score",
    "needs_review", "qc_factors", "qc_observations", "review_reasons", "aperture_shape", "aperture_evidence",
    "shear_evidence", "primer_diameter_mm", "fp_core_diameter_mm", "aperture_diameter_mm",
  ]);
  expect(Object.keys(s.areas_mm2 as object)).toEqual([
    "breech_face_impression", "aperture_shear", "firing_pin_shear", "firing_pin_impression", "primer_face",
  ]);
});
