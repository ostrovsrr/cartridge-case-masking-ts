/**
 * Sanity gates: primer plausibility, cautious polarity flips and hard review triggers
 * (synthetic parts of Python `tests/test_sanity_gates.py`; the real-scan regressions are covered
 * by `parity.test.ts`).
 */
import { describe, expect, it } from "vitest";
import { makeParams, Result, segment } from "../src/segment.js";
import { disk } from "../src/steps/geometry.js";
import { caseHead, plausibility } from "../src/steps/primer.js";
import { qualityControl } from "../src/steps/qc.js";
import { makeHead } from "../src/synthetic.js";
import type { Mask } from "../src/ndimage/index.js";

const UM = 8.0;
const MM = UM / 1000;

function ones(rows: number, cols: number): Mask {
  return { data: new Uint8Array(rows * cols).fill(1), rows, cols };
}

function finiteMask(z: { data: Float64Array; rows: number; cols: number }): Mask {
  return { data: Uint8Array.from(z.data, (v) => (Number.isFinite(v) ? 1 : 0)), rows: z.rows, cols: z.cols };
}

function qc(areas: Record<string, number> = {}, observations: Record<string, number> = {}, fpiR = 10): Result {
  const shape: [number, number] = [96, 96];
  const res = new Result("synthetic", UM, {
    primerEdgeInlierFrac: 1,
    apertureDetected: true,
    fpCoreCenterPx: [48, 48],
    fpCoreRadiusPx: fpiR,
  });
  res.areasMm2 = areas;
  Object.assign(res.qcObservations, observations);
  qualityControl(res, disk(shape, 48, 48, 38), ones(96, 96), disk(shape, 48, 48, fpiR), makeParams());
  return res;
}

describe("QC gates", () => {
  it("clean result passes all gates", () => {
    const res = qc({ breech_face_impression: 6.0, primer_face: 10.0 });
    expect(res.qcScore).toBe(1);
    expect(res.needsReview).toBe(false);
    expect(res.reviewReasons).toEqual([]);
  });

  it("zero breech face is routed to review", () => {
    const res = qc({ breech_face_impression: 0.0, primer_face: 10.0 });
    expect(res.needsReview).toBe(true);
    expect(res.reviewReasons).toContain("breech_face_too_small");
    expect(res.qcScore).toBe(1); // the composite score alone would have passed it
  });

  it("impression covering half the primer is routed to review", () => {
    const res = qc({ breech_face_impression: 3.0, primer_face: 10.0 }, {}, 30);
    expect(res.qcObservations.impression_fraction).toBeGreaterThanOrEqual(0.5);
    expect(res.reviewReasons).toContain("impression_too_large");
    expect(res.needsReview).toBe(true);
  });

  it("primer outside the diameter prior is routed to review", () => {
    const res = qc({ breech_face_impression: 6.0, primer_face: 10.0 }, { primer_in_prior: 0.0, primer_diameter_mm: 3.1 });
    expect(res.reviewReasons).toContain("primer_diameter_implausible");
    expect(res.warnings.some((w) => w.includes("3.10 mm is outside"))).toBe(true);
  });

  it("declined polarity flip is routed to review", () => {
    const res = qc({ breech_face_impression: 6.0, primer_face: 10.0 }, { polarity_ambiguous: 1.0 });
    expect(res.reviewReasons).toEqual(["polarity_ambiguous"]);
  });
});

describe("primer plausibility", () => {
  it("prefers a concentric primer of physical size", () => {
    const [surf, t] = makeHead({ sizeMm: 10.0, umPerPx: UM, primerCenterMm: [0.0, 0.0] });
    const valid = finiteMask(surf.z);
    const head = caseHead(valid, MM);
    expect(head).not.toBeNull();
    expect(Math.abs(head![2] - 10.0)).toBeLessThan(0.1);
    const [cx, cy] = t.primerCenterPx;
    const r = t.primerRadiusPx;
    const [good, why] = plausibility(surf.z, valid, cx, cy, r, MM, makeParams(), head);
    expect(good).toBe(1.0);
    expect(why.diameter_in_prior && why.crater_inside && !why.data_edge).toBe(true);
    // a 3.0 mm circle on the head 3.5 mm off axis: the headstamp failure mode
    const [decoy, why2] = plausibility(surf.z, valid, cx + 3.5 / MM, cy, 1.5 / MM, MM, makeParams(), head);
    expect(decoy).toBeLessThan(0.05);
    expect(why2.diameter_in_prior).toBe(false);
  });

  it("exempts a trimmed scan's primer at the data edge from the lower bound", () => {
    const [surf, t] = makeHead({ sizeMm: 7.0, umPerPx: UM, primerCenterMm: [0.0, 0.0], primerRadiusMm: 2.2 });
    const [cx, cy] = t.primerCenterPx;
    const keep = disk([surf.z.rows, surf.z.cols], cx, cy, 1.7 / MM); // operator crop: 3.4 mm of data
    for (let i = 0; i < keep.data.length; i++) if (!keep.data[i]) surf.z.data[i] = NaN;
    const valid = finiteMask(surf.z);
    expect(caseHead(valid, MM)).toBeNull();
    const [fac, why] = plausibility(surf.z, valid, cx, cy, 1.7 / MM, MM, makeParams(), null);
    expect(why.data_edge && why.diameter_in_prior).toBe(true);
    expect(fac).toBe(1.0);
  });
});

describe("polarity", () => {
  it("does not flip a crater on a flow-back plateau", () => {
    const [surf] = makeHead({
      seed: 4, dragAngleDeg: null, craterDepthUm: 80.0, flowbackUm: 350.0, apertureRadiusMm: 1.5, pinRadiusMm: 0.6,
    });
    const res = segment(surf);
    expect(res.warnings.some((w) => w.includes("polarity inverted"))).toBe(false);
    expect(res.needsReview).toBe(true);
    expect(res.reviewReasons).toContain("polarity_ambiguous");
  });

  it("still flips a cast", () => {
    const [surf] = makeHead({ seed: 2 });
    for (let i = 0; i < surf.z.data.length; i++) surf.z.data[i] = -surf.z.data[i];
    const res = segment(surf);
    expect(res.warnings.some((w) => w.includes("polarity inverted"))).toBe(true);
    expect(res.reviewReasons).not.toContain("polarity_ambiguous");
  });
});
