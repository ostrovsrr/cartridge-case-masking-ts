/**
 * Orientation-aware aperture shear, slotted apertures and the aperture-edge cue (port of Python
 * `tests/test_aperture_shear.py`).
 *
 * Synthetic heads from `makeHead` carry the true aperture-shear band and slot masks, so
 * detection is checked against known truth, and heads *without* the feature check that the
 * detectors stay silent.
 */
import { describe, expect, it } from "vitest";
import { binaryDilation, type Mask } from "../src/ndimage/index.js";
import { APERTURE_SHEAR, BREECH_FACE, DEFAULT_PARAMS, FIRING_PIN_IMPRESSION, makeParams, segment, type Params, type Result } from "../src/segment.js";
import { disk } from "../src/steps/geometry.js";
import { makeHead, type HeadOptions, type HeadTruth } from "../src/synthetic.js";

function run(kw: HeadOptions, p: Partial<Params> = {}): [Result, HeadTruth] {
  const [surf, truth] = makeHead(kw);
  return [segment(surf, p), truth];
}

const sum = (m: Uint8Array): number => m.reduce((s, v) => s + v, 0);
function shear(res: Result): Uint8Array {
  return Uint8Array.from(res.labels!.data, (v) => (v === APERTURE_SHEAR ? 1 : 0));
}
function andCount(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) if (a[i] && b[i]) s++;
  return s;
}
const label = (res: Result, v: number): Uint8Array => Uint8Array.from(res.labels!.data, (x) => (x === v ? 1 : 0));

// ------------------------------------------------------------------------------- aperture shear

describe("striated band is recovered", () => {
  const cases: HeadOptions[] = [
    { shearUm: 3.0 }, // drag at 30 deg, band opposite
    { shearUm: 1.5, seed: 9 }, // weak striations
    { shearUm: 3.0, dragAngleDeg: null, seed: 5 }, // no drag: radial striations at 180 deg
    { shearUm: 3.0, dragAngleDeg: -60.0, seed: 11 }, // another drag direction
  ];
  for (const kw of cases) {
    it(JSON.stringify(kw), () => {
      const [res, t] = run(kw);
      const sh = shear(res), tm = t.shearMask.data;
      expect(res.shearEvidence.accepted).toBe(true);
      expect(andCount(sh, tm) / sum(tm)).toBeGreaterThan(0.7); // recall
      expect(andCount(sh, tm) / sum(sh)).toBeGreaterThan(0.6); // precision
      expect(res.shearEvidence.axis_error_deg as number).toBeLessThan(15);
    });
  }
});

describe("no shear without striations", () => {
  const cases: HeadOptions[] = [
    {}, // plain dragged head
    { dragAngleDeg: null, seed: 1 },
    { dragAngleDeg: -120.0, seed: 12 },
    { roughBandUm: 3.0 }, // rough but isotropic metal
    { roughBandUm: 3.0, dragAngleDeg: null, seed: 6 },
  ];
  for (const kw of cases) {
    it(JSON.stringify(kw), () => {
      const [res] = run(kw);
      expect(sum(shear(res))).toBe(0);
      expect(res.shearEvidence.accepted).toBe(false);
      expect(res.shearEvidence.reason).toBeTruthy();
    });
  }
});

it("striations across the axis are not shear", () => {
  // the band sits at 210 deg but the drag (120 deg) puts the search sector at 300 deg
  const [res] = run({ shearUm: 3.0, shearAngleDeg: 210.0, shearPeriodMm: 0.06, dragAngleDeg: 120.0, seed: 13 });
  expect(sum(shear(res))).toBe(0);
});

it("breech face gives up the shear band", () => {
  const [res, t] = run({ shearUm: 3.0 });
  const bf = label(res, BREECH_FACE);
  expect(andCount(bf, shear(res))).toBe(0);
  expect(andCount(bf, t.shearMask.data) / sum(t.shearMask.data)).toBeLessThan(0.2);
});

// ------------------------------------------------------------------------------- slotted aperture

describe("slot is detected", () => {
  const cases: [HeadOptions, number][] = [
    [{ rectAperture: [1.3, 3.0], dragAngleDeg: 0.0 }, 0.0],
    [{ rectAperture: [1.3, 3.0], dragAngleDeg: null, seed: 3 }, 0.0],
    [{ rectAperture: [1.1, 2.6], rectAngleDeg: 45.0, dragAngleDeg: 45.0, seed: 4 }, 45.0],
    [{ rectAperture: [1.5, 3.2], rectAngleDeg: 100.0, dragAngleDeg: null, seed: 14 }, 100.0],
  ];
  for (const [kw, angle] of cases) {
    it(JSON.stringify(kw), () => {
      const [res, t] = run(kw);
      const ev = res.apertureEvidence;
      expect(res.apertureShape).toBe("rectangle");
      expect(res.apertureDetected).toBe(true);
      expect(Math.abs(((((ev.angle_deg as number) - angle + 90) % 180) + 180) % 180 - 90)).toBeLessThan(5);
      expect(Math.abs((ev.width_mm as number) - kw.rectAperture![0])).toBeLessThan(0.1 * kw.rectAperture![0]);
      const rect = t.rectMask;
      const nRect = sum(rect.data);
      // the breech face excludes the slot, and the slot is not taken from the breech face beyond it
      expect(andCount(label(res, BREECH_FACE), rect.data) / nRect).toBeLessThan(0.02);
      const [cx, cy] = res.primerCenterPx!;
      const face = disk([rect.rows, rect.cols], cx, cy, 0.9 * res.primerRadiusPx!);
      const near = binaryDilation(rect, 8);
      let lost = 0;
      for (let i = 0; i < face.data.length; i++) {
        if (face.data[i] && res.valid!.data[i] && res.labels!.data[i] !== BREECH_FACE && !near.data[i]) lost++;
      }
      expect(lost / nRect).toBeLessThan(0.1);
      expect(sum(shear(res))).toBe(0); // straight slot edges are not striations
    });
  }
});

describe("round heads stay round", () => {
  const cases: HeadOptions[] = [
    {},
    { dragAngleDeg: null, seed: 1 },
    { shearUm: 3.0 },
    { pinRadiusMm: 0.55, apertureRadiusMm: 0.95, seed: 15 },
    { primerRadiusMm: 2.6, pinRadiusMm: 0.9, apertureRadiusMm: 1.4, sizeMm: 7.6, seed: 16 },
  ];
  for (const kw of cases) {
    it(JSON.stringify(kw), () => {
      const [res] = run(kw);
      expect(res.apertureShape).toBe("circle");
      expect("reason" in res.apertureEvidence).toBe(true);
    });
  }
});

it("slot detection can be switched off", () => {
  const [res] = run({ rectAperture: [1.3, 3.0], dragAngleDeg: 0.0 }, { rectDetect: false });
  expect(res.apertureShape).toBe("circle");
});

it("impression is clipped to the slot", () => {
  // make the crater wider than the slot: everything outside the slot is lower primer face
  const [res, t] = run({ rectAperture: [1.0, 3.0], pinRadiusMm: 0.6, dragAngleDeg: null, seed: 17 });
  expect(res.apertureShape).toBe("rectangle");
  const fpi = label(res, FIRING_PIN_IMPRESSION);
  const slot: Mask = binaryDilation(t.rectMask, 6); // the zone keeps a 40 um margin
  let outside = 0;
  for (let i = 0; i < fpi.length; i++) if (fpi[i] && !slot.data[i]) outside++;
  expect(outside).toBeLessThanOrEqual(0.01 * sum(fpi));
});

// ------------------------------------------------------------------------------- aperture edge cue

const EDGE = makeParams({ apertureEdge: true });

it("takes the aperture from an edge step", () => {
  // no flow-back ring, only a 6 um raised plateau inside the aperture
  const [surf, t] = makeHead({ flowbackUm: 0.0, apertureStepUm: 6.0, dragAngleDeg: null, seed: 7 });
  const res = segment(surf, EDGE);
  expect(res.apertureDetected).toBe(true);
  expect(res.apertureEvidence.source).toBe("edge step");
  expect(Math.abs(res.apertureRadiusPx! - t.apertureRadiusPx)).toBeLessThan(0.1 * t.apertureRadiusPx);
});

it("keeps the default without an edge", () => {
  const [surf] = makeHead({ flowbackUm: 0.0, dragAngleDeg: null, seed: 8 });
  const res = segment(surf, EDGE);
  expect(res.apertureDetected).toBe(false);
  expect(res.apertureRadiusPx).toBeCloseTo(res.fpCoreRadiusPx! * DEFAULT_PARAMS.apertureDefaultRatio, 9);
  expect(res.qcFactors.aperture).toBeLessThan(1.0);
  expect((res.apertureEvidence.edge as Record<string, unknown>).accepted).toBeFalsy();
});

it("edge cue is off by default", () => {
  const [surf] = makeHead({ flowbackUm: 0.0, apertureStepUm: 6.0, dragAngleDeg: null, seed: 7 });
  const res = segment(surf);
  expect(res.apertureDetected).toBe(false);
  expect("edge" in res.apertureEvidence).toBe(false);
});

it("default head is unchanged by the new options", () => {
  const [a, ta] = makeHead();
  const [b] = makeHead({ rectAperture: null, apertureStepUm: 0.0, shearUm: 0.0, roughBandUm: 0.0 });
  for (let i = 0; i < a.z.data.length; i++) {
    const x = a.z.data[i], y = b.z.data[i];
    if (!(Object.is(x, y) || (Number.isNaN(x) && Number.isNaN(y)))) throw new Error(`differs at ${i}`);
  }
  expect(sum(ta.shearMask.data)).toBe(0);
  expect(sum(ta.rectMask.data)).toBe(0);
});
