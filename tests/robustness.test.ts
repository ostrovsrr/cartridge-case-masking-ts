/**
 * Robustness / stress tests of the masking pipeline on synthetic heads: a representative subset
 * of Python `tests/test_robustness.py` (every section, fewer parameter values). Each case is
 * compared against the ground truth returned by `makeHead`, with the Python tolerances:
 *
 *     primer centre  < 32 um            primer radius  within 6 %
 *     pin centre     < 0.15 x pin r     pin radius     within 20 %
 *     aperture       within 20 %        drag angle     within 15 deg
 *
 * Random NaN blobs and extra noise use the TypeScript `NormalRng`, so the realisations differ
 * from NumPy's; the expectations are the same.
 */
import { describe, expect, it } from "vitest";
import { gaussianFilter, percentile } from "../src/ndimage/index.js";
import { FIRING_PIN_IMPRESSION, FIRING_PIN_SHEAR, segment, type Result } from "../src/segment.js";
import { makeHead, NormalRng, type HeadOptions, type HeadTruth } from "../src/synthetic.js";
import type { Surface } from "../src/x3p.js";

const UM = 10.0; // default synthetic / working resolution for the sweeps (fast)
const TOL_PRIMER_CENTRE_MM = 0.032;
const TOL_PRIMER_R = 0.06;
const TOL_PIN_CENTRE = 0.15;
const TOL_PIN_R = 0.2;
const TOL_APERTURE = 0.2;
const TOL_ANGLE = 15.0;

const run = (surf: Surface, um = UM): Result => segment(surf, { workUmPerPx: um });

interface WorkTruth {
  primerCenterPx: [number, number];
  primerRadiusPx: number;
  pinCenterPx: [number, number];
  pinRadiusPx: number;
  apertureRadiusPx: number;
  dragAngleDeg: number | null;
}

/** Ground truth (source px grid) -> the Result's working px grid; `offset` = (col0, row0) of a crop. */
function toWork(t: HeadTruth, srcUm: number, res: Result, offset: [number, number] = [0, 0]): WorkTruth {
  const f = srcUm / res.umPerPx;
  const [c0, r0] = offset;
  const pt = (p: [number, number]): [number, number] => [(p[0] - c0 + 0.5) * f - 0.5, (p[1] - r0 + 0.5) * f - 0.5];
  return {
    primerCenterPx: pt(t.primerCenterPx),
    primerRadiusPx: t.primerRadiusPx * f,
    pinCenterPx: pt(t.pinCenterPx),
    pinRadiusPx: t.pinRadiusPx * f,
    apertureRadiusPx: t.apertureRadiusPx * f,
    dragAngleDeg: t.dragAngleDeg,
  };
}

const dist = (a: readonly number[], b: readonly number[]): number => Math.hypot(a[0] - b[0], a[1] - b[1]);
const angleErr = (a: number, b: number): number => Math.abs(((((a - b + 180) % 360) + 360) % 360) - 180);

function geometryErrors(
  res: Result,
  t: WorkTruth,
  { primer = true, pin = true, aperture = false, drag = false } = {},
): string[] {
  const mm = res.umPerPx / 1000;
  const e: string[] = [];
  if (res.status !== "ok") e.push(`status=${res.status} warnings=${res.warnings.join(" | ")}`);
  if (primer) {
    if (res.primerCenterPx === null) e.push("no primer");
    else {
      const dc = dist(res.primerCenterPx, t.primerCenterPx) * mm;
      if (dc >= TOL_PRIMER_CENTRE_MM) e.push(`primer centre off by ${(dc * 1000).toFixed(0)} um`);
      if (Math.abs(res.primerRadiusPx! - t.primerRadiusPx) >= TOL_PRIMER_R * t.primerRadiusPx)
        e.push(`primer radius ${(res.primerRadiusPx! * mm).toFixed(3)} mm, want ${(t.primerRadiusPx * mm).toFixed(3)}`);
    }
  }
  if (pin) {
    if (res.fpCoreCenterPx === null) e.push(`no firing pin (warnings=${res.warnings.join(" | ")})`);
    else {
      const dc = dist(res.fpCoreCenterPx, t.pinCenterPx);
      if (dc >= TOL_PIN_CENTRE * t.pinRadiusPx) e.push(`pin centre off by ${(dc * mm * 1000).toFixed(0)} um`);
      if (Math.abs(res.fpCoreRadiusPx! - t.pinRadiusPx) >= TOL_PIN_R * t.pinRadiusPx)
        e.push(`pin radius ${(res.fpCoreRadiusPx! * mm).toFixed(3)} mm, want ${(t.pinRadiusPx * mm).toFixed(3)}`);
    }
  }
  if (aperture) {
    if (!res.apertureDetected) e.push("aperture not detected");
    else if (Math.abs(res.apertureRadiusPx! - t.apertureRadiusPx) >= TOL_APERTURE * t.apertureRadiusPx)
      e.push(`aperture radius ${(res.apertureRadiusPx! * mm).toFixed(3)} mm, want ${(t.apertureRadiusPx * mm).toFixed(3)}`);
  }
  if (drag) {
    if (t.dragAngleDeg === null) {
      if (res.dragDetected) e.push(`false drag at ${res.dragAngleDeg!.toFixed(0)} deg`);
    } else if (!res.dragDetected) e.push(`drag at ${t.dragAngleDeg} deg not detected`);
    else if (angleErr(res.dragAngleDeg!, t.dragAngleDeg) >= TOL_ANGLE)
      e.push(`drag angle ${res.dragAngleDeg!.toFixed(1)} deg, want ${t.dragAngleDeg}`);
  }
  return e;
}

const head = (kw: HeadOptions = {}): [Surface, HeadTruth] => makeHead({ umPerPx: UM, sizeMm: 6.5, ...kw });

// 1. calibre ------------------------------------------------------------------------------------
describe("calibre generalisation", () => {
  const CALIBRES: Record<string, HeadOptions> = {
    "small_1.9mm": { primerRadiusMm: 1.9, sizeMm: 6.5 },
    "large_pistol_2.65mm": { primerRadiusMm: 2.65, sizeMm: 7.5 },
  };
  for (const [name, kw] of Object.entries(CALIBRES)) {
    it(name, () => {
      const [surf, truth] = head({ umPerPx: 8.0, seed: 11, ...kw });
      const res = run(surf, 8.0);
      expect(geometryErrors(res, toWork(truth, 8.0, res), { aperture: true, drag: true })).toEqual([]);
    });
  }
});

// 2. drag sweep -----------------------------------------------------------------------------------
describe("drag direction sweep", () => {
  for (const angle of [0, 120, 210, 330, null]) {
    it(String(angle), () => {
      const [surf, truth] = head({ dragAngleDeg: angle, seed: 20 + Math.floor((angle ?? 0) / 30) });
      const res = run(surf);
      expect(geometryErrors(res, toWork(truth, UM, res), { primer: false, pin: false, drag: true })).toEqual([]);
    });
  }
});

// 3. off-centre -----------------------------------------------------------------------------------
describe("off-centre", () => {
  const OFFSETS: [[number, number], [number, number]][] = [
    [[1.0, 0.0], [0.1, 0.05]],
    [[0.15, -0.1], [0.0, -0.8]],
    [[-0.6, -0.6], [-0.5, 0.5]],
  ];
  for (const [pc, off] of OFFSETS) {
    it(`P${pc}-F${off}`, () => {
      const [surf, truth] = head({ sizeMm: 7.8, primerCenterMm: pc, pinOffsetMm: off, seed: 31 });
      const res = run(surf);
      expect(geometryErrors(res, toWork(truth, UM, res), { aperture: true, drag: true })).toEqual([]);
    });
  }
});

// 4. tilt -----------------------------------------------------------------------------------------
describe("global tilt", () => {
  for (const [rampUm, dirDeg] of [[100, 45], [150, 250]]) {
    it(`${rampUm}um@${dirDeg}`, () => {
      const [surf, truth] = head({ seed: 41 });
      const { rows, cols, data } = surf.z;
      const a = (dirDeg * Math.PI) / 180;
      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cols; x++) {
          const u = (Math.cos(a) * (x - cols / 2) + Math.sin(a) * (y - rows / 2)) / cols;
          data[y * cols + x] += rampUm * 1e-6 * u;
        }
      }
      const res = run(surf);
      expect(geometryErrors(res, toWork(truth, UM, res), { aperture: true, drag: true })).toEqual([]);
    });
  }
});

// 5. missing data ---------------------------------------------------------------------------------
function nanBlobs(rows: number, cols: number, frac: number, seed: number, sigmaPx = 5.0): Uint8Array {
  const f = gaussianFilter(new NormalRng(seed).normalGrid(rows, cols), sigmaPx).data;
  const q = percentile(f, 100 * (1.0 - frac));
  return Uint8Array.from(f, (v) => (v > q ? 1 : 0));
}

describe("random NaN blobs", () => {
  for (const frac of [0.1, 0.2]) {
    it(String(frac), () => {
      const [surf, truth] = head({ seed: 51 });
      const b = nanBlobs(surf.z.rows, surf.z.cols, frac, Math.round(frac * 100));
      for (let i = 0; i < b.length; i++) if (b[i]) surf.z.data[i] = NaN;
      const res = run(surf);
      expect(geometryErrors(res, toWork(truth, UM, res))).toEqual([]);
    });
  }
});

describe("unmeasured crater centre", () => {
  for (const [holeFrac, drag] of [[0.5, null], [0.8, 30.0]] as [number, number | null][]) {
    it(`${holeFrac} drag=${drag}`, () => {
      const [surf, truth] = head({ seed: 52, dragAngleDeg: drag });
      const { rows, cols, data } = surf.z;
      const [cx, cy] = truth.pinCenterPx;
      const hole = new Uint8Array(rows * cols);
      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cols; x++) {
          if (Math.hypot(x - cx, y - cy) < holeFrac * truth.pinRadiusPx) {
            hole[y * cols + x] = 1;
            data[y * cols + x] = NaN;
          }
        }
      }
      const res = run(surf);
      const errs = geometryErrors(res, toWork(truth, UM, res), { drag: true });
      let lost = 0, nh = 0;
      for (let i = 0; i < hole.length; i++) {
        if (!hole[i]) continue;
        nh++;
        const l = res.labels!.data[i];
        if (l !== FIRING_PIN_IMPRESSION && l !== FIRING_PIN_SHEAR) lost++;
      }
      if (lost > 0.05 * nh) errs.push(`${((100 * lost) / nh).toFixed(0)}% of the NaN hole not labelled as impression`);
      expect(errs).toEqual([]);
    });
  }
});

// 6. noise ----------------------------------------------------------------------------------------
describe("noise levels", () => {
  for (const noiseUm of [3.0, 10.0]) {
    it(`${noiseUm}um`, () => {
      const [surf, truth] = head({ seed: 61 });
      const rng = new NormalRng(Math.round(noiseUm * 10));
      for (let i = 0; i < surf.z.data.length; i++) surf.z.data[i] += rng.normal(0.0, noiseUm * 1e-6);
      const res = run(surf);
      expect(geometryErrors(res, toWork(truth, UM, res), { aperture: true, drag: true })).toEqual([]);
    });
  }
});

// 7. resolution -----------------------------------------------------------------------------------
describe("resolution invariance", () => {
  const [surf, truth] = makeHead({ sizeMm: 6.5, umPerPx: 4.0, seed: 71 });
  const results = new Map<number, Result>();
  const get = (um: number): Result => {
    if (!results.has(um)) {
      const r = segment(surf, { workUmPerPx: um });
      expect(r.umPerPx).toBeCloseTo(um, 9);
      results.set(um, r);
    }
    return results.get(um)!;
  };
  for (const um of [6.0, 12.0]) {
    for (const what of ["primer", "pin"] as const) {
      it(`${what} at ${um} um/px`, () => {
        const diam = (r: Result): number => (2 * (what === "primer" ? r.primerRadiusPx! : r.fpCoreRadiusPx!) * r.umPerPx) / 1000;
        const dRef = diam(get(8.0)), d = diam(get(um));
        expect(Math.abs(d - dRef)).toBeLessThan(0.03 * dRef);
      });
    }
  }
  for (const um of [6.0, 8.0, 12.0]) {
    it(`geometry vs truth at ${um} um/px`, () => {
      const res = get(um);
      expect(geometryErrors(res, toWork(truth, 4.0, res), { aperture: true, drag: true })).toEqual([]);
    });
  }
});

// 8. non-square FOV -------------------------------------------------------------------------------
describe("non-square field of view", () => {
  const CROPS: Record<string, [[number, number], [number, number], boolean]> = {
    "7x6_centred": [[0.15, -0.1], [7.0, 6.0], true],
    "7x6_gap_near_left": [[-0.85, 0.0], [7.0, 6.0], false],
    "6x7_primer_0.1mm_from_right": [[0.7, 0.0], [6.0, 7.0], false],
  };
  for (const [name, [pc, [wMm, hMm], corners]] of Object.entries(CROPS)) {
    it(name, () => {
      const [surf, truth] = makeHead({ sizeMm: 7.6, umPerPx: UM, primerCenterMm: pc, seed: 81, invalidCorners: corners });
      const n = surf.z.rows;
      const w = Math.round((wMm * 1000) / UM), h = Math.round((hMm * 1000) / UM);
      const c0 = Math.floor((n - w) / 2), r0 = Math.floor((n - h) / 2);
      const crop = new Float64Array(w * h);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) crop[y * w + x] = surf.z.data[(r0 + y) * n + c0 + x];
      surf.z = { data: crop, rows: h, cols: w };
      const res = run(surf);
      expect([res.labels!.rows, res.labels!.cols]).toEqual([h, w]);
      expect(geometryErrors(res, toWork(truth, UM, res, [c0, r0]), { aperture: true, drag: true })).toEqual([]);
    });
  }
});

// 9. polarity -------------------------------------------------------------------------------------
describe("inverted polarity", () => {
  for (const seed of [3, 5]) {
    it(`seed ${seed}`, () => {
      const [surf, truth] = head({ seed, dragAngleDeg: 60.0 * seed });
      for (let i = 0; i < surf.z.data.length; i++) surf.z.data[i] = -surf.z.data[i];
      const res = run(surf);
      const errs = geometryErrors(res, toWork(truth, UM, res), { aperture: true, drag: true });
      if (!res.warnings.some((w) => w.includes("polarity"))) errs.push("no polarity warning");
      expect(errs).toEqual([]);
    });
  }
});

// 10. degenerate ----------------------------------------------------------------------------------
function flatNoise(n = 700): Float64Array {
  return gaussianFilter(new NormalRng(101).normalGrid(n, n, 0, 3e-6), 1.0).data;
}
const surface = (data: Float64Array, rows: number, cols: number, um = UM, name = "x"): Surface => ({
  z: { data, rows, cols },
  dx: um * 1e-6,
  dy: um * 1e-6,
  name,
});

describe("degenerate inputs do not crash", () => {
  const DEGENERATE: Record<string, () => Surface> = {
    all_nan: () => surface(new Float64Array(700 * 700).fill(NaN), 700, 700),
    constant: () => surface(new Float64Array(700 * 700).fill(1e-3), 700, 700),
    zeros: () => surface(new Float64Array(700 * 700), 700, 700),
    tiny_50x50: () => surface(new Float64Array(50 * 50), 50, 50),
    tiny_50x50_coarse_150um: () => surface(flatNoise(50), 50, 50, 150),
    single_row: () => surface(new Float64Array(800), 1, 800),
    primerless_flat_noise: () => surface(flatNoise(), 700, 700),
    primerless_with_nan_corners: () => {
      const f = flatNoise();
      for (let y = 0; y < 700; y++) for (let x = 0; x < 700; x++) if (Math.hypot(y - 350, x - 350) > 350) f[y * 700 + x] = NaN;
      return surface(f, 700, 700);
    },
    mostly_nan_25pct_valid: () => {
      const z = makeHead({ sizeMm: 7.0, umPerPx: UM })[0].z.data.slice();
      const b = nanBlobs(700, 700, 0.75, 7);
      for (let i = 0; i < b.length; i++) if (b[i]) z[i] = NaN;
      return surface(z, 700, 700);
    },
    inf_spikes: () => {
      const z = head()[0].z.data.slice();
      const rng = new NormalRng(3);
      for (let i = 0; i < z.length; i++) if (rng.random() < 0.01) z[i] = Infinity;
      return surface(z, 650, 650);
    },
  };
  for (const [name, mk] of Object.entries(DEGENERATE)) {
    it(name, () => {
      const res = run(mk());
      expect(["ok", "skipped", "failed", "partial"]).toContain(res.status);
    });
  }
});
