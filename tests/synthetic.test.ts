import { describe, expect, it } from "vitest";
import { headGeometry, makeHead, NormalRng, type HeadOptions } from "../src/synthetic.js";
import { decodeBits, decodeF64, loadJson } from "./core-fixtures.js";

const fx = loadJson("synthetic.json");

/** Python snake_case kwargs -> HeadOptions */
function toOptions(kw: Record<string, unknown>): HeadOptions {
  const o: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(kw)) o[k.replace(/_([a-z])/g, (_, ch: string) => ch.toUpperCase())] = v;
  return o as HeadOptions;
}

function maskDiff(a: Uint8Array, b: Uint8Array): number {
  let d = 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) d++;
  return d;
}

describe("headGeometry matches Python make_head minus its noise", () => {
  for (const [key, exp] of Object.entries<any>(fx.sets)) {
    it(key, () => {
      const opts = toOptions(exp.kwargs);
      const { zUm, truth, n } = headGeometry(opts);
      expect(n).toBe(exp.n);
      expect(truth.primerCenterPx[0]).toBeCloseTo(exp.primer_center_px[0], 9);
      expect(truth.primerCenterPx[1]).toBeCloseTo(exp.primer_center_px[1], 9);
      expect(truth.pinCenterPx[0]).toBeCloseTo(exp.pin_center_px[0], 9);
      expect(truth.pinCenterPx[1]).toBeCloseTo(exp.pin_center_px[1], 9);
      expect(truth.primerRadiusPx).toBeCloseTo(exp.primer_radius_px, 9);
      expect(truth.pinRadiusPx).toBeCloseTo(exp.pin_radius_px, 9);
      expect(truth.apertureRadiusPx).toBeCloseTo(exp.aperture_radius_px, 9);
      expect(truth.dragAngleDeg).toBe(exp.drag_angle_deg);
      expect(maskDiff(truth.tailMask.data, decodeBits(exp.tail_mask))).toBe(0);
      expect(maskDiff(truth.rectMask.data, decodeBits(exp.rect_mask))).toBe(0);
      expect(maskDiff(truth.shearMask.data, decodeBits(exp.shear_mask))).toBe(0);
      const g = decodeF64(exp.geometry_um); // float32 in the fixture
      let maxErr = 0;
      for (let i = 0; i < g.length; i++) maxErr = Math.max(maxErr, Math.abs(zUm.data[i]! - g[i]!));
      expect(maxErr).toBeLessThan(1e-3); // um; float32 storage + noise subtraction round-off
    });
  }
});

describe("makeHead", () => {
  it("default head: geometry, truth and masks match Python", () => {
    const exp = fx.default_full;
    const [s, truth] = makeHead();
    expect([s.z.rows, s.z.cols]).toEqual([exp.n, exp.n]);
    expect(s.dx).toBe(exp.dx);
    expect(s.dy).toBe(exp.dy);
    expect(s.name).toBe(exp.name);
    let nan = 0, sum = 0;
    for (const v of s.z.data) {
      if (Number.isNaN(v)) nan++;
      else sum += v;
    }
    expect(nan).toBe(exp.nan);
    // noise differs from numpy's stream; its mean is ~0 so the mean height agrees closely
    expect(Math.abs((sum / (s.z.data.length - nan)) * 1e6 - exp.finite_mean_um)).toBeLessThan(0.05);
    expect(truth.primerCenterPx).toEqual(exp.primer_center_px);
    expect(truth.pinCenterPx).toEqual(exp.pin_center_px);
    expect(truth.primerRadiusPx).toBe(exp.primer_radius_px);
    expect(truth.dragAngleDeg).toBe(30);
    expect(maskDiff(truth.tailMask.data, decodeBits(exp.tail_mask))).toBe(0);
    expect(truth.rectMask.data.reduce((a, b) => a + b, 0)).toBe(exp.rect_count);
    expect(truth.shearMask.data.reduce((a, b) => a + b, 0)).toBe(exp.shear_count);
  });

  it("is deterministic per seed and the noise has the Python distribution", () => {
    for (const [key, exp] of Object.entries<any>(fx.sets)) {
      const opts = toOptions(exp.kwargs);
      const [a] = makeHead({ ...opts, invalidCorners: false });
      const [b] = makeHead({ ...opts, invalidCorners: false });
      expect(a.z.data).toEqual(b.z.data);
      const [c] = makeHead(opts);
      expect(c.z.data.filter(Number.isNaN).length, key).toBe(exp.corner_nan);
      if (opts.roughBandUm) continue;
      const { zUm } = headGeometry(opts);
      let ss = 0, m = 0;
      const N = zUm.data.length;
      const noise = new Float64Array(N);
      for (let i = 0; i < N; i++) {
        noise[i] = a.z.data[i]! * 1e6 - zUm.data[i]!;
        m += noise[i]!;
      }
      m /= N;
      for (let i = 0; i < N; i++) ss += (noise[i]! - m) ** 2;
      const std = Math.sqrt(ss / N);
      expect(Math.abs(std / exp.noise_std_um - 1), key).toBeLessThan(0.08);
    }
  });

  it("different seeds give different noise", () => {
    const [a] = makeHead({ umPerPx: 40, seed: 1 });
    const [b] = makeHead({ umPerPx: 40, seed: 2 });
    expect(a.z.data).not.toEqual(b.z.data);
  });

  it("rough band adds unit-RMS-scaled texture only inside the shear mask", () => {
    const opts: HeadOptions = { umPerPx: 40, roughBandUm: 3.0, seed: 7, invalidCorners: false };
    const [withBand, truth] = makeHead(opts);
    const [without] = makeHead({ ...opts, roughBandUm: 0 });
    let ss = 0, k = 0;
    for (let i = 0; i < withBand.z.data.length; i++) {
      const d = (withBand.z.data[i]! - without.z.data[i]!) * 1e6;
      if (truth.shearMask.data[i]) {
        ss += d * d;
        k++;
      } else expect(Math.abs(d)).toBeLessThan(1e-9);
    }
    expect(k).toBeGreaterThan(100);
    expect(Math.sqrt(ss / k)).toBeGreaterThan(1.5);
    expect(Math.sqrt(ss / k)).toBeLessThan(6);
  });
});

describe("NormalRng", () => {
  it("produces standard normal deviates", () => {
    const rng = new NormalRng(42);
    const g = rng.normalGrid(300, 300);
    let m = 0;
    for (const v of g.data) m += v;
    m /= g.data.length;
    let s = 0;
    for (const v of g.data) s += (v - m) ** 2;
    s = Math.sqrt(s / g.data.length);
    expect(Math.abs(m)).toBeLessThan(0.01);
    expect(Math.abs(s - 1)).toBeLessThan(0.01);
  });
});
