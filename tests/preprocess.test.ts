import { describe, expect, it } from "vitest";
import { fillNearest, hillshade, resample } from "../src/preprocess.js";
import type { Surface } from "../src/x3p.js";
import { decode, decodeF64, loadJson, type ArrFixture } from "./core-fixtures.js";

const fx = loadJson("preprocess.json");
const input = decodeF64(fx.input);
const [R, C] = fx.input.shape as [number, number];

function surf(data: Float64Array, rows: number, cols: number, dx: number, dy: number, name: string): Surface {
  return { z: { data, rows, cols }, dx, dy, name };
}

function crop(data: Float64Array, cols: number, rows2: number, cols2: number): Float64Array {
  const out = new Float64Array(rows2 * cols2);
  for (let r = 0; r < rows2; r++) out.set(data.subarray(r * cols, r * cols + cols2), r * cols2);
  return out;
}

function compareFloat(actual: Float64Array, exp: ArrFixture, rtol: number, atol: number) {
  const e = decodeF64(exp);
  expect(actual.length).toBe(e.length);
  let nanMismatch = 0;
  let maxErr = 0;
  for (let i = 0; i < e.length; i++) {
    const a = actual[i]!, b = e[i]!;
    if (Number.isNaN(a) || Number.isNaN(b)) {
      if (Number.isNaN(a) !== Number.isNaN(b)) nanMismatch++;
      continue;
    }
    maxErr = Math.max(maxErr, Math.abs(a - b) / (atol + rtol * Math.abs(b)));
  }
  return { nanMismatch, maxErr };
}

describe("resample", () => {
  const cases: [string, Surface, number][] = [
    ["iso_70", surf(input, R, C, 50e-6, 50e-6, "pp"), 70.0],
    ["aniso_50x60_to_40", surf(input, R, C, 50e-6, 60e-6, "aniso"), 40.0],
    ["fine_12_to_35", surf(crop(input, C, 90, 110), 90, 110, 12e-6, 12e-6, "fine"), 35.0],
  ];
  for (const [key, s, um] of cases) {
    it(`matches Python (${key})`, () => {
      const exp = fx.resample[key];
      const out = resample(s, um);
      expect([out.z.rows, out.z.cols]).toEqual(exp.z.shape);
      expect(out.dx).toBe(exp.dx);
      expect(out.dy).toBe(exp.dy);
      expect(out.name).toBe(s.name);
      const { nanMismatch, maxErr } = compareFloat(out.z.data, exp.z, 1e-9, 1e-15);
      expect(nanMismatch).toBe(0);
      expect(maxErr).toBeLessThanOrEqual(1);
    });
  }

  it("never upsamples a square grid (returns the same object)", () => {
    const s = surf(input, R, C, 50e-6, 50e-6, "pp");
    expect(fx.resample.noop_is_same_object).toBe(true);
    expect(resample(s, 40.0)).toBe(s);
    expect(resample(s, 50.0)).toBe(s);
  });
});

describe("fillNearest", () => {
  it("matches scipy's nearest-index fill", () => {
    const out = fillNearest({ data: input, rows: R, cols: C });
    const e = decodeF64(fx.fill_nearest);
    let mismatch = 0;
    for (let i = 0; i < e.length; i++) if (out.data[i] !== e[i]) mismatch++;
    expect(mismatch).toBe(0);
  });

  it("returns a copy when nothing is invalid", () => {
    const data = Float64Array.from([1, 2, 3, 4]);
    const out = fillNearest({ data, rows: 2, cols: 2 });
    expect(Array.from(out.data)).toEqual([1, 2, 3, 4]);
    expect(out.data).not.toBe(data);
  });
});

describe("hillshade", () => {
  function compareU8(actual: Uint8Array, exp: ArrFixture) {
    const e = decode(exp) as Uint8Array;
    expect(actual.length).toBe(e.length);
    let maxDiff = 0, diff = 0;
    for (let i = 0; i < e.length; i++) {
      const d = Math.abs(actual[i]! - e[i]!);
      if (d) diff++;
      maxDiff = Math.max(maxDiff, d);
    }
    // float rounding (gaussian, percentile) can move a value across an integer boundary
    expect(maxDiff).toBeLessThanOrEqual(1);
    expect(diff / e.length).toBeLessThan(0.01);
  }

  it("default options match Python", () => {
    compareU8(hillshade({ data: input, rows: R, cols: C }, 50e-6).data, fx.hillshade_default);
  });

  it("custom light, exaggeration and valid mask match Python", () => {
    const valid = decode(fx.hillshade_opts_valid) as Uint8Array;
    const img = hillshade({ data: input, rows: R, cols: C }, 50e-6, {
      valid: { data: valid, rows: R, cols: C },
      azimuthDeg: 90.0,
      altitudeDeg: 50.0,
      exaggeration: 1.5,
    });
    compareU8(img.data, fx.hillshade_opts);
    for (let i = 0; i < valid.length; i++) if (!valid[i]) expect(img.data[i]).toBe(0);
  });

  it("resampled input matches Python", () => {
    const z = decodeF64(fx.hillshade_resampled_input);
    const [r, c] = fx.hillshade_resampled_input.shape;
    compareU8(hillshade({ data: z, rows: r, cols: c }, 70e-6).data, fx.hillshade_resampled);
  });
});
