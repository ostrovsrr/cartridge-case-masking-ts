import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { unzipSync, zipSync } from "fflate";
import { afterAll, describe, expect, it } from "vitest";
import { encodeX3P, makeSurface, parseX3P, readX3P, surfaceShape, umPerPx, writeX3P } from "../src/x3p.js";
import { checksum, close, decodeF64, FIXTURES, loadJson } from "./core-fixtures.js";

const fx = loadJson("x3p.json");
const tmp = mkdtempSync(join(tmpdir(), "x3p-test-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

function sameArray(a: Float64Array, b: Float64Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (Number.isNaN(a[i]!) && Number.isNaN(b[i]!)) continue;
    if (a[i] !== b[i]) return false;
  }
  return true;
}

describe("readX3P matches the Python reader on fixture files", () => {
  for (const name of Object.keys(fx.files)) {
    it(name, () => {
      const exp = fx.files[name];
      const s = readX3P(join(FIXTURES, name));
      expect(surfaceShape(s)).toEqual(exp.z.shape);
      expect(s.dx).toBe(exp.dx);
      expect(s.dy).toBe(exp.dy);
      expect(s.name).toBe(exp.name);
      expect(sameArray(s.z.data, decodeF64(exp.z))).toBe(true);
      const cs = checksum(s.z.data, s.z.rows, s.z.cols);
      expect(cs.nan).toBe(exp.stats.nan);
    });
  }

  it("imageOrientation=false keeps X3P (Y-up) row order", () => {
    const a = readX3P(join(FIXTURES, "int16_valid.x3p"));
    const b = readX3P(join(FIXTURES, "int16_valid.x3p"), { imageOrientation: false });
    const { rows, cols } = a.z;
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++) {
        const u = a.z.data[r * cols + c]!;
        const v = b.z.data[(rows - 1 - r) * cols + c]!;
        expect(Number.isNaN(u) ? Number.isNaN(v) : u === v).toBe(true);
      }
  });
});

describe("writeX3P", () => {
  it("round-trips a surface exactly (NaN included)", () => {
    const rows = 7, cols = 11;
    const data = new Float64Array(rows * cols).map((_, i) => Math.sin(i) * 1e-5);
    data[5] = NaN;
    data[40] = NaN;
    const s = makeSurface({ data, rows, cols }, 3.2e-6, 4.1e-6, "rt");
    const p = join(tmp, "rt.x3p");
    writeX3P(s, p);
    const back = readX3P(p);
    expect(surfaceShape(back)).toEqual([rows, cols]);
    expect(back.dx).toBe(3.2e-6);
    expect(back.dy).toBe(4.1e-6);
    expect(back.name).toBe("rt");
    expect(umPerPx(back)).toBeCloseTo(3.2, 12);
    expect(sameArray(back.z.data, data)).toBe(true);
  });

  it("re-encodes a Python-written file to identical XML and data", () => {
    const py = readFileSync(join(FIXTURES, "py_written_rect.x3p"));
    const s = parseX3P(py, "rect");
    const ours = unzipSync(encodeX3P(s));
    const theirs = unzipSync(py);
    expect(Object.keys(ours).sort()).toEqual(Object.keys(theirs).sort());
    expect(Buffer.from(ours["main.xml"]!).toString()).toBe(Buffer.from(theirs["main.xml"]!).toString());
    expect(Buffer.compare(Buffer.from(ours["bindata/data.bin"]!), Buffer.from(theirs["bindata/data.bin"]!))).toBe(0);
  });

  it("rejects a data block of the wrong size", () => {
    const s = readX3P(join(FIXTURES, "py_written.x3p"));
    const bad = makeSurface({ data: s.z.data, rows: s.z.rows, cols: s.z.cols }, s.dx, s.dy);
    const files = unzipSync(encodeX3P(bad));
    files["bindata/data.bin"] = files["bindata/data.bin"]!.subarray(0, 80);
    expect(() => parseX3P(zipSync(files), "bad")).toThrow(/expected 1600 values, got 10/);
  });
});

const DATA = process.env.CARTMASK_DATA;
describe.skipIf(!DATA)("real NIST scans (CARTMASK_DATA)", () => {
  for (const rel of Object.keys(fx.real ?? {})) {
    const path = join(DATA ?? "", ...rel.split("/"));
    it.skipIf(!existsSync(path))(rel, () => {
      const exp = fx.real[rel];
      const s = readX3P(path);
      expect(s.dx).toBe(exp.dx);
      expect(s.dy).toBe(exp.dy);
      expect(s.name).toBe(exp.name);
      const cs = checksum(s.z.data, s.z.rows, s.z.cols);
      expect(cs.shape).toEqual(exp.stats.shape);
      expect(cs.nan).toBe(exp.stats.nan);
      expect(cs.first_finite_index).toBe(exp.stats.first_finite_index);
      expect(cs.first_finite).toBe(exp.stats.first_finite);
      expect(cs.min).toBe(exp.stats.min);
      expect(cs.max).toBe(exp.stats.max);
      // summation order differs from numpy's pairwise sum: relative tolerance only
      expect(close(cs.sum, exp.stats.sum, 1e-9, 1e-18)).toBe(true);
      expect(close(cs.sumsq, exp.stats.sumsq, 1e-9, 1e-24)).toBe(true);
      expect(close(cs.wsum, exp.stats.wsum, 1e-9, 1e-16)).toBe(true);
    });
  }
});
