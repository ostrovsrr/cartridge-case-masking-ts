/**
 * Tests for src/evaluate.ts (port of tests/test_evaluate.py): metrics on masks with known overlap,
 * boundary-F tolerance, label I/O round trip, an end-to-end check on synthetic ground truth, the
 * blinded-packet workflow, and parity with the Python scorer (reference/score_reference.py) on
 * the same export; the parity cases are skipped when Python with `cartmask` is unavailable.
 */
import { spawnSync } from "node:child_process";
import { appendFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { beforeAll, describe, expect, it, vi } from "vitest";

vi.mock("../src/segment.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/segment.js")>();
  return { ...actual, segment: vi.fn(actual.segment) };
});

import { build } from "../scripts/build-annotator.js";
import {
  boundary,
  boundaryF,
  CLASSES,
  compareLabels,
  evaluate,
  findAnnotations,
  IGNORE,
  maskMetrics,
  METRICS,
  readLabels,
  run,
  type BlindMap,
  type MaskMetrics,
  type RunOutput,
} from "../src/evaluate.js";
import { encodePng, fromGrid } from "../src/image.js";
import type { Grid, Mask } from "../src/ndimage/index.js";
import {
  APERTURE_SHEAR,
  BREECH_FACE,
  FIRING_PIN_IMPRESSION,
  FIRING_PIN_SHEAR,
  segment,
  type Result,
} from "../src/segment.js";
import { makeHead, type HeadTruth } from "../src/synthetic.js";
import { readX3P, writeX3P, type Surface } from "../src/x3p.js";

const TS_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

function square(shape: [number, number] = [100, 100], y0 = 20, x0 = 20, size = 40): Mask {
  const [rows, cols] = shape;
  const data = new Uint8Array(rows * cols);
  for (let y = y0; y < Math.min(rows, y0 + size); y++) for (let x = x0; x < Math.min(cols, x0 + size); x++) data[y * cols + x] = 1;
  return { data, rows, cols };
}

function disc(n: number, cx: number, cy: number, r: number): Mask {
  const data = new Uint8Array(n * n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) data[y * n + x] = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2) < r ? 1 : 0;
  return { data, rows: n, cols: n };
}

const finite = (m: MaskMetrics): boolean => METRICS.every((k) => Number.isFinite(m[k]));

function labelGrid(rows: number, cols: number): Grid<Uint8Array> {
  return { data: new Uint8Array(rows * cols), rows, cols };
}

function fill(g: Grid<Uint8Array>, y0: number, y1: number, x0: number, x1: number, v: number): void {
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) g.data[y * g.cols + x] = v;
}

function writeX3PTo(s: Surface, path: string): void {
  mkdirSync(dirname(path), { recursive: true });
  writeX3P(s, path);
}

function writeLabels(path: string, g: Grid<Uint8Array>): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, encodePng(fromGrid(g)));
}

function tmp(): string {
  return mkdtempSync(join(tmpdir(), "cartmask-eval-"));
}

// --------------------------------------------------------------------------- overlap metrics
describe("overlap metrics", () => {
  it("perfect overlap", () => {
    const a = square();
    const m = maskMetrics(a, a);
    for (const k of METRICS) expect(m[k]).toBeCloseTo(1.0, 12);
  });

  it("disjoint", () => {
    const m = maskMetrics(square(undefined, 20, 0, 30), square(undefined, 20, 60, 30));
    for (const k of METRICS) expect(m[k]).toBe(0.0);
  });

  it("half overlap", () => {
    // two 40x40 squares shifted by 20 px: intersection 800, union 2400
    const m = maskMetrics(square(undefined, 20, 20), square(undefined, 20, 40));
    expect(m.iou).toBeCloseTo(800 / 2400, 12);
    expect(m.dice).toBeCloseTo((2 * 800) / 3200, 12);
    expect(m.precision).toBeCloseTo(0.5, 12);
    expect(m.recall).toBeCloseTo(0.5, 12);
  });

  it("contained mask precision / recall", () => {
    const big = square(undefined, 20, 20, 40), small = square(undefined, 30, 30, 20);
    const m = maskMetrics(small, big); // prediction inside truth
    expect(m.precision).toBeCloseTo(1.0, 12);
    expect(m.recall).toBeCloseTo(400 / 1600, 12);
  });

  it("empty classes give no NaN", () => {
    const empty = labelGrid(50, 50);
    const both = maskMetrics(empty, empty);
    expect(both.absent_in_both && finite(both) && both.iou === 1.0).toBe(true);
    const missed = maskMetrics(empty, square([50, 50], 10, 10, 10));
    const invented = maskMetrics(square([50, 50], 10, 10, 10), empty);
    for (const m of [missed, invented]) {
      expect(!m.absent_in_both && finite(m)).toBe(true);
      expect([m.iou, m.dice, m.bf_score]).toEqual([0, 0, 0]);
    }
  });

  it("compareLabels covers all classes", () => {
    const a = labelGrid(60, 60);
    fill(a, 10, 30, 10, 30, BREECH_FACE);
    fill(a, 40, 50, 40, 50, FIRING_PIN_IMPRESSION);
    const out = compareLabels(a, { ...a, data: a.data.slice() }, 8.0);
    expect(Object.keys(out).map(Number).sort()).toEqual([...CLASSES].sort());
    expect(out[BREECH_FACE].iou).toBe(1.0);
    expect(out[FIRING_PIN_IMPRESSION].iou).toBe(1.0);
    expect(out[APERTURE_SHEAR].absent_in_both && out[FIRING_PIN_SHEAR].absent_in_both).toBe(true);
    expect(() => compareLabels(a, { data: a.data.slice(0, 59 * 60), rows: 59, cols: 60 }, 8.0)).toThrow(/label shapes differ/);
  });
});

// --------------------------------------------------------------------------- boundary F
describe("boundary F", () => {
  it("boundary is a one-pixel ring", () => {
    const b = boundary(square(undefined, 20, 20, 10));
    expect(b.data.reduce((s, v) => s + v, 0)).toBe(36); // 4 * 10 - 4 corners counted once
    expect(boundary(labelGrid(5, 5)).data.some((v) => v)).toBe(false);
  });

  it.each([
    [0, 0, 1.0], // identical
    [1, 2, 1.0], // 1 px shift is within a 2 px tolerance
    [2, 2, 1.0], // exactly at the tolerance
    [1, 0, null], // any shift fails a zero tolerance (partially)
    [4, 2, null], // 4 px shift beyond 2 px tolerance
  ])("shift %d at tolerance %d", (shift, tol, expected) => {
    const a = square(undefined, 20, 20), b = square(undefined, 20, 20 + shift);
    const f = boundaryF(a, b, tol)[2];
    if (expected !== null) expect(f).toBeCloseTo(expected, 12);
    else {
      expect(f).toBeLessThan(0.75);
      // the same shift passes when the tolerance is large enough
      expect(boundaryF(a, b, shift)[2]).toBeCloseTo(1.0, 12);
    }
  });

  it("is monotone in the tolerance", () => {
    const a = disc(120, 60, 60, 40), b = disc(120, 62, 59, 37);
    const fs = [0, 1, 2, 4, 8].map((t) => boundaryF(a, b, t)[2]);
    for (let i = 1; i < fs.length; i++) expect(fs[i - 1]).toBeLessThanOrEqual(fs[i] + 1e-12);
    expect(fs[fs.length - 1]).toBeCloseTo(1.0, 12);
  });

  it("IoU is insensitive where boundary F is sensitive", () => {
    // large region, 3 px systematic offset: IoU stays high, BF at 2 px tolerance drops
    const a = disc(300, 150, 150, 120), b = disc(300, 150, 150, 117);
    const m = maskMetrics(b, a, 2.0);
    expect(m.iou).toBeGreaterThan(0.9);
    expect(m.bf_score).toBeLessThan(0.1);
  });
});

// --------------------------------------------------------------------------- end to end
/** Label image from the synthetic ground truth (see src/synthetic.ts makeHead). */
function truthLabels(rows: number, cols: number, t: HeadTruth): Grid<Uint8Array> {
  const lab = labelGrid(rows, cols);
  const [pcx, pcy] = t.primerCenterPx, [fcx, fcy] = t.pinCenterPx;
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const i = y * cols + x;
      const rPrimer = Math.sqrt((x - pcx) ** 2 + (y - pcy) ** 2), rPin = Math.sqrt((x - fcx) ** 2 + (y - fcy) ** 2);
      if (rPrimer <= t.primerRadiusPx && rPin > t.apertureRadiusPx) lab.data[i] = BREECH_FACE;
      if (rPin <= t.pinRadiusPx) lab.data[i] = FIRING_PIN_IMPRESSION;
      if (t.tailMask.data[i]) lab.data[i] = FIRING_PIN_SHEAR;
    }
  }
  return lab;
}

let surf: Surface, res: Result, lab: Grid<Uint8Array>;

describe("synthetic head", () => {
  beforeAll(() => {
    let t: HeadTruth;
    [surf, t] = makeHead({ dragAngleDeg: 30.0 });
    res = segment(surf, { workUmPerPx: 8.0 });
    lab = truthLabels(res.labels!.rows, res.labels!.cols, t);
  });

  it("end to end", () => {
    const m = evaluate(res, lab);
    expect(m.classes[BREECH_FACE].iou).toBeGreaterThan(0.75);
    expect(m.classes[FIRING_PIN_IMPRESSION].iou).toBeGreaterThan(0.7);
    expect(m.classes[FIRING_PIN_IMPRESSION].precision).toBeGreaterThan(0.95);
    expect(m.primer_center_err_um!).toBeLessThan(16); // < 2 px
    expect(m.pin_center_err_um!).toBeLessThan(16);
    for (const k of CLASSES) expect(finite(m.classes[k])).toBe(true);
  });

  it("self comparison is perfect", () => {
    const m = evaluate(res, { ...res.labels!, data: res.labels!.data.slice() });
    for (const k of CLASSES) {
      const c = m.classes[k];
      expect(c.absent_in_both || (c.iou === 1.0 && c.bf_score === 1.0)).toBe(true);
    }
  });

  it("shape mismatch raises", () => {
    const short = { data: lab.data.slice(0, (lab.rows - 3) * lab.cols), rows: lab.rows - 3, cols: lab.cols };
    expect(() => evaluate(res, short)).toThrow(/does not match the label grid/);
  });

  it("CLI round trip", () => {
    // Export-like PNG + sidecar -> run() reports IoU 1.0 against the algorithm's own masks,
    // and two annotators trigger the inter-annotator table.
    const dir = tmp();
    writeX3PTo(surf, join(dir, "raw", "synthetic.x3p"));
    for (const [name, img] of [
      ["alice", res.labels!],
      ["bob", lab],
      ["carol", lab],
    ] as const) {
      writeLabels(join(dir, "ann", name, "synthetic__labels.png"), img);
      writeFileSync(
        join(dir, "ann", name, "synthetic__labels.json"),
        JSON.stringify({ scan: "synthetic", annotator: name, prefilled: name === "alice", um_per_px: 8.0, label_shape: [img.rows, img.cols] }),
      );
    }
    expect(readLabels(join(dir, "ann", "alice", "synthetic__labels.png")).data).toBeInstanceOf(Uint8Array);
    expect(findAnnotations(join(dir, "ann"))).toHaveLength(3);
    const out = run(join(dir, "raw"), join(dir, "ann"), join(dir, "eval"));
    const by = Object.fromEntries(out.rows.map((r) => [r.annotator, r]));
    for (const k of CLASSES) {
      const c = by.alice.metrics.classes[k];
      expect(c.absent_in_both || Math.abs(c.iou - 1.0) < 1e-12).toBe(true);
    }
    expect(by.bob.metrics.classes[BREECH_FACE].iou).toBeGreaterThan(0.75);
    expect(new Set(out.inter.map((r) => `${r.annotator_a}|${r.annotator_b}`))).toEqual(new Set(["bob|carol"]));
    const breech = out.summary.find((r) => r.label === BREECH_FACE)!;
    expect(breech.n).toBe(2);
    expect(breech.iou_mean as number).toBeCloseTo(by.bob.metrics.classes[BREECH_FACE].iou, 12);
    for (const f of ["per_scan.csv", "per_scan_long.csv", "summary.csv", "summary.md", "inter_annotator.csv", "error_overlays.png"]) {
      expect(existsSync(join(dir, "eval", f)), f).toBe(true);
    }
    expect(readFileSync(join(dir, "eval", "summary.csv"), "utf8").toLowerCase()).not.toContain("nan");
  });

  it("ignored pixels cannot add errors or fake boundaries", () => {
    const truth = labelGrid(60, 60);
    fill(truth, 10, 45, 10, 45, BREECH_FACE);
    const pred = { ...truth, data: truth.data.slice() };
    fill(truth, 24, 34, 24, 34, IGNORE);
    fill(pred, 24, 34, 24, 34, FIRING_PIN_IMPRESSION);
    const scores = compareLabels(pred, truth, 8);
    expect(scores[BREECH_FACE].iou).toBe(1);
    expect(scores[BREECH_FACE].bf_score).toBe(1);
    expect(scores[FIRING_PIN_IMPRESSION].absent_in_both).toBe(true);
  });

  it("blind packet uses the private hash and the latest completed revision", () => {
    const dir = tmp();
    const source = join(dir, "private-specimen.x3p");
    writeX3PTo(surf, source);
    const seg = vi.mocked(segment);
    const before = seg.mock.calls.length;
    const priv = join(dir, "private.json");
    const page = join(dir, "packet.html");
    const packet = build([source], page, 800, 90, null, priv);
    expect(seg.mock.calls.length, "blinded builder called segmentation").toBe(before);
    expect(packet.scans).toHaveLength(1);
    const kase = packet.scans[0];
    expect("source" in kase || "prefill" in kase).toBe(false);
    expect(readFileSync(page, "utf8")).not.toContain("private-specimen.x3p");
    const meta = { scan: kase.name, packet: packet.packet, done: true, prefilled: false };
    for (const [revision, name, labels] of [
      ["01", "Examiner A", res.labels!],
      ["02", "examiner a", lab],
      ["03", "Examiner B", lab],
    ] as const) {
      const path = join(dir, "annotations", revision, `${kase.name}__labels.png`);
      writeLabels(path, labels);
      writeFileSync(path.replace(/\.png$/, ".json"), JSON.stringify({ ...meta, annotator: name, timestamp: `2026-01-${revision}T12:00:00Z` }));
    }
    const mapping = JSON.parse(readFileSync(priv, "utf8")) as BlindMap;
    const scored = run(dir, join(dir, "annotations"), join(dir, "scores"), undefined, null, mapping);
    expect(scored.rows.map((r) => r.annotator)).toEqual(["examiner a", "Examiner B"]);
    expect(scored.inter).toHaveLength(CLASSES.length);
    expect(scored.skipped).toEqual([]);

    // File name still matches but the bytes must agree with the private map.
    appendFileSync(source, "changed");
    const rejected = run(dir, join(dir, "annotations"), join(dir, "rejected"), undefined, null, mapping);
    expect(rejected.rows).toEqual([]);
    expect(rejected.skipped.length).toBeGreaterThan(0);
    for (const [, , reason] of rejected.skipped) expect(reason).toContain("hash");
  });
});

// --------------------------------------------------------------------------- Python parity
const PYTHON = process.env.CARTMASK_PYTHON ?? "python";
const PY_REPO = resolve(process.env.CARTMASK_PY ?? join(TS_ROOT, "..", "cartridge-case"));
const havePython =
  existsSync(join(PY_REPO, "cartmask", "evaluate.py")) &&
  spawnSync(PYTHON, ["-c", "import sys; sys.path.insert(0, sys.argv[1]); import cartmask.evaluate, cv2", PY_REPO], { encoding: "utf8" })
    .status === 0;
if (!havePython) console.warn("evaluate parity: Python with cartmask (CARTMASK_PY / CARTMASK_PYTHON) unavailable; skipped");

/** Write the TS prediction of `x3p` at `um` where score_reference.py substitutes it for segment(). */
function writePrediction(predDir: string, x3p: string, um: number, i: number): Result {
  const r = segment(readX3P(x3p), { workUmPerPx: um });
  const stem = `${r.name}__${i}`;
  writeLabels(join(predDir, `${stem}.png`), r.labels!);
  writeFileSync(
    join(predDir, `${stem}.json`),
    JSON.stringify({
      name: r.name,
      um_per_px: r.umPerPx,
      work_um_per_px: um,
      status: r.status,
      qc_score: r.qcScore,
      primer_center_px: r.primerCenterPx,
      primer_radius_px: r.primerRadiusPx,
      fp_core_center_px: r.fpCoreCenterPx,
    }),
  );
  return r;
}

/** Score with Python and compare every returned number and every written table with the TS run. */
function expectParity(raw: string, ann: string, dir: string, ts: RunOutput, blindMap?: string): void {
  const pyOut = join(dir, "py");
  const args = [join(TS_ROOT, "reference", "score_reference.py"), "--pred-from", raw, "--truth", ann, "--pred", join(dir, "pred"), "-o", pyOut];
  if (blindMap) args.push("--blind-map", blindMap);
  const p = spawnSync(PYTHON, args, { encoding: "utf8", env: { ...process.env, CARTMASK_PY: PY_REPO, PYTHONIOENCODING: "utf-8" } });
  expect(p.status, p.stderr).toBe(0);
  const py = JSON.parse(readFileSync(join(pyOut, "python.json"), "utf8"));
  const plain = (v: unknown): unknown => JSON.parse(JSON.stringify(v));
  const tsRows = ts.rows.map((r) => ({
    scan: r.scan,
    annotator: r.annotator,
    prefilled: r.prefilled,
    status: r.status,
    qc_score: r.qc_score,
    um_per_px: r.um_per_px,
    metrics: r.metrics,
    crop: r.crop,
  }));
  expect(plain(tsRows)).toEqual(py.rows);
  expect(plain(ts.summary)).toEqual(py.summary);
  expect(plain(ts.geometry)).toEqual(py.geometry);
  expect(plain(ts.inter)).toEqual(py.inter);
  expect(ts.skipped).toEqual(py.skipped);
  expect(ts.markdown).toBe(py.markdown);
  const tsOut = join(dir, "ts");
  for (const f of ["per_scan.csv", "per_scan_long.csv", "summary.csv", "inter_annotator.csv", "summary.md"]) {
    const read = (d: string): string | null => (existsSync(join(d, f)) ? readFileSync(join(d, f), "utf8").replace(/\r\n/g, f.endsWith(".md") ? "\n" : "\r\n") : null);
    expect(read(tsOut), f).toBe(read(pyOut));
  }
}

describe.skipIf(!havePython)("parity with the Python scorer", () => {
  it("assisted export with revisions, ignore pixels and skips", () => {
    const dir = tmp();
    const raw = join(dir, "raw"), ann = join(dir, "ann");
    const [s, t] = makeHead({ dragAngleDeg: 30.0 });
    const x3p = join(raw, "synthetic.x3p");
    writeX3PTo(s, x3p);
    const r = writePrediction(join(dir, "pred"), x3p, 8.0, 0);
    const { rows, cols } = r.labels!;
    const truth = truthLabels(rows, cols, t);
    const shifted = labelGrid(rows, cols);
    for (let y = 0; y < rows; y++) for (let x = 2; x < cols; x++) shifted.data[y * cols + x] = truth.data[y * cols + x - 2];
    const ignored = { ...truth, data: truth.data.slice() };
    fill(ignored, 300, 360, 300, 360, IGNORE);
    const older = { ...r.labels!, data: r.labels!.data.slice() };
    const cut = { data: truth.data.slice(0, (rows - 5) * cols), rows: rows - 5, cols };
    const exports: [string, string, Grid<Uint8Array>, Record<string, unknown>][] = [
      ["alice", "synthetic", r.labels!, { prefilled: true }],
      ["bob", "synthetic", older, { timestamp: "2026-01-01T12:00:00Z" }],
      ["bob-rev", "synthetic", truth, { annotator: "Bob", timestamp: "2026-01-02T09:00:00+01:00" }],
      ["carol", "synthetic", shifted, {}],
      ["dora", "synthetic", ignored, {}],
      ["erin", "synthetic", truth, { done: false }],
      ["fred", "synthetic", truth, { packet: "0123abcd" }],
      ["gina", "ghost", truth, {}],
      ["hank", "synthetic", truth, { um_per_px: "abc" }],
      ["ivan", "synthetic", cut, {}],
    ];
    for (const [who, scan, img, extra] of exports) {
      const base = join(ann, who, `${scan}__labels`);
      writeLabels(`${base}.png`, img);
      writeFileSync(`${base}.json`, JSON.stringify({ scan, annotator: who, um_per_px: 8.0, done: true, ...extra }));
    }
    const ts = run(raw, ann, join(dir, "ts"));
    expect(ts.rows.map((x) => x.annotator)).toEqual(["alice", "Bob", "carol", "dora"]);
    expect(ts.skipped).toHaveLength(5);
    expect(ts.rows[3].metrics.primer_center_err_um).toBeNull();
    expectParity(raw, ann, dir, ts);
  });

  it("blinded packet with a private map", () => {
    const dir = tmp();
    const [s, t] = makeHead({ dragAngleDeg: 30.0, seed: 3 });
    const source = join(dir, "raw", "private-specimen.x3p");
    writeX3PTo(s, source);
    const priv = join(dir, "private.json");
    const packet = build([source], join(dir, "packet.html"), 800, 90, null, priv);
    const map = JSON.parse(readFileSync(priv, "utf8")) as BlindMap;
    const name = packet.scans[0].name as string;
    const rec = map.scans[name];
    const r = writePrediction(join(dir, "pred"), source, rec.um_per_px, 0);
    const truth = truthLabels(r.labels!.rows, r.labels!.cols, t);
    const ann = join(dir, "ann");
    for (const [rev, who, img] of [
      ["20260101", "A", r.labels!],
      ["20260102", "a", truth],
      ["20260103", "B", truth],
    ] as const) {
      const base = join(ann, who, `${name}__${rev}120000__labels`);
      writeLabels(`${base}.png`, img);
      writeFileSync(
        `${base}.json`,
        JSON.stringify({ scan: name, packet: packet.packet, annotator: who, done: true, prefilled: false, timestamp: `${rev.slice(0, 4)}-${rev.slice(4, 6)}-${rev.slice(6)}T12:00:00Z` }),
      );
    }
    const ts = run(join(dir, "raw"), ann, join(dir, "ts"), undefined, null, map);
    expect(ts.rows.map((x) => x.annotator)).toEqual(["a", "B"]);
    expectParity(join(dir, "raw"), ann, dir, ts, priv);
  });
});
