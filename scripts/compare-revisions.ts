/**
 * Before/after evidence for a change to the 3D masks, on every pinned real scan.
 *
 * Port of `scripts/compare_revisions.py`.
 *
 * `run` segments every scan in the baseline and expanded manifests and stores the label grid
 * (`<key>.npz`, readable with `np.load(...)["labels"]`), the result summary (`<key>.json`) and a
 * shaded rendering (`<key>.png`) per scan. Point `--code` at an exported older revision of this
 * repository (`git archive`) to produce the "before" run with identical inputs.
 *
 * `compare` reads two runs and writes, per scan, the area change of every class, the per-class
 * IoU between old and new masks and the QC flags; a summary by study; and side-by-side
 * before/after figures for every scan that changed materially.
 *
 * `synthetic` scores a cartmask revision on synthetic heads with known aperture-shear and slot
 * masks. `--make` writes the heads with the current generator, so an older revision (`--code`)
 * is scored on identical inputs.
 *
 *     tsx scripts/compare-revisions.ts run --code old_src --out outputs/revisions/before
 *     tsx scripts/compare-revisions.ts run --out outputs/revisions/after
 *     tsx scripts/compare-revisions.ts compare outputs/revisions/before outputs/revisions/after \
 *         --out samples/aperture-and-shear
 *     tsx scripts/compare-revisions.ts synthetic --make outputs/revisions/synthetic.npz
 *     tsx scripts/compare-revisions.ts synthetic --cases outputs/revisions/synthetic.npz --code old_src
 */
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import type { HeadOptions } from "../src/synthetic.js";
import { gridNpy, loadNpz, npyGrid, saveNpz, scalarNpy, type NpyArray } from "./lib/npz.js";
import { decodePng, encodePng } from "../src/image.js";
import { pyFloat, pyRound, pyStr, readCsv, writeCsv } from "./lib/pyfmt.js";

export const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const SELF = fileURLToPath(import.meta.url);
export const MANIFESTS = [join(ROOT, "docs/scan-manifest.csv"), join(ROOT, "docs/expanded-scan-manifest.csv")];
export const CLASSES: Record<number, string> = {
  1: "breech_face",
  2: "aperture_shear",
  3: "firing_pin_shear",
  4: "firing_pin_impression",
};
/** A scan "changed materially" when any class IoU drops below this ... */
export const MATERIAL_IOU = 0.98;
/** ... or a class appears / disappears with at least this much area. */
export const MIN_AREA_MM2 = 0.02;

type Summary = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
type U8Grid = { data: Uint8Array; rows: number; cols: number };

/** `[cohort, relative path, absolute path]` for every manifest row. */
export function scanList(dataRoot: string): [string, string, string][] {
  const out: [string, string, string][] = [];
  MANIFESTS.forEach((manifest, i) => {
    const cohort = i === 0 ? "original" : "expanded";
    for (const row of readCsv(manifest)) {
      const rel = row["file"]!.replace(/^data\//, "");
      out.push([cohort, rel, join(dataRoot, ...rel.split("/"))]);
    }
  });
  return out;
}

/** Study name from `<calibre>__<study>__<specimen>`. */
export function studyOf(rel: string): string {
  const parts = basename(rel).replace(/\.[^.]*$/, "").split("__");
  return parts.length > 2 ? parts[1]! : "unknown";
}

/** Run key: the relative path with `/` -> `__` and no `.x3p`. */
export function runKey(rel: string): string {
  return rel.replace(/\//g, "__").replace(/\.x3p$/, "");
}

/** The cartmask API of a checkout: `<code>/src/index.ts` (an older revision) or this repository. */
async function cartmaskAt(code?: string): Promise<typeof import("../src/index.js")> {
  if (!code) return import("../src/index.js");
  const entry = ["src/index.ts", "src/index.js", "dist/src/index.js"].map((p) => join(code, p)).find(existsSync);
  if (!entry) throw new Error(`no src/index.ts under ${code}`);
  return import(pathToFileURL(entry).href);
}

/** `render(res)` of a checkout's visualize module, or of this repository's when the checkout has none. */
async function renderAt(code?: string): Promise<((res: any) => U8Grid) | null> {
  const candidates = [code && join(code, "src/visualize.ts"), join(ROOT, "src/visualize.ts")].filter(Boolean) as string[];
  for (const path of candidates) {
    if (!existsSync(path)) continue;
    const mod = await import(pathToFileURL(path).href);
    const { renderGray } = await import("./lib/figures.js");
    return (res) => renderGray(mod, res);
  }
  return null;
}

// --------------------------------------------------------------------------- run

async function runJobs(jobs: [string, string, string][], out: string, code: string | undefined, log: (l: string) => void, offset = 0, total = jobs.length): Promise<void> {
  const cm = await cartmaskAt(code);
  const render = await renderAt(code);
  if (!render) log("src/visualize.ts not found: skipping the shaded renderings (<key>.png)");
  for (let i = 0; i < jobs.length; i++) {
    const [cohort, rel, src] = jobs[i]!;
    const key = runKey(rel);
    let msg: string;
    try {
      const res = cm.segment(cm.readX3P(src));
      const summary: Summary = { ...res.summary(), cohort, file: rel, study: studyOf(rel) };
      writeFileSync(join(out, `${key}.json`), JSON.stringify(summary, null, 1));
      if (!res.labels) throw new Error("segment returned no labels");
      saveNpz(join(out, `${key}.npz`), { labels: gridNpy(res.labels) });
      if (res.z && render) writeFileSync(join(out, `${key}.png`), encodePng(gridImage(render(res))));
      msg = `${rel} ${res.status} review=${pyStr(res.needsReview)}`;
    } catch (exc) {
      const e = exc as Error;
      writeFileSync(join(out, `${key}.json`), JSON.stringify({ cohort, file: rel, status: "error", error: `${e.name}: ${e.message}` }));
      msg = `${rel} error ${e.message}`;
    }
    log(`${offset + i + 1}/${total} ${msg}`);
  }
}

/** Segment every manifest scan into `out`; `workers > 1` splits the list over child processes. */
export async function cmdRun(o: { dataRoot: string; out: string; code?: string; workers: number; only?: string[]; shard?: string }, log = console.log): Promise<void> {
  mkdirSync(o.out, { recursive: true });
  let jobs = scanList(o.dataRoot).filter(([, rel]) => !o.only?.length || o.only.some((s) => rel.includes(s)));
  if (o.shard) {
    const [k, n] = o.shard.split("/").map(Number) as [number, number];
    jobs = jobs.filter((_, i) => i % n === k);
    return runJobs(jobs, o.out, o.code, log);
  }
  if (o.workers <= 1 || jobs.length < 2) return runJobs(jobs, o.out, o.code, log);
  const n = Math.min(o.workers, jobs.length);
  const base = ["run", "--data-root", o.dataRoot, "--out", o.out, ...(o.code ? ["--code", o.code] : [])];
  if (o.only?.length) base.push("--only", ...o.only);
  await Promise.all(
    Array.from({ length: n }, (_, k) =>
      new Promise<void>((ok, fail) => {
        const child = spawn(process.execPath, [...process.execArgv, SELF, ...base, "--shard", `${k}/${n}`], { stdio: "inherit" });
        child.on("exit", (code) => (code === 0 ? ok() : fail(new Error(`worker ${k} exited with ${code}`))));
      }),
    ),
  );
}

// --------------------------------------------------------------------------- compare

function gridImage(g: U8Grid) {
  return { width: g.cols, height: g.rows, channels: 1 as const, data: g.data };
}

function count(m: Uint8Array): number {
  let n = 0;
  for (let i = 0; i < m.length; i++) n += m[i]!;
  return n;
}

function classMask(lab: U8Grid, v: number): Uint8Array {
  const m = new Uint8Array(lab.data.length);
  for (let i = 0; i < m.length; i++) m[i] = lab.data[i] === v ? 1 : 0;
  return m;
}

/** IoU of two 0/1 masks; 1 when both are empty. */
export function iou(a: Uint8Array, b: Uint8Array): number {
  let inter = 0, union = 0;
  for (let i = 0; i < a.length; i++) {
    inter += a[i]! & b[i]!;
    union += a[i]! | b[i]!;
  }
  return union === 0 ? 1.0 : inter / union;
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m]! : (s[m - 1]! + s[m]!) / 2;
}

const mean = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length;

/** `json.dumps(obj, sort_keys=True)` with Python separators and float formatting. */
function pyDumpsSorted(v: unknown): string {
  if (v === null || v === undefined) return "null";
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "number") return pyFloat(v);
  if (typeof v === "string") return JSON.stringify(v);
  if (Array.isArray(v)) return "[" + v.map(pyDumpsSorted).join(", ") + "]";
  const keys = Object.keys(v as object).sort();
  return "{" + keys.map((k) => `${JSON.stringify(k)}: ${pyDumpsSorted((v as Record<string, unknown>)[k])}`).join(", ") + "}";
}

function loadLabels(dir: string, key: string): U8Grid {
  return npyGrid<Uint8Array>(loadNpz(join(dir, `${key}.npz`))["labels"]!);
}

/** Per-scan and per-study comparison of two runs; figures for material changes (or all). */
export async function cmdCompare(o: { before: string; after: string; out: string; allFigures?: boolean }, log = console.log): Promise<void> {
  mkdirSync(join(o.out, "figures"), { recursive: true });
  const rows: Summary[] = [];
  let figure: typeof import("./lib/figures.js").revisionFigure | null | undefined;
  for (const name of readdirSync(o.after).filter((f) => f.endsWith(".json")).sort()) {
    const key = name.slice(0, -5);
    const sb: Summary = JSON.parse(readFileSync(join(o.after, name), "utf8"));
    const saPath = join(o.before, name);
    if (!existsSync(saPath) || sb.status === "error") continue;
    const sa: Summary = JSON.parse(readFileSync(saPath, "utf8"));
    if (sa.status === "error") continue;
    const la = loadLabels(o.before, key);
    const lb = loadLabels(o.after, key);
    const pxMm2 = (sb.um_per_px / 1000.0) ** 2;
    const row: Summary = { file: sb.file, cohort: sb.cohort, study: sb.study, status: sb.status };
    let material = false;
    for (const [v, cls] of Object.entries(CLASSES)) {
      const a = classMask(la, Number(v)), b = classMask(lb, Number(v));
      const na = count(a), nb = count(b);
      const j = iou(a, b);
      row[`${cls}_before_mm2`] = pyRound(na * pxMm2, 4);
      row[`${cls}_after_mm2`] = pyRound(nb * pxMm2, 4);
      row[`${cls}_delta_mm2`] = pyRound((nb - na) * pxMm2, 4);
      row[`${cls}_iou`] = pyRound(j, 4);
      if (j < MATERIAL_IOU && Math.max(na, nb) * pxMm2 >= MIN_AREA_MM2) material = true;
    }
    for (const k of ["aperture_detected", "needs_review", "qc_score"]) {
      row[`${k}_before`] = sa[k] ?? null;
      row[`${k}_after`] = sb[k] ?? null;
    }
    const wa: string[] = sa.warnings ?? [], wb: string[] = sb.warnings ?? [];
    row.review_reasons_before = (sa.review_reasons ?? []).join(";");
    row.review_reasons_after = (sb.review_reasons ?? []).join(";");
    row.aperture_shape_after = sb.aperture_shape ?? "circle";
    row.shear_evidence_after = pyDumpsSorted(sb.shear_evidence ?? {});
    row.warnings_added = wb.filter((w) => !wa.includes(w)).join("; ");
    row.warnings_removed = wa.filter((w) => !wb.includes(w)).join("; ");
    row.material_change = material;
    rows.push(row);
    if ((material || o.allFigures) && existsSync(join(o.after, `${key}.png`))) {
      if (figure === undefined) figure = await loadFigures(log, (m) => m.revisionFigure);
      const img = decodePng(readFileSync(join(o.after, `${key}.png`)));
      const base = { data: img.channels === 1 ? img.data : img.data.filter((_, i) => i % 3 === 0), rows: img.height, cols: img.width };
      if (figure) await figure(la, lb, base, sa, sb, join(o.out, "figures", `${key}.jpg`));
    }
  }
  if (!rows.length) throw new Error("no comparable scans");
  const cols = Object.keys(rows[0]!);
  writeCsv(join(o.out, "per-scan.csv"), cols, rows.map((r) => Object.fromEntries(cols.map((c) => [c, pyStr(r[c])]))));

  const groups = new Map<string, Summary[]>();
  for (const r of rows) {
    const k = JSON.stringify([r.cohort, r.study]);
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k)!.push(r);
  }
  const keys = [...groups.keys()].sort((x, y) => {
    const [a1, a2] = JSON.parse(x), [b1, b2] = JSON.parse(y);
    return a1 < b1 ? -1 : a1 > b1 ? 1 : a2 < b2 ? -1 : a2 > b2 ? 1 : 0;
  });
  const summary: Summary[] = [];
  const intCols = new Set(["scans", "material_change", "shear_before", "shear_after", "aperture_ring_before", "aperture_ring_after", "rectangular_after", "review_before", "review_after"]);
  for (const k of keys) {
    const [cohort, study] = JSON.parse(k);
    const rs = groups.get(k)!;
    const n = (f: (r: Summary) => boolean) => rs.filter(f).length;
    const s: Summary = {
      cohort, study, scans: rs.length,
      material_change: n((r) => r.material_change),
      shear_before: n((r) => r.aperture_shear_before_mm2 > 0),
      shear_after: n((r) => r.aperture_shear_after_mm2 > 0),
      aperture_ring_before: n((r) => !!r.aperture_detected_before),
      aperture_ring_after: n((r) => !!r.aperture_detected_after),
      rectangular_after: n((r) => r.aperture_shape_after === "rectangle"),
      review_before: n((r) => !!r.needs_review_before),
      review_after: n((r) => !!r.needs_review_after),
    };
    for (const cls of Object.values(CLASSES)) {
      s[`${cls}_median_iou`] = pyRound(median(rs.map((r) => r[`${cls}_iou`])), 4);
      s[`${cls}_min_iou`] = pyRound(Math.min(...rs.map((r) => r[`${cls}_iou`])), 4);
      s[`${cls}_mean_delta_mm2`] = pyRound(mean(rs.map((r) => r[`${cls}_delta_mm2`])), 4);
    }
    summary.push(s);
  }
  const scols = Object.keys(summary[0]!);
  writeCsv(join(o.out, "by-study.csv"), scols, summary.map((r) => Object.fromEntries(scols.map((c) => [c, pyStr(r[c], intCols.has(c))]))));
  log(`${rows.length} scans compared, ${rows.filter((r) => r.material_change).length} changed materially`);
  for (const s of summary) log(JSON.stringify(s));
}

/** The figure module, or null (with a message) when `src/visualize.ts` is not available. */
async function loadFigures<T>(log: (l: string) => void, pick: (m: typeof import("./lib/figures.js")) => T): Promise<T | null> {
  try {
    return pick(await import("./lib/figures.js"));
  } catch (exc) {
    log(`figures skipped (${(exc as Error).message})`);
    return null;
  }
}

// --------------------------------------------------------------------------- synthetic

type Case = HeadOptions & { _params?: Record<string, unknown> };
export const SYNTHETIC_CASES: Record<string, Case> = {
  "round, drag": {},
  "round, no drag": { dragAngleDeg: null, seed: 1 },
  "striated shear, drag": { shearUm: 3.0 },
  "weak striated shear, drag": { shearUm: 1.5, seed: 9 },
  "striated shear, no drag": { shearUm: 3.0, dragAngleDeg: null, seed: 5 },
  "isotropic rough band, drag": { roughBandUm: 3.0 },
  "isotropic rough band, no drag": { roughBandUm: 3.0, dragAngleDeg: null, seed: 6 },
  "slot 0 deg, drag": { rectAperture: [1.3, 3.0], dragAngleDeg: 0.0 },
  "slot 0 deg, no drag": { rectAperture: [1.3, 3.0], dragAngleDeg: null, seed: 3 },
  "slot 45 deg, drag": { rectAperture: [1.3, 3.0], rectAngleDeg: 45.0, dragAngleDeg: 45.0, seed: 4 },
  "edge step only": { flowbackUm: 0.0, apertureStepUm: 6.0, dragAngleDeg: null, seed: 7 },
  "edge step only, aperture_edge=True": { flowbackUm: 0.0, apertureStepUm: 6.0, dragAngleDeg: null, seed: 7, _params: { apertureEdge: true } },
  "no ring, no step": { flowbackUm: 0.0, dragAngleDeg: null, seed: 8 },
  "no ring, no step, aperture_edge=True": { flowbackUm: 0.0, dragAngleDeg: null, seed: 8, _params: { apertureEdge: true } },
};

/** `--make`: write the heads; otherwise score `code` (or this repository) on heads from `cases`. */
export async function cmdSynthetic(o: { make?: string; cases?: string; code?: string; out?: string }, log = console.log): Promise<Summary[] | void> {
  const cm = await cartmaskAt(o.code);
  const names = Object.keys(SYNTHETIC_CASES);
  if (o.make) {
    const arrays: Record<string, NpyArray> = {};
    names.forEach((name, i) => {
      const { _params, ...kw } = SYNTHETIC_CASES[name]!;
      void _params;
      const [surf, t] = cm.makeHead(kw);
      arrays[`z${i}`] = { dtype: "<f8", shape: [surf.z.rows, surf.z.cols], data: surf.z.data };
      arrays[`shear${i}`] = gridNpy(t.shearMask, "|b1");
      arrays[`rect${i}`] = gridNpy(t.rectMask, "|b1");
      arrays[`ap${i}`] = scalarNpy(t.apertureRadiusPx);
      arrays[`dx${i}`] = scalarNpy(surf.dx);
    });
    mkdirSync(dirname(resolve(o.make)), { recursive: true });
    saveNpz(o.make, arrays);
    log(`wrote ${names.length} heads to ${o.make}`);
    return;
  }
  if (!o.cases) throw new Error("synthetic: give --make or --cases");
  const data = loadNpz(o.cases);
  const rows: Summary[] = [];
  names.forEach((name, i) => {
    const dx = data[`dx${i}`]!.data[0]!;
    const extra = SYNTHETIC_CASES[name]!._params ?? {};
    const known = Object.fromEntries(Object.entries(extra).filter(([k]) => k in cm.DEFAULT_PARAMS)); // older revisions lack the switch
    const z = npyGrid<Float64Array>(data[`z${i}`]!);
    const res = cm.segment(cm.makeSurface(z, dx, dx, name), known);
    if (!res.labels) throw new Error(`${name}: no labels`);
    const lab = res.labels.data;
    const tm = data[`shear${i}`]!.data as Uint8Array, rm = data[`rect${i}`]!.data as Uint8Array;
    let sh = 0, shT = 0, nT = 0, nR = 0, bfR = 0;
    for (let p = 0; p < lab.length; p++) {
      const s = lab[p] === 2 ? 1 : 0;
      sh += s;
      shT += s & tm[p]!;
      nT += tm[p]!;
      nR += rm[p]!;
      bfR += lab[p] === 1 && rm[p] ? 1 : 0;
    }
    const ap = data[`ap${i}`]!.data[0]!;
    rows.push({
      case: name,
      shear_px: sh,
      shear_recall: nT ? pyRound(shT / nT, 3) : null,
      shear_precision: sh && nT ? pyRound(shT / sh, 3) : null,
      breech_face_in_slot: nR ? pyRound(bfR / nR, 3) : null,
      aperture_shape: (res as { apertureShape?: string }).apertureShape ?? "circle",
      aperture_detected: res.apertureDetected,
      aperture_radius_error: res.apertureRadiusPx && !nR ? pyRound(res.apertureRadiusPx / ap - 1, 3) : null,
    });
    log(JSON.stringify(rows[rows.length - 1]));
  });
  if (o.out) {
    mkdirSync(dirname(resolve(o.out)), { recursive: true });
    const { pyJson } = await import("./lib/pyfmt.js");
    writeFileSync(o.out, pyJson(rows, 1, (p) => p === "[].shear_px") + "\n", "utf8");
  }
  return rows;
}

// --------------------------------------------------------------------------- CLI

const USAGE = `usage: compare-revisions {run,compare,synthetic} ...

  run        [--data-root DIR] --out DIR [--code DIR] [--workers N] [--only SUBSTR ...]
  compare    BEFORE AFTER --out DIR [--all-figures]
  synthetic  [--make FILE.npz] [--cases FILE.npz] [--code DIR] [--out FILE.json]`;

export async function main(argv: string[] = process.argv.slice(2)): Promise<number> {
  const [cmd, ...rest] = argv;
  try {
    if (cmd === "run") {
      const { values, positionals } = parseArgs({
        args: rest,
        options: {
          "data-root": { type: "string", default: join(ROOT, "data") },
          out: { type: "string" },
          code: { type: "string" },
          workers: { type: "string", default: "6" },
          only: { type: "string", multiple: true },
          shard: { type: "string" },
        },
        allowPositionals: true,
      });
      if (!values.out) throw new Error("run: --out is required");
      // `--only a b c` (Python nargs="*"): positionals after --only are more substrings
      const only = [...(values.only ?? []), ...(values.only ? positionals : [])];
      await cmdRun({ dataRoot: resolve(values["data-root"]!), out: resolve(values.out), code: values.code && resolve(values.code), workers: Number(values.workers), only, shard: values.shard });
    } else if (cmd === "compare") {
      const { values, positionals } = parseArgs({
        args: rest,
        options: { out: { type: "string" }, "all-figures": { type: "boolean", default: false } },
        allowPositionals: true,
      });
      if (positionals.length !== 2 || !values.out) throw new Error("compare: give BEFORE AFTER --out DIR");
      await cmdCompare({ before: positionals[0]!, after: positionals[1]!, out: values.out, allFigures: values["all-figures"] });
    } else if (cmd === "synthetic") {
      const { values } = parseArgs({
        args: rest,
        options: { make: { type: "string" }, cases: { type: "string" }, code: { type: "string" }, out: { type: "string" } },
      });
      await cmdSynthetic({ ...values, code: values.code && resolve(values.code) });
    } else {
      console.log(USAGE);
      return cmd === "-h" || cmd === "--help" ? 0 : 2;
    }
  } catch (exc) {
    console.error(`compare-revisions: ${(exc as Error).message}`);
    return 1;
  }
  return 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().then((code) => process.exit(code));
}
