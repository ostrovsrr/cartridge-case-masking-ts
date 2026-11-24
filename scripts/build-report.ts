/**
 * Build a self-contained interactive HTML report (report/index.html).
 *
 * Port of `scripts/build_report.py`; the embedded data payload has the same format:
 *
 *     tsx scripts/build-report.ts                       # demo scans, saved evaluation
 *     tsx scripts/build-report.ts --data data/raw --matching samples/matching
 *
 * Every image is embedded as a data: URI, so the single HTML file can be opened anywhere (or
 * attached to an email) without the rest of the repository.
 *
 * The matching summary (AUC, bootstrap intervals) is read from `<matching>/summary.json`, which
 * `python python/compare.py` writes next to `pairs.csv`. When it is missing and Python is
 * available, the script runs `python python/compare.py <pairs.csv> --summarize-only` itself.
 */
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, extname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import { zlibSync } from "fflate";
import { crop, encodeJpeg, newImage, paste, resize, type Image } from "../src/image.js";
import { LABEL_NAMES, segment } from "../src/segment.js";
import { pyRoundN } from "../src/steps/np.js";
import { COLOURS, cropBox, render } from "../src/visualize.js";
import { readX3P } from "../src/x3p.js";
import { ROOT } from "./fetch-nbtrd.js";
import { readCsv } from "./lib/pyfmt.js";

export const SIZE = 520;
export const AUDIT_FILES = [join(ROOT, "samples/validation/current-3d.csv"), join(ROOT, "samples/validation/expanded-3d.csv")];
const MASK_KINDS = ["breech_face", "primer_disc", "full_scan"];

const stem = (p: string): string => basename(p, extname(p));

/** Deduplicate evaluated scans by source hash, independently of the viewer inputs. */
export function auditEntries(paths: string[]): Record<string, unknown>[] {
  const entries = new Map<string, Record<string, unknown>>();
  for (const path of paths) {
    for (const row of readCsv(path)) {
      const name = stem(row["file"]!);
      const parts = name.split("__");
      if (entries.has(row["source_sha256"]!)) continue;
      entries.set(row["source_sha256"]!, {
        name,
        study: parts.length > 1 ? parts[1] : "Other",
        status: row["status"],
        review: row["needs_review"] === "True",
        qc: row["qc_score"],
        coverage: row["crater_measured_fraction"],
        circle: row["impression_circle_iou"],
        reasons: row["review_reasons"] ?? "",
      });
    }
  }
  const cmp = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0);
  return [...entries.values()].sort((a, b) => cmp(a.study as string, b.study as string) || cmp(a.name as string, b.name as string));
}

// ---------------------------------------------------------------------------- PNG (RGBA)

const CRC = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function pngChunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length);
  const v = new DataView(out.buffer);
  v.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  let c = 0xffffffff;
  for (let i = 4; i < 8 + data.length; i++) c = CRC[(c ^ out[i]!) & 0xff]! ^ (c >>> 8);
  v.setUint32(8 + data.length, (c ^ 0xffffffff) >>> 0);
  return out;
}

/** RGBA PNG (colour type 6, filter 0, zlib level 9) of a w x h pixel buffer. */
export function encodePngRgba(rgba: Uint8Array, w: number, h: number): Uint8Array {
  const raw = new Uint8Array(h * (w * 4 + 1));
  for (let y = 0; y < h; y++) raw.set(rgba.subarray(y * w * 4, (y + 1) * w * 4), y * (w * 4 + 1) + 1);
  const ihdr = new Uint8Array(13);
  new DataView(ihdr.buffer).setUint32(0, w);
  new DataView(ihdr.buffer).setUint32(4, h);
  ihdr.set([8, 6, 0, 0, 0], 8);
  const parts = [Uint8Array.of(0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a), pngChunk("IHDR", ihdr), pngChunk("IDAT", zlibSync(raw, { level: 9 })), pngChunk("IEND", new Uint8Array(0))];
  const out = new Uint8Array(parts.reduce((a, p) => a + p.length, 0));
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

const uri = (bytes: Uint8Array, mime: string): string => `data:${mime};base64,` + Buffer.from(bytes).toString("base64");

// ---------------------------------------------------------------------------- scans

/** Viewer entry of one scan: summary fields, base image and one RGBA layer per label. */
export function scanEntry(path: string): Record<string, unknown> {
  let res;
  try {
    res = segment(readX3P(path));
  } catch (exc) {
    const msg = (exc as Error).message ?? String(exc);
    console.error(`  ${basename(path)}: ${msg}`);
    return { name: stem(path), status: "failed", review: true, warnings: [msg] };
  }
  const s = res.summary();
  const entry: Record<string, unknown> = {
    name: res.name, status: res.status, warnings: res.warnings, qc: s.qc_score,
    review: s.needs_review, qc_factors: s.qc_factors ?? {}, qc_observations: s.qc_observations ?? {},
    primer_d: s.primer_diameter_mm ?? null, pin_d: s.fp_core_diameter_mm ?? null, depth: s.fp_max_depth_um ?? null,
    aperture_d: s.aperture_diameter_mm ?? null, aperture_detected: s.aperture_detected, drag: res.dragDetected,
    drag_angle: s.drag_angle_deg ?? null, drag_len: s.drag_length_mm ?? null, areas: s.areas_mm2 ?? {},
  };
  if (res.status === "skipped" || res.status === "failed" || res.primerCenterPx == null || !res.labels) return entry;
  const [y0, y1, x0, x1] = cropBox(res);
  const side = Math.max(y1 - y0, x1 - x0);
  const scale = SIZE / side;
  const base = crop(render(res), y0, y1, x0, x1);
  const canvas: Image = newImage(side, side, 3, 0);
  paste(canvas, base, 0, 0);
  entry.base = uri(encodeJpeg(resize(canvas, SIZE, SIZE, "area"), 82), "image/jpeg");
  const layers: Record<string, string> = {};
  const L = res.labels;
  for (const [vs, name] of Object.entries(LABEL_NAMES)) {
    const v = Number(vs);
    const m: Image = newImage(side, side, 1, 0);
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) if (L.data[y * L.cols + x] === v) m.data[(y - y0) * side + (x - x0)] = 255;
    const mm = resize(m, SIZE, SIZE, "nearest");
    const c = COLOURS[v]!; // BGR
    const rgba = new Uint8Array(SIZE * SIZE * 4);
    for (let i = 0; i < SIZE * SIZE; i++) {
      rgba[i * 4] = c[2];
      rgba[i * 4 + 1] = c[1];
      rgba[i * 4 + 2] = c[0];
      rgba[i * 4 + 3] = mm.data[i]!;
    }
    layers[name] = uri(encodePngRgba(rgba, SIZE, SIZE), "image/png");
  }
  entry.layers = layers;
  const toCrop = (x: number, y: number) => [pyRoundN((x - x0) * scale, 1), pyRoundN((y - y0) * scale, 1)];
  entry.centre = toCrop(...res.primerCenterPx);
  if (res.fpCoreCenterPx != null) entry.pin = [...toCrop(...res.fpCoreCenterPx), pyRoundN(res.fpCoreRadiusPx! * scale, 1)];
  if (res.dragDetected && res.dragTipPx) entry.tip = toCrop(...res.dragTipPx);
  entry.mm_per_px = res.umPerPx / 1000 / scale;
  console.log(`  ${res.name}: ${res.status}, qc=${entry.qc}`);
  return entry;
}

// ---------------------------------------------------------------------------- matching

/** Rows of a pairs.csv with numeric columns parsed (Python `compare.read_pairs`). */
export function readPairs(path: string): [Record<string, unknown>[], string[]] {
  const rows = readCsv(path).map((r) => {
    const out: Record<string, unknown> = { ...r, match: parseInt(r["match"]!, 10) };
    for (const [k, v] of Object.entries(r)) {
      if (k.startsWith("ccf_") || k === "mask_iou") {
        if (v === "") delete out[k];
        else out[k] = parseFloat(v);
      }
    }
    return out;
  });
  const cols = new Set(rows.flatMap((r) => Object.keys(r).filter((c) => c.startsWith("ccf_")).map((c) => c.slice(4))));
  const kinds = [...MASK_KINDS.filter((k) => cols.has(k)), ...[...cols].filter((k) => !MASK_KINDS.includes(k)).sort()];
  return [rows, kinds];
}

/** `summary.json` beside pairs.csv, or recomputed by `python python/compare.py --summarize-only`. */
function matchingSummary(dir: string, reps: number, seed: number): Record<string, unknown> {
  const saved = join(dir, "summary.json");
  if (existsSync(saved)) return JSON.parse(readFileSync(saved, "utf8"));
  const script = join(ROOT, "python/compare.py");
  if (existsSync(script)) {
    const tmp = mkdtempSync(join(tmpdir(), "cartmask-summary-"));
    try {
      const py = process.env.PYTHON ?? "python";
      const r = spawnSync(py, [script, join(dir, "pairs.csv"), "--summarize-only", "-o", tmp, "--bootstrap-reps", String(reps), "--seed", String(seed)], { cwd: join(ROOT, "python"), stdio: ["ignore", "ignore", "inherit"] });
      if (r.status === 0 && existsSync(join(tmp, "summary.json"))) return JSON.parse(readFileSync(join(tmp, "summary.json"), "utf8"));
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  }
  console.error(`  no ${saved} (run python python/compare.py --summarize-only -o ${dir}); matching statistics omitted`);
  return {};
}

// ---------------------------------------------------------------------------- main

export function main(argv = process.argv.slice(2)): number {
  const { values, positionals } = parseArgs({
    args: argv,
    options: {
      data: { type: "string", multiple: true },
      audit: { type: "string", multiple: true },
      "no-audit": { type: "boolean", default: false },
      matching: { type: "string", default: "samples/matching" },
      failures: { type: "string", default: join(ROOT, "samples/failures/failure_gallery.jpg") },
      out: { type: "string", default: "report/index.html" },
      template: { type: "string", default: join(ROOT, "scripts/report_template.html") },
      fragment: { type: "boolean", default: false },
      "bootstrap-reps": { type: "string", default: "2000" },
      seed: { type: "string", default: "0" },
      help: { type: "boolean", short: "h", default: false },
    },
    allowPositionals: true,
  });
  if (values.help) {
    console.log(
      "usage: build-report [--data DIR_OR_X3P ...] [--audit CSV ...] [--no-audit] [--matching DIR] [--failures JPG]\n" +
        "                    [--out HTML] [--template HTML] [--fragment] [--bootstrap-reps N] [--seed N]",
    );
    return 0;
  }
  // `--data a b` (Python nargs="+"): bare positionals are extra data inputs
  const data = [...(values.data ?? []), ...positionals];
  const inputs = data.length ? data : ["data/raw", "data/glock"];
  const sources: string[] = [];
  for (const p of inputs) {
    if (extname(p).toLowerCase() === ".x3p") sources.push(p);
    else if (existsSync(p) && statSync(p).isDirectory()) sources.push(...readdirSync(p).filter((f) => f.endsWith(".x3p")).sort().map((f) => join(p, f)));
  }
  const scans = [...new Set(sources)].map(scanEntry);
  let matching: Record<string, unknown> = { pairs: [], summary: {} };
  const pairs = join(values.matching!, "pairs.csv");
  if (existsSync(pairs)) {
    const [rows, kinds] = readPairs(pairs);
    if (kinds.length) matching = { pairs: rows, summary: matchingSummary(values.matching!, Number(values["bootstrap-reps"]), Number(values.seed)) };
  }
  let html = readFileSync(values.template!, "utf8");
  let failures: Record<string, string> | null = null;
  if (values.failures && existsSync(values.failures)) {
    failures = {
      image: uri(readFileSync(values.failures), "image/jpeg"),
      note: "Left of each pair: commit 9bce6ec; right: current output. Single panels are current outputs. Full-size images and captions: samples/failures/.",
    };
  }
  const audit = values["no-audit"] ? [] : auditEntries(values.audit ?? AUDIT_FILES);
  const payload = JSON.stringify({ scans, audit, matching, failures }).replace(/</g, "\\u003c");
  html = html.replace("/*__DATA__*/null", () => payload);
  if (!values.fragment) {
    const nl = "\n";
    const head = "<!doctype html>" + nl + '<html lang="en"><head><meta charset="utf-8">' + nl +
      '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">' + nl;
    const baseCss = "<style>" + nl + "body{margin:0}img{max-width:100%}[hidden]{display:none!important}";
    html = head + html.replace("<style>", baseCss) + nl + "</html>" + nl;
  }
  mkdirSync(dirname(resolve(values.out!)), { recursive: true });
  writeFileSync(values.out!, html, "utf8");
  console.log(`wrote ${values.out} (${(statSync(values.out!).size / 1e6).toFixed(1)} MB)`);
  return 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  process.exit(main());
}

