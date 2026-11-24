#!/usr/bin/env node
/**
 * Batch command line interface (port of `cartmask/cli.py` and `__main__.py`).
 *
 *     cartmask data/raw -o outputs
 *     cartmask data/raw/some_scan.x3p -o outputs
 *     cartmask data/raw -o outputs -v
 *
 * Progress goes to stdout: one line per scan by default, per-file details and stack traces with
 * `-v`, only failures with `-q`.
 */
import { mkdirSync, readdirSync, realpathSync, statSync, writeFileSync } from "node:fs";
import { basename, extname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import { exportX3P } from "./export.js";
import { fromGrid, imwrite, type Image } from "./image.js";
import { csvRow, csvValue, pyFloatRepr, pyJsonDumps } from "./pyformat.js";
import { DEFAULT_PARAMS, LABEL_NAMES, makeParams, segment, type Params } from "./segment.js";
import { fmtF, pyRoundN } from "./steps/np.js";
import { canonical, contactSheet, cropBox, cropTo, overlay, render, setBriefColours, sideBySide } from "./visualize.js";
import { readX3P } from "./x3p.js";

export { imwrite } from "./image.js";

/** Log levels of the `cartmask` logger (Python `logging`). */
const DEBUG = 10, INFO = 20, WARNING = 30, ERROR = 40;
let level = INFO;
const log = {
  debug: (msg: string) => { if (level <= DEBUG) process.stdout.write(msg + "\n"); },
  info: (msg: string) => { if (level <= INFO) process.stdout.write(msg + "\n"); },
  error: (msg: string) => { if (level <= ERROR) process.stdout.write(msg + "\n"); },
};

/** Usage error (argparse `ap.error`): message on stderr, exit status 2. */
export class UsageError extends Error {
  readonly exitCode = 2;
}

const stemOf = (p: string): string => {
  const b = basename(p);
  return b.slice(0, b.length - extname(b).length);
};

/** Result of {@link processScan}: the `masks.json` dict and the primer close-ups for the contact sheet. */
export interface Processed {
  summary: Record<string, unknown>;
  crop: [Image, Image];
}

/**
 * Segment one scan and write all of its per-scan outputs.
 *
 * Files go to `outDir/<scan>/` plus a side-by-side figure in `outDir` itself; with `x3pMasks`
 * also `<scan>__masked.x3p` (original scan + embedded mask).
 */
export function processScan(path: string, outDir: string, params: Params, x3pMasks = false): Processed {
  const t0 = performance.now();
  const surf = readX3P(path);
  const res = segment(surf, params);
  const labels = res.labels;
  if (labels == null) throw new Error(`${basename(path)}: segmentation returned no label image`);
  const d = join(outDir, res.name);
  mkdirSync(d, { recursive: true });
  const base = render(res);
  const ov = overlay(res, base);
  imwrite(join(d, "original.png"), base);
  imwrite(join(d, "overlay.png"), ov);
  imwrite(join(d, "overlay_canonical.png"), canonical(res, ov));
  imwrite(join(d, "labels.png"), fromGrid(labels)); // 0=background, 1..4 see masks.json
  for (const [k, name] of Object.entries(LABEL_NAMES)) {
    const v = Number(k);
    const m = new Uint8Array(labels.data.length);
    for (let i = 0; i < m.length; i++) m[i] = labels.data[i] === v ? 255 : 0;
    imwrite(join(d, `mask_${v}_${name}.png`), { width: labels.cols, height: labels.rows, channels: 1, data: m });
  }
  imwrite(join(outDir, `${res.name}__side_by_side.png`), sideBySide(res));
  if (x3pMasks && res.status !== "skipped") exportX3P(path, res, join(d, `${res.name}__masked.x3p`));
  const summary = res.summary();
  summary.source = path;
  summary.seconds = pyRoundN((performance.now() - t0) / 1000, 2);
  summary.label_values = Object.fromEntries(Object.entries(LABEL_NAMES).map(([k, v]) => [String(k), v]));
  writeFileSync(join(d, "masks.json"), pyJsonDumps(summary, 2));
  log.debug(`wrote ${d}`);
  const box = cropBox(res);
  return { summary, crop: [cropTo(base, box), cropTo(ov, box)] };
}

const USAGE = `usage: cartmask [-h] [-o OUT] [--um-per-px UM_PER_PX] [--x3p-masks] [--brief-colours] [-v | -q]
                inputs [inputs ...]`;

const HELP = `${USAGE}

Automatically mask cartridge-case head scans (X3P).

positional arguments:
  inputs                X3P files or directories containing them

options:
  -h, --help            show this help message and exit
  -o OUT, --out OUT     output directory
  --um-per-px UM_PER_PX
                        working resolution
  --x3p-masks           also write <scan>__masked.x3p with the mask embedded (x3ptools format)
  --brief-colours       draw the round strike purple and the drag channel light blue, as in the
                        brief's figures (display only; label IDs are unchanged)
  -v, --verbose         also log per-file details and tracebacks
  -q, --quiet           only log failures
`;

function usageError(msg: string): never {
  throw new UsageError(`${USAGE}\ncartmask: error: ${msg}`);
}

/** `.x3p` files of a directory, sorted like Python's `sorted(Path.glob("*.x3p"))`. */
function listX3P(dir: string): string[] {
  const win = process.platform === "win32";
  const files = readdirSync(dir)
    .filter((f) => (win ? f.toLowerCase() : f).endsWith(".x3p"))
    .map((f) => join(dir, f));
  const key = (p: string) => (win ? p.toLowerCase() : p);
  return files.sort((a, b) => (key(a) < key(b) ? -1 : key(a) > key(b) ? 1 : 0));
}

/**
 * Entry point of `cartmask`.
 *
 * @param argv Command-line arguments without the program name; `process.argv.slice(2)` when omitted.
 * @throws UsageError for invalid arguments (the bin prints it and exits with status 2).
 */
export function main(argv: readonly string[] = process.argv.slice(2)): void {
  let parsed;
  try {
    parsed = parseArgs({
      args: [...argv],
      allowPositionals: true,
      options: {
        out: { type: "string", short: "o", default: "outputs" },
        "um-per-px": { type: "string" },
        "x3p-masks": { type: "boolean", default: false },
        "brief-colours": { type: "boolean", default: false },
        verbose: { type: "boolean", short: "v", default: false },
        quiet: { type: "boolean", short: "q", default: false },
        help: { type: "boolean", short: "h", default: false },
      },
    });
  } catch (e) {
    usageError((e as Error).message);
  }
  const a = parsed.values;
  if (a.help) {
    process.stdout.write(HELP);
    return;
  }
  if (a.verbose && a.quiet) usageError("argument -q/--quiet: not allowed with argument -v/--verbose");
  if (parsed.positionals.length === 0) usageError("the following arguments are required: inputs");
  let umPerPx = DEFAULT_PARAMS.workUmPerPx;
  if (a["um-per-px"] !== undefined) {
    umPerPx = Number(a["um-per-px"]);
    if (a["um-per-px"].trim() === "" || Number.isNaN(umPerPx)) {
      usageError(`argument --um-per-px: invalid float value: '${a["um-per-px"]}'`);
    }
  }
  level = a.verbose ? DEBUG : a.quiet ? WARNING : INFO;
  setBriefColours(a["brief-colours"]!);

  const files: string[] = [];
  for (const item of parsed.positionals) {
    let isDir = false;
    try { isDir = statSync(item).isDirectory(); } catch { /* a missing file fails per scan */ }
    if (isDir) files.push(...listX3P(item));
    else files.push(item);
  }
  if (files.length === 0) usageError("no .x3p files found");
  const out = a.out!;
  mkdirSync(out, { recursive: true });
  const params = makeParams({ workUmPerPx: umPerPx });
  log.debug(`${files.length} scan(s), ${JSON.stringify(params)}`);

  const rows: Record<string, unknown>[] = [];
  const sheet: [string, Image, Image][] = [];
  for (const f of files) {
    let r: Processed;
    try {
      r = processScan(f, out, params, a["x3p-masks"]);
    } catch (exc) {
      // keep the batch going; report the failure
      const err = exc instanceof Error ? exc : new Error(String(exc));
      const msg = `${err.name}: ${err.message}`;
      log.error(`[FAIL] ${basename(f)}: ${msg}`);
      log.debug(`traceback for ${basename(f)}\n${err.stack ?? ""}`);
      rows.push({ name: stemOf(f), status: "error", warnings: msg });
      continue;
    }
    const s = r.summary as Record<string, any>;
    const flag = s.needs_review ? "REVIEW" : "auto";
    const warnings = s.warnings as string[];
    log.info(
      `[${String(s.status).padStart(7)}|${flag.padStart(6)}|qc=${fmtF(s.qc_score ?? 0, 2)}] ${s.name}  (${pyFloatRepr(s.seconds)} s)  ` +
        warnings.join("; "),
    );
    rows.push({
      name: s.name,
      status: s.status,
      qc_score: s.qc_score,
      needs_review: s.needs_review,
      review_reasons: (s.review_reasons ?? []).join(";"),
      primer_diameter_mm: s.primer_diameter_mm,
      primer_edge_inlier_frac: s.primer_edge_inlier_frac,
      fp_core_diameter_mm: s.fp_core_diameter_mm,
      fp_max_depth_um: s.fp_max_depth_um,
      aperture_diameter_mm: s.aperture_diameter_mm,
      aperture_detected: s.aperture_detected,
      drag_detected: s.drag_detected,
      drag_angle_deg: s.drag_angle_deg,
      drag_length_mm: s.drag_length_mm,
      ...Object.fromEntries(Object.entries(s.areas_mm2 as Record<string, number>).map(([k, v]) => [`area_${k}_mm2`, v])),
      warnings: warnings.join(" | "),
    });
    if (s.status !== "skipped") sheet.push([s.name, ...r.crop]);
  }

  const keySet = new Set<string>();
  for (const r of rows) for (const k of Object.keys(r)) keySet.add(k);
  const rank = (k: string): [number, number, string] => [k !== "name" ? 1 : 0, k !== "status" ? 1 : 0, k];
  const keys = [...keySet].sort((x, y) => {
    const a1 = rank(x), b1 = rank(y);
    return a1[0] - b1[0] || a1[1] - b1[1] || (a1[2] < b1[2] ? -1 : a1[2] > b1[2] ? 1 : 0);
  });
  let csv = csvRow(keys);
  for (const r of rows) csv += csvRow(keys.map((k) => csvValue(r[k])));
  writeFileSync(join(out, "summary.csv"), csv);
  if (sheet.length) imwrite(join(out, "contact_sheet.png"), contactSheet(sheet));
  log.info(`wrote ${rows.length} results to ${out}`);
}

/** Run {@link main} as a program: usage errors exit with status 2. */
export function run(argv: readonly string[] = process.argv.slice(2)): void {
  try {
    main(argv);
  } catch (e) {
    if (e instanceof UsageError) {
      process.stderr.write(e.message + "\n");
      process.exit(e.exitCode);
    }
    throw e;
  }
}

function isEntry(): boolean {
  const arg = process.argv[1];
  if (!arg) return false;
  try {
    return pathToFileURL(realpathSync(resolve(arg))).href === pathToFileURL(realpathSync(new URL(import.meta.url))).href;
  } catch {
    return false;
  }
}

if (isEntry()) run();
