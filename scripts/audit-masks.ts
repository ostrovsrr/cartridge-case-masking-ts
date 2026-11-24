/**
 * Record deterministic 3D mask fingerprints and review outcomes for a fixed cohort.
 *
 * Port of `scripts/audit_masks.py`; same flags and the same output files, byte for byte:
 *
 *     tsx scripts/audit-masks.ts --out outputs/audit/baseline-3d.json \
 *         [--manifest docs/scan-manifest.csv] [--data-root data] \
 *         [--compare-with samples/validation/baseline-3d.json] [--csv outputs/audit/current-3d.csv]
 *
 * The label fingerprint is the SHA-256 of the label grid's raw bytes: uint8, row-major, specimen
 * view, exactly what Python's `labels.tobytes()` hashes. Because the TypeScript label grids are
 * identical to the Python ones, a TS audit reproduces the Python baseline's `label_sha256` values,
 * and `--compare-with samples/validation/baseline-3d.json` checks the port against the reference.
 */
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import { segment } from "../src/segment.js";
import { readX3P } from "../src/x3p.js";
import { loadManifest, MANIFEST, ROOT, verify } from "./fetch-nbtrd.js";
import { pyJson, pyStr, writeCsv } from "./lib/pyfmt.js";

export const CSV_COLUMNS = [
  "file", "source_sha256", "label_sha256", "label_height", "label_width", "status", "needs_review",
  "qc_score", "crater_measured_fraction", "impression_circle_iou", "primer_diameter_mm",
  "breech_face_fraction", "impression_fraction", "review_reasons",
] as const;

/** One audited scan; `result` is `Result.summary()` or an error summary. */
export interface AuditRecord {
  file: string;
  source_sha256: string;
  label_sha256: string | null;
  label_shape: [number, number] | null;
  result: Record<string, unknown>;
}

/** SHA-256 of the raw label bytes (uint8, row-major): Python's `hashlib.sha256(labels.tobytes())`. */
export function labelFingerprint(labels: Uint8Array): string {
  return createHash("sha256").update(labels).digest("hex");
}

/** The audit JSON exactly as Python writes it (`json.dumps(records, indent=2)` plus a newline). */
export function auditJson(records: AuditRecord[]): string {
  return pyJson(records, 2, (path) => path === "[].label_shape[]") + "\n";
}

/** Flatten audit records into the review table that `build-report.ts` reads. */
export function writeAuditCsv(records: AuditRecord[], path: string): void {
  const rows = records.map((rec) => {
    const res = rec.result;
    const obs = (res.qc_observations ?? {}) as Record<string, unknown>;
    const shape = rec.label_shape;
    return {
      file: rec.file,
      source_sha256: rec.source_sha256,
      label_sha256: rec.label_sha256 ?? "",
      label_height: shape ? String(shape[0]) : "",
      label_width: shape ? String(shape[1]) : "",
      status: pyStr(res.status),
      needs_review: pyStr(res.needs_review),
      qc_score: pyStr(res.qc_score),
      crater_measured_fraction: pyStr(obs.crater_measured_fraction),
      impression_circle_iou: pyStr(obs.impression_circle_iou),
      primer_diameter_mm: pyStr(res.primer_diameter_mm),
      breech_face_fraction: pyStr(obs.breech_face_fraction),
      impression_fraction: pyStr(obs.impression_fraction),
      review_reasons: ((res.review_reasons ?? []) as string[]).join(";"),
    };
  });
  writeCsv(path, CSV_COLUMNS, rows, "\n");
}

/** Files whose source hash, label shape or label hash differ between two audits (sorted). */
export function changedFiles(previous: AuditRecord[], records: AuditRecord[]): string[] {
  const key = (r: AuditRecord) => JSON.stringify([r.source_sha256, r.label_shape, r.label_sha256]);
  const old = new Map(previous.map((r) => [r.file, key(r)]));
  const now = new Map(records.map((r) => [r.file, key(r)]));
  const all = new Set([...old.keys(), ...now.keys()]);
  return [...all].filter((f) => old.get(f) !== now.get(f)).sort();
}

function pythonError(exc: unknown): string {
  const e = exc as NodeJS.ErrnoException;
  if (e?.code === "ENOENT") return `[Errno 2] No such file or directory: '${e.path ?? ""}'`;
  return e?.message ?? String(exc);
}

export interface AuditOptions {
  manifest?: string;
  dataRoot?: string;
  log?: (line: string) => void;
}

/** Segment every scan of a manifest and fingerprint its labels; never throws for one bad scan. */
export async function audit(opts: AuditOptions = {}): Promise<{ records: AuditRecord[]; integrityErrors: number; processingErrors: number }> {
  const log = opts.log ?? ((l: string) => console.log(l));
  const dataRoot = opts.dataRoot ?? join(ROOT, "data");
  const records: AuditRecord[] = [];
  let integrityErrors = 0;
  let processingErrors = 0;
  for (const scan of loadManifest(opts.manifest ?? MANIFEST)) {
    const source = join(dataRoot, ...scan.path.split("/"));
    let labels: { data: Uint8Array; rows: number; cols: number } | null = null;
    let summary: Record<string, unknown>;
    try {
      await verify(scan, source);
      try {
        const result = segment(readX3P(source));
        labels = result.labels;
        summary = result.summary();
      } catch (exc) {
        processingErrors += 1;
        const e = exc as Error;
        summary = { status: "error", needs_review: true, warnings: [`${e?.name ?? "Error"}: ${e?.message ?? exc}`] };
      }
    } catch (exc) {
      integrityErrors += 1;
      summary = { status: "integrity_error", needs_review: true, warnings: [pythonError(exc)] };
    }
    records.push({
      file: `data/${scan.path}`,
      source_sha256: scan.sha256,
      label_sha256: labels ? labelFingerprint(labels.data) : null,
      label_shape: labels ? [labels.rows, labels.cols] : null,
      result: summary,
    });
    log(`${records.length} ${scan.path} ${summary.status} review=${pyStr(summary.needs_review)}`);
  }
  return { records, integrityErrors, processingErrors };
}

const USAGE =
  "usage: audit-masks [-h] [--manifest MANIFEST] [--data-root DATA_ROOT] --out OUT [--compare-with COMPARE_WITH] [--csv CSV]";

/** CLI entry point; returns the exit code (0 ok, 1 errors or a parity break, 2 usage error). */
export async function main(argv: string[] = process.argv.slice(2), log = (l: string) => console.log(l), err = (l: string) => console.error(l)): Promise<number> {
  let values: { manifest?: string; "data-root"?: string; out?: string; "compare-with"?: string; csv?: string; help?: boolean };
  try {
    ({ values } = parseArgs({
      args: argv,
      options: {
        manifest: { type: "string", default: MANIFEST },
        "data-root": { type: "string", default: join(ROOT, "data") },
        out: { type: "string" },
        "compare-with": { type: "string" },
        csv: { type: "string" },
        help: { type: "boolean", short: "h", default: false },
      },
    }));
  } catch (e) {
    err(USAGE);
    err(`audit-masks: error: ${(e as Error).message}`);
    return 2;
  }
  if (values.help) {
    log(USAGE);
    log("\nRecord deterministic 3D mask fingerprints and review outcomes for a fixed cohort.\n");
    log("  --compare-with  require the same scan, label grid and mask bytes");
    log("  --csv           also write the per-scan review table used by the report");
    return 0;
  }
  if (!values.out) {
    err(USAGE);
    err("audit-masks: error: the following arguments are required: --out");
    return 2;
  }
  const { records, integrityErrors, processingErrors } = await audit({
    manifest: resolve(values.manifest!),
    dataRoot: resolve(values["data-root"]!),
    log,
  });
  mkdirSync(dirname(resolve(values.out)), { recursive: true });
  writeFileSync(values.out, auditJson(records), "utf8");
  if (values.csv) writeAuditCsv(records, values.csv);
  if (integrityErrors || processingErrors) {
    err(`${integrityErrors} integrity errors and ${processingErrors} scan-processing errors`);
    return 1;
  }
  if (values["compare-with"]) {
    const previous = JSON.parse(readFileSync(values["compare-with"], "utf8")) as AuditRecord[];
    const changed = changedFiles(previous, records);
    if (changed.length) {
      err("Changed mask inputs or outputs:\n" + changed.join("\n"));
      return 1;
    }
    log(`Exact 3D mask parity on ${records.length} files.`);
  }
  return 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().then((code) => process.exit(code));
}
