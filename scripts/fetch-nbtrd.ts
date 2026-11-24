/**
 * Download pinned NIST X3P scans or verify local copies offline.
 *
 * Port of `scripts/fetch_nbtrd.py`.  Same manifests (`docs/scan-manifest.csv`,
 * `docs/expanded-scan-manifest.csv`), same flags:
 *
 *     tsx scripts/fetch-nbtrd.ts [--set default|matching|glock|expanded|all]
 *                                [--data-root DIR] [--out DIR] [--dry-run | --verify-only]
 *
 * Requests are made one at a time (never in parallel), each with a 120 s
 * timeout and up to three attempts with a 2 s / 4 s back-off, exactly like the
 * Python script, so the NIST NBTRD servers see the same request pattern.
 */
import { createHash, randomBytes } from "node:crypto";
import { createReadStream, existsSync, mkdirSync, readFileSync, renameSync, statSync, unlinkSync, writeFileSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import { unzipSync } from "fflate";

export const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
export const MANIFEST = join(ROOT, "docs/scan-manifest.csv");
export const DATA_SETS = { default: "raw", matching: "matching", glock: "glock", expanded: "expanded" } as const;
export type SetName = keyof typeof DATA_SETS;
const DOWNLOAD_URL =
  /^https:\/\/tsapps\.nist\.gov\/NRBTD\/Studies\/(?:Cartridge\/DownloadMeasurements|CartridgeMeasurement\/DownloadMeasurement)\/[0-9a-f-]{36}$/;
const USER_AGENT = "cartmask-forensic-research/2.0";

/** One pinned scan; `path` is POSIX `<set folder>/<file>.x3p` (the manifest path without `data/`). */
export interface Scan {
  readonly path: string;
  readonly url: string;
  readonly size: number;
  readonly sha256: string;
}

/** Injectable side effects (network and sleeping) so tests never touch NIST. */
export interface Deps {
  fetch?: typeof globalThis.fetch;
  sleep?: (ms: number) => Promise<void>;
  stdout?: (line: string) => void;
  stderr?: (line: string) => void;
}

const defaultSleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

/** Minimal RFC 4180 CSV reader returning one record per row keyed by the header (like `csv.DictReader`). */
function readCsv(text: string): Record<string, string>[] {
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1); // utf-8-sig
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;
  let i = 0;
  while (i < text.length) {
    const ch = text[i]!;
    if (quoted) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i++;
        } else quoted = false;
      } else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else field += ch;
    i++;
  }
  if (field !== "" || row.length) {
    row.push(field);
    rows.push(row);
  }
  const nonEmpty = rows.filter((r) => !(r.length === 1 && r[0] === ""));
  const [header, ...body] = nonEmpty;
  if (!header) return [];
  return body.map((r) => Object.fromEntries(header.map((h, k) => [h, r[k] ?? ""])));
}

/** Parse and validate a manifest; rejects malformed, duplicate or off-site entries. */
export function loadManifest(path: string = MANIFEST): Scan[] {
  const scans: Scan[] = [];
  const seen = new Set<string>();
  const folders = new Set<string>(Object.values(DATA_SETS));
  for (const row of readCsv(readFileSync(path, "utf8"))) {
    const file = row["file"] ?? "";
    const parts = file.split("/").filter((p) => p !== "" && p !== ".");
    const name = parts.join("/");
    const url = row["download_url"] ?? "";
    const sizeText = (row["bytes"] ?? "").trim();
    const size = /^[+-]?\d+$/.test(sizeText) ? parseInt(sizeText, 10) : NaN;
    const digest = row["sha256"] ?? "";
    if (Number.isNaN(size)) throw new Error(`invalid literal for int() with base 10: '${row["bytes"]}'`);
    if (
      parts.length !== 3 ||
      parts[0] !== "data" ||
      !folders.has(parts[1]!) ||
      !/^[A-Za-z0-9_.-]+\.x3p$/.test(parts[2]!) ||
      seen.has(name) ||
      !DOWNLOAD_URL.test(url) ||
      size <= 0 ||
      !/^[0-9a-f]{64}$/.test(digest)
    ) {
      throw new Error(`invalid or duplicate manifest entry: ${JSON.stringify(row)}`);
    }
    seen.add(name);
    scans.push({ path: parts.slice(1).join("/"), url, size, sha256: digest });
  }
  if (!scans.length) throw new Error(`manifest is empty: ${path}`);
  return scans;
}

/** GET `url` with three attempts (2 s then 4 s pause between them) and a 120 s timeout each. */
export async function get(url: string, deps: Deps = {}): Promise<Uint8Array> {
  const doFetch = deps.fetch ?? globalThis.fetch;
  const sleep = deps.sleep ?? defaultSleep;
  const stderr = deps.stderr ?? ((l: string) => console.error(l));
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const response = await doFetch(url, {
        headers: { "User-Agent": USER_AGENT },
        signal: AbortSignal.timeout(120_000),
      });
      // urllib raises HTTPError (an OSError) for non-2xx responses
      if (!response.ok) throw new Error(`HTTP Error ${response.status}: ${response.statusText}`);
      return new Uint8Array(await response.arrayBuffer());
    } catch (exc) {
      if (attempt === 2) throw exc;
      stderr(`retry ${url}: ${(exc as Error).message ?? exc}`);
      await sleep(2000 * (attempt + 1));
    }
  }
  throw new Error("unreachable");
}

function sha256Hex(data: Uint8Array): string {
  return createHash("sha256").update(data).digest("hex");
}

/** True when `data` has the manifest byte count and SHA-256. */
export function matches(data: Uint8Array, scan: Scan): boolean {
  return data.byteLength === scan.size && sha256Hex(data) === scan.sha256;
}

/** The pinned X3P itself, or the matching `.x3p` member of a ZIP bundle returned by NIST. */
export function extract(data: Uint8Array, scan: Scan): Uint8Array {
  if (matches(data, scan)) return data;
  const members = unzipSync(data, {
    filter: (f) => f.name.toLowerCase().endsWith(".x3p") && f.originalSize === scan.size,
  });
  for (const payload of Object.values(members)) {
    if (matches(payload, scan)) return payload;
  }
  throw new Error("download contained no X3P matching the manifest SHA-256");
}

/** Check a local file against the manifest size and SHA-256 (streams the file). */
export async function verify(scan: Scan, target: string): Promise<void> {
  if (statSync(target).size !== scan.size) throw new Error("byte count does not match the manifest");
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(target)) hash.update(chunk as Buffer);
  if (hash.digest("hex") !== scan.sha256) throw new Error("SHA-256 does not match the manifest");
}

/** Download one scan and move it into place atomically (staging file in the target folder). */
export async function fetchScan(scan: Scan, target: string, deps: Deps = {}): Promise<void> {
  const payload = extract(await get(scan.url, deps), scan);
  mkdirSync(dirname(target), { recursive: true });
  const staging = join(dirname(target), `tmp${randomBytes(6).toString("hex")}`);
  try {
    writeFileSync(staging, payload);
    renameSync(staging, target);
  } finally {
    if (existsSync(staging)) unlinkSync(staging);
  }
}

const USAGE =
  "usage: fetch-nbtrd [-h] [--set {default,matching,glock,expanded,all}] [--data-root DATA_ROOT] [--out OUT] [--dry-run | --verify-only]";

/** CLI entry point; returns the process exit code (0 ok, 1 failures, 2 usage error). */
export async function main(argv: string[] = process.argv.slice(2), deps: Deps = {}): Promise<number> {
  const out = deps.stdout ?? ((l: string) => console.log(l));
  const err = deps.stderr ?? ((l: string) => console.error(l));
  const usageError = (msg: string): number => {
    err(USAGE);
    err(`fetch-nbtrd: error: ${msg}`);
    return 2;
  };
  let values: { set?: string; "data-root"?: string; out?: string; "dry-run"?: boolean; "verify-only"?: boolean; help?: boolean };
  try {
    ({ values } = parseArgs({
      args: argv,
      options: {
        set: { type: "string", default: "default" },
        "data-root": { type: "string", default: join(ROOT, "data") },
        out: { type: "string" },
        "dry-run": { type: "boolean", default: false },
        "verify-only": { type: "boolean", default: false },
        help: { type: "boolean", short: "h", default: false },
      },
      allowPositionals: false,
    }));
  } catch (e) {
    return usageError((e as Error).message);
  }
  if (values.help) {
    out(USAGE);
    out("\nDownload pinned NIST X3P scans or verify local copies offline.\n");
    out("  --set          all selects the 34 baseline files. Expanded is a separate cohort.");
    out("  --data-root    destination root (default: <repo>/data)");
    out("  --out          override the destination folder for one set");
    out("  --dry-run      list pinned URLs without making requests");
    out("  --verify-only  check existing files without making requests");
    return 0;
  }
  const set = values.set!;
  if (set !== "all" && !(set in DATA_SETS)) {
    return usageError(`argument --set: invalid choice: '${set}' (choose from 'default', 'matching', 'glock', 'expanded', 'all')`);
  }
  if (values["dry-run"] && values["verify-only"]) {
    return usageError("argument --verify-only: not allowed with argument --dry-run");
  }
  if (values.out && set === "all") {
    return usageError("--out requires a single set. Use --data-root with --set all.");
  }
  const manifest = set === "expanded" ? join(ROOT, "docs/expanded-scan-manifest.csv") : MANIFEST;
  const scans = loadManifest(manifest).filter(
    (scan) => set === "all" || scan.path.split("/")[0] === DATA_SETS[set as SetName],
  );
  let failures = 0;
  for (const scan of scans) {
    const target = values.out ? join(values.out, basename(scan.path)) : join(values["data-root"]!, ...scan.path.split("/"));
    if (values["dry-run"]) {
      out(`${target} <- ${scan.url}`);
      continue;
    }
    try {
      if (!values["verify-only"] && !existsSync(target)) await fetchScan(scan, target, deps);
      await verify(scan, target);
      out(`verified ${target}`);
    } catch (exc) {
      const e = exc as NodeJS.ErrnoException;
      err(`${target}: ${e.code === "ENOENT" ? `[Errno 2] No such file or directory: '${target}'` : e.message}`);
      failures += 1;
    }
  }
  if (!values["dry-run"]) out(`${scans.length - failures}/${scans.length} scans verified`);
  return failures > 0 ? 1 : 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().then((code) => process.exit(code));
}
