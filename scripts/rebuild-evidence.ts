/**
 * Rebuild project evidence offline from verified local scans and saved pairs.
 *
 * Port of `scripts/rebuild_evidence.py`. The TypeScript steps (masks, mask audit, report,
 * annotator) run in this Node process's runtime; the Python research experiments under
 * `python/` (matching, 2D brief evaluation and benchmark, replicate consistency) read the
 * TypeScript masks and are run with `python` (or `$PYTHON`). A step whose script is absent, or
 * a Python step when no Python interpreter is found, is skipped with a message.
 *
 *     tsx scripts/rebuild-evidence.ts [--data-root data] [--out outputs/release]
 *         [--x3p-masks] [--rerun-registration] [--replicates] [--audit]
 *
 * Writes, under OUT:
 *
 *   raw/, glock/          TypeScript CLI outputs (labels.png, masks.json, figures)
 *   audit/                (--audit) the mask audit, checked against samples/validation/
 *   matching/             pairs.csv + summary.json (python/compare.py)
 *   brief/                2D brief evaluation (python/image2d.py)
 *   2d-benchmark/         2D benchmark against the TypeScript 3D masks (python/benchmark_2d.py)
 *   replicates/           (--replicates) replicate consistency (python/replicate_consistency.py)
 *   report/index.html     interactive report; report/annotator.html
 */
import { spawnSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, readdirSync } from "node:fs";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import { loadManifest, ROOT, verify } from "./fetch-nbtrd.js";

const PYTHON = process.env.PYTHON ?? "python";

class StepError extends Error {}

function show(cmd: string[]): string {
  return cmd.map((a) => (/[\s"'$]/.test(a) ? JSON.stringify(a) : a)).join(" ");
}

/** Run a TypeScript script of this repository with the current runtime (tsx loader included). */
function runTs(script: string, ...args: string[]): void {
  const path = join(ROOT, script);
  if (!existsSync(path)) {
    console.log(`skip ${script}: not found`);
    return;
  }
  const cmd = [process.execPath, ...process.execArgv, path, ...args];
  console.log(show(["tsx", script, ...args]));
  const r = spawnSync(cmd[0]!, cmd.slice(1), { cwd: ROOT, stdio: "inherit" });
  if (r.status !== 0) throw new StepError(`${script} exited with ${r.status ?? r.signal}`);
}

let pythonOk: boolean | undefined;
/** Run `python/<script>`; skipped when the script or the interpreter is missing. */
function runPy(script: string, ...args: string[]): boolean {
  const path = join(ROOT, script);
  if (!existsSync(path)) {
    console.log(`skip ${script}: not found (Python evidence experiment not present in this checkout)`);
    return false;
  }
  if (pythonOk === undefined) {
    pythonOk = spawnSync(PYTHON, ["--version"], { stdio: "ignore" }).status === 0;
    if (!pythonOk) console.log(`skip Python experiments: '${PYTHON}' not found (set PYTHON, pip install -r python/requirements.txt)`);
  }
  if (!pythonOk) return false;
  const cmd = [PYTHON, "-B", path, ...args];
  console.log(show([PYTHON, "-B", script, ...args]));
  const r = spawnSync(cmd[0]!, cmd.slice(1), { cwd: ROOT, stdio: "inherit" });
  if (r.status !== 0) throw new StepError(`${script} exited with ${r.status ?? r.signal}`);
  return true;
}

export async function main(argv = process.argv.slice(2)): Promise<number> {
  const { values } = parseArgs({
    args: argv,
    options: {
      "data-root": { type: "string", default: join(ROOT, "data") },
      out: { type: "string", default: join(ROOT, "outputs/release") },
      "x3p-masks": { type: "boolean", default: false },
      "rerun-registration": { type: "boolean", default: false },
      replicates: { type: "boolean", default: false },
      audit: { type: "boolean", default: false },
      help: { type: "boolean", short: "h", default: false },
    },
  });
  if (values.help) {
    console.log("usage: rebuild-evidence [--data-root DIR] [--out DIR] [--x3p-masks] [--rerun-registration] [--replicates] [--audit]");
    console.log("\n  --rerun-registration  recompute comparisons from all pinned matching scans");
    console.log("  --replicates          also run the replicate-consistency experiment (needs the expanded set)");
    console.log("  --audit               also audit the 34 baseline masks against samples/validation/");
    return 0;
  }
  const data = resolve(values["data-root"]!);
  const out = resolve(values.out!);
  const rerun = values["rerun-registration"]!;
  const scans: Record<string, string[]> = { raw: [], glock: [], matching: [] };
  try {
    for (const scan of loadManifest()) {
      const key = scan.path.split("/")[0]!;
      if (key === "matching" && !rerun && !values.replicates) continue;
      const target = join(data, ...scan.path.split("/"));
      await verify(scan, target);
      scans[key]!.push(target);
    }
    const listed = new Set(scans.raw!.map((p) => p.split(/[\\/]/).pop()));
    const extras = existsSync(join(data, "raw")) ? readdirSync(join(data, "raw")).filter((f) => f.endsWith(".x3p") && !listed.has(f)).sort() : [];
    if (extras.length) throw new StepError(`raw directory contains unlisted scans: ${extras.join(", ")}`);
    const exportFlag = values["x3p-masks"] ? ["--x3p-masks"] : [];
    for (const key of ["raw", "glock"]) runTs("src/cli.ts", ...scans[key]!, "-o", join(out, key), ...exportFlag);
    if (values.audit) {
      runTs("scripts/audit-masks.ts", "--data-root", data, "--out", join(out, "audit/baseline-3d.json"),
        "--csv", join(out, "audit/current-3d.csv"), "--compare-with", join(ROOT, "samples/validation/baseline-3d.json"));
    }

    const matching = join(out, "matching");
    mkdirSync(matching, { recursive: true });
    if (rerun) {
      const masks = join(out, "ts-matching");
      runTs("src/cli.ts", ...scans.matching!, "-o", masks, "-q");
      runPy("python/compare.py", masks, "-o", matching, "--x3p-dir", join(data, "matching"));
    } else {
      const saved = join(ROOT, "samples/matching/pairs.csv");
      if (existsSync(saved)) {
        copyFileSync(saved, join(matching, "pairs.csv"));
        runPy("python/compare.py", join(matching, "pairs.csv"), "--summarize-only", "-o", matching);
      } else console.log("skip matching summary: samples/matching/pairs.csv not found");
    }

    const brief = join(out, "brief");
    const briefSrc = join(ROOT, "python/samples/brief");
    if (existsSync(briefSrc)) {
      mkdirSync(brief, { recursive: true });
      for (const stem of ["brief_p1_before", "brief_p1_after_reference", "brief_p2_reference"]) {
        copyFileSync(join(briefSrc, `${stem}.png`), join(brief, `${stem}.png`));
      }
      runPy("python/image2d.py", "--eval-brief", brief, "-o", brief);
    }
    runPy("python/benchmark_2d.py", "--masks", join(out, "raw"), "--masks", join(out, "glock"), "--data-root", data, "-o", join(out, "2d-benchmark"));

    if (values.replicates) {
      const reps = join(out, "ts-replicates");
      const expanded = existsSync(join(data, "expanded")) ? [join(data, "expanded")] : [];
      runTs("src/cli.ts", join(data, "matching"), join(data, "raw"), ...expanded, "-o", reps, "-q");
      runPy("python/replicate_consistency.py", reps, "-o", join(out, "replicates"));
    }

    runTs("scripts/build-report.ts", "--data", join(data, "raw"), join(data, "glock"), "--matching", matching, "--out", join(out, "report/index.html"));
    runTs("scripts/build-annotator.ts", ...scans.raw!, "-o", join(out, "report/annotator.html"));
  } catch (exc) {
    console.error(`rebuild failed: ${(exc as Error).message}`);
    return 1;
  }
  console.log(`Rebuilt evidence in ${out}`);
  return 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().then((code) => process.exit(code));
}
