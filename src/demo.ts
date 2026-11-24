/**
 * Offline demo: write synthetic X3P heads with known geometry and mask them (port of
 * `cartmask/demo.py`).
 *
 *     npx tsx src/demo.ts                       # writes outputs/demo
 *     npx tsx src/demo.ts -o somewhere --brief-colours
 *
 * No download is needed. Three synthetic small-pistol heads are written as X3P files (a dragged
 * strike, a round strike and a cast with inverted heights), then the normal batch command masks
 * them, so the demo exercises the same code path and output files as real scans: per-scan folders
 * with `labels.png`, `masks.json` and overlays, side-by-side figures, `summary.csv` and a contact
 * sheet. `truth.json` records the generating geometry for comparison.
 */
import { mkdirSync, realpathSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import * as cli from "./cli.js";
import { pyJsonDumps } from "./pyformat.js";
import { makeHead, type HeadOptions } from "./synthetic.js";
import { writeX3P } from "./x3p.js";

export const CASES: Readonly<Record<string, HeadOptions & { invert?: boolean }>> = {
  synthetic_drag_30deg: { dragAngleDeg: 30.0, seed: 0 },
  synthetic_round_strike: { dragAngleDeg: null, seed: 1 },
  synthetic_cast_inverted: { dragAngleDeg: 120.0, seed: 2, invert: true },
};

const snake = (k: string): string => k.replace(/[A-Z]/g, (c) => "_" + c.toLowerCase());

/** Write the synthetic scans and run the batch CLI on them. */
export function main(argv: readonly string[] = process.argv.slice(2)): void {
  const { values } = parseArgs({
    args: [...argv],
    options: {
      out: { type: "string", short: "o", default: "outputs/demo" },
      "brief-colours": { type: "boolean", default: false },
    },
  });
  const out = values.out!;
  const scans = join(out, "scans");
  mkdirSync(scans, { recursive: true });
  const truth: Record<string, Record<string, unknown>> = {};
  for (const [name, kw] of Object.entries(CASES)) {
    const { invert = false, ...opts } = kw;
    const [surf, t] = makeHead(opts);
    if (invert) surf.z = { ...surf.z, data: surf.z.data.map((v) => -v) };
    surf.name = name;
    writeX3P(surf, join(scans, `${name}.x3p`));
    const rec: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(t)) if (!k.endsWith("Mask")) rec[snake(k)] = v; // masks are arrays
    rec.inverted_heights = invert;
    truth[name] = rec;
  }
  writeFileSync(join(out, "truth.json"), pyJsonDumps(truth, 2));
  cli.main([scans, "-o", join(out, "masks"), ...(values["brief-colours"] ? ["--brief-colours"] : [])]);
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

if (isEntry()) main();
