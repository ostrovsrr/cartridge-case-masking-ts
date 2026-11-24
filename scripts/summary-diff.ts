/** Deep-compare TS Result.summary() with the Python masks.json-style summaries in reference/out/. */
import { join } from "node:path";
import { paramsFromPython, segment } from "../src/segment.js";
import { readX3P } from "../src/x3p.js";
import { loadManifest, loadSummary, REF } from "../tests/parity-utils.js";

const SKIP = new Set(["seconds", "labels_shape", "params", "kwargs", "source", "study", "set"]);
function diff(a: unknown, b: unknown, path: string, out: string[]): void {
  if (typeof a === "number" && typeof b === "number") {
    if (Math.abs(a - b) > 1e-9 * Math.max(1, Math.abs(b))) out.push(`${path}: ts ${a} py ${b}`);
    return;
  }
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) out.push(`${path}: len ${a.length} vs ${b.length}`);
    for (let i = 0; i < Math.min(a.length, b.length); i++) diff(a[i], b[i], `${path}[${i}]`, out);
    return;
  }
  if (a && b && typeof a === "object" && typeof b === "object") {
    const ka = Object.keys(a).filter((k) => !SKIP.has(k)), kb = Object.keys(b).filter((k) => !SKIP.has(k));
    if (ka.join() !== kb.join()) out.push(`${path}: keys ts [${ka}] py [${kb}]`);
    for (const k of kb) if (k in (a as object)) diff((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k], `${path}.${k}`, out);
    return;
  }
  if (a !== b) out.push(`${path}: ts ${JSON.stringify(a)} py ${JSON.stringify(b)}`);
}
const man = loadManifest()!;
const data = process.env.CARTMASK_DATA ?? man.data;
const which = process.argv[2] ?? "synthetic";
const entries = which === "synthetic" ? man.synthetic! : man.real!.filter((e) => e.set === which);
let bad = 0;
for (const e of entries) {
  const surf = readX3P(which === "synthetic" ? join(REF, e.x3p!) : join(data, e.file!));
  const ts = JSON.parse(JSON.stringify(segment(surf, paramsFromPython(e.params ?? {})).summary()));
  const py = loadSummary(which === "synthetic" ? `synthetic/${e.name}.json` : e.json!);
  const out: string[] = [];
  diff(ts, py, "", out);
  if (out.length) {
    bad++;
    console.log(`${e.name}:\n  ${out.join("\n  ")}`);
  }
}
console.log(`${entries.length - bad}/${entries.length} summaries identical`);
