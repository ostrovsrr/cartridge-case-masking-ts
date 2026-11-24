/**
 * Run the TypeScript pipeline on every Python reference case in reference/out/ and print a
 * per-case comparison table (IoU per class, geometry deltas, review agreement, timings).
 *
 *   npx tsx scripts/parity-report.ts [--synthetic] [--real] [--filter substring] [--json out.json]
 */
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { segment, paramsFromPython } from "../src/segment.js";
import { readX3P } from "../src/x3p.js";
import { compare, fmtIoU, HOW_TO, loadLabels, loadManifest, loadSummary, REF, type Comparison } from "../tests/parity-utils.js";

const args = process.argv.slice(2);
const only = args.includes("--synthetic") ? "synthetic" : args.includes("--real") ? "real" : null;
const fi = args.indexOf("--filter");
const filter = fi >= 0 ? args[fi + 1] : null;
const ji = args.indexOf("--json");
const jsonOut = ji >= 0 ? args[ji + 1] : null;

const man = loadManifest();
if (!man) {
  console.error(HOW_TO);
  process.exit(1);
}
const data = process.env.CARTMASK_DATA ?? man.data;
const rows: Comparison[] = [];
const line = (c: Comparison) =>
  `${c.name.slice(0, 44).padEnd(44)} ${c.iou.map(fmtIoU).join(" ")}  pc=${c.primerCentre.toFixed(2)} pr=${c.primerRadius.toFixed(2)} ` +
  `fc=${c.pinCentre.toFixed(2)} fr=${c.pinRadius.toFixed(2)} ar=${c.apertureRadius.toFixed(2)} da=${c.dragAngle.toFixed(2)} ` +
  `qc=${c.qc.toFixed(3)} rev=${c.reviewAgree ? "=" : "X"}${c.reasonsAgree ? "" : ` [${c.tsReasons}] vs [${c.pyReasons}]`} ` +
  `t=${c.seconds.toFixed(2)}s py=${c.pySeconds.toFixed(2)}s`;

if (only !== "real") {
  for (const e of man.synthetic ?? []) {
    if (filter && !e.name.includes(filter)) continue;
    const surf = readX3P(join(REF, e.x3p!));
    const t0 = performance.now();
    const res = segment(surf, paramsFromPython(e.params));
    const dt = (performance.now() - t0) / 1000;
    const c = compare(e.name, "synthetic", res, loadSummary(`synthetic/${e.name}.json`), loadLabels(`synthetic/${e.name}.labels.png`), dt);
    rows.push(c);
    console.log(line(c));
  }
}
if (only !== "synthetic") {
  for (const e of man.real ?? []) {
    if (filter && !e.name.includes(filter)) continue;
    const surf = readX3P(join(data, e.file!));
    const t0 = performance.now();
    const res = segment(surf);
    const dt = (performance.now() - t0) / 1000;
    const c = compare(e.name, `${e.study}/${e.set}`, res, loadSummary(e.json!), loadLabels(e.labels!), dt);
    rows.push(c);
    console.log(line(c));
  }
}
if (jsonOut) writeFileSync(jsonOut, JSON.stringify(rows, null, 1));
