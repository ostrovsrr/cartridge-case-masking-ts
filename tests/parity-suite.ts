/** Vitest suite builder for the Python parity tests (see parity-utils.ts for the comparison). */
import { existsSync } from "node:fs";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { paramsFromPython, segment } from "../src/segment.js";
import { readX3P } from "../src/x3p.js";
import {
  compare,
  fmtIoU,
  HOW_TO,
  loadLabels,
  loadManifest,
  loadSummary,
  REF,
  type Comparison,
  type ManifestEntry,
} from "./parity-utils.js";



/** Minimum IoU per present class: E3 full heads and synthetic inputs are held tighter. */
export function iouTarget(study: string): number {
  if (study === "synthetic") return 0.999;
  return study.startsWith("E3") ? 0.99 : 0.98;
}
/** Geometry tolerances [px] and [deg]. */
export const TOL_PX = 1.0;
export const TOL_DEG = 2.0;

/** Per-study summary: minimum IoU per class, review agreement, worst geometry, timings. */
export function summarise(rows: Comparison[]): string {
  const by = new Map<string, Comparison[]>();
  for (const r of rows) by.set(r.study, [...(by.get(r.study) ?? []), r]);
  const out = [
    "study                 n   minIoU BF  AS     FPS    FPI    review  reasons  max dpx  max ddeg  ts s/scan  py s/scan",
  ];
  for (const [study, rs] of by) {
    const mins = [0, 1, 2, 3].map((k) => {
      const v = rs.map((r) => r.iou[k]).filter((x): x is number => x !== null);
      return v.length ? Math.min(...v) : null;
    });
    const dpx = Math.max(...rs.map((r) => Math.max(r.primerCentre, r.primerRadius, r.pinCentre, r.pinRadius, r.apertureRadius)));
    const ddeg = Math.max(...rs.map((r) => r.dragAngle));
    const ts = rs.reduce((s, r) => s + r.seconds, 0) / rs.length;
    const py = rs.reduce((s, r) => s + r.pySeconds, 0) / rs.length;
    out.push(
      `${study.padEnd(20)} ${String(rs.length).padStart(3)}   ${mins.map(fmtIoU).join(" ")}  ` +
        `${rs.filter((r) => r.reviewAgree).length}/${rs.length}`.padEnd(8) +
        `${rs.filter((r) => r.reasonsAgree).length}/${rs.length}`.padEnd(9) +
        `${dpx.toExponential(1).padStart(8)} ${ddeg.toExponential(1).padStart(9)}  ${ts.toFixed(2).padStart(9)}  ${py.toFixed(2).padStart(9)}`,
    );
  }
  return out.join("\n");
}

/**
 * Build the parity suite for one part of the reference: `synthetic` (X3P inputs written by the
 * Python writer), or the `baseline` / `expanded` scan manifests (scans read from CARTMASK_DATA).
 */
export function paritySuite(part: "synthetic" | "baseline" | "expanded"): void {
  const man = loadManifest();
  const title = `python parity: ${part}`;
  if (!man) {
    console.warn(HOW_TO);
    describe.skip(title, () => it("reference/out missing", () => {}));
    return;
  }
  const data = process.env.CARTMASK_DATA;
  let entries: ManifestEntry[];
  if (part === "synthetic") entries = man.synthetic ?? [];
  else {
    if (!data) {
      console.warn(`${title}: set CARTMASK_DATA to the scan directory to compare real scans`);
      describe.skip(title, () => it("CARTMASK_DATA unset", () => {}));
      return;
    }
    if (part === "expanded" && process.env.CARTMASK_PARITY_EXPANDED !== "1") {
      describe.skip(`${title} (set CARTMASK_PARITY_EXPANDED=1)`, () => it("expanded manifest not requested", () => {}));
      return;
    }
    entries = (man.real ?? []).filter((e) => e.set === part);
  }
  if (entries.length === 0) {
    console.warn(`${title}: no entries in reference/out/manifest.json; ${HOW_TO}`);
    describe.skip(title, () => it("no reference entries", () => {}));
    return;
  }

  describe(title, () => {
    const rows: Comparison[] = [];
    afterAll(() => {
      if (rows.length) console.log(`\n${title}\n${summarise(rows)}`);
    });
    for (const e of entries) {
      it(e.name + (e.file ? ` (${e.file.split("/")[0]})` : ""), () => {
        let path: string, json: string, png: string;
        if (part === "synthetic") {
          path = join(REF, e.x3p!);
          json = `synthetic/${e.name}.json`;
          png = `synthetic/${e.name}.labels.png`;
        } else {
          path = join(data!, e.file!);
          json = e.json!;
          png = e.labels!;
          if (!existsSync(path)) {
            console.warn(`missing scan ${path}`);
            return;
          }
        }
        const surf = readX3P(path);
        const t0 = performance.now();
        const res = segment(surf, paramsFromPython(e.params ?? {}));
        const dt = (performance.now() - t0) / 1000;
        const study = part === "synthetic" ? "synthetic" : `${e.study}`;
        const c = compare(e.name, part === "synthetic" ? "synthetic" : `${study}/${part}`, res, loadSummary(json), loadLabels(png), dt);
        rows.push(c);
        expect(c.shapeOk, "label grid shape").toBe(true);
        const target = iouTarget(study);
        c.iou.forEach((v, k) => {
          if (v !== null) expect(v, `IoU of class ${k + 1}`).toBeGreaterThanOrEqual(target);
        });
        expect(c.statusAgree, "status").toBe(true);
        expect(c.reviewAgree, `needs_review (ts ${c.tsReasons} / py ${c.pyReasons})`).toBe(true);
        expect(c.tsReasons, "review_reasons").toEqual(c.pyReasons);
        for (const [k, v] of [
          ["primer centre", c.primerCentre],
          ["primer radius", c.primerRadius],
          ["pin centre", c.pinCentre],
          ["pin radius", c.pinRadius],
          ["aperture radius", c.apertureRadius],
        ] as const) {
          expect(v, `${k} [px]`).toBeLessThanOrEqual(TOL_PX);
        }
        expect(c.dragAgree, "drag_detected").toBe(true);
        expect(c.dragAngle, "drag angle [deg]").toBeLessThanOrEqual(TOL_DEG);
        expect(c.qc, "qc score").toBeLessThanOrEqual(0.01);
      });
    }
  });
}
