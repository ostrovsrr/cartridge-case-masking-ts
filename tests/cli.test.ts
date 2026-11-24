/**
 * Batch CLI (`src/cli.ts`): outputs per scan, per-file error capture, argument errors, and parity
 * of `masks.json` / `summary.csv` with the Python CLI on the same Python-written synthetic X3P
 * (`reference/out/cli`, from `python reference/export_masks.py`).
 */
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { UsageError, main } from "../src/cli.js";
import { decodePng } from "../src/image.js";
import { LABEL_NAMES } from "../src/segment.js";
import { makeHead } from "../src/synthetic.js";
import { writeX3P } from "../src/x3p.js";
import { REF } from "./parity-utils.js";

const tmp = mkdtempSync(join(tmpdir(), "cartmask-cli-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

/** Parse a `summary.csv` written with `\r\n` and minimal quoting. */
function parseCsv(text: string): Record<string, string>[] {
  const rows: string[][] = [];
  let row: string[] = [], cell = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (q) {
      if (ch === '"' && text[i + 1] === '"') { cell += '"'; i++; }
      else if (ch === '"') q = false;
      else cell += ch;
    } else if (ch === '"') q = true;
    else if (ch === ",") { row.push(cell); cell = ""; }
    else if (ch === "\r" && text[i + 1] === "\n") { row.push(cell); rows.push(row); row = []; cell = ""; i++; }
    else cell += ch;
  }
  const [head, ...body] = rows;
  return body.map((r) => Object.fromEntries(head.map((k, j) => [k, r[j]])));
}

describe("batch run", () => {
  const out = join(tmp, "out");
  beforeAll(() => {
    const [surf] = makeHead({ dragAngleDeg: 30.0 });
    surf.name = "head";
    writeX3P(surf, join(tmp, "head.x3p"));
    main([join(tmp, "head.x3p"), join(tmp, "missing.x3p"), "-o", out, "--x3p-masks", "-q"]);
  });

  it("writes every per-scan output", () => {
    for (const f of ["original.png", "overlay.png", "overlay_canonical.png", "labels.png", "masks.json", "head__masked.x3p"]) {
      expect(existsSync(join(out, "head", f)), f).toBe(true);
    }
    for (const [k, name] of Object.entries(LABEL_NAMES)) expect(existsSync(join(out, "head", `mask_${k}_${name}.png`))).toBe(true);
    expect(existsSync(join(out, "head__side_by_side.png"))).toBe(true);
    expect(existsSync(join(out, "contact_sheet.png"))).toBe(true);
    const labels = decodePng(readFileSync(join(out, "head", "labels.png")));
    expect(labels.channels).toBe(1);
    expect(labels.data.reduce((m, v) => Math.max(m, v), 0)).toBeLessThanOrEqual(4);
    const m = JSON.parse(readFileSync(join(out, "head", "masks.json"), "utf8"));
    expect(m.status).toBe("ok");
    expect(m.label_values).toEqual({ "1": "breech_face_impression", "2": "aperture_shear", "3": "firing_pin_shear", "4": "firing_pin_impression" });
  });

  it("captures per-file errors in summary.csv and keeps going", () => {
    const rows = parseCsv(readFileSync(join(out, "summary.csv"), "utf8"));
    expect(rows.map((r) => r.name)).toEqual(["head", "missing"]);
    expect(rows[0].status).toBe("ok");
    expect(rows[1].status).toBe("error");
    expect(rows[1].warnings).toMatch(/^Error: ENOENT/);
    expect(Object.keys(rows[0]).slice(0, 2)).toEqual(["name", "status"]);
  });
});

describe("arguments", () => {
  it("rejects -v with -q and empty inputs", () => {
    expect(() => main([tmp, "-v", "-q"])).toThrow(UsageError);
    expect(() => main([])).toThrow(UsageError);
    expect(() => main([join(tmp, "empty-dir-does-not-matter"), "--um-per-px", "abc"])).toThrow(UsageError);
  });
});

const CLI_REF = join(REF, "cli");
const CASES = ["default", "round_seed1", "drag20", "inverted_seed2"].filter(
  (n) => existsSync(join(CLI_REF, n, "masks.json")) && existsSync(join(REF, "synthetic", `${n}.x3p`)),
);

describe.skipIf(CASES.length === 0)("Python CLI parity (reference/out/cli)", () => {
  const out = join(tmp, "parity");
  beforeAll(() => {
    main([...CASES.map((n) => join(REF, "synthetic", `${n}.x3p`)), "-o", out, "-q"]);
  });

  it.each(CASES)("%s: masks.json equals Python's (except seconds and source)", (name) => {
    // Python's write_text() translates newlines on Windows; the TS CLI always writes \n
    const strip = (t: string) => t.split(/\r?\n/).filter((l) => !/^ {2}"(seconds|source)": /.test(l));
    const py = readFileSync(join(CLI_REF, name, "masks.json"), "utf8");
    const ts = readFileSync(join(out, name, "masks.json"), "utf8");
    const a = JSON.parse(py), b = JSON.parse(ts);
    for (const k of ["seconds", "source"]) { delete a[k]; delete b[k]; }
    expect(b).toEqual(a);
    expect(strip(ts)).toEqual(strip(py)); // same text, float formatting included
  });

  it("summary.csv has Python's columns and values", () => {
    const py = parseCsv(readFileSync(join(CLI_REF, "summary.csv"), "utf8")).filter((r) => CASES.includes(r.name));
    const ts = parseCsv(readFileSync(join(out, "summary.csv"), "utf8"));
    expect(ts).toEqual(py);
  });
});
