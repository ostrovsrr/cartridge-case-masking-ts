/**
 * The mask audit reproduces the Python baseline in samples/validation/ byte for byte.
 *
 * Offline checks (always run): Python-compatible JSON/CSV formatting and the fingerprint
 * definition. Real-data checks need CARTMASK_DATA (the folder holding raw/, glock/, matching/,
 * expanded/); set CARTMASK_AUDIT_EXPANDED=1 to also audit the 118-scan expanded cohort.
 */
import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { auditJson, changedFiles, labelFingerprint, main, type AuditRecord } from "../scripts/audit-masks.js";
import { ROOT } from "../scripts/fetch-nbtrd.js";
import { pyFloat, pyJson } from "../scripts/lib/pyfmt.js";

const VALIDATION = join(ROOT, "samples/validation");
const DATA = process.env.CARTMASK_DATA;
/** Committed text with any CRLF from a Windows checkout undone. */
const committed = (name: string): string => readFileSync(join(VALIDATION, name), "utf8").replace(/\r\n/g, "\n");
const tmp = mkdtempSync(join(tmpdir(), "cartmask-audit-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

describe("Python-compatible formatting", () => {
  it("formats floats like repr()", () => {
    expect(pyFloat(8)).toBe("8.0");
    expect(pyFloat(0)).toBe("0.0");
    expect(pyFloat(0.4583)).toBe("0.4583");
    expect(pyFloat(-1.5)).toBe("-1.5");
    expect(pyFloat(1e-5)).toBe("1e-05");
    expect(pyFloat(2.5e-7)).toBe("2.5e-07");
    expect(pyFloat(1e16)).toBe("1e+16");
    expect(pyFloat(123456.789)).toBe("123456.789");
  });

  it("dumps JSON like json.dumps(indent=2)", () => {
    const v = { a: [1, 2], b: {}, c: [], d: "x°", e: null, f: true };
    expect(pyJson(v, 2, (p) => p === "a[]")).toBe(
      '{\n  "a": [\n    1,\n    2\n  ],\n  "b": {},\n  "c": [],\n  "d": "x\\u00b0",\n  "e": null,\n  "f": true\n}',
    );
  });

  it("round-trips the committed baseline byte for byte", () => {
    const text = committed("baseline-3d.json");
    expect(auditJson(JSON.parse(text) as AuditRecord[])).toBe(text);
  });

  it("fingerprints the raw uint8 label bytes", () => {
    const labels = Uint8Array.from([0, 1, 2, 3, 4, 0]);
    expect(labelFingerprint(labels)).toBe(createHash("sha256").update(Buffer.from(labels)).digest("hex"));
  });

  it("reports changed files", () => {
    const rec = (file: string, sha: string): AuditRecord => ({
      file, source_sha256: "s", label_sha256: sha, label_shape: [2, 3], result: {},
    });
    expect(changedFiles([rec("a", "1"), rec("b", "2")], [rec("a", "1"), rec("b", "3"), rec("c", "4")])).toEqual(["b", "c"]);
  });
});

const baselineData = DATA && existsSync(join(DATA, "raw")) && existsSync(join(DATA, "glock")) && existsSync(join(DATA, "matching"));

describe.skipIf(!baselineData)("audit parity on real scans (CARTMASK_DATA)", () => {
  it("reproduces samples/validation/baseline-3d.json and current-3d.csv exactly", async () => {
    const out = join(tmp, "baseline-3d.json");
    const csv = join(tmp, "current-3d.csv");
    const log: string[] = [];
    const errors: string[] = [];
    const code = await main(
      ["--data-root", DATA!, "--out", out, "--csv", csv, "--compare-with", join(VALIDATION, "baseline-3d.json")],
      (l) => log.push(l),
      (l) => errors.push(l),
    );
    expect(errors).toEqual([]);
    expect(code).toBe(0);
    expect(log.at(-1)).toBe("Exact 3D mask parity on 34 files.");
    // fingerprints, review outcomes and every summary field, byte for byte
    expect(readFileSync(out, "utf8")).toBe(committed("baseline-3d.json"));
    expect(readFileSync(csv, "utf8")).toBe(committed("current-3d.csv"));
  }, 900_000);

  it.skipIf(!process.env.CARTMASK_AUDIT_EXPANDED || !existsSync(join(DATA ?? "", "expanded")))(
    "reproduces samples/validation/expanded-3d.csv exactly",
    async () => {
      const csv = join(tmp, "expanded-3d.csv");
      const code = await main(
        ["--manifest", join(ROOT, "docs/expanded-scan-manifest.csv"), "--data-root", DATA!, "--out", join(tmp, "expanded-3d.json"), "--csv", csv],
        () => {},
        () => {},
      );
      expect(code).toBe(0);
      expect(readFileSync(csv, "utf8")).toBe(committed("expanded-3d.csv"));
    },
    3_600_000,
  );
});
