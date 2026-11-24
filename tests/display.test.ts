/** Display conventions and the offline demo (port of Python `tests/test_display.py`). */
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import * as demo from "../src/demo.js";
import { FIRING_PIN_IMPRESSION, FIRING_PIN_SHEAR, segment } from "../src/segment.js";
import { makeHead } from "../src/synthetic.js";
import { COLOURS, LIGHT_BLUE, PURPLE, flatMask, setBriefColours } from "../src/visualize.js";

const tmp = mkdtempSync(join(tmpdir(), "cartmask-display-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

describe("display", () => {
  it("brief colours swap display only", () => {
    const res = segment(makeHead({ dragAngleDeg: 30.0 })[0]);
    const before = res.labels!.data.slice();
    try {
      setBriefColours(true);
      expect(COLOURS[FIRING_PIN_IMPRESSION]).toEqual(PURPLE);
      expect(COLOURS[FIRING_PIN_SHEAR]).toEqual(LIGHT_BLUE);
      const img = flatMask(res);
      const L = res.labels!.data;
      let n = 0;
      for (let i = 0; i < L.length; i++) {
        if (L[i] !== FIRING_PIN_IMPRESSION) continue;
        n++;
        expect([img.data[i * 3], img.data[i * 3 + 1], img.data[i * 3 + 2]]).toEqual([...PURPLE]);
      }
      expect(n).toBeGreaterThan(0);
    } finally {
      setBriefColours(false);
    }
    expect(COLOURS[FIRING_PIN_IMPRESSION]).toEqual(LIGHT_BLUE);
    expect(COLOURS[FIRING_PIN_SHEAR]).toEqual(PURPLE);
    expect(res.labels!.data).toEqual(before);
  });

  it("offline demo writes outputs", () => {
    demo.main(["-o", tmp]);
    const truth = JSON.parse(readFileSync(join(tmp, "truth.json"), "utf8"));
    expect(new Set(Object.keys(truth))).toEqual(new Set(Object.keys(demo.CASES)));
    const masks = join(tmp, "masks");
    expect(existsSync(join(masks, "summary.csv")) && existsSync(join(masks, "contact_sheet.png"))).toBe(true);
    for (const name of Object.keys(demo.CASES)) {
      const summary = JSON.parse(readFileSync(join(masks, name, "masks.json"), "utf8"));
      expect(summary.status).toBe("ok");
      expect(summary.needs_review, String(summary.review_reasons)).toBe(false);
    }
  });
});
