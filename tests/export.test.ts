/**
 * X3P mask export (`src/export.ts`): the Python `test_x3p_mask_export` round trip, plus parity of
 * the decoded mask RGB and main.xml with the Python exporter when `reference/out/cli` exists
 * (`python reference/export_masks.py`).
 */
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { BACKGROUND, LEGEND, exportX3P, hexToBgr, maskImage, maskXml, readMask } from "../src/export.js";
import { decodePng, grayToBgr } from "../src/image.js";
import { BREECH_FACE, FIRING_PIN_IMPRESSION, segment } from "../src/segment.js";
import { makeHead } from "../src/synthetic.js";
import { readX3P, writeX3P } from "../src/x3p.js";
import { readFileSync } from "node:fs";
import { REF } from "./parity-utils.js";

const tmp = mkdtempSync(join(tmpdir(), "cartmask-export-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

describe("exportX3P", () => {
  it("embeds the mask at native resolution and keeps the scan", () => {
    const [surf] = makeHead({ sizeMm: 6.0, umPerPx: 10.0 });
    const src = join(tmp, "s.x3p");
    writeX3P(surf, src);
    const res = segment(readX3P(src));
    const dst = exportX3P(src, res, join(tmp, "s_masked.x3p"));
    const [img, xml] = readMask(dst);
    expect([img.height, img.width]).toEqual([surf.z.rows, surf.z.cols]); // native resolution
    expect(xml).toContain("<Mask><Background>");
    expect(xml).toContain(LEGEND[BREECH_FACE][1]);
    const back = readX3P(dst); // still a valid scan
    const orig = readX3P(src);
    expect(back.z.data.length).toBe(orig.z.data.length);
    for (let i = 0; i < orig.z.data.length; i++) {
      const a = orig.z.data[i], b = back.z.data[i];
      if (Number.isNaN(a)) expect(Number.isNaN(b)).toBe(true);
      else expect(b).toBe(a);
    }
    let red = 0;
    for (let i = 0; i < img.width * img.height; i++) {
      if (img.data[i * 3 + 2] === 255 && img.data[i * 3 + 1] === 0 && img.data[i * 3] === 0) red++;
    }
    const L = res.labels!.data;
    const bf = L.reduce((s, v) => s + (v === BREECH_FACE ? 1 : 0), 0) / L.length;
    expect(Math.abs(red / (img.width * img.height) - bf)).toBeLessThan(0.01);
    // re-exporting replaces the old mask instead of adding a second one
    const again = exportX3P(dst, res, join(tmp, "s_masked2.x3p"));
    expect(readMask(again)[1].match(/<Mask>/g)!.length).toBe(1);
  });

  it("lists only the present labels, in label order", () => {
    expect(maskXml(new Set([0, FIRING_PIN_IMPRESSION, BREECH_FACE]))).toBe(
      `<Mask><Background>${BACKGROUND}</Background><Annotations>` +
        `<Annotation color="#ff0000">breech face impression</Annotation>` +
        `<Annotation color="#66ccff">firing pin impression</Annotation></Annotations></Mask>`,
    );
    expect(hexToBgr("#cd7f32")).toEqual([0x32, 0x7f, 0xcd]);
  });
});

const CLI_REF = join(REF, "cli");
const CASES = ["default", "round_seed1", "drag20", "inverted_seed2"].filter((n) =>
  existsSync(join(CLI_REF, n, "mask_rgb.png")) && existsSync(join(REF, "synthetic", `${n}.x3p`)),
);

describe.skipIf(CASES.length === 0)("Python parity of the exported mask (reference/out/cli)", () => {
  it.each(CASES)("%s: decoded mask RGB and main.xml equal Python's", (name) => {
    const src = join(REF, "synthetic", `${name}.x3p`);
    const res = segment(readX3P(src));
    const py = grayToBgr(decodePng(readFileSync(join(CLI_REF, name, "mask_rgb.png"))));
    const surf = readX3P(src);
    const ts = maskImage(res, [surf.z.rows, surf.z.cols]);
    expect([ts.width, ts.height]).toEqual([py.width, py.height]);
    let diff = 0;
    for (let i = 0; i < ts.data.length; i++) if (ts.data[i] !== py.data[i]) diff++;
    expect(diff).toBe(0);
    const dst = exportX3P(src, res, join(tmp, `${name}__masked.x3p`));
    const [img, xml] = readMask(dst);
    const [pyImg, pyXml] = readMask(join(CLI_REF, name, `${name}__masked.x3p`));
    expect(Buffer.from(img.data).equals(Buffer.from(pyImg.data))).toBe(true);
    expect(xml).toBe(pyXml);
  });
});
