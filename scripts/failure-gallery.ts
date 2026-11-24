/**
 * Build the failure gallery in samples/failures from two mask runs.
 *
 * Port of `scripts/failure_gallery.py`:
 *
 *     tsx scripts/compare-revisions.ts run --code <older checkout> --out outputs/gallery/before
 *     tsx scripts/compare-revisions.ts run --out outputs/gallery/after
 *     tsx scripts/failure-gallery.ts outputs/gallery/before outputs/gallery/after -o samples/failures
 *
 * Each case shows the rendered head with the detected primer circle (yellow) and the primer
 * close-up with the masks, strike circle and drag arrow. Cases that a release changed show the
 * earlier revision ("before") next to the current one ("after"); open failures show the current
 * masks only. The "before" panels of the committed gallery are Python commit 9bce6ec, the
 * revision those fixes were measured against; `--before-label` names the run you pass.
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import { decodePng, encodeJpeg, type Image } from "../src/image.js";
import { ROOT } from "./fetch-nbtrd.js";
import { galleryBlock, galleryMosaic, galleryPanels } from "./lib/figures.js";
import { loadNpz, npyGrid } from "./lib/npz.js";

/** `[run key, title, before/after?, caption]`; the captions describe the committed runs. */
export const CASES: [string, string, boolean, string][] = [
  ["matching__9mm__E3__G0232_A0006.001_CC03_TF", "Primer mislocalised (fixed)", true,
    "Before: 3.26 mm circle on the headstamp, QC 0.78. After: 4.15 mm primer, concentric with the head."],
  ["expanded__9mm__E3__G0242_A0006.001_CC01_TF", "Primer mislocalised (fixed)", true,
    "Before: 3.23 mm circle on the headstamp, QC 1.00. After: 4.03 mm primer."],
  ["expanded__9mm__E3__G0243_A0006.001_CC01_TF", "Primer mislocalised (fixed)", true,
    "Before: 3.14 mm circle on the headstamp letters. After: 4.06 mm primer."],
  ["expanded__9mm__E3__G0318_A0007.001_CC01_TF", "False polarity flip (now flagged, still wrong)", true,
    "Before: flipped, 3.67 mm 'crater', zero breech face, not flagged. After: not flipped, flagged; crater not separated."],
  ["expanded__9mm__E3__G0006_A0003.001_CC01_TF", "False polarity flip (now flagged, still wrong)", true,
    "Before: flipped, 2.70 mm 'crater'. After: not flipped, flagged; crater on flow-back plateau not separated."],
  ["glock__9mm__PopStatGlock__CWS111US_G34_RP1", "Rectangular aperture (improved, still flagged)", true,
    "Before: disk aperture, impression leaked below the slot, false drag. After: 1.38 mm slot, impression kept inside it; a short drag remains; crater unmeasured."],
  ["expanded__9mm__PopStatGlock__HCIFS_15-2769_RP1", "Unmeasured crater (flagged)", false,
    "Confocal scan leaves the steep crater unmeasured; the data gap fills 86% of the primer face."],
  ["matching__9mm__Fadul__Fadul_1-1", "Unmeasured crater (flagged)", false,
    "Primer is the data edge; crater outline is the no-data hole; the aperture-rim step is no longer called shear."],
  ["expanded__9mm__Weller__TW04-01", "False aperture shear (not flagged)", false,
    "The shallow drag channel at 3 o'clock is not detected as drag; its walls pass as shear, and the striated band at 9 o'clock is missed."],
];

/** Rendering, labels and summary of one scan of a `compare-revisions run`, as gallery panels. */
export function panels(run: string, key: string): [Image, Image] {
  const s = JSON.parse(readFileSync(join(run, `${key}.json`), "utf8"));
  const png = decodePng(readFileSync(join(run, `${key}.png`)));
  const base: Image = png.channels === 1 ? png : { ...png, channels: 1, data: png.data.filter((_, i) => i % 3 === 0) };
  const lab = npyGrid<Uint8Array>(loadNpz(join(run, `${key}.npz`))["labels"]!);
  return galleryPanels(base, lab, s);
}

export function main(argv = process.argv.slice(2)): number {
  const { values, positionals } = parseArgs({
    args: argv,
    options: {
      out: { type: "string", short: "o", default: join(ROOT, "samples/failures") },
      "before-label": { type: "string", default: "before (9bce6ec)" },
      help: { type: "boolean", short: "h", default: false },
    },
    allowPositionals: true,
  });
  if (values.help || positionals.length !== 2) {
    console.log("usage: failure-gallery BEFORE AFTER [-o OUT] [--before-label TEXT]");
    return values.help ? 0 : 2;
  }
  const [before, after] = positionals as [string, string];
  const out = values.out!;
  mkdirSync(out, { recursive: true });
  const blocks: Image[] = [];
  for (const [key, title, pair, caption] of CASES) {
    const b = galleryBlock(pair ? panels(before, key) : null, panels(after, key), title, key, caption, values["before-label"]!);
    const parts = key.split("__");
    writeFileSync(join(out, `${parts[2]}__${parts[3]}.jpg`), encodeJpeg(b, 85));
    blocks.push(b);
  }
  writeFileSync(join(out, "failure_gallery.jpg"), encodeJpeg(galleryMosaic(blocks), 85));
  console.log(`wrote ${blocks.length} cases to ${out}`);
  return 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  process.exit(main());
}
