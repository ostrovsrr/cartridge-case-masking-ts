/**
 * Build a self-contained HTML annotator for expert ground truth (port of `scripts/build_annotator.py`).
 *
 *     npx tsx scripts/build-annotator.ts                      # data/raw -> annotations/annotator.html
 *     npx tsx scripts/build-annotator.ts data/raw -o annotations/annotator.html --max-px 800
 *     npx tsx scripts/build-annotator.ts a.x3p b.x3p -o packet.html --blind-map private-map.json
 *
 * For every X3P in the folder the scan is segmented and rendered (shaded topography, as in the
 * rest of the pipeline). The primer close-up is embedded as a JPEG data URI, downscaled to at most
 * `--max-px` pixels. The annotation canvas itself always works on the *working-resolution label
 * grid* (the grid of `Result.labels`) inside the crop box, so an exported label image maps back
 * exactly: the export is a full-size uint8 label image with zeros outside the box.
 *
 * The automatic masks are embedded run-length encoded so the annotator can load them as an opt-in
 * starting point; the default is a blank canvas, and the export records which start was used.
 *
 * With `--blind-map` the page is a blinded packet: opaque case IDs in a shuffled order, the full
 * working grid rendered without running the segmentation, no source names or masks, plus an Ignore
 * class; the private map (source name, SHA-256, grid) is written separately for scoring with
 * `src/evaluate.ts --blind-map`. See docs/annotation.md for the protocol.
 */
import { randomInt, randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import { sha256File } from "../src/evaluate.js";
import { crop, encodeJpeg, fromGrid, resize, type Image } from "../src/image.js";
import { hillshade, resample } from "../src/preprocess.js";
import { DEFAULT_PARAMS, LABEL_NAMES, makeParams, segment, type Params } from "../src/segment.js";
import { pyRoundN } from "../src/steps/np.js";
import { COLOURS, cropBox, render } from "../src/visualize.js";
import { readX3P } from "../src/x3p.js";

export const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
export const TEMPLATE = join(dirname(fileURLToPath(import.meta.url)), "annotator_template.html");
const PLACEHOLDER = "/*__SCAN_DATA__*/";

/** Python `SystemExit(message)`. */
export class BuildExit extends Error {
  override name = "BuildExit";
}

/** Row-major run-length encoding: `[value, count, value, count, ...]`. */
export function rle(a: ArrayLike<number>): number[] {
  const out: number[] = [];
  for (let i = 0; i < a.length; ) {
    const v = a[i];
    let j = i + 1;
    while (j < a.length && a[j] === v) j++;
    out.push(v, j - i);
    i = j;
  }
  return out;
}

export function unrle(runs: readonly number[], rows: number, cols: number): Uint8Array {
  const out = new Uint8Array(rows * cols);
  let o = 0;
  for (let i = 0; i < runs.length; i += 2) {
    out.fill(runs[i], o, o + runs[i + 1]);
    o += runs[i + 1];
  }
  return out;
}

export function dataUri(img: Image, quality = 90): string {
  return "data:image/jpeg;base64," + Buffer.from(encodeJpeg(img, quality)).toString("base64");
}

/** Grey plane of an image (all channels are equal on a shaded relief). */
function grey(img: Image): Image {
  if (img.channels === 1) return img;
  const out = new Uint8Array(img.width * img.height);
  for (let i = 0; i < out.length; i++) out[i] = img.data[3 * i];
  return { width: img.width, height: img.height, channels: 1, data: out };
}

/** Scan entry of an assisted (non-blinded) page; null when the scan has no usable primer. */
export function scanEntry(path: string, params: Params, maxPx: number, quality: number): Record<string, unknown> | null {
  const res = segment(readX3P(path), params);
  if (res.status === "skipped" || res.status === "failed" || res.primerRadiusPx == null) {
    console.log(`  skip ${basename(path)}: status=${res.status} ${res.warnings.slice(0, 1).join("; ")}`);
    return null;
  }
  const [y0, y1, x0, x1] = cropBox(res);
  const base = grey(crop(render(res), y0, y1, x0, x1));
  const ch = base.height, cw = base.width;
  const s = Math.min(1.0, maxPx / Math.max(ch, cw));
  const img = s === 1.0 ? base : resize(base, Math.round(cw * s), Math.round(ch * s), "area");
  const labels = res.labels!;
  const pre = new Uint8Array((y1 - y0) * (x1 - x0));
  for (let y = y0; y < y1; y++) pre.set(labels.data.subarray(y * labels.cols + x0, y * labels.cols + x1), (y - y0) * (x1 - x0));
  return {
    name: res.name,
    source: basename(path),
    um_per_px: pyRoundN(res.umPerPx, 6),
    label_shape: [labels.rows, labels.cols],
    crop: [y0, y1, x0, x1],
    img: dataUri(img, quality),
    img_size: [img.width, img.height],
    img_scale: pyRoundN(s, 6),
    prefill: rle(pre),
    status: res.status,
  };
}

/** Render the complete working grid without invoking segmentation. */
export function blindedEntry(path: string, name: string, params: Params, maxPx: number, quality: number): Record<string, unknown> {
  const surface = resample(readX3P(path), params.workUmPerPx);
  let image = fromGrid(hillshade(surface.z, surface.dx));
  const h = image.height, w = image.width;
  const scale = Math.min(1.0, maxPx / Math.max(h, w));
  if (scale < 1) image = resize(image, Math.max(1, Math.round(w * scale)), Math.max(1, Math.round(h * scale)), "area");
  return {
    name,
    um_per_px: surface.dx * 1e6,
    label_shape: [h, w],
    crop: [0, h, 0, w],
    img: dataUri(image, quality),
    img_size: [image.width, image.height],
    img_scale: scale,
  };
}

/** Python `json.dumps` default `ensure_ascii`: non-ASCII characters as `\uXXXX`. */
function asciiJson(value: unknown, indent?: number): string {
  return JSON.stringify(value, null, indent).replace(/[\u0080-￿]/g, (c) => "\\u" + c.charCodeAt(0).toString(16).padStart(4, "0"));
}

function sortedX3p(dir: string): string[] {
  const isWin = process.platform === "win32";
  return readdirSync(dir)
    .filter((n) => (isWin ? n.toLowerCase() : n).endsWith(".x3p"))
    .map((n) => join(dir, n))
    .sort((a, b) => {
      const x = isWin ? a.toLowerCase() : a, y = isWin ? b.toLowerCase() : b;
      return x < y ? -1 : x > y ? 1 : 0;
    });
}

function localTimestamp(d = new Date()): string {
  const p = (v: number): string => String(v).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

/**
 * Build the annotator page `out` for the X3P files / folders in `inputs`.
 *
 * @param blindMap Path of a new private scoring map; when given, the page is a blinded packet.
 * @returns The embedded payload (`classes`, `scans`, `packet`, ...).
 * @throws BuildExit when there is nothing to annotate or the map path is unusable.
 */
export function build(
  inputs: string[],
  out: string,
  maxPx = 800,
  quality = 90,
  params: Partial<Params> | null = null,
  blindMap: string | null = null,
): Record<string, unknown> & { scans: Record<string, unknown>[]; packet: string | null } {
  const p = makeParams(params ?? {});
  let files: string[] = [];
  for (const i of inputs) files.push(...(existsSync(i) && statSync(i).isDirectory() ? sortedX3p(i) : [i]));
  if (!files.length) throw new BuildExit(`no .x3p files in ${inputs.join(", ")}`);
  if (blindMap && (resolve(blindMap) === resolve(out) || existsSync(blindMap))) {
    throw new BuildExit("choose a new private map path distinct from the annotator");
  }
  const packet = blindMap ? randomUUID().replace(/-/g, "") : null;
  const mapping: Record<string, { source: string; sha256: string; label_shape: unknown; um_per_px: unknown }> = {};
  if (blindMap) {
    files = files.slice();
    for (let i = files.length - 1; i > 0; i--) {
      const j = randomInt(i + 1);
      [files[i], files[j]] = [files[j], files[i]];
    }
  }
  const scans: Record<string, unknown>[] = [];
  for (const f of files) {
    const t0 = performance.now();
    let e: Record<string, unknown> | null;
    if (blindMap) {
      const digest = sha256File(f);
      if (Object.values(mapping).some((row) => row.sha256 === digest)) continue;
      const name = `case-${String(scans.length + 1).padStart(3, "0")}`;
      e = blindedEntry(f, name, p, maxPx, quality);
      mapping[name] = { source: basename(f), sha256: digest, label_shape: e.label_shape, um_per_px: e.um_per_px };
    } else e = scanEntry(f, p, maxPx, quality);
    if (e) {
      scans.push(e);
      const c = e.crop as number[];
      console.log(`  ${e.name}: grid ${c[1] - c[0]}x${c[3] - c[2]} px (${((performance.now() - t0) / 1000).toFixed(1)} s)`);
    }
  }
  if (!scans.length) throw new BuildExit("no usable primer scans to annotate");
  const classes: { value: number; name: string; colour: string }[] = [];
  for (const [k, name] of Object.entries(LABEL_NAMES)) {
    const [b, g, r] = COLOURS[Number(k)];
    const hex = (v: number): string => v.toString(16).padStart(2, "0");
    classes.push({ value: Number(k), name, colour: `#${hex(r)}${hex(g)}${hex(b)}` });
  }
  if (blindMap) classes.push({ value: 255, name: "Ignore", colour: "#FAC414" });
  const payload = { version: 1, built: localTimestamp(), classes, scans, work_um_per_px: p.workUmPerPx, packet };
  const js = asciiJson(payload).replaceAll("</", "<\\/");
  const html = readFileSync(TEMPLATE, "utf8");
  if (!html.includes(PLACEHOLDER)) throw new BuildExit(`placeholder ${PLACEHOLDER} missing from ${TEMPLATE}`);
  mkdirSync(dirname(resolve(out)), { recursive: true });
  if (blindMap) {
    mkdirSync(dirname(resolve(blindMap)), { recursive: true });
    writeFileSync(blindMap, asciiJson({ packet, scans: mapping }, 2), { encoding: "utf8", flag: "wx" });
  }
  writeFileSync(out, html.replace(PLACEHOLDER, () => js), "utf8");
  console.log(`wrote ${out} (${(statSync(out).size / 1e6).toFixed(2)} MB, ${scans.length} scans)`);
  return payload;
}

const USAGE =
  "usage: npx tsx scripts/build-annotator.ts [-h] [-o OUT] [--max-px MAX_PX] [--jpeg-quality JPEG_QUALITY] " +
  "[--um-per-px UM_PER_PX] [--blind-map BLIND_MAP] [inputs ...]";

/** Command-line entry point; returns the exit status. */
export function main(argv: string[] = process.argv.slice(2)): number {
  const fail = (msg: string): number => {
    console.error(`${USAGE}\nbuild-annotator: error: ${msg}`);
    return 2;
  };
  let parsed;
  try {
    parsed = parseArgs({
      args: argv,
      options: {
        out: { type: "string", short: "o", default: join(ROOT, "annotations", "annotator.html") },
        "max-px": { type: "string", default: "800" },
        "jpeg-quality": { type: "string", default: "90" },
        "um-per-px": { type: "string", default: String(DEFAULT_PARAMS.workUmPerPx) },
        "blind-map": { type: "string" },
        help: { type: "boolean", short: "h" },
      },
      allowPositionals: true,
      strict: true,
    });
  } catch (e) {
    return fail((e as Error).message);
  }
  const v = parsed.values;
  if (v.help) {
    console.log(`${USAGE}\n\nBuild a self-contained HTML annotator for expert ground truth.`);
    return 0;
  }
  const maxPx = Number(v["max-px"]), q = Number(v["jpeg-quality"]), um = Number(v["um-per-px"]);
  if (!Number.isInteger(maxPx) || !Number.isInteger(q)) return fail("--max-px and --jpeg-quality must be integers");
  if (maxPx < 1 || !(q >= 0 && q <= 100)) return fail("--max-px must be positive and --jpeg-quality must be between 0 and 100");
  if (!Number.isFinite(um) || um <= 0) return fail("--um-per-px must be positive and finite");
  const inputs = parsed.positionals.length ? parsed.positionals : [join(ROOT, "data", "raw")];
  try {
    build(inputs, v.out!, maxPx, q, { workUmPerPx: um }, v["blind-map"] ?? null);
  } catch (e) {
    if (e instanceof BuildExit) {
      console.error(e.message);
      return 1;
    }
    throw e;
  }
  return 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  process.exitCode = main();
}
