/** Helpers for the Python parity tests: reference loading, PNG decoding and label comparison. */
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { unzlibSync } from "fflate";
import type { Grid } from "../src/ndimage/index.js";
import type { Result } from "../src/segment.js";

export const REF = join(dirname(fileURLToPath(import.meta.url)), "..", "reference", "out");
export const HOW_TO =
  "Python parity reference missing: run `python reference/export_labels.py --expanded` " +
  "(needs CARTMASK_PY or ../cartridge-case and CARTMASK_DATA) to create reference/out/.";

export interface ManifestEntry {
  name: string;
  x3p?: string;
  params?: Record<string, unknown>;
  file?: string;
  study?: string;
  set?: "baseline" | "expanded";
  labels?: string;
  json?: string;
}
export interface Manifest {
  python_repo: string;
  data: string;
  synthetic?: ManifestEntry[];
  real?: ManifestEntry[];
}

export function loadManifest(): Manifest | null {
  const p = join(REF, "manifest.json");
  if (!existsSync(p)) return null;
  return JSON.parse(readFileSync(p, "utf8")) as Manifest;
}

/** Decode an 8-bit greyscale (or the first channel of an 8-bit RGB/RGBA) non-interlaced PNG. */
export function decodePngGray(bytes: Uint8Array): Grid<Uint8Array> {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let off = 8;
  let w = 0, h = 0, bitDepth = 0, colorType = 0, interlace = 0;
  const idat: Uint8Array[] = [];
  while (off < bytes.length) {
    const len = dv.getUint32(off);
    const type = String.fromCharCode(...bytes.subarray(off + 4, off + 8));
    const data = bytes.subarray(off + 8, off + 8 + len);
    if (type === "IHDR") {
      w = dv.getUint32(off + 8);
      h = dv.getUint32(off + 12);
      bitDepth = data[8];
      colorType = data[9];
      interlace = data[12];
    } else if (type === "IDAT") idat.push(data);
    else if (type === "IEND") break;
    off += 12 + len;
  }
  if (bitDepth !== 8 || interlace !== 0) throw new Error(`unsupported PNG (bit depth ${bitDepth}, interlace ${interlace})`);
  const bpp = colorType === 0 ? 1 : colorType === 2 ? 3 : colorType === 4 ? 2 : colorType === 6 ? 4 : 0;
  if (!bpp) throw new Error(`unsupported PNG colour type ${colorType}`);
  let total = 0;
  for (const c of idat) total += c.length;
  const z = new Uint8Array(total);
  for (let i = 0, o = 0; i < idat.length; o += idat[i].length, i++) z.set(idat[i], o);
  const raw = unzlibSync(z);
  const stride = w * bpp;
  const out = new Uint8Array(w * h);
  let prev = new Uint8Array(stride);
  for (let y = 0; y < h; y++) {
    const ft = raw[y * (stride + 1)];
    const line = raw.slice(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let x = 0; x < stride; x++) {
      const a = x >= bpp ? line[x - bpp] : 0;
      const b = prev[x];
      const c = x >= bpp ? prev[x - bpp] : 0;
      let v = line[x];
      if (ft === 1) v += a;
      else if (ft === 2) v += b;
      else if (ft === 3) v += (a + b) >> 1;
      else if (ft === 4) {
        const pp = a + b - c;
        const pa = Math.abs(pp - a), pb = Math.abs(pp - b), pc = Math.abs(pp - c);
        v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      }
      line[x] = v & 255;
    }
    for (let x = 0; x < w; x++) out[y * w + x] = line[x * bpp];
    prev = line;
  }
  return { data: out, rows: h, cols: w };
}

export function loadLabels(rel: string): Grid<Uint8Array> {
  return decodePngGray(readFileSync(join(REF, rel)));
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function loadSummary(rel: string): any {
  return JSON.parse(readFileSync(join(REF, rel), "utf8"));
}

/** Per-class IoU (classes 1..4); null where the class is absent from both. */
export function classIoU(a: Grid<Uint8Array>, b: Grid<Uint8Array>): (number | null)[] {
  const inter = new Float64Array(5), uni = new Float64Array(5);
  for (let i = 0; i < a.data.length; i++) {
    const x = a.data[i], y = b.data[i];
    if (x === y) {
      inter[x]++;
      uni[x]++;
    } else {
      uni[x]++;
      uni[y]++;
    }
  }
  const out: (number | null)[] = [];
  for (let k = 1; k <= 4; k++) out.push(uni[k] === 0 ? null : inter[k] / uni[k]);
  return out;
}

export function angleDiff(a: number, b: number): number {
  return Math.abs((((a - b + 180) % 360) + 360) % 360 - 180);
}

const dist = (a: number[] | null, b: number[] | null): number =>
  a && b ? Math.hypot(a[0] - b[0], a[1] - b[1]) : a === b ? 0 : Infinity;

const absd = (a: number | null, b: number | null): number =>
  a !== null && b !== null ? Math.abs(a - b) : a === b ? 0 : Infinity;

export interface Comparison {
  name: string;
  study: string;
  shapeOk: boolean;
  iou: (number | null)[];
  primerCentre: number;
  primerRadius: number;
  pinCentre: number;
  pinRadius: number;
  apertureRadius: number;
  dragAngle: number;
  dragAgree: boolean;
  qc: number;
  reviewAgree: boolean;
  reasonsAgree: boolean;
  statusAgree: boolean;
  tsReasons: string[];
  pyReasons: string[];
  seconds: number;
  pySeconds: number;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function compare(name: string, study: string, res: Result, py: any, pyLabels: Grid<Uint8Array>, seconds: number): Comparison {
  const ts = res.summary();
  const labels = res.labels!;
  const shapeOk = labels.rows === pyLabels.rows && labels.cols === pyLabels.cols;
  return {
    name,
    study,
    shapeOk,
    iou: shapeOk ? classIoU(labels, pyLabels) : [0, 0, 0, 0],
    primerCentre: dist(ts.primer_center_px as number[] | null, py.primer_center_px),
    primerRadius: absd(ts.primer_radius_px as number | null, py.primer_radius_px),
    pinCentre: dist(ts.fp_core_center_px as number[] | null, py.fp_core_center_px),
    pinRadius: absd(ts.fp_core_radius_px as number | null, py.fp_core_radius_px),
    apertureRadius: absd(ts.aperture_radius_px as number | null, py.aperture_radius_px),
    dragAngle:
      ts.drag_angle_deg !== null && py.drag_angle_deg !== null
        ? angleDiff(ts.drag_angle_deg as number, py.drag_angle_deg)
        : ts.drag_angle_deg === py.drag_angle_deg
          ? 0
          : Infinity,
    dragAgree: ts.drag_detected === py.drag_detected,
    qc: Math.abs((ts.qc_score as number) - py.qc_score),
    reviewAgree: ts.needs_review === py.needs_review,
    reasonsAgree: JSON.stringify(ts.review_reasons) === JSON.stringify(py.review_reasons),
    statusAgree: ts.status === py.status,
    tsReasons: ts.review_reasons as string[],
    pyReasons: py.review_reasons,
    seconds,
    pySeconds: py.seconds,
  };
}

export function fmtIoU(v: number | null): string {
  return v === null ? "  -   " : v.toFixed(4);
}
