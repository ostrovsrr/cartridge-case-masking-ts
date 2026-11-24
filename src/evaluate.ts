/**
 * Compare automatic masks with human annotations (port of `cartmask/evaluate.py`).
 *
 *     npx tsx src/evaluate.ts --pred-from data/raw --truth annotations/ -o outputs/evaluation
 *
 * Annotations are label images exported by the offline annotator (built by
 * `scripts/build-annotator.ts`): `<scan>__labels.png` is a uint8 image at the working-resolution
 * label grid (0 background, 1 breech face, 2 aperture shear, 3 firing-pin shear, 4 firing-pin
 * impression, 255 Ignore) and `<scan>__labels.json` is its sidecar (annotator, timestamp,
 * prefilled flag, packet ID for blinded packets). The truth folder is searched recursively, so
 * several examiners can keep their exports in sub-folders (`annotations/<name>/...`).
 *
 * Per class the script reports IoU, Dice, precision, recall and a boundary F-score (boundary
 * pixels count as matched within a tolerance, default 16 um = 2 px at 8 um/px), plus primer-centre
 * and pin-centre errors in um (centroids of the annotated regions vs. the geometry the pipeline
 * reports). When the same scan was annotated by more than one person, the human-vs-human
 * agreement is reported too, so algorithm-vs-human can be read against it.
 *
 * Conventions for empty classes (no NaN, no crash):
 *   - absent in both prediction and truth -> every score is 1.0 and the row is flagged
 *     `absent_in_both`; summary means leave such rows out;
 *   - present in only one of them -> IoU, Dice, precision, recall and boundary F are all 0.0.
 *
 * Metric records, rows and summary rows keep the Python dictionary keys (snake_case), so the
 * outputs can be compared with the Python reference one to one (`tests/evaluate.test.ts`).
 */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, extname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import {
  crop as cropImage,
  decodePng,
  draw,
  imwrite,
  newImage,
  paste,
  putText,
  rectangle,
  resize,
  vstack,
  type BGR,
  type Image,
} from "./image.js";
import { binaryFillHoles, edt, pairwiseSum, type Grid, type Mask } from "./ndimage/index.js";
import {
  APERTURE_SHEAR,
  BREECH_FACE,
  DEFAULT_PARAMS,
  FIRING_PIN_IMPRESSION,
  FIRING_PIN_SHEAR,
  LABEL_NAMES,
  makeParams,
  segment,
  type Params,
  type Result,
} from "./segment.js";
import { fmtF, fmtG, hypot, pyRoundN } from "./steps/np.js";
import { cropBox, render } from "./visualize.js";
import { readX3P } from "./x3p.js";

export const CLASSES = [BREECH_FACE, APERTURE_SHEAR, FIRING_PIN_SHEAR, FIRING_PIN_IMPRESSION] as const;
export const METRICS = ["iou", "dice", "precision", "recall", "bf_score"] as const;
export type MetricName = (typeof METRICS)[number];
export const DEFAULT_TOL_UM = 16.0;
export const IGNORE = 255;

// BGR colours of the error overlays (chosen to differ from the class legend)
export const TP_COLOUR: BGR = [60, 200, 60]; // agreement: green
export const FP_COLOUR: BGR = [220, 60, 220]; // algorithm only (false positive): magenta
export const FN_COLOUR: BGR = [240, 200, 0]; // annotator only (false negative): cyan

const IS_WIN = process.platform === "win32";

/** Python `ValueError` / `IOError` raised for bad inputs; `run` records its message as a skip reason. */
export class EvaluationError extends Error {
  override name = "EvaluationError";
}

/** Python `SystemExit(message)`: the CLI prints the message and exits with status 1. */
export class EvaluationExit extends Error {
  override name = "EvaluationExit";
}

type G = Grid<ArrayLike<number>>;

/** Per-class scores of one prediction/truth pair (Python `mask_metrics` dict). */
export interface MaskMetrics {
  tp: number;
  fp: number;
  fn: number;
  pred_px: number;
  truth_px: number;
  absent_in_both: boolean;
  iou: number;
  dice: number;
  precision: number;
  recall: number;
  bf_score: number;
}

/** Output of {@link evaluate}; centre errors are null when either side lacks the region. */
export interface Evaluation {
  classes: Record<number, MaskMetrics>;
  tol_px: number;
  primer_center_err_um: number | null;
  pin_center_err_um: number | null;
}

// --------------------------------------------------------------------------- metrics

function counts(pred: G, truth: G): [number, number, number] {
  const p = pred.data, t = truth.data;
  let tp = 0, fp = 0, fn = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i]) {
      if (t[i]) tp++;
      else fp++;
    } else if (t[i]) fn++;
  }
  return [tp, fp, fn];
}

function countOf(m: G): number {
  let n = 0;
  for (let i = 0; i < m.data.length; i++) if (m.data[i]) n++;
  return n;
}

function maskWhere(rows: number, cols: number, fn: (i: number) => boolean): Mask {
  const out = new Uint8Array(rows * cols);
  for (let i = 0; i < out.length; i++) out[i] = fn(i) ? 1 : 0;
  return { data: out, rows, cols };
}

function shapeStr(g: { rows: number; cols: number }): string {
  return `(${g.rows}, ${g.cols})`;
}

/** Inner boundary: mask pixels with a 4-neighbour outside (image edge counts as outside). */
export function boundary(mask: G): Mask {
  const { rows, cols, data } = mask;
  const out = new Uint8Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      if (!data[i]) continue;
      const inner = r > 0 && r < rows - 1 && c > 0 && c < cols - 1 && data[i - 1] && data[i + 1] && data[i - cols] && data[i + cols];
      out[i] = inner ? 0 : 1;
    }
  }
  return { data: out, rows, cols };
}

/**
 * Boundary precision, recall and F-score (Csurka et al. 2013 style).
 *
 * A predicted boundary pixel is correct when a true boundary pixel lies within `tolPx`
 * (Euclidean distance), and vice versa for recall. Only boundary pixels inside `domain` count.
 */
export function boundaryF(pred: G, truth: G, tolPx = 2.0, domain: G | null = null): [number, number, number] {
  const bp = boundary(pred), bt = boundary(truth);
  if (domain) {
    for (let i = 0; i < bp.data.length; i++) {
      if (!domain.data[i]) bp.data[i] = bt.data[i] = 0;
    }
  }
  const np_ = countOf(bp), nt = countOf(bt);
  if (np_ === 0 && nt === 0) return [1.0, 1.0, 1.0];
  if (np_ === 0 || nt === 0) return [0.0, 0.0, 0.0];
  const notBt = maskWhere(bt.rows, bt.cols, (i) => !bt.data[i]);
  const notBp = maskWhere(bp.rows, bp.cols, (i) => !bp.data[i]);
  const dtTruth = edt(notBt).data, dtPred = edt(notBp).data;
  const lim = tolPx + 1e-9;
  let hp = 0, hr = 0;
  for (let i = 0; i < bp.data.length; i++) {
    if (bp.data[i] && dtTruth[i] <= lim) hp++;
    if (bt.data[i] && dtPred[i] <= lim) hr++;
  }
  const p = hp / np_, r = hr / nt;
  const f = p + r === 0 ? 0.0 : (2 * p * r) / (p + r);
  return [p, r, f];
}

/** IoU, Dice, precision, recall and boundary F of two binary masks (never NaN). */
export function maskMetrics(pred: G, truth: G, tolPx = 2.0, boundaryDomain: G | null = null): MaskMetrics {
  const [tp, fp, fn] = counts(pred, truth);
  const nPred = tp + fp, nTruth = tp + fn;
  const absent = nPred === 0 && nTruth === 0;
  const base = { tp, fp, fn, pred_px: nPred, truth_px: nTruth, absent_in_both: absent };
  if (absent) return { ...base, iou: 1.0, dice: 1.0, precision: 1.0, recall: 1.0, bf_score: 1.0 };
  return {
    ...base,
    iou: tp / (tp + fp + fn),
    dice: (2 * tp) / (2 * tp + fp + fn),
    precision: nPred ? tp / nPred : 0.0,
    recall: nTruth ? tp / nTruth : 0.0,
    bf_score: boundaryF(pred, truth, tolPx, boundaryDomain)[2],
  };
}

function centroid(mask: G): [number, number] | null {
  const { rows, cols, data } = mask;
  let sx = 0, sy = 0, n = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (data[r * cols + c]) {
        sx += c;
        sy += r;
        n++;
      }
    }
  }
  return n ? [sx / n, sy / n] : null;
}

/** Primer face as annotated: union of all classes, holes filled. */
export function annotatedPrimer(labels: G): Mask {
  return binaryFillHoles(maskWhere(labels.rows, labels.cols, (i) => labels.data[i] > 0));
}

/** Per-class metrics between two label images of the same shape (Ignore pixels excluded). */
export function compareLabels(pred: G, truth: G, umPerPx: number, tolUm = DEFAULT_TOL_UM): Record<number, MaskMetrics> {
  if (pred.rows !== truth.rows || pred.cols !== truth.cols) {
    throw new EvaluationError(`label shapes differ: ${shapeStr(pred)} vs ${shapeStr(truth)}`);
  }
  const tolPx = tolUm / umPerPx;
  const { rows, cols } = pred;
  const p = pred.data, t = truth.data;
  const valid = maskWhere(rows, cols, (i) => p[i] !== IGNORE && t[i] !== IGNORE);
  let domain: Mask | null = null;
  if (countOf(valid) !== valid.data.length) {
    const d = edt(valid).data;
    const lim = Math.max(1, tolPx);
    domain = maskWhere(rows, cols, (i) => d[i] > lim);
  }
  const out: Record<number, MaskMetrics> = {};
  for (const k of CLASSES) {
    const pm = maskWhere(rows, cols, (i) => p[i] === k && valid.data[i] === 1);
    const tm = maskWhere(rows, cols, (i) => t[i] === k && valid.data[i] === 1);
    out[k] = maskMetrics(pm, tm, tolPx, domain);
  }
  return out;
}

/**
 * Compare one segmentation result with one annotated label image.
 *
 * Centre errors are null when either side lacks the region, and whenever the annotation marks
 * pixels as Ignore (an incomplete drawing has no exact centroid).
 *
 * @throws EvaluationError when the shapes differ or every pixel is Ignore.
 */
export function evaluate(result: Result, truth: G, tolUm = DEFAULT_TOL_UM): Evaluation {
  const labels = result.labels;
  if (labels == null || labels.rows !== truth.rows || labels.cols !== truth.cols) {
    throw new EvaluationError(
      `annotation shape ${shapeStr(truth)} does not match the label grid ${labels == null ? "None" : shapeStr(labels)}; ` +
        "was it exported at a different working resolution?",
    );
  }
  const um = result.umPerPx;
  let nIgnore = 0;
  for (let i = 0; i < truth.data.length; i++) if (truth.data[i] === IGNORE) nIgnore++;
  if (nIgnore === truth.data.length) throw new EvaluationError("all label pixels were marked Ignore");
  const classes = compareLabels(labels, truth, um, tolUm);
  const err = (cTruth: [number, number] | null, cPred: [number, number] | null): number | null =>
    cTruth === null || cPred === null ? null : hypot(cTruth[0] - cPred[0], cTruth[1] - cPred[1]) * um;
  const complete = nIgnore === 0;
  const pin = maskWhere(truth.rows, truth.cols, (i) => truth.data[i] === FIRING_PIN_IMPRESSION);
  return {
    classes,
    tol_px: tolUm / um,
    primer_center_err_um: complete ? err(centroid(annotatedPrimer(truth)), result.primerCenterPx) : null,
    pin_center_err_um: complete ? err(centroid(pin), result.fpCoreCenterPx) : null,
  };
}

// --------------------------------------------------------------------------- I/O

const PNG_SIGNATURE = [137, 80, 78, 71, 13, 10, 26, 10];

/**
 * Read a label PNG; colour PNGs with equal channels are accepted.
 *
 * @throws EvaluationError for unreadable files, colour images and values outside 0-4 / 255.
 */
export function readLabels(path: string): Grid<Uint8Array> {
  const bytes = new Uint8Array(readFileSync(path));
  let img: Image;
  try {
    if (bytes.length < 8 || PNG_SIGNATURE.some((v, i) => bytes[i] !== v)) throw new Error("not a PNG");
    img = decodePng(bytes);
  } catch {
    throw new EvaluationError(`cannot read ${path}`);
  }
  const n = img.width * img.height;
  let data: Uint8Array;
  if (img.channels === 3) {
    data = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const b = img.data[3 * i], g = img.data[3 * i + 1], r = img.data[3 * i + 2];
      if (b !== g || g !== r) throw new EvaluationError(`${basename(path)} is a colour image, expected a uint8 label image`);
      data[i] = b;
    }
  } else data = img.data;
  let bad = 0;
  for (let i = 0; i < n; i++) {
    const v = data[i];
    if (v !== IGNORE && v > FIRING_PIN_IMPRESSION) bad++;
  }
  if (bad) throw new EvaluationError(`${basename(path)}: ${bad} pixels are outside labels 0–4 and Ignore (255)`);
  return { data, rows: img.height, cols: img.width };
}

/** Python truthiness of a JSON value. */
function truthy(v: unknown): boolean {
  if (v === null || v === undefined || v === false || v === 0 || v === "") return false;
  if (Array.isArray(v)) return v.length > 0;
  if (typeof v === "object") return Object.keys(v as object).length > 0;
  return true;
}

/** Python `str()` of a JSON value. */
function pyStrValue(v: unknown): string {
  if (typeof v === "string") return v;
  if (v === null || v === undefined) return "None";
  if (v === true) return "True";
  if (v === false) return "False";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : pyRepr(v);
  return JSON.stringify(v);
}

/** `str.casefold()` (close enough: full case folding of the common cases). */
function casefold(s: string): string {
  return s.toUpperCase().toLowerCase();
}

/** Python `repr(float)`: shortest round-trip digits, exponent outside [1e-4, 1e16). */
export function pyRepr(x: number): string {
  if (Number.isNaN(x)) return "nan";
  if (!Number.isFinite(x)) return x > 0 ? "inf" : "-inf";
  if (x === 0) return Object.is(x, -0) ? "-0.0" : "0.0";
  const [mant, expS] = x.toExponential().split("e");
  const exp = Number(expS);
  const neg = mant.startsWith("-");
  const digits = mant.replace("-", "").replace(".", "");
  let s: string;
  if (exp < -4 || exp >= 16) {
    s = digits[0] + (digits.length > 1 ? "." + digits.slice(1) : "") + "e" + (exp < 0 ? "-" : "+") + String(Math.abs(exp)).padStart(2, "0");
  } else if (exp < 0) {
    s = "0." + "0".repeat(-exp - 1) + digits;
  } else {
    const frac = digits.slice(exp + 1);
    s = digits.slice(0, exp + 1).padEnd(exp + 1, "0") + "." + (frac || "0");
  }
  return (neg ? "-" : "") + s;
}

/** Python `float(v)` of a JSON value. */
function pyFloat(v: unknown): number {
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  if (typeof v === "string") {
    const t = v.trim().replace(/(?<=\d)_(?=\d)/g, "");
    if (/^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/i.test(t)) return Number(t);
    if (/^[+-]?(inf|infinity)$/i.test(t)) return t.startsWith("-") ? -Infinity : Infinity;
    if (/^[+-]?nan$/i.test(t)) return NaN;
    throw new EvaluationError(`could not convert string to float: '${v}'`);
  }
  const type = v === null || v === undefined ? "NoneType" : Array.isArray(v) ? "list" : "dict";
  throw new EvaluationError(`float() argument must be a string or a real number, not '${type}'`);
}

/** pathlib's ordering of paths (`sorted(Path.rglob(...))`): by parts, case-folded on Windows. */
function comparePaths(a: string, b: string): number {
  const norm = (p: string): string[] => (IS_WIN ? p.toLowerCase() : p).split(/[\\/]+/);
  const pa = norm(a), pb = norm(b);
  for (let i = 0; i < Math.min(pa.length, pb.length); i++) {
    if (pa[i] !== pb[i]) return pa[i] < pb[i] ? -1 : 1;
  }
  return pa.length - pb.length;
}

function listDir(dir: string): { name: string; isDir: boolean }[] {
  try {
    return readdirSync(dir, { withFileTypes: true }).map((d) => ({ name: d.name, isDir: d.isDirectory() }));
  } catch {
    return [];
  }
}

/**
 * `Path(root).rglob(pattern)` for a filename predicate, in the order Python 3.12 yields matches:
 * directories in `Path.walk` order (each directory's children listed when it is visited), the
 * matching entries of every directory in listing order; symlinked directories are not followed.
 */
function rglob(root: string, match: (name: string) => boolean): string[] {
  const dirs = [root];
  const walk = (d: string): void => {
    const subs = listDir(d).filter((e) => e.isDir).map((e) => join(d, e.name));
    dirs.push(...subs);
    for (const s of subs) walk(s);
  };
  walk(root);
  const out: string[] = [];
  for (const d of dirs) for (const e of listDir(d)) if (match(e.name)) out.push(join(d, e.name));
  return out;
}

const nameEq = (a: string, b: string): boolean => (IS_WIN ? a.toLowerCase() === b.toLowerCase() : a === b);
const endsWithName = (name: string, suffix: string): boolean =>
  IS_WIN ? name.toLowerCase().endsWith(suffix.toLowerCase()) : name.endsWith(suffix);

/** One exported annotation: label PNG, sidecar metadata and the annotator it is attributed to. */
export interface Annotation {
  /** Scan ID: the sidecar's `scan` when present (any JSON value), else the file-name stem. */
  scan: unknown;
  path: string;
  meta: Record<string, unknown>;
  annotator: string;
  prefilled: boolean;
}

/** All `*__labels.png` under `truthDir` with their sidecar metadata. */
export function findAnnotations(truthDir: string): Annotation[] {
  const suffix = "__labels.png";
  const pngs = rglob(truthDir, (n) => endsWithName(n, suffix)).sort(comparePaths);
  const root = resolve(truthDir);
  const items: Annotation[] = [];
  for (const png of pngs) {
    const name = basename(png);
    const scan = name.slice(0, name.length - suffix.length);
    let meta: Record<string, unknown> = {};
    const side = png.slice(0, png.length - extname(png).length) + ".json";
    if (existsSync(side)) {
      try {
        const value: unknown = JSON.parse(readFileSync(side, "utf8"));
        if (value === null || typeof value !== "object" || Array.isArray(value)) throw new Error("expected a JSON object");
        meta = value as Record<string, unknown>;
      } catch (e) {
        console.error(`warning: unreadable sidecar ${side}: ${(e as Error).message}`);
      }
    }
    let annotator = truthy(meta.annotator) ? pyStrValue(meta.annotator).trim() : "";
    if (!annotator) annotator = resolve(dirname(png)) !== root ? basename(dirname(png)) : "unknown";
    items.push({
      scan: "scan" in meta ? meta.scan : scan,
      path: png,
      meta,
      annotator,
      prefilled: truthy(meta.prefilled ?? false),
    });
  }
  return items;
}

/** Python `Path(s).name` (drive and separators stripped). */
function pathName(s: string): string {
  return basename(s);
}

function findX3p(predFrom: string, scan: unknown, meta: Record<string, unknown>): string | null {
  if (typeof scan !== "string" || pathName(scan) !== scan || ["", ".", ".."].includes(scan)) return null;
  const cands = [join(predFrom, `${scan}.x3p`)];
  if (truthy(meta.source)) cands.push(join(predFrom, basename(pyStrValue(meta.source))));
  for (const c of cands) if (existsSync(c)) return c;
  const hits = rglob(predFrom, (n) => endsWithName(n, ".x3p")).filter((p) => {
    const n = basename(p);
    return (n.startsWith(".") && n.length === 4 ? n : n.slice(0, n.length - 4)) === scan;
  });
  return hits.length ? hits[0] : null;
}

/** Python `datetime.fromisoformat(s).timestamp()` (naive times as UTC); null when unparseable. */
function isoTimestamp(s: string): number | null {
  const m =
    /^(\d{4})(-?)(\d{2})\2(\d{2})(?:[^](\d{2})(?:(:?)(\d{2})(?:\6(\d{2})(?:[.,](\d+))?)?)?(Z|[+-]\d{2}(?::?\d{2}(?::?\d{2}(?:[.,]\d+)?)?)?)?)?$/u.exec(
      s,
    );
  if (!m) return null;
  const [Y, M, D] = [Number(m[1]), Number(m[3]), Number(m[4])];
  const [h, mi, sec] = [Number(m[5] ?? 0), Number(m[7] ?? 0), Number(m[8] ?? 0)];
  const micro = m[9] ? Number(m[9].slice(0, 6).padEnd(6, "0")) : 0;
  if (M < 1 || M > 12 || D < 1 || h > 23 || mi > 59 || sec > 59 || Y < 1) return null;
  const d = new Date(0);
  d.setUTCFullYear(Y, M - 1, D);
  if (d.getUTCMonth() !== M - 1) return null; // day out of range
  d.setUTCHours(h, mi, sec, 0);
  let offset = 0;
  const tz = m[10];
  if (tz && tz !== "Z") {
    const t = /^([+-])(\d{2}):?(\d{2})?:?(\d{2})?/.exec(tz)!;
    const oh = Number(t[2]), om = Number(t[3] ?? 0), os = Number(t[4] ?? 0);
    if (oh > 23 || om > 59 || os > 59) return null;
    offset = (t[1] === "-" ? -1 : 1) * (oh * 3600 + om * 60 + os);
  }
  const us = (d.getTime() / 1000 - offset) * 1e6 + micro;
  return us / 1e6;
}

function revisionTime(meta: Record<string, unknown>): number {
  return isoTimestamp(pyStrValue(meta.timestamp ?? "")) ?? 0;
}

/** Private scoring map written by `scripts/build-annotator.ts --blind-map`. */
export interface BlindMap {
  packet: string;
  scans: Record<string, BlindRecord>;
}
export interface BlindRecord {
  source: string;
  sha256: string;
  label_shape: [number, number];
  um_per_px: number;
}

/** SHA-256 hex digest of a file. */
export function sha256File(path: string): string {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function blindSource(predFrom: string, record: BlindRecord): string {
  const name = record.source;
  if (typeof name !== "string" || pathName(name) !== name) throw new EvaluationError("private map source must be a filename");
  for (const path of rglob(predFrom, (n) => nameEq(n, name)).sort(comparePaths)) {
    if (sha256File(path) === record.sha256) return path;
  }
  throw new EvaluationError(`no scan matching private map hash for ${name}`);
}

function isFsError(e: unknown): e is Error {
  return e instanceof Error && typeof (e as NodeJS.ErrnoException).code === "string";
}

/** Mean, sample standard deviation and count of the non-null values (NumPy summation). */
function meanStd(vals: (number | null)[]): [number | null, number | null, number] {
  const v = vals.filter((x): x is number => x !== null);
  if (!v.length) return [null, null, 0];
  const m = pairwiseSum(v) / v.length;
  if (v.length === 1) return [m, 0.0, 1];
  const sq = v.map((x) => (x - m) * (x - m));
  return [m, Math.sqrt(pairwiseSum(sq) / (v.length - 1)), v.length];
}

function fmt(m: number | null, s: number | null, n: number, digits = 3): string {
  if (m === null) return "n/a";
  return `${fmtF(m, digits)} ± ${fmtF(s!, digits)} (n=${n})`;
}

/** Python `_r` for a float: rounded and written as `repr`; "" for None. */
function rf(v: number | null, nd = 2): string {
  return v === null ? "" : pyRepr(pyRoundN(v, nd));
}

const pyBool = (b: boolean): string => (b ? "True" : "False");

/** `csv.writer` row (QUOTE_MINIMAL, `\r\n` terminator). */
function csvRow(cells: (string | number)[]): string {
  return (
    cells
      .map((c) => {
        const s = String(c);
        return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
      })
      .join(",") + "\r\n"
  );
}

// --------------------------------------------------------------------------- figures

/** Crop of `base` with TP / FP / FN blended in, scaled (nearest) into a white `size` square. */
export function errorTile(
  base: Image,
  pred: G,
  truth: G,
  crop: readonly [number, number, number, number],
  size = 260,
  alpha = 0.6,
): Image {
  const [y0, y1, x0, x1] = crop;
  const img = cropImage(base, y0, y1, x0, x1);
  const W = pred.cols;
  const cy0 = Math.min(Math.max(0, y0), pred.rows), cx0 = Math.min(Math.max(0, x0), W);
  const c = img.channels;
  for (let y = 0; y < img.height; y++) {
    for (let x = 0; x < img.width; x++) {
      const i = (cy0 + y) * W + cx0 + x;
      const p = pred.data[i], t = truth.data[i];
      const col = p && t ? TP_COLOUR : p ? FP_COLOUR : t ? FN_COLOUR : null;
      if (!col) continue;
      const o = (y * img.width + x) * c;
      for (let k = 0; k < c; k++) img.data[o + k] = Math.trunc(img.data[o + k] * (1 - alpha) + col[c === 3 ? k : 0] * alpha);
    }
  }
  const s = size / Math.max(img.height, img.width);
  const tile = resize(img, Math.max(1, Math.trunc(img.width * s)), Math.max(1, Math.trunc(img.height * s)), "nearest");
  const out = newImage(size, size, 3, 255);
  paste(out, tile, Math.floor((size - tile.width) / 2), Math.floor((size - tile.height) / 2));
  return out;
}

const SHORT: Record<number, string> = {
  [BREECH_FACE]: "1 breech face",
  [APERTURE_SHEAR]: "2 aperture shear",
  [FIRING_PIN_SHEAR]: "3 firing-pin shear",
  [FIRING_PIN_IMPRESSION]: "4 firing-pin impression",
};
const INK: BGR = [20, 20, 20];

/** Grid: one row per (scan, annotator), one column per class; TP / FP / FN coloured. */
export function errorFigure(rows: readonly Row[], size = 260): Image {
  const head = 30, labelW = 330, pad = 4;
  const width = labelW + CLASSES.length * (size + pad);
  const top = newImage(width, head + 34, 3, 255);
  draw(top, (ctx) => {
    let x = 10;
    for (const [c, text] of [
      [TP_COLOUR, "both (true positive)"],
      [FP_COLOUR, "algorithm only (false positive)"],
      [FN_COLOUR, "annotator only (false negative)"],
    ] as const) {
      rectangle(ctx, x, 8, x + 22, 26, c, -1);
      putText(ctx, text, x + 28, 23, 0.5, INK);
      x += 40 + 9 * text.length;
    }
    CLASSES.forEach((k, i) => putText(ctx, SHORT[k], labelW + i * (size + pad) + 6, head + 26, 0.55, INK));
  });
  const lines = [top];
  for (const r of rows) {
    const row = newImage(width, size + pad, 3, 255);
    const txt = [[...r.scan].slice(0, 38).join(""), `annotator: ${r.annotator}` + (r.prefilled ? " (prefilled)" : "")];
    for (const k of CLASSES) {
      const m = r.metrics.classes[k];
      txt.push(`${k}: IoU ${fmtF(m.iou, 2)}` + (m.absent_in_both ? " (absent)" : ""));
    }
    draw(row, (ctx) => txt.forEach((t, j) => putText(ctx, t, 8, 24 + 22 * j, 0.45, INK)));
    const { rows: h, cols: w } = r.truth;
    CLASSES.forEach((k, i) => {
      const pm = maskWhere(h, w, (p) => r.pred.data[p] === k && r.truth.data[p] !== IGNORE);
      const tm = maskWhere(h, w, (p) => r.truth.data[p] === k);
      paste(row, errorTile(r.base, pm, tm, r.crop, size), labelW + i * (size + pad), 0);
    });
    lines.push(row);
  }
  return vstack(lines);
}

// --------------------------------------------------------------------------- driver

/** One scored annotation (Python row dict). */
export interface Row {
  scan: string;
  annotator: string;
  prefilled: boolean;
  status: string;
  qc_score: number;
  um_per_px: number;
  metrics: Evaluation;
  pred: Grid<Uint8Array>;
  truth: Grid<Uint8Array>;
  base: Image;
  crop: [number, number, number, number];
  file: string;
}

/** Inter-annotator comparison of one class on one scan. */
export interface InterRow extends Record<MetricName, number> {
  scan: string;
  annotator_a: string;
  annotator_b: string;
  cls: string;
  label: number;
  absent_in_both: boolean;
}

/** Per-class summary over the independently drawn annotations. */
export interface SummaryRow {
  cls: string;
  label: number;
  n: number;
  n_absent_in_both: number;
  [key: string]: string | number | null;
}

export type MeanStd = [number | null, number | null, number];

export interface RunOutput {
  rows: Row[];
  summary: SummaryRow[];
  geometry: { primer_center_err_um: MeanStd; pin_center_err_um: MeanStd };
  inter: InterRow[];
  /** `[scan, annotator, reason]` of every annotation that was not scored. */
  skipped: [string, string, string][];
  markdown: string;
}

/**
 * Score every annotation under `truthDir` against the segmentation of its X3P in `predFrom` and
 * write `per_scan.csv`, `per_scan_long.csv`, `summary.csv`, `summary.md`, `inter_annotator.csv`
 * (when two annotators drew the same scan), `error_overlays.png` and `figures/` to `outDir`.
 *
 * With `blindMap` only completed, non-prefilled exports of that packet are scored, and the scan is
 * located by the SHA-256 in the private map. Per scan and annotator (case-insensitive) the latest
 * revision by sidecar timestamp is used.
 *
 * @throws EvaluationExit when there are no annotations at all.
 */
export function run(
  predFrom: string,
  truthDir: string,
  outDir: string,
  tolUm = DEFAULT_TOL_UM,
  params: Partial<Params> | null = null,
  blindMap: BlindMap | null = null,
): RunOutput {
  const anns = findAnnotations(truthDir);
  if (!anns.length) throw new EvaluationExit(`no *__labels.png annotations found under ${truthDir}`);
  mkdirSync(join(outDir, "figures"), { recursive: true });

  const cache = new Map<string, [Result, Image]>();
  const rows: Row[] = [];
  const skipped: [string, string, string][] = [];
  const truthsByScan = new Map<string, [string, Grid<Uint8Array>, number][]>();
  const latest = new Map<string, Annotation & { scan: string }>();
  for (const a of anns) {
    const { scan, meta, annotator } = a;
    if (typeof scan !== "string" || !scan || meta.done === false) {
      skipped.push([pyStrValue(scan), annotator, "missing scan ID or unfinished label"]);
      continue;
    }
    if (
      blindMap &&
      (meta.packet !== blindMap.packet || !Object.prototype.hasOwnProperty.call(blindMap.scans, scan) || meta.done !== true || a.prefilled)
    ) {
      skipped.push([scan, annotator, "export is not a completed independent label for this packet"]);
      continue;
    }
    if (!blindMap && truthy(meta.packet)) {
      skipped.push([scan, annotator, "blinded labels require their private map"]);
      continue;
    }
    const key = `${scan}\u0000${casefold(annotator)}`;
    const previous = latest.get(key);
    if (previous === undefined || revisionTime(meta) >= revisionTime(previous.meta)) latest.set(key, { ...a, scan });
  }

  const sources = new Map<string, string>();
  for (const a of latest.values()) {
    const { scan, meta } = a;
    const defaultSpacing = params?.workUmPerPx ?? DEFAULT_PARAMS.workUmPerPx;
    const record = blindMap ? blindMap.scans[scan] : null;
    let um: number;
    try {
      um = pyFloat(truthy(record) ? record!.um_per_px : "um_per_px" in meta ? meta.um_per_px : defaultSpacing);
      if (!Number.isFinite(um) || um <= 0) throw new EvaluationError("pixel spacing must be positive and finite");
    } catch (e) {
      if (!(e instanceof EvaluationError)) throw e;
      skipped.push([scan, a.annotator, `invalid pixel spacing: ${e.message}`]);
      continue;
    }
    let x3p: string | null;
    try {
      if (record && truthy(record)) {
        if (!sources.has(scan)) sources.set(scan, blindSource(predFrom, record));
        x3p = sources.get(scan)!;
      } else x3p = findX3p(predFrom, scan, meta);
    } catch (e) {
      if (!(e instanceof EvaluationError) && !isFsError(e)) throw e;
      skipped.push([scan, a.annotator, (e as Error).message]);
      continue;
    }
    if (x3p === null) {
      skipped.push([scan, a.annotator, "no matching X3P in --pred-from"]);
      continue;
    }
    let truth: Grid<Uint8Array>;
    try {
      truth = readLabels(a.path);
      if (record && truthy(record)) {
        const ls = record.label_shape as unknown;
        if (!(Array.isArray(ls) && ls.length === 2 && ls[0] === truth.rows && ls[1] === truth.cols)) {
          throw new EvaluationError("label shape does not match the private map");
        }
      }
    } catch (e) {
      if (!(e instanceof EvaluationError) && !isFsError(e)) throw e;
      skipped.push([scan, a.annotator, (e as Error).message]);
      continue;
    }
    const ck = `${x3p}\u0000${um}`;
    if (!cache.has(ck)) {
      const res = segment(readX3P(x3p), makeParams({ ...(params ?? {}), workUmPerPx: um }));
      cache.set(ck, [res, render(res)]);
    }
    const [res, base] = cache.get(ck)!;
    let m: Evaluation;
    try {
      m = evaluate(res, truth, tolUm);
    } catch (e) {
      if (!(e instanceof EvaluationError)) throw e;
      skipped.push([scan, a.annotator, e.message]);
      continue;
    }
    if (!a.prefilled) {
      if (!truthsByScan.has(scan)) truthsByScan.set(scan, []);
      truthsByScan.get(scan)!.push([a.annotator, truth, res.umPerPx]);
    }
    let crop = cropBox(res);
    if (res.primerRadiusPx == null) {
      // fall back to the annotated extent
      let ymin = Infinity, ymax = -Infinity, xmin = Infinity, xmax = -Infinity;
      for (let y = 0; y < truth.rows; y++) {
        for (let x = 0; x < truth.cols; x++) {
          const v = truth.data[y * truth.cols + x];
          if (v >= 1 && v <= 4) {
            if (y < ymin) ymin = y;
            if (y > ymax) ymax = y;
            if (x < xmin) xmin = x;
            if (x > xmax) xmax = x;
          }
        }
      }
      if (ymin !== Infinity) crop = [Math.max(0, ymin - 10), ymax + 10, Math.max(0, xmin - 10), xmax + 10];
    }
    rows.push({
      scan,
      annotator: a.annotator,
      prefilled: a.prefilled,
      status: res.status,
      qc_score: res.qcScore,
      um_per_px: res.umPerPx,
      metrics: m,
      pred: res.labels!,
      truth,
      base,
      crop,
      file: a.path,
    });
  }

  // ---- per-scan CSV (one row per scan x annotator, wide) and long CSV
  const names: Record<number, string> = Object.fromEntries(CLASSES.map((k) => [k, LABEL_NAMES[k]]));
  const perClass = ["truth_px", "pred_px", ...METRICS, "absent_in_both"] as const;
  const wideCols = ["scan", "annotator", "prefilled", "status", "qc_score", "um_per_px", "primer_center_err_um", "pin_center_err_um"];
  for (const k of CLASSES) wideCols.push(...perClass.map((c) => `${names[k]}_${c}`));
  let csv = csvRow(wideCols);
  for (const r of rows) {
    const m = r.metrics;
    const cells: (string | number)[] = [
      r.scan,
      r.annotator,
      pyBool(r.prefilled),
      r.status,
      rf(r.qc_score, 3),
      rf(r.um_per_px, 4),
      rf(m.primer_center_err_um),
      rf(m.pin_center_err_um),
    ];
    for (const k of CLASSES) {
      const cm = m.classes[k];
      cells.push(cm.truth_px, cm.pred_px, ...METRICS.map((x) => rf(cm[x], 4)), pyBool(cm.absent_in_both));
    }
    csv += csvRow(cells);
  }
  writeFileSync(join(outDir, "per_scan.csv"), csv, "utf8");
  csv = csvRow(["scan", "annotator", "prefilled", "class", "label", "truth_px", "pred_px", "tp", "fp", "fn", ...METRICS, "absent_in_both"]);
  for (const r of rows) {
    for (const k of CLASSES) {
      const cm = r.metrics.classes[k];
      csv += csvRow([
        r.scan,
        r.annotator,
        pyBool(r.prefilled),
        names[k],
        k,
        cm.truth_px,
        cm.pred_px,
        cm.tp,
        cm.fp,
        cm.fn,
        ...METRICS.map((x) => rf(cm[x], 4)),
        pyBool(cm.absent_in_both),
      ]);
    }
  }
  writeFileSync(join(outDir, "per_scan_long.csv"), csv, "utf8");

  // ---- inter-annotator agreement
  const inter: InterRow[] = [];
  for (const [scan, lst] of truthsByScan) {
    for (let i = 0; i < lst.length; i++) {
      for (let j = i + 1; j < lst.length; j++) {
        const [n1, t1, um] = lst[i], [n2, t2] = lst[j];
        if (casefold(n1) === casefold(n2) || t1.rows !== t2.rows || t1.cols !== t2.cols) continue;
        const cmp = compareLabels(t1, t2, um, tolUm);
        for (const k of CLASSES) {
          const cm = cmp[k];
          inter.push({
            scan,
            annotator_a: n1,
            annotator_b: n2,
            cls: names[k],
            label: k,
            iou: cm.iou,
            dice: cm.dice,
            precision: cm.precision,
            recall: cm.recall,
            bf_score: cm.bf_score,
            absent_in_both: cm.absent_in_both,
          });
        }
      }
    }
  }
  if (inter.length) {
    csv = csvRow(["scan", "annotator_a", "annotator_b", "class", "label", ...METRICS, "absent_in_both"]);
    for (const r of inter) {
      csv += csvRow([r.scan, r.annotator_a, r.annotator_b, r.cls, r.label, ...METRICS.map((x) => rf(r[x], 4)), pyBool(r.absent_in_both)]);
    }
    writeFileSync(join(outDir, "inter_annotator.csv"), csv, "utf8");
  }

  // ---- summary of independently drawn annotations
  const independent = rows.filter((r) => !r.prefilled);
  const summary: SummaryRow[] = [];
  for (const k of CLASSES) {
    const present = independent.map((r) => r.metrics.classes[k]).filter((c) => !c.absent_in_both);
    const absent = independent.filter((r) => r.metrics.classes[k].absent_in_both).length;
    const s: SummaryRow = { cls: names[k], label: k, n: present.length, n_absent_in_both: absent };
    for (const x of METRICS) [s[`${x}_mean`], s[`${x}_std`]] = meanStd(present.map((c) => c[x]));
    const hh = inter.filter((r) => r.label === k && !r.absent_in_both).map((r) => r.iou);
    [s.human_vs_human_iou_mean, s.human_vs_human_iou_std, s.human_vs_human_n] = meanStd(hh);
    summary.push(s);
  }
  const geometry = {
    primer_center_err_um: meanStd(independent.map((r) => r.metrics.primer_center_err_um)),
    pin_center_err_um: meanStd(independent.map((r) => r.metrics.pin_center_err_um)),
  };
  const num = (v: string | number | null): number | null => v as number | null;
  csv = csvRow([
    "class",
    "label",
    "n",
    "n_absent_in_both",
    ...METRICS.flatMap((x) => [`${x}_mean`, `${x}_std`]),
    "human_vs_human_iou_mean",
    "human_vs_human_iou_std",
    "human_vs_human_n",
  ]);
  for (const s of summary) {
    csv += csvRow([
      s.cls,
      s.label,
      s.n,
      s.n_absent_in_both,
      ...METRICS.flatMap((x) => [rf(num(s[`${x}_mean`]), 4), rf(num(s[`${x}_std`]), 4)]),
      rf(num(s.human_vs_human_iou_mean), 4),
      rf(num(s.human_vs_human_iou_std), 4),
      num(s.human_vs_human_n)!,
    ]);
  }
  for (const [key, [m, sd, n]] of Object.entries(geometry)) {
    csv += csvRow([key, "", n, "", rf(m, 2), rf(sd, 2), ...Array<string>(2 * METRICS.length - 2).fill(""), "", "", ""]);
  }
  writeFileSync(join(outDir, "summary.csv"), csv, "utf8");

  const md = [
    `# Algorithm vs. annotation (${rows.length} annotations, ${new Set(rows.map((r) => r.scan)).size} scans, ` +
      `${new Set(rows.map((r) => r.annotator)).size} annotators)`,
    "",
    `Boundary tolerance ${fmtG(tolUm, 6)} µm. Mean ± sample std over independently drawn annotations. Classes absent in both ` +
      "masks are left out of the mean (count in the last column).",
    "",
    "| Class | IoU | Dice | Precision | Recall | Boundary F | Human-vs-human IoU | absent in both |",
    "|---|---|---|---|---|---|---|---|",
  ];
  for (const s of summary) {
    md.push(
      `| ${s.cls} | ` +
        METRICS.map((x) => fmt(num(s[`${x}_mean`]), num(s[`${x}_std`]), s.n)).join(" | ") +
        ` | ${fmt(num(s.human_vs_human_iou_mean), num(s.human_vs_human_iou_std), num(s.human_vs_human_n)!)}` +
        ` | ${s.n_absent_in_both} |`,
    );
  }
  md.push("", "| Geometry | error (µm) |", "|---|---|");
  for (const [key, [m, sd, n]] of Object.entries(geometry)) {
    md.push(`| ${key.replaceAll("_err_um", "").replaceAll("_", " ")} | ${fmt(m, sd, n, 1)} |`);
  }
  const nPref = rows.filter((r) => r.prefilled).length;
  md.push("", `Pre-filled annotations: ${nPref} of ${rows.length}. The detail files include them and the aggregate excludes them.`);
  if (skipped.length) md.push("", "Skipped:", ...skipped.map(([s, a, why]) => `- ${s} (${a}): ${why}`));
  const markdown = md.join("\n");
  writeFileSync(join(outDir, "summary.md"), markdown + "\n", "utf8");

  // ---- figures
  if (rows.length) {
    imwrite(join(outDir, "error_overlays.png"), errorFigure(rows));
    for (const r of rows) {
      const safe = [...`${r.scan}__${r.annotator}`].map((ch) => (/^[\p{L}\p{N}\-_.]$/u.test(ch) ? ch : "_")).join("");
      imwrite(join(outDir, "figures", `${safe}__errors.png`), errorFigure([r]));
    }
  }
  return { rows, summary, geometry, inter, skipped, markdown };
}

// --------------------------------------------------------------------------- CLI

const USAGE =
  "usage: npx tsx src/evaluate.ts [-h] --pred-from PRED_FROM --truth TRUTH [-o OUT] [--tol-um TOL_UM] [--blind-map BLIND_MAP]";
const HELP = `${USAGE}

Score automatic masks against annotated label images.

options:
  -h, --help            show this help message and exit
  --pred-from PRED_FROM
                        folder with the X3P scans that were annotated
  --truth TRUTH         folder with <scan>__labels.png exports (searched recursively)
  -o OUT, --out OUT     output folder
  --tol-um TOL_UM       boundary-F tolerance in µm
  --blind-map BLIND_MAP
                        private map created with build-annotator.ts --blind-map`;

/** Command-line entry point; returns the process exit status (2 = usage error, 1 = no annotations). */
export function main(argv: string[] = process.argv.slice(2)): number {
  const usageError = (msg: string): number => {
    console.error(`${USAGE}\nevaluate: error: ${msg}`);
    return 2;
  };
  let values: Record<string, string | boolean | undefined>;
  try {
    ({ values } = parseArgs({
      args: argv,
      options: {
        "pred-from": { type: "string" },
        truth: { type: "string" },
        out: { type: "string", short: "o", default: "outputs/evaluation" },
        "tol-um": { type: "string" },
        "blind-map": { type: "string" },
        help: { type: "boolean", short: "h" },
      },
      strict: true,
      allowPositionals: false,
    }));
  } catch (e) {
    return usageError((e as Error).message);
  }
  if (values.help) {
    console.log(HELP);
    return 0;
  }
  const missing = ["pred-from", "truth"].filter((k) => values[k] === undefined).map((k) => `--${k}`);
  if (missing.length) return usageError(`the following arguments are required: ${missing.join(", ")}`);
  let tol = DEFAULT_TOL_UM;
  if (values["tol-um"] !== undefined) {
    try {
      tol = pyFloat(values["tol-um"]);
    } catch {
      return usageError(`argument --tol-um: invalid float value: '${values["tol-um"]}'`);
    }
  }
  if (!Number.isFinite(tol) || tol < 0) return usageError("--tol-um must be nonnegative and finite");
  const blindMap = values["blind-map"] ? (JSON.parse(readFileSync(values["blind-map"] as string, "utf8")) as BlindMap) : null;
  let out: RunOutput;
  try {
    out = run(values["pred-from"] as string, values.truth as string, values.out as string, tol, null, blindMap);
  } catch (e) {
    if (e instanceof EvaluationExit) {
      console.error(e.message);
      return 1;
    }
    throw e;
  }
  console.log(out.markdown);
  console.log(`\nwritten to ${values.out}`);
  return 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  process.exitCode = main();
}
