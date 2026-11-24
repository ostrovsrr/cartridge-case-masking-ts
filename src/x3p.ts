/**
 * Minimal reader for ISO 25178-72 X3P surface-topography files.
 *
 * An X3P file is a ZIP archive holding `main.xml` (metadata) and a binary
 * height matrix (usually `bindata/data.bin`).  Only the subset used by
 * ballistics databases is supported: a single regular grid (FeatureType SUR)
 * whose Z values are stored as little-endian I/L/F/D numbers, X varying fastest.
 *
 * Port of `cartmask/x3p.py`; `writeX3P` is the port of `write_x3p` from
 * `cartmask/synthetic.py`.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { basename, extname } from "node:path";
import { unzipSync, zipSync, strToU8 } from "fflate";
import { XMLParser } from "fast-xml-parser";
import type { Grid } from "./ndimage/index.js";

type ZType = "I" | "L" | "F" | "D";
const BYTES: Record<ZType, number> = { I: 2, L: 4, F: 4, D: 8 };

/**
 * Height map with physical sampling.
 *
 * - `z`: 2-D grid of heights in metres, shape (rows=Y, cols=X); NaN = invalid.
 * - `dx`, `dy`: pixel spacing in metres.
 */
export interface Surface {
  z: Grid<Float64Array>;
  dx: number;
  dy: number;
  name: string;
}

/** `[rows, cols]` of the height map (Python `Surface.shape`). */
export function surfaceShape(s: Surface): [number, number] {
  return [s.z.rows, s.z.cols];
}

/** Pixel spacing in micrometres (Python `Surface.um_per_px`). */
export function umPerPx(s: Surface): number {
  return s.dx * 1e6;
}

/** Build a `Surface` (Python dataclass constructor; `name` defaults to ""). */
export function makeSurface(z: Grid<Float64Array>, dx: number, dy: number, name = ""): Surface {
  return { z, dx, dy, name };
}

// ---------------------------------------------------------------- XML helpers

type XNode = Record<string, unknown>;

const parser = new XMLParser({
  removeNSPrefix: true, // equivalent of Python's _strip_ns
  ignoreAttributes: true,
  parseTagValue: false,
  trimValues: false,
  isArray: () => false,
});

function child(node: unknown, key: string): unknown {
  if (node === null || typeof node !== "object") return undefined;
  const v = (node as XNode)[key];
  // repeated elements: ElementTree.find returns the first match
  return Array.isArray(v) ? v[0] : v;
}

function elementText(el: unknown): string | undefined {
  if (el === undefined || el === null) return undefined;
  if (typeof el === "string") return el;
  if (typeof el === "number" || typeof el === "boolean") return String(el);
  const t = (el as XNode)["#text"];
  return t === undefined ? undefined : String(t);
}

/** Port of Python `_text`: stripped text at a slash path, or `dflt` when the element is missing or empty. */
function text(root: unknown, path: string, dflt?: string): string | undefined {
  let node: unknown = root;
  for (const part of path.split("/")) {
    node = child(node, part);
    if (node === undefined) return dflt;
  }
  const t = elementText(node);
  return t ? t.trim() : dflt; // Python: `el.text.strip() if el is not None and el.text else default`
}

function req(v: string | undefined, what: string): string {
  if (v === undefined) throw new Error(`X3P: missing ${what}`);
  return v;
}

/** Python `float()`: trims whitespace, rejects garbage. */
function pyFloat(s: string): number {
  const t = s.trim();
  const v = Number(t);
  if (t === "" || (Number.isNaN(v) && !/^[+-]?nan$/i.test(t))) throw new Error(`could not convert string to float: '${s}'`);
  return v;
}

/** Python `int()` on a decimal string. */
function pyInt(s: string): number {
  const t = s.trim();
  if (!/^[+-]?\d+$/.test(t)) throw new Error(`invalid literal for int(): '${s}'`);
  return parseInt(t, 10);
}

// ---------------------------------------------------------------- reader

/**
 * Read an X3P file.
 *
 * X3P stores rows with the Y axis pointing up (row 0 = y = 0).  With
 * `imageOrientation=true` the rows are flipped so that the array displays
 * the way the specimen is seen (headstamps read correctly).
 */
export function readX3P(path: string, opts: { imageOrientation?: boolean } = {}): Surface {
  const imageOrientation = opts.imageOrientation ?? true;
  return parseX3P(readFileSync(path), stem(path), imageOrientation);
}

function stem(path: string): string {
  const b = basename(path);
  return b.slice(0, b.length - extname(b).length);
}

/** Parse X3P bytes (the ZIP archive) into a `Surface`; `readX3P` without the file system. */
export function parseX3P(bytes: Uint8Array, name = "", imageOrientation = true): Surface {
  const files = unzipSync(bytes);
  const names = Object.keys(files);
  const xmlName = names.find((n) => n.toLowerCase().endsWith("main.xml"));
  if (xmlName === undefined) throw new Error(`${name}: no main.xml in archive`);
  const doc = parser.parse(Buffer.from(files[xmlName]!).toString("utf8")) as XNode;
  // the root element (ISO5436_2) is the single non-declaration key
  const rootKey = Object.keys(doc).find((k) => !k.startsWith("?"));
  const root = rootKey === undefined ? undefined : doc[rootKey];
  const r3 = child(root, "Record3");
  const r1 = child(root, "Record1");
  const nx = pyInt(req(text(r3, "MatrixDimension/SizeX"), "SizeX"));
  const ny = pyInt(req(text(r3, "MatrixDimension/SizeY"), "SizeY"));
  const nz = pyInt(text(r3, "MatrixDimension/SizeZ", "1")!);
  const dx = pyFloat(req(text(r1, "Axes/CX/Increment"), "CX/Increment"));
  const dy = pyFloat(req(text(r1, "Axes/CY/Increment"), "CY/Increment"));
  const dtype = text(r1, "Axes/CZ/DataType", "D")! as ZType;
  if (!(dtype in BYTES)) throw new Error(`KeyError: '${dtype}'`);
  const zIncText = text(r1, "Axes/CZ/Increment", "1");
  const zOffText = text(r1, "Axes/CZ/Offset", "0");
  const zInc = pyFloat(zIncText || "1"); // Python: float(_text(...) or 1)
  const zOff = pyFloat(zOffText || "0");

  const link = text(r3, "DataLink/PointDataLink", "bindata/data.bin")!;
  const member = names.find((n) => n.endsWith(link)) ?? link;
  const raw = files[member];
  if (raw === undefined) throw new Error(`KeyError: "There is no item named '${member}' in the archive"`);
  const size = BYTES[dtype];
  if (raw.byteLength % size !== 0) throw new Error("buffer size must be a multiple of element size");
  const count = raw.byteLength / size;
  if (count !== nx * ny * nz) throw new Error(`${name}: expected ${nx * ny * nz} values, got ${count}`);
  const n = nx * ny; // first Z layer only
  const z = new Float64Array(n);
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  switch (dtype) {
    case "I":
      // Integer encodings: apply scale/offset. Floats are already in metres.
      for (let i = 0; i < n; i++) z[i] = view.getInt16(i * 2, true) * zInc + zOff;
      break;
    case "L":
      for (let i = 0; i < n; i++) z[i] = view.getInt32(i * 4, true) * zInc + zOff;
      break;
    case "F":
      for (let i = 0; i < n; i++) z[i] = view.getFloat32(i * 4, true);
      break;
    case "D":
      for (let i = 0; i < n; i++) z[i] = view.getFloat64(i * 8, true);
      break;
  }

  const validLink = text(r3, "DataLink/ValidPointsLink");
  if (validLink) {
    const vm = names.find((nm) => nm.endsWith(validLink));
    if (vm !== undefined) {
      // np.unpackbits(..., bitorder="little"): bit k of byte b is sample 8*b + k;
      // a short mask counts the missing bits as valid
      const bits = files[vm]!;
      for (let i = 0; i < n; i++) {
        const b = i >> 3;
        if (b < bits.length && ((bits[b]! >> (i & 7)) & 1) === 0) z[i] = NaN;
      }
    }
  }

  let out = z;
  if (imageOrientation) {
    out = new Float64Array(n);
    for (let r = 0; r < ny; r++) out.set(z.subarray((ny - 1 - r) * nx, (ny - r) * nx), r * nx);
  }
  return { z: { data: out, rows: ny, cols: nx }, dx, dy, name };
}

// ---------------------------------------------------------------- writer

/** Python `repr(float)`-like formatting so the XML matches the Python writer. */
function pyRepr(v: number): string {
  if (Number.isInteger(v) && Math.abs(v) < 1e16) return v.toFixed(1);
  const s = String(v);
  const abs = Math.abs(v);
  if (abs !== 0 && (abs < 1e-4 || abs >= 1e16)) {
    // Python switches to exponent notation below 1e-4: 8e-06 rather than 0.000008
    const [m, e] = v.toExponential().split("e");
    const ev = Number(e);
    return `${m}e${ev < 0 ? "-" : "+"}${String(Math.abs(ev)).padStart(2, "0")}`;
  }
  return s;
}

/** Serialise a surface to X3P bytes (see `writeX3P`). */
export function encodeX3P(surf: Surface): Uint8Array {
  const { rows: ny, cols: nx, data } = surf.z;
  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<p:ISO5436_2 xmlns:p="http://www.opengps.eu/2008/ISO5436_2">
 <Record1><Revision>ISO5436 - 2000</Revision><FeatureType>SUR</FeatureType>
  <Axes>
   <CX><AxisType>I</AxisType><DataType>D</DataType><Increment>${pyRepr(surf.dx)}</Increment><Offset>0</Offset></CX>
   <CY><AxisType>I</AxisType><DataType>D</DataType><Increment>${pyRepr(surf.dy)}</Increment><Offset>0</Offset></CY>
   <CZ><AxisType>A</AxisType><DataType>D</DataType></CZ>
  </Axes></Record1>
 <Record3><MatrixDimension><SizeX>${nx}</SizeX><SizeY>${ny}</SizeY><SizeZ>1</SizeZ></MatrixDimension>
  <DataLink><PointDataLink>bindata/data.bin</PointDataLink></DataLink></Record3>
</p:ISO5436_2>`;
  const bin = new Uint8Array(nx * ny * 8);
  const view = new DataView(bin.buffer);
  // back to X3P row order (y up)
  for (let r = 0; r < ny; r++) {
    const src = (ny - 1 - r) * nx;
    for (let c = 0; c < nx; c++) view.setFloat64((r * nx + c) * 8, data[src + c]!, true);
  }
  return zipSync({ "main.xml": strToU8(xml), "bindata/data.bin": bin }, { level: 6 });
}

/** Write a minimal X3P (float64, Y-up row order) - for tests and demos. */
export function writeX3P(surf: Surface, path: string): void {
  writeFileSync(path, encodeX3P(surf));
}
