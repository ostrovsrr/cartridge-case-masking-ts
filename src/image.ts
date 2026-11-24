/**
 * 8-bit raster images for the display and export code (the role NumPy arrays + OpenCV play in
 * Python): a small BGR/grey image type, the array operations `visualize.py` uses (crop, stack,
 * pad, resize, warp), a lossless PNG codec that keeps grey images grey (like `cv2.imencode`), and
 * the bridge to `@napi-rs/canvas` for drawing primitives, text and JPEG.
 *
 * Channel order is BGR, as in OpenCV, so colour constants and pixel tests read like the Python.
 */
import { writeFileSync } from "node:fs";
import { extname } from "node:path";
import { createCanvas, type SKRSContext2D } from "@napi-rs/canvas";
import { unzlibSync, zlibSync } from "fflate";
import type { Grid } from "./ndimage/grid.js";
import { resizeArea, resizeLinear, resizeNearest, warpAffine } from "./ndimage/interp.js";

/** BGR colour triple (OpenCV order). */
export type BGR = readonly [number, number, number];

/** Row-major 8-bit image: 1 channel (grey) or 3 channels (BGR, interleaved). */
export interface Image {
  width: number;
  height: number;
  channels: 1 | 3;
  data: Uint8Array;
}

/** New image filled with `fill` (a grey level or a BGR colour). */
export function newImage(width: number, height: number, channels: 1 | 3 = 3, fill: number | BGR = 0): Image {
  const data = new Uint8Array(width * height * channels);
  if (typeof fill === "number") data.fill(fill);
  else if (channels === 1) data.fill(fill[0]);
  else for (let i = 0; i < data.length; i += 3) { data[i] = fill[0]; data[i + 1] = fill[1]; data[i + 2] = fill[2]; }
  return { width, height, channels, data };
}

export function cloneImage(img: Image): Image {
  return { ...img, data: img.data.slice() };
}

/** Grey image from a uint8 grid (`labels`, masks). */
export function fromGrid(g: Grid<ArrayLike<number>>): Image {
  return { width: g.cols, height: g.rows, channels: 1, data: Uint8Array.from(g.data as ArrayLike<number>) };
}

/** `cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)`. */
export function grayToBgr(img: Image): Image {
  if (img.channels === 3) return cloneImage(img);
  const out = new Uint8Array(img.width * img.height * 3);
  for (let i = 0, j = 0; i < img.data.length; i++, j += 3) out[j] = out[j + 1] = out[j + 2] = img.data[i];
  return { width: img.width, height: img.height, channels: 3, data: out };
}

/** BGR value of pixel (x, y) (grey replicated). */
export function pixel(img: Image, x: number, y: number): [number, number, number] {
  const i = (y * img.width + x) * img.channels;
  return img.channels === 3 ? [img.data[i], img.data[i + 1], img.data[i + 2]] : [img.data[i], img.data[i], img.data[i]];
}

/** `img[y0:y1, x0:x1].copy()` (bounds clipped like NumPy slicing). */
export function crop(img: Image, y0: number, y1: number, x0: number, x1: number): Image {
  const cl = (v: number, n: number) => Math.max(0, Math.min(n, v));
  y0 = cl(y0, img.height); y1 = Math.max(y0, cl(y1, img.height));
  x0 = cl(x0, img.width); x1 = Math.max(x0, cl(x1, img.width));
  const w = x1 - x0, h = y1 - y0, c = img.channels;
  const out = new Uint8Array(w * h * c);
  for (let y = 0; y < h; y++) {
    const s = ((y0 + y) * img.width + x0) * c;
    out.set(img.data.subarray(s, s + w * c), y * w * c);
  }
  return { width: w, height: h, channels: c, data: out };
}

/** Copy `src` into `dst` with its top-left corner at (x0, y0), in place. */
export function paste(dst: Image, src: Image, x0: number, y0: number): void {
  const s3 = src.channels === dst.channels ? src : grayToBgr(src);
  const c = dst.channels;
  for (let y = 0; y < s3.height; y++) {
    const s = y * s3.width * c;
    dst.data.set(s3.data.subarray(s, s + s3.width * c), ((y0 + y) * dst.width + x0) * c);
  }
}

/** `np.hstack` (equal heights). */
export function hstack(imgs: Image[]): Image {
  const out = newImage(imgs.reduce((s, m) => s + m.width, 0), imgs[0].height, imgs[0].channels);
  let x = 0;
  for (const m of imgs) { paste(out, m, x, 0); x += m.width; }
  return out;
}

/** `np.vstack` (equal widths). */
export function vstack(imgs: Image[]): Image {
  const out = newImage(imgs[0].width, imgs.reduce((s, m) => s + m.height, 0), imgs[0].channels);
  let y = 0;
  for (const m of imgs) { paste(out, m, 0, y); y += m.height; }
  return out;
}

/** `np.pad(img, ((top, bottom), (left, right), (0, 0)), constant_values=value)`. */
export function pad(img: Image, top: number, bottom: number, left: number, right: number, value = 255): Image {
  const out = newImage(img.width + left + right, img.height + top + bottom, img.channels, value);
  paste(out, img, left, top);
  return out;
}

/** OpenCV `saturate_cast<uchar>(float)`: round half to even, clamp to 0..255. */
function sat8(v: number): number {
  if (!(v > 0)) return 0;
  if (v >= 255) return 255;
  const r = Math.round(v);
  return r - v === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

function channel(img: Image, k: number): Grid<Uint8Array> {
  const n = img.width * img.height;
  if (img.channels === 1) return { data: img.data, rows: img.height, cols: img.width };
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = img.data[i * 3 + k];
  return { data: out, rows: img.height, cols: img.width };
}

function merge(chs: Grid<ArrayLike<number>>[], round: boolean): Image {
  const { rows, cols } = chs[0];
  const c = chs.length as 1 | 3;
  const out = new Uint8Array(rows * cols * c);
  for (let k = 0; k < c; k++) {
    const d = chs[k].data;
    for (let i = 0; i < rows * cols; i++) out[i * c + k] = round ? sat8(d[i]) : d[i];
  }
  return { width: cols, height: rows, channels: c, data: out };
}

/** `cv2.resize(img, (width, height), interpolation=...)`, channel by channel. */
export function resize(img: Image, width: number, height: number, mode: "area" | "linear" | "nearest"): Image {
  width = Math.max(1, width);
  height = Math.max(1, height);
  const chs: Grid<ArrayLike<number>>[] = [];
  for (let k = 0; k < img.channels; k++) {
    const g = channel(img, k);
    chs.push(mode === "nearest" ? resizeNearest(g, height, width) : mode === "area" ? resizeArea(g, height, width) : resizeLinear(g, height, width));
  }
  return merge(chs, mode !== "nearest");
}

/** `cv2.warpAffine(img, M, (width, height), flags=INTER_LINEAR, borderValue=0)`; `M` row-major 2x3. */
export function warp(img: Image, M: ArrayLike<number>, width: number, height: number): Image {
  const chs: Grid<ArrayLike<number>>[] = [];
  for (let k = 0; k < img.channels; k++) chs.push(warpAffine(channel(img, k), M, height, width, { border: 0 }));
  return merge(chs, true);
}

// ------------------------------------------------------------------------------------ canvas

/** CSS colour of a BGR triple. */
export function css(c: BGR): string {
  return `rgb(${c[2]}, ${c[1]}, ${c[0]})`;
}

/**
 * Draw on a colour image with a 2-D canvas context, in place. Coordinates are OpenCV pixel
 * indices; add 0.5 to address a pixel centre (see {@link px}).
 */
export function draw(img: Image, fn: (ctx: SKRSContext2D) => void): void {
  if (img.channels !== 3) throw new Error("draw() needs a BGR image");
  const { width: w, height: h } = img;
  const canvas = createCanvas(w, h);
  const ctx = canvas.getContext("2d");
  const id = ctx.createImageData(w, h);
  const d = id.data, s = img.data;
  for (let i = 0, j = 0; i < w * h * 3; i += 3, j += 4) { d[j] = s[i + 2]; d[j + 1] = s[i + 1]; d[j + 2] = s[i]; d[j + 3] = 255; }
  ctx.putImageData(id, 0, 0);
  fn(ctx);
  const back = ctx.getImageData(0, 0, w, h).data;
  for (let i = 0, j = 0; i < w * h * 3; i += 3, j += 4) { s[i] = back[j + 2]; s[i + 1] = back[j + 1]; s[i + 2] = back[j]; }
}

/**
 * Canvas stroke width that matches an OpenCV `thickness`: OpenCV's thick lines come out wider
 * than nominal (measured: 2 -> 3 px, 3 -> 5 px, 5 -> 7 px).
 */
export function strokeWidth(thickness: number): number {
  return thickness > 1 ? thickness + 1 + (thickness & 1) : 1;
}

/** Pixel-centre coordinate of an OpenCV integer pixel index. */
export const px = (v: number): number => v + 0.5;

/** `cv2.line` (round caps, as OpenCV draws thick lines). */
export function line(ctx: SKRSContext2D, x0: number, y0: number, x1: number, y1: number, color: BGR, thickness: number): void {
  ctx.strokeStyle = css(color);
  ctx.lineWidth = strokeWidth(thickness);
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(px(x0), px(y0));
  ctx.lineTo(px(x1), px(y1));
  ctx.stroke();
}

const cvRound = (x: number): number => {
  const r = Math.round(x);
  return r - x === 0.5 && r % 2 !== 0 ? r - 1 : r;
};

/** `cv2.arrowedLine(img, (x0, y0), (x1, y1), color, thickness, tipLength=tipLength)`. */
export function arrowedLine(
  ctx: SKRSContext2D, x0: number, y0: number, x1: number, y1: number, color: BGR, thickness: number, tipLength = 0.1,
): void {
  const tipSize = Math.hypot(x0 - x1, y0 - y1) * tipLength;
  line(ctx, x0, y0, x1, y1, color, thickness);
  const angle = Math.atan2(y0 - y1, x0 - x1);
  for (const s of [1, -1]) {
    const px_ = cvRound(x1 + tipSize * Math.cos(angle + (s * Math.PI) / 4));
    const py_ = cvRound(y1 + tipSize * Math.sin(angle + (s * Math.PI) / 4));
    line(ctx, px_, py_, x1, y1, color, thickness);
  }
}

/** `cv2.circle` outline. */
export function circle(ctx: SKRSContext2D, cx: number, cy: number, r: number, color: BGR, thickness: number): void {
  ctx.strokeStyle = css(color);
  ctx.lineWidth = strokeWidth(thickness);
  ctx.beginPath();
  ctx.arc(px(cx), px(cy), r, 0, 2 * Math.PI);
  ctx.stroke();
}

/** `cv2.rectangle` between inclusive corners; `thickness < 0` fills. */
export function rectangle(ctx: SKRSContext2D, x0: number, y0: number, x1: number, y1: number, color: BGR, thickness: number): void {
  if (thickness < 0) {
    ctx.fillStyle = css(color);
    ctx.fillRect(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
    return;
  }
  ctx.strokeStyle = css(color);
  ctx.lineWidth = strokeWidth(thickness);
  ctx.strokeRect(px(x0), px(y0), x1 - x0, y1 - y0);
}

/**
 * `cv2.putText(img, text, (x, y), FONT_HERSHEY_SIMPLEX, scale, color, thickness, LINE_AA)`:
 * (x, y) is the left end of the baseline. A sans-serif face sized so the cap height matches the
 * Hershey font (about 22 px at scale 1), stretched horizontally to the Hershey advance width;
 * the glyphs themselves differ from OpenCV's.
 */
export function putText(ctx: SKRSContext2D, text: string, x: number, y: number, scale: number, color: BGR, thickness = 1): void {
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(1.18, 1);
  ctx.font = `${(30 * scale).toFixed(1)}px Arial, Helvetica, sans-serif`;
  ctx.textBaseline = "alphabetic";
  ctx.textAlign = "left";
  ctx.fillStyle = css(color);
  ctx.fillText(text, 0, 0);
  if (thickness > 1) {
    ctx.strokeStyle = css(color);
    ctx.lineWidth = thickness - 1;
    ctx.lineJoin = "round";
    ctx.strokeText(text, 0, 0);
  }
  ctx.restore();
}

// ------------------------------------------------------------------------------------ PNG

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(buf: Uint8Array, start: number, end: number): number {
  let c = 0xffffffff;
  for (let i = start; i < end; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length);
  const v = new DataView(out.buffer);
  v.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  v.setUint32(8 + data.length, crc32(out, 4, 8 + data.length));
  return out;
}

const paeth = (a: number, b: number, c: number): number => {
  const p = a + b - c;
  const pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
};

/**
 * Lossless 8-bit PNG: grey images as colour type 0, BGR images as RGB (type 2), like
 * `cv2.imencode(".png", img)`. Rows are filtered with the minimum-sum-of-absolute-differences
 * heuristic and deflated with zlib.
 */
export function encodePng(img: Image, level: 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 = 6): Uint8Array {
  const { width: w, height: h, channels: c } = img;
  const stride = w * c;
  const rgb = new Uint8Array(stride * h);
  if (c === 3) for (let i = 0; i < rgb.length; i += 3) { rgb[i] = img.data[i + 2]; rgb[i + 1] = img.data[i + 1]; rgb[i + 2] = img.data[i]; }
  else rgb.set(img.data);
  const raw = new Uint8Array((stride + 1) * h);
  const cand = Array.from({ length: 5 }, () => new Uint8Array(stride));
  for (let y = 0; y < h; y++) {
    const r = y * stride, pr = r - stride;
    let best = 0, bestSum = Infinity;
    for (let f = 0; f < 5; f++) {
      const o = cand[f];
      let sum = 0;
      for (let i = 0; i < stride; i++) {
        const x = rgb[r + i];
        const a = i >= c ? rgb[r + i - c] : 0;
        const b = y > 0 ? rgb[pr + i] : 0;
        const cc = y > 0 && i >= c ? rgb[pr + i - c] : 0;
        const v = (f === 0 ? x : f === 1 ? x - a : f === 2 ? x - b : f === 3 ? x - ((a + b) >> 1) : x - paeth(a, b, cc)) & 0xff;
        o[i] = v;
        sum += v < 128 ? v : 256 - v;
      }
      if (sum < bestSum) { bestSum = sum; best = f; }
    }
    raw[y * (stride + 1)] = best;
    raw.set(cand[best], y * (stride + 1) + 1);
  }
  const ihdr = new Uint8Array(13);
  const v = new DataView(ihdr.buffer);
  v.setUint32(0, w);
  v.setUint32(4, h);
  ihdr[8] = 8;
  ihdr[9] = c === 3 ? 2 : 0;
  const parts = [
    Uint8Array.from([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk("IHDR", ihdr),
    chunk("IDAT", zlibSync(raw, { level })),
    chunk("IEND", new Uint8Array(0)),
  ];
  const out = new Uint8Array(parts.reduce((s, p) => s + p.length, 0));
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

/**
 * Decode an 8-bit, non-interlaced PNG (grey, grey+alpha, RGB, RGBA, palette). Grey stays one
 * channel; colour comes back as BGR (alpha dropped), like `cv2.imdecode(..., IMREAD_UNCHANGED)`
 * without the alpha plane.
 */
export function decodePng(bytes: Uint8Array): Image {
  const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let o = 8, w = 0, h = 0, depth = 0, ct = 0, interlace = 0;
  let plte: Uint8Array | null = null;
  const idat: Uint8Array[] = [];
  while (o < bytes.length) {
    const len = v.getUint32(o);
    const type = String.fromCharCode(...bytes.subarray(o + 4, o + 8));
    const data = bytes.subarray(o + 8, o + 8 + len);
    if (type === "IHDR") {
      w = v.getUint32(o + 8); h = v.getUint32(o + 12); depth = data[8]; ct = data[9]; interlace = data[12];
    } else if (type === "PLTE") plte = data;
    else if (type === "IDAT") idat.push(data);
    else if (type === "IEND") break;
    o += 12 + len;
  }
  if (depth !== 8 || interlace !== 0) throw new Error(`unsupported PNG (bit depth ${depth}, interlace ${interlace})`);
  const bpp = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 }[ct];
  if (bpp === undefined) throw new Error(`unsupported PNG colour type ${ct}`);
  const z = new Uint8Array(idat.reduce((s, p) => s + p.length, 0));
  let p = 0;
  for (const d of idat) { z.set(d, p); p += d.length; }
  const raw = unzlibSync(z);
  const stride = w * bpp;
  const px_ = new Uint8Array(stride * h);
  for (let y = 0; y < h; y++) {
    const f = raw[y * (stride + 1)];
    const s = y * (stride + 1) + 1, r = y * stride, pr = r - stride;
    for (let i = 0; i < stride; i++) {
      const a = i >= bpp ? px_[r + i - bpp] : 0;
      const b = y > 0 ? px_[pr + i] : 0;
      const c = y > 0 && i >= bpp ? px_[pr + i - bpp] : 0;
      const x = raw[s + i];
      px_[r + i] = (f === 0 ? x : f === 1 ? x + a : f === 2 ? x + b : f === 3 ? x + ((a + b) >> 1) : x + paeth(a, b, c)) & 0xff;
    }
  }
  if (ct === 0) return { width: w, height: h, channels: 1, data: px_ };
  if (ct === 4) {
    const g = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) g[i] = px_[i * 2];
    return { width: w, height: h, channels: 1, data: g };
  }
  const out = new Uint8Array(w * h * 3);
  for (let i = 0; i < w * h; i++) {
    if (ct === 3) {
      const k = px_[i] * 3;
      out[i * 3] = plte![k + 2]; out[i * 3 + 1] = plte![k + 1]; out[i * 3 + 2] = plte![k];
    } else {
      const k = i * bpp;
      out[i * 3] = px_[k + 2]; out[i * 3 + 1] = px_[k + 1]; out[i * 3 + 2] = px_[k];
    }
  }
  return { width: w, height: h, channels: 3, data: out };
}

/** JPEG bytes through `@napi-rs/canvas` (quality 0..100, OpenCV's default 95). */
export function encodeJpeg(img: Image, quality = 95): Uint8Array {
  const c = grayToBgr(img);
  const canvas = createCanvas(c.width, c.height);
  const ctx = canvas.getContext("2d");
  const id = ctx.createImageData(c.width, c.height);
  for (let i = 0, j = 0; i < c.data.length; i += 3, j += 4) {
    id.data[j] = c.data[i + 2]; id.data[j + 1] = c.data[i + 1]; id.data[j + 2] = c.data[i]; id.data[j + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
  return canvas.encodeSync("jpeg", quality);
}

/**
 * `cv2.imwrite` replacement: the suffix of `path` selects PNG (lossless, grey kept grey) or JPEG.
 *
 * @throws Error for any other suffix.
 */
export function imwrite(path: string, img: Image): void {
  const ext = extname(path).toLowerCase();
  if (ext === ".png") writeFileSync(path, encodePng(img));
  else if (ext === ".jpg" || ext === ".jpeg") writeFileSync(path, encodeJpeg(img));
  else throw new Error(`could not encode ${path}: unsupported image format ${ext}`);
}
