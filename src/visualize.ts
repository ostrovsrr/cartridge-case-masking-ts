/**
 * Overlay rendering, side-by-side figures and contact sheets (port of `cartmask/visualize.py`).
 *
 * Images are {@link Image}s in BGR order, so the colour constants are the Python tuples verbatim.
 * Shapes are drawn with `@napi-rs/canvas`; text uses a sans-serif face instead of OpenCV's
 * Hershey font, with the same positions and content.
 */
import { findExternalContours } from "./ndimage/contours.js";
import { rotationMatrix } from "./ndimage/interp.js";
import { hillshade } from "./preprocess.js";
import { APERTURE_SHEAR, BREECH_FACE, FIRING_PIN_IMPRESSION, FIRING_PIN_SHEAR, type Result } from "./segment.js";
import { fmtF, pyIntRound } from "./steps/np.js";
import {
  arrowedLine,
  circle,
  cloneImage,
  crop,
  draw,
  fromGrid,
  grayToBgr,
  hstack,
  line,
  newImage,
  pad,
  paste,
  px,
  putText,
  rectangle,
  resize,
  vstack,
  warp,
  css,
  type BGR,
  type Image,
} from "./image.js";
import type { Grid } from "./ndimage/grid.js";

export type { BGR, Image } from "./image.js";

export const PURPLE: BGR = [200, 70, 150];
export const LIGHT_BLUE: BGR = [230, 220, 90];
/**
 * BGR display colours. Default ("physical") convention: the drag channel the pin sheared is
 * firing-pin shear (purple) and the round strike is the firing-pin impression (light blue). The
 * brief's figures colour the round strike purple and the tail light blue; `setBriefColours(true)`
 * (CLI flag `--brief-colours`) swaps the two display colours to match them. Label IDs never
 * change: 3 is always the drag channel and 4 the round strike.
 */
export const COLOURS: Record<number, BGR> = {
  [BREECH_FACE]: [40, 40, 220], // red
  [APERTURE_SHEAR]: [60, 200, 60], // green
  [FIRING_PIN_SHEAR]: PURPLE,
  [FIRING_PIN_IMPRESSION]: LIGHT_BLUE,
};
export const ARROW: BGR = [255, 60, 20]; // blue
export const LEGEND: ReadonlyArray<readonly [number | null, string]> = [
  [BREECH_FACE, "1 Breech-face impression"],
  [APERTURE_SHEAR, "2 Aperture shear"],
  [FIRING_PIN_SHEAR, "3 Firing-pin shear (drag channel)"],
  [FIRING_PIN_IMPRESSION, "4 Firing-pin impression"],
  [null, "5 Firing-pin drag (arrow)"],
];

/** `COLOURS.items()` in label order. */
function colourItems(): [number, BGR][] {
  return Object.keys(COLOURS).map((k) => [Number(k), COLOURS[Number(k)]]);
}

/**
 * Switch the display colours of labels 3 and 4 to the brief's figures (or back).
 *
 * With `enabled` the round strike (label 4, firing-pin impression) is drawn purple and the drag
 * channel (label 3, firing-pin shear) light blue, as in both figures of the brief. The change is
 * in place, so every module that imported {@link COLOURS} follows it. Label IDs, mask files and
 * the X3P export colours are unaffected.
 */
export function setBriefColours(enabled = true): void {
  COLOURS[FIRING_PIN_SHEAR] = enabled ? LIGHT_BLUE : PURPLE;
  COLOURS[FIRING_PIN_IMPRESSION] = enabled ? PURPLE : LIGHT_BLUE;
}

/** The label grid shared by 2D and 3D results. */
function labelsOf(res: Result): Grid<Uint8Array> {
  if (res.labels == null) throw new Error(`${res.name}: result carries no label image`);
  return res.labels;
}

/** Shaded-relief rendering of the (working-resolution) height map: BGR, light from the upper left, invalid pixels black. */
export function render(res: Result): Image {
  if (res.z == null || res.valid == null) throw new Error(`${res.name}: result carries no height map`);
  return grayToBgr(fromGrid(hillshade(res.z, res.umPerPx * 1e-6, { valid: res.valid })));
}

/**
 * Masks blended over the rendering, with outlines, strike centre and drag arrow.
 *
 * @param base BGR background; `render(res)` when omitted.
 * @param alpha Opacity of the mask colours.
 * @returns BGR image, same size as `base`.
 */
export function overlay(res: Result, base?: Image | null, alpha = 0.55): Image {
  const labels = labelsOf(res);
  const b = base ?? render(res);
  const out = cloneImage(b);
  const cols = colourItems();
  const lut: (BGR | null)[] = new Array(256).fill(null);
  for (const [v, c] of cols) lut[v] = c;
  const L = labels.data, d = out.data, s = b.data;
  for (let i = 0; i < L.length; i++) {
    if (L[i] > 0) {
      const c = lut[L[i]] ?? ([0, 0, 0] as BGR);
      for (let k = 0; k < 3; k++) d[i * 3 + k] = Math.trunc(s[i * 3 + k] * (1 - alpha) + c[k] * alpha);
    }
  }
  const lw = Math.max(2, Math.floor(b.height / 300));
  draw(out, (ctx) => {
    // thin outlines make adjacent regions readable
    for (const [v, c] of cols) {
      const m = new Uint8Array(L.length);
      for (let i = 0; i < L.length; i++) m[i] = L[i] === v ? 1 : 0;
      ctx.strokeStyle = css(c);
      ctx.lineWidth = 1;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      for (const cnt of findExternalContours({ data: m, rows: labels.rows, cols: labels.cols })) {
        ctx.beginPath();
        ctx.moveTo(px(cnt[0][0]), px(cnt[0][1]));
        for (let j = 1; j < cnt.length; j++) ctx.lineTo(px(cnt[j][0]), px(cnt[j][1]));
        ctx.closePath();
        ctx.stroke();
      }
    }
    if (res.fpCoreCenterPx != null && res.fpCoreRadiusPx != null) {
      const [cx, cy] = res.fpCoreCenterPx;
      const rr = Math.max(4, pyIntRound(0.25 * res.fpCoreRadiusPx));
      circle(ctx, pyIntRound(cx), pyIntRound(cy), rr, ARROW, lw);
      if (res.dragDetected && res.dragTipPx != null) {
        const [tx, ty] = res.dragTipPx;
        const len = Math.hypot(tx - cx, ty - cy);
        arrowedLine(ctx, Math.trunc(cx), Math.trunc(cy), Math.trunc(tx), Math.trunc(ty), ARROW, lw, Math.min(0.35, 18 / Math.max(len, 1)));
      }
    }
  });
  return out;
}

/** Label image in legend colours on white (invalid data light grey); BGR. */
export function flatMask(res: Result): Image {
  const labels = labelsOf(res);
  const out = newImage(labels.cols, labels.rows, 3, 255);
  const d = out.data;
  if (res.valid != null) {
    const v = res.valid.data;
    for (let i = 0; i < v.length; i++) if (!v[i]) d[i * 3] = d[i * 3 + 1] = d[i * 3 + 2] = 235;
  }
  for (const [lv, c] of colourItems()) {
    const L = labels.data;
    for (let i = 0; i < L.length; i++) if (L[i] === lv) { d[i * 3] = c[0]; d[i * 3 + 1] = c[1]; d[i * 3 + 2] = c[2]; }
  }
  return out;
}

/** Python `f"{mm:g}"` for the scale-bar values used here. */
const fmtGeneral = (v: number): string => String(Number(v.toPrecision(6)));

/** Draw a `mm`-long scale bar in the lower-left corner, in place. */
function scaleBar(img: Image, umPerPx: number, mm = 1.0): void {
  const L = pyIntRound((mm * 1000) / umPerPx);
  const h = img.height;
  const x0 = 15, y0 = h - 20;
  draw(img, (ctx) => {
    line(ctx, x0, y0, x0 + L, y0, [255, 255, 255], 5);
    line(ctx, x0, y0, x0 + L, y0, [0, 0, 0], 2);
    putText(ctx, `${fmtGeneral(mm)} mm`, x0, y0 - 10, 0.6, [0, 0, 0], 3);
    putText(ctx, `${fmtGeneral(mm)} mm`, x0, y0 - 10, 0.6, [255, 255, 255], 1);
  });
}

/** Scale `img` to fit a `size` x `size` white canvas, centred. */
function fit(img: Image, size: number): Image {
  const { width: w, height: h } = img;
  const s = size / Math.max(h, w);
  const r = resize(grayToBgr(img), Math.trunc(w * s), Math.trunc(h * s), "area");
  const canvas = newImage(size, size, 3, 255);
  paste(canvas, r, Math.floor((size - r.width) / 2), Math.floor((size - r.height) / 2));
  return canvas;
}

/**
 * Square window around the primer, clipped to the image.
 *
 * @param pad Half-size of the window in primer radii.
 * @returns `[y0, y1, x0, x1]`; the whole image when no primer was found.
 */
export function cropBox(res: Result, pad_ = 1.25): [number, number, number, number] {
  const labels = labelsOf(res);
  const h = labels.rows, w = labels.cols;
  if (res.primerRadiusPx == null || res.primerCenterPx == null) return [0, h, 0, w];
  const [cx, cy] = res.primerCenterPx;
  const r = res.primerRadiusPx * pad_;
  return [Math.max(0, Math.trunc(cy - r)), Math.min(h, Math.trunc(cy + r)), Math.max(0, Math.trunc(cx - r)), Math.min(w, Math.trunc(cx + r))];
}

/** `img[y0:y1, x0:x1]` for a {@link cropBox}. */
export function cropTo(img: Image, box: readonly [number, number, number, number]): Image {
  return crop(img, box[0], box[1], box[2], box[3]);
}

/**
 * Four-panel figure: [full scan | primer close-up | masked close-up | flat mask] + legend and a
 * measurement footer.
 *
 * @param size Edge length of each square panel in pixels.
 */
export function sideBySide(res: Result, size = 640): Image {
  const base = render(res);
  const ov = overlay(res, base);
  const [y0, y1, x0, x1] = cropBox(res);
  const scale = size / Math.max(y1 - y0, x1 - x0);
  const panels: [string, Image][] = [];
  const full = cloneImage(base);
  if (res.primerRadiusPx != null) {
    draw(full, (ctx) => rectangle(ctx, x0, y0, x1, y1, [0, 200, 255], Math.max(2, Math.floor(base.height / 250))));
  }
  panels.push(["Original scan (rendered topography)", fit(full, size)]);
  for (const [title, img] of [
    ["Primer region - original", base],
    ["Generated masks", ov],
    ["Label mask", flatMask(res)],
  ] as [string, Image][]) {
    let c = crop(img, y0, y1, x0, x1);
    c = resize(c, Math.trunc(c.width * scale), Math.trunc(c.height * scale), scale < 1 ? "area" : "linear");
    scaleBar(c, res.umPerPx / scale);
    panels.push([title, fit(c, size)]);
  }
  const head = 34;
  const tiles: Image[] = [];
  for (const [title, img] of panels) {
    const t = newImage(size, head + size, 3, 255);
    paste(t, img, 0, head);
    draw(t, (ctx) => putText(ctx, title, 8, 24, 0.62, [20, 20, 20], 1));
    tiles.push(t);
  }
  const row = hstack(tiles.map((t) => pad(t, 0, 0, 4, 4, 255)));

  // footer: legend + key measurements
  const foot = newImage(row.width, 70, 3, 255);
  const s = res.summary();
  const info = [res.name, `status=${res.status}`];
  if (res.primerRadiusPx) info.push(`primer d=${fmtF(s.primer_diameter_mm as number, 2)} mm`);
  if (res.fpCoreRadiusPx) {
    info.push(`FP d=${fmtF(s.fp_core_diameter_mm as number, 2)} mm, depth=${fmtF(res.fpMaxDepthUm ?? NaN, 0)} um`);
  }
  if (res.dragDetected) info.push(`drag ${fmtF(res.dragLengthMm ?? NaN, 2)} mm @ ${fmtF(res.dragAngleDeg ?? NaN, 0)} deg`);
  draw(foot, (ctx) => {
    let x = 10;
    for (const [v, text] of LEGEND) {
      if (v === null) arrowedLine(ctx, x, 22, x + 26, 22, ARROW, 3, 0.4);
      else rectangle(ctx, x, 12, x + 24, 32, COLOURS[v], -1);
      putText(ctx, text, x + 32, 28, 0.55, [20, 20, 20], 1);
      x += 40 + 9 * text.length + 25;
    }
    putText(ctx, info.join("   "), 10, 58, 0.5, [90, 90, 90], 1);
  });
  return vstack([row, foot]);
}

/**
 * Grid of before/after pairs.
 *
 * @param items `[title, original crop, overlay crop]` per scan; not empty.
 * @param tile Edge length of each image in pixels.
 * @param cols Pairs per row; the last row is padded with white.
 */
export function contactSheet(items: ReadonlyArray<readonly [string, Image, Image]>, tile = 300, cols = 4): Image {
  if (items.length === 0) throw new Error("contactSheet needs at least one item");
  const pairs: Image[] = [];
  for (const [title, a, b] of items) {
    const p = hstack([fit(a, tile), newImage(4, tile, 3, 255), fit(b, tile)]);
    const t = newImage(p.width, 26, 3, 255);
    draw(t, (ctx) => putText(ctx, title.slice(0, 60), 6, 18, 0.48, [20, 20, 20], 1));
    pairs.push(pad(vstack([t, p]), 6, 6, 6, 6, 255));
  }
  while (pairs.length % cols) pairs.push(newImage(pairs[0].width, pairs[0].height, 3, 255));
  const rows: Image[] = [];
  for (let i = 0; i < pairs.length; i += cols) rows.push(hstack(pairs.slice(i, i + cols)));
  return vstack(rows);
}

/**
 * Primer close-up rotated so the firing-pin drag points to 6 o'clock.
 *
 * Examiners compare cases in a common orientation (the brief's own masked example is rotated this
 * way). Without a drag the crop is not rotated.
 *
 * @param img Image on the working grid (rendering, overlay or a grey mask).
 * @param pad Half-size of the output in primer radii.
 * @returns Square crop centred on the primer; `img` unchanged when no primer was found.
 */
export function canonical(res: Result, img: Image, pad_ = 1.25): Image {
  if (res.primerCenterPx == null || res.primerRadiusPx == null) return img;
  const [cx, cy] = res.primerCenterPx;
  const angle = res.dragDetected && res.dragAngleDeg != null ? res.dragAngleDeg - 90.0 : 0.0;
  const M = rotationMatrix(cx, cy, angle, 1.0);
  const r = Math.trunc(res.primerRadiusPx * pad_);
  M[2] += r - cx;
  M[5] += r - cy;
  return warp(img, M, 2 * r, 2 * r);
}
