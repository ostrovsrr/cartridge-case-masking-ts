/**
 * Figures for the revision comparison and the failure gallery (the OpenCV drawing of
 * `compare_revisions.py` and `failure_gallery.py`), built on `src/visualize.ts` and `src/image.ts`.
 *
 * Layout, colours, crops and captions follow the Python scripts; glyphs come from a sans-serif
 * face instead of OpenCV's Hershey font (see `putText`).
 */
import { writeFileSync } from "node:fs";
import {
  arrowedLine, circle, crop, css, draw, encodeJpeg, grayToBgr, hstack, newImage, pad, paste, putText, px, resize, vstack,
  type BGR, type Image,
} from "../../src/image.js";
import { findExternalContours } from "../../src/ndimage/contours.js";
import { ARROW, COLOURS } from "../../src/visualize.js";
import { fmtF } from "../../src/steps/np.js";

type Summary = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
type U8Grid = { data: Uint8Array; rows: number; cols: number };

/** `render(res)[..., 0]` of a visualize module (this repository's or an older checkout's). */
export function renderGray(mod: { render: (res: any) => Image }, res: unknown): U8Grid {
  const img = mod.render(res);
  const g = new Uint8Array(img.width * img.height);
  for (let i = 0; i < g.length; i++) g[i] = img.data[i * img.channels]!;
  return { data: g, rows: img.height, cols: img.width };
}

/** A grey PNG (as stored by `run`) as an image. */
export function greyImage(g: U8Grid | Image): Image {
  return "width" in g ? g : { width: g.cols, height: g.rows, channels: 1, data: g.data };
}

function colourItems(): [number, BGR][] {
  return Object.keys(COLOURS).map((k) => [Number(k), COLOURS[Number(k)]!]);
}

/** Masks blended at 55% over the grey rendering, without outlines (`ov[m] = ov*0.45 + col*0.55`). */
function blend(base: Image, lab: U8Grid): Image {
  const img = grayToBgr(base);
  const lut: (BGR | undefined)[] = [];
  for (const [v, c] of colourItems()) lut[v] = c;
  for (let i = 0; i < lab.data.length; i++) {
    const v = lab.data[i]!;
    if (v === 0) continue;
    const c = lut[v] ?? ([0, 0, 0] as const);
    for (let k = 0; k < 3; k++) img.data[i * 3 + k] = Math.trunc(img.data[i * 3 + k]! * 0.45 + c[k] * 0.55);
  }
  return img;
}

/** `cv2.drawContours` of every class outline, 1 px, in the class colour. */
function outlines(img: Image, lab: U8Grid): void {
  draw(img, (ctx) => {
    for (const [v, c] of colourItems()) {
      const m = new Uint8Array(lab.data.length);
      for (let i = 0; i < m.length; i++) m[i] = lab.data[i] === v ? 1 : 0;
      ctx.strokeStyle = css(c);
      ctx.lineWidth = 1;
      for (const cnt of findExternalContours({ data: m, rows: lab.rows, cols: lab.cols })) {
        ctx.beginPath();
        ctx.moveTo(px(cnt[0]![0]), px(cnt[0]![1]));
        for (let j = 1; j < cnt.length; j++) ctx.lineTo(px(cnt[j]![0]), px(cnt[j]![1]));
        ctx.closePath();
        ctx.stroke();
      }
    }
  });
}

function text(img: Image, s: string, x: number, y: number, scale: number, color: BGR): void {
  draw(img, (ctx) => putText(ctx, s, x, y, scale, color, 1));
}

const fmt2 = (v: unknown): string => fmtF(Number(v ?? 0), 2);

/** Before/after figure of one scan: [rendered | before | after] + a two-line footer (JPEG q80). */
export async function revisionFigure(old: U8Grid, nw: U8Grid, baseGrid: U8Grid, sa: Summary, sb: Summary, path: string): Promise<void> {
  const base = greyImage(baseGrid);
  const ov = (lab: U8Grid): Image => {
    const img = blend(base, lab);
    outlines(img, lab);
    return img;
  };
  const h = base.height, w = base.width;
  let [y0, y1, x0, x1] = [0, h, 0, w];
  if (sb.primer_center_px && sb.primer_radius_px) {
    const [cx, cy] = sb.primer_center_px as [number, number];
    const r = sb.primer_radius_px * 1.2;
    [y0, y1, x0, x1] = [Math.max(0, Math.trunc(cy - r)), Math.min(h, Math.trunc(cy + r)), Math.max(0, Math.trunc(cx - r)), Math.min(w, Math.trunc(cx + r))];
  }
  const size = 380;
  const tiles: Image[] = [];
  for (const [title, img] of [["rendered", grayToBgr(base)], ["before", ov(old)], ["after", ov(nw)]] as const) {
    let c = crop(img, y0, y1, x0, x1);
    const s = size / Math.max(c.width, c.height);
    c = resize(c, Math.trunc(c.width * s), Math.trunc(c.height * s), "area");
    const t = newImage(size, size + 30, 3, 255);
    paste(t, c, 0, 30);
    text(t, title, 6, 21, 0.6, [20, 20, 20]);
    tiles.push(pad(t, 0, 0, 3, 3, 255));
  }
  const row = hstack(tiles);
  const foot = newImage(row.width, 62, 3, 255);
  const info = (s: Summary): string => {
    const a = (s.areas_mm2 ?? {}) as Record<string, number>;
    return (
      `BF ${fmt2(a.breech_face_impression)}  AS ${fmt2(a.aperture_shear)}  FPI ${fmt2(a.firing_pin_impression)} mm2  ` +
      `ap=${s.aperture_detected ? "ring" : "dflt"}  qc=${fmt2(s.qc_score)}${s.needs_review ? " REVIEW" : ""}`
    );
  };
  text(foot, `${sb.file}`, 6, 16, 0.45, [60, 60, 60]);
  text(foot, "before: " + info(sa), 6, 35, 0.42, [20, 20, 20]);
  text(foot, "after:  " + info(sb) + `  shape=${sb.aperture_shape ?? "circle"}`, 6, 54, 0.42, [20, 20, 20]);
  writeFileSync(path, encodeJpeg(vstack([row, foot]), 80));
}

/** Gallery panel size [px]. */
export const T = 300;

/** Full head with the primer circle (yellow) and the primer close-up with masks, strike circle and drag arrow. */
export function galleryPanels(base: Image, lab: U8Grid, s: Summary): [Image, Image] {
  const full = grayToBgr(base);
  const ov = blend(base, lab);
  const h = base.height, w = base.width;
  const lw = Math.max(2, Math.floor(h / 250));
  const [cx, cy] = s.primer_center_px as [number, number];
  const rp = s.primer_radius_px as number;
  draw(full, (ctx) => circle(ctx, Math.trunc(cx), Math.trunc(cy), Math.trunc(rp), [0, 200, 255], lw));
  if (s.fp_core_center_px) {
    const [fx, fy] = s.fp_core_center_px as [number, number];
    draw(ov, (ctx) => {
      circle(ctx, Math.trunc(fx), Math.trunc(fy), Math.trunc(s.fp_core_radius_px * 0.3), ARROW, lw);
      if (s.drag_detected && s.drag_tip_px) {
        const [tx, ty] = s.drag_tip_px as [number, number];
        arrowedLine(ctx, Math.trunc(fx), Math.trunc(fy), Math.trunc(tx), Math.trunc(ty), ARROW, lw, 0.15);
      }
    });
  }
  const r = rp * 1.2;
  const box = [Math.max(0, Math.trunc(cy - r)), Math.min(h, Math.trunc(cy + r)), Math.max(0, Math.trunc(cx - r)), Math.min(w, Math.trunc(cx + r))] as const;
  return [full, crop(ov, box[0], box[1], box[2], box[3])];
}

/** Scale into a black `T` x `T` square, centred. */
export function fitPanel(img: Image): Image {
  const sc = T / Math.max(img.width, img.height);
  const r = resize(grayToBgr(img), Math.trunc(img.width * sc), Math.trunc(img.height * sc), "area");
  const out = newImage(T, T, 3, 0);
  paste(out, r, Math.floor((T - r.width) / 2), Math.floor((T - r.height) / 2));
  return out;
}

/** One gallery case: title, 2 or 4 panels, caption. */
export function galleryBlock(
  before: [Image, Image] | null, after: [Image, Image], title: string, key: string, caption: string, beforeLabel: string,
): Image {
  const tiles: Image[] = [];
  if (before) {
    tiles.push(fitPanel(before[0]), fitPanel(before[1]));
    text(tiles[0]!, beforeLabel, 6, 18, 0.45, [60, 60, 230]);
  }
  tiles.push(fitPanel(after[0]), fitPanel(after[1]));
  if (before) text(tiles[2]!, "after", 6, 18, 0.45, [60, 200, 60]);
  let row = hstack(tiles.map((t) => pad(t, 0, 0, 0, 4, 255)));
  const width = 4 * (T + 4);
  row = pad(row, 0, 0, 0, width - row.width, 255);
  const head = newImage(width, 28, 3, 255);
  text(head, `${title}: ${key.split("__").slice(2).join("__")}`, 4, 20, 0.55, [20, 20, 20]);
  const foot = newImage(width, 26, 3, 255);
  text(foot, caption, 4, 17, 0.38, [60, 60, 60]);
  return vstack([head, row, foot, newImage(width, 10, 3, 255)]);
}

/** Two columns, each case appended to the currently shorter one; white padding and gutter. */
export function galleryMosaic(blocks: Image[]): Image {
  const cols: Image[][] = [[], []];
  const height = (c: Image[]) => c.reduce((a, x) => a + x.height, 0);
  for (const b of blocks) (height(cols[0]!) <= height(cols[1]!) ? cols[0]! : cols[1]!).push(b);
  const hmax = Math.max(...cols.map(height));
  const stacked = cols
    .filter((c) => c.length)
    .map((c) => {
      const h = height(c);
      const parts = hmax > h ? [...c, newImage(c[0]!.width, hmax - h, 3, 255)] : c;
      return pad(vstack(parts), 0, 0, 0, 12, 255);
    });
  return hstack(stacked);
}

export { encodeJpeg };
