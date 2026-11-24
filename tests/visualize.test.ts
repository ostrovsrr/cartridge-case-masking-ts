/**
 * Display code (`src/visualize.ts`, `src/image.ts`): exact mask blending away from the drawn
 * outlines, the canonical rotation direction (Python `test_qc_and_canonical_orientation`), the
 * `--brief-colours` swap, figure sizes and the PNG codec.
 */
import { afterEach, beforeAll, describe, expect, it } from "vitest";
import { decodePng, encodePng, newImage, pixel, type Image } from "../src/image.js";
import { BREECH_FACE, FIRING_PIN_IMPRESSION, FIRING_PIN_SHEAR, Result, segment } from "../src/segment.js";
import { makeHead } from "../src/synthetic.js";
import {
  ARROW,
  COLOURS,
  LIGHT_BLUE,
  PURPLE,
  canonical,
  contactSheet,
  cropBox,
  cropTo,
  flatMask,
  overlay,
  render,
  setBriefColours,
  sideBySide,
} from "../src/visualize.js";

/** Interior pixels of label `v`: the whole (2k+1)^2 neighbourhood carries `v`, away from the strike circle and arrow. */
function interior(res: Result, v: number, k = 4): [number, number][] {
  const L = res.labels!;
  const out: [number, number][] = [];
  const [cx, cy] = res.fpCoreCenterPx!;
  const [tx, ty] = res.dragTipPx ?? [cx, cy];
  const keepOut = Math.max(4, Math.round(0.25 * res.fpCoreRadiusPx!)) + 8;
  for (let y = k; y < L.rows - k; y += 7) {
    for (let x = k; x < L.cols - k; x += 7) {
      let ok = true;
      for (let dy = -k; dy <= k && ok; dy++) for (let dx = -k; dx <= k && ok; dx++) ok = L.data[(y + dy) * L.cols + x + dx] === v;
      if (!ok) continue;
      // distance to the arrow segment and the strike circle
      const sx = tx - cx, sy = ty - cy;
      const t = Math.max(0, Math.min(1, ((x - cx) * sx + (y - cy) * sy) / Math.max(sx * sx + sy * sy, 1e-9)));
      if (Math.hypot(x - cx - t * sx, y - cy - t * sy) < keepOut) continue;
      out.push([x, y]);
    }
  }
  return out;
}

describe("overlay", () => {
  let res: Result;
  let base: Image;
  let ov: Image;
  beforeAll(() => {
    res = segment(makeHead({ dragAngleDeg: 30.0 })[0]);
    base = render(res);
    ov = overlay(res, base);
  });

  it("renders the height map as grey BGR", () => {
    expect(base.channels).toBe(3);
    expect([base.width, base.height]).toEqual([res.labels!.cols, res.labels!.rows]);
    const [b, g, r] = pixel(base, 437, 300);
    expect(b).toBe(g);
    expect(g).toBe(r);
  });

  it("blends the label colours exactly (uint8 truncation, alpha 0.55)", () => {
    for (const v of [BREECH_FACE, FIRING_PIN_IMPRESSION]) {
      const pts = interior(res, v);
      expect(pts.length).toBeGreaterThan(20);
      const c = COLOURS[v];
      for (const [x, y] of pts) {
        const b = pixel(base, x, y);
        const want = b.map((bv, k) => Math.trunc(bv * (1 - 0.55) + c[k] * 0.55));
        expect(pixel(ov, x, y)).toEqual(want);
      }
    }
    // unlabelled pixels keep the rendering
    for (const [x, y] of interior(res, 0).slice(0, 200)) expect(pixel(ov, x, y)).toEqual(pixel(base, x, y));
  });

  it("draws the strike circle and drag arrow in the arrow colour", () => {
    const [cx, cy] = res.fpCoreCenterPx!;
    const [tx, ty] = res.dragTipPx!;
    let hits = 0;
    for (let s = 0.4; s <= 0.8; s += 0.05) {
      const p = pixel(ov, Math.round(cx + s * (tx - cx)), Math.round(cy + s * (ty - cy)));
      if (Math.abs(p[0] - ARROW[0]) < 30 && Math.abs(p[1] - ARROW[1]) < 30 && Math.abs(p[2] - ARROW[2]) < 30) hits++;
    }
    expect(hits).toBeGreaterThan(5);
  });

  it("flat mask uses the legend colours on white", () => {
    const fm = flatMask(res);
    const L = res.labels!.data;
    for (let i = 0; i < L.length; i += 97) {
      const want = L[i] ? [...COLOURS[L[i]]] : res.valid!.data[i] ? [255, 255, 255] : [235, 235, 235];
      expect([fm.data[i * 3], fm.data[i * 3 + 1], fm.data[i * 3 + 2]]).toEqual(want);
    }
  });

  it("figures have the Python layout sizes", () => {
    const sbs = sideBySide(res);
    expect([sbs.width, sbs.height]).toEqual([4 * (640 + 8), 34 + 640 + 70]);
    const box = cropBox(res);
    const sheet = contactSheet([
      ["a", cropTo(base, box), cropTo(ov, box)],
      ["b", cropTo(base, box), cropTo(ov, box)],
    ]);
    expect([sheet.width, sheet.height]).toEqual([4 * (2 * 300 + 4 + 12), 26 + 300 + 12]);
  });
});

describe("canonical orientation", () => {
  it("rotates the drag to 6 o'clock", () => {
    const res = segment(makeHead({ dragAngleDeg: 20.0 })[0]);
    expect(res.qcScore).toBeGreaterThanOrEqual(0);
    expect(res.qcScore).toBeLessThanOrEqual(1);
    expect(res.needsReview).toBe(false);
    const L = res.labels!;
    const m = new Uint8Array(L.data.length);
    for (let i = 0; i < m.length; i++) m[i] = L.data[i] === FIRING_PIN_SHEAR ? 255 : 0;
    const tail = canonical(res, { width: L.cols, height: L.rows, channels: 1, data: m });
    let n = 0, sx = 0, sy = 0;
    for (let y = 0; y < tail.height; y++) for (let x = 0; x < tail.width; x++) if (tail.data[y * tail.width + x]) { n++; sx += x; sy += y; }
    expect(n).toBeGreaterThan(0);
    const h = tail.height, w = tail.width;
    expect(sy / n - h / 2).toBeGreaterThan(3 * Math.abs(sx / n - w / 2)); // drag now points to 6 o'clock
  });

  it("returns the image unchanged without a primer", () => {
    const res = new Result("x", 8, { labels: { data: new Uint8Array(4), rows: 2, cols: 2 } });
    const img = newImage(2, 2, 3, 7);
    expect(canonical(res, img)).toBe(img);
  });
});

describe("brief colours", () => {
  afterEach(() => setBriefColours(false));

  it("swaps the display colours of labels 3 and 4 only", () => {
    const before = { ...COLOURS };
    setBriefColours(true);
    expect(COLOURS[FIRING_PIN_IMPRESSION]).toBe(PURPLE);
    expect(COLOURS[FIRING_PIN_SHEAR]).toBe(LIGHT_BLUE);
    expect(COLOURS[BREECH_FACE]).toBe(before[BREECH_FACE]);
    setBriefColours(false);
    expect(COLOURS[FIRING_PIN_IMPRESSION]).toBe(LIGHT_BLUE);
    expect(COLOURS[FIRING_PIN_SHEAR]).toBe(PURPLE);
  });
});

describe("PNG codec", () => {
  it("round-trips grey and BGR images losslessly", () => {
    const g = newImage(37, 23, 1);
    for (let i = 0; i < g.data.length; i++) g.data[i] = (i * 31 + (i >> 3)) & 255;
    expect(decodePng(encodePng(g))).toEqual(g);
    const c = newImage(19, 41, 3);
    for (let i = 0; i < c.data.length; i++) c.data[i] = (i * 7 + (i % 13) * 19) & 255;
    const back = decodePng(encodePng(c));
    expect(back).toEqual(c);
    // RGB on disk: byte 0 of the first pixel is the red (last BGR) channel
    expect(encodePng(c)[25]).toBe(2); // colour type RGB
  });
});
