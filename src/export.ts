/**
 * Write the masks back into the X3P file (x3ptools / CSAFE convention); port of
 * `cartmask/export.py`.
 *
 * Forensic research tools (the R packages x3ptools and cmcR) read a mask stored inside the X3P
 * container as `bindata/mask.png` with a colour legend in `Record3/Mask`:
 *
 * ```xml
 * <Mask>
 *   <Background>#cd7f32</Background>
 *   <Annotations>
 *     <Annotation color="#ff0000">breech face impression</Annotation>
 *     ...
 *   </Annotations>
 * </Mask>
 * ```
 *
 * The exported file is the original scan, untouched, plus that mask at the scan's native
 * resolution, so the result opens directly in those tools.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { unzipSync, zipSync, type Zippable } from "fflate";
import { decodePng, encodePng, grayToBgr, type BGR, type Image } from "./image.js";
import { resizeNearest } from "./ndimage/interp.js";
import { APERTURE_SHEAR, BREECH_FACE, FIRING_PIN_IMPRESSION, FIRING_PIN_SHEAR, type Result } from "./segment.js";

/** x3ptools default "copper" background. */
export const BACKGROUND = "#cd7f32";
export const LEGEND: Readonly<Record<number, readonly [string, string]>> = {
  [BREECH_FACE]: ["#ff0000", "breech face impression"],
  [APERTURE_SHEAR]: ["#00b400", "aperture shear"],
  [FIRING_PIN_SHEAR]: ["#9932cc", "firing pin shear (drag channel)"],
  [FIRING_PIN_IMPRESSION]: ["#66ccff", "firing pin impression"],
};

const legendKeys = (): number[] => Object.keys(LEGEND).map(Number);

export function hexToBgr(h: string): BGR {
  const s = h.replace(/^#+/, "");
  return [parseInt(s.slice(4, 6), 16), parseInt(s.slice(2, 4), 16), parseInt(s.slice(0, 2), 16)];
}

/** Colour mask (BGR) at native scan resolution, rows top-to-bottom as displayed. */
export function maskImage(res: Result, nativeShape: readonly [number, number]): Image {
  if (res.labels == null) throw new Error(`${res.name}: result carries no label image`);
  const [ny, nx] = nativeShape;
  const lab = resizeNearest(res.labels, ny, nx);
  const data = new Uint8Array(ny * nx * 3);
  const lut: BGR[] = new Array(256).fill(hexToBgr(BACKGROUND));
  for (const v of legendKeys()) lut[v] = hexToBgr(LEGEND[v][0]);
  for (let i = 0; i < ny * nx; i++) {
    const c = lut[lab.data[i]];
    data[i * 3] = c[0]; data[i * 3 + 1] = c[1]; data[i * 3 + 2] = c[2];
  }
  return { width: nx, height: ny, channels: 3, data };
}

/** PNG bytes of {@link maskImage} (8-bit RGB, like `cv2.imencode(".png", img)`). */
export function maskPng(res: Result, nativeShape: readonly [number, number]): Uint8Array {
  return encodePng(maskImage(res, nativeShape));
}

/** The `Record3/Mask` legend listing the labels in `present`. */
export function maskXml(present: ReadonlySet<number>): string {
  const ann = legendKeys()
    .filter((v) => present.has(v))
    .map((v) => `<Annotation color="${LEGEND[v][0]}">${LEGEND[v][1]}</Annotation>`)
    .join("");
  return `<Mask><Background>${BACKGROUND}</Background><Annotations>${ann}</Annotations></Mask>`;
}

/** Copy `src` to `dst` adding `bindata/mask.png` and the legend; returns `dst`. */
export function exportX3P(src: string, res: Result, dst: string): string {
  if (res.labels == null) throw new Error(`${res.name}: result carries no label image`);
  const files = unzipSync(readFileSync(src));
  const names = Object.keys(files);
  const xmlName = names.find((n) => n.toLowerCase().endsWith("main.xml"));
  if (xmlName === undefined) throw new Error("StopIteration: no main.xml in archive");
  let xml = new TextDecoder("utf-8").decode(files[xmlName]);
  const mx = /<(?:\w+:)?SizeX>\s*(\d+)/.exec(xml);
  const my = /<(?:\w+:)?SizeY>\s*(\d+)/.exec(xml);
  if (!mx || !my) throw new Error("main.xml has no SizeX/SizeY");
  const nx = parseInt(mx[1], 10), ny = parseInt(my[1], 10);
  xml = xml.replace(/<(\w+:)?Mask>[\s\S]*?<\/(\w+:)?Mask>/g, ""); // replace old mask
  const present = new Set<number>(res.labels.data);
  const mxml = maskXml(present);
  let n = 0;
  xml = xml.replace(/(<\/(?:\w+:)?Record3>)/, (m) => { n++; return mxml + m; });
  if (n !== 1) throw new Error("main.xml has no Record3 element");
  const png = maskPng(res, [ny, nx]);
  const out: Zippable = {};
  for (const name of names) {
    if (name.toLowerCase().endsWith("mask.png")) continue;
    out[name] = name === xmlName ? new TextEncoder().encode(xml) : files[name];
  }
  out["bindata/mask.png"] = png;
  writeFileSync(dst, zipSync(out, { level: 6 }));
  return dst;
}

/** `[BGR mask image, main.xml text]` from an exported file - for tests. */
export function readMask(path: string): [Image, string] {
  const files = unzipSync(readFileSync(path));
  const names = Object.keys(files);
  const pngName = names.find((n) => n.endsWith("mask.png"));
  const xmlName = names.find((n) => n.toLowerCase().endsWith("main.xml"));
  if (pngName === undefined || xmlName === undefined) throw new Error(`${path}: no mask.png / main.xml`);
  const img = grayToBgr(decodePng(files[pngName])); // IMREAD_COLOR
  return [img, new TextDecoder("utf-8").decode(files[xmlName])];
}
