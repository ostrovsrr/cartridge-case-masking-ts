/**
 * Synthetic cartridge-case head topography with known ground truth.
 *
 * Used by the test-suite to check the pipeline end to end without real data:
 * a flat case head, a primer-pocket gap, a primer face with fine roughness, a
 * hemispherical firing-pin crater with an optional drag tail, and a raised
 * flow-back ring marking the firing-pin aperture.
 *
 * Optional features with known masks (all off by default, so the default head is
 * unchanged):
 *
 * - `rectAperture` - a rectangular (slotted, Glock-like) flow-back plateau with
 *   straight parallel step edges instead of the round ring;
 * - `apertureStepUm` - a flat raised plateau inside the round aperture whose
 *   only trace is a step at the aperture edge (no ring profile);
 * - `shearUm` - a band of striations parallel to the drag axis at the
 *   aperture edge, on the side opposite the drag (aperture shear);
 * - `roughBandUm` - the same band filled with isotropic roughness instead
 *   (rough metal that is *not* shear).
 *
 * Port of `cartmask/synthetic.py`.  The geometry is identical to the Python
 * version (same formulas, same pixel grid); the random parts (fine roughness
 * and the `roughBandUm` texture) come from a seeded xoshiro128** generator with
 * Box-Muller normals instead of NumPy's `default_rng` (PCG64 + ziggurat), so the
 * noise realisation differs from Python for the same `seed` while having the
 * same distribution (normal, sigma 3 um, Gaussian-smoothed with sigma 1 px).
 * The writer `write_x3p` of the Python module lives in `x3p.ts` (`writeX3P`).
 */
import { gaussianFilter } from "./ndimage/index.js";
import type { Grid, Mask } from "./ndimage/index.js";
import type { Surface } from "./x3p.js";

// ---------------------------------------------------------------- seeded PRNG

/** Python's built-in `round()` (round half to even). */
function pyRound(v: number): number {
  const f = Math.floor(v);
  const d = v - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/**
 * Seeded normal generator (xoshiro128** state seeded by splitmix32, Box-Muller
 * transform).  Deterministic for a given integer seed, but *not* the NumPy stream.
 */
export class NormalRng {
  private s0: number;
  private s1: number;
  private s2: number;
  private s3: number;
  private spare: number | null = null;

  constructor(seed: number) {
    let x = (Math.trunc(seed) ^ 0x9e3779b9) >>> 0;
    const next = (): number => {
      // splitmix32
      x = (x + 0x9e3779b9) >>> 0;
      let z = x;
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
      return (z ^ (z >>> 16)) >>> 0;
    };
    this.s0 = next();
    this.s1 = next();
    this.s2 = next();
    this.s3 = next();
    if ((this.s0 | this.s1 | this.s2 | this.s3) === 0) this.s0 = 1;
  }

  /** Next 32-bit unsigned integer (xoshiro128**). */
  nextU32(): number {
    const result = Math.imul(rotl(Math.imul(this.s1, 5) >>> 0, 7), 9) >>> 0;
    const t = (this.s1 << 9) >>> 0;
    this.s2 = (this.s2 ^ this.s0) >>> 0;
    this.s3 = (this.s3 ^ this.s1) >>> 0;
    this.s1 = (this.s1 ^ this.s2) >>> 0;
    this.s0 = (this.s0 ^ this.s3) >>> 0;
    this.s2 = (this.s2 ^ t) >>> 0;
    this.s3 = rotl(this.s3, 11);
    return result;
  }

  /** Uniform double in [0, 1) with 53 random bits. */
  random(): number {
    const a = this.nextU32() >>> 5;
    const b = this.nextU32() >>> 6;
    return (a * 67108864 + b) / 9007199254740992;
  }

  /** Standard normal deviate scaled to `N(loc, scale)`. */
  normal(loc = 0, scale = 1): number {
    if (this.spare !== null) {
      const v = this.spare;
      this.spare = null;
      return loc + scale * v;
    }
    let u = 0;
    while (u === 0) u = this.random();
    const v = this.random();
    const r = Math.sqrt(-2 * Math.log(u));
    this.spare = r * Math.sin(2 * Math.PI * v);
    return loc + scale * r * Math.cos(2 * Math.PI * v);
  }

  /** Grid of normal deviates (`rng.normal(loc, scale, (rows, cols))`). */
  normalGrid(rows: number, cols: number, loc = 0, scale = 1): Grid<Float64Array> {
    const data = new Float64Array(rows * cols);
    for (let i = 0; i < data.length; i++) data[i] = this.normal(loc, scale);
    return { data, rows, cols };
  }
}

function rotl(x: number, k: number): number {
  return ((x << k) | (x >>> (32 - k))) >>> 0;
}

// ---------------------------------------------------------------- head model

/** Keyword arguments of `makeHead` (Python names in camelCase; same defaults). */
export interface HeadOptions {
  sizeMm?: number;
  umPerPx?: number;
  primerCenterMm?: [number, number];
  primerRadiusMm?: number;
  pinOffsetMm?: [number, number];
  pinRadiusMm?: number;
  craterDepthUm?: number;
  /** `null` = no drag tail. */
  dragAngleDeg?: number | null;
  dragLengthMm?: number;
  apertureRadiusMm?: number;
  flowbackUm?: number;
  seed?: number;
  invalidCorners?: boolean;
  /** `[widthMm, lengthMm]` of a slotted aperture, or `null` for the round ring. */
  rectAperture?: [number, number] | null;
  rectAngleDeg?: number;
  apertureStepUm?: number;
  shearUm?: number;
  /** `null` = opposite the drag (or 180 deg without a drag). */
  shearAngleDeg?: number | null;
  shearPeriodMm?: number;
  roughBandUm?: number;
}

/**
 * Ground truth of a synthetic head (Python `truth` dict; keys in camelCase).
 * Centres are `[x, y]` in pixels; masks are 0/1 grids, empty when the feature is off.
 */
export interface HeadTruth {
  primerCenterPx: [number, number];
  primerRadiusPx: number;
  pinCenterPx: [number, number];
  pinRadiusPx: number;
  apertureRadiusPx: number;
  dragAngleDeg: number | null;
  tailMask: Mask;
  rectMask: Mask;
  shearMask: Mask;
}

const DEFAULTS: Required<HeadOptions> = {
  sizeMm: 7.0,
  umPerPx: 8.0,
  primerCenterMm: [0.15, -0.1],
  primerRadiusMm: 2.2,
  pinOffsetMm: [0.1, 0.05],
  pinRadiusMm: 0.7,
  craterDepthUm: 400.0,
  dragAngleDeg: 30.0,
  dragLengthMm: 0.45,
  apertureRadiusMm: 1.1,
  flowbackUm: 60.0,
  seed: 0,
  invalidCorners: true,
  rectAperture: null,
  rectAngleDeg: 0.0,
  apertureStepUm: 0.0,
  shearUm: 0.0,
  shearAngleDeg: null,
  shearPeriodMm: 0.06,
  roughBandUm: 0.0,
};

/**
 * `np.hypot` as computed by the reference NumPy build (sqrt of the sum of squares);
 * `Math.hypot` rounds differently in the last ulp and flips pixels lying exactly
 * on a radius threshold (e.g. the primer-pocket gap edge).
 */
const hypot = (x: number, y: number): number => Math.sqrt(x * x + y * y);

const clip01 = (v: number): number => (v < 0 ? 0 : v > 1 ? 1 : v);
const deg2rad = (d: number): number => (d * Math.PI) / 180;

/**
 * Deterministic part of `makeHead`: the noiseless height map in micrometres
 * (before the fine roughness, the `roughBandUm` texture and the invalid corners)
 * plus the truth.  Exposed so the geometry can be checked against Python exactly.
 */
export function headGeometry(options: HeadOptions = {}): { zUm: Grid<Float64Array>; truth: HeadTruth; n: number; px: number } {
  const o = { ...DEFAULTS, ...options };
  const n = Math.trunc(pyRound((o.sizeMm * 1000) / o.umPerPx));
  const px = o.umPerPx / 1000.0;
  const [pcx, pcy] = o.primerCenterMm;
  const fcx = pcx + o.pinOffsetMm[0];
  const fcy = pcy + o.pinOffsetMm[1];
  const N = n * n;
  const z = new Float64Array(N); // case head at 0 um
  const rectMask = new Uint8Array(N);
  const shearMask = new Uint8Array(N);
  const tailMask = new Uint8Array(N);
  const pr = o.primerRadiusMm;

  // rectangular aperture frame
  const ra = deg2rad(o.rectAngleDeg);
  const cra = Math.cos(ra);
  const sra = Math.sin(ra);
  // shear band axis
  const shearOn = o.shearUm !== 0 || o.roughBandUm !== 0;
  const cDeg = o.shearAngleDeg !== null ? o.shearAngleDeg : o.dragAngleDeg !== null ? o.dragAngleDeg + 180.0 : 180.0;
  const c = deg2rad(cDeg);
  const sc = Math.sin(c);
  const cc = Math.cos(c);
  const halfAngle = deg2rad(50);
  // spherical cap of radius R with given depth
  const d = o.craterDepthUm / 1000;
  const R = (o.pinRadiusMm ** 2 + d ** 2) / (2 * d);
  // drag axis
  const a = o.dragAngleDeg !== null ? deg2rad(o.dragAngleDeg) : 0;
  const ux = Math.cos(a);
  const uy = Math.sin(a);

  for (let i = 0; i < n; i++) {
    const yy = (i - n / 2) * px; // mm, centred
    for (let j = 0; j < n; j++) {
      const xx = (j - n / 2) * px;
      const k = i * n + j;
      const rPrimer = hypot(xx - pcx, yy - pcy);
      const rPin = hypot(xx - fcx, yy - fcy);
      let zv = 0;
      if (rPrimer > pr && rPrimer < pr + 0.35) zv = -300.0; // primer-pocket gap
      const primer = rPrimer <= pr;
      if (primer) {
        zv = 15.0; // primer face
        // rounded primer-cup edge
        if (rPrimer > pr - 0.1) zv -= ((rPrimer - (pr - 0.1)) / 0.1) * 150.0;
      }

      if (o.rectAperture !== null) {
        // slotted aperture: flat raised plateau between straight parallel steps
        const along = (xx - fcx) * cra + (yy - fcy) * sra;
        const across = -(xx - fcx) * sra + (yy - fcy) * cra;
        if (primer && Math.abs(across) <= o.rectAperture[0] / 2 && Math.abs(along) <= o.rectAperture[1] / 2) {
          rectMask[k] = 1;
          zv += o.flowbackUm;
        }
      } else if (primer && rPin < o.apertureRadiusMm) {
        // flow-back ring inside the aperture
        zv += o.flowbackUm * clip01(1.0 - (rPin - o.pinRadiusMm) / (o.apertureRadiusMm - o.pinRadiusMm));
        if (o.apertureStepUm) zv += o.apertureStepUm;
      }

      if (shearOn) {
        const phi = Math.atan2(yy - fcy, xx - fcx);
        const dphi = phi - c;
        if (
          primer &&
          rPin > 1.05 * o.pinRadiusMm &&
          rPin < o.apertureRadiusMm &&
          Math.abs(Math.atan2(Math.sin(dphi), Math.cos(dphi))) <= halfAngle
        ) {
          shearMask[k] = 1;
          if (o.shearUm) {
            // striations run along the shear axis (the line through the pin centre and the band)
            const acrossAxis = -(xx - fcx) * sc + (yy - fcy) * cc;
            zv += o.shearUm * Math.sin((2 * Math.PI * acrossAxis) / o.shearPeriodMm);
          }
        }
      }

      // hemispherical crater
      let crater = 0;
      if (rPin < o.pinRadiusMm) crater = -(Math.sqrt(R ** 2 - rPin ** 2) - (R - d)) * 1000;
      // drag tail: elongated groove continuing from the crater
      if (o.dragAngleDeg !== null) {
        const t = (xx - fcx) * ux + (yy - fcy) * uy; // along drag
        const s = -(xx - fcx) * uy + (yy - fcy) * ux; // across drag
        const halfW = o.pinRadiusMm * 0.55 * clip01(1 - (t - o.pinRadiusMm * 0.6) / (o.dragLengthMm + 0.4));
        const tail = t > 0 && t < o.pinRadiusMm + o.dragLengthMm && Math.abs(s) < halfW;
        let tailDepth = 0;
        if (tail) {
          const depth = 0.45 * o.craterDepthUm * clip01(1 - (t - o.pinRadiusMm) / o.dragLengthMm);
          tailDepth = -depth * clip01(1 - (s / Math.max(halfW, 1e-6)) ** 2);
        }
        crater = Math.min(crater, tailDepth);
        if (tail && rPin > o.pinRadiusMm && tailDepth < -20) tailMask[k] = 1;
      }
      if (primer) zv += crater;
      z[k] = zv;
    }
  }

  const truth: HeadTruth = {
    primerCenterPx: [pcx / px + n / 2, pcy / px + n / 2],
    primerRadiusPx: pr / px,
    pinCenterPx: [fcx / px + n / 2, fcy / px + n / 2],
    pinRadiusPx: o.pinRadiusMm / px,
    apertureRadiusPx: o.apertureRadiusMm / px,
    dragAngleDeg: o.dragAngleDeg,
    tailMask: { data: tailMask, rows: n, cols: n },
    rectMask: { data: rectMask, rows: n, cols: n },
    shearMask: { data: shearMask, rows: n, cols: n },
  };
  return { zUm: { data: z, rows: n, cols: n }, truth, n, px };
}

/**
 * Synthetic head scan and its ground truth.
 *
 * `rectAperture` is `[widthMm, lengthMm]` of a slot centred on the
 * pin with its long axis at `rectAngleDeg` (image coordinates); the
 * plateau stands `flowbackUm` above the primer face and replaces the
 * round ring.  `shearUm` is the amplitude of the striated band and
 * `roughBandUm` the RMS of isotropic roughness in the same band; the
 * band spans +-50 deg around `shearAngleDeg` (default: opposite the
 * drag, or 180 deg without a drag) between 1.05 x the pin radius and the
 * aperture radius.  The truth holds `rectMask` and `shearMask`
 * (0/1, empty when the feature is off) besides the geometry.
 *
 * Noise differs from Python for the same seed (see the module comment).
 */
export function makeHead(options: HeadOptions = {}): [Surface, HeadTruth] {
  const o = { ...DEFAULTS, ...options };
  const { zUm, truth, n, px } = headGeometry(o);
  const z = zUm.data;
  const N = n * n;
  if (o.roughBandUm) {
    const bump = gaussianFilter(new NormalRng(o.seed + 1).normalGrid(n, n, 0, 1), 1.5);
    let mean = 0;
    for (let i = 0; i < N; i++) mean += bump.data[i]!;
    mean /= N;
    let ss = 0;
    for (let i = 0; i < N; i++) ss += (bump.data[i]! - mean) ** 2;
    const std = Math.sqrt(ss / N); // numpy std: population (ddof=0)
    const m = truth.shearMask.data;
    for (let i = 0; i < N; i++) if (m[i]) z[i] += (o.roughBandUm * bump.data[i]!) / std;
  }
  const noise = gaussianFilter(new NormalRng(o.seed).normalGrid(n, n, 0, 3.0), 1.0); // fine roughness
  const zm = new Float64Array(N);
  for (let i = 0; i < n; i++) {
    const yy = (i - n / 2) * px;
    for (let j = 0; j < n; j++) {
      const xx = (j - n / 2) * px;
      const k = i * n + j;
      zm[k] = (z[k]! + noise.data[k]!) * 1e-6;
      if (o.invalidCorners && hypot(xx, yy) > o.sizeMm / 2) zm[k] = NaN;
    }
  }
  const surf: Surface = { z: { data: zm, rows: n, cols: n }, dx: o.umPerPx * 1e-6, dy: o.umPerPx * 1e-6, name: "synthetic" };
  return [surf, truth];
}

/** Python keeps `write_x3p` in this module; re-exported for parity. */
export { writeX3P } from "./x3p.js";
