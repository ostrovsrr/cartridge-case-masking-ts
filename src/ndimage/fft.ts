/**
 * Exact FFTs of arbitrary length (mixed radix 2/3/4/5 + generic small primes, Bluestein for lengths
 * with a prime factor > 13) and the 2-D real transforms `scipy.fft.rfft2` / `irfft2`.
 *
 * Complex data is kept as separate re/im Float64Arrays. Normalisation follows SciPy's default
 * ("backward"): forward unscaled, inverse scaled by 1/n.
 */
import type { Grid } from "./grid.js";

// ----------------------------------------------------------------------------- next_fast_len

function isSmooth(n: number, primes: readonly number[]): boolean {
  for (const p of primes) while (n % p === 0) n /= p;
  return n === 1;
}

/**
 * `scipy.fft.next_fast_len(n, real)`: the smallest 5-smooth (real=true) or 11-smooth
 * (real=false) integer >= n.
 */
export function nextFastLen(n: number, real = true): number {
  if (n <= 6) return Math.max(n, 1);
  const primes = real ? [2, 3, 5] : [2, 3, 5, 7, 11];
  let m = n;
  while (!isSmooth(m, primes)) m++;
  return m;
}

// ----------------------------------------------------------------------------- 1-D complex FFT

const MAX_DIRECT_PRIME = 13;

class Plan {
  readonly n: number;
  readonly factors: number[] = [];
  readonly twr: Float64Array;
  readonly twi: Float64Array;
  readonly blue: { m: number; plan: Plan; wr: Float64Array; wi: Float64Array; Br: Float64Array; Bi: Float64Array } | null = null;
  private scratchR = new Float64Array(0);
  private scratchI = new Float64Array(0);

  constructor(n: number) {
    this.n = n;
    this.twr = new Float64Array(n);
    this.twi = new Float64Array(n);
    for (let k = 0; k < n; k++) {
      const a = (-2 * Math.PI * k) / n;
      this.twr[k] = Math.cos(a);
      this.twi[k] = Math.sin(a);
    }
    // kissfft factorisation: 4s first, then 2, 3, 5, 7, ...
    let p = 4;
    let rem = n;
    const fsqrt = Math.floor(Math.sqrt(n));
    if (n > 1) {
      do {
        while (rem % p) {
          if (p === 4) p = 2;
          else if (p === 2) p = 3;
          else p += 2;
          if (p > fsqrt) p = rem;
        }
        rem /= p;
        this.factors.push(p, rem);
      } while (rem > 1);
    }
    let maxP = 0;
    for (let i = 0; i < this.factors.length; i += 2) maxP = Math.max(maxP, this.factors[i]);
    if (maxP > MAX_DIRECT_PRIME) {
      // Bluestein: X_k = w_k * sum_j (x_j w_j) conj(w_{k-j}), w_k = exp(-i pi k^2 / n)
      let m = 1;
      while (m < 2 * n - 1) m <<= 1;
      const wr = new Float64Array(n), wi = new Float64Array(n);
      for (let k = 0; k < n; k++) {
        const k2 = (k * k) % (2 * n); // exact for n < 2^26
        const a = (-Math.PI * k2) / n;
        wr[k] = Math.cos(a);
        wi[k] = Math.sin(a);
      }
      const plan = getPlan(m);
      const Br = new Float64Array(m), Bi = new Float64Array(m);
      Br[0] = wr[0]; Bi[0] = -wi[0];
      for (let k = 1; k < n; k++) {
        Br[k] = Br[m - k] = wr[k];
        Bi[k] = Bi[m - k] = -wi[k];
      }
      plan.forward(Br, Bi);
      this.blue = { m, plan, wr, wi, Br, Bi };
    }
  }

  /** In-place forward transform of re/im (length n). */
  forward(re: Float64Array, im: Float64Array): void {
    const n = this.n;
    if (n <= 1) return;
    if (this.blue) return this.bluestein(re, im);
    if (this.scratchR.length < n) {
      this.scratchR = new Float64Array(n);
      this.scratchI = new Float64Array(n);
    }
    const inR = this.scratchR, inI = this.scratchI;
    inR.set(re.subarray(0, n));
    inI.set(im.subarray(0, n));
    this.work(re, im, 0, inR, inI, 0, 1, 0);
  }

  private bluestein(re: Float64Array, im: Float64Array): void {
    const { m, plan, wr, wi, Br, Bi } = this.blue!;
    const n = this.n;
    const ar = new Float64Array(m), ai = new Float64Array(m);
    for (let k = 0; k < n; k++) {
      ar[k] = re[k] * wr[k] - im[k] * wi[k];
      ai[k] = re[k] * wi[k] + im[k] * wr[k];
    }
    plan.forward(ar, ai);
    for (let k = 0; k < m; k++) {
      const r = ar[k] * Br[k] - ai[k] * Bi[k];
      const i = ar[k] * Bi[k] + ai[k] * Br[k];
      ar[k] = r;
      ai[k] = -i; // conj for inverse via forward
    }
    plan.forward(ar, ai);
    const s = 1 / m;
    for (let k = 0; k < n; k++) {
      const cr = ar[k] * s, ci = -ai[k] * s;
      re[k] = cr * wr[k] - ci * wi[k];
      im[k] = cr * wi[k] + ci * wr[k];
    }
  }

  /** kissfft kf_work: out[o..o+p*m) from input in[f + j*fstride]. */
  private work(oR: Float64Array, oI: Float64Array, o: number, iR: Float64Array, iI: Float64Array, f: number, fstride: number, fi: number): void {
    const p = this.factors[fi], m = this.factors[fi + 1];
    if (m === 1) {
      for (let j = 0; j < p; j++) {
        oR[o + j] = iR[f + j * fstride];
        oI[o + j] = iI[f + j * fstride];
      }
    } else {
      for (let j = 0; j < p; j++) this.work(oR, oI, o + j * m, iR, iI, f + j * fstride, fstride * p, fi + 2);
    }
    switch (p) {
      case 2: this.bfly2(oR, oI, o, fstride, m); break;
      case 3: this.bfly3(oR, oI, o, fstride, m); break;
      case 4: this.bfly4(oR, oI, o, fstride, m); break;
      case 5: this.bfly5(oR, oI, o, fstride, m); break;
      default: this.bflyGeneric(oR, oI, o, fstride, m, p); break;
    }
  }

  private bfly2(R: Float64Array, I: Float64Array, o: number, fs: number, m: number): void {
    const twr = this.twr, twi = this.twi;
    for (let k = 0; k < m; k++) {
      const a = o + k, b = a + m;
      const wr = twr[k * fs], wi = twi[k * fs];
      const tr = R[b] * wr - I[b] * wi, ti = R[b] * wi + I[b] * wr;
      R[b] = R[a] - tr; I[b] = I[a] - ti;
      R[a] += tr; I[a] += ti;
    }
  }

  private bfly3(R: Float64Array, I: Float64Array, o: number, fs: number, m: number): void {
    const twr = this.twr, twi = this.twi;
    const epi = twi[fs * m];
    for (let k = 0; k < m; k++) {
      const a = o + k, b = a + m, c = b + m;
      const w1r = twr[k * fs], w1i = twi[k * fs], w2r = twr[2 * k * fs], w2i = twi[2 * k * fs];
      const s1r = R[b] * w1r - I[b] * w1i, s1i = R[b] * w1i + I[b] * w1r;
      const s2r = R[c] * w2r - I[c] * w2i, s2i = R[c] * w2i + I[c] * w2r;
      const s3r = s1r + s2r, s3i = s1i + s2i;
      const s0r = (s1r - s2r) * epi, s0i = (s1i - s2i) * epi;
      const br = R[a] - s3r * 0.5, bi = I[a] - s3i * 0.5;
      R[a] += s3r; I[a] += s3i;
      R[c] = br + s0i; I[c] = bi - s0r;
      R[b] = br - s0i; I[b] = bi + s0r;
    }
  }

  private bfly4(R: Float64Array, I: Float64Array, o: number, fs: number, m: number): void {
    const twr = this.twr, twi = this.twi;
    for (let k = 0; k < m; k++) {
      const a = o + k, b = a + m, c = b + m, d = c + m;
      const w1r = twr[k * fs], w1i = twi[k * fs];
      const w2r = twr[2 * k * fs], w2i = twi[2 * k * fs];
      const w3r = twr[3 * k * fs], w3i = twi[3 * k * fs];
      const s0r = R[b] * w1r - I[b] * w1i, s0i = R[b] * w1i + I[b] * w1r;
      const s1r = R[c] * w2r - I[c] * w2i, s1i = R[c] * w2i + I[c] * w2r;
      const s2r = R[d] * w3r - I[d] * w3i, s2i = R[d] * w3i + I[d] * w3r;
      const s5r = R[a] - s1r, s5i = I[a] - s1i;
      let ar = R[a] + s1r, ai = I[a] + s1i;
      const s3r = s0r + s2r, s3i = s0i + s2i;
      const s4r = s0r - s2r, s4i = s0i - s2i;
      R[c] = ar - s3r; I[c] = ai - s3i;
      ar += s3r; ai += s3i;
      R[a] = ar; I[a] = ai;
      R[b] = s5r + s4i; I[b] = s5i - s4r;
      R[d] = s5r - s4i; I[d] = s5i + s4r;
    }
  }

  private bfly5(R: Float64Array, I: Float64Array, o: number, fs: number, m: number): void {
    const twr = this.twr, twi = this.twi;
    const yar = twr[fs * m], yai = twi[fs * m];
    const ybr = twr[2 * fs * m], ybi = twi[2 * fs * m];
    for (let u = 0; u < m; u++) {
      const i0 = o + u, i1 = i0 + m, i2 = i1 + m, i3 = i2 + m, i4 = i3 + m;
      const s0r = R[i0], s0i = I[i0];
      let wr = twr[u * fs], wi = twi[u * fs];
      const s1r = R[i1] * wr - I[i1] * wi, s1i = R[i1] * wi + I[i1] * wr;
      wr = twr[2 * u * fs]; wi = twi[2 * u * fs];
      const s2r = R[i2] * wr - I[i2] * wi, s2i = R[i2] * wi + I[i2] * wr;
      wr = twr[3 * u * fs]; wi = twi[3 * u * fs];
      const s3r = R[i3] * wr - I[i3] * wi, s3i = R[i3] * wi + I[i3] * wr;
      wr = twr[4 * u * fs]; wi = twi[4 * u * fs];
      const s4r = R[i4] * wr - I[i4] * wi, s4i = R[i4] * wi + I[i4] * wr;
      const s7r = s1r + s4r, s7i = s1i + s4i;
      const s10r = s1r - s4r, s10i = s1i - s4i;
      const s8r = s2r + s3r, s8i = s2i + s3i;
      const s9r = s2r - s3r, s9i = s2i - s3i;
      R[i0] = s0r + (s7r + s8r); I[i0] = s0i + (s7i + s8i);
      const s5r = s0r + s7r * yar + s8r * ybr, s5i = s0i + s7i * yar + s8i * ybr;
      const s6r = s10i * yai + s9i * ybi, s6i = -(s10r * yai) - s9r * ybi;
      R[i1] = s5r - s6r; I[i1] = s5i - s6i;
      R[i4] = s5r + s6r; I[i4] = s5i + s6i;
      const s11r = s0r + s7r * ybr + s8r * yar, s11i = s0i + s7i * ybr + s8i * yar;
      const s12r = -(s10i * ybi) + s9i * yai, s12i = s10r * ybi - s9r * yai;
      R[i2] = s11r + s12r; I[i2] = s11i + s12i;
      R[i3] = s11r - s12r; I[i3] = s11i - s12i;
    }
  }

  private bflyGeneric(R: Float64Array, I: Float64Array, o: number, fs: number, m: number, p: number): void {
    const twr = this.twr, twi = this.twi, N = this.n;
    const sr = new Float64Array(p), si = new Float64Array(p);
    for (let u = 0; u < m; u++) {
      let k = u;
      for (let q1 = 0; q1 < p; q1++, k += m) { sr[q1] = R[o + k]; si[q1] = I[o + k]; }
      k = u;
      for (let q1 = 0; q1 < p; q1++, k += m) {
        let tw = 0;
        let accR = sr[0], accI = si[0];
        for (let q = 1; q < p; q++) {
          tw += fs * k;
          if (tw >= N) tw %= N;
          accR += sr[q] * twr[tw] - si[q] * twi[tw];
          accI += sr[q] * twi[tw] + si[q] * twr[tw];
        }
        R[o + k] = accR; I[o + k] = accI;
      }
    }
  }
}

const plans = new Map<number, Plan>();
function getPlan(n: number): Plan {
  let p = plans.get(n);
  if (!p) {
    p = new Plan(n);
    plans.set(n, p);
  }
  return p;
}

/** In-place forward complex FFT (`np.fft.fft`). */
export function fft(re: Float64Array, im: Float64Array): void {
  getPlan(re.length).forward(re, im);
}

/** In-place inverse complex FFT with 1/n scaling (`np.fft.ifft`). */
export function ifft(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  for (let i = 0; i < n; i++) im[i] = -im[i];
  getPlan(n).forward(re, im);
  const s = 1 / n;
  for (let i = 0; i < n; i++) {
    re[i] *= s;
    im[i] = -im[i] * s;
  }
}

// ----------------------------------------------------------------------------- 2-D transforms

/** Complex 2-D spectrum / signal, row-major rows x cols. */
export interface ComplexGrid {
  re: Float64Array;
  im: Float64Array;
  rows: number;
  cols: number;
}

/** Transform every column of a complex grid in place (forward or inverse, unscaled). */
function columns(z: ComplexGrid, ncols: number, inverse: boolean): void {
  const { rows, cols } = z;
  if (rows <= 1) return;
  const plan = getPlan(rows);
  const br = new Float64Array(rows), bi = new Float64Array(rows);
  for (let c = 0; c < ncols; c++) {
    for (let r = 0; r < rows; r++) {
      br[r] = z.re[r * cols + c];
      bi[r] = inverse ? -z.im[r * cols + c] : z.im[r * cols + c];
    }
    plan.forward(br, bi);
    for (let r = 0; r < rows; r++) {
      z.re[r * cols + c] = br[r];
      z.im[r * cols + c] = inverse ? -bi[r] : bi[r];
    }
  }
}

/** Forward 2-D complex FFT (`np.fft.fft2`), returns a new grid. */
export function fft2(z: ComplexGrid): ComplexGrid {
  const { rows, cols } = z;
  const out: ComplexGrid = { re: z.re.slice(), im: z.im.slice(), rows, cols };
  const plan = getPlan(cols);
  for (let r = 0; r < rows; r++) plan.forward(out.re.subarray(r * cols, (r + 1) * cols), out.im.subarray(r * cols, (r + 1) * cols));
  columns(out, cols, false);
  return out;
}

/** Inverse 2-D complex FFT with 1/(rows*cols) scaling (`np.fft.ifft2`). */
export function ifft2(z: ComplexGrid): ComplexGrid {
  const { rows, cols } = z;
  const out: ComplexGrid = { re: z.re.slice(), im: z.im.slice(), rows, cols };
  const plan = getPlan(cols);
  for (let r = 0; r < rows; r++) {
    const re = out.re.subarray(r * cols, (r + 1) * cols), im = out.im.subarray(r * cols, (r + 1) * cols);
    for (let i = 0; i < cols; i++) im[i] = -im[i];
    plan.forward(re, im);
    for (let i = 0; i < cols; i++) im[i] = -im[i];
  }
  columns(out, cols, true);
  const s = 1 / (rows * cols);
  for (let i = 0; i < out.re.length; i++) {
    out.re[i] *= s;
    out.im[i] *= s;
  }
  return out;
}

/** Half spectrum from `rfft2`: rows = shape[0], cols = shape[1]/2 + 1, plus the real shape. */
export interface RealSpectrum extends ComplexGrid {
  shape: [number, number];
}

/**
 * `scipy.fft.rfft2(g, shape)`: real 2-D FFT of `g` zero-padded (or cropped) to `shape`
 * (default g's shape). Two real rows are packed into one complex FFT.
 */
export function rfft2(g: Grid<ArrayLike<number>>, shape?: [number, number]): RealSpectrum {
  const [s0, s1] = shape ?? [g.rows, g.cols];
  const h = (s1 >> 1) + 1;
  const out: RealSpectrum = { re: new Float64Array(s0 * h), im: new Float64Array(s0 * h), rows: s0, cols: h, shape: [s0, s1] };
  const plan = getPlan(s1);
  const zr = new Float64Array(s1), zi = new Float64Array(s1);
  const nr = Math.min(s0, g.rows), nc = Math.min(s1, g.cols);
  for (let r = 0; r < nr; r += 2) {
    zr.fill(0);
    zi.fill(0);
    const two = r + 1 < nr;
    for (let c = 0; c < nc; c++) {
      zr[c] = g.data[r * g.cols + c];
      if (two) zi[c] = g.data[(r + 1) * g.cols + c];
    }
    plan.forward(zr, zi);
    for (let k = 0; k < h; k++) {
      const nk = k === 0 ? 0 : s1 - k;
      // X = (Z[k] + conj(Z[n-k])) / 2 ; Y = (Z[k] - conj(Z[n-k])) / (2i)
      const ar = zr[k], ai = zi[k], br = zr[nk], bi = -zi[nk];
      out.re[r * h + k] = 0.5 * (ar + br);
      out.im[r * h + k] = 0.5 * (ai + bi);
      if (two) {
        out.re[(r + 1) * h + k] = 0.5 * (ai - bi);
        out.im[(r + 1) * h + k] = -0.5 * (ar - br);
      }
    }
  }
  columns(out, h, false);
  return out;
}

/**
 * `scipy.fft.irfft2(spec, shape)`: inverse of `rfft2`. The spectrum is cropped / zero-padded to
 * `shape[0]` rows and `shape[1]/2 + 1` columns; the imaginary parts of the DC and Nyquist bins
 * along the last axis are ignored (as pocketfft's c2r).
 */
export function irfft2(spec: ComplexGrid, shape: [number, number]): Grid<Float64Array> {
  const [s0, s1] = shape;
  const h = (s1 >> 1) + 1;
  const z: ComplexGrid = { re: new Float64Array(s0 * h), im: new Float64Array(s0 * h), rows: s0, cols: h };
  const nr = Math.min(s0, spec.rows), nc = Math.min(h, spec.cols);
  for (let r = 0; r < nr; r++) {
    for (let c = 0; c < nc; c++) {
      z.re[r * h + c] = spec.re[r * spec.cols + c];
      z.im[r * h + c] = spec.im[r * spec.cols + c];
    }
  }
  columns(z, h, true);
  const out = new Float64Array(s0 * s1);
  const plan = getPlan(s1);
  const zr = new Float64Array(s1), zi = new Float64Array(s1);
  const scale = 1 / (s0 * s1);
  const even = (s1 & 1) === 0;
  for (let r = 0; r < s0; r += 2) {
    const two = r + 1 < s0;
    // Z = Xa + i Xb (Hermitian-completed), inverse via conj-forward-conj
    for (let k = 0; k < s1; k++) {
      let ar: number, ai: number, br = 0, bi = 0;
      if (k < h) {
        ar = z.re[r * h + k]; ai = z.im[r * h + k];
        if (two) { br = z.re[(r + 1) * h + k]; bi = z.im[(r + 1) * h + k]; }
        if (k === 0 || (even && k === h - 1)) { ai = 0; bi = 0; }
      } else {
        const m = s1 - k;
        ar = z.re[r * h + m]; ai = -z.im[r * h + m];
        if (two) { br = z.re[(r + 1) * h + m]; bi = -z.im[(r + 1) * h + m]; }
      }
      // Z = A + iB = (ar - bi) + i(ai + br); conj for inverse
      zr[k] = ar - bi;
      zi[k] = -(ai + br);
    }
    plan.forward(zr, zi);
    for (let c = 0; c < s1; c++) {
      out[r * s1 + c] = zr[c] * scale;
      if (two) out[(r + 1) * s1 + c] = -zi[c] * scale;
    }
  }
  return { data: out, rows: s0, cols: s1 };
}

/** Element-wise complex product of two spectra of equal size (new grid). */
export function cmul<T extends ComplexGrid>(a: T, b: ComplexGrid): T {
  const n = a.re.length;
  const re = new Float64Array(n), im = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    re[i] = a.re[i] * b.re[i] - a.im[i] * b.im[i];
    im[i] = a.re[i] * b.im[i] + a.im[i] * b.re[i];
  }
  return { ...a, re, im };
}
