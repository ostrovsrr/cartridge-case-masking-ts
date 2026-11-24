/**
 * Small dense least squares (`np.linalg.lstsq(A, b, rcond=None)`) by Householder QR.
 *
 * Full-rank problems are solved by back substitution on R. If R is (numerically) rank deficient
 * the minimum-norm solution is taken from a Jacobi SVD of R with NumPy's default cutoff
 * (singular values below eps * max(m, n) * s_max are dropped), which is what NumPy's SVD-based
 * solver returns.
 */
import type { Grid } from "./grid.js";

export type MatrixLike = Grid<ArrayLike<number>> | ArrayLike<ArrayLike<number>>;

function toColumnMajor(A: MatrixLike): { a: Float64Array; m: number; n: number } {
  if ("rows" in A && "cols" in A && "data" in A) {
    const g = A as Grid<ArrayLike<number>>;
    const a = new Float64Array(g.rows * g.cols);
    for (let i = 0; i < g.rows; i++) for (let j = 0; j < g.cols; j++) a[j * g.rows + i] = g.data[i * g.cols + j];
    return { a, m: g.rows, n: g.cols };
  }
  const rowsArr = A as ArrayLike<ArrayLike<number>>;
  const m = rowsArr.length;
  const n = m ? rowsArr[0].length : 0;
  const a = new Float64Array(m * n);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) a[j * m + i] = rowsArr[i][j];
  return { a, m, n };
}

export interface LstsqResult {
  x: Float64Array;
  rank: number;
  /** Sum of squared residuals ||A x - b||^2. */
  residual: number;
}

/** Least-squares solution with rank and residual. */
export function lstsqFull(A: MatrixLike, b: ArrayLike<number>): LstsqResult {
  const { a: a0, m: m0, n } = toColumnMajor(A);
  if (b.length !== m0) throw new Error(`lstsq: b has ${b.length} entries, A has ${m0} rows`);
  // pad with zero rows if underdetermined (does not change the solution set)
  const m = Math.max(m0, n);
  const a = new Float64Array(m * n);
  for (let j = 0; j < n; j++) a.set(a0.subarray(j * m0, (j + 1) * m0), j * m);
  const c = new Float64Array(m);
  for (let i = 0; i < m0; i++) c[i] = b[i];

  // Householder QR: a := R (upper part), c := Q^T b
  for (let j = 0; j < n; j++) {
    const col = j * m;
    let scale = 0;
    for (let i = j; i < m; i++) scale = Math.max(scale, Math.abs(a[col + i]));
    if (scale === 0) continue;
    let norm2 = 0;
    for (let i = j; i < m; i++) {
      const v = a[col + i] / scale;
      norm2 += v * v;
    }
    let alpha = scale * Math.sqrt(norm2);
    if (a[col + j] > 0) alpha = -alpha;
    // v = x - alpha e1 stored in place; beta = 2 / (v^T v)
    a[col + j] -= alpha;
    let vtv = 0;
    for (let i = j; i < m; i++) vtv += a[col + i] * a[col + i];
    if (vtv === 0) { a[col + j] = alpha; continue; }
    const beta = 2 / vtv;
    for (let k = j + 1; k < n; k++) {
      const ck = k * m;
      let s = 0;
      for (let i = j; i < m; i++) s += a[col + i] * a[ck + i];
      s *= beta;
      for (let i = j; i < m; i++) a[ck + i] -= s * a[col + i];
    }
    let s = 0;
    for (let i = j; i < m; i++) s += a[col + i] * c[i];
    s *= beta;
    for (let i = j; i < m; i++) c[i] -= s * a[col + i];
    a[col + j] = alpha;
    for (let i = j + 1; i < m; i++) a[col + i] = 0;
  }
  let residual = 0;
  for (let i = n; i < m; i++) residual += c[i] * c[i];

  const R = new Float64Array(n * n); // row-major n x n
  for (let i = 0; i < n; i++) for (let j = i; j < n; j++) R[i * n + j] = a[j * m + i];
  let rmax = 0;
  for (let i = 0; i < n; i++) rmax = Math.max(rmax, Math.abs(R[i * n + i]));
  const tol = Number.EPSILON * Math.max(m0, n) * rmax;
  let fullRank = rmax > 0;
  for (let i = 0; i < n && fullRank; i++) if (!(Math.abs(R[i * n + i]) > tol * 10)) fullRank = false;

  const x = new Float64Array(n);
  if (fullRank) {
    for (let i = n - 1; i >= 0; i--) {
      let s = c[i];
      for (let j = i + 1; j < n; j++) s -= R[i * n + j] * x[j];
      x[i] = s / R[i * n + i];
    }
    return { x, rank: n, residual };
  }
  const { rank } = minNormSolve(R, n, c, Math.max(m0, n), x);
  return { x, rank, residual };
}

/** `np.linalg.lstsq(A, b, rcond=None)[0]` for a small dense A (rows x k, row-major grid or rows). */
export function lstsq(A: MatrixLike, b: ArrayLike<number>): Float64Array {
  return lstsqFull(A, b).x;
}

/** Minimum-norm solution of R x = c (R n x n row-major) by one-sided Jacobi SVD. */
function minNormSolve(R: Float64Array, n: number, c: Float64Array, mn: number, x: Float64Array): { rank: number } {
  // columns of U (column-major copies of R's columns), V = I
  const U = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) U[j * n + i] = R[i * n + j];
  const V = new Float64Array(n * n);
  for (let j = 0; j < n; j++) V[j * n + j] = 1;
  for (let sweep = 0; sweep < 60; sweep++) {
    let rotated = false;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        let al = 0, be = 0, ga = 0;
        for (let i = 0; i < n; i++) {
          const up = U[p * n + i], uq = U[q * n + i];
          al += up * up; be += uq * uq; ga += up * uq;
        }
        if (ga === 0 || Math.abs(ga) <= 1e-15 * Math.sqrt(al * be)) continue;
        rotated = true;
        const zeta = (be - al) / (2 * ga);
        const t = Math.sign(zeta || 1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        const cs = 1 / Math.sqrt(1 + t * t), sn = cs * t;
        for (let i = 0; i < n; i++) {
          const up = U[p * n + i], uq = U[q * n + i];
          U[p * n + i] = cs * up - sn * uq;
          U[q * n + i] = sn * up + cs * uq;
          const vp = V[p * n + i], vq = V[q * n + i];
          V[p * n + i] = cs * vp - sn * vq;
          V[q * n + i] = sn * vp + cs * vq;
        }
      }
    }
    if (!rotated) break;
  }
  const sv = new Float64Array(n);
  let smax = 0;
  for (let j = 0; j < n; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += U[j * n + i] * U[j * n + i];
    sv[j] = Math.sqrt(s);
    smax = Math.max(smax, sv[j]);
  }
  const cut = Number.EPSILON * mn * smax;
  let rank = 0;
  x.fill(0);
  for (let j = 0; j < n; j++) {
    if (!(sv[j] > cut)) continue;
    rank++;
    let uc = 0;
    for (let i = 0; i < n; i++) uc += U[j * n + i] * c[i];
    const coef = uc / (sv[j] * sv[j]); // U_j is unnormalised: (u/s)^T c / s
    for (let i = 0; i < n; i++) x[i] += coef * V[j * n + i];
  }
  return { rank };
}
