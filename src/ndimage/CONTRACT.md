# ndimage contract (TypeScript replacements for the SciPy/OpenCV calls cartmask uses)

All 2-D arrays are row-major `Grid`s. Semantics follow `scipy.ndimage` defaults unless stated, so
the TypeScript pipeline can be checked against the Python reference (`reference/make_fixtures.py`).

```ts
export type Num = Float64Array | Float32Array;
export interface Grid<T extends ArrayLike<number> = Float64Array> { data: T; rows: number; cols: number }
export type Mask = Grid<Uint8Array>;            // 0/1
export type Labels = Grid<Int32Array>;          // 0 = background, 1..n
```

| Module | Function | Python equivalent / notes |
|---|---|---|
| grid.ts | `grid`, `zeros`, `full`, `clone`, `map`, `at`, `isFinite mask`, `where`, `percentile(values, q)` (linear, numpy default), `median`, `mad` | numpy basics |
| filters.ts | `gaussianFilter(g, sigma, {order?: [dy,dx], truncate=4})` | `ndi.gaussian_filter`, mode `reflect` (half-sample symmetric), separable; `order` 0/1 per axis for structure tensors |
| filters.ts | `uniformFilter(g, size)`, `medianFilter1d(arr, size)`, `medianFilter(g, size)`, `maximumFilter(g, size)` | `ndi.uniform_filter`, `ndi.median_filter`, `ndi.maximum_filter` (reflect) |
| morphology.ts | `binaryDilation(m, iterations=1, structure=cross3)`, `binaryErosion`, `binaryOpening`, `binaryClosing`, `binaryFillHoles(m)`, `binaryPropagation(seed, mask)`, `disk(rows, cols, cx, cy, r)`, `ellipseKernel(k)` | `ndi.binary_*` (default structure = 4-connected cross, border = 0); fill holes = background not 4-connected to border |
| measure.ts | `label(m, connectivity=4)` → `{labels, n}`; `labelSums(values, labels, n)`; `labelSizes` | `ndi.label` (default 4-connectivity), `ndi.sum` |
| distance.ts | `edt(m)` → distances to nearest zero; `edtIndices(invalid)` → nearest valid index per pixel | `ndi.distance_transform_edt(…, return_indices=True)` (Felzenszwalb–Huttenlocher exact EDT) |
| interp.ts | `mapCoordinates(g, ys, xs)` (bilinear, mode nearest) | `ndi.map_coordinates(order=1, mode="nearest")` |
| interp.ts | `resizeArea`, `resizeNearest`, `resizeLinear(g, rows, cols)` | `cv2.resize` INTER_AREA / NEAREST / LINEAR (OpenCV pixel-centre conventions) |
| interp.ts | `rotationMatrix(cx, cy, angleDeg)`, `warpAffine(g, M, rows, cols, {nearest?, border=0})` | `cv2.getRotationMatrix2D`, `cv2.warpAffine` |
| contours.ts | `externalContour(m)` → points of the outer boundary of the largest component, in order | `cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_NONE)` largest contour |
| fft.ts | `nextFastLen(n)`, `rfft2/irfft2` or complex `fft2/ifft2` on padded grids | `scipy.fft` (any exact algorithm; Bluestein or mixed radix) |
| linalg.ts | `lstsq(A rows×k, b)` small dense least squares (QR or normal equations with care) | `np.linalg.lstsq` for plane/circle fits |

## Implementation notes (phase 1A)

Concrete choices where the table above left room (all names above are exported unchanged from
`src/ndimage/index.ts`):

- `isFinite mask` is `isFiniteMask(g)`. Extra helpers: `select(g, mask)`, `nanmedian`, `pairwiseSum`
  (NumPy's summation), `gradient(g, spacing)` (= `np.gradient`, returns `{gy, gx}`),
  `gaussianFilter1d`, `correlate1d`, `CROSS3`, `SQUARE3`, `ones(r, c)`, `maskAnd/Or/Not/Count`,
  `labelMask`, `featureTransform` (`{iy, ix}`), `findExternalContours`, `invertAffine`,
  `fft/ifft` (1-D, in place), `fft2/ifft2`, `cmul`, `lstsqFull` (`{x, rank, residual}`).
- `gaussianFilter(g, sigma | [sy, sx], {order, truncate})` returns Float64; for a Float32Array input
  every axis pass is rounded to float32 (SciPy keeps float32 dtype). Same for `uniformFilter`.
- `mad(values)` is the raw `median(|v - median(v)|)` (multiply by 1.4826 yourself).
- `percentile(values, q)` takes a number or an array of q (returns number / number[]).
- Morphology signatures are positional: `binaryErosion(m, iterations = 1, structure = CROSS3)`;
  `binaryFillHoles(m, structure?)`, `binaryPropagation(seed, mask?, structure?)`.
  `ellipseKernel(k)` = `disk(2k+1, 2k+1, k, k, k)`. A 1-D signal is a `1 x n` grid (use
  `ones(1, 2*gap+1)` for the aperture plateau closing).
- `label(m, 4 | 8)`; `labelSums(values, labels, n)` and `labelSizes(labels, n)` return arrays of
  length `n + 1` indexed by label (index 0 = background).
- `edt(m)` → Float64 grid; `edtIndices(invalid)` → `Int32Array` of flat indices `row*cols + col`
  (-1 if there is no valid pixel). Same algorithm as SciPy, so ties resolve identically.
- `resizeArea` / `resizeLinear(g, rows, cols)` return Float64 grids (OpenCV double path; float32
  inputs are rounded to float32); `resizeNearest` keeps the array type.
- `rotationMatrix(cx, cy, angleDeg, scale = 1)` → `Float64Array(6)` (row-major 2x3).
  `warpAffine(g, M, rows, cols, {nearest?, border = 0, inverse?})`: nearest keeps the array type,
  bilinear returns Float64.
- `externalContour(m)` → `[x, y][]` of the longest external contour
  (`max(contours, key=len)`), `[]` for an empty mask.
- `rfft2(g, shape?)` → `{re, im, rows, cols, shape}` half spectrum; `irfft2(spec, shape)` → Float64
  grid; `nextFastLen(n, real = true)`.
- `lstsq(A, b)` takes a row-major `Grid` (rows x k) or an array of rows; returns `Float64Array`.

Parity fixtures: `reference/ndimage_fixtures.py` → `tests/fixtures/ndimage/*.json`, checked by
`tests/ndimage.test.ts`.

Images for display/encoding use `@napi-rs/canvas` (PNG/JPEG encode/decode, text, lines, circles).
Real NIST scans are read from `process.env.CARTMASK_DATA` (e.g. `D:\rodio\repos\cartridge-case\data`);
tests that need them are skipped when it is unset.
