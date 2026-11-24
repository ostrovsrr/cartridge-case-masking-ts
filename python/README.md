# python/: research evidence

The product is the TypeScript package in `src/`. This folder holds the few
**research-evidence experiments** that stay in the scientific Python ecosystem,
because they depend on SciPy (FFT-based masked cross-correlation, Gaussian
filters, bootstrap statistics) and OpenCV (SIFT + RANSAC registration, area
resampling, contour drawing). Porting them would mean re-implementing and
re-validating numerics that are not part of the product.

**None of these scripts segments a scan.** They read the masks that the
TypeScript CLI writes, so the evidence describes the TypeScript product's
masks, not those of the original Python `cartmask` package.

```
npx tsx src/cli.ts <scans or folders> -o outputs/ts-<name>   # per scan: labels.png + masks.json
```

## What lives here

| File | What it does | Ported from (`cartmask` repo) |
|---|---|---|
| `compare.py` | Downstream check of the breech-face mask: band-pass, circular-symmetry removal, rotation + translation registration by masked NCC, AUC / EER / d' / ROC, firearm-level cluster bootstrap, `--summarize-only` | `cartmask/compare.py` |
| `replicate_consistency.py` | Repeat firings of one firearm: mask IoU after registration, consensus masks | `scripts/replicate_consistency.py` |
| `image2d.py` | Best-effort 2D mode for shaded images; `--eval-brief` (SIFT + RANSAC onto the brief's annotated figure), `--eval-renders` (2D mode vs the TS 3D masks) | `cartmask/image2d.py` |
| `benchmark_2d.py` | Low-shading fallback vs the shipped 2D detector: brief, synthetic heads, real renders with the TS masks as 3D proxy | `scripts/benchmark_2d.py` |
| `ts_masks.py` | Reads a TS CLI output back into a `Result` and rebuilds the height map and datum (see below) | new |
| `cartmask_shared.py` | Verbatim copies of the `cartmask` helpers these scripts need (label values, geometry, resampling, datum plane fit, hillshade, overlay figures) | `segment.py`, `steps/*.py`, `preprocess.py`, `visualize.py` |
| `x3p_reader.py`, `synthetic.py` | X3P reader and synthetic heads | `cartmask/x3p.py`, `cartmask/synthetic.py` |
| `samples/brief/` | The brief's figures (inputs of `--eval-brief`) and the reference `brief_eval.json` | `samples/brief/` |
| `tests/` | Statistics and registration tests, 2D CLI test, TS-output loader tests | `tests/test_compare.py`, `tests/test_image2d.py` |

## How the height map is rebuilt

The TS CLI writes `labels.png` (uint8 labels on the working grid) and
`masks.json` (scalars rounded to 4 decimals), but not the heights. The
comparison needs the depth below the primer datum, so `ts_masks.load_scan`
rebuilds what `cartmask.segment` held in memory:

1. It re-reads the X3P named by `masks.json["source"]` (or `<name>.x3p` in
   `--x3p-dir`) and area-resamples it to the working grid. The grid must match
   `labels.png` and `masks.json["um_per_px"]`.
2. It negates the heights when `masks.json` reports the polarity flip.
3. It re-fits the datum plane with `robust_plane` on the annulus that
   `fit_datum` uses, with the primer circle from `masks.json`. The plane
   coefficients are not stored, but the same inputs give the same plane.
4. It checks itself. The primer-face roughness recomputed from the new depth
   must equal `masks.json["roughness_um"]` to 4 decimals. The outcome is in
   `Result.datum_check`, and `compare.py` prints it for every scan.

Verification: the rebuilt depth was compared with `cartmask.segment`'s own
depth on the 28 distinct usable scans in `data/matching` and `data/raw`. 27
were bit-identical. On the last one (Fadul 9-2) the largest difference was
0.0008 um, caused by the 4-decimal rounding of the primer circle. All 28
passed the roughness check. The polarity-flip path has no real example and is
covered by `tests/test_ts_masks.py`.

## Agreement with the Python reference evidence

These results use TS CLI masks, compared with the `cartmask` repository's committed samples:

- `compare.py` on all 19 `data/matching` scans (81 pairs x 3 regions):
  breech-face and full-scan scores, angles, shifts and mask IoUs are identical
  to `samples/matching/pairs.csv`. One primer-disc score differs in the 4th
  decimal. `summary.json` is identical.
- `compare.py --summarize-only` on the reference `pairs.csv` reproduces
  `summary.json` byte for byte.
- `replicate_consistency.py` on the matching scans gives the same pairwise
  IoUs for all 8 firearms.
- `benchmark_2d.py` reproduces `samples/validation/2d-benchmark.json`, and
  `image2d.py --eval-brief` reproduces `brief_eval.json`.

## Running

```
pip install -r python/requirements.txt
pytest python/tests -q

npx tsx src/cli.ts <data>/matching -o outputs/ts-matching
python python/compare.py outputs/ts-matching -o outputs/matching           # ~7 min, 81 pairs x 3 regions
python python/compare.py --summarize-only -o outputs/matching              # stats + plots from pairs.csv
python python/compare.py outputs/ts-matching -o outputs/check --pairs-from <reference pairs.csv> --max-pairs 3

npx tsx src/cli.ts <data>/matching <data>/raw <data>/expanded -o outputs/ts-replicates
python python/replicate_consistency.py outputs/ts-replicates -o outputs/replicates

python python/image2d.py --eval-brief python/samples/brief
python python/image2d.py --eval-renders outputs/ts-raw -o outputs/2d
python python/benchmark_2d.py --masks outputs/ts-2d-proxy -o outputs/2d-benchmark   # REAL list in the script
```

Requirements are pinned to the versions that produced the reference evidence
(numpy 1.26.2, scipy 1.11.3, opencv-python 4.11.0.86, matplotlib 3.8.2).
