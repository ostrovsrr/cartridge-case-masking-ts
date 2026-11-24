# Changelog

## Unreleased

TypeScript port of the Python reference implementation. The masks are unchanged: label grids are identical to the reference on all 152 real scan files and 36 synthetic heads. For the history of the method itself, see the reference implementation's changelog.

- Ported X3P reading and writing, the 3D pipeline and its stage modules, synthetic heads, the batch CLI, exports, overlays and the offline demo to TypeScript (Node 20+, strict ESM).
- Ported the NBTRD fetcher with the same pinned manifests, request pattern and offline verification.
- Ported the mask audit (`scripts/audit-masks.ts`). It writes the reference's audit format byte for byte and reproduces the committed `samples/validation/` baseline; `tests/audit.test.ts` checks this when `CARTMASK_DATA` is set.
- Ported the revision comparison (`scripts/compare-revisions.ts`) and failure gallery (`scripts/failure-gallery.ts`). Runs are stored as `.json` + `.npz` + `.png`, the same layout as the reference, so runs from either implementation can be compared.
- Ported the interactive report (`scripts/build-report.ts`, same template and data payload), the offline annotator and annotation scoring.
- Added `scripts/rebuild-evidence.ts`, which runs the TypeScript steps and the Python evidence experiments, skipping the latter when Python is absent.
- Kept the research experiments (matching, 2D image mode and benchmark, replicate consistency) in Python under `python/`; they now read the TypeScript masks.
- Added CI on Node 20 and 22 (Ubuntu and Windows) and a Python 3.12 job for `python/tests`.
