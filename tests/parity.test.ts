/**
 * Parity with the Python pipeline on the 34 scans of docs/scan-manifest.csv (label grid shape,
 * per-class IoU, geometry, QC and review routing). Needs `reference/out/` from
 * `python reference/export_labels.py` and CARTMASK_DATA; skipped otherwise.
 * Synthetic inputs: parity-synthetic.test.ts; expanded manifest: parity-expanded.test.ts.
 */
import { paritySuite } from "./parity-suite.js";

paritySuite("baseline");
