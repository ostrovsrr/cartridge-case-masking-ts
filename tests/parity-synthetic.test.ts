/**
 * Parity with the Python pipeline on synthetic heads: the X3P inputs and the Python labels come
 * from `reference/export_labels.py` (reference/out/, gitignored); skipped when absent.
 */
import { paritySuite } from "./parity-suite.js";

paritySuite("synthetic");
