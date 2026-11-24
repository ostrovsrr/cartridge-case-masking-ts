/**
 * Parity on docs/expanded-scan-manifest.csv (118 more scans, several minutes): runs only with
 * CARTMASK_PARITY_EXPANDED=1, CARTMASK_DATA and a reference exported with `--expanded`.
 */
import { paritySuite } from "./parity-suite.js";

paritySuite("expanded");
