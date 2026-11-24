"""Run the Python ``cartmask`` CLI and X3P mask exporter as the parity reference for
``src/cli.ts`` and ``src/export.ts``.

Inputs are the Python-written synthetic heads from ``reference/export_labels.py``
(``reference/out/synthetic/<name>.x3p``), so both sides read identical heights.  Writes:

* ``reference/out/cli/``            output of ``python -m cartmask <x3ps> -o ... --x3p-masks``
                                    (per-scan ``masks.json``, ``<name>__masked.x3p``, ``summary.csv`` ...);
* ``reference/out/cli/<name>/mask_rgb.png``  the decoded ``bindata/mask.png`` re-encoded as plain RGB.

Usage (system Python with numpy / scipy / opencv; run ``export_labels.py`` first)::

    python reference/export_masks.py

Environment: ``CARTMASK_PY`` (Python repo, default ``../cartridge-case``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TS_ROOT = HERE.parent
PY_ROOT = Path(os.environ.get("CARTMASK_PY", TS_ROOT.parent / "cartridge-case")).resolve()
sys.path.insert(0, str(PY_ROOT))

import cv2  # noqa: E402

from cartmask import cli  # noqa: E402
from cartmask.export import read_mask  # noqa: E402

SYN = HERE / "out" / "synthetic"
OUT = HERE / "out" / "cli"
CASES = ["default", "round_seed1", "drag20", "inverted_seed2"]


def main() -> None:
    inputs = [SYN / f"{n}.x3p" for n in CASES]
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        sys.exit(f"missing synthetic inputs (run reference/export_labels.py first): {missing}")
    cli.main([*map(str, inputs), "-o", str(OUT), "--x3p-masks", "-q"])
    for n in CASES:
        img, _ = read_mask(OUT / n / f"{n}__masked.x3p")
        cv2.imwrite(str(OUT / n / "mask_rgb.png"), img)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
