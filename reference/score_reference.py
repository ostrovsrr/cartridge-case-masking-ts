"""Score an annotation export with the Python ``cartmask.evaluate`` (parity reference).

Used by ``tests/evaluate.test.ts``: the TypeScript side segments the scans, writes the
predictions to ``--pred`` and scores the export itself; this script runs the Python
``cartmask.evaluate.run`` on the same X3P folder, annotation folder and (optional) private map,
with ``segment`` replaced by the TypeScript predictions, so any difference is in the scoring.

    python reference/score_reference.py --pred-from RAW --truth ANN --pred PRED -o OUT [--blind-map MAP]

``--pred`` holds ``<x3p stem>__<n>.json`` (Result fields: name, um_per_px, work_um_per_px, status,
qc_score, primer_center_px, primer_radius_px, fp_core_center_px) and ``<x3p stem>__<n>.png``
(the label grid). The scored rows, summary, geometry, inter-annotator rows, skip reasons and
Markdown are written to ``OUT/python.json``; the CSV / Markdown outputs land in ``OUT`` as usual.

Environment: ``CARTMASK_PY`` = path of the Python repo (default ``../cartridge-case``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY_ROOT = Path(os.environ.get("CARTMASK_PY", HERE.parent.parent / "cartridge-case")).resolve()
sys.path.insert(0, str(PY_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import cartmask.evaluate as ev  # noqa: E402
import cartmask.visualize as vis  # noqa: E402
import cartmask.x3p as x3p  # noqa: E402
from cartmask.segment import Result  # noqa: E402


def load_predictions(folder: Path) -> dict:
    preds = {}
    for js in sorted(folder.glob("*.json")):
        d = json.loads(js.read_text(encoding="utf-8"))
        labels = cv2.imdecode(np.fromfile(str(js.with_suffix(".png")), np.uint8), cv2.IMREAD_UNCHANGED)
        tup = lambda v: None if v is None else tuple(float(x) for x in v)  # noqa: E731
        res = Result(name=d["name"], um_per_px=float(d["um_per_px"]), status=d["status"],
                     qc_score=float(d["qc_score"]), primer_center_px=tup(d["primer_center_px"]),
                     primer_radius_px=d["primer_radius_px"], fp_core_center_px=tup(d["fp_core_center_px"]),
                     labels=labels.astype(np.uint8))
        preds[(d["name"], float(d["work_um_per_px"]))] = res
    return preds


def plain(o):
    if isinstance(o, dict):
        return {str(k): plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-from", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--tol-um", type=float, default=ev.DEFAULT_TOL_UM)
    ap.add_argument("--blind-map", type=Path)
    a = ap.parse_args()
    preds = load_predictions(Path(a.pred))

    def fake_segment(surface, params):
        return preds[(surface.name, float(params.work_um_per_px))]

    ev.segment = fake_segment
    x3p.read_x3p = lambda path, *args, **kw: types.SimpleNamespace(name=Path(path).stem)
    vis.render = lambda res: np.zeros((*res.labels.shape, 3), np.uint8)
    blind = json.loads(a.blind_map.read_text(encoding="utf-8")) if a.blind_map else None
    out = ev.run(Path(a.pred_from), Path(a.truth), Path(a.out), a.tol_um, blind_map=blind)
    rows = [dict(scan=r["scan"], annotator=r["annotator"], prefilled=r["prefilled"], status=r["status"],
                 qc_score=r["qc_score"], um_per_px=r["um_per_px"], metrics=r["metrics"], crop=list(r["crop"]))
            for r in out["rows"]]
    doc = dict(rows=rows, summary=out["summary"], geometry=out["geometry"], inter=out["inter"],
               skipped=out["skipped"], markdown=out["markdown"])
    (Path(a.out) / "python.json").write_text(json.dumps(plain(doc), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
