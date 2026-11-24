"""Export Python ``cartmask`` segmentations as the parity reference for the TypeScript port.

For (a) a fixed set of synthetic heads and (b) every scan in the Python repo's
``docs/scan-manifest.csv`` (plus ``docs/expanded-scan-manifest.csv`` with
``--expanded``), this writes into ``reference/out/`` (gitignored):

* ``synthetic/<name>.x3p``         the synthetic input, written by the Python X3P writer,
                                   so the TypeScript side reads exactly the same heights;
* ``<set>/<name>.labels.png``      the Python label grid (8-bit grey, values 0..4);
* ``<set>/<name>.json``            ``Result.summary()`` plus ``seconds`` (segment() only)
                                   and the label grid shape;
* ``manifest.json``                what was exported, with the Params overrides per case.

Usage (system Python with numpy / scipy / opencv)::

    python reference/export_labels.py                 # synthetic + baseline manifest
    python reference/export_labels.py --expanded      # also the expanded manifest
    python reference/export_labels.py --only synthetic

Environment:

``CARTMASK_PY``    path of the Python repo (default: ``../cartridge-case`` next to this repo)
``CARTMASK_DATA``  scan directory (default: ``$CARTMASK_PY/data``); manifest paths
                   ``data/<sub>/<file>.x3p`` are resolved under it.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TS_ROOT = HERE.parent
PY_ROOT = Path(os.environ.get("CARTMASK_PY", TS_ROOT.parent / "cartridge-case")).resolve()
sys.path.insert(0, str(PY_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

from cartmask import Params, read_x3p, segment  # noqa: E402
from cartmask.synthetic import make_head, write_x3p  # noqa: E402

OUT = HERE / "out"
DATA = Path(os.environ.get("CARTMASK_DATA", PY_ROOT / "data")).resolve()


# --------------------------------------------------------------------------- synthetic set

def _nan_blobs(shape, frac, seed, sigma_px=5.0):
    rng = np.random.default_rng(seed)
    f = ndi.gaussian_filter(rng.normal(size=shape), sigma_px)
    return f > np.quantile(f, 1.0 - frac)


def _negate(surf):
    surf.z = -surf.z
    return surf


def _tilt(surf, ramp_um, direction_deg):
    n_rows, n_cols = surf.z.shape
    yy, xx = np.indices(surf.z.shape)
    a = np.deg2rad(direction_deg)
    u = (np.cos(a) * (xx - n_cols / 2) + np.sin(a) * (yy - n_rows / 2)) / n_cols
    surf.z = surf.z + ramp_um * 1e-6 * u
    return surf


def _blobs(surf, frac):
    surf.z = np.where(_nan_blobs(surf.z.shape, frac, seed=int(frac * 100)), np.nan, surf.z)
    return surf


def _hole(surf, truth, hole_frac):
    yy, xx = np.indices(surf.z.shape)
    cx, cy = truth["pin_center_px"]
    surf.z = np.where(np.hypot(xx - cx, yy - cy) < hole_frac * truth["pin_radius_px"], np.nan, surf.z)
    return surf


def _noise(surf, noise_um):
    rng = np.random.default_rng(int(noise_um * 10))
    surf.z = surf.z + rng.normal(0.0, noise_um * 1e-6, surf.z.shape)
    return surf


def _crop(surf, w_mm, h_mm, um):
    n = surf.z.shape[0]
    w, h = int(round(w_mm * 1000 / um)), int(round(h_mm * 1000 / um))
    c0, r0 = (n - w) // 2, (n - h) // 2
    surf.z = surf.z[r0:r0 + h, c0:c0 + w].copy()
    return surf


R = dict(um_per_px=10.0, size_mm=6.5)     # test_robustness defaults

#: name -> (make_head kwargs, post-processing(surf, truth) or None, Params overrides)
SYNTHETIC = {
    "default": ({}, None, {}),
    "round_seed1": (dict(drag_angle_deg=None, seed=1), None, {}),
    "drag20": (dict(drag_angle_deg=20.0), None, {}),
    "inverted_seed2": (dict(seed=2), lambda s, t: _negate(s), {}),
    "flowback_plateau": (dict(seed=4, drag_angle_deg=None, crater_depth_um=80.0, flowback_um=350.0,
                              aperture_radius_mm=1.5, pin_radius_mm=0.6), None, {}),
    "shear": (dict(shear_um=3.0), None, {}),
    "shear_weak": (dict(shear_um=1.5, seed=9), None, {}),
    "shear_nodrag": (dict(shear_um=3.0, drag_angle_deg=None, seed=5), None, {}),
    "shear_drag-60": (dict(shear_um=3.0, drag_angle_deg=-60.0, seed=11), None, {}),
    "shear_across_axis": (dict(shear_um=3.0, shear_angle_deg=210.0, shear_period_mm=0.06,
                               drag_angle_deg=120.0, seed=13), None, {}),
    "rough_band": (dict(rough_band_um=3.0), None, {}),
    "rough_band_nodrag": (dict(rough_band_um=3.0, drag_angle_deg=None, seed=6), None, {}),
    "rect_0": (dict(rect_aperture=(1.3, 3.0), drag_angle_deg=0.0), None, {}),
    "rect_0_nodrag": (dict(rect_aperture=(1.3, 3.0), drag_angle_deg=None, seed=3), None, {}),
    "rect_45": (dict(rect_aperture=(1.1, 2.6), rect_angle_deg=45.0, drag_angle_deg=45.0, seed=4), None, {}),
    "rect_100": (dict(rect_aperture=(1.5, 3.2), rect_angle_deg=100.0, drag_angle_deg=None, seed=14), None, {}),
    "rect_clip": (dict(rect_aperture=(1.0, 3.0), pin_radius_mm=0.6, drag_angle_deg=None, seed=17), None, {}),
    "rect_off": (dict(rect_aperture=(1.3, 3.0), drag_angle_deg=0.0), None, dict(rect_detect=False)),
    "small_pin": (dict(pin_radius_mm=0.55, aperture_radius_mm=0.95, seed=15), None, {}),
    "large_primer": (dict(primer_radius_mm=2.6, pin_radius_mm=0.9, aperture_radius_mm=1.4, size_mm=7.6, seed=16),
                     None, {}),
    "edge_step": (dict(flowback_um=0.0, aperture_step_um=6.0, drag_angle_deg=None, seed=7), None,
                  dict(aperture_edge=True)),
    "edge_none": (dict(flowback_um=0.0, drag_angle_deg=None, seed=8), None, dict(aperture_edge=True)),
    "edge_off": (dict(flowback_um=0.0, aperture_step_um=6.0, drag_angle_deg=None, seed=7), None, {}),
    "calibre_large": (dict(R, um_per_px=8.0, seed=11, primer_radius_mm=2.65, size_mm=7.5), None,
                      dict(work_um_per_px=8.0)),
    "drag_sweep_90": (dict(R, drag_angle_deg=90, seed=23), None, dict(work_um_per_px=10.0)),
    "drag_sweep_240": (dict(R, drag_angle_deg=240, seed=28), None, dict(work_um_per_px=10.0)),
    "off_centre": (dict(R, size_mm=7.8, primer_center_mm=(-0.6, -0.6), pin_offset_mm=(-0.5, 0.5), seed=31),
                   None, dict(work_um_per_px=10.0)),
    "tilt_150_120": (dict(R, seed=41), lambda s, t: _tilt(s, 150, 120), dict(work_um_per_px=10.0)),
    "nan_blobs_15": (dict(R, seed=51), lambda s, t: _blobs(s, 0.15), dict(work_um_per_px=10.0)),
    "hole_80_drag": (dict(R, seed=52, drag_angle_deg=30.0), lambda s, t: _hole(s, t, 0.8),
                     dict(work_um_per_px=10.0)),
    "noise_5": (dict(R, seed=61), lambda s, t: _noise(s, 5.0), dict(work_um_per_px=10.0)),
    "res4_work6": (dict(size_mm=6.5, um_per_px=4.0, seed=71), None, dict(work_um_per_px=6.0)),
    "res4_work12": (dict(size_mm=6.5, um_per_px=4.0, seed=71), None, dict(work_um_per_px=12.0)),
    "crop_7x6_left": (dict(size_mm=7.6, um_per_px=10.0, primer_center_mm=(-0.85, 0.0), seed=81,
                           invalid_corners=False), lambda s, t: _crop(s, 7.0, 6.0, 10.0),
                      dict(work_um_per_px=10.0)),
    "inverted_seed4": (dict(R, seed=4, drag_angle_deg=240.0), lambda s, t: _negate(s), dict(work_um_per_px=10.0)),
    "small_fov": (dict(size_mm=1.0), None, {}),
}


# --------------------------------------------------------------------------- output

def _clean(o):
    """NaN/inf -> None so the JSON is standard (TypeScript JSON.parse rejects NaN)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def _write(dst: Path, name: str, res, seconds: float, extra: dict) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    labels = res.labels if res.labels is not None else np.zeros((1, 1), np.uint8)
    if not cv2.imwrite(str(dst / f"{name}.labels.png"), labels.astype(np.uint8)):
        raise RuntimeError(f"could not write {name}.labels.png")
    s = res.summary()
    s.update(extra)
    s["seconds"] = round(seconds, 3)
    s["labels_shape"] = list(labels.shape)
    (dst / f"{name}.json").write_text(json.dumps(_clean(s), indent=1))


def export_synthetic(names: list[str] | None) -> list[dict]:
    dst = OUT / "synthetic"
    dst.mkdir(parents=True, exist_ok=True)
    out = []
    for name, (kw, post, pover) in SYNTHETIC.items():
        if names and name not in names:
            continue
        surf, truth = make_head(**kw)
        if post is not None:
            surf = post(surf, truth)
        x3p = dst / f"{name}.x3p"
        write_x3p(surf, x3p)
        s = read_x3p(x3p)                     # segment exactly what TypeScript will read
        p = Params(**pover)
        t0 = time.perf_counter()
        res = segment(s, p)
        dt = time.perf_counter() - t0
        _write(dst, name, res, dt, {"params": pover, "kwargs": _clean(dict(kw))})
        print(f"synthetic {name:22s} {res.status:8s} {dt:6.2f}s review={res.needs_review}")
        out.append({"name": name, "x3p": f"synthetic/{name}.x3p", "params": pover})
    return out


def export_real(manifests: list[tuple[str, Path]], limit: int | None) -> list[dict]:
    out = []
    seen = set()
    for set_name, csv_path in manifests:
        with open(csv_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        for row in rows[:limit] if limit else rows:
            rel = row["file"].replace("\\", "/")
            rel = rel[len("data/"):] if rel.startswith("data/") else rel
            if rel in seen:
                continue
            seen.add(rel)
            path = DATA / rel
            name = Path(rel).stem
            study = name.split("__")[1] if name.count("__") >= 2 else "unknown"
            if not path.exists():
                print(f"missing {path} (skipped)")
                continue
            surf = read_x3p(path)
            t0 = time.perf_counter()
            res = segment(surf)
            dt = time.perf_counter() - t0
            sub = rel.split("/")[0]
            _write(OUT / "real" / sub, name, res, dt, {"source": rel, "study": study, "set": set_name})
            print(f"{set_name:8s} {rel:60s} {res.status:8s} {dt:6.2f}s review={res.needs_review}")
            out.append({"name": name, "file": rel, "study": study, "set": set_name,
                        "labels": f"real/{sub}/{name}.labels.png", "json": f"real/{sub}/{name}.json"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--expanded", action="store_true", help="also export docs/expanded-scan-manifest.csv")
    ap.add_argument("--only", choices=["synthetic", "real"], help="export only one part")
    ap.add_argument("--names", nargs="*", help="synthetic case names to export (default: all)")
    ap.add_argument("--limit", type=int, help="first N rows of each manifest (debugging)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    man_path = OUT / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    manifest.update(python_repo=str(PY_ROOT), data=str(DATA), numpy=np.__version__)
    if a.only != "real":
        manifest["synthetic"] = export_synthetic(a.names)
    if a.only != "synthetic":
        mans = [("baseline", PY_ROOT / "docs" / "scan-manifest.csv")]
        if a.expanded:
            mans.append(("expanded", PY_ROOT / "docs" / "expanded-scan-manifest.csv"))
        manifest["real"] = export_real(mans, a.limit)
    man_path.write_text(json.dumps(manifest, indent=1))
    print(f"wrote {man_path}")


if __name__ == "__main__":
    main()
