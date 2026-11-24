"""Compare a low-shading fallback to the shipped 2D detector on fixed fixtures.

    npx tsx src/cli.ts <data>/raw/9mm__E3__G0042_A0005.002_CC01_TF.x3p ... -o outputs/ts-2d-proxy
    python python/benchmark_2d.py --masks outputs/ts-2d-proxy -o outputs/2d-benchmark

Port of ``scripts/benchmark_2d.py``.  Three fixture sets:

* the brief's page-1 figure (``samples/brief``; SIFT + RANSAC registration onto
  the annotated reference, see ``image2d.evaluate_brief``);
* synthetic heads with known geometry under eight lightings (``synthetic.make_head``);
* real scans: the 2D mode runs on the hillshade render of the working-grid
  heights and is scored against the 3D masks.  Those 3D proxy masks are now
  the TypeScript CLI's ``labels.png`` (``ts_masks``), not ``cartmask.segment``.
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import image2d
from cartmask_shared import (BREECH_FACE, FIRING_PIN_IMPRESSION, FIRING_PIN_SHEAR, disk, hillshade,
                             render)
from synthetic import make_head
from ts_masks import load_scan, scan_dirs

HERE = Path(__file__).resolve().parent
DETECT_IMPRESSION = image2d.find_impression_2d
REAL = (
    "raw/9mm__E3__G0042_A0005.002_CC01_TF.x3p",
    "raw/9mm__E3__G0257_A0006.001_CC01_TF.x3p",
    "raw/40SW__Lightstone__Lightstone_SW_6-1.x3p",
    "raw/9mm__Fadul__Fadul_1-1.x3p",
    "glock/9mm__PopStatGlock__AADW896_G17_RP2.x3p",
    "glock/9mm__PopStatGlock__CWS111US_G34_RP1.x3p",
)


def fallback(g, valid, cx, cy, radius, params, warnings):
    initial = DETECT_IMPRESSION(g, valid, cx, cy, radius, params, warnings)
    if initial[1] is not None:
        return initial
    candidate = DETECT_IMPRESSION(g, valid, cx, cy, radius, replace(params, fpi_min_score=1.5), [])
    mask, core, _, _ = candidate
    if core is None:
        return initial
    x, y, r = core
    if not (0.15 * radius <= r <= 0.50 * radius and np.hypot(x - cx, y - cy) <= 0.35 * radius):
        return initial
    if (image2d.iou(mask, disk(mask.shape, x, y, r)) or 0) < 0.60:
        return initial
    return candidate


def scores(image, face, impression):
    original = image2d.segment_image(image).labels_at(image.shape[:2])
    with patch.object(image2d, "find_impression_2d", fallback):
        candidate = image2d.segment_image(image).labels_at(image.shape[:2])

    def measure(labels):
        return dict(breech_face=image2d.iou(labels == BREECH_FACE, face),
                    firing_pin=image2d.iou(np.isin(labels, (FIRING_PIN_IMPRESSION, FIRING_PIN_SHEAR)), impression))

    return dict(baseline=measure(original), candidate=measure(candidate))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--masks", action="append", required=True,
                        help="TypeScript CLI output folder(s) holding the REAL scans' masks (repeatable)")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="X3P root (REAL paths are relative to it); only needed when "
                             "masks.json['source'] does not resolve")
    parser.add_argument("--brief", type=Path, default=HERE / "samples/brief")
    parser.add_argument("-o", "--out", type=Path, default=Path("outputs/2d-benchmark"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    synthetic = []
    for index, (azimuth, altitude, depth) in enumerate([
        (315, 35, 400), (90, 35, 400), (0, 70, 230), (180, 15, 200),
        (270, 35, 100), (45, 65, 90), (135, 35, 40), (225, 60, 180),
    ]):
        surface, truth = make_head(um_per_px=12.0, seed=index + 31, crater_depth_um=depth,
                                   drag_angle_deg=None if index % 2 else 47)
        shape = surface.z.shape
        primer = disk(shape, *truth["primer_center_px"], 0.95 * truth["primer_radius_px"])
        aperture = disk(shape, *truth["pin_center_px"], truth["aperture_radius_px"])
        impression = disk(shape, *truth["pin_center_px"], truth["pin_radius_px"]) | truth["tail_mask"]
        image = hillshade(surface.z, surface.dx, azimuth_deg=azimuth, altitude_deg=altitude)
        synthetic.append(dict(seed=index + 31, azimuth=azimuth, altitude=altitude, depth_um=depth,
                              **scores(image, primer & ~aperture & ~impression, impression)))
    dirs = {d.name: d for d in scan_dirs(args.masks)}
    x3p_dirs = sorted({str(args.data_root / Path(r).parent) for r in REAL}) if args.data_root else []
    proxy = []
    for relative in REAL:
        name = Path(relative).stem
        if name not in dirs:
            raise SystemExit(f"{name}: no TypeScript CLI output under {args.masks}; run src/cli.ts on it first")
        result = load_scan(dirs[name], x3p_dirs)
        proxy.append(dict(file=relative, **scores(render(result), result.labels == BREECH_FACE,
                          np.isin(result.labels, (FIRING_PIN_IMPRESSION, FIRING_PIN_SHEAR)))))
    brief = args.out / "brief"
    brief.mkdir(exist_ok=True)
    for stem in ("brief_p1_before", "brief_p1_after_reference", "brief_p2_reference"):
        shutil.copyfile(args.brief / f"{stem}.png", brief / f"{stem}.png")
    baseline = image2d.evaluate_brief(brief)["page1_evofinder"]["iou"]
    with patch.object(image2d, "find_impression_2d", fallback):
        development = image2d.evaluate_brief(brief)["page1_evofinder"]["iou"]
    payload = dict(brief_development=dict(baseline=baseline, candidate=development),
                   synthetic_known_geometry=synthetic, real_render_3d_proxy=proxy)
    (args.out / "benchmark.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for name, rows in (("Synthetic", synthetic), ("Real proxy", proxy)):
        for metric in ("breech_face", "firing_pin"):
            paired = [(r["baseline"][metric], r["candidate"][metric]) for r in rows]
            pairs = np.asarray([(x, y) for x, y in paired if x is not None and y is not None])
            print(f"{name} {metric}: n={len(pairs)}, baseline {pairs[:, 0].mean():.3f}, "
                  f"candidate {pairs[:, 1].mean():.3f}, worst change {(pairs[:, 1]-pairs[:, 0]).min():+.3f}")
    print("Brief development case:", baseline, development)


if __name__ == "__main__":
    main()
