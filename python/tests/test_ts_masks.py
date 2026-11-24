"""Tests for reading TypeScript CLI outputs (labels.png + masks.json) and rebuilding the datum."""
import json

import cv2
import numpy as np
import pytest

import compare
from cartmask_shared import BREECH_FACE, disk
from synthetic import make_head, write_x3p
from ts_masks import FLIP_WARNING, fit_depth, load_scan, read_labels, scan_dirs


def _write_scan(root, name, surf, truth, warnings=(), roughness=None, rgba=False):
    """A scan as the TypeScript CLI writes it, with the primer circle from the synthetic truth."""
    x3p = root / f"{name}.x3p"
    surf.name = name
    write_x3p(surf, x3p)
    d = root / "out" / name
    d.mkdir(parents=True)
    cx, cy = truth["primer_center_px"]
    rp = truth["primer_radius_px"]
    labels = (disk(surf.z.shape, cx, cy, 0.95 * rp) & ~disk(surf.z.shape, *truth["pin_center_px"],
                                                             truth["aperture_radius_px"])).astype(np.uint8)
    labels[~np.isfinite(surf.z)] = 0
    img = cv2.cvtColor(labels, cv2.COLOR_GRAY2BGRA) if rgba else labels
    cv2.imwrite(str(d / "labels.png"), img)
    z = -surf.z if FLIP_WARNING in " ".join(warnings) else surf.z
    if roughness is None:
        _, roughness, _ = fit_depth(z, np.isfinite(z), cx, cy, rp)
    meta = dict(name=name, um_per_px=surf.dx * 1e6, status="ok", warnings=list(warnings),
                primer_center_px=[round(cx, 4), round(cy, 4)], primer_radius_px=round(rp, 4),
                roughness_um=round(roughness, 4), needs_review=False, source=str(x3p))
    (d / "masks.json").write_text(json.dumps(meta))
    return d


def _tilted(seed=0, a=2e-7, b=-1e-7, c=5e-6):
    surf, truth = make_head(um_per_px=12.0, seed=seed, drag_angle_deg=None)
    yy, xx = np.indices(surf.z.shape)
    flat = surf.z.copy()
    surf.z = surf.z + a * xx + b * yy + c
    return surf, truth, flat


def test_datum_recovers_plane_and_depth(tmp_path):
    surf, truth, flat = _tilted()
    res = load_scan(_write_scan(tmp_path, "9mm__Toy__Toy_1-1", surf, truth), strict=True)
    surf.z = flat
    ref = load_scan(_write_scan(tmp_path, "9mm__Toy__Toy_1-2", surf, truth), strict=True)
    assert res.datum_check["ok"] and ref.datum_check["ok"]
    (a, b, c), (a0, b0, c0) = res.datum_check["plane_coef"], ref.datum_check["plane_coef"]
    assert (a - a0, b - b0, c - c0) == pytest.approx((2e-7, -1e-7, 5e-6), abs=1e-12)
    v = res.valid
    assert np.abs(res.depth[v] - ref.depth[v]).max() < 1e-12          # the tilt is removed exactly
    face = disk(v.shape, *truth["primer_center_px"], 0.9 * truth["primer_radius_px"]) & v
    face &= ~disk(v.shape, *truth["pin_center_px"], truth["aperture_radius_px"])
    assert abs(np.median(res.depth[face])) < 1e-6                      # primer face is the datum
    assert res.labels.max() == BREECH_FACE and res.um_per_px == pytest.approx(12.0)
    assert res.primer_center_px == pytest.approx(truth["primer_center_px"], abs=1e-4)


def test_polarity_flip_is_reapplied(tmp_path):
    surf, truth, _ = _tilted(seed=1)
    ref = load_scan(_write_scan(tmp_path, "9mm__Toy__Toy_1-1", surf, truth))
    surf.z = -surf.z                                   # cast / inverted export, flipped by the CLI
    flipped = load_scan(_write_scan(tmp_path, "9mm__Toy__Toy_1-2", surf, truth,
                                    warnings=["height polarity inverted (cast?) - flipped"]), strict=True)
    v = ref.valid
    assert np.array_equal(v, flipped.valid)
    assert np.allclose(ref.depth[v], flipped.depth[v], atol=1e-12)


def test_roughness_mismatch_is_detected(tmp_path):
    surf, truth, _ = _tilted(seed=2)
    d = _write_scan(tmp_path, "9mm__Toy__Toy_2-1", surf, truth, roughness=99.0)
    assert not load_scan(d).datum_check["ok"]
    with pytest.raises(ValueError, match="roughness"):
        load_scan(d, strict=True)


def test_labels_png_formats_and_scan_discovery(tmp_path):
    surf, truth, _ = _tilted(seed=3)
    d = _write_scan(tmp_path, "9mm__Toy__Toy_3-1", surf, truth, rgba=True)
    assert read_labels(d).ndim == 2 and read_labels(d).max() == 1
    assert scan_dirs([tmp_path / "out"]) == [d] == scan_dirs([d])
    bad = np.zeros((4, 4, 3), np.uint8)
    bad[0, 0] = (0, 0, 1)
    cv2.imwrite(str(d / "labels.png"), bad)
    with pytest.raises(ValueError, match="channels differ"):
        read_labels(d)


def test_compare_end_to_end_on_cli_style_outputs(tmp_path):
    for i, (gun, rep) in enumerate([(1, 1), (1, 2), (2, 1)]):
        surf, truth, _ = _tilted(seed=10 + gun)        # same "firearm" = same texture seed
        surf.z = surf.z + np.random.default_rng(i).normal(0, 2e-7, surf.z.shape)
        _write_scan(tmp_path, f"9mm__Toy__Toy_{gun}-{rep}", surf, truth)
    out = tmp_path / "matching"
    compare.main([str(tmp_path / "out"), "-o", str(out), "--kinds", "breech_face",
                  "--bootstrap-reps", "20"])
    rows, kinds = compare.read_pairs(out / "pairs.csv")
    assert kinds == ["breech_face"] and len(rows) == 3
    km = [r["ccf_breech_face"] for r in rows if r["match"]]
    knm = [r["ccf_breech_face"] for r in rows if not r["match"]]
    assert min(km) > max(knm)
    assert json.loads((out / "summary.json").read_text())["Toy"]["breech_face"]["auc"] == 1.0
