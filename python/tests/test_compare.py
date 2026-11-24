"""Tests for the breech-face comparison used to validate the masks downstream."""
import csv
import itertools
import json

import cv2
import numpy as np
import pytest
from scipy import ndimage as ndi

from compare import (Prepared, auc, cluster_bootstrap, eer, identity, main, masked_ncc,
                     register, resample_pairs, roc_curve, separation, summarize)


def _texture(n=200, seed=0):
    rng = np.random.default_rng(seed)
    t = ndi.gaussian_filter(rng.normal(size=(n, n)), 2.0).astype(np.float32)
    yy, xx = np.indices((n, n))
    r = np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2)
    m = ((r < 0.45 * n) & (r > 0.15 * n)).astype(np.float32)   # annulus like a breech face
    return t * m, m


def test_masked_ncc_recovers_shift():
    f, m = _texture()
    g, mg = np.roll(f, (4, -6), axis=(0, 1)), np.roll(m, (4, -6), axis=(0, 1))
    c, dy, dx = masked_ncc(f, m, g, mg, max_shift=10)
    assert c == pytest.approx(1.0, abs=1e-3)
    assert (dy, dx) == (-4, 6)


def test_register_recovers_rotation():
    f, m = _texture(seed=1)
    h, w = f.shape
    M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), -37.0, 1.0)
    g = cv2.warpAffine(f, M, (w, h))
    mg = (cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST) > 0.5).astype(np.float32)
    a = Prepared("a", "s", "1", 12.0, f, m)
    b = Prepared("b", "s", "1", 12.0, g * mg, mg)
    c, ang, dy, dx = register(a, b)
    assert abs(((ang - 37.0) + 180) % 360 - 180) <= 1.0
    assert c > 0.9


def test_unrelated_textures_score_low():
    f, m = _texture(seed=2)
    g, mg = _texture(seed=3)
    c, *_ = register(Prepared("a", "s", "1", 12.0, f, m), Prepared("b", "s", "2", 12.0, g, mg))
    assert c < 0.35


def test_auc_and_identity():
    assert auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auc([0.1], [0.9]) == 0.0
    assert identity("9mm__E3__G0002_A0002.001_CC03_TF") == ("E3", "G0002")
    assert identity("9mm__Fadul__Fadul_7-2") == ("Fadul", "Fadul7")


# --------------------------------------------------------------------------- statistics

def _toy_pairs(n_guns=5, n_scans=3, seed=0, sep=0.25):
    """All within-study pairs of a toy study; same-gun pairs score higher on average."""
    rng = np.random.default_rng(seed)
    names = [f"9mm__Toy__Toy_{g}-{s}" for g in range(1, n_guns + 1) for s in range(1, n_scans + 1)]
    rows = []
    for a, b in itertools.combinations(names, 2):
        m = int(identity(a)[1] == identity(b)[1])
        base = rng.normal(0.4 + sep * m, 0.12)
        rows.append(dict(a=a, b=b, study="Toy", match=m, ccf_breech_face=base,
                         ccf_primer_disc=base - 0.1 * m + rng.normal(0, 0.05)))
    return rows


def test_bootstrap_ci_contains_point_estimate_and_is_seeded():
    rows = _toy_pairs()
    kinds = ["breech_face", "primer_disc"]
    s1 = summarize(rows, kinds, n_boot=400, seed=7)["Toy"]
    s2 = summarize(rows, kinds, n_boot=400, seed=7)["Toy"]
    s3 = summarize(rows, kinds, n_boot=400, seed=8)["Toy"]
    for k in kinds:
        lo, hi = s1[k]["auc_ci95"]
        assert lo <= s1[k]["auc"] <= hi
        assert lo < hi
    assert s1 == s2                                     # reproducible with a seed
    assert s1["breech_face"]["auc_ci95"] != s3["breech_face"]["auc_ci95"]
    d = s1["paired_auc_difference"]["breech_face_minus_primer_disc"]
    lo, hi = d["ci95"]
    assert lo <= d["auc_diff"] <= hi
    assert 0.0 <= d["frac_reps_gt_0"] + d["frac_reps_eq_0"] <= 1.0
    b = s1["bootstrap"]
    assert (b["firearms"], b["scans"], b["reps"]) == (5, 15, 400)
    assert b["possible_distinct_resamples"] == 121       # C(9, 5) - 5


def test_cluster_resample_never_pairs_a_firearm_with_itself_as_non_match():
    rows = _toy_pairs(n_guns=3, n_scans=3)
    rng = np.random.default_rng(0)
    saw_duplicate_draw = False
    for _ in range(300):
        idx = resample_pairs(rows, rng)
        for i in idx:
            fa, fb = identity(rows[i]["a"])[1], identity(rows[i]["b"])[1]
            assert rows[i]["match"] == int(fa == fb)       # labels follow the original firearms
        km = [i for i in idx if rows[i]["match"]]
        # a firearm drawn twice contributes its 3 same-gun pairs twice, never cross-copy pairs
        saw_duplicate_draw |= len(km) > len(set(km))
        assert len(km) % 3 == 0
    assert saw_duplicate_draw


def test_cluster_resample_rebuilds_pairs_with_multiplicity():
    rows = _toy_pairs(n_guns=3, n_scans=2)

    class Fixed:                                          # draws Toy1 twice and Toy3 once
        def choice(self, n, size, replace):
            return np.array([0, 0, 2])

    idx = resample_pairs(rows, Fixed())
    pairs = [tuple(sorted((identity(rows[i]["a"])[1], identity(rows[i]["b"])[1]))) for i in idx]
    assert pairs.count(("Toy1", "Toy1")) == 2            # 1 same-gun pair x 2 copies
    assert pairs.count(("Toy3", "Toy3")) == 1
    assert pairs.count(("Toy1", "Toy3")) == 2 * 4         # 2x2 scans, between each copy and Toy3
    assert not any("Toy2" in p for p in pairs)


def test_single_firearm_resamples_are_dropped():
    rows = _toy_pairs(n_guns=2, n_scans=2)
    boot = cluster_bootstrap(rows, ["breech_face"], n_boot=200, seed=1)["breech_face"]
    assert np.isnan(boot).any() and np.isfinite(boot).any()   # P(one gun drawn twice) = 1/2


def test_eer_toy_examples():
    # threshold 0.55: one false reject (0.4) and one false accept (0.7) out of four each
    e, t = eer([0.4, 0.6, 0.8, 0.9], [0.1, 0.2, 0.5, 0.7])
    assert e == pytest.approx(0.25)
    assert t == pytest.approx(0.55)
    # perfect separation: EER 0, threshold in the middle of the gap
    e, t = eer([0.8, 0.9], [0.1, 0.3, 0.6])
    assert e == 0.0 and t == pytest.approx(0.7)
    # complete overlap of identical scores: chance level
    e, _ = eer([0.5, 0.5], [0.5, 0.5])
    assert e == pytest.approx(0.5)


def test_separation_and_roc():
    d, dp = separation([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])
    assert d == pytest.approx(1.0) and dp == pytest.approx(1.0)
    far, tar = roc_curve([0.9, 0.8], [0.1, 0.85])
    assert far[0] == tar[0] == 0.0 and far[-1] == tar[-1] == 1.0
    assert np.all(np.diff(far) >= 0) and np.all(np.diff(tar) >= 0)
    # trapezoid area under the step ROC equals the rank AUC
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    assert trapezoid(tar, far) == pytest.approx(auc([0.9, 0.8], [0.1, 0.85]))


def test_summarize_only_round_trip(tmp_path):
    rows = _toy_pairs(n_guns=4, n_scans=2)
    keys = ["a", "b", "study", "match", "ccf_breech_face", "ccf_primer_disc"]
    with open(tmp_path / "pairs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    main(["--summarize-only", "-o", str(tmp_path), "--bootstrap-reps", "50"])
    s = json.loads((tmp_path / "summary.json").read_text())["Toy"]
    for key in ("known_match_pairs", "known_non_match_pairs"):     # backward-compatible keys
        assert key in s
    for key in ("auc", "km_mean", "knm_mean", "gap_min_km_minus_max_knm", "auc_ci95", "eer", "d_prime"):
        assert key in s["breech_face"]
    assert (tmp_path / "score_distributions.png").exists()
    assert (tmp_path / "roc_curves.png").exists()
