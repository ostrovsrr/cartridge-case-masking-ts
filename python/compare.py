"""Does the mask help?  Breech-face comparison of repeat firings.

The point of masking a breech-face impression is to compare cartridge cases:
two cases fired from the same pistol should correlate better than cases from
different pistols.  This module measures that, as a *downstream* check of the
masks that needs no hand-drawn ground truth:

  1. each scan's masks are read from the TypeScript CLI output (``labels.png``
     + ``masks.json``; the height map and datum are rebuilt from the X3P by
     ``ts_masks``); the chosen region (our breech-face mask, the whole primer
     disc, or the full scan) is cropped around the primer centre;
  2. the height residual is band-pass filtered (mask-aware Gaussian filters,
     keeping roughly 20-250 um wavelengths, i.e. tool marks, not form);
  3. every pair is registered over rotation (coarse-to-fine) and translation
     with a masked normalised cross-correlation (Padfield 2012), and the
     maximum ACCF is the similarity score;
  4. known-match (same firearm) and known-non-match (different firearm, same
     study) score distributions are compared (ROC AUC, separation gap).

Pairs share scans and firearms, so they are not independent observations.
Uncertainty is therefore estimated with a firearm-level cluster bootstrap:
firearms are resampled with replacement within a study, all pairs among the
resampled firearms' scans are rebuilt, and AUC is recomputed (see
``cluster_bootstrap``).  The same resamples give a paired comparison of the
region choices.  Equal-error rate and d' are reported alongside.

    npx tsx src/cli.ts path/to/matching -o outputs/ts-matching          # masks (TypeScript)
    python python/compare.py outputs/ts-matching -o outputs/matching    # registration + summary
    python python/compare.py --summarize-only -o outputs/matching       # reuse pairs.csv

Port of ``cartmask/compare.py``: the registration, band-pass, statistics and
plots are unchanged; only the source of the masks differs (the TypeScript
product instead of ``cartmask.segment``), so the evidence is about the masks
the TypeScript CLI ships.  The X3P of each scan is found from
``masks.json["source"]`` or, failing that, as ``<name>.x3p`` in ``--x3p-dir``.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import fft as sfft
from scipy import ndimage as ndi

from cartmask_shared import BREECH_FACE, Result
from ts_masks import load_scan, scan_dirs

MASK_KINDS = ("breech_face", "primer_disc", "full_scan")


@dataclass
class Prepared:
    name: str
    study: str
    firearm: str
    um_per_px: float
    img: np.ndarray       # band-passed height (um), 0 outside mask
    mask: np.ndarray      # float32 0/1


def identity(name: str) -> tuple[str, str]:
    """(study, firearm) from file names written by scripts/fetch_nbtrd.py."""
    parts = name.split("__")
    study = parts[1] if len(parts) > 2 else "unknown"
    spec = parts[-1]
    m = re.match(r"(G\d+)_", spec)                 # E3: G0002_A0002.001_CC01_TF
    if m:
        return study, m.group(1)
    m = re.match(r"(.+?)_?(\d+)-\d+$", spec)        # Fadul_1-2, Lightstone_SW_6-1
    if m:
        return study, f"{m.group(1)}{m.group(2)}"
    return study, spec


def _masked_gauss(img, mask, sigma):
    num = ndi.gaussian_filter(img * mask, sigma)
    den = ndi.gaussian_filter(mask, sigma)
    return num / np.maximum(den, 1e-6)


def region_mask(res: Result, kind: str) -> np.ndarray:
    if kind == "breech_face":
        return res.labels == BREECH_FACE
    if kind == "primer_disc":
        yy, xx = np.indices(res.labels.shape)
        cx, cy = res.primer_center_px
        return (np.hypot(xx - cx, yy - cy) <= res.primer_radius_px) & res.valid
    if kind == "full_scan":
        return res.valid.copy()
    raise ValueError(kind)


def _remove_circular(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Subtract the per-radius median around the crop centre (= primer centre).

    Rotationally symmetric structure (primer-cup rounding, aperture step,
    flow-back ring) is shared by every cartridge and correlates at any
    rotation; removing it leaves the individual tool marks (cf. Tai & Eddy 2018).
    """
    h, w = img.shape
    yy, xx = np.indices(img.shape)
    rb = np.round(np.hypot(xx - (w - 1) / 2, yy - (h - 1) / 2)).astype(int)
    out = img.copy()
    r = rb[mask]
    v = img[mask]
    order = np.argsort(r, kind="stable")
    r, v = r[order], v[order]
    cuts = np.flatnonzero(np.diff(r)) + 1
    med = np.concatenate([np.full(len(c), np.median(c)) for c in np.split(v, cuts)])
    vals = np.empty_like(v)
    vals[order] = v - med
    out[mask] = vals
    return out * mask


def prepare(res: Result, kind: str, um_per_px: float = 12.0, crop_mm: float = 5.0,
            band_um: tuple[float, float] = (20.0, 250.0), remove_circular: bool = True) -> Prepared:
    """Crop a square around the primer centre, band-pass, resample."""
    study, firearm = identity(res.name)
    m = region_mask(res, kind) & res.valid
    d = np.where(res.valid, res.depth, 0.0) * 1e6
    # crop (pad if needed) a fixed physical window centred on the primer
    half = int(round(crop_mm / 2 * 1000 / res.um_per_px))
    cx, cy = (int(round(v)) for v in res.primer_center_px)
    pad = half + 2
    d = np.pad(d, pad)
    m = np.pad(m, pad)
    sl = (slice(cy + pad - half, cy + pad + half), slice(cx + pad - half, cx + pad + half))
    d, m = d[sl], m[sl].astype(np.float64)
    # band-pass at native working resolution (sigma = cutoff / (2*pi) in px, roughly)
    lo, hi = band_um
    s_lo = lo / res.um_per_px / 2.0
    s_hi = hi / res.um_per_px / 2.0
    smooth = _masked_gauss(d, m, max(s_lo, 0.5))
    form = _masked_gauss(d, m, s_hi)
    bp = (smooth - form) * m
    if remove_circular:
        bp = _remove_circular(bp, m > 0)
    # resample to the comparison grid
    size = int(round(crop_mm * 1000 / um_per_px))
    bp = cv2.resize(bp, (size, size), interpolation=cv2.INTER_AREA)
    mk = cv2.resize(m, (size, size), interpolation=cv2.INTER_AREA) > 0.75
    # erode a little: filter edge effects live at the mask boundary
    mk = ndi.binary_erosion(mk, iterations=1)
    bp = np.where(mk, bp, 0.0)
    if mk.any():
        bp[mk] -= bp[mk].mean()
    return Prepared(res.name, study, firearm, um_per_px, bp.astype(np.float32), mk.astype(np.float32))


# --------------------------------------------------------------------------- registration

def masked_ncc(f, mf, g, mg, max_shift: int, min_overlap_frac: float = 0.3):
    """Masked normalised cross-correlation for all shifts |dx|,|dy| <= max_shift.

    Padfield, "Masked object registration in the Fourier domain", IEEE TIP 2012.
    Returns (best ncc, dy, dx) where g shifted by (dy, dx) best matches f.
    """
    H, W = f.shape
    # pad to fast FFT lengths: sizes with large prime factors are very slow
    s = (sfft.next_fast_len(2 * H - 1, real=True), sfft.next_fast_len(2 * W - 1, real=True))
    fft = lambda a, shape: sfft.rfft2(a, shape, workers=-1)          # noqa: E731
    ifft = lambda a, shape: sfft.irfft2(a, shape, workers=-1)        # noqa: E731
    gr, mgr = g[::-1, ::-1], mg[::-1, ::-1]
    F, F2, MF = fft(f, s), fft(f * f, s), fft(mf, s)
    G, G2, MG = fft(gr, s), fft(gr * gr, s), fft(mgr, s)
    n = ifft(MF * MG, s)
    sf, sg = ifft(F * MG, s), ifft(MF * G, s)
    sff, sgg = ifft(F2 * MG, s), ifft(MF * G2, s)
    sfg = ifft(F * G, s)
    nn = np.maximum(n, 1)
    num = sfg - sf * sg / nn
    den = np.sqrt(np.maximum((sff - sf ** 2 / nn) * (sgg - sg ** 2 / nn), 1e-12))
    ncc = num / den
    min_n = min_overlap_frac * min(mf.sum(), mg.sum())
    ncc[n < min_n] = -1.0
    c0, c1 = H - 1, W - 1                               # zero shift
    win = ncc[c0 - max_shift:c0 + max_shift + 1, c1 - max_shift:c1 + max_shift + 1]
    i = int(np.argmax(win))
    dy, dx = np.unravel_index(i, win.shape)
    return float(win.flat[i]), int(dy - max_shift), int(dx - max_shift)


def _rotate(img, angle_deg, order=cv2.INTER_LINEAR):
    h, w = img.shape
    M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=order, borderValue=0)


def _down(p: Prepared, k: int):
    if k == 1:
        return p.img, p.mask
    h, w = p.img.shape
    img = cv2.resize(p.img, (w // k, h // k), interpolation=cv2.INTER_AREA)
    msk = cv2.resize(p.mask, (w // k, h // k), interpolation=cv2.INTER_AREA) > 0.75
    return np.where(msk, img, 0).astype(np.float32), msk.astype(np.float32)


def register(a: Prepared, b: Prepared, max_shift_mm: float = 0.5, coarse_step: float = 4.0):
    """Best (ncc, angle, dy, dx) aligning b onto a; coarse 2x-down search, then refine."""
    f2, mf2 = _down(a, 2)
    g2, mg2 = _down(b, 2)
    ms2 = int(round(max_shift_mm * 1000 / (a.um_per_px * 2)))
    best = (-2.0, 0.0, 0, 0)
    for ang in np.arange(-180, 180, coarse_step):
        g, mg = _rotate(g2, ang), (_rotate(mg2, ang, cv2.INTER_NEAREST) > 0.5).astype(np.float32)
        c, dy, dx = masked_ncc(f2, mf2, g * mg, mg, ms2)
        if c > best[0]:
            best = (c, ang, dy, dx)
    ms = int(round(max_shift_mm * 1000 / a.um_per_px))
    fine = (-2.0, 0.0, 0, 0)
    for ang in np.arange(best[1] - coarse_step, best[1] + coarse_step + 1e-9, 1.0):
        g, mg = _rotate(b.img, ang), (_rotate(b.mask, ang, cv2.INTER_NEAREST) > 0.5).astype(np.float32)
        c, dy, dx = masked_ncc(a.img, a.mask, g * mg, mg, ms)
        if c > fine[0]:
            fine = (c, float(ang), dy, dx)
    return fine


def mask_iou_after_registration(ra: Result, rb: Result, angle, dy, dx, um_per_px, crop_mm=5.0):
    """IoU of the two breech-face masks once b is aligned onto a."""
    pa = prepare(ra, "breech_face", um_per_px, crop_mm)
    pb = prepare(rb, "breech_face", um_per_px, crop_mm)
    mb = _rotate(pb.mask, angle, cv2.INTER_NEAREST)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    mb = cv2.warpAffine(mb, M, mb.shape[::-1], flags=cv2.INTER_NEAREST) > 0.5
    ma = pa.mask > 0.5
    return float((ma & mb).sum() / max((ma | mb).sum(), 1))


def auc(pos, neg):
    """Probability a random known-match outscores a random known-non-match."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


# --------------------------------------------------------------------------- statistics

def eer(pos, neg) -> tuple[float, float]:
    """Equal-error rate and the score threshold where it occurs.

    A pair is called a match when ``score >= threshold``.  Candidate thresholds
    are midpoints between consecutive distinct scores (plus one below and one
    above all scores); the EER is (FAR + FRR) / 2 at the candidate where false
    accepts (KNM >= t) and false rejects (KM < t) are closest.  Ties use the lowest total error and then the lowest candidate threshold.
    """
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan"), float("nan")
    u = np.unique(np.concatenate([pos, neg]))
    t = np.concatenate([[u[0] - 1.0], (u[:-1] + u[1:]) / 2, [u[-1] + 1.0]])
    far = (neg[None, :] >= t[:, None]).mean(axis=1)
    frr = (pos[None, :] < t[:, None]).mean(axis=1)
    gap = np.abs(far - frr)
    best = np.flatnonzero(np.isclose(gap, gap.min()))
    # among equally balanced candidates prefer the lowest total error
    tot = (far + frr)[best]
    best = best[np.isclose(tot, tot.min())]
    return float((far[best[0]] + frr[best[0]]) / 2), float(t[best[0]])


def roc_curve(pos, neg) -> tuple[np.ndarray, np.ndarray]:
    """(false-accept rate, true-accept rate) as the threshold sweeps high to low."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    t = np.unique(np.concatenate([pos, neg]))[::-1]
    tar = np.concatenate([[0.0], (pos[None, :] >= t[:, None]).mean(axis=1)])
    far = np.concatenate([[0.0], (neg[None, :] >= t[:, None]).mean(axis=1)])
    return far, tar


def separation(pos, neg) -> tuple[float, float]:
    """(Cohen's d with pooled SD, d' = mean gap / sqrt of the mean variance)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 2 or len(neg) < 2:
        return float("nan"), float("nan")
    vp, vn = pos.var(ddof=1), neg.var(ddof=1)
    diff = pos.mean() - neg.mean()
    pooled = np.sqrt(((len(pos) - 1) * vp + (len(neg) - 1) * vn) / (len(pos) + len(neg) - 2))
    mean_sd = np.sqrt((vp + vn) / 2)
    return (float(diff / pooled) if pooled > 0 else float("nan"),
            float(diff / mean_sd) if mean_sd > 0 else float("nan"))


def _firearms(rows) -> tuple[list[str], dict, dict]:
    """Firearms of a study, KM pair indices per firearm, KNM pair indices per firearm pair."""
    km, knm, guns = {}, {}, set()
    for i, r in enumerate(rows):
        fa, fb = identity(r["a"])[1], identity(r["b"])[1]
        guns |= {fa, fb}
        if fa == fb:
            km.setdefault(fa, []).append(i)
        else:
            knm.setdefault(frozenset((fa, fb)), []).append(i)
    return sorted(guns), km, knm


def resample_pairs(rows, rng: np.random.Generator) -> list[int]:
    """One firearm-level cluster-bootstrap resample of the pairs of one study.

    Firearms are drawn with replacement and every pair among the drawn
    firearms' scans is rebuilt (as indices into ``rows``, repeated when a
    firearm is drawn more than once):

    * each drawn copy of a firearm contributes its own known-match pairs;
    * each two drawn copies of *different* firearms contribute all known-non-
      match pairs between them;
    * two copies of the *same* firearm contribute nothing: their cross pairs
      would be same-gun pairs mislabelled as different-gun (or a scan paired
      with itself), so they are never formed.
    """
    guns, km, knm = _firearms(rows)
    draw = rng.choice(len(guns), size=len(guns), replace=True)
    out: list[int] = []
    for i in draw:
        out += km.get(guns[i], [])
    for x, y in itertools.combinations(draw, 2):
        if x != y:
            out += knm.get(frozenset((guns[x], guns[y])), [])
    return out


def cluster_bootstrap(rows, kinds, n_boot: int = 2000, seed: int = 0) -> dict[str, np.ndarray]:
    """Bootstrap AUC per region (same resamples for every region, so differences are paired).

    Resamples that contain a single distinct firearm have no known-non-match
    pairs; their AUC is undefined and they are returned as NaN.
    """
    rng = np.random.default_rng(seed)
    match = np.array([int(r["match"]) for r in rows], bool)
    scores = {k: np.array([float(r[f"ccf_{k}"]) for r in rows]) for k in kinds}
    out = {k: np.full(n_boot, np.nan) for k in kinds}
    for b in range(n_boot):
        idx = np.asarray(resample_pairs(rows, rng), int)
        if not len(idx):
            continue
        m = match[idx]
        for k in kinds:
            s = scores[k][idx]
            out[k][b] = auc(s[m], s[~m])
    return out


def _ci(x: np.ndarray, level: float = 0.95) -> list[float] | None:
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    a = (1 - level) / 2
    return [round(float(np.quantile(x, a)), 3), round(float(np.quantile(x, 1 - a)), 3)]


def _r(v, d=3):
    return None if v is None or not np.isfinite(v) else round(float(v), d)


def summarize(rows, kinds, n_boot: int = 2000, seed: int = 0) -> dict:
    """Per-study summary: point estimates, firearm-level bootstrap CIs, paired differences."""
    summary = {}
    for study in sorted({r["study"] for r in rows}):
        rs = [r for r in rows if r["study"] == study]
        guns, _, _ = _firearms(rs)
        scans = {r["a"] for r in rs} | {r["b"] for r in rs}
        summary[study] = {"known_match_pairs": sum(int(r["match"]) for r in rs),
                          "known_non_match_pairs": sum(1 - int(r["match"]) for r in rs)}
        # seed per study from its name, so a study's numbers do not depend on which others are present
        boot = cluster_bootstrap(rs, kinds, n_boot, seed=seed + zlib.crc32(study.encode()))
        for k in kinds:
            pos = [float(r[f"ccf_{k}"]) for r in rs if int(r["match"])]
            neg = [float(r[f"ccf_{k}"]) for r in rs if not int(r["match"])]
            e, thr = eer(pos, neg)
            d, dp = separation(pos, neg)
            summary[study][k] = dict(
                auc=_r(auc(pos, neg)),
                km_mean=round(float(np.mean(pos)), 3) if pos else None,
                knm_mean=round(float(np.mean(neg)), 3) if neg else None,
                gap_min_km_minus_max_knm=round(float(min(pos) - max(neg)), 3) if pos and neg else None,
                auc_ci95=_ci(boot[k]),
                eer=_r(e), eer_threshold=_r(thr),
                cohens_d=_r(d, 2), d_prime=_r(dp, 2))
        if "breech_face" in kinds:
            diffs = {}
            for other in (k for k in kinds if k != "breech_face"):
                dd = boot["breech_face"] - boot[other]
                dd = dd[np.isfinite(dd)]
                base_auc = summary[study]["breech_face"]["auc"]
                other_auc = summary[study][other]["auc"]
                diffs[f"breech_face_minus_{other}"] = dict(
                    auc_diff=_r(base_auc - other_auc) if base_auc is not None and other_auc is not None else None,
                    ci95=_ci(dd),
                    frac_reps_gt_0=_r(np.mean(dd > 1e-12) if len(dd) else np.nan),
                    frac_reps_eq_0=_r(np.mean(np.abs(dd) <= 1e-12) if len(dd) else np.nan))
            summary[study]["paired_auc_difference"] = diffs
        summary[study]["bootstrap"] = dict(
            method="firearm-level cluster bootstrap within study; percentile 95% CI",
            firearms=len(guns), scans=len(scans), reps=n_boot,
            valid_reps=int(np.isfinite(boot[kinds[0]]).sum()), seed=seed,
            # multisets of n firearms drawn from n, minus the n single-firearm ones: with few
            # firearms the bootstrap distribution has only a handful of distinct values
            possible_distinct_resamples=math.comb(2 * len(guns) - 1, len(guns)) - len(guns),
            note="reps drawing a single distinct firearm have no different-gun pairs and are dropped")
        ious = [float(r["mask_iou"]) for r in rs if r.get("mask_iou") not in (None, "")]
        if ious:
            summary[study]["breech_face_mask_iou_known_matches"] = round(float(np.mean(ious)), 3)
    return summary


# --------------------------------------------------------------------------- CLI

def read_pairs(path: Path) -> tuple[list[dict], list[str]]:
    """Rows of a pairs.csv written by this module, with numeric columns parsed."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["match"] = int(r["match"])
        for k, v in list(r.items()):
            if k.startswith("ccf_") or k == "mask_iou":
                if v in (None, ""):
                    del r[k]
                else:
                    r[k] = float(v)
    cols = {c[4:] for r in rows for c in r if c.startswith("ccf_")}
    kinds = [k for k in MASK_KINDS if k in cols] + sorted(cols - set(MASK_KINDS))
    return rows, kinds


def write_summary(rows, kinds, out: Path, n_boot: int = 2000, seed: int = 0) -> dict:
    summary = summarize(rows, kinds, n_boot, seed)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    _plot(rows, kinds, summary, out / "score_distributions.png")
    _plot_roc(rows, kinds, summary, out / "roc_curves.png")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*",
                    help="TypeScript CLI output folders: an output root or per-scan folders with "
                         "labels.png + masks.json (or, with --summarize-only, a pairs.csv)")
    ap.add_argument("-o", "--out", default="outputs/matching")
    ap.add_argument("--x3p-dir", action="append", default=[],
                    help="where to look for <scan>.x3p when masks.json['source'] does not resolve "
                         "(repeatable)")
    ap.add_argument("--pairs-from", type=Path, default=None,
                    help="only register the (a, b) pairs listed in this pairs.csv (e.g. a reference run)")
    ap.add_argument("--max-pairs", type=int, default=None, help="stop after this many pairs")
    ap.add_argument("--um-per-px", type=float, default=12.0, help="comparison resolution")
    ap.add_argument("--kinds", default=",".join(MASK_KINDS))
    ap.add_argument("--summarize-only", action="store_true",
                    help="skip registration: recompute summary.json and plots from an existing "
                         "pairs.csv (the input if given, else OUT/pairs.csv)")
    ap.add_argument("--bootstrap-reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.summarize_only:
        src = Path(args.inputs[0]) if args.inputs else out / "pairs.csv"
        if src.is_dir():
            src = src / "pairs.csv"
        rows, kinds = read_pairs(src)
        print(f"summarising {len(rows)} pairs from {src}")
    else:
        if not args.inputs:
            ap.error("give TypeScript CLI output folders, or --summarize-only")
        only = None
        if args.pairs_from:
            with open(args.pairs_from, newline="") as fh:
                only = {(r["a"], r["b"]) for r in csv.DictReader(fh)}
        rows, kinds = _compare_scans(args.inputs, args.kinds.split(","), args.um_per_px, out,
                                     args.x3p_dir, only, args.max_pairs)
    summary = write_summary(rows, kinds, out, args.bootstrap_reps, args.seed)
    print(json.dumps(summary, indent=2))


def _compare_scans(inputs, kinds, um_per_px, out: Path, x3p_dirs=(), only=None, max_pairs=None):
    dirs = scan_dirs(inputs)
    if only is not None:
        wanted = {n for pair in only for n in pair}
        dirs = [d for d in dirs if d.name in wanted]

    results: dict[str, Result] = {}
    for d in dirs:
        t = time.time()
        r = load_scan(d, x3p_dirs)
        if r.status in ("skipped", "failed"):
            print(f"skip {d.name}: {r.status}")
            continue
        r.z = None  # free memory; depth/labels/valid are kept
        results[r.name] = r
        chk = r.datum_check
        print(f"loaded {r.name} ({time.time() - t:.1f}s; datum check "
              f"{'ok' if chk and chk['ok'] else 'MISMATCH'})")

    prepared = {k: {n: prepare(r, k, um_per_px) for n, r in results.items()} for k in kinds}
    names = sorted(results)
    rows = []
    for a, b in itertools.combinations(names, 2):
        if only is not None and (a, b) not in only:
            continue
        if max_pairs is not None and len(rows) >= max_pairs:
            break
        sa, fa = identity(a)
        sb, fb = identity(b)
        if sa != sb:
            continue  # compare within a study (same instrument and ammunition protocol)
        row = dict(a=a, b=b, study=sa, match=int(fa == fb))
        for k in kinds:
            t = time.time()
            c, ang, dy, dx = register(prepared[k][a], prepared[k][b])
            row[f"ccf_{k}"] = round(c, 4)
            if k == "breech_face":
                row["angle"], row["dy"], row["dx"] = ang, dy, dx
                if fa == fb:
                    row["mask_iou"] = round(mask_iou_after_registration(
                        results[a], results[b], ang, dy, dx, um_per_px), 3)
            row[f"sec_{k}"] = round(time.time() - t, 1)
        rows.append(row)
        print(f"{'KM ' if row['match'] else 'KNM'} {a[-24:]:>24} vs {b[-24:]:<24} " +
              " ".join(f"{k}={row[f'ccf_{k}']:.3f}" for k in kinds))

    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("a", "b", "study", "match"), k))
    with open(out / "pairs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return rows, kinds


# --------------------------------------------------------------------------- plots

TITLES = {"breech_face": "Our breech-face mask", "primer_disc": "Whole primer disc",
          "full_scan": "No mask (full scan)"}
COLOURS = {"breech_face": "#d0342c", "primer_disc": "#2f6fb0", "full_scan": "#8a8f98"}


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def _auc_label(st: dict) -> str:
    ci = st.get("auc_ci95")
    return f"AUC {st['auc']:.2f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")


def _plot(rows, kinds, summary, path):
    plt = _pyplot()
    if plt is None:
        return
    studies = sorted({r["study"] for r in rows})
    fig, axes = plt.subplots(len(studies), len(kinds), figsize=(4.2 * len(kinds), 3.1 * len(studies)),
                             squeeze=False, sharex=True)
    for i, s in enumerate(studies):
        rs = [r for r in rows if r["study"] == s]
        for j, k in enumerate(kinds):
            ax = axes[i, j]
            pos = [r[f"ccf_{k}"] for r in rs if r["match"]]
            neg = [r[f"ccf_{k}"] for r in rs if not r["match"]]
            ax.scatter(neg, np.random.default_rng(0).uniform(0.05, 0.45, len(neg)), s=22,
                       color="#8a8f98", label="different gun")
            ax.scatter(pos, np.random.default_rng(1).uniform(0.55, 0.95, len(pos)), s=26,
                       color="#d0342c", label="same gun")
            st = summary[s][k]
            if st.get("eer_threshold") is not None:
                ax.axvline(st["eer_threshold"], color="#444", lw=0.8, ls=":")
            ax.set_yticks([])
            ax.set_xlim(-0.05, 1.0)
            ax.set_title(f"{s}: {TITLES.get(k, k)}\n{_auc_label(st)}, EER {st['eer']:.2f}", fontsize=10)
            if i == len(studies) - 1:
                ax.set_xlabel("max areal cross-correlation")
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
    axes[0, 0].legend(loc="upper right", fontsize=8, frameon=False)
    fig.text(0.5, 0.005, "AUC with 95% firearm-level cluster-bootstrap interval; dotted line = EER threshold",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_roc(rows, kinds, summary, path):
    plt = _pyplot()
    if plt is None:
        return
    studies = sorted({r["study"] for r in rows})
    fig, axes = plt.subplots(1, len(studies), figsize=(4.6 * len(studies), 4.6), squeeze=False)
    for ax, s in zip(axes[0], studies):
        rs = [r for r in rows if r["study"] == s]
        for k in kinds:
            pos = [r[f"ccf_{k}"] for r in rs if r["match"]]
            neg = [r[f"ccf_{k}"] for r in rs if not r["match"]]
            far, tar = roc_curve(pos, neg)
            ours = k == "breech_face"
            ax.plot(far, tar, color=COLOURS.get(k), lw=2.2 if ours else 1.4, zorder=3 if ours else 2,
                    label=f"{TITLES.get(k, k)}: {_auc_label(summary[s][k])}")
        ax.plot([0, 1], [1, 0], color="#bbb", lw=0.8, ls=":")   # FAR = FRR: EER where a curve crosses
        ax.plot([0, 1], [0, 1], color="#ddd", lw=0.8)
        st = summary[s]
        ax.set_title(f"{s}: {st['known_match_pairs']} same-gun / {st['known_non_match_pairs']} different-gun "
                     f"pairs,\n{st['bootstrap']['firearms']} firearms", fontsize=10)
        ax.set_xlabel("false-accept rate (different-gun pairs above threshold)")
        ax.set_ylabel("true-accept rate (same-gun pairs above threshold)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.legend(loc="lower right", fontsize=7.5, frameon=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
