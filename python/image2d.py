"""Best-effort 2D mode: mask a single shaded *image* of a cartridge-case head.

The main pipeline (``cartmask.segment``) works on calibrated 3D topography.
A reviewer may instead have only a picture - for example the shaded renders
in the assignment brief.  Shading does not determine height, so this mode is
weaker by construction; it is provided for convenience and its output is
always flagged for review.

    python python/image2d.py some_render.png -o outputs/2d [--px-um 3.57]
    python python/image2d.py --eval-brief python/samples/brief      # brief figures
    python python/image2d.py --eval-renders outputs/ts-raw          # 2D vs the TS 3D masks

Copy of ``cartmask/image2d.py``.  The 2D method is unchanged; ``--eval-renders``
now takes the TypeScript CLI's output (``labels.png`` + ``masks.json`` per
scan, X3P re-read for the shaded render) instead of running ``cartmask.segment``,
so the 3D reference masks are the TypeScript product's.

Method (all sizes relative to the image or to the detected primer):

1. Primer - unsigned gradient-voting circular Hough (shading flips sign
   around a circular edge under oblique light) plus ``cv2.HoughCircles``
   candidates, radius 0.14-0.55 x image size (1.5-3.0 mm when ``px_um`` is
   given).  Every candidate is scored by closed-contour support: the fraction
   of 180 rays with a strong radial edge on the circle.  Circles whose outside
   is image background (the case-head outline of a render) are rejected, and
   an enclosing circle is preferred over a nested one (the crater).  The
   winner is refined by a robust circle fit to the strongest radial edge on
   360 rays.
2. Firing-pin impression - inside the primer the crater is the region whose
   *low-pass* shading departs most from the breech-face level and whose
   low-pass shading gradient is strongest (a crater gives a large, smooth,
   roughly radially symmetric light/dark pattern; breech-face marks are
   high-frequency).  A second gradient-voting Hough search at 0.15-0.5 x the
   primer radius gives the pin-strike circle; the two are cross-checked.
3. Drag - the 3D pipeline's test (``segment.find_drag``) on the outline:
   a tail outside the strike circle that points one way.  Only then is the
   tail labelled firing-pin shear.
4. Breech face - primer disc minus the rounded rim, minus 1.5 x pin radius
   around the strike and a margin around the whole impression.
5. Aperture shear is *not* attempted: in a shaded image the striations
   cannot be separated from lighting texture.  It is left empty and the
   output says so.

Label values and colours are the 3D pipeline's.  Physical scale is unknown
unless ``px_um`` is given; geometry is reported in input-image pixels and as
fractions of the primer radius.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

from cartmask_shared import (APERTURE_SHEAR, ARROW, BREECH_FACE, COLOURS, FIRING_PIN_IMPRESSION,
                             FIRING_PIN_SHEAR, LABEL_NAMES, LEGEND, Params, Result, _fit, _round,
                             _scale_bar, contact_sheet, crop_box, find_drag, fit_core, flat_mask,
                             overlay, render, set_brief_colours)
from cartmask_shared import disk as _disk
from cartmask_shared import robust_circle as _robust_circle
from cartmask_shared import sample as _sample
from ts_masks import load_scans


@dataclass
class Params2D:
    work_max_px: int = 700                     # images are processed at most this size
    primer_r_frac: tuple[float, float] = (0.14, 0.55)   # of min(h, w), scale unknown
    primer_r_mm: tuple[float, float] = (1.5, 3.0)        # when px_um is given
    pin_r_frac: tuple[float, float] = (0.15, 0.5)        # of primer radius
    rim_margin_frac: float = 0.05              # as in the 3D pipeline
    fpi_min_score: float = 3.0                 # crater score vs breech-face median
    aperture_ratio: float = 1.5                # aperture radius / pin radius (3D default)
    fpi_margin_frac: float = 0.25              # extra margin around the whole impression / pin radius
    min_primer_inside: float = 0.75


@dataclass
class Result2D(Result):
    """``segment.Result`` plus 2D-specific fields; arrays are at working size."""
    work_scale: float = 1.0                    # working px / input px
    input_shape: tuple[int, int] | None = None
    px_um: float | None = None                 # input-image pixel size, if known
    primer_support: float | None = None
    fpi_source: str = "none"                   # "blob+circle", "blob", "circle", "none"
    pin_circle_support: float | None = None
    drag_length_px: float | None = None
    image: np.ndarray | None = None            # working-size BGR image (not serialised)

    def labels_at(self, shape) -> np.ndarray:
        """Labels resampled (nearest) to ``shape`` - e.g. the input image size."""
        return cv2.resize(self.labels, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

    def summary(self) -> dict:
        d = asdict(self)
        for k in ("z", "valid", "depth", "labels", "image", "um_per_px", "fp_max_depth_um",
                  "roughness_um", "drag_length_mm"):
            d.pop(k, None)
        s = 1.0 / self.work_scale
        geo = {}
        if self.primer_radius_px:
            R = self.primer_radius_px
            geo["primer_center_px"] = [c * s for c in self.primer_center_px]
            geo["primer_radius_px"] = R * s
            if self.fp_core_radius_px:
                fx, fy = self.fp_core_center_px
                geo["fp_core_center_px"] = [fx * s, fy * s]
                geo["fp_core_radius_px"] = self.fp_core_radius_px * s
                geo["fp_core_radius_frac"] = self.fp_core_radius_px / R
                geo["fp_offset_frac"] = float(np.hypot(fx - self.primer_center_px[0],
                                                       fy - self.primer_center_px[1]) / R)
            if self.aperture_radius_px:
                geo["aperture_radius_px"] = self.aperture_radius_px * s
                geo["aperture_radius_frac"] = self.aperture_radius_px / R
            if self.drag_detected:
                geo["drag_tip_px"] = [c * s for c in self.drag_tip_px]
                geo["drag_length_px"] = self.drag_length_px * s
                geo["drag_length_frac"] = self.drag_length_px / R
            if self.px_um:
                mm = self.px_um / 1000.0
                geo["primer_diameter_mm"] = 2 * R * s * mm
                if self.fp_core_radius_px:
                    geo["fp_core_diameter_mm"] = 2 * self.fp_core_radius_px * s * mm
                if self.drag_detected:
                    geo["drag_length_mm"] = self.drag_length_px * s * mm
        for k in ("primer_center_px", "primer_radius_px", "fp_core_center_px", "fp_core_radius_px",
                  "aperture_radius_px", "drag_tip_px"):
            d.pop(k, None)
        d["geometry_input_px"] = geo
        px_area = (1.0 / self.work_scale) ** 2
        R2 = (self.primer_radius_px or np.nan) ** 2 * np.pi
        d["areas_input_px"] = {n: float((self.labels == v).sum() * px_area) for v, n in LABEL_NAMES.items()}
        d["areas_frac_of_primer_disc"] = {n: float((self.labels == v).sum() / R2) if R2 == R2 else None
                                          for v, n in LABEL_NAMES.items()}
        d["mode"] = "2d_image"
        return _round(d)


# --------------------------------------------------------------------------- helpers

def _to_gray(img: np.ndarray) -> np.ndarray:
    """Any BGR/BGRA/gray uint8/uint16/float image -> float32 gray in [0, 1]."""
    a = np.asarray(img)
    if a.ndim == 3:
        a = a[..., :3]
        if a.dtype != np.uint8:
            a = cv2.normalize(a.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    a = a.astype(np.float32)
    a[~np.isfinite(a)] = 0
    hi = 255.0 if np.asarray(img).dtype == np.uint8 else max(float(a.max()), 1e-9)
    return np.clip(a / hi, 0, 1)


def _background(g: np.ndarray) -> np.ndarray:
    """Flat black (or flat white) areas connected to the image border: no data."""
    bg = np.zeros(g.shape, bool)
    r = max(1, int(round(0.01 * min(g.shape))))
    st = _disk((2 * r + 1, 2 * r + 1), r, r, r)
    for m in (g <= 1.5 / 255, g >= 254 / 255):
        m = ndi.binary_opening(m, structure=st)       # thin gaps / glints are not background
        lab, n = ndi.label(m)
        if n == 0:
            continue
        border = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
        border = border[border > 0]
        if border.size:
            sizes = ndi.sum(m, lab, border)
            keep = border[sizes > 0.002 * g.size]
            bg |= np.isin(lab, keep)
    return bg


def _peaks(acc: np.ndarray, k: int, min_dist: int) -> list[tuple[float, int, int]]:
    mx = ndi.maximum_filter(acc, size=2 * min_dist + 1)
    ys, xs = np.nonzero((acc == mx) & (acc > 0))
    v = acc[ys, xs]
    o = np.argsort(v)[::-1][:k]
    return [(float(v[i]), int(xs[i]), int(ys[i])) for i in o]


def _hough_unsigned(gx, gy, sel, radii, centre_ok=None, step=2, per_r=3):
    """Gradient-voting circular Hough that votes on *both* sides of each edge.

    Under oblique lighting a circular edge is bright on one side and dark on
    the other, so the gradient points inwards on half of the circle and
    outwards on the rest: unsigned voting handles both.  Returns a list of
    (normalised score, cx, cy, r) candidates in full-resolution pixels.
    """
    h, w = gx.shape
    ey, ex = np.nonzero(sel)
    if ey.size < 20:
        return []
    mag = np.hypot(gx[ey, ex], gy[ey, ex]) + 1e-12
    ux, uy = gx[ey, ex] / mag, gy[ey, ex] / mag
    wt = np.clip(mag / np.percentile(mag, 99), 0, 1)
    ah, aw = h // step + 1, w // step + 1
    out = []
    for r in radii:
        acc = np.zeros(ah * aw)
        for sg in (1.0, -1.0):
            cx = np.round((ex + sg * r * ux) / step).astype(int)
            cy = np.round((ey + sg * r * uy) / step).astype(int)
            ok = (cx >= 0) & (cx < aw) & (cy >= 0) & (cy < ah)
            acc += np.bincount(cy[ok] * aw + cx[ok], weights=wt[ok], minlength=ah * aw)
        acc = ndi.gaussian_filter(acc.reshape(ah, aw), 1.0) / (2 * np.pi * r / step)
        if centre_ok is not None:
            cm = np.zeros((ah, aw))
            sub = centre_ok[::step, ::step]
            cm[:sub.shape[0], :sub.shape[1]] = sub
            acc *= cm
        for v, x, y in _peaks(acc, per_r, max(2, int(0.15 * r / step))):
            out.append((v, float(x * step), float(y * step), float(r)))
    return out


def _ray_support(gx, gy, thr, cx, cy, r, band=0.05, n=180, valid=None):
    """Closed-contour support of a circle: fraction of rays with a radial edge.

    Returns (support, inside_fraction, strength) where strength is the median
    best radial-gradient magnitude on the ray divided by ``thr``.
    """
    h, w = gx.shape
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rs = np.linspace(r * (1 - band), r * (1 + band), 7)
    X = cx + np.outer(np.cos(t), rs)
    Y = cy + np.outer(np.sin(t), rs)
    inside = (X[:, 3] >= 0) & (X[:, 3] < w - 1) & (Y[:, 3] >= 0) & (Y[:, 3] < h - 1)
    if inside.sum() < n * 0.25:
        return 0.0, float(inside.mean()), 0.0
    gxs = _sample(gx, X.ravel(), Y.ravel()).reshape(X.shape)
    gys = _sample(gy, X.ravel(), Y.ravel()).reshape(X.shape)
    radial = np.abs(gxs * np.cos(t)[:, None] + gys * np.sin(t)[:, None]).max(axis=1)
    ok = inside
    if valid is not None:
        vv = _sample(valid.astype(np.float32), X[:, 3], Y[:, 3]) > 0.5
        ok = ok & vv
    if ok.sum() < 10:
        return 0.0, float(inside.mean()), 0.0
    v = radial[ok]
    return float(np.mean(v > thr)), float(inside.mean()), float(np.median(v) / thr)


def _outside_is_background(bg, cx, cy, r) -> bool:
    """True when the band just outside the circle is mostly no-data background."""
    h, w = bg.shape
    t = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    rs = np.linspace(1.03 * r, 1.15 * r, 5)
    X = (cx + np.outer(np.cos(t), rs)).ravel()
    Y = (cy + np.outer(np.sin(t), rs)).ravel()
    ins = (X >= 0) & (X < w) & (Y >= 0) & (Y < h)
    if ins.mean() < 0.2:
        return False
    return bool(bg[Y[ins].astype(int), X[ins].astype(int)].mean() > 0.5)


def _refine_circle(gx, gy, cx, cy, r, lo=0.88, hi=1.12, n=360):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rs = np.arange(lo * r, hi * r, 0.5)
    X = cx + np.outer(np.cos(t), rs)
    Y = cy + np.outer(np.sin(t), rs)
    h, w = gx.shape
    ins = (X >= 2) & (X < w - 3) & (Y >= 2) & (Y < h - 3)
    inside = ins.mean(axis=1) > 0.5
    gxs = _sample(gx, X.ravel(), Y.ravel()).reshape(X.shape)
    gys = _sample(gy, X.ravel(), Y.ravel()).reshape(X.shape)
    radial = np.abs(gxs * np.cos(t)[:, None] + gys * np.sin(t)[:, None]) * ins
    # innermost strong edge on each ray: the primer edge is the inner border of
    # the primer-pocket gap, whose outer border is often just as strong
    rmax = radial.max(axis=1, keepdims=True)
    loc = np.zeros_like(radial, bool)
    loc[:, 1:-1] = (radial[:, 1:-1] >= radial[:, :-2]) & (radial[:, 1:-1] >= radial[:, 2:])
    strong = loc & (radial >= 0.6 * rmax)
    k = np.where(strong.any(axis=1), strong.argmax(axis=1), radial.argmax(axis=1))
    st = radial[np.arange(n), k]
    good = inside & (st > 0.4 * np.median(st[inside])) if inside.any() else inside
    if good.sum() < 20:
        return cx, cy, r, 0.0
    ex, ey = X[np.arange(n), k][good], Y[np.arange(n), k][good]
    fcx, fcy, fr, keep = _robust_circle(ex, ey, min_tol=1.5)
    inl = keep.sum() / max(inside.sum(), 1)
    if not np.isfinite(fr) or not (0.85 * r <= fr <= 1.15 * r) or inl < 0.3:
        return cx, cy, r, float(inl)
    return float(fcx), float(fcy), float(fr), float(inl)


# --------------------------------------------------------------------------- steps

def find_primer_2d(g, bg, p: Params2D, px_um_work, warnings):
    """Return (cx, cy, r, support) of the primer in working pixels, or None."""
    h, w = g.shape
    m = min(h, w)
    if px_um_work:
        rmin, rmax = (r * 1000.0 / px_um_work for r in p.primer_r_mm)
    else:
        rmin, rmax = p.primer_r_frac[0] * m, p.primer_r_frac[1] * m
    rmin, rmax = max(rmin, 8.0), max(min(rmax, 0.75 * max(h, w)), 10.0)
    gs = ndi.gaussian_filter(g, max(1.0, 0.006 * m))
    gy, gx = np.gradient(gs)
    mag = np.hypot(gx, gy)
    fg = ndi.binary_erosion(~bg, iterations=3)
    if fg.sum() < 100:
        return None
    thr = float(np.percentile(mag[fg], 80)) + 1e-6
    sel = fg & (mag > np.percentile(mag[fg], 88))
    radii = np.arange(rmin, rmax, max(1.5, (rmax - rmin) / 60))
    cands = _hough_unsigned(gx, gy, sel, radii, step=2, per_r=2)
    # OpenCV's own detector as a second opinion
    u8 = (gs * 255).astype(np.uint8)
    try:
        hc = cv2.HoughCircles(u8, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=0.05 * m, param1=60,
                              param2=0.5, minRadius=int(rmin), maxRadius=int(rmax))
    except cv2.error:
        hc = None
    if hc is not None:
        cands += [(0.0, float(x), float(y), float(r)) for x, y, r in hc[0][:20]]
    if not cands:
        return None
    # keep the strongest Hough peaks plus all OpenCV ones, then score them
    cands.sort(key=lambda c: -c[0])
    pool = [c for c in cands if c[0] > 0][:40] + [c for c in cands if c[0] == 0]
    diag = 0.5 * np.hypot(h, w)
    scored = []
    for _, cx, cy, r in pool:
        sup, ins, strength = _ray_support(gx, gy, thr, cx, cy, r, valid=~bg)
        if ins < p.min_primer_inside or sup <= 0:
            continue
        prior = 1.0 - 0.35 * np.hypot(cx - w / 2, cy - h / 2) / diag
        score = sup * min(strength, 3.0) * prior
        rim = _outside_is_background(bg, cx, cy, r)
        scored.append(dict(cx=cx, cy=cy, r=r, sup=sup, score=score, rim=rim))
    if not scored:
        return None
    if any(not c["rim"] for c in scored):
        if any(c["rim"] for c in scored):
            best_rim = max((c for c in scored if c["rim"]), key=lambda c: c["score"])
            if best_rim["score"] >= max(c["score"] for c in scored):
                warnings.append("strongest circle is the case-head outline; ignored")
        scored = [c for c in scored if not c["rim"]]
    best = max(scored, key=lambda c: c["score"])
    # prefer an enclosing circle over a nested one (crater inside the primer)
    changed = True
    while changed:
        changed = False
        enc = [c for c in scored if c["r"] > 1.6 * best["r"]
               and np.hypot(c["cx"] - best["cx"], c["cy"] - best["cy"]) < 0.6 * c["r"]
               and c["score"] >= 0.5 * best["score"]]
        if enc:
            best = max(enc, key=lambda c: c["score"])
            changed = True
    cx, cy, r, inl = best["cx"], best["cy"], best["r"], 0.0
    for _ in range(2):
        cx, cy, r, inl = _refine_circle(gx, gy, cx, cy, r)
    sup, _, _ = _ray_support(gx, gy, thr, cx, cy, r, band=0.03, valid=~bg)
    if sup < 0.5:
        warnings.append(f"primer edge supported on only {sup:.0%} of rays")
    return cx, cy, r, sup


def find_impression_2d(g, valid, cx, cy, R, p: Params2D, warnings):
    """Firing-pin impression from shading statistics + pin-circle Hough.

    Returns (fpi mask, (pcx, pcy, rc) or None, source string, circle support).
    """
    shape = g.shape
    face = _disk(shape, cx, cy, 0.92 * R) & valid
    inner = _disk(shape, cx, cy, 0.85 * R) & valid
    if inner.sum() < 200:
        return np.zeros(shape, bool), None, "none", None
    s1 = max(1.0, 0.035 * R)
    low = ndi.gaussian_filter(g, s1)
    med = float(np.median(low[face]))
    dev = ndi.gaussian_filter(np.abs(low - med), s1)
    gy, gx = np.gradient(low)
    lg = ndi.gaussian_filter(np.hypot(gx, gy), s1 / 2)
    devn = dev / (np.median(dev[face]) + 1e-6)
    lgn = lg / (np.median(lg[face]) + 1e-6)
    S = 0.5 * (np.minimum(devn, 20) + np.minimum(lgn, 20))

    # region: strongest, most central connected blob above threshold
    vals = S[inner]
    t_otsu, _ = cv2.threshold(np.clip(vals / 20 * 255, 0, 255).astype(np.uint8), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(p.fpi_min_score, t_otsu / 255 * 20)
    cand = inner & (S > t)
    yy, xx = np.indices(shape)
    central = np.clip(1.0 - np.hypot(xx - cx, yy - cy) / R, 0, 1)
    blob = np.zeros(shape, bool)
    lab, n = ndi.label(cand)
    if n:
        sc = ndi.sum(S * central, lab, np.arange(1, n + 1))
        k = int(np.argmax(sc)) + 1
        if (lab == k).sum() > np.pi * (0.06 * R) ** 2:
            blob = lab == k
            rr = max(1, int(round(0.02 * R)))
            blob = ndi.binary_fill_holes(ndi.binary_closing(blob, iterations=rr))
            blob = ndi.binary_opening(blob, iterations=rr) & inner
            if blob.any():
                lab2, n2 = ndi.label(blob)
                sizes = ndi.sum(blob, lab2, np.arange(1, n2 + 1))
                blob = lab2 == (int(np.argmax(sizes)) + 1)

    # pin-strike circle: second Hough search inside the primer
    gs = ndi.gaussian_filter(g, max(1.0, 0.012 * R))
    hy, hx = np.gradient(gs)
    mag = np.hypot(hx, hy)
    thr = float(np.percentile(mag[inner], 80)) + 1e-6
    sel = inner & (mag > np.percentile(mag[inner], 88))
    rmin, rmax = p.pin_r_frac[0] * R, p.pin_r_frac[1] * R
    radii = np.arange(rmin, rmax, max(1.0, (rmax - rmin) / 40))
    near = _disk(shape, cx, cy, 0.6 * R)
    cands = _hough_unsigned(hx, hy, sel, radii, centre_ok=near.astype(float), step=1, per_r=2)
    circle, csup = None, None
    best = -1.0
    for _, px, py, r in sorted(cands, key=lambda c: -c[0])[:30]:
        sup, ins, strength = _ray_support(hx, hy, thr, px, py, r, band=0.08, valid=valid)
        sc = sup * min(strength, 3.0) * (1 - 0.5 * np.hypot(px - cx, py - cy) / R)
        if sc > best:
            best, circle, csup = sc, (px, py, r), sup

    src = "none"
    core = None
    if blob.any():
        bcx, bcy, brc = fit_core(blob)
        core, src = (bcx, bcy, brc), "blob"
        if circle is not None and csup >= 0.5:
            px, py, r = circle
            inside = blob[int(np.clip(py, 0, shape[0] - 1)), int(np.clip(px, 0, shape[1] - 1))]
            if inside and 0.6 * brc <= r <= 1.5 * brc and np.hypot(px - bcx, py - bcy) < 0.5 * brc:
                core, src = (px, py, r), "blob+circle"
        if src == "blob":
            warnings.append("pin-strike circle not confirmed by a second Hough search")
        # keep the blob near the strike (drag tails may extend ~2.5 x pin radius)
        fpi = (blob & _disk(shape, core[0], core[1], 2.6 * core[2])) | (
            _disk(shape, core[0], core[1], core[2]) & inner)
        fpi = ndi.binary_fill_holes(fpi)
    elif circle is not None and csup >= 0.6:
        core, src = circle, "circle"
        fpi = _disk(shape, *circle) & inner
        warnings.append("impression outline from circle only (no shading anomaly found)")
    else:
        fpi = np.zeros(shape, bool)
        warnings.append("no firing-pin impression found")
    return fpi, core, src, csup


# --------------------------------------------------------------------------- main

def segment_image(img: np.ndarray, px_um: float | None = None, name: str = "image",
                  p: Params2D | None = None) -> Result2D:
    """Mask a single shaded image of a cartridge-case head.

    img    BGR/BGRA/gray array (any dtype).
    px_um  input-image pixel size in micrometres, if known.
    """
    p = p or Params2D()
    g0 = _to_gray(img)
    h0, w0 = g0.shape
    s = min(1.0, p.work_max_px / max(h0, w0, 1))
    g = cv2.resize(g0, (max(1, round(w0 * s)), max(1, round(h0 * s))), interpolation=cv2.INTER_AREA) \
        if s < 1 else g0.copy()
    bg = _background(g)
    valid = ~bg
    px_um_work = px_um / s if px_um else None
    base = cv2.cvtColor((g * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    res = Result2D(name=name, um_per_px=px_um_work or 1.0, work_scale=s, input_shape=(h0, w0),
                   px_um=px_um, valid=valid, image=base, z=None, depth=None)
    res.labels = np.zeros(g.shape, np.uint8)
    res.warnings += [
        "2D image mode: no height data - crater depth, flow-back ring and aperture cannot be measured",
        "aperture shear is not attempted in 2D mode (left empty)",
    ]
    if not px_um:
        res.warnings.append("physical scale unknown: geometry in pixels and fractions of primer radius")
    if valid.mean() < 0.2 or float(np.std(g[valid])) < 1e-3:
        res.status = "failed"
        res.warnings.append("image is blank or almost entirely background")
        return _finish(res, None, p)

    prim = find_primer_2d(g, bg, p, px_um_work, res.warnings)
    if prim is None:
        res.status = "failed"
        res.warnings.append("no primer circle found")
        return _finish(res, None, p)
    cx, cy, R, sup = prim
    res.primer_center_px, res.primer_radius_px = (cx, cy), R
    res.primer_support = res.primer_edge_inlier_frac = sup
    face = _disk(g.shape, cx, cy, R * (1 - p.rim_margin_frac)) & valid

    fpi, core, src, csup = find_impression_2d(g, valid, cx, cy, R, p, res.warnings)
    res.fpi_source, res.pin_circle_support = src, csup
    tail = np.zeros_like(fpi)
    zone = np.zeros_like(fpi)
    if core is not None and fpi.any():
        fcx, fcy, rc = core
        res.fp_core_center_px, res.fp_core_radius_px = (fcx, fcy), rc
        res.aperture_radius_px = p.aperture_ratio * rc
        res.warnings.append(f"aperture taken as {p.aperture_ratio} x firing-pin radius (no height data)")
        px_mm = (px_um_work or 1000.0) / 1000.0
        drag = find_drag(fpi, fcx, fcy, rc, face, px_mm, Params())
        if drag is not None:
            tail = drag["mask"] & fpi
            res.drag_detected = True
            res.drag_tip_px, res.drag_angle_deg = drag["tip"], drag["angle"]
            res.drag_length_px = float(np.hypot(drag["tip"][0] - fcx, drag["tip"][1] - fcy))
            res.warnings.append("drag tail labelled firing-pin shear from outline elongation only")
        else:
            res.warnings.append("no firing-pin drag detected; firing-pin shear left empty")
        margin = max(1, int(round(p.fpi_margin_frac * rc)))
        zone = _disk(g.shape, fcx, fcy, res.aperture_radius_px) | ndi.binary_dilation(fpi, iterations=margin)
    else:
        res.status = "partial"
    bf = face & ~zone
    bf = ndi.binary_opening(bf, iterations=1)
    res.labels[bf] = BREECH_FACE
    res.labels[fpi & face] = FIRING_PIN_IMPRESSION
    res.labels[tail & face] = FIRING_PIN_SHEAR
    return _finish(res, fpi, p)


def _finish(res: Result2D, fpi, p: Params2D) -> Result2D:
    f = {"two_d_only": 0.7,
         "primer_edge": float(np.clip(((res.primer_support or 0) - 0.3) / 0.5, 0, 1)),
         "firing_pin": {"blob+circle": 1.0, "blob": 0.8, "circle": 0.6}.get(res.fpi_source, 0.3)}
    res.qc_factors = {k: round(v, 3) for k, v in f.items()}
    res.qc_score = float(np.prod(list(f.values())))
    res.needs_review = True        # 2D masks are always a suggestion for an examiner
    res.review_reasons = ["two_d_only"]
    if res.px_um:
        a = (res.px_um / res.work_scale / 1000.0) ** 2
        res.areas_mm2 = {n: float((res.labels == v).sum() * a) for v, n in LABEL_NAMES.items()}
    return res


# --------------------------------------------------------------------------- figures

def side_by_side_2d(res: Result2D, size: int = 520) -> np.ndarray:
    """[input image | primer close-up | masks | label image] + legend, as in 3D mode."""
    base = res.image
    ov = overlay(res, base)
    y0, y1, x0, x1 = crop_box(res)
    scale = size / max(y1 - y0, x1 - x0, 1)
    full = base.copy()
    if res.primer_radius_px is not None:
        cv2.rectangle(full, (x0, y0), (x1, y1), (0, 200, 255), max(2, base.shape[0] // 250))
    panels = [("Input image (2D, no height)", _fit(full, size))]
    for title, img in [("Primer region - original", base), ("Generated masks (2D mode)", ov),
                       ("Label mask", flat_mask(res))]:
        crop = img[y0:y1, x0:x1].copy()
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
                          interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        if res.px_um:
            _scale_bar(crop, res.px_um / res.work_scale / scale)
        panels.append((title, _fit(crop, size)))
    head = 34
    tiles = []
    for title, img in panels:
        t = np.full((head + size, size, 3), 255, np.uint8)
        t[head:] = img
        cv2.putText(t, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 1, cv2.LINE_AA)
        tiles.append(t)
    row = np.hstack([np.pad(t, ((0, 0), (4, 4), (0, 0)), constant_values=255) for t in tiles])
    foot = np.full((90, row.shape[1], 3), 255, np.uint8)
    x = 10
    for v, text in LEGEND:
        if v is None:
            cv2.arrowedLine(foot, (x, 22), (x + 26, 22), ARROW, 3, tipLength=0.4)
        else:
            cv2.rectangle(foot, (x, 12), (x + 24, 32), COLOURS[v], -1)
        cv2.putText(foot, text, (x + 32, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        x += 40 + 8 * len(text) + 20
    info = [res.name, f"status={res.status}", f"qc={res.qc_score:.2f} (review)"]
    if res.primer_radius_px:
        info.append(f"primer r={res.primer_radius_px / res.work_scale:.0f} px (support {res.primer_support:.0%})")
    if res.fp_core_radius_px:
        info.append(f"pin r={res.fp_core_radius_px / res.primer_radius_px:.2f} x primer r [{res.fpi_source}]")
    if res.drag_detected:
        info.append(f"drag {res.drag_length_px / res.primer_radius_px:.2f} x primer r @ {res.drag_angle_deg:.0f} deg")
    cv2.putText(foot, "   ".join(info), (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(foot, "2D mode: shading only, no height data. Aperture shear not attempted. Always needs review.",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 160), 1, cv2.LINE_AA)
    return np.vstack([row, foot])


def _imwrite(path: Path, img) -> None:
    ok, buf = cv2.imencode(path.suffix, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise IOError(f"could not encode {path.name}")
    buf.tofile(str(path))


def _imread(path: Path) -> np.ndarray:
    a = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise IOError(f"could not read {path}")
    return a


def process_image(path: Path, out_dir: Path, px_um: float | None = None) -> dict:
    res = segment_image(_imread(path), px_um=px_um, name=path.stem)
    d = out_dir / res.name
    d.mkdir(parents=True, exist_ok=True)
    _imwrite(d / "overlay.png", overlay(res, res.image))
    _imwrite(d / "labels.png", res.labels_at(res.input_shape))
    _imwrite(out_dir / f"{res.name}__side_by_side_2d.png", side_by_side_2d(res))
    s = res.summary()
    s["source"] = str(path)
    s["label_values"] = {str(k): v for k, v in LABEL_NAMES.items()}
    (d / "masks.json").write_text(json.dumps(s, indent=2))
    return s


# --------------------------------------------------------------------------- evaluation

def iou(a: np.ndarray, b: np.ndarray):
    u = np.logical_or(a, b).sum()
    return None if u == 0 else float(np.logical_and(a, b).sum() / u)


def reference_regions(ref_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Colour-threshold the brief's annotation colours (semi-transparent overlays)."""
    hsv = cv2.cvtColor(ref_bgr[..., :3], cv2.COLOR_BGR2HSV)
    H, S, V = (hsv[..., i].astype(int) for i in range(3))
    red = ((H < 8) | (H >= 170)) & (S > 120) & (V > 60)
    green = (H >= 40) & (H < 80) & (S > 45)
    cyan = (H >= 80) & (H < 100) & (S > 45)
    blue = (H >= 110) & (H < 128) & (S > 150) & (V > 120)       # drag arrow / strike circle
    purple = (H >= 128) & (H < 170) & (S > 35) & (V > 25)
    def clean(mask):
        return ndi.binary_opening(mask, iterations=1)
    red, green, cyan, purple = clean(red), clean(green), clean(cyan), clean(purple)
    fp = ndi.binary_fill_holes(ndi.binary_closing(cyan | purple | blue, iterations=3))
    lab, n = ndi.label(fp)
    if n:  # the firing-pin region is one blob; drop speckle
        sizes = ndi.sum(fp, lab, np.arange(1, n + 1))
        fp = lab == (int(np.argmax(sizes)) + 1)
    red = ndi.binary_closing(red, iterations=2) & ~fp
    return dict(breech_face=red, aperture_shear=green, cyan=cyan, purple=purple & fp,
                firing_pin_region=fp)


def _similarity_sift(src_g8, dst_g8):
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.015)
    k1, d1 = sift.detectAndCompute(src_g8, None)
    k2, d2 = sift.detectAndCompute(dst_g8, None)
    if d1 is None or d2 is None:
        return None, 0, 0
    bf = cv2.BFMatcher(cv2.NORM_L2)
    good = []
    for m in bf.knnMatch(d1, d2, k=2):
        if len(m) == 2 and m[0].distance < 0.8 * m[1].distance:
            good.append(m[0])
    if len(good) < 8:
        return None, 0, len(good)
    a = np.float32([k1[m.queryIdx].pt for m in good])
    b = np.float32([k2[m.trainIdx].pt for m in good])
    M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=3.0,
                                         maxIters=20000, confidence=0.999)
    return M, int(inl.sum()) if inl is not None else 0, len(good)


def evaluate_brief(folder: Path) -> dict:
    folder = Path(folder)
    out: dict = {"note": "IoU of 2D-mode masks against the brief's own annotated figures. "
                         "Reference regions are colour-thresholded from semi-transparent overlays."}
    # --- page 1: before -> after (rotated / rescaled) ---------------------------------
    before = _imread(folder / "brief_p1_before.png")[..., :3]
    after = _imread(folder / "brief_p1_after_reference.png")[..., :3]
    res = segment_image(before, name="brief_p1_before")
    _imwrite(folder / "brief_p1_before__side_by_side_2d.png", side_by_side_2d(res))
    lab = res.labels_at(before.shape[:2])
    g1 = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    M, ninl, ngood = _similarity_sift(g1, g2)
    reg = dict(method="SIFT + RANSAC similarity", matches=ngood, inliers=ninl)
    if M is not None:
        sc = float(np.hypot(M[0, 0], M[1, 0]))
        reg.update(scale=sc, rotation_deg=float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))),
                   tx=float(M[0, 2]), ty=float(M[1, 2]))
    registered = M is not None and ninl >= 15 and 0.2 <= sc <= 3.0
    if not registered:
        reg["warning"] = "registration did not meet support or scale checks; IoU omitted"
    ref = reference_regions(after)
    p1 = {"segmentation": res.summary(), "registration": reg}
    if registered:
        H, W = after.shape[:2]
        warped = cv2.warpAffine(lab, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
        footprint = cv2.warpAffine(np.ones(lab.shape, np.uint8), M, (W, H), flags=cv2.INTER_NEAREST) > 0
        ours_fp = (warped == FIRING_PIN_IMPRESSION) | (warped == FIRING_PIN_SHEAR)
        p1["iou"] = {
            "breech_face (red)": iou(warped == BREECH_FACE, ref["breech_face"] & footprint),
            "firing_pin_region (cyan+purple vs ours FPI+drag)": iou(ours_fp, ref["firing_pin_region"] & footprint),
            "drag_channel (cyan vs ours firing-pin shear)": iou(warped == FIRING_PIN_SHEAR, ref["cyan"] & footprint),
            "aperture_shear (green; ours never labelled in 2D)": iou(warped == APERTURE_SHEAR,
                                                                    ref["aperture_shear"] & footprint),
        }
        p1["reference_area_px"] = {k: int(v.sum()) for k, v in ref.items()}
        p1["ours_area_px_in_reference_frame"] = {n: int((warped == v).sum()) for v, n in LABEL_NAMES.items()}
        # visual check: our outlines drawn on the reference
        vis = after.copy()
        for v, c in COLOURS.items():
            cnts, _ = cv2.findContours((warped == v).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, cnts, -1, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.drawContours(vis, cnts, -1, c, 2, cv2.LINE_AA)
        # The brief's page-1 reference is stored upside down (its caption reads
        # inverted); show it upright.  IoU above is computed before this rotation.
        vis = cv2.rotate(vis, cv2.ROTATE_180)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 34), (255, 255, 255), -1)
        cv2.putText(vis, "brief reference (fill, shown upright) vs 2D mode (outlines)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 2, cv2.LINE_AA)
        _imwrite(folder / "brief_p1_compare.png", np.hstack([cv2.resize(overlay(res, res.image), None,
                 fx=H / res.image.shape[0], fy=H / res.image.shape[0]), vis]))
    out["page1_evofinder"] = p1

    # --- page 2: Glock image that already carries the annotations -------------------------
    p2img = _imread(folder / "brief_p2_reference.png")[..., :3]
    # crude overlay removal: max channel keeps the shading under a single-hue tint
    deov = p2img.max(axis=2)
    res2 = segment_image(deov, name="brief_p2_deoverlaid")
    _imwrite(folder / "brief_p2__side_by_side_2d.png", side_by_side_2d(res2))
    out["page2_glock"] = {
        "note": "The brief supplies only the annotated figure. A maximum-channel image is "
                "shown for qualitative review. The source includes the label overlays, so "
                "there is no independent IoU measurement for this figure.",
        "segmentation": res2.summary(),
    }
    return out


def evaluate_renders(masks: Path | list, out_dir: Path, x3p_dirs=()) -> dict:
    """Render each scan of a TypeScript CLI output, run the 2D mode on the render, IoU vs the 3D labels."""
    inputs = masks if isinstance(masks, (list, tuple)) else [masks]
    rows, sheet = [], []
    classes = {"breech_face": [BREECH_FACE], "firing_pin_impression": [FIRING_PIN_IMPRESSION],
               "firing_pin_shear": [FIRING_PIN_SHEAR], "aperture_shear": [APERTURE_SHEAR],
               "firing_pin_region": [FIRING_PIN_IMPRESSION, FIRING_PIN_SHEAR]}
    for r3 in load_scans(inputs, x3p_dirs):
        row = {"scan": r3.name, "status_3d": r3.status}
        if r3.status == "skipped" or r3.primer_radius_px is None:
            row["note"] = "skipped by 3D pipeline"
            rows.append(row)
            continue
        img = render(r3)
        for tag, pxu in (("scale_unknown", None), ("scale_known", r3.um_per_px)):
            r2 = segment_image(img, px_um=pxu, name=r3.name)
            lab2 = r2.labels_at(img.shape[:2])
            e = {"status_2d": r2.status, "fpi_source": r2.fpi_source}
            if r2.primer_radius_px:
                c2 = np.array(r2.primer_center_px) / r2.work_scale
                e["primer_centre_err_frac"] = float(np.hypot(*(c2 - np.array(r3.primer_center_px)))
                                                    / r3.primer_radius_px)
                e["primer_radius_ratio"] = float(r2.primer_radius_px / r2.work_scale / r3.primer_radius_px)
                d2 = _disk(img.shape[:2], c2[0], c2[1], r2.primer_radius_px / r2.work_scale)
                e["iou_primer_disc"] = iou(d2, _disk(img.shape[:2], *r3.primer_center_px, r3.primer_radius_px))
            for k, vs in classes.items():
                e[f"iou_{k}"] = iou(np.isin(lab2, vs), np.isin(r3.labels, vs))
            e["drag_3d"], e["drag_2d"] = r3.drag_detected, r2.drag_detected
            row[tag] = e
            if tag == "scale_unknown":
                y0, y1, x0, x1 = crop_box(r3)
                o3 = overlay(r3, img)[y0:y1, x0:x1]
                o2 = overlay(r2, r2.image)
                o2 = cv2.resize(o2, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)[y0:y1, x0:x1]
                sheet.append((f"{r3.name[:34]}  3D | 2D", o3, o2))
        rows.append(row)
    def mean(key, tag):
        v = [r[tag][key] for r in rows if tag in r and r[tag].get(key) is not None]
        return (float(np.mean(v)), len(v)) if v else (None, 0)
    keys = ["iou_primer_disc", "iou_breech_face", "iou_firing_pin_impression", "iou_firing_pin_region",
            "iou_firing_pin_shear", "iou_aperture_shear", "primer_centre_err_frac"]
    summary = {tag: {k: dict(zip(("mean", "n"), mean(k, tag))) for k in keys}
               for tag in ("scale_unknown", "scale_known")}
    if sheet:
        _imwrite(out_dir / "render_vs_3d_contact.jpg", contact_sheet(sheet, tile=260, cols=3))
    return _round({"note": "2D mode run on the 3D pipeline's own hillshade render; IoU against the 3D "
                           "labels on the same pixel grid. IoU is None when both masks are empty.",
                   "summary_mean": summary, "per_scan": rows})


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python python/image2d.py",
                                 description="Best-effort masking of a 2D shaded image of a cartridge-case head.")
    ap.add_argument("inputs", nargs="*", help="image files (png/jpg/tif) or folders")
    ap.add_argument("-o", "--out", default="outputs/2d", help="output directory")
    ap.add_argument("--px-um", type=float, default=None, help="input pixel size in micrometres, if known")
    ap.add_argument("--eval-brief", metavar="DIR", help="evaluate on the brief figures in DIR")
    ap.add_argument("--eval-renders", metavar="MASKDIR",
                    help="2D mode on renders of the scans in a TypeScript CLI output vs their 3D labels")
    ap.add_argument("--x3p-dir", action="append", default=[],
                    help="where to look for <scan>.x3p when masks.json['source'] does not resolve")
    ap.add_argument("--brief-colours", action="store_true",
                    help="round strike purple, drag channel light blue, as in the brief's figures (display only)")
    a = ap.parse_args(argv)
    set_brief_colours(a.brief_colours)
    if not (a.inputs or a.eval_brief or a.eval_renders):
        ap.error("provide images, --eval-brief, or --eval-renders")
    if a.px_um is not None and (not np.isfinite(a.px_um) or a.px_um <= 0):
        ap.error("--px-um must be positive and finite")
    if a.eval_brief:
        r = evaluate_brief(Path(a.eval_brief))
        (Path(a.eval_brief) / "brief_eval.json").write_text(json.dumps(r, indent=2))
        print(json.dumps({k: v.get("iou") for k, v in r.items() if isinstance(v, dict)}, indent=2))
    if a.eval_renders:
        od = Path(a.out)
        od.mkdir(parents=True, exist_ok=True)
        r = evaluate_renders(Path(a.eval_renders), od, a.x3p_dir)
        (od / "render_vs_3d.json").write_text(json.dumps(r, indent=2))
        print(json.dumps(r["summary_mean"], indent=2))
    out = Path(a.out)
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    files = []
    for s in a.inputs:
        q = Path(s)
        files += sorted(f for f in q.iterdir() if f.suffix.lower() in exts) if q.is_dir() else [q]
    for f in files:
        out.mkdir(parents=True, exist_ok=True)
        s = process_image(f, out, a.px_um)
        print(f"{f.name}: status={s['status']} qc={s['qc_score']} fpi={s['fpi_source']} "
              f"drag={s['drag_detected']}")


if __name__ == "__main__":
    main()
