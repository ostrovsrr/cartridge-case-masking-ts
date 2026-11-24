"""Helpers shared by the research-evidence scripts, copied from the Python ``cartmask`` package.

Everything here is a verbatim (or field-for-field) copy of the reference
implementation, so that the evidence scripts compute exactly what they did in
the Python repository.  Nothing here segments a scan: the masks come from the
TypeScript CLI (see :mod:`ts_masks`).

Sources, in ``cartmask``:

* label values and :class:`Result` - ``segment.py``
* :func:`disk`, :func:`fit_circle`, :func:`robust_circle`, :func:`sample` - ``steps/geometry.py``
* :func:`fit_core` - ``steps/firing_pin.py``; :func:`find_drag` - ``steps/drag.py``
* :func:`robust_plane` - ``steps/datum.py``
* :func:`resample`, :func:`fill_nearest`, :func:`hillshade` - ``preprocess.py``
* colours, :func:`render`, :func:`overlay`, :func:`flat_mask`, :func:`crop_box`,
  :func:`contact_sheet` - ``visualize.py``
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np
from scipy import ndimage as ndi

from x3p_reader import Surface

# --------------------------------------------------------------------------- labels

BREECH_FACE, APERTURE_SHEAR, FIRING_PIN_SHEAR, FIRING_PIN_IMPRESSION = 1, 2, 3, 4
LABEL_NAMES = {
    BREECH_FACE: "breech_face_impression",
    APERTURE_SHEAR: "aperture_shear",
    FIRING_PIN_SHEAR: "firing_pin_shear",
    FIRING_PIN_IMPRESSION: "firing_pin_impression",
}


@dataclass
class Params:
    """The subset of ``cartmask.segment.Params`` the evidence scripts use (same defaults)."""

    work_um_per_px: float = 8.0
    rim_margin_frac: float = 0.05
    drag_min_reach: float = 1.15
    drag_min_area_frac: float = 0.01
    drag_min_coherence: float = 0.75


@dataclass
class Result:
    """Field-for-field copy of ``cartmask.segment.Result`` (same order, so ``asdict`` matches).

    For a 3D scan it is filled from the TypeScript CLI's ``masks.json`` and
    ``labels.png`` plus the height map re-read from the X3P (see :mod:`ts_masks`).
    """

    name: str
    um_per_px: float
    status: str = "ok"
    warnings: list[str] = field(default_factory=list)
    primer_center_px: tuple[float, float] | None = None
    primer_radius_px: float | None = None
    primer_edge_inlier_frac: float | None = None
    fp_core_center_px: tuple[float, float] | None = None
    fp_core_radius_px: float | None = None
    fp_max_depth_um: float | None = None
    roughness_um: float | None = None
    aperture_radius_px: float | None = None
    aperture_detected: bool = False
    drag_detected: bool = False
    drag_tip_px: tuple[float, float] | None = None
    drag_angle_deg: float | None = None
    drag_length_mm: float | None = None
    areas_mm2: dict[str, float] = field(default_factory=dict)
    qc_score: float = 0.0
    needs_review: bool = True
    qc_factors: dict[str, float] = field(default_factory=dict)
    z: np.ndarray | None = None
    valid: np.ndarray | None = None
    depth: np.ndarray | None = None
    labels: np.ndarray | None = None
    qc_observations: dict[str, float] = field(default_factory=dict)
    review_reasons: list[str] = field(default_factory=list)
    aperture_shape: str = "circle"
    aperture_evidence: dict[str, Any] = field(default_factory=dict)
    shear_evidence: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("z", "valid", "depth", "labels"):
            d.pop(k)
        return _round(d)


def _round(o: Any) -> Any:
    """Recursively round floats to 4 decimals and turn numpy scalars into Python ones."""
    if isinstance(o, float):
        return round(o, 4)
    if isinstance(o, (list, tuple)):
        return [_round(v) for v in o]
    if isinstance(o, dict):
        return {k: _round(v) for k, v in o.items()}
    if isinstance(o, (np.floating, np.integer)):
        return _round(o.item())
    return o


# --------------------------------------------------------------------------- geometry

def disk(shape: tuple[int, ...], cx: float, cy: float, r: float) -> np.ndarray:
    """Boolean disk mask (boundary included)."""
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def fit_circle(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Algebraic (Kasa) least-squares circle fit."""
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x ** 2 + y ** 2
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    return cx, cy, float(np.sqrt(max(c + cx ** 2 + cy ** 2, 1e-12)))


def robust_circle(x: np.ndarray, y: np.ndarray, iters: int = 6, k: float = 2.5,
                  min_tol: float = 1.0) -> tuple[float, float, float, np.ndarray]:
    """Circle fit with iterative MAD-based outlier rejection."""
    keep = np.ones(len(x), bool)
    cx = cy = r = np.nan
    for _ in range(iters):
        if keep.sum() < 8:
            break
        cx, cy, r = fit_circle(x[keep], y[keep])
        res = np.abs(np.hypot(x - cx, y - cy) - r)
        mad = np.median(res[keep]) * 1.4826
        new = res <= max(k * mad, min_tol)
        if (new == keep).all():
            break
        keep = new
    return cx, cy, r, keep


def sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of ``img`` at sub-pixel points (nearest at the border)."""
    return ndi.map_coordinates(img, [ys, xs], order=1, mode="nearest")


def fit_core(fpi: np.ndarray) -> tuple[float, float, float]:
    """Circle of the pin strike: robust fit to the round part of the impression outline."""
    dt = ndi.distance_transform_edt(fpi)
    iy, ix = np.unravel_index(int(dt.argmax()), dt.shape)
    cx, cy, r = float(ix), float(iy), float(dt.max())
    contours, _ = cv2.findContours(fpi.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = max(contours, key=len)[:, 0, :].astype(float)
    for _ in range(3):  # re-centre on the outline points that belong to the round part
        rad = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        near = rad <= np.percentile(rad, 50) * 1.15
        if near.sum() < 10:
            break
        cx, cy, r, _ = robust_circle(pts[near, 0], pts[near, 1], min_tol=1.5)
    return cx, cy, r


def find_drag(fpi: np.ndarray, cx: float, cy: float, rc: float, face: np.ndarray, px_mm: float,
              p: Params) -> dict | None:
    """Firing-pin drag: the part of the impression outside the strike circle."""
    yy, xx = np.indices(fpi.shape)
    outside = np.hypot(xx - cx, yy - cy) > rc * 1.1
    tail = ndi.binary_opening(fpi & outside, iterations=2)
    if not tail.any():
        return None
    lab, n = ndi.label(tail)
    sizes = ndi.sum(tail, lab, range(1, n + 1))
    tail = lab == (int(np.argmax(sizes)) + 1)
    ty, tx = np.nonzero(tail)
    vx, vy = tx - cx, ty - cy
    dist = np.hypot(vx, vy)
    if dist.max() < p.drag_min_reach * rc or tail.sum() < p.drag_min_area_frac * np.pi * rc * rc:
        return None
    w = np.clip(dist - rc, 0, None)
    ux, uy = np.sum(vx / dist * w), np.sum(vy / dist * w)
    coherence = np.hypot(ux, uy) / max(w.sum(), 1e-9)   # 1 = one direction, 0 = ring
    if coherence < p.drag_min_coherence:
        return None
    ux, uy = ux / np.hypot(ux, uy), uy / np.hypot(ux, uy)
    length = float(np.max(vx * ux + vy * uy))
    return dict(mask=tail & face, tip=(cx + ux * length, cy + uy * length),
                angle=float(np.degrees(np.arctan2(uy, ux))), length_mm=length * px_mm,
                coherence=float(coherence))


# --------------------------------------------------------------------------- height maps

def robust_plane(z: np.ndarray, mask: np.ndarray, iters: int = 5) -> tuple[np.ndarray, float]:
    """Plane ``z = a*x + b*y + c`` fitted with iterative 3-sigma (MAD) rejection."""
    yy, xx = np.nonzero(mask)
    zz = z[yy, xx]
    keep = np.ones(zz.size, bool)
    coef = np.zeros(3)
    for _ in range(iters):
        A = np.c_[xx[keep], yy[keep], np.ones(keep.sum())]
        coef, *_ = np.linalg.lstsq(A, zz[keep], rcond=None)
        res = zz - (coef[0] * xx + coef[1] * yy + coef[2])
        s = 1.4826 * np.median(np.abs(res[keep]))
        keep = np.abs(res) < 3 * max(s, 1e-9)
    res = zz - (coef[0] * xx + coef[1] * yy + coef[2])
    sigma = 1.4826 * np.median(np.abs(res[keep]))
    return coef, sigma


def resample(surf: Surface, um_per_px: float) -> Surface:
    """Area-average the height map to a coarser isotropic grid (NaN-aware)."""
    sx = surf.dx * 1e6 / um_per_px
    sy = surf.dy * 1e6 / um_per_px
    if sx >= 1.0 and sy >= 1.0 and abs(surf.dx - surf.dy) <= 1e-9 * surf.dx:
        return surf
    if sx > 1.0 or sy > 1.0:
        um_per_px = max(surf.dx, surf.dy) * 1e6
        sx, sy = surf.dx * 1e6 / um_per_px, surf.dy * 1e6 / um_per_px
    h, w = surf.z.shape
    size = (max(1, round(w * sx)), max(1, round(h * sy)))
    valid = np.isfinite(surf.z)
    z0 = np.where(valid, surf.z, 0.0)
    num = cv2.resize(z0, size, interpolation=cv2.INTER_AREA)
    den = cv2.resize(valid.astype(np.float64), size, interpolation=cv2.INTER_AREA)
    z = np.where(den >= 0.5, num / np.maximum(den, 1e-9), np.nan)
    return Surface(z=z, dx=um_per_px * 1e-6, dy=um_per_px * 1e-6, name=surf.name)


def fill_nearest(z: np.ndarray) -> np.ndarray:
    """Replace NaNs by the nearest valid value (for filtering/gradients only)."""
    invalid = ~np.isfinite(z)
    if not invalid.any():
        return z.copy()
    idx = ndi.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return z[tuple(idx)]


def hillshade(z: np.ndarray, px_m: float, valid: np.ndarray | None = None,
              azimuth_deg: float = 315.0, altitude_deg: float = 35.0,
              exaggeration: float = 3.0) -> np.ndarray:
    """Render a height map (metres) as an 8-bit shaded relief image."""
    if valid is None:
        valid = np.isfinite(z)
    zf = fill_nearest(z) if not valid.all() else z
    zf = ndi.gaussian_filter(zf, 0.7)
    gy, gx = np.gradient(zf * exaggeration, px_m)
    az, alt = np.deg2rad(azimuth_deg), np.deg2rad(altitude_deg)
    lx, ly, lz = np.cos(alt) * np.sin(az), -np.cos(alt) * np.cos(az), np.sin(alt)
    shade = (-gx * lx - gy * ly + lz) / np.sqrt(1.0 + gx ** 2 + gy ** 2)
    v = shade[valid] if valid.any() else shade.ravel()
    lo, hi = np.percentile(v, [0.5, 99.5])
    img = np.clip((shade - lo) / max(hi - lo, 1e-9), 0, 1)
    img = (img * 235 + 10).astype(np.uint8)
    img[~valid] = 0
    return img


# --------------------------------------------------------------------------- figures

PURPLE = (200, 70, 150)
LIGHT_BLUE = (230, 220, 90)
COLOURS = {
    BREECH_FACE: (40, 40, 220),            # red
    APERTURE_SHEAR: (60, 200, 60),         # green
    FIRING_PIN_SHEAR: PURPLE,
    FIRING_PIN_IMPRESSION: LIGHT_BLUE,
}
ARROW = (255, 60, 20)                      # blue
LEGEND = [
    (BREECH_FACE, "1 Breech-face impression"),
    (APERTURE_SHEAR, "2 Aperture shear"),
    (FIRING_PIN_SHEAR, "3 Firing-pin shear (drag channel)"),
    (FIRING_PIN_IMPRESSION, "4 Firing-pin impression"),
    (None, "5 Firing-pin drag (arrow)"),
]


def set_brief_colours(enabled: bool = True) -> None:
    """Switch the display colours of labels 3 and 4 to the brief's figures (or back)."""
    COLOURS[FIRING_PIN_SHEAR], COLOURS[FIRING_PIN_IMPRESSION] = (
        (LIGHT_BLUE, PURPLE) if enabled else (PURPLE, LIGHT_BLUE))


def _labels(res: Result) -> np.ndarray:
    if res.labels is None:
        raise ValueError(f"{res.name}: result carries no label image")
    return res.labels


def render(res: Result) -> np.ndarray:
    """Shaded-relief rendering of the (working-resolution) height map, BGR."""
    if res.z is None or res.valid is None:
        raise ValueError(f"{res.name}: result carries no height map")
    g = hillshade(res.z, res.um_per_px * 1e-6, res.valid)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def overlay(res: Result, base: np.ndarray | None = None, alpha: float = 0.55) -> np.ndarray:
    """Masks blended over the rendering, with outlines, strike centre and drag arrow."""
    labels = _labels(res)
    base = render(res) if base is None else base
    out = base.copy()
    col = np.zeros_like(base)
    for v, c in COLOURS.items():
        col[labels == v] = c
    m = labels > 0
    out[m] = (base[m] * (1 - alpha) + col[m] * alpha).astype(np.uint8)
    for v, c in COLOURS.items():
        cnts, _ = cv2.findContours((labels == v).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, c, 1, cv2.LINE_AA)
    lw = max(2, base.shape[0] // 300)
    if res.fp_core_center_px is not None and res.fp_core_radius_px is not None:
        cx, cy = res.fp_core_center_px
        rr = max(4, int(round(0.25 * res.fp_core_radius_px)))
        cv2.circle(out, (int(round(cx)), int(round(cy))), rr, ARROW, lw, cv2.LINE_AA)
        if res.drag_detected and res.drag_tip_px is not None:
            tx, ty = res.drag_tip_px
            L = np.hypot(tx - cx, ty - cy)
            cv2.arrowedLine(out, (int(cx), int(cy)), (int(tx), int(ty)), ARROW, lw, cv2.LINE_AA,
                            tipLength=min(0.35, 18 / max(L, 1)))
    return out


def flat_mask(res: Result) -> np.ndarray:
    """Label image in legend colours on white (invalid data light grey)."""
    labels = _labels(res)
    out = np.full((*labels.shape, 3), 255, np.uint8)
    if res.valid is not None:
        out[~res.valid] = (235, 235, 235)
    for v, c in COLOURS.items():
        out[labels == v] = c
    return out


def _scale_bar(img: np.ndarray, um_per_px: float, mm: float = 1.0) -> None:
    """Draw a ``mm``-long scale bar in the lower-left corner, in place."""
    L = int(round(mm * 1000 / um_per_px))
    h, _ = img.shape[:2]
    x0, y0 = 15, h - 20
    cv2.line(img, (x0, y0), (x0 + L, y0), (255, 255, 255), 5)
    cv2.line(img, (x0, y0), (x0 + L, y0), (0, 0, 0), 2)
    cv2.putText(img, f"{mm:g} mm", (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, f"{mm:g} mm", (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def _fit(img: np.ndarray, size: int) -> np.ndarray:
    """Scale ``img`` to fit a ``size`` x ``size`` white canvas, centred."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 255, np.uint8)
    y0, x0 = (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
    canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    return canvas


def crop_box(res: Result, pad: float = 1.25) -> tuple[int, int, int, int]:
    """Square window ``(y0, y1, x0, x1)`` around the primer, clipped to the image."""
    labels = _labels(res)
    h, w = labels.shape
    if res.primer_radius_px is None or res.primer_center_px is None:
        return 0, h, 0, w
    cx, cy = res.primer_center_px
    r = res.primer_radius_px * pad
    return (max(0, int(cy - r)), min(h, int(cy + r)), max(0, int(cx - r)), min(w, int(cx + r)))


def contact_sheet(items: list[tuple[str, np.ndarray, np.ndarray]], tile: int = 300, cols: int = 4) -> np.ndarray:
    """Grid of before/after pairs."""
    pairs = []
    for title, a, b in items:
        pa, pb = _fit(a, tile), _fit(b, tile)
        p = np.hstack([pa, np.full((tile, 4, 3), 255, np.uint8), pb])
        t = np.full((26, p.shape[1], 3), 255, np.uint8)
        cv2.putText(t, title[:60], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
        pairs.append(np.pad(np.vstack([t, p]), ((6, 6), (6, 6), (0, 0)), constant_values=255))
    while len(pairs) % cols:
        pairs.append(np.full_like(pairs[0], 255))
    rows = [np.hstack(pairs[i:i + cols]) for i in range(0, len(pairs), cols)]
    return np.vstack(rows)
