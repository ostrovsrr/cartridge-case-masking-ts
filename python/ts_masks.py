"""Read the TypeScript CLI's per-scan outputs back as a cartmask-style :class:`Result`.

``npx tsx src/cli.ts <scans> -o <out>`` writes, for every scan, ``<out>/<scan>/``
with ``labels.png`` (uint8 label image on the working grid, 0 = background,
1..4 as in ``masks.json["label_values"]``) and ``masks.json`` (the scalar
fields of the result, floats rounded to 4 decimals).  The evidence scripts
also need the height map, which the CLI does not write.  It is rebuilt here
from the X3P, exactly as ``cartmask.segment.segment`` builds it:

=====================  ==================================================================
field                  source
=====================  ==================================================================
``labels``             ``labels.png`` (the TypeScript masks: this is what is evaluated)
``z``, ``valid``       X3P re-read (:mod:`x3p_reader`) and area-resampled to the working
                       grid (:func:`cartmask_shared.resample`, the function the TS port
                       reproduces pixel for pixel); negated when ``masks.json`` reports
                       the polarity flip (warning ``height polarity inverted``)
``um_per_px``          the resampled grid's exact spacing (``masks.json`` holds it
                       rounded; it is used as a consistency check)
``depth``              ``z - plane``: the datum plane is re-fitted with
                       :func:`cartmask_shared.robust_plane` on the annulus that
                       ``cartmask.steps.datum.fit_datum`` uses (primer face minus the
                       inner 0.45 primer radii), with the primer circle from
                       ``masks.json``.  The plane coefficients are not in
                       ``masks.json``; refitting reproduces them because the inputs
                       (heights, valid mask, primer circle) are the same.
primer centre/radius   ``masks.json`` (``primer_center_px``, ``primer_radius_px``)
everything else        ``masks.json``
=====================  ==================================================================

Check: the primer-face roughness recomputed from the re-derived depth (the
same formula as ``fit_datum``) must equal ``masks.json["roughness_um"]`` to the
4 decimals stored.  The result is kept in ``Result.datum_check``; a mismatch is
printed and raises with ``strict=True``.
"""
from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

from cartmask_shared import Params, Result, disk, fill_nearest, resample, robust_plane
from x3p_reader import read_x3p

FLIP_WARNING = "height polarity inverted"
_TUPLE_FIELDS = ("primer_center_px", "fp_core_center_px", "drag_tip_px")
_RESULT_FIELDS = set(Result.__dataclass_fields__) - {"z", "valid", "depth", "labels"}


def scan_dirs(inputs: Iterable[str | Path]) -> list[Path]:
    """Per-scan output directories (holding ``masks.json``) under each input, sorted by name.

    An input is either one scan's directory or a CLI output root whose
    sub-directories are scans.
    """
    found: dict[str, Path] = {}
    for item in map(Path, inputs):
        if (item / "masks.json").is_file():
            found.setdefault(item.name, item)
        elif item.is_dir():
            for d in sorted(item.iterdir()):
                if (d / "masks.json").is_file():
                    found.setdefault(d.name, d)
        else:
            raise FileNotFoundError(f"{item}: not a TypeScript CLI output directory")
    return [found[k] for k in sorted(found)]


def read_meta(scan_dir: Path) -> dict:
    return json.loads((Path(scan_dir) / "masks.json").read_text(encoding="utf-8"))


def read_labels(scan_dir: Path) -> np.ndarray:
    """``labels.png`` as a 2-D uint8 array (grey, or RGB(A) with equal colour channels)."""
    path = Path(scan_dir) / "labels.png"
    a = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise OSError(f"could not read {path}")
    if a.ndim == 3:
        c = a[..., :3]
        if not ((c[..., 0] == c[..., 1]).all() and (c[..., 1] == c[..., 2]).all()):
            raise ValueError(f"{path}: colour channels differ; expected a grey label image")
        a = c[..., 0]
    a = a.astype(np.uint8)
    if a.max() > 4:
        raise ValueError(f"{path}: label value {a.max()} > 4")
    return a


def find_x3p(meta: dict, scan_dir: Path, x3p_dirs: Sequence[str | Path] = ()) -> Path:
    """The scan's X3P: ``masks.json["source"]`` if it exists, else ``<name>.x3p`` in ``x3p_dirs``."""
    name = meta.get("name") or Path(scan_dir).name
    cands = []
    src = meta.get("source")
    if src:
        cands += [Path(src), Path(scan_dir).parent / src]
    cands += [Path(d) / f"{name}.x3p" for d in x3p_dirs]
    for c in cands:
        if c.is_file():
            return c
    raise FileNotFoundError(f"{name}: X3P not found (source={src!r}; searched {list(map(str, x3p_dirs))})")


def _working_surface(x3p: Path, labels_shape: tuple[int, int], um_meta: float):
    surf = read_x3p(x3p)
    for target in (Params.work_um_per_px, um_meta):
        s = resample(surf, target)
        if s.z.shape == labels_shape and abs(s.dx * 1e6 - um_meta) < 1e-3:
            return s
    raise ValueError(f"{x3p.name}: no working grid matches labels.png {labels_shape} at {um_meta} um/px")


def fit_depth(z: np.ndarray, valid: np.ndarray, cx: float, cy: float, rp: float,
              rim_margin_frac: float = Params.rim_margin_frac) -> tuple[np.ndarray, float, np.ndarray]:
    """``cartmask.steps.datum.fit_datum`` on the primer face of ``segment``.

    Returns ``(depth, roughness_um, plane_coef)``; depth is NaN where ``z`` is.
    """
    face = disk(z.shape, cx, cy, rp * (1 - rim_margin_frac))
    yy, xx = np.indices(z.shape)
    ann = face & valid & ~disk(z.shape, cx, cy, 0.45 * rp)
    if ann.sum() < 200:
        ann = face & valid
    coef, _ = robust_plane(z, ann)
    d = z - (coef[0] * xx + coef[1] * yy + coef[2])
    df = fill_nearest(np.where(valid, d, np.nan))
    hp = (df - ndi.gaussian_filter(df, 4.0))[ann]
    rough_um = float(1.4826 * np.median(np.abs(hp - np.median(hp))) * 1e6)
    return d, rough_um, coef


def load_scan(scan_dir: str | Path, x3p_dirs: Sequence[str | Path] = (), heights: bool = True,
              strict: bool = False) -> Result:
    """One scan of the TypeScript CLI output as a :class:`Result`.

    With ``heights=False`` only ``labels.png`` and ``masks.json`` are read
    (``z``, ``valid`` and ``depth`` stay None).
    """
    scan_dir = Path(scan_dir)
    meta = read_meta(scan_dir)
    labels = read_labels(scan_dir)
    kw = {k: v for k, v in meta.items() if k in _RESULT_FIELDS}
    for k in _TUPLE_FIELDS:
        if kw.get(k) is not None:
            kw[k] = tuple(kw[k])
    res = Result(**kw)
    res.labels = labels
    res.source = meta.get("source")            # type: ignore[attr-defined]
    res.datum_check = None                     # type: ignore[attr-defined]
    if not heights:
        return res
    x3p = find_x3p(meta, scan_dir, x3p_dirs)
    s = _working_surface(x3p, labels.shape, float(meta["um_per_px"]))
    z = s.z
    if any(FLIP_WARNING in w for w in meta.get("warnings", [])):
        z = -z
    res.z, res.valid, res.um_per_px = z, np.isfinite(z), s.dx * 1e6
    res.x3p = x3p                               # type: ignore[attr-defined]
    if res.status == "failed" or res.primer_radius_px is None:
        return res                              # cartmask leaves depth unset: nothing to compare
    if res.status == "skipped":
        res.depth = np.zeros_like(z)
        return res
    cx, cy = res.primer_center_px
    d, rough, coef = fit_depth(z, res.valid, cx, cy, res.primer_radius_px)
    res.depth = d
    want = meta.get("roughness_um")
    ok = want is not None and abs(round(rough, 4) - want) <= 1.5e-4
    res.datum_check = dict(roughness_um=rough, masks_json_roughness_um=want, ok=ok,  # type: ignore[attr-defined]
                           plane_coef=[float(c) for c in coef])
    if not ok:
        msg = f"{res.name}: re-derived roughness {rough:.4f} um != masks.json {want}; datum may differ"
        if strict:
            raise ValueError(msg)
        print("warning:", msg, file=sys.stderr)
    return res


def load_scans(inputs: Iterable[str | Path], x3p_dirs: Sequence[str | Path] = (), heights: bool = True,
               strict: bool = False) -> list[Result]:
    return [load_scan(d, x3p_dirs, heights, strict) for d in scan_dirs(inputs)]
