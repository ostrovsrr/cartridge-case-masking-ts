"""Write Python reference fixtures for the core TypeScript port (x3p, preprocess, synthetic).

Run with the system Python (numpy, scipy, opencv-python):

    python reference/make_fixtures.py

``cartmask`` is imported from ``$CARTMASK_PY`` (default: the sibling Python repo
``D:\\rodio\\repos\\cartridge-case``).  Real-scan checksums are computed from
``$CARTMASK_DATA`` (default: ``<CARTMASK_PY>/data``) and skipped when absent.
Everything is written to ``tests/fixtures/core/``.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "tests" / "fixtures" / "core"
PY_ROOT = Path(os.environ.get("CARTMASK_PY", r"D:\rodio\repos\cartridge-case"))
DATA = Path(os.environ.get("CARTMASK_DATA", PY_ROOT / "data"))
sys.path.insert(0, str(PY_ROOT))

from scipy import ndimage as ndi  # noqa: E402

from cartmask.preprocess import fill_nearest, hillshade, resample  # noqa: E402
from cartmask.synthetic import make_head, write_x3p  # noqa: E402
from cartmask.x3p import Surface, read_x3p  # noqa: E402


def arr(a: np.ndarray, dtype: str) -> dict:
    a = np.ascontiguousarray(a, dtype=dtype)
    return {"shape": list(a.shape), "dtype": dtype, "b64": base64.b64encode(a.tobytes()).decode()}


def bits(m: np.ndarray) -> dict:
    m = np.asarray(m, bool)
    return {"shape": list(m.shape), "count": int(m.sum()),
            "b64": base64.b64encode(np.packbits(m.ravel(), bitorder="little").tobytes()).decode()}


def checksum(z: np.ndarray) -> dict:
    """Order-insensitive-ish fingerprint of a height map (finite values only)."""
    fin = np.isfinite(z)
    v = z[fin]
    flat = np.flatnonzero(fin.ravel())
    w = (flat % 97 + 1).astype(np.float64)
    return {"shape": list(z.shape), "nan": int((~fin).sum()), "sum": float(v.sum()),
            "sumsq": float((v * v).sum()), "wsum": float((v * w).sum()),
            "min": float(v.min()) if v.size else None, "max": float(v.max()) if v.size else None,
            "first_finite_index": int(flat[0]) if flat.size else -1,
            "first_finite": float(v[0]) if v.size else None}


def write_custom_x3p(path: Path, raw: np.ndarray, dtype_code: str, nx: int, ny: int, nz: int = 1,
                     z_inc: str | None = None, z_off: str | None = None, valid: bytes | None = None,
                     xml_name: str = "main.xml", prefix_all: bool = False, link: str = "bindata/data.bin") -> None:
    """X3P variants the Python writer cannot produce (int types, valid-points mask, prefixed tags)."""
    p = "p:" if prefix_all else ""
    cz = f"<{p}AxisType>A</{p}AxisType><{p}DataType>{dtype_code}</{p}DataType>"
    if z_inc is not None:
        cz += f"<{p}Increment>{z_inc}</{p}Increment>"
    if z_off is not None:
        cz += f"<{p}Offset> {z_off} </{p}Offset>"
    vlink = f"<{p}ValidPointsLink>bindata/valid.bin</{p}ValidPointsLink>" if valid is not None else ""
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<p:ISO5436_2 xmlns:p="http://www.opengps.eu/2008/ISO5436_2">
 <{p}Record1><{p}Revision>ISO5436 - 2000</{p}Revision><{p}FeatureType>SUR</{p}FeatureType>
  <{p}Axes>
   <{p}CX><{p}AxisType>I</{p}AxisType><{p}DataType>D</{p}DataType><{p}Increment>3.125e-06</{p}Increment></{p}CX>
   <{p}CY><{p}AxisType>I</{p}AxisType><{p}DataType>D</{p}DataType><{p}Increment> 3.5e-06 </{p}Increment></{p}CY>
   <{p}CZ>{cz}</{p}CZ>
  </{p}Axes></{p}Record1>
 <{p}Record3><{p}MatrixDimension><{p}SizeX>{nx}</{p}SizeX><{p}SizeY>{ny}</{p}SizeY><{p}SizeZ>{nz}</{p}SizeZ></{p}MatrixDimension>
  <{p}DataLink><{p}PointDataLink>{link}</{p}PointDataLink>{vlink}</{p}DataLink></{p}Record3>
</p:ISO5436_2>"""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(xml_name, xml)
        zf.writestr(link, raw.tobytes())
        if valid is not None:
            zf.writestr("bindata/valid.bin", valid)


def x3p_fixtures() -> dict:
    out: dict = {"files": {}}
    # (a) files written by the Python writer
    surf, _ = make_head(size_mm=1.0, um_per_px=25.0, seed=3)                 # 40 x 40 with NaN corners
    write_x3p(surf, OUT / "py_written.x3p")
    rect = Surface(z=surf.z[3:33, :].copy(), dx=25e-6, dy=30e-6, name="rect")  # non-square, anisotropic
    write_x3p(rect, OUT / "py_written_rect.x3p")

    rng = np.random.default_rng(11)
    # int16 with scale/offset, short valid-points mask, every tag prefixed, odd width
    nx, ny = 13, 7
    raw16 = rng.integers(-30000, 30000, nx * ny).astype("<i2")
    mask_bits = rng.random(nx * ny) > 0.2
    packed = np.packbits(mask_bits, bitorder="little").tobytes()[:-3]          # short: last bits count as valid
    write_custom_x3p(OUT / "int16_valid.x3p", raw16, "I", nx, ny, z_inc="1e-8", z_off="2e-6",
                     valid=packed, prefix_all=True)
    # int32 with an increment but no offset, data under a custom link
    nx, ny = 9, 6
    raw32 = rng.integers(-2**30, 2**30, nx * ny).astype("<i4")
    write_custom_x3p(OUT / "int32.x3p", raw32, "L", nx, ny, z_inc="2.5e-10", link="data/heights.bin")
    # float32, two Z layers (first one is used), XML member in a folder with capitals
    nx, ny = 8, 5
    raw_f = rng.normal(0, 1e-5, 2 * nx * ny).astype("<f4")
    raw_f[[3, 17]] = np.nan
    write_custom_x3p(OUT / "float32_nz2.x3p", raw_f, "F", nx, ny, nz=2, xml_name="meta/Main.XML")

    for name in ["py_written.x3p", "py_written_rect.x3p", "int16_valid.x3p", "int32.x3p", "float32_nz2.x3p"]:
        s = read_x3p(OUT / name)
        s_up = read_x3p(OUT / name, image_orientation=False)
        assert np.array_equal(s_up.z[::-1], s.z, equal_nan=True)
        out["files"][name] = {"dx": s.dx, "dy": s.dy, "name": s.name, "z": arr(s.z, "<f8"),
                              "stats": checksum(s.z)}

    real = ["glock/9mm__PopStatGlock__AADW896_G17_RP2.x3p", "raw/9mm__E3__G0002_A0002.001_CC01_TF.x3p",
            "raw/40SW__CTS15-526__G1_B006_1.x3p"]
    out["real"] = {}
    for rel in real:
        if (DATA / rel).exists():
            s = read_x3p(DATA / rel)
            out["real"][rel] = {"dx": s.dx, "dy": s.dy, "name": s.name, "stats": checksum(s.z)}
    return out


def preprocess_fixtures() -> dict:
    surf, _ = make_head(size_mm=7.0, um_per_px=50.0, seed=2)                 # 140 x 140
    z = surf.z.copy()
    rng = np.random.default_rng(5)
    z.ravel()[rng.choice(z.size, 250, replace=False)] = np.nan              # scattered holes
    z[60:67, 80:86] = np.nan                                                # a block hole
    base = Surface(z=z, dx=50e-6, dy=50e-6, name="pp")
    aniso = Surface(z=z, dx=50e-6, dy=60e-6, name="aniso")
    fine = Surface(z=z[:90, :110].copy(), dx=12e-6, dy=12e-6, name="fine")
    out = {"input": arr(z, "<f8")}
    r1 = resample(base, 70.0)
    r2 = resample(aniso, 40.0)
    r3 = resample(fine, 35.0)
    r_same = resample(base, 40.0)
    out["resample"] = {
        "iso_70": {"dx": r1.dx, "dy": r1.dy, "z": arr(r1.z, "<f8")},
        "aniso_50x60_to_40": {"dx": r2.dx, "dy": r2.dy, "z": arr(r2.z, "<f8")},
        "fine_12_to_35": {"dx": r3.dx, "dy": r3.dy, "z": arr(r3.z, "<f8")},
        "noop_is_same_object": r_same is base,
    }
    out["fill_nearest"] = arr(fill_nearest(z), "<f8")
    out["hillshade_default"] = arr(hillshade(z, 50e-6), "u1")
    vmask = np.isfinite(z)
    vmask[:, :20] = False
    out["hillshade_opts_valid"] = arr(vmask, "u1")
    out["hillshade_opts"] = arr(hillshade(z, 50e-6, valid=vmask, azimuth_deg=90.0, altitude_deg=50.0,
                                          exaggeration=1.5), "u1")
    zr = resample(base, 70.0).z
    out["hillshade_resampled_input"] = arr(zr, "<f8")
    out["hillshade_resampled"] = arr(hillshade(zr, 70e-6), "u1")
    return out


SYN_SETS = {
    "default_40": dict(um_per_px=40.0),
    "nodrag_step": dict(um_per_px=40.0, drag_angle_deg=None, aperture_step_um=20.0, seed=4),
    "slot_shear": dict(um_per_px=40.0, rect_aperture=(0.6, 1.6), rect_angle_deg=25.0, shear_um=4.0),
    "shear_angle": dict(um_per_px=40.0, shear_um=5.0, shear_angle_deg=200.0, shear_period_mm=0.08,
                        primer_center_mm=(-0.2, 0.1), pin_offset_mm=(-0.15, 0.2), drag_angle_deg=-60.0),
    "rough_band": dict(um_per_px=40.0, rough_band_um=3.0, seed=7, crater_depth_um=250.0, drag_length_mm=0.6),
    "banker_round": dict(size_mm=1.0, um_per_px=16.0, primer_radius_mm=0.3, pin_radius_mm=0.1,
                         aperture_radius_mm=0.2, primer_center_mm=(0.0, 0.0), pin_offset_mm=(0.0, 0.0)),
}


def synthetic_fixtures() -> dict:
    out: dict = {"sets": {}}
    for key, kw in SYN_SETS.items():
        surf, truth = make_head(invalid_corners=False, **kw)
        n = surf.z.shape[0]
        seed = kw.get("seed", 0)
        noise = ndi.gaussian_filter(np.random.default_rng(seed).normal(0, 3.0, (n, n)), 1.0)
        geom = surf.z * 1e6 - noise
        rough = kw.get("rough_band_um", 0.0)
        if rough:
            bump = ndi.gaussian_filter(np.random.default_rng(seed + 1).normal(0, 1, (n, n)), 1.5)
            geom[truth["shear_mask"]] -= rough * bump[truth["shear_mask"]] / bump.std()
        surf_c, _ = make_head(**kw)
        out["sets"][key] = {
            "kwargs": {k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()},
            "n": n, "dx": surf.dx,
            "primer_center_px": list(truth["primer_center_px"]), "primer_radius_px": truth["primer_radius_px"],
            "pin_center_px": list(truth["pin_center_px"]), "pin_radius_px": truth["pin_radius_px"],
            "aperture_radius_px": truth["aperture_radius_px"], "drag_angle_deg": truth["drag_angle_deg"],
            "tail_mask": bits(truth["tail_mask"]), "rect_mask": bits(truth["rect_mask"]),
            "shear_mask": bits(truth["shear_mask"]),
            "geometry_um": arr(geom, "<f4"),
            "corner_nan": int(np.isnan(surf_c.z).sum()),
            "noise_std_um": float(noise.std()),
        }
    surf, truth = make_head()
    out["default_full"] = {
        "n": surf.z.shape[0], "dx": surf.dx, "dy": surf.dy, "name": surf.name,
        "nan": int(np.isnan(surf.z).sum()),
        "primer_center_px": list(truth["primer_center_px"]), "primer_radius_px": truth["primer_radius_px"],
        "pin_center_px": list(truth["pin_center_px"]), "pin_radius_px": truth["pin_radius_px"],
        "aperture_radius_px": truth["aperture_radius_px"], "drag_angle_deg": truth["drag_angle_deg"],
        "tail_mask": bits(truth["tail_mask"]), "rect_count": int(truth["rect_mask"].sum()),
        "shear_count": int(truth["shear_mask"].sum()),
        "finite_mean_um": float(np.nanmean(surf.z) * 1e6),
    }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("x3p.json", x3p_fixtures), ("preprocess.json", preprocess_fixtures),
                     ("synthetic.json", synthetic_fixtures)]:
        (OUT / name).write_text(json.dumps(fn(), indent=1), encoding="utf-8")
    total = sum(p.stat().st_size for p in OUT.iterdir())
    print(f"wrote {OUT} ({total / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
