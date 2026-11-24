"""Generate SciPy/OpenCV/NumPy parity fixtures for src/ndimage (TypeScript).

Run with the system Python (numpy, scipy, opencv-python):

    python reference/ndimage_fixtures.py

Writes tests/fixtures/ndimage/*.json. Inputs are small and deterministic; every float is written with
``repr`` precision so the TypeScript tests compare against the exact SciPy/OpenCV output.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import scipy
import scipy.fft as sfft
from scipy import ndimage as ndi

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "ndimage"
RNG = np.random.default_rng(20241124)


def arr(a: np.ndarray) -> dict:
    a = np.asarray(a)
    if a.dtype == bool:
        a = a.astype(np.uint8)
    if a.ndim == 1:
        a = a[None, :]
    flat = a.ravel().tolist()
    if a.dtype.kind == "f":
        flat = [float(v) if np.isfinite(v) else None for v in a.ravel()]  # NaN/inf -> null
    return {"shape": list(a.shape), "dtype": str(a.dtype), "data": flat}


def dump(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"versions": {"numpy": np.__version__, "scipy": scipy.__version__, "cv2": cv2.__version__},
               **payload}
    (OUT / f"{name}.json").write_text(json.dumps(payload, separators=(",", ":")))


def disk(shape, cx, cy, r):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def blobs(shape, n, seed):
    rng = np.random.default_rng(seed)
    m = np.zeros(shape, bool)
    for _ in range(n):
        cy, cx = rng.uniform(0, shape[0]), rng.uniform(0, shape[1])
        r = rng.uniform(1.0, 5.0)
        m |= disk(shape, cx, cy, r)
    return m


# --------------------------------------------------------------------------- filters
def filters():
    a = RNG.normal(0, 1, (23, 17))
    b = RNG.normal(0, 1, (9, 6))
    cases = []
    for img, tag in ((a, "a"), (b, "b")):
        for sigma in (0.7, 1.0, 2.5, [1.5, 3.0], 4.0):
            for order in ([0, 0], [0, 1], [1, 0], [1, 1]):
                if tag == "b" and order != [0, 0] and sigma not in (1.0, 4.0):
                    continue
                cases.append({"input": tag, "sigma": sigma, "order": order, "truncate": 4.0,
                              "out": arr(ndi.gaussian_filter(img, sigma, order=order))})
    cases.append({"input": "a", "sigma": 1.3, "order": [0, 0], "truncate": 2.0,
                  "out": arr(ndi.gaussian_filter(a, 1.3, truncate=2.0))})
    uni = [{"input": t, "size": s, "out": arr(ndi.uniform_filter(i, s))}
           for i, t in ((a, "a"), (b, "b")) for s in (3, 4, 9)]
    mx = [{"input": t, "size": s, "out": arr(ndi.maximum_filter(i, size=s))}
          for i, t in ((a, "a"), (b, "b")) for s in (3, 4, 7)]
    med = [{"input": t, "size": s, "out": arr(ndi.median_filter(i, s))}
           for i, t in ((a, "a"), (b, "b")) for s in (3, 5)]
    line = RNG.normal(0, 1, 15)
    line[[3, 4]] = line[2]   # ties
    med1 = [{"size": s, "out": arr(ndi.median_filter(line, s))} for s in (3, 5, 7)]
    gy, gx = np.gradient(a)
    gy2, gx2 = np.gradient(a, 0.25)
    dump("filters", {"a": arr(a), "b": arr(b), "line": arr(line), "gaussian": cases, "uniform": uni,
                     "maximum": mx, "median": med, "median1d": med1,
                     "gradient": {"gy": arr(gy), "gx": arr(gx), "gy_s": arr(gy2), "gx_s": arr(gx2)}})


# --------------------------------------------------------------------------- morphology
def morphology():
    m = RNG.random((30, 28)) < 0.62
    m[0, :] = True                       # touch the border
    m2 = blobs((32, 30), 9, 3)
    m2[10:14, 10:14] = False             # holes inside blobs
    ring = disk((25, 25), 12, 12, 9) & ~disk((25, 25), 12, 12, 5)
    ring |= disk((25, 25), 12, 12, 2)    # island inside the hole
    ring[0:3, 20:25] = True
    cross = ndi.generate_binary_structure(2, 1)
    sq = np.ones((3, 3), bool)
    dk = disk((5, 5), 2, 2, 2)
    even = np.array([[1, 1], [0, 1]], bool)
    structs = {"cross": cross, "square": sq, "disk2": dk, "even": even}
    inputs = {"m": m, "m2": m2, "ring": ring}
    cases = []
    for iname, img in inputs.items():
        for sname, st in structs.items():
            for it in (1, 2, 3):
                if sname in ("disk2", "even") and it == 3:
                    continue
                cases.append({"input": iname, "structure": sname, "iterations": it,
                              "erosion": arr(ndi.binary_erosion(img, st, iterations=it)),
                              "dilation": arr(ndi.binary_dilation(img, st, iterations=it)),
                              "opening": arr(ndi.binary_opening(img, st, iterations=it)),
                              "closing": arr(ndi.binary_closing(img, st, iterations=it))})
    # 1-D closing with a line structure (aperture.py plateau logic)
    plateau = RNG.random(40) < 0.55
    line_cases = [{"gap": g, "closing": arr(ndi.binary_closing(plateau, structure=np.ones(2 * g + 1, bool))),
                   "label": arr(ndi.label(ndi.binary_closing(plateau, structure=np.ones(2 * g + 1, bool)))[0])}
                  for g in (1, 2, 4)]
    fill = {k: arr(ndi.binary_fill_holes(v)) for k, v in inputs.items()}
    seed = np.zeros((32, 30), bool)
    seed[[3, 15, 28], [4, 20, 9]] = True
    seed[0, 0] = True                    # seed outside the mask stays set
    prop = arr(ndi.binary_propagation(seed, mask=m2))
    seed_r = np.zeros((25, 25), bool)
    seed_r[12, 3] = True
    prop_r = arr(ndi.binary_propagation(seed_r, mask=ring))
    dump("morphology", {"inputs": {k: arr(v) for k, v in inputs.items()},
                        "structures": {k: arr(v) for k, v in structs.items()},
                        "cases": cases, "plateau": arr(plateau), "line": line_cases, "fill_holes": fill,
                        "propagation": {"seed": arr(seed), "mask": "m2", "out": prop},
                        "propagation_ring": {"seed": arr(seed_r), "mask": "ring", "out": prop_r},
                        "disk": {"shape": [9, 11], "cx": 4.5, "cy": 3.2, "r": 3.1,
                                 "out": arr(disk((9, 11), 4.5, 3.2, 3.1))}})


# --------------------------------------------------------------------------- measure
def measure():
    m = RNG.random((26, 24)) < 0.45
    vals = RNG.normal(0, 1, m.shape)
    out = {"m": arr(m), "vals": arr(vals)}
    for name, st in (("c4", None), ("c8", np.ones((3, 3), bool))):
        lab, n = ndi.label(m, structure=st)
        idx = np.arange(1, n + 1)
        out[name] = {"labels": arr(lab), "n": int(n),
                     "sums": [float(v) for v in ndi.sum(vals, lab, idx)],
                     "sizes": [int(v) for v in ndi.sum(m, lab, idx)]}
    dump("measure", out)


# --------------------------------------------------------------------------- distance
def distance():
    cases = {}
    m1 = RNG.random((21, 19)) < 0.8
    m2 = np.ones((17, 23), bool)
    m2[8, 11] = False                    # single zero: many ties for row/col symmetric points
    m2[2, 2] = False
    m2[14, 20] = False
    m3 = ~blobs((24, 22), 4, 7)          # large zero regions
    m4 = np.ones((12, 12), bool)
    m4[[0, 0, 11, 11, 5], [0, 11, 0, 11, 6]] = False   # equidistant corners
    for k, m in (("m1", m1), ("m2", m2), ("m3", m3), ("m4", m4)):
        dt, (iy, ix) = ndi.distance_transform_edt(m, return_indices=True)
        cases[k] = {"input": arr(m), "dist": arr(dt), "iy": arr(iy), "ix": arr(ix)}
    dump("distance", cases)


# --------------------------------------------------------------------------- interpolation
def interp():
    a = RNG.normal(0, 1, (13, 11))
    ys = np.r_[RNG.uniform(-2, 14, 40), [0, 12, 5.5, -0.5, 12.5, 6]]
    xs = np.r_[RNG.uniform(-2, 12, 40), [0, 10, 3.25, 10.5, -0.25, 4]]
    mc = ndi.map_coordinates(a, [ys, xs], order=1, mode="nearest")

    src = RNG.normal(0, 1, (23, 17))
    src_i = (RNG.random((23, 17)) * 5).astype(np.int32)
    sizes = [(10, 7), (7, 5), (11, 8), (5, 4), (23, 17), (40, 30), (12, 30), (46, 34), (1, 1), (3, 17)]
    res = []
    for (h, w) in sizes:
        res.append({"rows": h, "cols": w,
                    "area": arr(cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)),
                    "linear": arr(cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)),
                    "nearest": arr(cv2.resize(src_i, (w, h), interpolation=cv2.INTER_NEAREST))})
    # exact integer down-scales (area fast path) and 2x linear
    big = RNG.normal(0, 1, (24, 18))
    fast = []
    for (h, w) in [(12, 9), (8, 6), (6, 6), (4, 3), (12, 6)]:
        fast.append({"rows": h, "cols": w,
                     "area": arr(cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)),
                     "linear": arr(cv2.resize(big, (w, h), interpolation=cv2.INTER_LINEAR)),
                     "area32": arr(cv2.resize(big.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA))})

    rot = []
    img = RNG.normal(0, 1, (19, 23))
    lab = (RNG.random((19, 23)) * 4).astype(np.int32)
    for ang, (cx, cy) in ((17.0, (11.0, 9.0)), (-33.3, (5.5, 12.25)), (90.0, (11.0, 9.0)), (1.7, (0.0, 0.0)),
                          (180.0, (11.0, 9.0)), (-123.4, (7.3, 4.9))):
        M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
        rot.append({"angle": ang, "cx": cx, "cy": cy, "M": M.ravel().tolist(),
                    "linear": arr(cv2.warpAffine(img, M, (23, 19), flags=cv2.INTER_LINEAR, borderValue=0)),
                    "linear32": arr(cv2.warpAffine(img.astype(np.float32), M, (23, 19), flags=cv2.INTER_LINEAR,
                                                   borderValue=0)),
                    "nearest": arr(cv2.warpAffine(lab, M, (23, 19), flags=cv2.INTER_NEAREST, borderValue=0)),
                    "linear_big": arr(cv2.warpAffine(img, M, (27, 21), flags=cv2.INTER_LINEAR, borderValue=0))})
    M2 = cv2.getRotationMatrix2D((3.0, 4.0), 25.0, 1.3)
    shift = np.float32([[1, 0, 2.37], [0, 1, -1.61]]).astype(np.float64)
    extra = {"scaled": {"M": M2.ravel().tolist(),
                        "linear": arr(cv2.warpAffine(img, M2, (23, 19), flags=cv2.INTER_LINEAR, borderValue=0))},
             "shift": {"M": shift.ravel().tolist(),
                       "linear": arr(cv2.warpAffine(img, shift, (23, 19), flags=cv2.INTER_LINEAR, borderValue=0)),
                       "nearest": arr(cv2.warpAffine(lab, shift, (23, 19), flags=cv2.INTER_NEAREST,
                                                     borderValue=0))}}
    dump("interp", {"a": arr(a), "ys": arr(ys), "xs": arr(xs), "map_coordinates": arr(mc),
                    "src": arr(src), "src_i": arr(src_i), "resize": res, "big": arr(big), "fast": fast,
                    "img": arr(img), "lab": arr(lab), "rotate": rot, "extra": extra})


# --------------------------------------------------------------------------- contours
def contours():
    cases = []
    shapes = []
    a = disk((20, 22), 10, 9, 6.5)
    shapes.append(a)
    b = blobs((30, 30), 7, 11)
    b[12:16, 12:16] = False
    shapes.append(b)
    c = np.zeros((15, 15), bool)
    c[3, 3] = True                        # single pixel
    c[6, 2:12] = True                     # 1-px line
    c[9:12, 5] = True                     # vertical line
    c[13, 13] = True
    c[0, 7:10] = True                     # touches border
    shapes.append(c)
    r = disk((25, 25), 12, 12, 10) & ~disk((25, 25), 12, 12, 6)
    r |= disk((25, 25), 12, 12, 2)        # inside the hole: not external
    r[20:25, 0:4] = True
    shapes.append(r)
    d = np.zeros((12, 14), bool)
    d[2:6, 2:6] = True
    d[6:9, 6:10] = True                   # diagonal 8-connection
    d[1, 11] = True
    d[2, 12] = True
    shapes.append(d)
    full = np.ones((6, 7), bool)
    shapes.append(full)
    for s in shapes:
        cnts, _ = cv2.findContours(s.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cases.append({"input": arr(s), "contours": [c[:, 0, :].tolist() for c in cnts]})
    dump("contours", {"cases": cases})


# --------------------------------------------------------------------------- fft
def fft():
    nfl = {"n": list(range(1, 260)) + [1001, 2399, 4801],
           "real": [sfft.next_fast_len(n, real=True) for n in list(range(1, 260)) + [1001, 2399, 4801]],
           "complex": [sfft.next_fast_len(n) for n in list(range(1, 260)) + [1001, 2399, 4801]]}
    cases = []
    for (h, w), shape in (((7, 6), (13, 11)), ((9, 10), None), ((5, 8), (9, 15)), ((6, 7), (16, 18)),
                          ((4, 5), (7, 13)), ((11, 9), (21, 17)), ((3, 3), (49, 46))):
        a = RNG.normal(0, 1, (h, w))
        s = shape or (h, w)
        F = sfft.rfft2(a, s)
        back = sfft.irfft2(F, s)
        # product of two spectra and inverse (masked-NCC style)
        b = RNG.normal(0, 1, (h, w))
        G = sfft.rfft2(b[::-1, ::-1], s)
        corr = sfft.irfft2(F * G, s)
        cases.append({"input": arr(a), "other": arr(b), "shape": list(s), "re": arr(F.real), "im": arr(F.imag),
                      "back": arr(back), "corr": arr(corr)})
    c = RNG.normal(0, 1, (6, 10)) + 1j * RNG.normal(0, 1, (6, 10))
    C = np.fft.fft2(c)
    cplx = {"re": arr(c.real), "im": arr(c.imag), "Fre": arr(C.real), "Fim": arr(C.imag)}
    lens = {}
    for n in (1, 2, 3, 5, 7, 11, 13, 17, 31, 64, 97, 100, 121, 127, 210, 243, 250, 257):
        x = RNG.normal(0, 1, n) + 1j * RNG.normal(0, 1, n)
        X = np.fft.fft(x)
        lens[str(n)] = {"re": x.real.tolist(), "im": x.imag.tolist(), "Xre": X.real.tolist(), "Xim": X.imag.tolist()}
    dump("fft", {"next_fast_len": nfl, "rfft2": cases, "fft2": cplx, "fft1d": lens})


# --------------------------------------------------------------------------- linalg + stats
def linalg_stats():
    cases = []
    x = RNG.uniform(-5, 5, 50)
    y = RNG.uniform(-5, 5, 50)
    z = 0.3 * x - 1.2 * y + 4.0 + RNG.normal(0, 0.1, 50)
    A = np.c_[x, y, np.ones_like(x)]
    cases.append({"A": arr(A), "b": z.tolist(), "x": np.linalg.lstsq(A, z, rcond=None)[0].tolist()})
    t = RNG.uniform(0, 2 * np.pi, 40)
    px, py = 100 + 30 * np.cos(t) + RNG.normal(0, 0.3, 40), 80 + 30 * np.sin(t) + RNG.normal(0, 0.3, 40)
    A = np.c_[2 * px, 2 * py, np.ones_like(px)]
    b = px ** 2 + py ** 2
    cases.append({"A": arr(A), "b": b.tolist(), "x": np.linalg.lstsq(A, b, rcond=None)[0].tolist()})
    A = RNG.normal(0, 1, (12, 6))
    b = RNG.normal(0, 1, 12)
    cases.append({"A": arr(A), "b": b.tolist(), "x": np.linalg.lstsq(A, b, rcond=None)[0].tolist()})
    A = np.c_[x[:10], 2 * x[:10], np.ones(10)]                       # rank deficient: min-norm solution
    b = z[:10]
    cases.append({"A": arr(A), "b": b.tolist(), "x": np.linalg.lstsq(A, b, rcond=None)[0].tolist(),
                  "rank_deficient": True})
    A = RNG.normal(0, 1, (3, 3))                                      # square
    b = RNG.normal(0, 1, 3)
    cases.append({"A": arr(A), "b": b.tolist(), "x": np.linalg.lstsq(A, b, rcond=None)[0].tolist()})

    stats = []
    for n in (1, 2, 5, 10, 101, 1000):
        v = RNG.normal(0, 1, n)
        if n >= 10:
            v[: n // 5] = np.round(v[: n // 5], 1)   # duplicates
        qs = [0, 0.5, 1, 10, 25, 33.3, 50, 75, 88, 90, 95, 99, 99.5, 100]
        stats.append({"v": v.tolist(), "q": qs, "p": [float(np.percentile(v, q)) for q in qs],
                      "median": float(np.median(v)), "mad": float(np.median(np.abs(v - np.median(v))))})
    dump("linalg", {"lstsq": cases, "stats": stats})


if __name__ == "__main__":
    filters()
    morphology()
    measure()
    distance()
    interp()
    contours()
    fft()
    linalg_stats()
    total = sum(p.stat().st_size for p in OUT.glob("*.json"))
    print(f"wrote {len(list(OUT.glob('*.json')))} fixture files to {OUT} ({total / 1e6:.2f} MB)")
