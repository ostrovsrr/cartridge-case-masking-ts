"""Replicate-scan consistency: do repeat firings of one firearm get the same breech-face mask?

    npx tsx src/cli.ts <data>/matching <data>/raw <data>/expanded -o outputs/ts-replicates
    python python/replicate_consistency.py outputs/ts-replicates -o outputs/replicates

Port of ``scripts/replicate_consistency.py``: the masks are the TypeScript
CLI's (``labels.png`` + ``masks.json`` per scan, heights re-read from the X3P by
``ts_masks``), the registration and statistics are unchanged.

Scans are grouped by study and firearm (``compare.identity``); only
firearms with two or more scans are used.  Every pair within a firearm is
registered exactly as in the matching experiment (band-passed breech-face
heights, rotation and translation by masked normalised cross-correlation on a
5 mm crop centred on the primer), and the two breech-face masks are compared
by IoU after registration.  All replicates are then aligned onto the first
scan of the firearm and a consensus mask is formed by majority vote (a pixel
is kept when more than half of the replicates label it breech face; with two
replicates this is their intersection).

Writes ``consistency.json`` (per-firearm pairwise IoU and consensus areas)
and ``consensus_<firearm>.png`` figures.  Mask agreement after registration
measures repeatability of the masking, not its correctness: a mask that is
wrong the same way on every replicate still agrees.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from compare import _rotate, identity, prepare, register
from ts_masks import load_scan, scan_dirs

UM = 12.0
CROP_MM = 5.0


def aligned_mask(pa, pb, angle: float, dy: int, dx: int) -> np.ndarray:
    """Breech-face mask of ``pb`` moved into the frame of ``pa``."""
    mb = _rotate(pb.mask, angle, cv2.INTER_NEAREST)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(mb, M, mb.shape[::-1], flags=cv2.INTER_NEAREST) > 0.5


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def figure(name: str, ref_img: np.ndarray, masks: list[np.ndarray], labels: list[str], path: Path) -> None:
    """Vote-count map and consensus outline over the reference scan's band-passed heights."""
    size = 360
    g = ref_img.astype(float)
    lim = np.percentile(np.abs(g[g != 0]), 99) if (g != 0).any() else 1.0
    grey = np.clip(128 + 127 * g / max(lim, 1e-9), 0, 255).astype(np.uint8)
    base = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    votes = np.sum(masks, axis=0)
    n = len(masks)
    consensus = votes * 2 > n
    vote_img = np.full((*votes.shape, 3), 255, np.uint8)
    shades = [(235, 235, 235)] + [tuple(int(v) for v in c) for c in
                                  np.linspace((150, 150, 240), (30, 30, 170), n).astype(int)]
    for k in range(n + 1):
        vote_img[votes == k] = shades[k]
    panels = []
    over = base.copy()
    palette = [(40, 40, 220), (220, 120, 30), (60, 170, 60), (180, 60, 180)]
    for m, c in zip(masks, itertools.cycle(palette)):
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(over, cnts, -1, c, 2, cv2.LINE_AA)
    cons = base.copy()
    cons[consensus] = (0.45 * cons[consensus] + 0.55 * np.array((40, 40, 220))).astype(np.uint8)
    for title, img in (("Registered replicate masks", over), (f"Votes (0-{n} of {n})", vote_img),
                       ("Consensus (majority)", cons)):
        tile = np.full((size + 30, size, 3), 255, np.uint8)
        tile[30:] = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
        cv2.putText(tile, title, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        panels.append(np.pad(tile, ((0, 0), (4, 4), (0, 0)), constant_values=255))
    row = np.hstack(panels)
    foot = np.full((48, row.shape[1], 3), 255, np.uint8)
    cv2.putText(foot, f"{name}: " + ", ".join(labels), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(foot, "5 mm crop centred on the primer; replicates registered onto the first scan",
                (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", np.vstack([row, foot]))
    if ok:
        buf.tofile(str(path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="TypeScript CLI output folders (roots or per-scan folders)")
    ap.add_argument("--x3p-dir", action="append", default=[],
                    help="where to look for <scan>.x3p when masks.json['source'] does not resolve")
    ap.add_argument("-o", "--out", type=Path, default=Path("outputs/replicates"))
    ap.add_argument("--studies", default="E3,Fadul,Weller,Lightstone")
    ap.add_argument("--figures", default="E3:G0002,E3:G0232,Fadul:Fadul1,Weller:TW01",
                    help="study:firearm groups to draw")
    args = ap.parse_args()
    studies = set(args.studies.split(","))
    groups: dict[tuple[str, str], dict[str, Path]] = defaultdict(dict)
    for d in scan_dirs(args.inputs):                   # same scan in two folders counts once
        study, gun = identity(d.name)
        if study in studies:
            groups[(study, gun)].setdefault(d.name, d)
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    args.out.mkdir(parents=True, exist_ok=True)
    wanted = {tuple(s.split(":")) for s in args.figures.split(",") if ":" in s}
    method = next(p for p in __doc__.split("\n\n") if p.startswith("Scans are grouped"))
    report: dict = {"note": method.replace("\n", " "), "masks": "TypeScript CLI (labels.png)",
                    "comparison_um_per_px": UM,
                    "crop_mm": CROP_MM, "firearms": []}
    for (study, gun), scans in sorted(groups.items()):
        names = sorted(scans)
        results = {n: load_scan(scans[n], args.x3p_dir) for n in names}
        prep = {n: prepare(r, "breech_face", UM, CROP_MM) for n, r in results.items()}
        pairs = []
        for a, b in itertools.combinations(names, 2):
            _, ang, dy, dx = register(prep[a], prep[b])
            pairs.append({"a": a, "b": b, "mask_iou": round(iou(prep[a].mask > 0.5,
                                                                 aligned_mask(prep[a], prep[b], ang, dy, dx)), 3)})
        ref = names[0]
        masks = [prep[ref].mask > 0.5]
        for other in names[1:]:
            _, ang, dy, dx = register(prep[ref], prep[other])
            masks.append(aligned_mask(prep[ref], prep[other], ang, dy, dx))
        votes = np.sum(masks, axis=0)
        consensus = votes * 2 > len(masks)
        px_mm2 = (UM / 1000) ** 2
        entry = {
            "study": study, "firearm": gun, "scans": names,
            "flagged": [n for n in names if results[n].needs_review],
            "pairwise": pairs,
            "mean_pairwise_iou": round(float(np.mean([p["mask_iou"] for p in pairs])), 3),
            "replicate_mask_mm2": [round(float(m.sum() * px_mm2), 2) for m in masks],
            "consensus_mm2": round(float(consensus.sum() * px_mm2), 2),
            "consensus_over_union": round(float(consensus.sum() / max((votes > 0).sum(), 1)), 3),
        }
        report["firearms"].append(entry)
        print(f"{study:10s} {gun:16s} n={len(names)} mean IoU {entry['mean_pairwise_iou']:.3f} "
              f"consensus {entry['consensus_mm2']} mm2 (of union {entry['consensus_over_union']:.2f})", flush=True)
        if (study, gun) in wanted:
            figure(f"{study} {gun}", prep[ref].img, masks, [n.split("__")[-1] for n in names],
                   args.out / f"consensus_{study}_{gun}.png")
    by_study: dict[str, list[float]] = defaultdict(list)
    for e in report["firearms"]:
        by_study[e["study"]] += [p["mask_iou"] for p in e["pairwise"]]
    report["summary"] = {s: {"pairs": len(v), "mean_iou": round(float(np.mean(v)), 3),
                             "min_iou": round(float(np.min(v)), 3)} for s, v in sorted(by_study.items())}
    (args.out / "consistency.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
