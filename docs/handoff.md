# Cartmask technical handoff

Cartmask segments primer regions on calibrated topography of fired cartridge cases. This repository is the TypeScript port of the Python reference implementation; the history below is the reference's, and the port reproduces its released masks byte for byte. The intended deliverables are an inspectable mask, a record of the measurements behind it, and an explicit review decision. The review decision can be audited independently of the geometry.

## How the project was built

| Round | Work | Evidence |
| --- | --- | --- |
| Initial build | Read X3P scans, segment the primer and toolmark regions, and export images and native-grid masks. | [Historical baseline](../CHANGELOG.md), [sample overlays](../samples/) and known-geometry tests. |
| Verification | Refactor the pipeline into stages, reproduce the original masks, add clustered matching statistics, and build offline review and annotation tools. | [Method](method.md), [matching data](../samples/matching/) and [report](../report/index.html). |
| Release audit | Freeze commit `37bc9fc` (Python reference), verify the input hashes, test candidate changes, review the integrated output, and rebuild the handoff. | The benchmark and acceptance checks below. |
| Release fixes | Visual review of the 150-scan audit found primers placed on headstamps and false polarity flips on E3 full heads. Add plausibility re-ranking, a stricter flip rule and hard review gates; recompute matching; publish the failures. | [Failure modes](evaluation.md#known-failure-modes), [regression tests](../tests/sanity-gates.test.ts). |
| TypeScript port | Port the pipeline, X3P I/O, exports, report, annotator and audit tools to TypeScript; keep the research experiments (matching, 2D, replicates) in Python, reading the TypeScript masks. Accept only byte-identical label grids. | [Parity](../README.md#parity-with-the-python-reference), [audit baseline](../samples/validation/baseline-3d.json), [CHANGELOG](../CHANGELOG.md). |

## Why the design looks this way

| Decision | Reason | Trade-off |
| --- | --- | --- |
| Work in calibrated height units | Relative depth and physical scale survive changes in image rendering. A rule describes which part of the measurement produced a label. | A flat render can support shape cues but cannot recover the missing elevation signal. |
| Fit the outer primer and local datum before the crater | The surrounding case head and specimen tilt otherwise distort the reference surface. | The fitter needs enough primer in view and a valid reference region. Too-small crops are routed to review or skipped. |
| Use a circular model as the starting point | The regular specimens in the initial E3 set have a useful circular prior. A simple model can be checked against the observed footprint. | Rectangular apertures violate it, so a slot detector runs first and replaces the disk when two straight parallel steps are found (29 of 39 Glock scans). Incomplete impressions still belong in review diagnostics. |
| Judge aperture shear by striation direction, not roughness alone | Shear is cut by the same unlocking motion that drags the pin, so its striations run along the drag axis. Roughness alone also fired on rim steps and rough metal. | Coverage of the striated band is often partial, and an undetected drag can still mislead it. |
| Preserve deterministic region labels in the X3P export | A discrete native-grid mask can be reopened alongside the original measurement. | Preview colours are a presentation convention. The label IDs and coordinate transform are the data contract. |
| Evaluate masks and matching separately | Pair ranking measures one downstream use. A direct label comparison measures agreement on where a region ends. | Pair results share sources and specimens, so their uncertainty must account for those groups. Independent label collection requires an examiner workflow. |
| Keep model-assisted labels out of the primary evaluation | Assisted review can speed up corrections. Drawing independently avoids showing the evaluator the model's answer. | Independent labelling takes longer and may reveal genuine ambiguity. Record ignored pixels and examiner agreement where applicable. |

The alternative to an explicit rule pipeline was a supervised segmentation model. The project began without independent region labels. The current pipeline makes physical assumptions visible and can be used to design the first labelled benchmark. A learned replacement would need sufficient independent labels and a held-out evaluation by source and instrument.

## Dataset roles

The original manifest contains **34 files representing 32 distinct scans**. Two files appear in more than one workflow. This fixed set includes the masking demo, rectangular-impression review cases, and the saved downstream pairing experiment. Keep it as the regression baseline.

The expanded NIST cohort adds **118** measurements for a total of **150 distinct scans**. Its selection favours distinct source firearms where inventory permits. E3 adds full-head 9 mm coverage. PopStat Glock adds rectangular-impression cases. Lightstone adds .40 S&W crops. Weller adds measurements acquired with photometric stereo. The cohort uses a fixed seed, stores specimen and measurement identifiers, and records SHA-256 for each acquired scan. [Expanded manifest](expanded-scan-manifest.csv) · [NIST database overview](https://www.nist.gov/programs-projects/nist-ballistics-toolmark-database) · [E3 study](https://tsapps.nist.gov/NRBTD/Studies/Studies/Details/a023199a-b9f3-4a1a-89e8-c94054a7cf61)

More unlabelled scans support coverage estimates and failure discovery. The relevant next data for per-region accuracy are independently drawn labels from a stratified subset, with entire source groups held out during tuning. The 81 historical pair scores remain a separate experiment.

## Reproduce the evidence

Run from the repository root with Node 20 or newer after `npm ci`. The Python evidence experiments need Python 3.12 and `python -m pip install -r python/requirements.txt`; the rebuild skips them with a message when Python or a script is missing.

```powershell
npx tsx scripts/fetch-nbtrd.ts --set all --verify-only
npx tsx scripts/audit-masks.ts --out outputs/audit/current-3d.json --csv outputs/audit/current-3d.csv --compare-with samples/validation/baseline-3d.json
npx tsx scripts/audit-masks.ts --manifest docs/expanded-scan-manifest.csv --out outputs/audit/expanded-3d.json --csv outputs/audit/expanded-3d.csv
npx tsx scripts/rebuild-evidence.ts --out outputs/release
npx tsc --noEmit
npx vitest run
python -m pytest python/tests -q
```

The audit writes the same bytes as the Python reference's `scripts/audit_masks.py`: the first audit reproduces [`baseline-3d.json`](../samples/validation/baseline-3d.json) and [`current-3d.csv`](../samples/validation/current-3d.csv) exactly, the second [`expanded-3d.csv`](../samples/validation/expanded-3d.csv). `npm run audit` and `npm run audit:expanded` are shortcuts.

Add `--data-root <folder>` to the fetch, audit and rebuild scripts when the scans live outside `data/`. The real-scan tests read `CARTMASK_DATA` (the folder holding `raw/`, `glock/`, `matching/` and `expanded/`) and skip when it is unset; set `CARTMASK_AUDIT_EXPANDED=1` to include the 118-scan audit in `tests/audit.test.ts`.

The rebuild uses the saved pairing scores by default. Use `--rerun-registration` to recompute the comparisons from the verified matching scans if the masks or registration changed. The two modes answer different reproducibility questions and have different runtimes. `--replicates` adds the replicate-consistency experiment and `--audit` the mask audit.

## What the verification changed

- **3D review.** Two measurable warning signs, crater coverage and circle agreement, were separated from the composite score.
- **Release fixes.** Three E3 primers were placed on the headstamp and two E3 scans were wrongly flipped as casts; one of these had zero breech face and was not flagged. Primer candidates are now re-ranked by physical plausibility, a flip needs three agreeing pieces of evidence, and six hard gates give every review flag a reason code. Of the 34 original files, 33 label grids are byte-identical; G0232 CC03 changed because its primer is now found. In the expanded cohort six label grids changed, all of them intended.
- **Aperture shear and slots.** Shear now needs striations along the shear axis that stand out from the breech face; Glock slots are detected from two straight parallel steps; an aperture-edge cue exists but is off by default. Against the previous release on all 152 files: no review flag changed, no E3 impression changed, and one E3 breech face moved below 0.98 IoU (G0002 CC01, false shear removed). [Aperture and shear](aperture-and-shear.md)
- **Acquisition.** The original fetcher reselected specimens from live pages. The release fetcher uses the pinned URL and checksum. The acquisition run respected NIST's rate limit, recovered, and fetched all 118 planned files.
- **Annotation.** A blinded packet replaced scan names with opaque IDs and excluded algorithmic prefill. Repeated exports in one Downloads folder could be silently renamed, so exports are timestamped and paired to keep both revisions discoverable.
- **2D.** A shading fallback helped on the development figure. It produced no gain on eight synthetic and six real-proxy comparisons, so it remains a benchmark candidate. [Evaluation and measured results](evaluation.md)

Independent examiner labels are still needed for real-scan region accuracy and review false-negative estimates. The [annotation packet protocol](annotation.md) identifies 24 cases across four study strata, with six distinct sources per stratum. More unlabeled scans would be most useful after these labels show which instruments or failure modes are underrepresented.

## Five-minute interview demo

1. Open the [offline report](../report/index.html). Show the 150-scan evaluation and its per-study breakdown. In the viewer, toggle the region masks and show the review factors on one ordinary scan and one ambiguous scan.
2. Explain the calibrated assumptions using the displayed primer, datum and impression. Show the exported label map beside the native scan.
3. Open the matching table. Say plainly that on E3 the breech-face mask (AUC 0.975) does not beat the full 5 mm crop centred on the primer (1.000), and that the earlier opposite result came from a primer placed on the headstamp. Then distinguish that result from direct boundary accuracy.
4. Open the [annotator](../report/annotator.html). Show how the label PNG and metadata are exported and explain the independent scoring protocol.
5. End with the [failure gallery](evaluation.md#known-failure-modes): one fixed headstamp case, one flagged flow-back plateau, one unmeasured crater and the improved Glock slot. Use the 2D brief result to distinguish image-only suggestions from the calibrated workflow, and explain what further labels would change.
