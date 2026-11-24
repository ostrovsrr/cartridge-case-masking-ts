# Aperture shear and slotted apertures

This page covers how Cartmask finds aperture shear and the firing-pin aperture, what changed from the previous release, and the evidence behind the change. The changes address three weaknesses of the earlier masks:

1. **Texture-only aperture shear.** Shear was any rough patch near the aperture. It missed weakly rough shear, for example on Lightstone 6-1. It also fired on rough metal and on the aperture-rim step.
2. **Rectangular (Glock) apertures as disks.** The slot plateau counted as breech face, and the impression often leaked onto the lower primer face beside the slot.
3. **A fixed 1.5 × fallback aperture.** When no flow-back ring was found, the aperture was set to 1.5 × the pin radius.

There are no examiner labels. The evidence comes from three sources: synthetic heads with known masks, consistency with the mechanism and the Lightstone (2010) observations, and visual inspection of before/after figures for every scan whose masks changed. None of it measures accuracy against an examiner.

## What the pipeline does now

### Aperture shear follows the striation direction

The aperture edge cuts the shear while the barrel unlocks. The same motion drags the firing pin, so shear striations run **along the drag axis**. When no drag is detected, they run **radially** through the sheared sector.

1. A structure tensor on the band-passed depth (24 µm high-pass, 48 µm window) gives each pixel a striation direction and a coherence. Gradient magnitudes are capped, so a single step edge cannot dominate the window.
2. The **oriented texture energy** is texture energy × coherence × alignment with the shear axis. The axis is the drag axis, or the radial direction when there is no drag. This energy is divided by its breech-face mean. Breech-face filing marks that happen to share the direction therefore raise the reference instead of becoming shear. This matters on Lightstone, where the breech face is horizontally striated too.
3. Candidates are connected regions of the aperture band above 2.5 × that mean. With drag, only the ±75° sector opposite it is searched. The strongest candidate is accepted only when all of these hold:
   - it covers at least 0.05 mm²;
   - its texture is at least 1.4 × (with drag) or 1.6 × (without drag) the breech-face median;
   - its own striations are consistent (orientation consistency ≥ 0.3);
   - they lie within 20° of the axis.
4. Accepted candidates grow by hysteresis above 1.5 ×.

`Result.shearEvidence` (`shear_evidence` in `masks.json`) records the measurements, or the reason no shear was accepted.

### Slotted apertures are detected before the circular model

A rectangular firing-pin hole leaves a raised flow-back plateau between two long, straight, parallel steps. For every direction, the height gradient along that direction is averaged over lines perpendicular to it. This is a signed Radon projection, equivalent to a line Hough transform for steps. A raised slot gives a rising edge on one side of the pin and a falling edge on the other. The strongest pair is a slot only when all of these hold:

- both edges stand out: score ≥ 10 × the projection noise;
- the response is direction-selective: ≥ 3 × its median over all directions. A round flow-back ring scores alike in every direction; the most selective round-pin scan in the cohort reaches 2.8.
- the edges are 0.8–2.2 mm apart.

Along the slot, the zone follows the raised plateau, bridging unmeasured stretches and the crater. The slot then changes three things:

- It replaces the aperture disk, so the breech face excludes it.
- *Measured* impression pixels outside the slot are dropped. There, depth below the datum is the lower primer face, not crater. Unmeasured crater is kept.
- Strips of ±200 µm along the two edges are kept out of the shear search, because a straight step along the drag axis would read as one long striation.

Round-pin scans take the circular path unchanged. `Result.aperture_shape` reports `circle` or `rectangle`, and `Result.aperture_evidence` holds the slot score, anisotropy, width and angle.

### Aperture-edge cue (available, off by default)

`segment(surf, { apertureEdge: true })` (`aperture_edge` in the Python reference) tries two further cues when no flow-back ring is found within 2.1 × the pin radius. Each needs evidence all around the crater:

- **Wide ring.** The flow-back fall-off may lie out to 2.6 × the pin radius, but within 0.6 × the primer radius. It must be seen in at least 9 of 12 sectors.
- **Edge step.** A closed outward down-step: a robust circle through the strongest step on each of 120 rays, supported by at least half of them and centred on the pin.

**Why it is off.** On real scans it moves the breech-face boundary of 3 E3 and 10 Lightstone scans:

- On E3 G0143, G0232 CC02 and G0242 the aperture moves out to the outer edge of a visibly raised flow-back ring. Breech-face IoU against the default is 0.77–0.86.
- On Lightstone the circle follows the outer arc of the striated crescent.

The new boundaries look right on the figures, but the E3 breech faces were meant to stay stable, and nothing independent confirms them. The comparison with the cue on is kept in [`samples/aperture-and-shear/edge-cue-on/`](../samples/aperture-and-shear/edge-cue-on/). That folder includes the E3 figures and a matching rerun (E3 breech-face AUC 0.967 [0.951, 1.000]).

## Evidence on 152 real scan files

The previous release (commit `f2e3c91`) and the current code were run on all 152 pinned files: the 34 original files (32 distinct scans) and the 118 expanded scans. [`per-scan.csv`](../samples/aperture-and-shear/per-scan.csv) holds, per file:

- the area of each class before and after;
- the per-class IoU between old and new masks;
- the QC score, review flag and review reasons;
- the aperture shape and the shear evidence.

[`by-study.csv`](../samples/aperture-and-shear/by-study.csv) aggregates them. A file "changed materially" when any class with at least 0.02 mm² has IoU below 0.98. That is the case for 78 files, and each has a before/after figure in [`figures/`](../samples/aperture-and-shear/figures/).

| Study | Files | Changed | Breech-face IoU median (min) | Breech face Δ mean mm² | Scans with shear before→after | Impression IoU median (min) | Aperture from evidence before→after | Slot | Flagged before→after |
| --- | ---: | ---: | --- | ---: | --- | --- | --- | ---: | --- |
| E3, original | 17 | 3 | 1.000 (0.979) | +0.02 | 3→0 | 1.000 (1.000) | 9→9 | 0 | 0→0 |
| E3, expanded | 52 | 3 | 1.000 (0.992) | +0.00 | 3→1 | 1.000 (1.000) | 19→19 | 0 | 5→5 |
| Fadul | 11 | 11 | 0.958 (0.935) | +0.17 | 11→1 | 1.000 (1.000) | 11→11 | 0 | 11→11 |
| Weller | 16 | 9 | 0.983 (0.949) | +0.13 | 8→1 | 1.000 (1.000) | 16→16 | 0 | 0→0 |
| Lightstone, original | 2 | 2 | 0.994 (0.989) | +0.01 | 1→2 | 1.000 (1.000) | 0→0 | 0 | 2→2 |
| Lightstone, expanded | 14 | 14 | 0.991 (0.974) | +0.01 | 12→14 | 1.000 (1.000) | 0→0 | 0 | 14→14 |
| PopStat Glock, original | 3 | 3 | 0.397 (0.060) | +0.99 | 3→0 | 0.835 (0.541) | 2→3 | 3 | 3→3 |
| PopStat Glock, expanded | 36 | 33 | 0.590 (0.000) | +0.61 | 31→10 | 0.957 (0.465) | 18→31 | 26 | 36→36 |
| CTS (firing-pin crop, skipped) | 1 | 0 | – | – | – | – | – | – | 1→1 |

"Original" counts files, so the duplicate copies of E3 G0002 CC01 and Fadul 1-1 appear twice. No scan changed its review flag. QC scores changed only on the 14 Glock files where a slot replaced the 1.5 × default aperture.

### E3 round-pin scans

- **No regressions.** Every E3 impression mask is unchanged (IoU 1.000), no E3 scan is taken for a slot, and every E3 review flag and reason is unchanged.
- **One breech face below 0.98 IoU.** E3 G0002 CC01 scores 0.979. The 1.04 mm² previously called shear sat on the concentric flow-back ring step, whose striations are tangential, so it now returns to the breech face. [Figure](../samples/aperture-and-shear/figures/raw__9mm__E3__G0002_A0002.001_CC01_TF.jpg)
- **Small shear removals** on G0004, G0174 and G0300 (0.05–0.40 mm²).
- **G0107 keeps a smaller shear area** (0.17 mm²). It has drag, and the candidate lies within the axis tolerance.

### Aperture shear

- **Lightstone (16 files).** The striated crescent at 9 o'clock, opposite the 3 o'clock drag or teardrop, is found on every file:
  - candidate centroids lie at 134°–227°, all on the left;
  - striation axes lie within 15° of horizontal.
  - The three previous misses (6-1, 5-1, 8-1) are found, and 7-1 is no longer taken on the wrong side.
  - Coverage of the crescent is often partial. Median shear IoU with the old mask is 0.39, and the old mask spilled onto the crater rim.
  - [6-1](../samples/aperture-and-shear/figures/raw__40SW__Lightstone__Lightstone_SW_6-1.jpg) · [7-1](../samples/aperture-and-shear/figures/expanded__40SW__Lightstone__Lightstone_SW_7-1.jpg) · [3-2](../samples/aperture-and-shear/figures/expanded__40SW__Lightstone__Lightstone_SW_3-2.jpg)
- **Fadul, Weller and E3.** The old detector labelled the aperture-rim step or rough patches as shear on 11, 8 and 6 files. Those striations are tangential or absent and are no longer accepted.
  - Shear remains on Fadul 3-1, whose left side has a band striated along the drag axis.
  - It also remains on Weller TW04-01, a false positive described below, and on E3 G0107.
  - [Fadul 1-1](../samples/aperture-and-shear/figures/matching__9mm__Fadul__Fadul_1-1.jpg) · [Weller TW05-01](../samples/aperture-and-shear/figures/expanded__9mm__Weller__TW05-01.jpg)
- **Glock (39 files).** Shear drops from 34 files to 10.
  - Most of the old Glock shear outlined the crater rim or the slot edges.
  - The striated end of the flow-back plateau now lies inside the slot zone, so it is no longer breech face. It is labelled as shear only on a few scans, for example DP773, because Glock breech faces are often striated in the same direction.
  - [DP773](../samples/aperture-and-shear/figures/expanded__9mm__PopStatGlock__FBI_DP773US_RP1.jpg)

### Slotted apertures

- **Detection.** A slot is accepted on 29 of 39 Glock files and on none of the 113 round-pin files.
- **Leaks removed.** Where the impression had leaked onto the lower primer face beside the raised slot, clipping restores the breech face. BCN147 grows from 0.66 to 3.92 mm², FRB198 from 0.01 to 3.30 mm² and BWC506 from 0.23 to 3.75 mm². The spurious purple drag bands on those scans disappear.
- **Plateau no longer counted.** Where the plateau had been counted as breech face, the breech face shrinks; AADW896 G17 goes from 5.11 to 4.54 mm².
- **Review reasons.**
  - Five Glock files (BCN147, CGS762, FRB198, HCIFS 15-2401 and BWC506) no longer trip `impression_too_large` or `breech_face_too_small`.
  - AKL198 no longer trips `impression_not_circular`, and DE893 now trips it because its clipped crater is flatter.
  - Every Glock scan remains flagged, mainly because the crater is unmeasured.
- [AADW896 G17](../samples/aperture-and-shear/figures/glock__9mm__PopStatGlock__AADW896_G17_RP2.jpg) · [BCN147](../samples/aperture-and-shear/figures/expanded__9mm__PopStatGlock__FBI_BCN147US_RP1.jpg) · [FRB198](../samples/aperture-and-shear/figures/expanded__9mm__PopStatGlock__FBI_FRB198_RP1.jpg) · [CBS776](../samples/aperture-and-shear/figures/expanded__9mm__PopStatGlock__FBI_CBS776US_RP1.jpg)

### Downstream matching and replicate consistency

The matching experiment and the replicate-consistency check were both re-run with the current masks. The numbers are in [Results at a glance](evaluation.md#results-at-a-glance). E3 breech-face AUC is 0.975 [0.951, 1.000], against 0.971 before; the primer disc (0.926) and the full 5 mm crop (1.000) are unchanged, and Fadul stays at 1.000. Mean replicate breech-face IoU barely moves (E3 0.82 → 0.83, Fadul 0.81 → 0.82). Both are downstream checks (81 matching pairs from 19 scans, and 27 replicate pairs). They do not measure mask accuracy.

## Synthetic ground truth

[`synthetic-before.json`](../samples/aperture-and-shear/synthetic-before.json) and [`synthetic-after.json`](../samples/aperture-and-shear/synthetic-after.json) score both versions on the same 14 heads. `makeHead` in [`src/synthetic.ts`](../src/synthetic.ts) (`make_head` in the Python reference) has four optional features with known masks, all off by default:

- a rectangular plateau;
- a flat aperture plateau whose only trace is an edge step;
- a band striated along the shear axis;
- an isotropic rough band in the same place.

| Case | Previous release: shear px (recall / precision) | Current: shear px (recall / precision) | Other |
| --- | --- | --- | --- |
| Round head, drag / no drag | 4803 / 4845, all false | 0 / 0 | |
| Striated band, drag | 13392 (0.99 / 0.68) | 12067 (0.96 / 0.73) | |
| Weak striated band (1.5 µm), drag | 12883 (0.98 / 0.69) | 10664 (0.85 / 0.72) | |
| Striated band, no drag | 13274 (0.99 / 0.68) | 8228 (0.86 / 0.95) | |
| Isotropic rough band, drag / no drag | 13488 / 13350, all false | 0 / 0 | |
| Slot 0°, drag / no drag; slot 45° | 6308 / 6944 / 6576, all false | 0 / 0 / 0 | Breech face inside the slot: 43–49 % before, 0 % now |
| Edge step only (6 µm) | 5707, all false | 0 | Aperture radius −11 % (default); +0.1 % with `apertureEdge: true` |
| No ring, no step | 3755, all false | 0 | Default kept, with the cue on or off |

The previous detector could not tell a striated band from rough metal: its recall and precision were the same for both. It also labelled shear on every plain head. [`tests/aperture-shear.test.ts`](../tests/aperture-shear.test.ts) checks the following:

- recovery of striated bands under four drag configurations;
- silence on plain heads, isotropic rough bands and striations outside the search sector;
- slots at 0°, 45° and 100°, with the breech face outside the slot and the impression clipped to it;
- round heads of several sizes staying round;
- the edge-step aperture, and the switches.

## Remaining failure cases

- **Undetected drag.** On Weller TW04-01, the drag channel at 3 o'clock is shallower than the impression threshold, so no drag is detected. Its radial walls pass as shear, and a visible striated area at 9 o'clock is missed. This scan is not flagged. [Figure](../samples/aperture-and-shear/figures/expanded__9mm__Weller__TW04-01.jpg)
- **Partial shear coverage.** On Lightstone the mask often covers only the strongest part of the crescent (3-2, 6-1), with occasional small blobs elsewhere.
- **Glock plateau-end shear.** The striated end of the slot plateau is now excluded from the breech face, but it is usually not labelled as shear.
- **Ten Glock files keep the circular model.** These are DW621, CC843, DR850, ZDU872, F1789438, 16-0046 and HCIFS 15-2364, -2542, -2696 and -2769. On some, the impression already covers most of the face and hides the slot edges. On others the edges are weak or oblique. Their masks are unchanged and they remain flagged.
- **Flat-sided craters.** Where the slot is narrower than the measured crater, the clipped impression has flat sides (DE893, HMV446). This is plausible for a pin passing through a slot, but unverified.

## Reproducing the evidence

The before/after numbers above were measured with the Python reference implementation: the previous release is its commit `f2e3c91`, the current masks are the release this TypeScript port reproduces byte for byte (see the [README](../README.md#parity-with-the-python-reference)). The revision tools here read and write the same run layout as the Python ones (`<key>.json`, `<key>.npz` with `labels`, `<key>.png`), so a "before" run made by the reference's `scripts/compare_revisions.py` can be compared with a TypeScript "after" run:

```powershell
# in the Python reference repository: the previous release
git archive -o prev.tar f2e3c91 cartmask          # unpack to prev/
python scripts/compare_revisions.py run --code prev --data-root <data> --out outputs/revisions/before

# here
npx tsx scripts/compare-revisions.ts run --data-root <data> --out outputs/revisions/after
npx tsx scripts/compare-revisions.ts compare <reference>/outputs/revisions/before outputs/revisions/after --out samples/aperture-and-shear

# synthetic heads: made by the reference generator, scored here
python scripts/compare_revisions.py synthetic --make outputs/revisions/synthetic.npz        # in the reference
npx tsx scripts/compare-revisions.ts synthetic --cases <reference>/outputs/revisions/synthetic.npz --out outputs/revisions/synthetic-after.json
```

Scoring the reference's heads this way reproduces [`synthetic-after.json`](../samples/aperture-and-shear/synthetic-after.json) byte for byte. `compare-revisions.ts synthetic --make` writes heads with the TypeScript generator instead: same geometry and known masks, but its fine roughness comes from a different random stream than NumPy's, so the three striated-band cases score slightly differently (for example 12217 instead of 12067 shear pixels for "striated shear, drag"). The reference scores those TypeScript-made heads exactly as this repository does.

For a future change to this repository, export the earlier revision with `git archive -o prev.tar <commit> src` (unpack to `prev/`) and pass `--code prev` to `run` and `synthetic`; the script then imports `prev/src/index.ts` instead of the working tree. `--workers N` (default 6) splits `run` over N processes.
