# Interview walkthrough

1. Open the side-by-side X3P example in the README. Explain the four labels and how the separate drag arrow encodes direction. Mention that the brief's figures swap purple and light blue, and show `--brief-colours`.
2. Explain the strongest physical cue. Where the visible primer edge drops into its pocket, the signed gradient votes towards the centre. Then explain why that alone failed on some full heads: headstamp letters vote too, so candidates are re-ranked by primer size, concentricity with the case head and a crater inside. Fit a local datum so the crater threshold follows depth relative to the primer face.
3. Open the failure gallery. Show one fixed primer placement, one flow-back plateau that is now flagged but still wrong, one unmeasured crater and the Glock slot that is now detected. Point to the reason codes on each flagged scan.
4. Show [aperture and shear](aperture-and-shear.md): shear is accepted only where striations run along the drag axis (Lightstone 6-1 now found, rim steps on Fadul no longer called shear), and the Glock slot replaces the circular aperture. Mention that the aperture-edge cue is off by default and why.
5. Show the matching table. On E3 the full 5 mm crop centred on the primer separates every pair (AUC 1.000) and the breech-face mask scores 0.975, so the mask does not improve matching in this sample. The earlier claim that it did was caused by a primer placed on the headstamp; fixing it changed the ranking. Fadul shows a ceiling effect. Say exactly what "full scan" means: the 5 mm crop centred on the detected primer.
6. Show the replicate consistency figure: repeat firings receive nearly the same breech-face mask after registration, and the consensus mask keeps what most firings agree on.
7. Open the annotator. Explain how two independent labels on the same image help quantify both algorithm agreement and the intrinsic ambiguity of the boundary.

## Questions to be ready for

**Why use a rule-based pipeline?** Calibrated height and known primer geometry provide useful cues without a labelled training set. Individual rules are easy to inspect, and when one fails, as the headstamp and flow-back cases did, the failure can be located and gated. Shear coverage, the remaining circular-model cases and the off-by-default aperture-edge cue identify where annotated data would help most.

**How would the method scale?** Process each scan independently, preserve per-file failures and metadata, and batch by instrument. Sort the batch on `needs_review` and `review_reasons` for triage. Matching evaluates pairs and grows quadratically within a study, so measure alignment cost separately from segmentation.

**What if the scan is cropped or inverted?** Small fields are skipped. The pipeline checks how much of a detected primer is visible and marks uncertain results for review. Heights are flipped only when a central bump, the primer-edge evidence and a plausible flipped crater all agree; on the NIST data no scan is flipped any more, and the two former flips are flagged instead.

**How many failures does the review flag miss?** No examiner labels exist yet, so only an estimate: in my visual review of the 68 E3 full heads, none of the 63 unflagged scans had a gross failure, but about 9 had a visible boundary error, mostly an unlabelled drag tail or an oversized aperture zone. Before the fixes, one gross failure (G0318, zero breech face) was unflagged. The blinded annotation packet is the way to measure this properly.

**How would you test direct accuracy?** Freeze the sample and scoring policy, collect blinded polygon labels from two examiners, and report per-class IoU and boundary F-score along with agreement between examiners. Keep input types and instruments identifiable in the results.

**Why TypeScript, and why is some of it still Python?** The product (reading X3P, masking, exports, the report and the annotator) runs on Node with three small dependencies, so it installs with `npm ci` and the same code can move into a browser. The research experiments (registration for the matching check, the 2D image mode, replicate consistency) stay in Python because they lean on OpenCV, SciPy and matplotlib, and porting them would change the statistics being reported. They read the TypeScript masks, so the evidence is about the masks that ship. The port was accepted only when every label grid matched the Python reference byte for byte.

**Where did AI contribute?** Claude and Codex assisted with implementation, review, and documentation. The repository records measured results and reproducible checks. The candidate should be prepared to explain the method and demonstrate it live.
