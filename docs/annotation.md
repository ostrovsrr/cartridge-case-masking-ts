# Annotation and scoring

## Independent evaluation packet

The [frozen selection](annotation-cohort.csv) includes six cases from distinct source groups in each of the four added studies. Selection was seeded and did not use the algorithm's outputs. Generate the blinded packet from the repository root.

```powershell
$caseFiles = (Import-Csv docs/annotation-cohort.csv).file
npx tsx scripts/build-annotator.ts @caseFiles -o outputs/annotation/packet.html --blind-map outputs/annotation/private-map.json
```

Give the same HTML to each examiner and retain the private map for scoring. The packet uses opaque case IDs, a shuffled order, and the full working-grid render. It omits source filenames, algorithm masks, and prefill controls. Use **Ignore** for genuinely unassignable pixels and explain the reason in Notes. Draw background only where the regions are confidently absent. Mark each annotation done before export.

```powershell
npx tsx src/evaluate.ts --pred-from data/expanded --truth annotations --blind-map outputs/annotation/private-map.json -o outputs/evaluation
```

The scorer verifies the source hash and grid against the private map and takes the latest completed revision per case and examiner. It excludes Ignore pixels from region metrics and removes neighbouring boundaries from boundary scoring within the matching tolerance. If any pixels are ignored, the centre error is omitted to avoid claiming an exact centroid on an incomplete drawing.

## Assisted review

Open the [offline annotator](../report/annotator.html) for the baseline scans, or build a page for your own X3P inputs.

```powershell
npx tsx scripts/build-annotator.ts data/raw -o report/annotator.html
```

1. Enter an annotator ID. Select a scan and leave the canvas blank to begin an independent annotation.
2. Draw each visible region with the brush or polygon tool. Use background for the case head outside the primer and for areas you cannot assign to a class. Record ambiguous boundaries and missing data in Notes.
3. Mark the scan done and export it. Keep the label PNG and matching JSON sidecar together in `annotations/<annotator>/`. These are discrete label values, so keep the label PNG at its exported dimensions and avoid image editors that apply colour conversion.
4. Ask each examiner to work independently on the same X3P inputs. The scorer keeps pre-filled exports in the detail files and excludes them from independent summary metrics.

```powershell
npx tsx src/evaluate.ts --pred-from data/raw --truth annotations -o outputs/evaluation
```

The scorer processes the X3P on the annotation grid recorded in the JSON. It writes per-scan and per-class CSVs, a Markdown summary and error overlays. When different annotators used the same grid, it also reports inter-annotator overlap. A class absent in both masks is counted separately. Boundary F defaults to a 16 µm tolerance and can be set with `--tol-um`.

The browser may block the second download. Allow multiple downloads for the local annotator and export the scan again. If the scorer reports a shape mismatch, check that the PNG is the label export and that the matching JSON and source X3P are present.
