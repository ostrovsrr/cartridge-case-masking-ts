/** Port of Python `tests/test_qc_observations.py`. */
import { expect, it } from "vitest";
import { DEFAULT_PARAMS, makeParams, Result } from "../src/segment.js";
import { disk } from "../src/steps/geometry.js";
import { qualityControl } from "../src/steps/qc.js";
import type { Mask } from "../src/ndimage/index.js";

const shape: [number, number] = [96, 96];
const ones = (): Mask => ({ data: new Uint8Array(96 * 96).fill(1), rows: 96, cols: 96 });
const fresh = (): Result =>
  new Result("synthetic", 8, { primerEdgeInlierFrac: 1, apertureDetected: true, fpCoreCenterPx: [48, 48], fpCoreRadiusPx: 10 });

it("routes a poor circle fit to review even with a high composite score", () => {
  const result = fresh();
  const face = disk(shape, 48, 48, 38);
  const footprint: Mask = { data: new Uint8Array(96 * 96), rows: 96, cols: 96 };
  for (let y = 43; y < 54; y++) for (let x = 34; x < 63; x++) footprint.data[y * 96 + x] = 1;
  qualityControl(result, face, ones(), footprint, makeParams());
  expect(result.qcScore).toBe(1);
  expect(result.needsReview).toBe(true);
  expect(result.qcObservations.impression_circle_iou).toBeLessThan(0.85);
});

it("gates on the measured crater and accepts a clean circle", () => {
  const face = disk(shape, 48, 48, 38);
  const impression = disk(shape, 48, 48, 10);
  const valid = ones();
  for (const missing of [false, true]) {
    const result = fresh();
    if (missing) for (let i = 0; i < 48 * 96; i++) if (impression.data[i]) valid.data[i] = 0;
    qualityControl(result, face, valid, impression, makeParams());
    expect(result.qcObservations.impression_circle_iou).toBe(1);
    expect(result.needsReview).toBe(missing);
    expect(result.qcScore).toBeGreaterThanOrEqual(DEFAULT_PARAMS.qcReviewBelow);
  }
});
