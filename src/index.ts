/** Public API of the TypeScript port of `cartmask`. */
export {
  APERTURE_SHEAR,
  BREECH_FACE,
  DEFAULT_PARAMS,
  FIRING_PIN_IMPRESSION,
  FIRING_PIN_SHEAR,
  LABEL_NAMES,
  Result,
  makeParams,
  paramsFromPython,
  segment,
  type Params,
  type Status,
} from "./segment.js";
export { fillNearest, hillshade, resample } from "./preprocess.js";
export { encodeX3P, makeSurface, parseX3P, readX3P, writeX3P, type Surface } from "./x3p.js";
export { makeHead, type HeadOptions, type HeadTruth } from "./synthetic.js";
export { apertureZone, findAperture, findApertureEdge, findRectAperture, type Evidence, type RectAperture } from "./steps/aperture.js";
export { fitDatum, isInverted, robustPlane } from "./steps/datum.js";
export { findDrag, type Drag } from "./steps/drag.js";
export { fitCore, largestSeededComponent, segmentFiringPin } from "./steps/firing_pin.js";
export { disk, fitCircle, robustCircle, sample } from "./steps/geometry.js";
export { caseHead, findPrimer, plausibility, primerInsideFraction, type PrimerInfo } from "./steps/primer.js";
export {
  MAX_IMPRESSION_FRACTION,
  MIN_BREECH_FACE_FRACTION,
  MIN_CIRCLE_AGREEMENT,
  MIN_CRATER_MEASURED,
  qualityControl,
} from "./steps/qc.js";
export { findApertureShear, orientedEnergy, striationField, textureEnergy } from "./steps/shear.js";
