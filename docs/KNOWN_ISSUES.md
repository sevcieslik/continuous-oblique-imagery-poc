# Known issues / risks

## Pass classifier can be too aggressive

On the DH1 benchmark, frames used in the manual final product were sometimes labelled
`CROSSING_PASS` or `UNKNOWN_PASS`. The current classifier uses the ratio of along-circuit progression
to total camera motion plus an overall chainage delta check. This can misclassify short curved or
locally transverse but still legitimate acquisition fragments.

**Next:** direction/pass confidence using local tangent windows, neighbouring pass context and audited
manual sequences.

## Backward reference assignment has a systematic off-by-one pattern

Clean manual-vs-AI disagreements are dominated by an adjacent span shift on backward travel.
Do not fix this with a global `+1/-1` hack. The resolver should use target chainage, pass direction,
boundary distance and incoming/outgoing tangent context.

## Existing/manual 163 is not perfect ground truth

The manual product is internally very continuous but misses topology spans and contains images that
cannot always be confidently mapped back to raw. Calibration must use the audited trusted subset and
human adjudication.

## EO coordinate incompatibility

A project may contain EO sources in different or incorrectly declared coordinate spaces. The pipeline
currently uses spatial distance to alignment to reject incompatible sources. This is necessary but should
be accompanied by clearer CRS diagnostics and a fail-fast report.

## LiDAR structure class is project-specific

Class 215 worked empirically on DH1. Auto profiling previously selected an implausible class because
vertical statistics alone were not constrained enough. Never assume 215 on a new client/dataset without
confirming the classification schema.

## Camera projection is approximate

The pipeline uses EO/OPK, footprints and LiDAR-derived geometry but does not yet include a complete
lens-distortion-aware camera calibration. Geometry should constrain search and provide QA, not be treated
as guaranteed pixel truth.

## Parallel circuits / neighbouring towers

On shared corridors (DH1 + DH2) the neighbouring tower can be closer to the geometry prior than the
target. v0.14 selected the neighbour in 1 of 5 smoke-run crops. v0.15 adds competitor hypotheses,
registration and ownership. Residual risk: if neither LiDAR clusters nor a neighbour CSV describe the
other circuit, and only one tower is detected, registration cannot run and the assignment falls back to
bias-corrected ownership. Provide `--competitor-structures` for known parallel circuits.

## Prior offset varies inside a pass

Smoke-run residuals were 117-1470 px with a large along-track spread; a per-pass pixel bias is a first
order correction only. A proper fix is EO/OPK refinement (bundle adjustment against tie points from the
catalogue) and lens-distortion modelling.

## Grounding DINO false positives

Whole-image/tiled detection produced many background/field boxes. v0.14 significantly reduces this by
limiting search to local geometry ROIs, but structure detection still needs an adjudicated bbox/crop
benchmark across multiple structure types and projects.

## Crop resolution for FLI

Crops capped by `--crop-max-dimension` (1600 default, 2048 in runners) are upscaled ~3.5x by FLI v4.
Use `--crop-max-dimension 0` when crops feed FLI framing.

## CPU performance

DINOv3 and Grounding DINO can run on CPU but full-circuit runs are expensive. GPU batching and resumable
stage execution are roadmap items.

## Capture date configuration

A hard-coded `--capture-date` is unsafe for projects containing multiple acquisition dates. Prefer EXIF or
EO-source inference and use an explicit date only as a controlled override.
