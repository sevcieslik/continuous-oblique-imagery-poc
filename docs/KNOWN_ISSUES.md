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

## Grounding DINO false positives

Whole-image/tiled detection produced many background/field boxes. v0.14 significantly reduces this by
limiting search to local geometry ROIs, but structure detection still needs an adjudicated bbox/crop
benchmark across multiple structure types and projects.

## CPU performance

DINOv3 and Grounding DINO can run on CPU but full-circuit runs are expensive. GPU batching and resumable
stage execution are roadmap items.

## Capture date configuration

A hard-coded `--capture-date` is unsafe for projects containing multiple acquisition dates. Prefer EXIF or
EO-source inference and use an explicit date only as a controlled override.
