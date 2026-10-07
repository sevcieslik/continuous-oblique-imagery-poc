# Continuous Oblique Imagery POC

Internal proof-of-concept for building and validating a continuous oblique imagery product from
raw oblique photography, camera EO, image footprints, PLS-CADD staking, classified LiDAR and
visual foundation models.

The current reference implementation is `src/continuous_oblique_v014.py`.

## Core product definition

A continuous-oblique product is a **series of legitimate acquisition frames covering the circuit**.
It is not a collection of one "best" image per span. A frame remains part of the legal sequence
when it preserves continuity even if only one structure, part of a span, or no complete span is
visible in that frame.

Product membership and QA are deliberately separated:

- sequence membership is based on acquisition geometry, circuit alignment, pass direction and AOI;
- span/structure labels are metadata attached to retained frames;
- DINOv3 continuity, duplicate signals and span visibility are QA evidence only;
- structure-crop detection never removes an image from product 163;
- the historical/manual 163 product is a benchmark, not assumed to be perfect ground truth.

## Inputs

Typical project root:

```text
<ProjectRoot>/
├── 130_raw_overlapping_oblique_imagery/
├── 163_continuous_oblique_imagery/      # optional manual benchmark
├── 277_camera_eo/
├── 30_classified_tiled_point_cloud/
├── 40_pls_cadd_staking_table/
└── *_footprints*.dxf
```

## Main workflow

1. Parse PLS-CADD staking and build a canonical reference topology.
2. Build circuit alignment and chainage.
3. Join raw imagery, EO and footprint metadata.
4. Segment acquisition into flight passes.
5. Classify `FORWARD_PASS`, `BACKWARD_PASS`, `CROSSING_PASS`, `UNKNOWN_PASS`.
6. Build legal continuous sequences independently by direction.
7. Assign span/structure references without using visibility as a deletion gate.
8. Check pass-specific geometric continuity and gaps.
9. Run DINOv3 visual-continuity QA between adjacent legal frames.
10. Detect structures for downstream crops using geometry-constrained Grounding DINO.
11. Render annotated 163 candidates, QA CSV/GPKG outputs and structure crops.
12. Optionally benchmark against an existing/manual 163 product.

## Current status

The DH1 validation dataset demonstrated:

- strong recovery of manual raw-frame selection among confidently mapped images;
- strong DINO continuity in the manual reference product;
- a recurring off-by-one span-label pattern, especially on backward travel;
- false `CROSSING_PASS` classifications in some frames used by the manual product;
- the need to audit manual production before using it as ground truth;
- substantially better tower localisation when Grounding DINO is constrained to a local geometry ROI.

See `docs/OBSERVATIONS.md` and `docs/VALIDATION.md` for measured results and caveats.

## Quick start

Install the tested CPU environment:

```powershell
python -m pip install -r requirements.txt
```

DINOv3 requires access to the gated Meta checkpoint on Hugging Face:

```text
facebook/dinov3-vits16-pretrain-lvd1689m
```

Example:

```powershell
python .\src\continuous_oblique_v014.py `
  "D:\path\to\project" `
  --circuit CIRCUIT_ID `
  --structure-class 215 `
  --render-mode full `
  --output "D:\path\to\project\_continuous_oblique_CIRCUIT_ID_full_v014"
```

`215` is an empirical class used on one validation project. It is **not a universal structure-class
standard**; profile/confirm the classification schema on every new dataset.

## Repository layout

```text
src/       current pipeline and delivery packager
tools/     benchmarking, manual-audit and adjudication utilities
runners/   PowerShell runners/examples
docs/      product rules, architecture, findings and roadmap
samples/   anonymised schemas/examples only
archive/   historical POC versions and run notes
```

## Project philosophy

Fail closed for structure crops; fail open for legitimate sequence continuity.

A missed tower crop is preferable to saving a field/shadow as a tower. Conversely, a legal sequence
frame must not be removed simply because a detector or span-visibility test is uncertain.
