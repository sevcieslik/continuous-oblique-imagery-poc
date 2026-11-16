# Continuous Oblique Imagery POC

Internal proof-of-concept for building and validating a continuous oblique imagery product from
raw oblique photography, camera EO, image footprints, PLS-CADD staking, classified LiDAR and
visual foundation models.

The current reference implementation is `src/continuous_oblique_v015.py` (with its helper module
`src/structure_detection_v015.py`). v0.15 adds competitor-aware structure detection and a
georeferenced **structure catalogue** (`src/build_structure_catalog_v01.py`) that feeds
[FLI_frame_maker](https://github.com/sevcieslik/FLI_frame_maker) and provides tie points for
georeference validation. See `docs/STRUCTURE_CATALOG.md`.

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
10. Detect structures for downstream crops using geometry-constrained Grounding DINO, with
    competitor-aware assignment (parallel circuits / neighbouring towers) and per-pass prior calibration.
11. Render annotated 163 candidates, QA CSV/GPKG outputs and structure crops.
12. Build the structure catalogue (FLI export, tie points, GeoJSON).
13. Optionally benchmark against an existing/manual 163 product.

## Current status

The DH1 validation dataset demonstrated:

- strong recovery of manual raw-frame selection among confidently mapped images;
- strong DINO continuity in the manual reference product;
- a recurring off-by-one span-label pattern, especially on backward travel;
- false `CROSSING_PASS` classifications in some frames used by the manual product;
- the need to audit manual production before using it as ground truth;
- substantially better tower localisation when Grounding DINO is constrained to a local geometry ROI;
- on parallel-circuit corridors (DH1/DH2 ~163 ft apart) the v0.14 nearest-to-prior assignment can crop
  the neighbouring circuit's tower; v0.15 resolves this with competitor hypotheses and registration.

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
python .\src\continuous_oblique_v015.py `
  "D:\path\to\project" `
  --circuit CIRCUIT_ID `
  --structure-class 215 `
  --render-mode full `
  --render-other-views `
  --competitor-structures "D:\path\to\00_structures_by_circuit\NEIGHBOUR_structures.csv" `
  --output "D:\path\to\project\_continuous_oblique_CIRCUIT_ID_full_v015"

python .\src\build_structure_catalog_v01.py "D:\path\to\project\_continuous_oblique_CIRCUIT_ID_full_v015" --circuit CIRCUIT_ID
```

Or run everything with `runners\RUN_full_circuit_v015.ps1 -Circuit CIRCUIT_ID`.

Tests (no model weights or project data needed):

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests
```

`215` is an empirical class used on one validation project. It is **not a universal structure-class
standard**; profile/confirm the classification schema on every new dataset.

## Repository layout

```text
src/       current pipeline, detection helpers, delivery packager, structure catalogue builder
tools/     benchmarking, manual-audit, adjudication and detector-evaluation utilities
tests/     pytest unit + synthetic integration tests
runners/   PowerShell runners/examples
docs/      product rules, architecture, findings and roadmap
samples/   anonymised schemas/examples only
archive/   historical POC versions and run notes
```

## Project philosophy

Fail closed for structure crops; fail open for legitimate sequence continuity.

A missed tower crop is preferable to saving a field/shadow as a tower. Conversely, a legal sequence
frame must not be removed simply because a detector or span-visibility test is uncertain.
