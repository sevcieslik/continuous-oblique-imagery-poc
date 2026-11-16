# Architecture

## Pipeline stages

### A. Reference geometry

- discover latest versioned input folders;
- parse PLS-CADD staking XML;
- determine working CRS;
- build physical PLS topology;
- derive canonical reference structures/spans;
- create circuit alignment and chainage;
- optionally profile LiDAR classes and structure heights.

### B. Image/EO/footprint model

- index raw imagery;
- parse EO records (`Image, X, Y, Z, O, P, K` variants);
- join DXF footprint polygons to source imagery;
- compute camera chainage and corridor distance;
- derive approximate OPK optical target when usable;
- keep geometry and optical reference candidates separately.

### C. Pass model

Pass segmentation is performed within each EO source. Current direction classification uses camera
motion projected onto the circuit alignment. `CROSSING_PASS` is used when along-circuit progression
is too small relative to flight motion.

This classifier is a known calibration area: the manual benchmark identified legitimate frames that
were classified as crossing/unknown on DH1.

### D. Sequence model

- build legal product images;
- stitch compatible pass fragments into circuit runs;
- calculate ordered sequence index per run and direction;
- calculate pass-specific coverage/gaps;
- preserve partial-span frames.

### E. Reference resolver

Reference assignment retains evidence such as:

- `primary_span_geometry`;
- `primary_span_optical`;
- `primary_span_method`;
- `reference_chainage`;
- `reference_assignment_status`;
- `reference_confidence`;
- `alternative_reference` near boundaries.

Backward-pass off-by-one errors observed in benchmarking show that this layer needs a direction-aware
resolver before production use.

### F. Visual continuity QA

DINOv3 features are extracted from adjacent logical product frames. Current metrics include global
similarity, dense patch similarity, reciprocal/mutual matches and spatial coherence. Geometry and
visual evidence are combined into review statuses without changing membership.

### G. Structure detection/crops

Current v0.15 flow (`render_candidate_products_and_crops_v15`, helpers in `structure_detection_v015.py`):

1. LiDAR/EO/footprint geometry predicts a source-pixel prior for every target structure in the frame.
2. The prior is shifted by the per-pass robust median of earlier confident detection offsets.
3. Competitor hypotheses are projected with the same camera model: LiDAR structure-class clusters near
   the target that are not the target, plus optional neighbour-circuit structures (FLI step-00 CSV).
4. Local ROIs are tiled at source resolution and Grounding DINO runs only there.
5. Tile-edge artefacts are rejected, then containment NMS (IoU or intersection-over-min) merges duplicates.
6. Full-image and ROI plausibility filters (as v0.14).
7. **Registration:** a single pixel translation aligning all hypotheses (targets + competitors) with the
   detections is estimated; it is accepted only when several hypotheses support it with a clear margin.
   This makes assignment invariant to EO offsets larger than half the tower spacing.
8. **Ownership:** each detection is softly owned by the nearest (registered) hypothesis. A target may
   only take detections it owns.
9. Scores combine ownership, detector confidence, absolute geometry and optional DINOv3 tracking.
   One-to-one assignment with explicit `AMBIGUOUS_ASSIGNMENT`, `REJECTED_COMPETITOR_OWNED`,
   `LOW_ASSIGNMENT_SCORE` statuses. DINO templates update only from `HIGH` confidence matches.
10. Camera-model quality, projected staking base and detection residuals are written per row.

Crop-only frames (crossing/side views) are processed when `--render-other-views` is set. Rows outside
`--render-limit` are `NOT_IN_RENDER_SUBSET`.

Previous v0.14 flow:

1. LiDAR/EO/footprint geometry predicts a source-pixel prior.
2. Build a local ROI around the prior.
3. Tile only the local ROI at source resolution.
4. Run Grounding DINO tower detection.
5. Filter boxes by full-image and ROI plausibility.
6. Apply a geometry association gate.
7. Score candidates using detection confidence + geometry distance (+ optional DINO tracking).
8. Assign detections one-to-one where multiple structures share an image.
9. Render crop only after an accepted visual detection.

No acceptable detection = no crop. The continuous frame remains untouched.

### H. Packaging

The delivery packager separates:

- annotated continuous 163 imagery;
- clean structure crops for ML/FLI use;
- tagged structure crops for human review;
- per-structure observations;
- global structure coverage;
- selected QA metadata.

### I. Structure catalogue

`build_structure_catalog_v01.py` turns accepted detections into a tagged, georeferenced catalogue:
observation and structure tables, GeoJSON, best-view ranking, per-pass residual consistency flags,
image↔world tie points and an FLI_frame_maker assignment CSV + EXIF-bearing images. It never changes
detection results. See `STRUCTURE_CATALOG.md`.

## High-level flow

```mermaid
flowchart LR
    A[Raw oblique JPGs] --> M[Image/EO/footprint model]
    B[Camera EO] --> M
    C[Footprint DXF] --> M
    D[PLS staking] --> R[Reference topology + chainage]
    E[Classified LiDAR] --> R
    R --> M
    M --> P[Pass segmentation + direction]
    P --> S[Legal sequence / circuit runs]
    S --> L[Reference assignment]
    S --> Q[DINOv3 continuity QA]
    L --> O[Annotated 163 product]
    M --> T[Geometry ROI]
    E --> T
    T --> G[Grounding DINO structure detection]
    G --> CROP[Structure crops + observations]
    CROP --> CAT[Structure catalogue]
    CAT --> FLI[FLI_frame_maker]
    CAT --> GV[Georeference validation tie points]
    Q --> QA[QA / review queue]
    L --> QA
    P --> QA
```
