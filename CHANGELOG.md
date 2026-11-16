# Changelog

## v0.15 - structure detection + catalogue (2026-11, branch feature/structure-catalog-v015)

Pipeline `src/continuous_oblique_v015.py` (sequence/span logic unchanged from v0.14):

- competitor-aware tower assignment: LiDAR structure-class clusters and optional neighbour-circuit
  structures (`--competitor-structures`, FLI step-00 CSV) are projected as competing hypotheses;
- per-image registration of all hypotheses to detections (`--registration-*`), invariant to EO offsets
  larger than half the tower spacing;
- per-pass online prior-bias calibration (`--bias-*`);
- tile-edge artefact rejection (`--tower-tile-artifact-fraction`) and containment NMS (`--tower-nms-iomin`);
- explicit `AMBIGUOUS_ASSIGNMENT` / `REJECTED_COMPETITOR_OWNED` / `LOW_ASSIGNMENT_SCORE` statuses and
  `HIGH`/`MEDIUM` confidence; DINO templates updated from `HIGH` matches only;
- camera-model quality, projected staking base and residuals persisted (were computed but dropped);
- crop-only frames processed with `--render-other-views` (v0.14 skipped frames outside the manifest);
- `NOT_IN_RENDER_SUBSET` status; QA overlays split into `ok/`, `review/`, `no_crop/` with a separate
  review budget (`--crop-qa-review-limit`); `summary.json` `poc_version` fixed.

New:

- `src/structure_detection_v015.py` - torch-free detection helpers;
- `src/build_structure_catalog_v01.py` - georeferenced structure catalogue, FLI_frame_maker export with
  lossless EXIF insertion, tie points, residual tables, GeoJSON, per-pass residual outlier flags;
- `tools/evaluate_structure_detector_v01.py` - detector metrics and adjudication pack;
- `runners/RUN_full_circuit_v015.ps1`; `tests/` (pytest); `requirements-dev.txt`; `.gitignore`;
- `docs/STRUCTURE_CATALOG.md`.

Changed:

- delivery packager refuses to delete a non-empty folder that is not a previous delivery package;
- `src/continuous_oblique_v014.py` moved out of `src/` (identical copy already in `archive/`).


## Repository baseline - 2026-10

- consolidated continuous-oblique POC v0.1-v0.14;
- added current v0.14 delivery packager;
- added manual-vs-AI benchmark tool;
- added manual-product DINO/geometry audit v0.1-v0.3;
- added focused adjudication tool;
- documented product semantics, architecture, validation findings and roadmap;
- added anonymised sample schemas and safe default `.gitignore`.
