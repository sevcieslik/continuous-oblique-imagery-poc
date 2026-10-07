# Data contract

## Expected input folders

The pipeline discovers the highest `vN` directory under the standard product folders.

| Product | Purpose |
|---|---|
| `130_raw_overlapping_oblique_imagery` | source JPGs |
| `163_continuous_oblique_imagery` | optional existing/manual benchmark |
| `277_camera_eo` | camera exterior orientation |
| `30_classified_tiled_point_cloud` | classified LiDAR |
| `40_pls_cadd_staking_table` | PLS-CADD staking XML |
| `*_footprints*.dxf` | image footprint polygons/labels |

## Key image fields

Recommended stable concepts for downstream consumers:

| Field | Meaning |
|---|---|
| `filename` | raw source image name |
| `pass_id` | segmented acquisition pass |
| `pass_direction` | forward/backward/crossing/unknown acquisition direction |
| `circuit_run_id` | stitched logical circuit run |
| `reference_chainage` | target chainage used for reference assignment |
| `primary_span` | selected span label |
| `reference_id` | product label entity (span or structure) |
| `view_direction` | target-relative camera view |
| `valid_for_product` / `sequence_keep` | canonical sequence membership |
| `rejection_reason` | reason excluded from canonical sequence |
| `reference_confidence` | confidence in selected reference |
| `alternative_reference` | adjacent plausible reference near a boundary |

## Useful structure-crop fields

- `structure_id`
- `structure_idx`
- `crop_role`
- `crop_bucket`
- `tower_detector_status`
- `tower_detector_score`
- `tower_detector_final_score`
- `tower_dino_similarity`
- `geometry_prior_distance_px`
- `tower_search_mode`
- `tower_*bbox*`
- `crop_path`

## Output policy

CSV and GPKG outputs are QA/audit artefacts. Rendered images are derived products. Do not infer that
a row labelled `REVIEW` or `POSSIBLE_*` is automatically wrong; several fields intentionally expose
uncertainty rather than hiding it.
