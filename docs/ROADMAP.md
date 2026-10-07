# Roadmap

## v0.15 - reference and pass logic

Highest priority because the benchmark indicates these are larger errors than raw-frame selection.

- direction-aware span resolver, especially for backward travel;
- explicit incoming/outgoing tangent handling at sharp structures;
- replace hard pass classification thresholds with local evidence and confidence;
- audit frames currently called `CROSSING_PASS`/`UNKNOWN_PASS` against coherent manual sequences;
- make pass acceptance capable of retaining a legal circuit fragment even when a short local segment
  has low along-track progression;
- preserve an adjudication trace for every reference/pass decision.

## v0.16 - multi-project calibration

- validate DH1 learnings on PA27 and at least one additional circuit;
- calculate per-project and pooled metrics;
- establish project-agnostic thresholds vs project-specific configuration;
- formalise structure-class profiling instead of carrying class 215 between datasets;
- improve coordinate-system diagnostics and fail early on incompatible EO.

## v0.17 - structure detection/crops

- benchmark Grounding DINO crop precision against manually reviewed structure boxes/crops;
- add structure-type-aware prompts and thresholds;
- evaluate larger Grounding DINO checkpoint / GPU batching where useful;
- use DINOv3 temporal/sequence tracking to stabilise tower identity without overriding geometry;
- preserve multiple valid views and add `is_best_view` ranking as metadata only.

## v0.18 - evaluation harness

- create a versioned adjudicated ground-truth package;
- unit tests for span boundary logic, direction handling and gap classification;
- regression tests against small anonymised fixtures;
- benchmark metrics: membership precision/recall, span exact/adjacent accuracy, directional coverage,
  continuity outliers, crop detection precision/recall;
- compare each release automatically against frozen benchmark outputs.

## v0.19 - performance and packaging

- configurable YAML/TOML project profile;
- GPU/CPU execution profiles;
- batched DINO/Grounding DINO inference;
- resumable stages and robust cache versioning;
- structured logging and run manifest with code/config/model hashes;
- consistent CLI entry point instead of versioned monolithic scripts.

## v1.0 - production candidate

- tested multi-project configuration;
- formal schemas for inputs/outputs;
- deterministic and reproducible run manifests;
- CI static checks + regression fixtures;
- clear operator review queue instead of implicit automatic decisions;
- delivery packager with configurable product conventions;
- documented failure/recovery process.

## Separate downstream track: FLI comparison

FLI visual comparison should remain a separate module from initial 163 construction. It should consume
structure crops/observations and rank/compare views without coupling FLI decisions back into product
membership.
