# Development history

The project evolved through rapid POC iterations. Historical scripts remain under `archive/` for
traceability; new work should branch from v0.14 rather than resurrect an earlier version.

| Version | Main change |
|---|---|
| v0.1 | project discovery, PLS/EO/footprint geometry, basic pass/reference/coverage QA |
| v0.2 | circuit runs and manual-final-to-raw appearance benchmark |
| v0.3 | approximate OPK optical target, missing-footprint recovery, richer gap classification |
| v0.4 | canonical reference topology separated from physical PLS staking rows |
| v0.5 | rendered candidate 163 product, reference confidence, initial structure crops |
| v0.6 | experimental complete-span visibility gate (later rejected) |
| v0.6.1 | corrected product semantics: partial/no-complete-span frames remain in sequence |
| v0.7 | DINOv3 visual continuity QA; QA never changes membership |
| v0.8 | footprint-constrained homography structure projection |
| v0.8.1 | corrected reference-plane projection consistency |
| v0.9 | LiDAR-driven 3-D structure projection and class profiling |
| v0.9.1 | v0.9 bugfix release |
| v0.10 | visual tower detector (OWL-ViT) with geometry as broad prior |
| v0.11 | source-resolution tiled Grounding DINO detector |
| v0.12 | detector diagnostics/fallback and safer LiDAR auto-class profiling |
| v0.13 | plausibility filtering + geometry-gated one-to-one detector assignment |
| v0.14 | local geometry ROI Grounding DINO; detector no longer searches the whole image |

### Important reversals

The most important design correction was v0.6 -> v0.6.1. Complete-span visibility is not a valid
product membership rule. It remains useful QA metadata only.
