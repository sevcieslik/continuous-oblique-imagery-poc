# Validation and benchmarking

## Why the existing/manual product is not treated as absolute truth

The manual 163 product is useful because it captures human production decisions, but it can contain
missing spans, inconsistent labels, sequence choices that are hard to map back to raw imagery, and
historical workflow compromises. The benchmark therefore has two stages:

1. audit the manual product internally;
2. compare AI to the trusted/auditable manual subset.

## DH1 AI vs manual benchmark

Observed comparison:

- manual images: **400**;
- AI images: **252**;
- manual unique spans: **289**;
- AI unique spans: **212**;
- confidently comparable manual source matches: **264**;
- same raw selected by AI: **249 / 264 = 94.32%**;
- exact span when the same raw was selected: **226 / 249 = 90.76%**;
- common spans: **208**;
- manual-only spans: **81**;
- AI-only spans: **4**.

Interpretation: raw-frame selection was already strong on the comparable subset. The larger remaining
issues were sequence coverage/pass classification and reference attribution, not primarily image
selection.

## Manual-product DINO/geometry audit v0.3

The independent manual audit produced:

- 400 manual images;
- 250 geometry-auditable images;
- 248 geometry transitions;
- 241 `CONTINUOUS` transitions;
- 6 `VISUAL_REVIEW` transitions;
- 1 `GEOMETRY_JUMP_VISUAL_OK` transition;
- 236 Grade-A calibration images (59%);
- 14 Grade-B calibration images;
- 243 expected high-overlap pairs;
- 0 true duplicate suspects;
- 0 label suspects from the DINO test;
- 9 spans missing from the manual product that are present in the reference topology.

The 136 LOW/unresolved manual-to-raw matches are not considered manual-product failures. They indicate
that the appearance fingerprint benchmark cannot confidently recover the original raw frame.

## Important residual disagreement

Within the clean Grade-A subset, reference disagreements were dominated by a systematic one-span
shift, particularly on backward travel. This supports a direction-aware span resolver rather than a
simple global offset or blind tuning to the manual labels.

## Validation tooling

- `tools/compare_manual_vs_ai_v01.py`: source selection and span attribution comparison;
- `tools/audit_manual_163_v03.py`: independent DINO + geometry audit of the manual product;
- `tools/adjudicate_manual_vs_ai_v01.py`: focused human adjudication pack for clean span disagreements
  and possible pass-classifier conflicts.
