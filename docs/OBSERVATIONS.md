# Engineering observations

This file captures the main lessons from iterative development and review.

## Product semantics

1. Continuous coverage is more important than complete-span visibility in every individual frame.
2. Partial-span frames can be exactly the frames that bridge a visual sequence.
3. Two directional sequences can both be legitimate; aggregate coverage can hide directional gaps.
4. Redundancy is not automatically an error in high-overlap acquisition.
5. A structure-crop failure must not change 163 membership.

## Reference topology

- Physical PLS staking rows are not always equivalent to product/reference structures.
- On DH1, tow ordinal information was required to collapse intermediate staking rows into the intended
  reference topology.
- A valid topology should be explicit, monotonic and auditable rather than inferred from row count.

## CRS / EO

- EO sources must be checked for spatial compatibility before use.
- One DH1 EO source was hundreds of thousands of project units away from the alignment and was correctly
  treated as incompatible; the other aligned within metres/feet-scale proximity.
- Do not merge directional coverage from incompatible coordinate sources just to increase counts.

## LiDAR class profiling

Automatic selection can fail badly if a non-structure class has unrealistic vertical statistics.
On DH1, class 7 produced implausible height/range values while class 215 had repeatable tower-like
coverage and dimensions. Class 215 was therefore pinned for that project.

**This is project-specific empirical evidence, not a standard class definition.**

## Span visibility

v0.6 temporarily used complete-span visibility as a membership gate. Review showed that this violated
the product definition. v0.6.1 corrected the rule: span visibility is QA only.

## DINOv3 continuity

- DINO is useful for detecting visual outliers, view changes and suspicious jumps.
- Very high similarity is common and expected for adjacent high-overlap frames; it must not be treated
  as a duplicate by itself.
- A duplicate flag needs sequence/chainage context and trustworthy source mapping.
- Manual-product audit showed 243 expected high-overlap pairs and no strong duplicate suspects after
  correcting the duplicate logic.

## Grounding DINO structure detection

Whole-image/tiled detection generated many false positives in fields/background. Geometry is therefore
best used as a **search prior** rather than as a final crop itself:

- geometry-only crop projection was not reliably pixel-accurate;
- full-image Grounding DINO was noisy;
- local geometry ROI + detector + geometry gate materially improved tower selection;
- if no credible detection exists, write no crop.

## Manual ground truth

The manual product is valuable but not perfect:

- it contains spans absent from the AI and also misses some topology spans;
- some manual frames map back to raw with low confidence;
- some frames used manually are classified by the current model as crossing/unknown;
- clean reference disagreements show a strong backward off-by-one pattern.

Therefore calibration should use audited Grade A/B subsets plus explicit adjudication, not all manual
labels blindly.

## Capture date

Do not force one `--capture-date` for a dataset that contains multiple acquisition dates. Prefer raw
EXIF, then EO-source mode, and use an explicit date only when independent metadata is unavailable.
