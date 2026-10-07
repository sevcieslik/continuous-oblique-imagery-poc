# Operating guide

## Environment

The known-good Windows CPU combination during development was:

```text
Python 3.10 environment
torch 2.6.0+cpu
torchvision 0.21.0+cpu
transformers 5.17.0
```

Use the same interpreter for the pipeline and all audit tools. A common failure mode was launching the
base Conda Python, where `torch` failed to load `c10.dll`.

## First run on a new dataset

1. Confirm the standard input folders and the footprint DXF.
2. Run a geometry-only/small render smoke test.
3. Check working CRS and EO spatial compatibility.
4. Inspect reference structure/span counts.
5. Profile LiDAR classes before pinning a structure class.
6. Review pass segmentation and direction counts.
7. Run 3-10 structure detections before a full-circuit detector run.
8. Only then run the full-resolution package.

## Suggested smoke command

```powershell
python .\src\continuous_oblique_v014.py `
  "D:\path\project" `
  --circuit CIRCUIT `
  --structure-class 215 `
  --render-limit 3 `
  --crop-qa-limit 12 `
  --skip-dino `
  --skip-tower-dino `
  --output "D:\path\project\_continuous_oblique_smoke"
```

`215` is only an example from DH1. Change it when the new point-cloud schema differs.

## Full run

Use `runners/RUN_full_circuit_portable_v014.ps1` or adapt one of the project-specific runners.

## Post-run checks

Always inspect:

- console counts for raw / EO / structures / reference spans / passes / circuit runs;
- `eo_sources.csv` for incompatible EO sources;
- `flight_passes.csv` for suspicious `CROSSING_PASS` blocks;
- `direction_coverage.csv` and gaps independently by direction;
- `qa/visual_continuity` for DINO outliers;
- `qa/structure_detection` for ROI and selected tower detections;
- `product_manifest.csv` for reference confidence/boundary ambiguity;
- `structure_crop_candidates.csv` for no-detection and detector-quality patterns.

## Benchmarking an existing/manual product

Run, in order:

1. `compare_manual_vs_ai_v01.py`
2. `audit_manual_163_v03.py`
3. `adjudicate_manual_vs_ai_v01.py`

Do not use v0.1/v0.2 manual-audit duplicate counts for calibration; v0.3 corrected the false duplicate
logic by separating local high overlap from untrusted raw-mapping collisions.
