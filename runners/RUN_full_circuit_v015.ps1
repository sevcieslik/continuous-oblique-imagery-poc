param(
    [Parameter(Mandatory=$true)]
    [string]$Circuit,

    [string]$ProjectRoot = $PSScriptRoot,
    [string]$Python = "python",
    [int]$StructureClass = 215,
    [string]$CaptureDate = "",
    # Optional: neighbour-circuit structures (FLI_frame_maker step-00 *_structures.csv, ';'-separated)
    [string]$CompetitorStructures = "",
    # 0 = keep crops at source resolution (recommended when crops feed FLI framing)
    [int]$CropMaxDimension = 0,
    [string]$Output = "",
    [string]$RawDir = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $ProjectRoot "_continuous_oblique_${Circuit}_full_v015"
}

function Find-Script([string]$name) {
    $local = Join-Path $ProjectRoot $name
    if (Test-Path $local) { return $local }
    $repo = Join-Path $PSScriptRoot "..\src\$name"
    if (Test-Path $repo) { return (Resolve-Path $repo).Path }
    throw "Script not found: $name"
}

$MainScript = Find-Script "continuous_oblique_v015.py"
$null       = Find-Script "structure_detection_v015.py"   # must sit next to the main script
$Packager   = Find-Script "package_continuous_oblique_delivery_v01.py"
$Catalog    = Find-Script "build_structure_catalog_v01.py"

Write-Host "Project root: $ProjectRoot"
Write-Host "Circuit:      $Circuit"
Write-Host "Output:       $Output"
Write-Host "Competitors:  $CompetitorStructures"

$mainArgs = @(
    $MainScript, $ProjectRoot,
    "--circuit", $Circuit,
    "--structure-class", $StructureClass,
    "--render-mode", "full",
    "--jpeg-quality", "94",
    "--crop-max-dimension", $CropMaxDimension,
    "--crop-qa-limit", "60",
    "--crop-qa-review-limit", "300",
    "--render-other-views",
    "--tower-detector-device", "cpu",
    "--tower-dino-device", "cpu",
    "--dino-device", "cpu",
    "--output", $Output
)
if ($CaptureDate) { $mainArgs += @("--capture-date", $CaptureDate) }
if ($CompetitorStructures) { $mainArgs += @("--competitor-structures", $CompetitorStructures) }

Write-Host "=== 1/4 Continuous-oblique run (v0.15) ==="
& $Python @mainArgs
if ($LASTEXITCODE -ne 0) { throw "continuous_oblique_v015.py failed with exit code $LASTEXITCODE" }

Write-Host "=== 2/4 Delivery package ==="
& $Python $Packager $Output --circuit $Circuit --output-version v1
if ($LASTEXITCODE -ne 0) { throw "Delivery packager failed with exit code $LASTEXITCODE" }

Write-Host "=== 3/4 Structure catalogue (FLI export + georef tie points) ==="
$catArgs = @($Catalog, $Output, "--circuit", $Circuit, "--min-confidence", "MEDIUM")
if ($RawDir) { $catArgs += @("--raw-dir", $RawDir) }
& $Python @catArgs
if ($LASTEXITCODE -ne 0) { throw "Catalogue builder failed with exit code $LASTEXITCODE" }

Write-Host "=== 4/4 Detector evaluation pack ==="
$Eval = Join-Path $PSScriptRoot "..\tools\evaluate_structure_detector_v01.py"
if (Test-Path $Eval) {
    & $Python $Eval $Output
} else {
    Write-Host "evaluate_structure_detector_v01.py not found next to runners; skipped."
}

Write-Host ""
Write-Host "DONE"
Write-Host "Catalogue: $Output\structure_catalog"
Write-Host "FLI CSV:   $Output\structure_catalog\fli\structure_photo_match_qa.csv"
Write-Host "FLI imgs:  $Output\structure_catalog\fli\images"
Write-Host "Review:    $Output\qa\structure_detection\review  and  $Output\_detector_eval\adjudication.html"
