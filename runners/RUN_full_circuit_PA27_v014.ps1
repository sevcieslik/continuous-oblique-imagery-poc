$ErrorActionPreference = "Stop"

$Root = $PSScriptRoot
$Circuit = "PA27"

# Re-use the Python environment that already worked for DINOv3.
$Python = "D:\Sev\Obliques\NYPA\.venv-v07\Scripts\python.exe"

$MainScript = Join-Path $Root "continuous_oblique_v014.py"
$Packager = Join-Path $Root "package_continuous_oblique_delivery_v01.py"
$Output = Join-Path $Root "_continuous_oblique_${Circuit}_full_v014"

Write-Host "============================================================"
Write-Host "Continuous Oblique v0.14 - full circuit"
Write-Host "============================================================"
Write-Host "Root:      $Root"
Write-Host "Circuit:   $Circuit"
Write-Host "Python:    $Python"
Write-Host "Main:      $MainScript"
Write-Host "Output:    $Output"
Write-Host "============================================================"

if (-not (Test-Path $Python)) { throw "Python environment not found: $Python" }
if (-not (Test-Path $MainScript)) { throw "Main script not found: $MainScript" }

# capture-date omitted intentionally: let the pipeline infer dates from source imagery.
# Structure class 215 is retained from the previous validated dataset.

& $Python $MainScript `
    $Root `
    --circuit $Circuit `
    --structure-class 215 `
    --render-mode full `
    --jpeg-quality 94 `
    --crop-max-dimension 2048 `
    --crop-qa-limit 40 `
    --render-other-views `
    --tower-detector-device cpu `
    --tower-dino-device cpu `
    --dino-device cpu `
    --output $Output

if ($LASTEXITCODE -ne 0) {
    throw "continuous_oblique_v014.py failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Main pipeline completed successfully."

if (Test-Path $Packager) {
    Write-Host ""
    Write-Host "Packaging deliverables..."

    & $Python $Packager `
        $Output `
        --circuit $Circuit `
        --output-version v1

    if ($LASTEXITCODE -ne 0) {
        throw "Delivery packager failed with exit code $LASTEXITCODE"
    }

    Write-Host "Packaging completed successfully."
}
else {
    Write-Warning "Packager not found: $Packager"
    Write-Warning "Skipping delivery packaging."
}

Write-Host ""
Write-Host "============================================================"
Write-Host "DONE"
Write-Host "Output: $Output"
Write-Host "============================================================"
