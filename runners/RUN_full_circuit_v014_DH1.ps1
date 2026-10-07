$ErrorActionPreference = "Stop"

$Root = "D:\Sev\Obliques\NYPA"
$Python = Join-Path $Root ".venv-v07\Scripts\python.exe"
$MainScript = Join-Path $Root "continuous_oblique_v014.py"
$Packager = Join-Path $Root "package_continuous_oblique_delivery_v01.py"
$Output = Join-Path $Root "_continuous_oblique_DH1_full_v014"

if (-not (Test-Path $Python)) { throw "Python venv not found: $Python" }
if (-not (Test-Path $MainScript)) { throw "Main script not found: $MainScript" }
if (-not (Test-Path $Packager)) { throw "Packager not found: $Packager" }

Write-Host "=== 1/2 Full DH1 continuous-oblique run ==="
& $Python $MainScript `
  $Root `
  --circuit DH1 `
  --capture-date 042226 `
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

if ($LASTEXITCODE -ne 0) { throw "continuous_oblique_v014.py failed with exit code $LASTEXITCODE" }

Write-Host "=== 2/2 Building delivery package ==="
& $Python $Packager `
  $Output `
  --circuit DH1 `
  --output-version v1

if ($LASTEXITCODE -ne 0) { throw "Packaging failed with exit code $LASTEXITCODE" }

Write-Host ""
Write-Host "DONE"
Write-Host "Delivery: $Output\deliverable"
Write-Host "163:      $Output\deliverable\163_continuous_oblique_imagery\v1"
Write-Host "Crops:    $Output\deliverable\structure_crops\by_structure"
Write-Host "ZIP:      $Output\deliverable\DH1_structure_crops_v1.zip"
