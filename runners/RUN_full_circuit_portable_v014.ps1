param(
    [Parameter(Mandatory=$true)]
    [string]$Circuit,

    [string]$ProjectRoot = $PSScriptRoot,
    [string]$Python = "python",
    [int]$StructureClass = 215,
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $ProjectRoot "_continuous_oblique_${Circuit}_full_v014"
}

$MainScript = Join-Path $ProjectRoot "continuous_oblique_v014.py"
$Packager = Join-Path $ProjectRoot "package_continuous_oblique_delivery_v01.py"

if (-not (Test-Path $MainScript)) {
    # Repository layout fallback.
    $MainScript = Join-Path $PSScriptRoot "..\src\continuous_oblique_v014.py"
}
if (-not (Test-Path $Packager)) {
    $Packager = Join-Path $PSScriptRoot "..\src\package_continuous_oblique_delivery_v01.py"
}

Write-Host "Project root: $ProjectRoot"
Write-Host "Circuit:      $Circuit"
Write-Host "Python:       $Python"
Write-Host "Output:       $Output"
Write-Host "Class:        $StructureClass"

& $Python $MainScript `
    $ProjectRoot `
    --circuit $Circuit `
    --structure-class $StructureClass `
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

if (Test-Path $Packager) {
    & $Python $Packager $Output --circuit $Circuit --output-version v1
    if ($LASTEXITCODE -ne 0) {
        throw "Delivery packager failed with exit code $LASTEXITCODE"
    }
}
