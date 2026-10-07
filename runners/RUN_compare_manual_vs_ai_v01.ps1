param(
    [string]$ProjectRoot = "D:\Sev\Obliques\DH1",
    [int]$ReviewLimit = 150
)

$ErrorActionPreference = "Stop"

$venvPython = Join-Path $ProjectRoot ".venv-v07\Scripts\python.exe"
if (Test-Path $venvPython) {
    $python = $venvPython
} else {
    $python = "python"
}

$script = Join-Path $ProjectRoot "compare_manual_vs_ai_v01.py"
if (-not (Test-Path $script)) {
    throw "Script not found: $script"
}

& $python $script $ProjectRoot `
    --review-limit $ReviewLimit `
    --output (Join-Path $ProjectRoot "_benchmark_manual_vs_ai_v01")

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
