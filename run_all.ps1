# Run the complete pipeline (Phase 1 -> 6).
# Usage (from THIS folder):  .\run_all.ps1 [-Quick]
# If scripts are blocked:    powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Quick
param(
    [switch]$Quick
)

# "Continue" so Python warnings on stderr do not abort the script; failures are caught via exit codes below.
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot

if (Get-Command python -ErrorAction SilentlyContinue) { $PY = "python" }
elseif (Get-Command py -ErrorAction SilentlyContinue) { $PY = "py" }
else { Write-Host "Python was not found. Install Python 3.10+ from python.org (tick 'Add Python to PATH')." -ForegroundColor Red; exit 1 }

function Run-Step([string]$Title, [string[]]$PyArgs) {
    Write-Host "`n== $Title ==" -ForegroundColor Cyan
    & $PY @PyArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Title (exit code $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Run-Step "Install dependencies" @("-m", "pip", "install", "-r", "requirements.txt")
Run-Step "Phase 1: correctness tests" @("tests/run_tests.py")
if ($Quick) {
    Run-Step "Phase 2: baselines" @("scripts/01_baselines.py", "--quick")
    Run-Step "Phase 3: dataset" @("scripts/02_generate_dataset.py", "--games", "50", "--out", "data/positions_quick.csv")
    Run-Step "Phase 4: training" @("scripts/03_train_models.py", "--data", "data/positions_quick.csv")
    Run-Step "Phase 5+6: experiments" @("scripts/04_run_experiments.py", "--quick")
} else {
    Run-Step "Phase 2: baselines" @("scripts/01_baselines.py")
    Run-Step "Phase 3: dataset" @("scripts/02_generate_dataset.py", "--games", "600")
    Run-Step "Phase 4: training" @("scripts/03_train_models.py")
    Run-Step "Phase 5+6: experiments" @("scripts/04_run_experiments.py")
}
Run-Step "Plots" @("scripts/05_make_plots.py")

Write-Host "`nAll phases completed successfully!" -ForegroundColor Green
