@echo off
rem Run the complete pipeline (Phase 1 -> 6).  Usage (from THIS folder): run_all.bat [--quick]
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  where py >nul 2>nul
  if errorlevel 1 ( echo Python not found. Install Python 3.10+ and tick "Add Python to PATH". & exit /b 1 )
  set PY=py
) else ( set PY=python )

set QUICK=0
if "%1"=="--quick" set QUICK=1

echo == Install dependencies ==
%PY% -m pip install -r requirements.txt || exit /b 1
echo == Phase 1: tests ==
%PY% tests/run_tests.py || exit /b 1

if "%QUICK%"=="1" (
  echo == Phase 2: baselines ==
  %PY% scripts/01_baselines.py --quick || exit /b 1
  echo == Phase 3: dataset ==
  %PY% scripts/02_generate_dataset.py --games 50 --out data/positions_quick.csv || exit /b 1
  echo == Phase 4: training ==
  %PY% scripts/03_train_models.py --data data/positions_quick.csv || exit /b 1
  echo == Phase 5+6: experiments ==
  %PY% scripts/04_run_experiments.py --quick || exit /b 1
) else (
  echo == Phase 2: baselines ==
  %PY% scripts/01_baselines.py || exit /b 1
  echo == Phase 3: dataset ==
  %PY% scripts/02_generate_dataset.py --games 600 || exit /b 1
  echo == Phase 4: training ==
  %PY% scripts/03_train_models.py || exit /b 1
  echo == Phase 5+6: experiments ==
  %PY% scripts/04_run_experiments.py || exit /b 1
)
echo == Plots ==
%PY% scripts/05_make_plots.py || exit /b 1
echo All phases completed successfully!
