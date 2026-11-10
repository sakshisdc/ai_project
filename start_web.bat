@echo off
rem Double-click to start the web demo (Windows). Opens http://127.0.0.1:8000
cd /d "%~dp0"
where python >nul 2>nul && (set PY=python) || (set PY=py)
%PY% -c "import numpy, pandas, sklearn, scipy, joblib" 2>nul || %PY% -m pip install -r requirements.txt
%PY% serve.py
pause
