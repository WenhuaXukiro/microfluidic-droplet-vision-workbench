@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
py -3 -V >nul 2>&1
if errorlevel 1 (
  python -m venv .venv
) else (
  py -3 -m venv .venv
)
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip check
if errorlevel 1 goto fail
.venv\Scripts\python.exe download_pretrained_weights.py
if errorlevel 1 goto fail
echo Training environment installed. Restart the workbench.
pause
exit /b 0
:fail
echo Installation failed. Check Python installation and internet access.
pause
exit /b 1
