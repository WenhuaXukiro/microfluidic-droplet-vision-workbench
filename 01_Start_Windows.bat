@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  set "LAB_PYTHON=.venv\Scripts\python.exe"
) else (
  py -3 -V >nul 2>&1
  if errorlevel 1 (
    python -V >nul 2>&1
    if errorlevel 1 (
      echo Install Python 3.12 from python.org first. See Windows_README.md
      pause
      exit /b 1
    )
    set "LAB_PYTHON=python"
  ) else (
    set "LAB_PYTHON=py -3"
  )
)
start "" "http://127.0.0.1:8765/"
echo Keep this window open. Press Ctrl+C to stop the workbench.
%LAB_PYTHON% -X utf8 server.py
pause
