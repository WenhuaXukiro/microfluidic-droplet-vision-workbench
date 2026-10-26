@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run 02_Install_Training_Windows.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -c "import sys,torch,ultralytics; print('Python:',sys.version); print('PyTorch:',torch.__version__); print('Ultralytics:',ultralytics.__version__); print('NVIDIA CUDA available:',torch.cuda.is_available())"
pause
