#!/bin/zsh
set -eu
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  print "请先运行 安装训练环境.command。"
  exit 1
fi
.venv/bin/python -c 'import sys,torch,torchvision,cv2,ultralytics; print("Python:",sys.version); print("PyTorch:",torch.__version__); print("TorchVision:",torchvision.__version__); print("OpenCV:",cv2.__version__); print("Ultralytics:",ultralytics.__version__); print("Apple GPU:",torch.backends.mps.is_available()); print("NVIDIA GPU:",torch.cuda.is_available())'
.venv/bin/python -m pip check
read "?按回车关闭…"
