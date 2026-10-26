#!/bin/zsh
set -eu
cd "$(dirname "$0")"
export PYTHONUTF8=1
if ! command -v python3 >/dev/null 2>&1; then
  print "请先安装 python.org 的 Python 3.12 或 3.13（macOS universal2）。"
  exit 1
fi
python3 -c 'import platform,sys; assert sys.platform == "darwin", "此脚本用于macOS"; assert platform.machine() == "arm64", "请使用原生Apple芯片Python，不要通过Rosetta运行"; assert int(platform.mac_ver()[0].split(".")[0]) >= 14, "此环境需要macOS 14或以上"; assert (3,12) <= sys.version_info[:2] <= (3,13), "请安装Python 3.12或3.13"'
trap 'print "安装未完成，请检查上方提示、Python版本和网络。"' ZERR
if [[ ! -x .venv/bin/python ]]; then
  if ! command -v python3 >/dev/null 2>&1; then
    print "请先安装 Python 3.13，或查看 解压后先读.md。"
    exit 1
  fi
  python3 -c 'import sys; assert sys.version_info >= (3, 12), "请安装Python 3.12或以上，建议3.13"'
  python3 -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-macos.txt
.venv/bin/python -m pip check
.venv/bin/python download_pretrained_weights.py
print "训练环境安装完成。请重新启动工作台。"
read "?按回车关闭…"
