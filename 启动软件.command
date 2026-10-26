#!/bin/zsh
set -u
cd "$(dirname "$0")" || exit 1
export PYTHONUTF8=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
if [[ -x .venv/bin/python ]]; then
  lab_python=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  lab_python="$(command -v python3)"
else
  print "请先安装 Python 3.13，或查看 解压后先读.md。"
  read "?按回车关闭…"
  exit 1
fi
print "工作台地址：http://127.0.0.1:8765/"
print "请保留此窗口。按 Control+C 停止工作台。"
(sleep 1; open "http://127.0.0.1:8765/") &
"$lab_python" -X utf8 server.py
lab_status=$?
if [[ $lab_status -ne 0 ]]; then
  print "启动失败。若端口8765已占用，请先关闭旧工作台；其他问题见上方提示。"
  read "?按回车关闭…"
fi
exit $lab_status
