#!/bin/zsh
cd -- "${0:A:h}" || exit 1
if command -v python3 >/dev/null 2>&1; then
  python3 launch.py start --open
else
  print "未找到 Python 3。请安装 Python 3 后重试。"
  read "reply?按回车关闭窗口。"
  exit 1
fi
result=$?
if (( result != 0 )); then
  read "reply?启动失败。请阅读上面的提示，按回车关闭窗口。"
fi
exit "$result"
