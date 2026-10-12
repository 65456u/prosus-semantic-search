#!/bin/zsh
cd -- "${0:A:h}" || exit 1
python3 launch.py stop
result=$?
if (( result != 0 )); then
  read "reply?请阅读上面的提示，按回车关闭窗口。"
fi
exit "$result"
