"""Start and stop the local explorer independently of a chat or terminal window."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser

HERE = Path(__file__).resolve().parent
PORT = 8765
URL = f"http://127.0.0.1:{PORT}/"
RUNTIME = HERE / ".runtime"
PID_FILE = RUNTIME / "server.json"
LOG_FILE = RUNTIME / "server.log"


def is_explorer_running():
    try:
        with urlopen(URL, timeout=1) as response:
            body = response.read(16000).decode("utf-8", errors="replace")
            return response.status == 200 and "<title>Prosus · 数据浏览器</title>" in body
    except (OSError, URLError):
        return False


def port_is_open():
    try:
        with socket.create_connection(("127.0.0.1", PORT), timeout=1):
            return True
    except OSError:
        return False


def read_own_pid():
    try:
        record = json.loads(PID_FILE.read_text(encoding="utf-8"))
        pid = record["pid"]
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
            return None
        command = subprocess.run(
            ["ps", "-ww", "-p", str(pid), "-o", "args="],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        expected = record.get("command")
        if not isinstance(expected, str) or command != expected:
            return None
        if str(HERE / "serve.py") not in command:
            return None
        return pid
    except (OSError, ValueError, KeyError, TypeError):
        return None


def start(open_browser=False):
    if is_explorer_running():
        print(f"网页已在运行：{URL}")
        if open_browser:
            webbrowser.open(URL)
        return 0
    if port_is_open():
        print(f"端口 {PORT} 被其他程序占用。没有停止或替换该程序。", file=sys.stderr)
        return 1
    if not (HERE / "data.js").exists():
        print("首次启动，正在读取原始 CSV…", flush=True)
        subprocess.run([sys.executable, str(HERE / "prepare_data.py")], check=True)
    RUNTIME.mkdir(mode=0o700, exist_ok=True)
    command = [sys.executable, str(HERE / "serve.py"), "--port", str(PORT)]
    with LOG_FILE.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=str(HERE), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            close_fds=True,
        )
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if process.poll() is not None:
            print(f"网页服务未能启动。详细原因保存在：{LOG_FILE}", file=sys.stderr)
            PID_FILE.unlink(missing_ok=True)
            return 1
        if is_explorer_running():
            # macOS Python can re-exec during startup; record its command after readiness.
            # Save the command as well as the PID to reject unrelated reused PIDs.
            os_command = subprocess.run(
                ["ps", "-ww", "-p", str(process.pid), "-o", "args="],
                capture_output=True, text=True, check=False,
            ).stdout.strip()
            PID_FILE.write_text(json.dumps({"pid": process.pid, "command": os_command}), encoding="utf-8")
            print(f"网页已启动：{URL}")
            print("可以关闭这个终端窗口；服务会继续在本机后台运行。")
            if open_browser:
                webbrowser.open(URL)
            return 0
        time.sleep(0.15)
    # This handle belongs to the process we just created; do not leave a failed start running.
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    PID_FILE.unlink(missing_ok=True)
    print(f"启动超时。请查看：{LOG_FILE}", file=sys.stderr)
    return 1


def stop():
    pid = read_own_pid()
    if pid is None:
        if is_explorer_running():
            print("网页由另一个启动方式运行，未停止该进程。请到原启动窗口停止服务。")
            return 1
        PID_FILE.unlink(missing_ok=True)
        print("网页服务没有运行。")
        return 0
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and is_explorer_running():
        time.sleep(0.1)
    if is_explorer_running():
        print("服务尚未停止，请稍后检查。", file=sys.stderr)
        return 1
    PID_FILE.unlink(missing_ok=True)
    print("网页服务已停止。数据文件保持不变。")
    return 0


def main():
    parser = argparse.ArgumentParser(description="启动、停止或检查 Prosus 本地数据浏览器")
    parser.add_argument("action", nargs="?", default="start", choices=["start", "stop", "status"])
    parser.add_argument("--open", action="store_true", help="启动后在默认浏览器中打开网页")
    args = parser.parse_args()
    if args.action == "start":
        return start(args.open)
    if args.action == "stop":
        return stop()
    running = is_explorer_running()
    print(f"网页正在运行：{URL}" if running else "网页服务没有运行。")
    return 0 if running else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.SubprocessError) as error:
        print(f"操作未完成：{error}", file=sys.stderr)
        raise SystemExit(1)
