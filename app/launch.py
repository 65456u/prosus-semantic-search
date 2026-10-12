"""Start or stop this demo locally without keeping a terminal window open."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".demo-cache" / "runtime"
STATE = RUNTIME / "server.json"


def health(port: int):
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1) as response:
            data = json.load(response)
        return data if data.get("app") == "prosus-search-demo" else None
    except (URLError, TimeoutError, ValueError, OSError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "stop", "status"], nargs="?", default="start")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    port = args.port
    if args.action != "start" and STATE.exists():
        port = json.loads(STATE.read_text())["port"]
    current = health(port)
    url = f"http://127.0.0.1:{port}/"
    if args.action == "status":
        print(json.dumps(current, ensure_ascii=False, indent=2) if current else "The demo is stopped.")
        return
    if args.action == "stop":
        if not current:
            print("The demo is already stopped.")
            return
        if not STATE.exists() or json.loads(STATE.read_text()).get("pid") != current.get("pid"):
            raise SystemExit("This launcher does not own the running server. Leave it running or stop it in its terminal.")
        os.kill(current["pid"], signal.SIGTERM)
        STATE.unlink(missing_ok=True)
        # Wait for shutdown so an immediate `start` does not find the old server still answering.
        for _ in range(40):
            if not health(port):
                break
            time.sleep(0.25)
        print("The demo has stopped.")
        return
    if current:
        print(f"The demo is already running: {url}")
        if args.open:
            webbrowser.open(url)
        return
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise SystemExit(f"Port {port} is in use by another service. Choose --port with a free port.")
    python = ROOT / ".venv" / "bin" / "python"
    if not python.is_file() or not (ROOT / "app/frontend/dist/index.html").is_file():
        raise SystemExit("Run the setup steps in app/README.md first.")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("HF_HOME", str(ROOT / ".demo-cache" / "huggingface"))
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    with (RUNTIME / "server.log").open("ab") as log:
        process = subprocess.Popen(
            [str(python), "-m", "uvicorn", "app.server:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            start_new_session=True,
        )
    STATE.write_text(json.dumps({"pid": process.pid, "port": port, "started_at": time.time()}))
    for _ in range(40):
        if health(port):
            print(f"The demo is ready: {url}")
            if args.open:
                webbrowser.open(url)
            return
        if process.poll() is not None:
            raise SystemExit(f"The server exited. Read {RUNTIME / 'server.log'}.")
        time.sleep(0.25)
    raise SystemExit(f"The server is still starting. Read {RUNTIME / 'server.log'}.")


if __name__ == "__main__":
    main()
