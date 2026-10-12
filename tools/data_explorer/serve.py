"""Serve only this explorer directory, bound to the local computer."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8765)
args = parser.parse_args()
directory = str(Path(__file__).resolve().parent)
handler = partial(SimpleHTTPRequestHandler, directory=directory)
server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
print(f"Data explorer: http://127.0.0.1:{args.port}/", flush=True)
try:
    server.serve_forever()
except KeyboardInterrupt:
    server.server_close()
