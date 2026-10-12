"""Blind human relevance labelling for validating the LLM judge.

    .venv/bin/python -m tools.annotation.annotate prepare [--queries 20]
    .venv/bin/python -m tools.annotation.annotate serve      # http://127.0.0.1:8766/
    .venv/bin/python -m tools.annotation.annotate serve --host 0.0.0.0   # phone on the same Wi-Fi; prints a link with an access token
    .venv/bin/python -m tools.annotation.annotate retest [--pairs 40]    # blind re-labelling of already labelled pairs
    .venv/bin/python -m tools.annotation.annotate serve --retest --port 8767
    .venv/bin/python -m tools.annotation.annotate topup [--systems ...]   # bring later systems into the human check

`prepare` draws a query sample stratified by query type with a fixed seed, takes
each sampled query's full judged pool, and shuffles the listings. The annotator
never sees the producing system, its rank, or the LLM grade. Chinese
translations (local Gemma 4 E4B, not the judge model) and product images help
a non-Portuguese annotator; the Portuguese original is always shown. Labels are appended to
output/eval/human_labels.jsonl; the latest label for a pair wins.
"""

from __future__ import annotations

import argparse
import hmac
import json
import random
import secrets
import socket
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from core.llm import LLM

from core.common import HUMAN, JUDGMENTS, OUT, POOL_DEPTH, QUERY_INFO, RUNS, load_corpus, read_jsonl

TASKS = OUT / "annotation_tasks.json"
RETEST_TASKS = OUT / "retest_tasks.json"
HUMAN_RETEST = OUT / "human_labels_retest.jsonl"
PAGE = Path(__file__).with_name("annotate.html")
TRANSLATE_MODEL = "ollama/gemma4:e4b-it-q4_K_M"
TRANSLATE_SYSTEM = """Translate a Brazilian food-delivery catalog listing into Simplified Chinese for a reader who does not know Portuguese.
Translate faithfully. Keep brand names. When a dish name is a Brazilian specialty, add a short explanation in parentheses.
Return empty strings for empty fields."""
TRANSLATE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name_zh", "description_zh", "category_zh"],
    "properties": {"name_zh": {"type": "string"}, "description_zh": {"type": "string"}, "category_zh": {"type": "string"}},
}


def stratified_sample(query_info: list[dict], size: int, seed: int = 13) -> list[int]:
    """Proportional allocation per query type, at least one query from each type."""
    by_type: dict[str, list[int]] = defaultdict(list)
    for record in sorted(query_info, key=lambda record: record["query_id"]):
        by_type[record["query_type"]].append(record["query_id"])
    rng = random.Random(seed)
    total = sum(len(ids) for ids in by_type.values())
    chosen = []
    for query_type, ids in sorted(by_type.items()):
        quota = max(1, round(size * len(ids) / total))
        chosen.extend(rng.sample(ids, min(quota, len(ids))))
    rng.shuffle(chosen)
    return chosen[:size]


def prepare(size: int) -> None:
    corpus = load_corpus()
    llm = LLM()
    info = {record["query_id"]: record for record in read_jsonl(QUERY_INFO)}
    pooled: dict[int, list[str]] = defaultdict(list)
    for judgment in read_jsonl(JUDGMENTS):
        pooled[judgment["query_id"]].append(judgment["document_id"])
    sample = stratified_sample(list(info.values()), size)
    rng = random.Random(29)

    documents = sorted({document_id for query_id in sample for document_id in pooled[query_id]})
    translations = _translations(corpus, llm, documents)
    tasks = []
    for query_id in sample:
        items = [_item(corpus, document_id, translations[document_id]) for document_id in pooled[query_id]]
        rng.shuffle(items)
        record = info[query_id]
        tasks.append({"query_id": query_id, "query": record["query"], "translation_zh": record["translation_zh"],
                      "translation_en": record["translation_en"], "explanation_zh": record["explanation_zh"],
                      "query_type": record["query_type"], "items": items})
    TASKS.write_text(json.dumps({"queries": tasks}, ensure_ascii=False), encoding="utf-8")
    print(f"Prepared {sum(len(task['items']) for task in tasks)} listings over {len(tasks)} queries in {TASKS}")


def _translations(corpus, llm: LLM, documents: list[str]) -> dict[str, dict]:
    def translate(document_id: str) -> dict:
        detail = corpus.get_item(document_id)
        source = {"name": detail["name"], "description": detail["description"], "category": detail["category"]}
        return llm.json("annotation-translation", TRANSLATE_MODEL, TRANSLATE_SYSTEM, json.dumps(source, ensure_ascii=False), TRANSLATE_SCHEMA)

    with ThreadPoolExecutor(8) as executor:
        return dict(zip(documents, executor.map(translate, documents)))


def _item(corpus, document_id: str, translation: dict) -> dict:
    detail = corpus.get_item(document_id)
    return {"document_id": document_id, "name": detail["name"], "description": detail["description"],
            "category": detail["category"], "price": detail["price"], "image_url": detail["image_url"], **translation}


TOPUP_SYSTEMS = ["qe-hybrid-neutral", "hybrid-vl", "qe-hybrid-vl"]


def topup(systems: list[str], seed: int = 31) -> None:
    """Bring systems added after the labelling round into the human check on the same queries: each sampled
    query keeps its labelled listings and gains the unlabelled listings in these systems' top 5. Labels are
    untouched; the page jumps to the first unlabelled listing."""
    corpus, llm = load_corpus(), LLM()
    data = json.loads(TASKS.read_text(encoding="utf-8"))
    labelled = {(r["query_id"], r["document_id"]) for r in read_jsonl(HUMAN)}
    top: dict[int, list[str]] = defaultdict(list)
    for run in read_jsonl(RUNS):
        if run["system"] in systems:
            top[run["query_id"]] += run["document_ids"][:POOL_DEPTH]
    new = {}
    for task in data["queries"]:
        q = task["query_id"]
        task["items"] = [item for item in task["items"] if (q, item["document_id"]) in labelled]
        new[q] = [d for d in dict.fromkeys(top[q]) if (q, d) not in labelled]
    translations = _translations(corpus, llm, sorted({d for documents in new.values() for d in documents}))
    llm.unload(TRANSLATE_MODEL)
    rng = random.Random(seed)
    for task in data["queries"]:
        items = [_item(corpus, d, translations[d]) for d in new[task["query_id"]]]
        rng.shuffle(items)
        task["items"] += items
    TASKS.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"{sum(map(len, new.values()))} listings to label over {sum(bool(v) for v in new.values())} queries "
          f"for {', '.join(systems)} in {TASKS}")


def latest_labels(path: Path = HUMAN) -> dict[str, dict]:
    labels = {}
    for record in read_jsonl(path):
        labels[f"{record['query_id']}|{record['document_id']}"] = record
    return labels


def prepare_retest(pairs: int, seed: int = 41) -> None:
    """Sample already labelled pairs, balanced across the four grades, for a blind second pass.

    The retest page never shows the first label; agreement between the two passes is the
    annotator's own consistency, the ceiling any judge can be expected to reach.
    """
    first = {key: record["grade"] for key, record in latest_labels().items() if record["grade"] is not None}
    by_grade: dict[int, list[str]] = defaultdict(list)
    for key, grade in sorted(first.items()):
        by_grade[grade].append(key)
    rng = random.Random(seed)
    chosen: list[str] = []
    for grade in sorted(by_grade):
        chosen += rng.sample(by_grade[grade], min(pairs // len(by_grade), len(by_grade[grade])))
    rest = [key for key in sorted(first) if key not in set(chosen)]
    chosen += rng.sample(rest, max(0, pairs - len(chosen)))
    wanted = set(chosen)
    tasks = []
    for task in json.loads(TASKS.read_text(encoding="utf-8"))["queries"]:
        items = [item for item in task["items"] if f"{task['query_id']}|{item['document_id']}" in wanted]
        if items:
            rng.shuffle(items)
            tasks.append({**task, "items": items})
    rng.shuffle(tasks)
    RETEST_TASKS.write_text(json.dumps({"queries": tasks, "mode": "retest"}, ensure_ascii=False), encoding="utf-8")
    print(f"Prepared {sum(len(t['items']) for t in tasks)} retest pairs over {len(tasks)} queries in {RETEST_TASKS}")


class Handler(BaseHTTPRequestHandler):
    tasks_path, labels_path = TASKS, HUMAN  # switched to the retest files by `serve --retest`
    # Set when serving beyond localhost. The page contains confidential data, so every request
    # must carry this token: once in the link (?k=...), then as a cookie.
    token: str | None = None

    def _authorized(self) -> bool:
        if not self.token:
            return True
        supplied = parse_qs(urlsplit(self.path).query).get("k", [""])[0]
        cookie = SimpleCookie(self.headers.get("Cookie", ""))
        if "k" in cookie:
            supplied = supplied or cookie["k"].value
        return hmac.compare_digest(supplied.encode(), self.token.encode())

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlsplit(self.path)
        if not self._authorized():
            return self._send(403, "Use the link with the access token.".encode(), "text/plain; charset=utf-8")
        if url.path == "/" and self.token and "k=" in url.query:
            # Move the token into a cookie and drop it from the address bar.
            self.send_response(303)
            self.send_header("Set-Cookie", f"k={self.token}; HttpOnly; SameSite=Strict; Path=/; Max-Age=172800")
            self.send_header("Location", "/")
            self.end_headers()
            return
        if url.path == "/":
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/tasks":
            payload = json.loads(self.tasks_path.read_text(encoding="utf-8"))
            payload["labels"] = {key: record["grade"] for key, record in latest_labels(self.labels_path).items()}
            self._send(200, json.dumps(payload, ensure_ascii=False).encode(), "application/json")
        else:
            self._send(404, b"Not found", "text/plain")

    def do_POST(self):
        if not self._authorized():
            return self._send(403, b"Forbidden", "text/plain")
        if urlsplit(self.path).path != "/api/label":
            return self._send(404, b"Not found", "text/plain")
        record = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if record.get("grade") not in (0, 1, 2, 3, None):
            return self._send(422, b"Invalid grade", "text/plain")
        entry = {"query_id": int(record["query_id"]), "document_id": str(record["document_id"]),
                 "grade": record["grade"], "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
        with self.labels_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._send(200, b"{}", "application/json")

    def log_message(self, *args):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "retest", "serve", "topup"])
    parser.add_argument("--pairs", type=int, default=40)
    parser.add_argument("--retest", action="store_true", help="serve the retest tasks and record to the retest file")
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--systems", nargs="+", default=TOPUP_SYSTEMS, help="systems whose top 5 `topup` adds")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 serves the local network with an access token")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.queries)
        return
    if args.command == "retest":
        prepare_retest(args.pairs)
        return
    if args.command == "topup":
        topup(args.systems)
        return
    if args.retest:
        Handler.tasks_path, Handler.labels_path = RETEST_TASKS, HUMAN_RETEST
    address = "127.0.0.1"
    if args.host not in ("127.0.0.1", "localhost"):
        Handler.token = secrets.token_urlsafe(9)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 1))  # UDP connect sends nothing; it only selects the LAN interface
            address = probe.getsockname()[0]
    suffix = f"?k={Handler.token}" if Handler.token else ""
    print(f"Labelling interface: http://{address}:{args.port}/{suffix}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
