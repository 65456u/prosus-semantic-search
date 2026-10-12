"""Listwise human labelling: order a set of listings for one query, and mark where relevance ends.

    .venv/bin/python -m tools.annotation.listwise prepare      # 20 labelled queries x 8 listings
    .venv/bin/python -m tools.annotation.listwise serve        # http://127.0.0.1:8770/
    .venv/bin/python -m tools.annotation.listwise summary

For each query the set holds the top 3 of qe-hybrid, harrier, and hybrid (the systems whose
order is in dispute), filled to 8 from the judged pool across grades. Listings start in a
random order with no system shown. The annotator drags them into order and places a divider
above the first listing that no longer satisfies the query, so one pass yields a ranking and
a relevance threshold. Rankings are appended to output/eval/human_listwise.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from core.common import HUMAN, JUDGMENTS, OUT, RUNS, read_jsonl
from evaluation.metrics import kendall_tau_b

TASKS = OUT / "listwise_tasks.json"
LABELS = OUT / "human_listwise.jsonl"
PAGE = Path(__file__).with_name("listwise.html")
SET_SIZE = 8
PRIORITY = ["qe-hybrid", "harrier", "hybrid"]


def prepare(seed: int = 11) -> None:
    annotation = json.loads((OUT / "annotation_tasks.json").read_text(encoding="utf-8"))["queries"]
    runs = {(r["system"], r["query_id"]): r["document_ids"] for r in read_jsonl(RUNS)}
    human = {(r["query_id"], r["document_id"]): r["grade"] for r in read_jsonl(HUMAN)}
    rng = random.Random(seed)
    tasks = []
    for query in annotation:
        q = query["query_id"]
        pool = {item["document_id"]: item for item in query["items"]}
        chosen: list[str] = []
        for system in PRIORITY:
            for document_id in runs.get((system, q), [])[:3]:
                if document_id in pool and document_id not in chosen:
                    chosen.append(document_id)
        rest = [d for d in pool if d not in chosen]
        rng.shuffle(rest)
        # Fill with listings of grades not yet represented, so every set spans the scale.
        rest.sort(key=lambda d: sum(human.get((q, c)) == human.get((q, d)) for c in chosen))
        chosen = (chosen + rest)[:SET_SIZE]
        items = [pool[d] for d in chosen]
        rng.shuffle(items)
        tasks.append({"query_id": q, "query": query["query"], "translation_zh": query["translation_zh"],
                      "translation_en": query["translation_en"], "explanation_zh": query["explanation_zh"], "items": items})
    rng.shuffle(tasks)
    TASKS.write_text(json.dumps({"tasks": tasks}, ensure_ascii=False), encoding="utf-8")
    print(f"Prepared {len(tasks)} ranking tasks of up to {SET_SIZE} listings in {TASKS}")


def latest() -> dict[int, dict]:
    return {record["query_id"]: record for record in read_jsonl(LABELS)}


def summary() -> dict:
    """Agreement of the human ranking with the annotator's own grades and with the judge's grades."""
    human = {(r["query_id"], r["document_id"]): r["grade"] for r in read_jsonl(HUMAN)}
    judge = {(r["query_id"], r["document_id"]): r["grade"] for r in read_jsonl(JUDGMENTS)}
    result: dict[str, list[float]] = defaultdict(list)
    cut_agree = []
    for q, record in latest().items():
        order = record["order"]
        position = [-i for i in range(len(order))]  # higher is better
        for name, grades in (("own_grades", human), ("judge_grades", judge)):
            pairs = [(p, grades.get((q, d))) for p, d in zip(position, order) if grades.get((q, d)) is not None]
            if len(pairs) >= 3 and len({g for _, g in pairs}) > 1:
                result[name].append(kendall_tau_b([p for p, _ in pairs], [g for _, g in pairs]))
        above = set(order[:record["cut"]])
        cut_agree += [(d in above) == (human.get((q, d)) is not None and human[(q, d)] >= 2) for d in order if human.get((q, d)) is not None]
    out = {name: {"queries": len(v), "mean_kendall_tau_b": statistics.fmean(v)} for name, v in result.items() if v}
    if cut_agree:
        out["divider_matches_own_grade_2_plus"] = sum(cut_agree) / len(cut_agree)
    print(json.dumps(out, indent=2))
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/tasks":
            payload = json.loads(TASKS.read_text(encoding="utf-8"))
            payload["rankings"] = {str(q): {"order": r["order"], "cut": r["cut"], "reviewed": bool(r.get("divider_reviewed"))}
                                   for q, r in latest().items()}
            return self._send(200, json.dumps(payload, ensure_ascii=False).encode(), "application/json")
        self._send(404, b"Not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/rank":
            return self._send(404, b"Not found", "text/plain")
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        tasks = {task["query_id"]: task for task in json.loads(TASKS.read_text(encoding="utf-8"))["tasks"]}
        task = tasks.get(body.get("query_id"))
        order, cut = body.get("order"), body.get("cut")
        if task is None or sorted(order or []) != sorted(i["document_id"] for i in task["items"]) or not isinstance(cut, int) or not 0 <= cut <= len(order):
            return self._send(422, b"Invalid ranking", "text/plain")
        record = {"query_id": task["query_id"], "order": order, "cut": cut, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if body.get("reviewed"):
            record["divider_reviewed"] = True  # the divider was placed deliberately (second pass)
        with LABELS.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._send(200, b"{}", "application/json")

    def log_message(self, *args):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "serve", "summary"])
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "summary":
        summary()
    else:
        print(f"Listwise ranking: http://127.0.0.1:{args.port}/", flush=True)
        ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
