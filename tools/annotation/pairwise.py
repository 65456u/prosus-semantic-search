"""Blind side-by-side (pairwise) comparison of two systems' top 5, by a person or the LLM judge.

    .venv/bin/python -m tools.annotation.pairwise prepare      # 20 labelled queries x 2 system pairs
    .venv/bin/python -m tools.annotation.pairwise serve        # http://127.0.0.1:8768/
    .venv/bin/python -m tools.annotation.pairwise summary
    .venv/bin/python -m tools.annotation.pairwise judge        # LLM judge on all 100 queries, both orders
    .venv/bin/python -m tools.annotation.pairwise prepare-items     # query + two listings: which is more relevant?
    .venv/bin/python -m tools.annotation.pairwise serve --items     # http://127.0.0.1:8769/
    .venv/bin/python -m tools.annotation.pairwise judge-items       # LLM judge on the same listing pairs, both orders
    .venv/bin/python -m tools.annotation.pairwise summary-items
    .venv/bin/python -m tools.annotation.pairwise prepare-duels [--depth 2]   # same-rank listings of compared systems, all queries
    .venv/bin/python -m tools.annotation.pairwise serve --duels [--host 0.0.0.0]  # http://127.0.0.1:8772/
    .venv/bin/python -m tools.annotation.pairwise summary-duels
    .venv/bin/python -m tools.annotation.pairwise export-offline        # unlabelled duels as one self-contained page
    .venv/bin/python -m tools.annotation.pairwise import-offline CODE   # the result code that page shows

Pointwise grades depend on each grader's scale (the judge is about one grade stricter than
the annotator); a preference between two lists does not. Each task shows one query and two
top-5 lists in random left/right order, without system names, and asks which list better
serves the query. Identical lists are skipped. Preferences are appended to
output/eval/human_pairwise.jsonl. The LLM judge sees each comparison twice with the lists
swapped; only a preference that survives the swap counts, which cancels position bias.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import random
import secrets
import socket
import time
from collections import Counter
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from concurrent.futures import ThreadPoolExecutor

from core.llm import LLM
from retrieval.pipeline import item_card

from core.common import OUT, RUNS, load_corpus, read_jsonl, write_jsonl
from evaluation.judge import JUDGE_MODEL

PAIRS = [("hybrid", "qe-hybrid"), ("harrier", "qe-hybrid")]
TASKS = OUT / "pairwise_tasks.json"
LABELS = OUT / "human_pairwise.jsonl"
PAGE = Path(__file__).with_name("pairwise.html")
LLM_LABELS = OUT / "llm_pairwise.jsonl"
ITEM_TASKS = OUT / "pairwise_item_tasks.json"
ITEM_LABELS = OUT / "human_item_pairwise.jsonl"
LLM_ITEM_LABELS = OUT / "llm_item_pairwise.jsonl"
DUEL_TASKS = OUT / "pairwise_duel_tasks.json"
DUEL_LABELS = OUT / "human_duel_pairwise.jsonl"
OFFLINE_TEMPLATE = Path(__file__).with_name("duels_offline.html")
OFFLINE_PAGE = OUT / "duels_offline.html"
OFFLINE_INDEX = OUT / "duels_offline.json"
OFFLINE_CHOICES = {"L": "left", "R": "right", "U": "tie", "D": "both_bad"}
# System comparisons the duels decide: (system, the system it modifies).
DUELS = [("hybrid-vl", "hybrid"), ("qe-hybrid-neutral", "hybrid"), ("qe-hybrid-vl", "hybrid-vl"), ("harrier", "e5"),
         ("qe-hybrid-rerank", "qe-hybrid"), ("hybrid", "harrier"), ("harrier-ft", "harrier"), ("hybrid-ft", "hybrid")]

ITEM_SYSTEM = """You compare two catalog listings from iFood, a Brazilian food-delivery app, for one Brazilian Portuguese query.
Decide which listing better satisfies what the user most plausibly wants; queries may be literal translations of
English food phrases, so judge against the intended food. Answer "tie" when both are about equally relevant or
equally irrelevant. Give a brief reason (at most 25 words)."""

PAIR_SYSTEM = """You compare two search result lists from iFood, a Brazilian food-delivery app.
For one Brazilian Portuguese query you see List A and List B, each with its top 5 listings in rank order.
Decide which list better serves what the user most plausibly wants; queries may be literal translations of
English food phrases, so judge against the intended food. Earlier positions matter more than later ones.
Answer "tie" when the lists are about equally good or equally poor. Give a brief reason (at most 30 words)."""
PAIR_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["reason", "preference"],
               "properties": {"reason": {"type": "string"}, "preference": {"type": "string", "enum": ["A", "B", "tie"]}}}


def prepare(seed: int = 3) -> None:
    """Tasks on the human-labelled queries, whose pooled listings already have translations."""
    annotation = json.loads((OUT / "annotation_tasks.json").read_text(encoding="utf-8"))["queries"]
    runs = {(r["system"], r["query_id"]): r["document_ids"][:5] for r in read_jsonl(RUNS)}
    rng = random.Random(seed)
    tasks = []
    for query in annotation:
        items = {item["document_id"]: item for item in query["items"]}
        for a, b in PAIRS:
            lists = {system: runs[(system, query["query_id"])] for system in (a, b)}
            if lists[a] == lists[b] or not all(d in items for docs in lists.values() for d in docs):
                continue
            left, right = (a, b) if rng.random() < 0.5 else (b, a)
            tasks.append({"id": f"{query['query_id']}:{a}:{b}", "query_id": query["query_id"], "query": query["query"],
                          "translation_zh": query["translation_zh"], "translation_en": query["translation_en"],
                          "explanation_zh": query["explanation_zh"], "pair": [a, b], "left_system": left, "right_system": right,
                          "left": [items[d] for d in lists[left]], "right": [items[d] for d in lists[right]]})
    rng.shuffle(tasks)
    TASKS.write_text(json.dumps({"tasks": tasks}, ensure_ascii=False), encoding="utf-8")
    print(f"Prepared {len(tasks)} side-by-side tasks in {TASKS}")


def prepare_items(per_kind: int = 15, seed: int = 5) -> None:
    """Listing pairs within one labelled query, three kinds of 15:

    tied      the annotator gave both the same grade (1-3): can a preference separate them?
    conflict  judge and annotator order the two the opposite way: who is right?
    clear     the annotator's grades differ by 2+: does the preference agree with the grades?
    """
    from core.common import HUMAN, JUDGMENTS

    annotation = json.loads((OUT / "annotation_tasks.json").read_text(encoding="utf-8"))["queries"]
    human = {(r["query_id"], r["document_id"]): r["grade"] for r in read_jsonl(HUMAN)}
    judge = {(r["query_id"], r["document_id"]): r["grade"] for r in read_jsonl(JUDGMENTS)}
    rng = random.Random(seed)
    kinds: dict[str, list] = {"tied": [], "conflict": [], "clear": []}
    for query in annotation:
        q = query["query_id"]
        items = [item for item in query["items"] if human.get((q, item["document_id"])) is not None and (q, item["document_id"]) in judge]
        for i, a in enumerate(items):
            for b in items[i + 1:]:
                ha, hb = human[(q, a["document_id"])], human[(q, b["document_id"])]
                ja, jb = judge[(q, a["document_id"])], judge[(q, b["document_id"])]
                if a["name"] == b["name"]:
                    continue
                if ha == hb and ha >= 1:
                    kinds["tied"].append((query, a, b))
                elif (ha - hb) * (ja - jb) < 0:
                    kinds["conflict"].append((query, a, b))
                elif abs(ha - hb) >= 2:
                    kinds["clear"].append((query, a, b))
    tasks = []
    for kind, candidates in kinds.items():
        rng.shuffle(candidates)
        per_query: Counter = Counter()
        for query, a, b in candidates:
            if sum(t["kind"] == kind for t in tasks) >= per_kind:
                break
            if per_query[query["query_id"]] >= 2:  # spread the pairs over many queries
                continue
            per_query[query["query_id"]] += 1
            left, right = (a, b) if rng.random() < 0.5 else (b, a)
            tasks.append({"id": f"{query['query_id']}:{a['document_id']}:{b['document_id']}", "kind": kind, "query_id": query["query_id"],
                          "query": query["query"], "translation_zh": query["translation_zh"], "translation_en": query["translation_en"],
                          "explanation_zh": query["explanation_zh"], "left_doc": left["document_id"], "right_doc": right["document_id"],
                          "left": [left], "right": [right]})
    rng.shuffle(tasks)
    ITEM_TASKS.write_text(json.dumps({"tasks": tasks, "mode": "items"}, ensure_ascii=False), encoding="utf-8")
    print(f"Prepared {len(tasks)} listing pairs in {ITEM_TASKS}: {Counter(t['kind'] for t in tasks)}")


def judge_all(workers: int = 8) -> None:
    """LLM side-by-side on every query where the two lists differ, both orders."""
    corpus = load_corpus()
    index_of = {document_id: index for index, document_id in enumerate(corpus.document_ids)}
    runs = {(r["system"], r["query_id"]): r["document_ids"][:5] for r in read_jsonl(RUNS)}
    llm = LLM()

    def listing(documents: list[str]) -> str:
        return "\n".join(f"{rank}. {item_card(corpus, index_of[d])}" for rank, d in enumerate(documents, start=1))

    def ask(job):
        query, a, b = job
        lists = {s: runs[(s, query["id"])] for s in (a, b)}
        votes = []
        for first, second in ((a, b), (b, a)):
            prompt = f"Query: {query['text']}\n\nList A:\n{listing(lists[first])}\n\nList B:\n{listing(lists[second])}"
            answer = llm.json("pairwise-judge", JUDGE_MODEL, PAIR_SYSTEM, prompt, PAIR_SCHEMA)
            votes.append("tie" if answer["preference"] == "tie" else first if answer["preference"] == "A" else second)
        winner = votes[0] if votes[0] == votes[1] else "tie"  # inconsistent across orders counts as a tie
        return {"id": f"{query['id']}:{a}:{b}", "query_id": query["id"], "pair": [a, b], "votes": votes, "winner": winner}

    jobs = [(query, a, b) for query in corpus.queries for a, b in PAIRS if runs[(a, query["id"])] != runs[(b, query["id"])]]
    with ThreadPoolExecutor(workers) as pool:
        records = list(pool.map(ask, jobs))
    write_jsonl(LLM_LABELS, records)
    consistent = sum(r["votes"][0] == r["votes"][1] for r in records)
    print(f"Judged {len(records)} comparisons; consistent across orders: {consistent}/{len(records)}")


def judge_items() -> None:
    corpus = load_corpus()
    index_of = {document_id: index for index, document_id in enumerate(corpus.document_ids)}
    tasks = json.loads(ITEM_TASKS.read_text(encoding="utf-8"))["tasks"]
    llm = LLM()

    def ask(task):
        a, b = task["left_doc"], task["right_doc"]
        votes = []
        for first, second in ((a, b), (b, a)):
            prompt = (f"Query: {task['query']}\n\nListing A: {item_card(corpus, index_of[first])}\n\n"
                      f"Listing B: {item_card(corpus, index_of[second])}")
            answer = llm.json("pairwise-item-judge", JUDGE_MODEL, ITEM_SYSTEM, prompt, PAIR_SCHEMA)
            votes.append("tie" if answer["preference"] == "tie" else first if answer["preference"] == "A" else second)
        return {"id": task["id"], "kind": task["kind"], "votes": votes, "winner": votes[0] if votes[0] == votes[1] else "tie"}

    with ThreadPoolExecutor(8) as pool:
        records = list(pool.map(ask, tasks))
    write_jsonl(LLM_ITEM_LABELS, records)
    print(f"Judged {len(records)} listing pairs; consistent across orders: {sum(r['votes'][0] == r['votes'][1] for r in records)}/{len(records)}")


def summary_items() -> dict:
    """What preferences add over pointwise grades, on the three kinds of listing pairs."""
    from core.common import HUMAN, JUDGMENTS

    tasks = {t["id"]: t for t in json.loads(ITEM_TASKS.read_text(encoding="utf-8"))["tasks"]}
    human_pref = {r["id"]: r for r in read_jsonl(ITEM_LABELS)}
    llm_pref = {r["id"]: r for r in read_jsonl(LLM_ITEM_LABELS)}
    grade = {(r["query_id"], r["document_id"]): r["grade"] for r in read_jsonl(HUMAN)}
    judge = {(r["query_id"], r["document_id"]): r["grade"] for r in read_jsonl(JUDGMENTS)}

    def better(scores, task):
        a, b = scores[(task["query_id"], task["left_doc"])], scores[(task["query_id"], task["right_doc"])]
        return "tie" if a == b else task["left_doc"] if a > b else task["right_doc"]

    result = {}
    for kind in ("tied", "conflict", "clear"):
        ids = [i for i in human_pref if tasks[i]["kind"] == kind]
        if not ids:
            continue
        row = {"pairs": len(ids), "human_preference_decided": sum(human_pref[i]["winner"] != "tie" for i in ids)}
        decided = [i for i in ids if human_pref[i]["winner"] != "tie"]
        row["agrees_with_own_grades"] = sum(human_pref[i]["winner"] == better(grade, tasks[i]) for i in decided)
        row["agrees_with_judge_grades"] = sum(human_pref[i]["winner"] == better(judge, tasks[i]) for i in decided)
        shared = [i for i in ids if i in llm_pref]
        row["llm_preference_matches_human"] = sum(llm_pref[i]["winner"] == human_pref[i]["winner"] for i in shared)
        row["llm_compared"] = len(shared)
        result[kind] = row
        print(f"{kind:9} {row}")
    return result


def latest() -> dict[str, dict]:
    return {record["id"]: record for record in read_jsonl(LABELS)}


def summary(records: list[dict] | None = None, label: str = "human") -> dict:
    records = records if records is not None else list(latest().values())
    print(f"--- {label} ({len(records)} comparisons)")
    result = {}
    for a, b in PAIRS:
        votes = Counter(record["winner"] for record in records if record["pair"] == [a, b])
        wins_b, wins_a, ties = votes[b], votes[a], votes["tie"]
        decided = wins_a + wins_b
        # Two-sided exact sign test on decided comparisons.
        tail = sum(math.comb(decided, k) for k in range(0, min(wins_a, wins_b) + 1)) / 2 ** decided if decided else 1.0
        result[f"{b} vs {a}"] = {"wins": wins_b, "ties": ties, "losses": wins_a, "p_sign_test": min(1.0, 2 * tail)}
        print(f"{b:>10} vs {a:<8}: {wins_b} wins, {ties} ties, {wins_a} losses (sign test p={min(1.0, 2 * tail):.3f})")
    return result


def prepare_duels(depth: int = 2, seed: int = 11) -> None:
    """Listing pairs that decide system comparisons, for every query: the two systems' listings at the same
    rank (1..depth) whenever they differ. Round r holds the rank-r duels of all queries, so stopping early
    still leaves complete rank-1 coverage. Pairs of a query are kept together. Re-running only appends pairs
    for newly added systems: existing tasks and labels never change. Pairs already chosen in the earlier
    listing-pair round are not asked again."""
    from tools.annotation.annotate import _item, _translations
    from core.common import QUERY_INFO

    corpus = load_corpus()
    runs = {(r["system"], r["query_id"]): r["document_ids"] for r in read_jsonl(RUNS)}
    info = {r["query_id"]: r for r in read_jsonl(QUERY_INFO)}
    previous = json.loads(DUEL_TASKS.read_text(encoding="utf-8"))["tasks"] if DUEL_TASKS.exists() else []
    known = {task["id"] for task in previous}
    earlier = {f"{r['query_id']}:{':'.join(sorted((r['left_doc'], r['right_doc'])))}" for r in read_jsonl(ITEM_LABELS)}
    rng = random.Random(seed)
    queries = sorted({q for _, q in runs})
    rng.shuffle(queries)
    comparisons = [(a, b) for a, b in DUELS if any(s == a for s, _ in runs) and any(s == b for s, _ in runs)]
    new: dict[str, dict] = {}
    for rank in range(1, depth + 1):
        for q in queries:
            for a, b in comparisons:
                da, db = runs[(a, q)][rank - 1], runs[(b, q)][rank - 1]
                if da == db:
                    continue
                pair_id = f"{q}:{':'.join(sorted((da, db)))}"
                serves = {"system": a, "baseline": b, "rank": rank, "system_doc": da, "baseline_doc": db}
                if pair_id in new:
                    new[pair_id]["serves"].append(serves)
                elif pair_id not in known and pair_id not in earlier:
                    new[pair_id] = {"id": pair_id, "query_id": q, "round": rank, "docs": (da, db), "serves": [serves]}
    translations = _translations(corpus, LLM(), sorted({d for task in new.values() for d in task["docs"]}))
    tasks = []
    for task in new.values():
        left, right = task["docs"] if rng.random() < 0.5 else task["docs"][::-1]
        record = info[task["query_id"]]
        tasks.append({"id": task["id"], "query_id": task["query_id"], "round": task["round"], "serves": task["serves"],
                      "query": record["query"], "translation_zh": record["translation_zh"], "translation_en": record["translation_en"],
                      "explanation_zh": record["explanation_zh"], "left_doc": left, "right_doc": right,
                      "left": [_item(corpus, left, translations[left])], "right": [_item(corpus, right, translations[right])]})
    order = {q: i for i, q in enumerate(queries)}
    tasks.sort(key=lambda t: (t["round"], order[t["query_id"]]))
    merged = sorted(previous + tasks, key=lambda t: t["round"])  # stable: new pairs join the end of their round
    DUEL_TASKS.write_text(json.dumps({"tasks": merged, "mode": "duels"}, ensure_ascii=False), encoding="utf-8")
    print(f"Added {len(tasks)} duel pairs ({Counter(t['round'] for t in tasks)} by rank) over {len({t['query_id'] for t in tasks})} queries; "
          f"{len(previous) + len(tasks)} in {DUEL_TASKS}; comparisons: {comparisons}")


def export_offline() -> None:
    """The unlabelled duels as one page that works without the local server (labels stay in the browser and
    come back as a short code: one letter per pair)."""
    tasks = json.loads(DUEL_TASKS.read_text(encoding="utf-8"))["tasks"]
    labelled = {record["id"] for record in read_jsonl(DUEL_LABELS)}
    todo = [task for task in tasks if task["id"] not in labelled]
    set_id = hashlib.sha1("|".join(task["id"] for task in todo).encode()).hexdigest()[:8]
    OFFLINE_INDEX.write_text(json.dumps({"set": set_id, "ids": [task["id"] for task in todo]}), encoding="utf-8")

    def slim(item: dict) -> dict:
        return {"n": item["name"], "zh": item.get("name_zh"), "c": item["category"], "czh": item.get("category_zh"),
                "d": item.get("description"), "dzh": item.get("description_zh"), "p": item.get("price"), "img": item.get("image_url")}

    data = {"set": set_id, "tasks": [{"qid": t["query_id"], "q": t["query"], "zh": t["translation_zh"], "en": t["translation_en"],
                                      "ex": t["explanation_zh"], "l": slim(t["left"][0]), "r": slim(t["right"][0])} for t in todo]}
    page = OFFLINE_TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    OFFLINE_PAGE.write_text(page, encoding="utf-8")
    print(f"{len(todo)} unlabelled duels in {OFFLINE_PAGE} (set {set_id})")


def import_offline(code: str) -> None:
    index = json.loads(OFFLINE_INDEX.read_text(encoding="utf-8"))
    set_id, _, letters = code.strip().partition(":")
    if set_id != index["set"] or len(letters) != len(index["ids"]):
        raise SystemExit(f"Code does not match the exported set {index['set']} with {len(index['ids'])} pairs.")
    tasks = {task["id"]: task for task in json.loads(DUEL_TASKS.read_text(encoding="utf-8"))["tasks"]}
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    added = 0
    with DUEL_LABELS.open("a", encoding="utf-8") as stream:
        for pair_id, letter in zip(index["ids"], letters):
            if letter not in OFFLINE_CHOICES:
                continue
            task, choice = tasks[pair_id], OFFLINE_CHOICES[letter]
            winner = task[f"{choice}_doc"] if choice in ("left", "right") else "tie"
            stream.write(json.dumps({"id": pair_id, "query_id": task["query_id"], "round": task["round"], "left_doc": task["left_doc"],
                                     "right_doc": task["right_doc"], "choice": choice, "winner": winner, "time": stamp,
                                     "source": "offline"}) + "\n")
            added += 1
    print(f"Imported {added} labels from set {set_id}")


def duel_preferences() -> dict[str, dict]:
    """Latest human choice per listing pair, from the duels and the earlier listing-pair round."""
    choices = {}
    for record in read_jsonl(ITEM_LABELS) + read_jsonl(DUEL_LABELS):
        pair_id = f"{record['query_id']}:{':'.join(sorted((record['left_doc'], record['right_doc'])))}"
        choices[pair_id] = record
    return choices


def summary_duels() -> dict:
    """Per system comparison: how often the system's listing beat the baseline's at the same rank."""
    runs = {(r["system"], r["query_id"]): r["document_ids"] for r in read_jsonl(RUNS)}
    choices = duel_preferences()
    result = {}
    for a, b in DUELS:
        if not any(s == a for s, _ in runs) or not any(s == b for s, _ in runs):
            continue
        for rank in (1, 2, 3):
            wins = losses = ties = both_bad = same = 0
            for q in sorted({q for _, q in runs}):
                da, db = runs[(a, q)][rank - 1], runs[(b, q)][rank - 1]
                if da == db:
                    same += 1
                    continue
                record = choices.get(f"{q}:{':'.join(sorted((da, db)))}")
                if record is None:
                    continue
                if record["winner"] == da:
                    wins += 1
                elif record["winner"] == db:
                    losses += 1
                else:
                    ties += 1
                    both_bad += record.get("choice") == "both_bad"
            decided = wins + losses
            tail = sum(math.comb(decided, k) for k in range(0, min(wins, losses) + 1)) / 2 ** decided if decided else 1.0
            if wins + losses + ties:
                result[f"{a} vs {b} @rank{rank}"] = {"wins": wins, "ties": ties, "both_bad": both_bad, "losses": losses,
                                                     "identical": same, "p_sign_test": min(1.0, 2 * tail)}
                print(f"{a:>18} vs {b:<16} rank {rank}: {wins} wins, {ties} ties ({both_bad} both bad), {losses} losses, "
                      f"{same} identical (sign test p={min(1.0, 2 * tail):.3f})")
    (OUT / "duels.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return result


class Handler(BaseHTTPRequestHandler):
    tasks_path, labels_path = TASKS, LABELS  # switched to the listing-pair files by `serve --items` / `--duels`
    # Set when serving beyond localhost: the page shows confidential data, so every request must carry this
    # token, once in the link (?k=...) and then as a cookie.
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
        if not self._authorized():
            return self._send(403, "Use the link with the access token.".encode(), "text/plain; charset=utf-8")
        url = urlsplit(self.path)
        if url.path == "/" and self.token and "k=" in url.query:
            self.send_response(303)
            self.send_header("Set-Cookie", f"k={self.token}; Path=/; HttpOnly; SameSite=Strict")
            self.send_header("Location", "/")
            self.end_headers()
            return
        if url.path == "/":
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if url.path == "/api/tasks":
            payload = json.loads(self.tasks_path.read_text(encoding="utf-8"))
            labels = {record["id"]: record for record in read_jsonl(self.labels_path)}
            # System names and pair kinds never reach the page: the comparison stays blind.
            secret = ("pair", "left_system", "right_system", "kind", "left_doc", "right_doc", "serves")
            hidden = [{key: value for key, value in task.items() if key not in secret} for task in payload["tasks"]]
            choices = {task_id: record["choice"] for task_id, record in labels.items()}
            body = {"tasks": hidden, "labels": choices, "mode": payload.get("mode", "lists")}
            return self._send(200, json.dumps(body, ensure_ascii=False).encode(), "application/json")
        self._send(404, b"Not found", "text/plain")

    def do_POST(self):
        if not self._authorized():
            return self._send(403, b"Forbidden", "text/plain")
        if urlsplit(self.path).path != "/api/label":
            return self._send(404, b"Not found", "text/plain")
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        tasks = {task["id"]: task for task in json.loads(self.tasks_path.read_text(encoding="utf-8"))["tasks"]}
        task, choice = tasks.get(body.get("id")), body.get("choice")
        allowed = ("left", "right", "tie", "both_bad") if task and "serves" in task else ("left", "right", "tie")
        if task is None or choice not in allowed:
            return self._send(422, b"Invalid label", "text/plain")
        if "serves" in task:
            winner = task[f"{choice}_doc"] if choice in ("left", "right") else "tie"
            record = {"id": task["id"], "query_id": task["query_id"], "round": task["round"], "left_doc": task["left_doc"],
                      "right_doc": task["right_doc"], "choice": choice, "winner": winner}
        elif "left_system" in task:
            winner = "tie" if choice == "tie" else task[f"{choice}_system"]
            record = {"id": task["id"], "query_id": task["query_id"], "pair": task["pair"], "left_system": task["left_system"],
                      "right_system": task["right_system"], "choice": choice, "winner": winner}
        else:
            winner = "tie" if choice == "tie" else task[f"{choice}_doc"]
            record = {"id": task["id"], "query_id": task["query_id"], "kind": task["kind"], "left_doc": task["left_doc"],
                      "right_doc": task["right_doc"], "choice": choice, "winner": winner}
        record["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self.labels_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._send(200, b"{}", "application/json")

    def log_message(self, *args):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "serve", "summary", "judge", "prepare-items", "judge-items", "summary-items",
                                            "prepare-duels", "summary-duels", "export-offline", "import-offline"])
    parser.add_argument("code", nargs="?", help="import-offline: the result code from the offline page")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--items", action="store_true", help="serve the query + two listings tasks")
    parser.add_argument("--duels", action="store_true", help="serve the same-rank listing duels")
    parser.add_argument("--depth", type=int, default=2, help="prepare-duels: ranks to compare")
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 serves the local network with an access token")
    args = parser.parse_args()
    if args.command == "export-offline":
        export_offline()
    elif args.command == "import-offline":
        import_offline(args.code or "")
    elif args.command == "prepare-duels":
        prepare_duels(args.depth)
    elif args.command == "summary-duels":
        summary_duels()
    elif args.command == "prepare":
        prepare()
    elif args.command == "prepare-items":
        prepare_items()
    elif args.command == "judge-items":
        judge_items()
        summary_items()
    elif args.command == "summary-items":
        summary_items()
    elif args.command == "judge":
        judge_all()
        summary(read_jsonl(LLM_LABELS), "LLM judge, all queries")
    elif args.command == "summary":
        human = latest()
        summary(list(human.values()))
        machine = {record["id"]: record for record in read_jsonl(LLM_LABELS)}
        if machine:
            summary(list(machine.values()), "LLM judge, all queries")
            shared = [task_id for task_id in human if task_id in machine]
            summary([machine[task_id] for task_id in shared], "LLM judge, same comparisons as the human")
            agree = sum(human[t]["winner"] == machine[t]["winner"] for t in shared)
            print(f"Human and judge pick the same winner (or both tie) in {agree}/{len(shared)} shared comparisons")
    else:
        port = args.port or (8772 if args.duels else 8769 if args.items else 8768)
        if args.items:
            Handler.tasks_path, Handler.labels_path = ITEM_TASKS, ITEM_LABELS
        if args.duels:
            Handler.tasks_path, Handler.labels_path = DUEL_TASKS, DUEL_LABELS
        address = "127.0.0.1"
        if args.host not in ("127.0.0.1", "localhost"):
            Handler.token = secrets.token_urlsafe(9)
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect(("10.255.255.255", 1))  # UDP connect sends nothing; it only selects the LAN interface
                address = probe.getsockname()[0]
        suffix = f"?k={Handler.token}" if Handler.token else ""
        print(f"Side-by-side labelling: http://{address}:{port}/{suffix}", flush=True)
        ThreadingHTTPServer((args.host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
