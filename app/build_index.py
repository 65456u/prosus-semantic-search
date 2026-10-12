"""Prepare persistent document vectors using the same core as the demo."""

from pathlib import Path
import json
import threading
import time

from core.corpus import Corpus
from retrieval.engine import RetrievalEngine


def main():
    root = Path(__file__).resolve().parents[1]
    corpus = Corpus(root / "data" / "5k_items_curated.csv", root / "data" / "queries.csv")
    engine = RetrievalEngine(corpus, root / ".demo-cache/retrieval", root / ".demo-cache/models/multilingual-e5-small")
    errors = []

    def prepare():
        try:
            engine.build_semantic()
        except Exception as exc:
            errors.append(exc)

    started = time.perf_counter()
    worker = threading.Thread(target=prepare)
    worker.start()
    last_message = None
    while worker.is_alive():
        status = engine.semantic_status
        message = status["message"]
        if message != last_message:
            print(message, flush=True)
            last_message = message
        worker.join(timeout=3)
    print(json.dumps({"seconds": round(time.perf_counter() - started, 2), **engine.semantic_status}, indent=2), flush=True)
    if errors:
        raise errors[0]


if __name__ == "__main__":
    main()
