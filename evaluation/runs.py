"""Run every system on all 100 queries and save the top-10 distinct listings.

    .venv/bin/python -m evaluation.runs [--systems ...] [--workers 4]

Query understanding runs first for every query and the LLM is then unloaded,
so the local LLM and the PyTorch retrieval models are never resident together
on a 16 GB machine. LLM responses, query vectors, and reranker scores are
cached, so reruns reproduce the same rankings. Per-run elapsed times exclude
the LLM call (reported separately from the usage ledger).
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

from retrieval.pipeline import LLM_SYSTEMS, SYSTEMS

from core.common import RUNS, load_pipeline, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=SYSTEMS)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    pipeline = load_pipeline()
    corpus = pipeline.corpus
    if LLM_SYSTEMS & set(args.systems):
        start = time.perf_counter()
        variants = {system == "qe-hybrid-neutral" for system in LLM_SYSTEMS & set(args.systems)}
        with ThreadPoolExecutor(4) as pool:
            list(pool.map(lambda job: pipeline.understand(*job), [(query["text"], neutral) for query in corpus.queries for neutral in variants]))
        pipeline.llm.unload(pipeline.understand_model)
        print(f"Query understanding ready for {len(corpus.queries)} queries ({time.perf_counter() - start:.0f}s)", flush=True)
    start = time.perf_counter()
    _ = pipeline.dense_matrix  # embed the catalog once before parallel queries
    print(f"Dense catalog vectors ready ({time.perf_counter() - start:.0f}s)", flush=True)

    def run(task):
        query, system = task
        start = time.perf_counter()
        result = pipeline.run(system, query["text"], args.top_k)
        ranking = result.pop("ranking")
        result.pop("query")
        return {"query_id": query["id"], "query": query["text"], **result,
                "document_ids": [corpus.document_ids[index] for index in ranking],
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 1)}

    tasks = [(query, system) for system in args.systems for query in corpus.queries]
    records = []
    with ThreadPoolExecutor(args.workers) as pool:
        for number, record in enumerate(pool.map(run, tasks), start=1):
            records.append(record)
            if number % 100 == 0:
                print(f"{number}/{len(tasks)} runs", flush=True)
    previous = [record for record in read_jsonl(RUNS) if record["system"] not in args.systems]
    write_jsonl(RUNS, previous + records)
    print(f"Saved {len(records)} runs to {RUNS}")


if __name__ == "__main__":
    main()
