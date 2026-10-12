"""Controlled single-request latency per system, with warm models.

    .venv/bin/python -m evaluation.latency [--queries 30]

Runs one request at a time after a warm-up, so numbers are not inflated by
concurrent workers or model loading. Query vectors (e5, harrier) and reranker
scores are computed fresh: the benchmark uses empty temporary caches for them.
LLM query understanding stays cached here and is measured separately with the
response cache bypassed. Writes output/eval/latency.json.
"""

from __future__ import annotations

import argparse
import functools
import json
import statistics
import tempfile
import time
from pathlib import Path

from core.local_models import _ArrayStore
from retrieval.pipeline import SYSTEMS, UNDERSTAND_MODEL, UNDERSTAND_SCHEMA, UNDERSTAND_SYSTEM

from core.common import OUT, load_pipeline
from retrieval.multimodal import MODEL_DIR as VL_MODEL_DIR, VLEncoder, document_matrices, hybrid_vl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--llm-samples", type=int, default=5)
    parser.add_argument("--skip-vl", action="store_true", help="skip the Qwen3-VL systems")
    args = parser.parse_args()
    pipeline = load_pipeline()
    scratch = Path(tempfile.mkdtemp(prefix="latency-"))

    def fresh_caches(name: str) -> None:
        pipeline.dense._queries = _ArrayStore(scratch / f"{name}-dense-queries.sqlite3")
        pipeline.reranker._scores = _ArrayStore(scratch / f"{name}-rerank-scores.sqlite3")

    fresh_caches("warmup")
    pipeline.engine.e5_scores = functools.partial(pipeline.engine.e5_scores, use_cache=False)
    queries = [query["text"] for query in pipeline.corpus.queries[1:args.queries + 1]]
    for system in SYSTEMS:  # warm-up on a query outside the measured set: loads every model
        pipeline.run(system, pipeline.corpus.queries[0]["text"])
    result: dict = {"queries": len(queries), "systems_ms": {}}
    for system in SYSTEMS:
        fresh_caches(system)  # no system reuses vectors or scores computed for another
        timings = []
        for query in queries:
            start = time.perf_counter()
            pipeline.run(system, query)
            timings.append((time.perf_counter() - start) * 1000)
        result["systems_ms"][system] = {"median": statistics.median(timings), "p90": sorted(timings)[int(0.9 * (len(timings) - 1))]}
        print(f"{system:18} median {result['systems_ms'][system]['median']:8.1f} ms", flush=True)
    if not args.skip_vl and VL_MODEL_DIR.is_dir():
        # Photo-aware systems: document vectors are precomputed; the query is embedded fresh by the 2B model.
        encoder = VLEncoder()
        fused = document_matrices(encoder, pipeline.corpus)["fused"]
        encoder.store = _ArrayStore(scratch / "vl-warmup.sqlite3")
        encoder.queries([pipeline.corpus.queries[0]["text"]])  # warm-up: loads the model
        for system in ("vl-fused", "hybrid-vl"):
            fresh_caches(system)
            encoder.store = _ArrayStore(scratch / f"{system}-vl-queries.sqlite3")  # no reuse across systems
            timings = []
            for query in queries:
                start = time.perf_counter()
                vector = encoder.queries([query])[0]
                if system == "vl-fused":
                    pipeline.engine.rank_distinct(fused @ vector, 10)
                else:
                    hybrid_vl(pipeline, query, vector, fused)
                timings.append((time.perf_counter() - start) * 1000)
            result["systems_ms"][system] = {"median": statistics.median(timings), "p90": sorted(timings)[int(0.9 * (len(timings) - 1))]}
            print(f"{system:18} median {result['systems_ms'][system]['median']:8.1f} ms", flush=True)
        del encoder
    timings = []
    for query in queries[:args.llm_samples]:
        start = time.perf_counter()
        pipeline.llm.json("latency-probe", UNDERSTAND_MODEL, UNDERSTAND_SYSTEM, f"Query: {query}", UNDERSTAND_SCHEMA, use_cache=False)
        timings.append(time.perf_counter() - start)
    pipeline.llm.unload(UNDERSTAND_MODEL)
    result["understanding_uncached_s"] = {"median": statistics.median(timings), "samples": len(timings), "model": UNDERSTAND_MODEL}
    print(f"LLM understanding (uncached) median {result['understanding_uncached_s']['median']:.1f} s", flush=True)
    note = ("Single request at a time on an Apple M4 (16 GB), warm models, fresh query vectors and reranker scores. "
            "System times use cached LLM output; add the uncached understanding time for a new query to "
            "qe-hybrid and qe-hybrid-rerank. Photo-aware systems embed the query with Qwen3-VL-Embedding-2B (fp16).")
    result["note"] = note
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latency.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
