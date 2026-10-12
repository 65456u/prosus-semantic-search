"""Ablation: does past search behaviour help when it is written into the listing representation?

    .venv/bin/python -m retrieval.log_expansion

No other retrieval system uses behaviour. Here each listing's retrieval text gets one extra line
with its most frequent past search terms ("Buscas: ..."), taken from the 80% training split of
terms only (document expansion, as with doc2query but from real logs). The hashed 20% held-out
split never enters a listing and stays the behavioural test set. Systems:

    harrier-logs   harrier over the expanded texts: search behaviour inside the item embedding
    hybrid-logs    RRF(bm25-pt, harrier), both channels over the expanded texts

Writes runs on the 100 evaluation queries to runs.jsonl (judge them with evaluation.judge) and
held-out-term metrics, paired against harrier and hybrid, to behavioral.json. Near-duplicate
terms can sit on both sides of the split ("acai" held out, "acai da cidade" in training); that is
the realistic case for logs, so the held-out gain is an upper bound on what logs add for new terms.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import numpy as np
from rank_bm25 import BM25Okapi

from retrieval.pipeline import CANDIDATE_DEPTH
from retrieval.engine import tokenize_portuguese

from evaluation.behavioral import RESULT, evaluate_terms, held_out, held_out_terms, paired
from core.common import RUNS, load_pipeline, read_jsonl, write_jsonl

LOG_SYSTEMS = ["harrier-logs", "hybrid-logs"]
BASELINE = {"harrier-logs": "harrier", "hybrid-logs": "hybrid"}
MAX_TERMS = 10


def expanded_texts(corpus) -> list[str]:
    texts = []
    for item, text in zip(corpus.items, corpus.texts):
        counts: Counter[str] = Counter()
        for entry in corpus.get_item(item["document_id"])["item_profile"].get("search") or []:
            term = str(entry.get("term") or "").strip().lower()
            if term and not held_out(term):
                counts[term] += int(entry.get("count") or 1)
        top = [term for term, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:MAX_TERMS]]
        texts.append(text + ("\nBuscas: " + ", ".join(top) if top else ""))
    return texts


class LogExpanded:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.texts = expanded_texts(pipeline.corpus)
        self.bm25 = BM25Okapi([tokenize_portuguese(text, stem=True) for text in self.texts], k1=1.5, b=0.75)
        self.matrix = pipeline.dense.documents(self.texts)

    def run(self, system: str, query: str, top_k: int = 10) -> list[int]:
        engine = self.pipeline.engine
        dense = self.matrix @ self.pipeline.dense.queries([query])[0]
        if system == "harrier-logs":
            return engine.rank_distinct(dense, top_k)
        lexical = np.asarray(self.bm25.get_scores(tokenize_portuguese(query, stem=True)), dtype=np.float64)
        ranking, _ = self.pipeline._fuse([engine.rank_distinct(lexical, CANDIDATE_DEPTH, positive_only=True),
                                          engine.rank_distinct(dense, CANDIDATE_DEPTH)])
        return ranking[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    pipeline = load_pipeline()
    corpus = pipeline.corpus
    start = time.perf_counter()
    expanded = LogExpanded(pipeline)
    added = sum(text != original for text, original in zip(expanded.texts, corpus.texts))
    training_terms = {line.removeprefix("Buscas: ") for text in expanded.texts for line in text.splitlines() if line.startswith("Buscas: ")}
    training_terms = {term for line in training_terms for term in line.split(", ")}
    overlap = sum(query["text"].strip().lower() in training_terms for query in corpus.queries)
    print(f"Expanded {added}/{len(corpus.texts)} listings with training-split search terms "
          f"({time.perf_counter() - start:.0f}s); evaluation queries that are also training terms: {overlap}", flush=True)

    # 1. The 100 evaluation queries, for the LLM judge.
    records = []
    for system in LOG_SYSTEMS:
        for query in corpus.queries:
            start = time.perf_counter()
            ranking = expanded.run(system, query["text"], args.top_k)
            records.append({"query_id": query["id"], "query": query["text"], "system": system,
                            "document_ids": [corpus.document_ids[index] for index in ranking],
                            "elapsed_ms": round((time.perf_counter() - start) * 1000, 1)})
    write_jsonl(RUNS, [r for r in read_jsonl(RUNS) if r["system"] not in LOG_SYSTEMS] + records)
    print(f"Saved {len(records)} runs to {RUNS}", flush=True)

    # 2. Held-out search terms, paired with the same system without logs.
    terms = held_out_terms(corpus)
    names = sorted(terms)
    result = json.loads(RESULT.read_text()) if RESULT.is_file() else {"systems": {}}
    ablation = {}
    for system in LOG_SYSTEMS:
        result["systems"][system] = evaluate_terms(corpus, terms, names, lambda term: expanded.run(system, term, 10))
        comparison = paired(result, system, BASELINE[system])
        if comparison:
            ablation[system] = comparison
        print(f"{system:13} held-out terms {len(names)}  MRR@10 {result['systems'][system]['mrr@10']:.3f}"
              + (f"  vs {comparison['baseline']} {comparison['baseline_mrr@10']:.3f} (p={comparison['p_value']:.4f})" if comparison else ""),
              flush=True)
    result["behavior_ablation"] = {"listings_expanded": added, "max_terms": MAX_TERMS, "eval_queries_in_training_terms": overlap, **ablation}
    RESULT.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {RESULT}")


if __name__ == "__main__":
    main()
