"""A second, LLM-free source of relevance: historical search behaviour.

    .venv/bin/python -m evaluation.behavioral [--systems ...] [--llm-sample 300]

Queries are historical search terms from itemProfile.search; a listing counts as relevant
when users who searched that term reached it. The terms come from the same hashed 20%
split used for the identifier diagnostics. No system except generative retrieval ever
trains on search logs, and generative retrieval (trained on most of them) is excluded.

These labels are incomplete (only listings inside this 5k sample, only those users
reached), so the metrics are Hit@10 and MRR@10 of the first relevant listing, reported
per system with bootstrap CIs, then compared with the LLM-judge ordering. Terms differ
from the evaluation queries: short, often navigational (brands, merchants).
Writes output/eval/behavioral.json.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import statistics

from retrieval.pipeline import LLM_SYSTEMS, SYSTEMS

from core.common import OUT, load_pipeline
from evaluation.metrics import bootstrap_ci, kendall_tau, paired_permutation_p

RESULT = OUT / "behavioral.json"


def held_out(term: str) -> bool:
    """The hashed 20% of search terms kept for testing; the other 80% may be used for training."""
    return int(hashlib.sha256(term.encode()).hexdigest(), 16) % 5 == 0


def held_out_terms(corpus) -> dict[str, set[tuple[str, str]]]:
    """Term -> relevant listings (item_id, merchant_id), for the hashed 20% of terms."""
    relevant: dict[str, set[tuple[str, str]]] = collections.defaultdict(set)
    for item in corpus.items:
        for entry in corpus.get_item(item["document_id"])["item_profile"].get("search") or []:
            term = str(entry.get("term") or "").strip().lower()
            if term and held_out(term):
                relevant[term].add((item["item_id"], item["merchant_id"]))
    return dict(relevant)


def evaluate_terms(corpus, terms: dict[str, set[tuple[str, str]]], names: list[str], retrieve, sampled: bool = False) -> dict:
    """Hit@10 and MRR@10 of the first listing users reached, for `retrieve(term) -> ranked row indices`."""
    hits, reciprocal = [], []
    for term in names:
        listings = [(corpus.items[i]["item_id"], corpus.items[i]["merchant_id"]) for i in retrieve(term)[:10]]
        rank = next((position for position, listing in enumerate(listings, start=1) if listing in terms[term]), None)
        hits.append(float(rank is not None))
        reciprocal.append(1.0 / rank if rank else 0.0)
    return {"terms": len(names), "sampled": sampled,
            "hit@10": statistics.fmean(hits), "hit@10_ci95": bootstrap_ci(hits),
            "mrr@10": statistics.fmean(reciprocal), "mrr@10_ci95": bootstrap_ci(reciprocal),
            "per_term_mrr": dict(zip(names, reciprocal))}


def paired(result: dict, system: str, baseline: str) -> dict | None:
    """MRR@10 difference against a baseline on the same terms, with a paired permutation test."""
    entry, base = result["systems"].get(system), result["systems"].get(baseline)
    if not entry or not base:
        return None
    names = sorted(entry["per_term_mrr"])
    after, before = [entry["per_term_mrr"][t] for t in names], [base["per_term_mrr"][t] for t in names]
    return {"baseline": baseline, "mrr@10": statistics.fmean(after), "baseline_mrr@10": statistics.fmean(before),
            "delta": statistics.fmean(after) - statistics.fmean(before), "p_value": paired_permutation_p(after, before)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=[s for s in SYSTEMS if s not in LLM_SYSTEMS])
    parser.add_argument("--llm-sample", type=int, default=300, help="terms evaluated for systems that call the LLM")
    args = parser.parse_args()
    pipeline = load_pipeline()
    corpus = pipeline.corpus
    terms = held_out_terms(corpus)
    names = sorted(terms)
    sample = sorted(random.Random(7).sample(names, min(args.llm_sample, len(names))))
    previous = json.loads(RESULT.read_text()) if RESULT.is_file() else {"systems": {}}
    for system in args.systems:
        evaluated = sample if system in LLM_SYSTEMS else names
        entry = evaluate_terms(corpus, terms, evaluated, lambda term: pipeline.run(system, term, top_k=10)["ranking"],
                               sampled=system in LLM_SYSTEMS)
        previous["systems"][system] = entry
        print(f"{system:17} terms {len(evaluated):5}  Hit@10 {entry['hit@10']:.3f}  MRR@10 {entry['mrr@10']:.3f}", flush=True)
    if LLM_SYSTEMS & set(args.systems):
        pipeline.llm.unload(pipeline.understand_model)
    # Same-sample comparison: every system on the LLM sample, so orderings are comparable.
    done = [s for s in SYSTEMS if s in previous["systems"]]
    previous["sample_mrr@10"] = {s: statistics.fmean(previous["systems"][s]["per_term_mrr"][t] for t in sample) for s in done}
    report = json.loads((OUT / "report.json").read_text()) if (OUT / "report.json").is_file() else {}
    judge = {s: report.get("llm_judge", {}).get(s, {}).get("nDCG@5", {}).get("mean") for s in done}
    if len(done) >= 3 and all(v is not None for v in judge.values()):
        previous["kendall_tau_vs_judge"] = kendall_tau([previous["sample_mrr@10"][s] for s in done], [judge[s] for s in done])
    previous.update({"held_out_terms": len(names), "llm_sample": len(sample),
                     "relevant_per_term_median": statistics.median(len(v) for v in terms.values())})
    RESULT.write_text(json.dumps(previous, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {RESULT}; Kendall tau vs judge: {previous.get('kendall_tau_vs_judge')}")


if __name__ == "__main__":
    main()
