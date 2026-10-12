"""Behavioural evaluation, refined: typed searches only, graded by count, split by kind of search.

    .venv/bin/python -m evaluation.behavioral_graded

Labels come from itemProfile.search on the hashed 20% held-out terms (the same split as
evaluation.behavioral, never used by any system), with two refinements:

  typed only   only entries with method "term" (the user typed the search), not clicks on suggestions,
               history, or trends
  graded       a listing's gain for a term is log2(1 + number of times users reached it from that term),
               so nDCG@10 rewards ranking the most-reached listings first

Each held-out term is also classified by the local LLM (Gemma 4, cached) as navigational (a brand,
store, product line, or medicine), generic (a plain dish or product name), or descriptive (a dish or
product with attributes or a need). The LLM only chooses the subset; the labels stay behavioural.
The descriptive subset is closest to the evaluation queries, so systems that call the LLM at query
time (qe-hybrid-neutral, qe-hybrid-vl) are evaluated there. Metrics: nDCG@10 (graded), MRR@10,
Hit@10, plus a traffic-weighted nDCG@10 (each term weighted by how often it was searched).
Writes output/eval/behavioral_graded.json and .md.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from core.llm import LLM

from retrieval.log_expansion import LogExpanded
from evaluation.behavioral import held_out
from core.common import OUT, load_pipeline
from evaluation.metrics import bootstrap_ci, paired_permutation_p
from retrieval import multimodal, two_tower

RESULT = OUT / "behavioral_graded.json"
KIND_MODEL = "ollama/gemma4:e4b-it-q4_K_M"
KIND_SYSTEM = """You classify short searches typed into iFood, a Brazilian food-delivery app. Return the kind:
- navigational: names a specific brand, store or restaurant, product line, or medicine (e.g. "burger king", "coca cola zero",
  "dipirona 500mg", "acai da cidade", "activia").
- generic: a plain dish or product name with no further detail (e.g. "pizza", "açaí", "hambúrguer", "sabonete", "fralda").
- descriptive: describes the food or product wanted with attributes, ingredients, preparation, style, or a need
  (e.g. "pizza de calabresa com borda recheada", "lanche sem glúten", "sobremesa gelada", "comida caseira barata")."""
KIND_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["kind"],
               "properties": {"kind": {"type": "string", "enum": ["navigational", "generic", "descriptive"]}}}
BASE_SYSTEMS = ["bm25", "bm25-pt", "e5", "harrier", "hybrid", "vl-text", "vl-fused", "hybrid-vl",
                "harrier-logs", "hybrid-logs", "harrier-ft", "harrier-ft-syn", "hybrid-ft"]
LLM_SYSTEMS = ["qe-hybrid-neutral", "qe-hybrid-vl"]  # descriptive subset only
COMPARISONS = [("harrier", "e5"), ("hybrid", "harrier"), ("hybrid", "bm25-pt"), ("hybrid-vl", "hybrid"), ("vl-fused", "harrier"),
               ("qe-hybrid-neutral", "hybrid"), ("qe-hybrid-vl", "hybrid-vl"), ("harrier-ft", "harrier"), ("hybrid-ft", "hybrid"),
               ("hybrid-logs", "hybrid")]


def labels(corpus) -> tuple[dict[str, dict[tuple[str, str], float]], dict[str, int]]:
    """Held-out term -> {listing: gain} from typed searches, and term -> total typed searches."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for item in corpus.items:
        for entry in corpus.get_item(item["document_id"])["item_profile"].get("search") or []:
            term = str(entry.get("term") or "").strip().lower()
            if term and entry.get("method") == "term" and held_out(term):
                counts[term][(item["item_id"], item["merchant_id"])] += int(entry.get("count") or 1)
    gains = {term: {listing: math.log2(1 + n) for listing, n in counter.items()} for term, counter in counts.items()}
    return gains, {term: sum(counter.values()) for term, counter in counts.items()}


def kinds(terms: list[str]) -> dict[str, str]:
    llm = LLM()
    with ThreadPoolExecutor(8) as pool:
        found = list(pool.map(lambda t: llm.json("search-kind", KIND_MODEL, KIND_SYSTEM, f"Search: {t}", KIND_SCHEMA)["kind"], terms))
    return dict(zip(terms, found))


def score(corpus, ranking: list[int], gain: dict) -> dict[str, float]:
    listings = list(dict.fromkeys((corpus.items[i]["item_id"], corpus.items[i]["merchant_id"]) for i in ranking[:10]))
    gains = [gain.get(listing, 0.0) for listing in listings]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(gain.values(), reverse=True)[:10]))
    first = next((i + 1 for i, g in enumerate(gains) if g > 0), None)
    return {"nDCG@10": dcg / ideal if ideal else 0.0, "MRR@10": 1 / first if first else 0.0, "Hit@10": float(first is not None)}


def main() -> None:
    pipeline = load_pipeline()
    corpus = pipeline.corpus
    gains, traffic = labels(corpus)
    terms = sorted(gains)
    kind = kinds(terms)
    descriptive = [t for t in terms if kind[t] == "descriptive"]
    print(f"{len(terms)} held-out typed terms: {Counter(kind.values())}", flush=True)
    # LLM query understanding for the descriptive terms first, then free the LLM before the encoders load.
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda t: pipeline.understand(t, neutral=True), descriptive))
    pipeline.llm.unload(KIND_MODEL)

    retrieve = {s: (lambda q, s=s: pipeline.run(s, q, 10)["ranking"]) for s in ("bm25", "bm25-pt", "e5", "harrier", "hybrid", "qe-hybrid-neutral")}
    vl = multimodal.retrievers(pipeline)
    vl["_encoder"].queries(terms + [pipeline.dense_expansion(pipeline.understand(t, neutral=True)) for t in descriptive])
    retrieve.update({s: f for s, f in vl.items() if not s.startswith("_")})
    expanded = LogExpanded(pipeline)
    retrieve.update({s: (lambda q, s=s: expanded.run(s, q, 10)) for s in ("harrier-logs", "hybrid-logs")})
    ft = two_tower.retrievers(pipeline)
    for encoder in ft["_ranking"].encoders.values():
        encoder.queries(terms)
    retrieve.update({s: f for s, f in ft.items() if not s.startswith("_")})
    pipeline.dense.queries(terms)

    per_term: dict[str, dict[str, dict[str, float]]] = {}
    for system in BASE_SYSTEMS + LLM_SYSTEMS:
        evaluated = descriptive if system in LLM_SYSTEMS else terms
        per_term[system] = {t: score(corpus, retrieve[system](t), gains[t]) for t in evaluated}
        print(f"{system:18} {len(evaluated):5} terms", flush=True)

    subsets = {"all": terms, "descriptive": descriptive, "generic": [t for t in terms if kind[t] == "generic"],
               "navigational": [t for t in terms if kind[t] == "navigational"]}
    summary = {}
    for name, subset in subsets.items():
        summary[name] = {"terms": len(subset), "systems": {}}
        for system, scores in per_term.items():
            shared = [t for t in subset if t in scores]
            if len(shared) < len(subset):
                continue
            values = [scores[t]["nDCG@10"] for t in shared]
            weights = [traffic[t] for t in shared]
            summary[name]["systems"][system] = {
                "nDCG@10": statistics.fmean(values), "nDCG@10_ci95": bootstrap_ci(values),
                "MRR@10": statistics.fmean(scores[t]["MRR@10"] for t in shared), "Hit@10": statistics.fmean(scores[t]["Hit@10"] for t in shared),
                "traffic_weighted_nDCG@10": sum(v * w for v, w in zip(values, weights)) / sum(weights)}
        summary[name]["comparisons"] = {}
        for a, b in COMPARISONS:
            if a in summary[name]["systems"] and b in summary[name]["systems"]:
                x, y = [per_term[a][t]["nDCG@10"] for t in subset], [per_term[b][t]["nDCG@10"] for t in subset]
                summary[name]["comparisons"][f"{a} vs {b}"] = {"delta": statistics.fmean(x) - statistics.fmean(y), "p": paired_permutation_p(x, y)}
    RESULT.write_text(json.dumps({"kinds": dict(Counter(kind.values())), "kind_of_term": kind, "summary": summary}, indent=1, ensure_ascii=False),
                      encoding="utf-8")

    lines = ["# Behavioural evaluation: typed searches, graded by count, by kind of search", "",
             f"{len(terms)} held-out terms with typed searches; kinds (local LLM): {dict(Counter(kind.values()))}. "
             "Gain = log2(1 + times users reached the listing from the term). LLM systems run on the descriptive subset only.", ""]
    for name, block in summary.items():
        lines += [f"## {name} ({block['terms']} terms)", "", "| System | nDCG@10 [95% CI] | MRR@10 | Hit@10 | traffic-weighted nDCG@10 |", "|---|---|---|---|---|"]
        for system, v in block["systems"].items():
            ci = v["nDCG@10_ci95"]
            lines.append(f"| `{system}` | {v['nDCG@10']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {v['MRR@10']:.3f} | {v['Hit@10']:.3f} | {v['traffic_weighted_nDCG@10']:.3f} |")
        lines += ["", "| Comparison | Δ nDCG@10 | p |", "|---|---|---|"]
        lines += [f"| {k} | {v['delta']:+.3f} | {v['p']:.4f} |" for k, v in block["comparisons"].items()]
        lines.append("")
    (OUT / "behavioral_graded.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
