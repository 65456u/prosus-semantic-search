"""Pooled, blind, pointwise relevance judging by a local LLM from a different model family than every system component.

    .venv/bin/python -m evaluation.judge [--judge-model ollama/qwen3.5:9b-q4_K_M] [--workers 8]

Pool: the union of every system's top-5 for each query (TREC-style pooling), so
all systems are scored on the same judged items. Each (query, listing) pair is
judged on its own, without the rank, the producing system, or other candidates,
so a grade cannot depend on which systems happen to be in the pool.

The script also annotates each query once (translation and query type) for
stratified analysis and the human-labelling interface. Annotations come from a
different model than the judge, so human labels made with their help stay
independent of the judge they validate.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core.llm import LLM
from retrieval.pipeline import item_card

from core.common import JUDGMENTS, POOL_DEPTH, QUERY_INFO, RUNS, load_corpus, read_jsonl, write_jsonl

JUDGE_MODEL = "ollama/qwen3.5:9b-q4_K_M"
ANNOTATION_MODEL = "ollama/gemma4:e4b-it-q4_K_M"

JUDGE_SYSTEM = """You are an expert relevance assessor for iFood, a Brazilian food-delivery app.
You see one search query and one catalog listing. Decide how well the listing satisfies the query.
Use Brazilian culinary knowledge: regional dishes, menu conventions, and what Brazilian users mean by common phrases.
Queries may be literal word-for-word translations of English food phrases. Judge against the food the query most plausibly
means, not its literal wording. Menus often use English loanwords (wrap, bowl, burger, combo).

Grades:
3 = Exact: the listing is the food the user asked for, including the requested style or key detail when one is specified.
2 = Good substitute: the same dish or dish family, but a requested style, ingredient, or detail is missing or uncertain.
1 = Weak: loosely related only, for example a shared main ingredient, a side, or the same broad category.
0 = Irrelevant: a different food or product than requested, for example a drink when a dish was asked.

Judge only from the listing text. Do not reward a listing for sharing words with the query when the food differs.
Give a brief reason (at most 25 words), then the grade."""

JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["reason", "grade"],
    "properties": {"reason": {"type": "string"}, "grade": {"type": "integer", "enum": [0, 1, 2, 3]}},
}

QUERY_TYPES = ["specific_dish", "dish_with_modifier", "cuisine_or_style", "meal_or_vague"]

QUERY_SYSTEM = """You annotate Brazilian Portuguese food-search queries for an evaluation report.
Return:
- translation_en: the English food phrase the query most plausibly means (queries may be literal translations from English).
- translation_zh: a faithful Simplified Chinese translation.
- explanation_zh: one Simplified Chinese sentence explaining what food the query refers to, for a reader who does not know Brazilian cuisine.
- query_type, one of:
  specific_dish: names one well-defined dish (e.g. "Pão de queijo").
  dish_with_modifier: a dish plus a style, preparation, or attribute (e.g. "Pizza de massa fina assada em forno a lenha").
  cuisine_or_style: a cuisine, region, or style rather than one dish (e.g. "Comida caseira mineira").
  meal_or_vague: a meal occasion or an abstract description (e.g. "Tigela crocante e poderosa")."""

QUERY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["translation_en", "translation_zh", "explanation_zh", "query_type"],
    "properties": {
        "translation_en": {"type": "string"},
        "translation_zh": {"type": "string"},
        "explanation_zh": {"type": "string"},
        "query_type": {"type": "string", "enum": QUERY_TYPES},
    },
}


def pool(runs: list[dict], depth: int = POOL_DEPTH) -> dict[int, list[str]]:
    """Union of each system's top-`depth` per query, in first-seen order."""
    pooled: dict[int, dict[str, None]] = {}
    for run in sorted(runs, key=lambda record: (record["query_id"], record["system"])):
        pooled.setdefault(run["query_id"], {}).update(dict.fromkeys(run["document_ids"][:depth]))
    return {query_id: list(documents) for query_id, documents in pooled.items()}


def judge_prompt(query: str, card: str) -> str:
    return f"Search query (Brazilian Portuguese): {query}\n\nListing: {card}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--output", default=str(JUDGMENTS))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    corpus = load_corpus()
    llm = LLM()
    index_of = {document_id: index for index, document_id in enumerate(corpus.document_ids)}
    queries = {query["id"]: query["text"] for query in corpus.queries}
    pooled = pool(read_jsonl(RUNS))

    def annotate(query_id):
        return {"query_id": query_id, "query": queries[query_id],
                **llm.json("query-annotation", ANNOTATION_MODEL, QUERY_SYSTEM, f"Query: {queries[query_id]}", QUERY_SCHEMA)}

    def judge(pair):
        query_id, document_id = pair
        card = item_card(corpus, index_of[document_id])
        verdict = llm.json("judge", args.judge_model, JUDGE_SYSTEM, judge_prompt(queries[query_id], card), JUDGE_SCHEMA)
        return {"query_id": query_id, "document_id": document_id, "judge": args.judge_model, **verdict}

    pairs = [(query_id, document_id) for query_id, documents in sorted(pooled.items()) for document_id in documents]
    print(f"Pool: {len(pairs)} (query, listing) pairs over {len(pooled)} queries", flush=True)
    with ThreadPoolExecutor(args.workers) as executor:
        write_jsonl(QUERY_INFO, executor.map(annotate, sorted(queries)))
        llm.unload(ANNOTATION_MODEL)
        print(f"Annotated {len(queries)} queries with {ANNOTATION_MODEL}", flush=True)
        judgments = []
        for number, record in enumerate(executor.map(judge, pairs), start=1):
            judgments.append(record)
            if number % 250 == 0:
                print(f"Judged {number}/{len(pairs)}", flush=True)
    write_jsonl(Path(args.output), judgments)
    print(f"Saved {len(judgments)} judgments to {args.output}")


if __name__ == "__main__":
    main()
