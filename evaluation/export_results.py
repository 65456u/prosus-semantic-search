"""Export the submitted rankings: the top 10 distinct listings per query for the recommended systems.

    .venv/bin/python -m evaluation.export_results

Reads output/eval/runs.jsonl (written by evaluation.runs and the experiment scripts) and writes
results/<system>.tsv with query_id, rank, itemId, merchantId. Listing names and descriptions stay
out of the repository; join on the IDs with 5k_items_curated.csv to read them.
"""

from __future__ import annotations

from core.common import ROOT, RUNS, load_corpus, read_jsonl

SYSTEMS = ["qe-hybrid-vl", "hybrid-vl"]  # best overall; best without an LLM call at query time


def main() -> None:
    corpus = load_corpus()
    listing = {item["document_id"]: (item["item_id"], item["merchant_id"]) for item in corpus.items}
    runs = {(r["system"], r["query_id"]): r["document_ids"] for r in read_jsonl(RUNS)}
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    for system in SYSTEMS:
        lines = ["query_id\trank\titemId\tmerchantId"]
        for query in corpus.queries:
            ranking = runs[(system, query["id"])]
            lines += [f"{query['id']}\t{rank}\t{listing[d][0]}\t{listing[d][1]}" for rank, d in enumerate(ranking[:10], 1)]
        (out / f"{system}.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{system}: {len(lines) - 1} rows -> results/{system}.tsv")


if __name__ == "__main__":
    main()
