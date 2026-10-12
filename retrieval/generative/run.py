"""Decode the 100 evaluation queries with each trained variant and add them to the evaluation runs.

    .venv/bin/python -m retrieval.generative.run [--variants semantic tu-raw tu-clean] [--tag=-long]

System names are gr-<variant><tag> (checkpoints from `train --tag`). Each decoded identifier expands to every listing with that
content; the top 10 distinct listings are kept. Records merge into output/eval/runs.jsonl
like evaluation.runs, so evaluation.judge judges only the new pooled pairs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

from core.local_models import pick_device
from core.common import RUNS, load_corpus, read_jsonl, write_jsonl

from .decode import IdTrie, encode_targets, generate
from .ids import IDS, VARIANTS
from .train import CHECKPOINTS, load_model

TAGS = ["", "-long", "-50ep"]  # "-long": 8 epochs, "-50ep": up to 50 epochs (Snellius), compared with shorter runs
GR_SYSTEMS = [f"gr-{variant}{tag}" for tag in TAGS for variant in VARIANTS]
BEAMS = 20


def decode_queries(model, tokenizer, ids: dict, variant: str, corpus, system: str, device: str) -> tuple[list[dict], list[float]]:
    """Top-10 distinct listings for every evaluation query: each decoded identifier expands to all its listings."""
    index_of = {document_id: index for index, document_id in enumerate(corpus.document_ids)}
    trie = IdTrie(encode_targets(tokenizer, ids["variants"][variant]), tokenizer.eos_token_id)
    records, timings = [], []
    for query in corpus.queries:
        start = time.perf_counter()
        ranked = generate(model, tokenizer, trie, [query["text"]], beams=BEAMS, top=BEAMS, device=device)[0]
        documents, listings = [], set()
        for content, _ in ranked:
            for document_id in ids["listings"][str(content)]:
                item = corpus.items[index_of[document_id]]
                if (item["item_id"], item["merchant_id"]) not in listings:
                    listings.add((item["item_id"], item["merchant_id"]))
                    documents.append(document_id)
        timings.append((time.perf_counter() - start) * 1000)
        records.append({"query_id": query["id"], "query": query["text"], "system": system,
                        "document_ids": documents[:10], "identifiers": [ids["variants"][variant][c] for c, _ in ranked[:10]],
                        "elapsed_ms": round(timings[-1], 1)})
    return records, timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--tag", choices=TAGS, default="")
    args = parser.parse_args()
    device = pick_device()
    corpus = load_corpus()
    ids = json.loads(IDS.read_text(encoding="utf-8"))
    records = []
    for variant in args.variants:
        tokenizer, model = load_model(device, CHECKPOINTS / f"{variant}{args.tag}")
        model.eval()
        decoded, timings = decode_queries(model, tokenizer, ids, variant, corpus, f"gr-{variant}{args.tag}", device)
        records += decoded
        print(f"gr-{variant}{args.tag}: median {statistics.median(timings):.0f} ms per query (beam {BEAMS}, batch 1)", flush=True)
        del model
    systems = {record["system"] for record in records}
    write_jsonl(RUNS, [record for record in read_jsonl(RUNS) if record["system"] not in systems] + records)
    print(f"Saved {len(records)} runs to {RUNS}")


if __name__ == "__main__":
    main()
