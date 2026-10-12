"""LLM pass over every distinct catalog content: a clean identifier phrase plus synthetic training queries.

    .venv/bin/python -m retrieval.generative.prepare_items [--workers 4]

Listing names are often poor identifiers ("2115 - salmão…", "PEQUENA (4 PEDAÇOS)", "Ninho").
A local LLM (Gemma 4 E4B, a different family from the Qwen judge) writes for each content:
tipo (kind), variante (flavour/filling/substance), detalhe (brand/size), and three
customer-style queries. Variante and detalhe must be faithful: every word of four or more
letters must occur in the listing text, otherwise the item is retried alone and finally
falls back to the cleaned original name. Queries see only the listing, never the
evaluation queries. Writes output/genret/items.jsonl.
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from core.corpus import normalize_text
from core.llm import LLM
from core.common import OUT, load_corpus, write_jsonl

MODEL = "ollama/gemma4:e4b-it-q4_K_M"
ITEMS = OUT.parent / "genret" / "items.jsonl"
BATCH = 5

SYSTEM = """You prepare Brazilian food-delivery catalog listings for a search index. For each listing return:
- tipo: the generic kind of product or dish, 1-2 common Brazilian Portuguese words in lower case (e.g. "pizza", "hambúrguer", "açaí", "batata frita", "analgésico", "refrigerante").
- variante: the main distinguishing flavour, filling, ingredient, or active substance, 1-4 words; empty if none.
- detalhe: brand, size, or quantity if stated, 1-4 words; empty if none.
- consultas: 3 different short searches (2-6 words) a Brazilian customer could type to find this listing: one naming the product, one describing it, one about a need or occasion it fits. Do not copy the listing name verbatim.
For tipo, variante, and detalhe use only information stated in the listing; never invent flavours, brands, or sizes.
Drop menu numbers, promotions, prices, and marketing words. Return one entry per listing id."""

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {"items": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["id", "tipo", "variante", "detalhe", "consultas"],
    "properties": {"id": {"type": "string"}, "tipo": {"type": "string"}, "variante": {"type": "string"}, "detalhe": {"type": "string"},
                   "consultas": {"type": "array", "items": {"type": "string"}}}}}}}


def distinct_contents(corpus) -> list[int]:
    """First source row of every distinct retrieval text; identical listings share one identifier."""
    seen: dict[str, int] = {}
    for index, text in enumerate(corpus.texts):
        seen.setdefault(text, index)
    return list(seen.values())


def listing_line(corpus, index: int, local_id: int) -> str:
    metadata = corpus.get_item(corpus.document_ids[index])["item_metadata"]
    taxonomy = metadata.get("taxonomy") or {}
    path = " > ".join(str(taxonomy[k]) for k in ("l0", "l1", "l2") if taxonomy.get(k))
    description = (metadata.get("description") or "").strip()[:200]
    return (f"id={local_id} | name: {metadata.get('name') or ''} | description: {description} | "
            f"menu section: {metadata.get('category_name') or ''} | taxonomy: {path}")


def unfaithful(entry: dict[str, Any], source: str) -> list[str]:
    phrase = normalize_text(f"{entry.get('variante', '')} {entry.get('detalhe', '')}")
    return [word for word in re.findall(r"[a-z]{4,}", phrase) if word not in source]


def clean_name(name: str) -> str:
    name = re.sub(r"^\s*\d+\s*[-.)]\s*", "", name)
    name = re.sub(r"(?i)\b(promo\.?|promoção|oferta|até \d+% ?off|\d+% ?off)\b", " ", name)
    return re.sub(r"\s+", " ", name).strip(" -|")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    corpus = load_corpus()
    llm = LLM()
    contents = distinct_contents(corpus)
    sources = {index: normalize_text(corpus.texts[index]) for index in contents}

    def ask(indices: list[int]) -> dict[int, dict[str, Any]]:
        prompt = "\n".join(listing_line(corpus, index, local) for local, index in enumerate(indices))
        try:
            entries = llm.json("item-canonical", MODEL, SYSTEM, prompt, SCHEMA)["items"]
        except RuntimeError:
            return {}
        by_local = {entry["id"].strip(): entry for entry in entries}
        return {index: by_local[str(local)] for local, index in enumerate(indices) if str(local) in by_local}

    def process(batch: list[int]) -> list[dict[str, Any]]:
        found = ask(batch)
        records = []
        for index in batch:
            entry = found.get(index)
            status = "batch"
            if entry is None or not entry.get("tipo", "").strip() or unfaithful(entry, sources[index]):
                entry = ask([index]).get(index)  # retry alone: batches sometimes drop or blur items
                status = "retried"
            if entry is None or not entry.get("tipo", "").strip() or unfaithful(entry, sources[index]):
                name = corpus.items[index]["name"]
                tipo = (entry or {}).get("tipo", "").strip().lower() or normalize_text(corpus.items[index]["taxonomy_l0"]).replace("_", " ")
                entry = {"tipo": tipo, "variante": clean_name(name), "detalhe": "", "consultas": (entry or {}).get("consultas", [])}
                status = "fallback"
            queries = [q.strip() for q in entry.get("consultas", []) if q.strip()][:3]
            records.append({"row": index, "document_id": corpus.document_ids[index], "tipo": entry["tipo"].strip().lower(),
                            "variante": entry.get("variante", "").strip(), "detalhe": entry.get("detalhe", "").strip(),
                            "queries": queries, "status": status})
        return records

    batches = [contents[start:start + BATCH] for start in range(0, len(contents), BATCH)]
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(args.workers) as pool:
        for number, batch_records in enumerate(pool.map(process, batches), start=1):
            records.extend(batch_records)
            if number % 50 == 0:
                print(f"{len(records)}/{len(contents)} contents", flush=True)
    write_jsonl(ITEMS, records)
    statuses = {status: sum(r["status"] == status for r in records) for status in ("batch", "retried", "fallback")}
    print(f"Saved {len(records)} contents to {ITEMS}: {statuses}")


if __name__ == "__main__":
    main()
