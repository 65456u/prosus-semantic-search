"""Word-based item identifiers for generative retrieval, in three variants for an ablation.

    .venv/bin/python -m retrieval.generative.ids

One identifier per distinct product content (identical listings share it and are
expanded after decoding). All identifiers are plain Portuguese text, so the seq2seq
model reuses its pretrained vocabulary; uniqueness comes from a " #k" suffix.

    tu-raw     PAMT-style "title + URL path": listing name | menu section | taxonomy l2 | l1
    tu-clean   same structure, with the LLM-normalised name (tipo variante detalhe) as the title
    semantic   coarse-to-fine: LLM-named semantic cluster | canonical tipo | variante detalhe
    semantic-mm  the same, with clusters over text and photo vectors (retrieval/generative/ids_mm.py)

The semantic cluster is k-means (64) over harrier content embeddings, named by Gemma
from its members, so the first decoding step chooses among balanced, meaningful groups.
Writes output/genret/ids.json with the identifiers and per-variant diagnostics.
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from core.llm import LLM
from core.local_models import DenseEncoder
from retrieval.pipeline import DENSE_INSTRUCTION
from retrieval.engine import tokenize_portuguese
from core.common import ROOT, load_corpus, read_jsonl

from .prepare_items import ITEMS, MODEL

IDS = ITEMS.parent / "ids.json"
VARIANTS = ["tu-raw", "tu-clean", "semantic", "semantic-mm"]
CLUSTERS = 64
MAX_ID_TOKENS = 40
TOKENIZER_DIR = ROOT / ".demo-cache/models/t5gemma-2-270m-270m"

NAME_SYSTEM = """You name groups of Brazilian food-delivery catalog listings.
Given sample listings from one group, return a 1-3 word Brazilian Portuguese category name, in lower case,
that covers most of them (for example "pizzas", "medicamentos", "chocolates", "pratos executivos")."""
NAME_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["nome"], "properties": {"nome": {"type": "string"}}}


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip(" |")


def consolidate_tipos(tipos: list[str], dense: DenseEncoder, threshold: float = 0.95) -> dict[str, str]:
    """Map free-text kinds to canonical ones: same Snowball stems, then near-identical embeddings."""
    by_stem: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for tipo in tipos:
        by_stem[" ".join(tokenize_portuguese(tipo, stem=True)) or tipo][tipo] += 1
    surface = {key: counter.most_common(1)[0][0] for key, counter in by_stem.items()}
    frequency = {key: sum(counter.values()) for key, counter in by_stem.items()}
    keys = sorted(by_stem, key=lambda key: -frequency[key])
    vectors = dense.documents([surface[key] for key in keys])
    kept: list[int] = []
    target: dict[str, str] = {}
    for position, key in enumerate(keys):
        similar = [k for k in kept if float(vectors[k] @ vectors[position]) >= threshold]
        canonical = surface[keys[similar[0]]] if similar else surface[key]
        if not similar:
            kept.append(position)
        for tipo in by_stem[key]:
            target[tipo] = canonical
    return target


def name_clusters(labels: np.ndarray, phrases: list[str], vectors: np.ndarray, llm: LLM) -> dict[int, str]:
    names: dict[int, str] = {}
    for cluster in range(labels.max() + 1):
        members = np.where(labels == cluster)[0]
        centre = vectors[members].mean(axis=0)
        closest = members[np.argsort(-(vectors[members] @ centre))[:20]]
        prompt = "Listings:\n" + "\n".join(f"- {phrases[m]}" for m in closest)
        names[cluster] = squash(llm.json("cluster-name", MODEL, NAME_SYSTEM, prompt, NAME_SCHEMA)["nome"]).lower() or f"grupo {cluster}"
    counts = collections.Counter(names.values())
    for cluster, name in names.items():  # duplicate names get the cluster's most common kind appended
        if counts[name] > 1:
            common = collections.Counter(phrases[m].split(" ")[0] for m in np.where(labels == cluster)[0]).most_common(1)[0][0]
            names[cluster] = f"{name} {common}" if common not in name else f"{name} {cluster}"
    return names


def unique_ids(raw: list[str], tokenizer) -> list[str]:
    """Cap length on word boundaries, then add " #k" to every repeated identifier after its first use."""
    capped = []
    for text in raw:
        words = text.split(" ")
        while len(words) > 1 and len(tokenizer(" ".join(words), add_special_tokens=False)["input_ids"]) > MAX_ID_TOKENS:
            words.pop()
        capped.append(" ".join(words).strip(" |"))
    seen: collections.Counter = collections.Counter()
    result = []
    for text in capped:
        seen[text] += 1
        result.append(text if seen[text] == 1 else f"{text} #{seen[text]}")
    return result


def diagnostics(raw: list[str], ids: list[str], tokenizer, held_out: dict[str, set[int]]) -> dict[str, Any]:
    lengths = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in ids]
    first = [text.split(" ")[0].lower() for text in ids]  # the first word is the first decoding commitment
    cover1, cover3 = [], []
    for members in held_out.values():
        counts = collections.Counter(first[m] for m in members).most_common()
        cover1.append(counts[0][1] / len(members))
        cover3.append(sum(v for _, v in counts[:3]) / len(members))
    repeated = collections.Counter(raw)
    return {"collided_before_suffix": round(sum(v for v in repeated.values() if v > 1) / len(raw), 4),
            "tokens_median": int(np.median(lengths)), "tokens_p95": int(np.percentile(lengths, 95)),
            "distinct_first_words": len(set(first)), "held_out_term_coverage_top1": round(float(np.mean(cover1)), 3),
            "held_out_term_coverage_top3": round(float(np.mean(cover3)), 3)}


def held_out_members(corpus, rows: list[int]) -> dict[str, set[int]]:
    """Held-out search terms (same 20% hash split as the design diagnostics) -> content positions, for terms with 3+ contents."""
    position = {row: i for i, row in enumerate(rows)}
    content_of = {text: position[row] for row in rows for text in [corpus.texts[row]]}
    held_out: dict[str, set[int]] = collections.defaultdict(set)
    for index, item in enumerate(corpus.items):
        for entry in corpus.get_item(item["document_id"])["item_profile"].get("search") or []:
            term = str(entry.get("term") or "").strip().lower()
            if term and int(hashlib.sha256(term.encode()).hexdigest(), 16) % 5 == 0:
                held_out[term].add(content_of[corpus.texts[index]])
    return {term: members for term, members in held_out.items() if len(members) >= 3}


def main() -> None:
    from transformers import AutoTokenizer

    corpus = load_corpus()
    prepared = read_jsonl(ITEMS)
    rows = [record["row"] for record in prepared]
    llm = LLM()
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    dense = DenseEncoder(ROOT / ".demo-cache/models/harrier-oss-v1-0.6b", ROOT / ".demo-cache/retrieval", DENSE_INSTRUCTION)
    vectors = dense.documents(corpus.texts)[rows]

    metadata = [corpus.get_item(corpus.document_ids[row])["item_metadata"] for row in rows]
    path = [f"{squash(m.get('category_name'))} | {str((m.get('taxonomy') or {}).get('l2') or '').replace('_', ' ').lower()} | "
            f"{str((m.get('taxonomy') or {}).get('l1') or '').replace('_', ' ').lower()}" for m in metadata]
    tipo_map = consolidate_tipos([record["tipo"] for record in prepared], dense)
    tipos = [tipo_map[record["tipo"]] for record in prepared]
    titles = [squash(f"{record['tipo']} {record['variante']} {record['detalhe']}") for record in prepared]
    labels = KMeans(n_clusters=CLUSTERS, n_init=4, random_state=0).fit(vectors).labels_
    cluster_names = name_clusters(labels, titles, vectors, llm)

    raw = {
        "tu-raw": [squash(f"{squash(corpus.items[row]['name'])} | {p}") for row, p in zip(rows, path)],
        "tu-clean": [squash(f"{title} | {p}") for title, p in zip(titles, path)],
        "semantic": [squash(f"{cluster_names[label]} | {tipo} | {squash(record['variante'] + ' ' + record['detalhe'])}")
                     for label, tipo, record in zip(labels, tipos, prepared)],
    }
    held_out = held_out_members(corpus, rows)
    position = {row: i for i, row in enumerate(rows)}
    content_of = {text: position[row] for row in rows for text in [corpus.texts[row]]}

    output: dict[str, Any] = {"rows": rows, "listings": {}, "variants": {}, "diagnostics": {},
                              "tipos": {"before": len(set(r["tipo"] for r in prepared)), "after": len(set(tipos))},
                              "clusters": {str(k): v for k, v in sorted(cluster_names.items())}}
    for index, item in enumerate(corpus.items):
        output["listings"].setdefault(str(content_of[corpus.texts[index]]), []).append(corpus.document_ids[index])
    for variant in raw:  # semantic-mm is added afterwards by retrieval.generative.ids_mm
        ids = unique_ids(raw[variant], tokenizer)
        output["variants"][variant] = ids
        output["diagnostics"][variant] = diagnostics(raw[variant], ids, tokenizer, held_out)
        print(variant, output["diagnostics"][variant], flush=True)
    print(f"tipos: {output['tipos']}; held-out terms: {len(held_out)}")
    IDS.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    llm.unload(MODEL)  # free memory before training
    print(f"Saved {IDS}")


if __name__ == "__main__":
    main()
