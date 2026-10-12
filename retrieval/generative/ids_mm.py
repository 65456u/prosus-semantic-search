"""Multimodal semantic identifiers: the `semantic` structure, clustered on text and photo together.

    .venv/bin/python -m retrieval.multimodal        # caches the text+photo vectors first
    .venv/bin/python -m retrieval.generative.ids_mm

The identifier keeps the coarse-to-fine form of `semantic` (cluster name | canonical tipo |
variante detalhe); only the first level changes. k-means (64) runs over the concatenation of the
harrier text vector and the Qwen3-VL-Embedding text+photo vector, each L2-normalised so both
count equally, as in multimodal semantic IDs for generative retrieval. Clusters are named by the
same LLM prompt. Adds variant `semantic-mm` to output/genret/ids.json; other variants are untouched.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.cluster import KMeans

from core.llm import LLM
from core.local_models import DenseEncoder
from retrieval.pipeline import DENSE_INSTRUCTION
from core.common import ROOT, load_corpus, read_jsonl
from retrieval.multimodal import VLEncoder, document_matrices

from .ids import (CLUSTERS, IDS, TOKENIZER_DIR, consolidate_tipos, diagnostics, held_out_members, name_clusters,
                  squash, unique_ids)
from .prepare_items import ITEMS, MODEL


def main() -> None:
    from transformers import AutoTokenizer

    corpus = load_corpus()
    prepared = read_jsonl(ITEMS)
    rows = [record["row"] for record in prepared]
    output = json.loads(IDS.read_text(encoding="utf-8"))
    assert output["rows"] == rows, "ids.json and the prepared items disagree; rerun retrieval.generative.ids first"
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    dense = DenseEncoder(ROOT / ".demo-cache/models/harrier-oss-v1-0.6b", ROOT / ".demo-cache/retrieval", DENSE_INSTRUCTION)
    text_vectors = dense.documents(corpus.texts)[rows]
    photo_vectors = document_matrices(VLEncoder(), corpus)["fused"][rows]
    vectors = np.hstack([text_vectors, photo_vectors]) / np.sqrt(2)

    tipo_map = consolidate_tipos([record["tipo"] for record in prepared], dense)
    tipos = [tipo_map[record["tipo"]] for record in prepared]
    titles = [squash(f"{record['tipo']} {record['variante']} {record['detalhe']}") for record in prepared]
    labels = KMeans(n_clusters=CLUSTERS, n_init=4, random_state=0).fit(vectors).labels_
    llm = LLM()
    cluster_names = name_clusters(labels, titles, vectors, llm)
    llm.unload(MODEL)
    raw = [squash(f"{cluster_names[label]} | {tipo} | {squash(record['variante'] + ' ' + record['detalhe'])}")
           for label, tipo, record in zip(labels, tipos, prepared)]
    ids = unique_ids(raw, tokenizer)
    output["variants"]["semantic-mm"] = ids
    output["diagnostics"]["semantic-mm"] = diagnostics(raw, ids, tokenizer, held_out_members(corpus, rows))
    output["clusters_mm"] = {str(k): v for k, v in sorted(cluster_names.items())}
    # How much the photo moved the grouping: adjusted Rand index against the text-only clusters.
    from sklearn.metrics import adjusted_rand_score

    text_labels = KMeans(n_clusters=CLUSTERS, n_init=4, random_state=0).fit(text_vectors).labels_
    output["diagnostics"]["semantic-mm"]["ari_vs_text_clusters"] = round(float(adjusted_rand_score(text_labels, labels)), 3)
    print("semantic-mm", output["diagnostics"]["semantic-mm"], flush=True)
    IDS.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {IDS}")


if __name__ == "__main__":
    main()
