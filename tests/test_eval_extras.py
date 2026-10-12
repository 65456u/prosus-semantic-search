"""Leakage and isolation checks for the behaviour ablation and the multimodal judge."""

from __future__ import annotations

import csv
import json

from core.corpus import Corpus
from retrieval.log_expansion import expanded_texts
from evaluation.behavioral import held_out
from evaluation.judge import JUDGE_SYSTEM
from evaluation.judge_mm import JUDGE_SYSTEM_MM


def test_log_expansion_never_uses_held_out_terms(tmp_path):
    held = next(f"termo {i}" for i in range(100) if held_out(f"termo {i}"))
    trained = next(f"termo {i}" for i in range(100) if not held_out(f"termo {i}"))
    items, queries = tmp_path / "items.csv", tmp_path / "queries.csv"
    with items.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["_id", "itemId", "itemMetadata", "itemProfile", "merchantId"])
        writer.writeheader()
        writer.writerow({"_id": "d0", "itemId": "i0", "merchantId": "m0",
                         "itemMetadata": json.dumps({"name": "Coxinha", "category_name": "Salgados"}),
                         "itemProfile": json.dumps({"search": [{"term": held, "count": 9}, {"term": trained, "count": 1}],
                                                    "metrics": {}})})
    with queries.open("w", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=["search_term_pt"]).writeheader()
    [text] = expanded_texts(Corpus(items, queries))
    assert trained in text and held not in text


def test_multimodal_judge_prompt_differs_only_in_the_evidence_line():
    text_lines, mm_lines = JUDGE_SYSTEM.splitlines(), JUDGE_SYSTEM_MM.splitlines()
    assert len(text_lines) == len(mm_lines)
    assert sum(a != b for a, b in zip(text_lines, mm_lines)) == 1


def test_two_tower_training_never_sees_held_out_terms_and_batches_have_no_duplicates(tmp_path):
    from retrieval.two_tower import batches, training_pairs

    terms = [f"termo {i}" for i in range(40)]
    items, queries = tmp_path / "items.csv", tmp_path / "queries.csv"
    with items.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["_id", "itemId", "itemMetadata", "itemProfile", "merchantId"])
        writer.writeheader()
        for i in range(6):
            writer.writerow({"_id": f"d{i}", "itemId": f"i{i}", "merchantId": "m0",
                             "itemMetadata": json.dumps({"name": f"Prato {i}", "category_name": "Pratos"}),
                             "itemProfile": json.dumps({"search": [{"term": t, "count": 1} for t in terms[i::6]], "metrics": {}})})
    with queries.open("w", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=["search_term_pt"]).writeheader()
    pairs, dev = training_pairs(Corpus(items, queries), "logs")
    used = {query for query, _ in pairs} | set(dev)
    assert used and not any(held_out(term) for term in used)
    assert not {query for query, _ in pairs} & set(dev)
    for batch in batches(pairs, 4, seed=1):
        assert len({q for q, _ in batch}) == len(batch) == len({r for _, r in batch})
    assert sum(map(len, batches(pairs, 4, seed=1))) == len(pairs)
