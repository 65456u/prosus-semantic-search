"""Training and development examples for generative retrieval (no evaluation queries are used).

Per distinct content: indexing views (full product text, name only), LLM-written customer
queries, and historical search terms from itemProfile.search (aggregated over identical
listings). Five percent of the query-style pairs, chosen by a hash, form a development set
for checkpoint selection; indexing views always stay in training, as in DSI.
"""

from __future__ import annotations

import collections
import hashlib
import random
from typing import Any

from core.common import read_jsonl

from .prepare_items import ITEMS

MAX_CONTENTS_PER_TERM = 20  # navigational terms ("burger king") keep their most-searched contents only


def _is_dev(text: str, target: int) -> bool:
    return int(hashlib.sha256(f"{text}\t{target}".encode()).hexdigest(), 16) % 20 == 0


def build_examples(corpus, ids: dict[str, Any], variant: str, seed: int = 0) -> tuple[list[tuple[str, int]], list[tuple[str, int]], dict[str, int]]:
    """Return (train, dev, counts) as (input text, content position) pairs; targets are looked up in ids."""
    prepared = read_jsonl(ITEMS)
    rows = ids["rows"]
    position = {row: i for i, row in enumerate(rows)}
    content_of = {corpus.texts[row]: position[row] for row in rows}
    train: list[tuple[str, int]] = []
    queries: list[tuple[str, int]] = []
    for i, (row, record) in enumerate(zip(rows, prepared)):
        train.append((corpus.texts[row].replace("\n", " | "), i))
        train.append((corpus.items[row]["name"], i))
        queries.extend((query, i) for query in record["queries"])
    searches: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for index, item in enumerate(corpus.items):
        for entry in corpus.get_item(item["document_id"])["item_profile"].get("search") or []:
            term = str(entry.get("term") or "").strip().lower()
            if term:
                searches[term][content_of[corpus.texts[index]]] += int(entry.get("count") or 1)
    log_pairs = [(term, content) for term, counter in searches.items()
                 for content, _ in counter.most_common(MAX_CONTENTS_PER_TERM)]
    dev = [pair for pair in queries + log_pairs if _is_dev(*pair)]
    train += [pair for pair in queries + log_pairs if not _is_dev(*pair)]
    train = list(dict.fromkeys(train))
    random.Random(seed).shuffle(train)
    counts = {"indexing": 2 * len(rows), "llm_queries": len(queries), "search_log_pairs": len(log_pairs),
              "train": len(train), "dev": len(dev), "contents": len(rows)}
    return train, dev, counts
