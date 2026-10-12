"""Unified human gold: pointwise grades, listing pairs, and listwise rankings in one latent score.

Each source becomes pairwise evidence between two listings of the same query:
  pointwise   higher grade beats lower (weight 1); equal grades are a tie (weight 0.5)
  listwise    earlier position beats later (weight 1) for every pair in the ranking
  pairwise    the chosen listing beats the other (weight 2: a direct, deliberate comparison)
A Bradley-Terry model per query (MM updates, with one tie against a reference item as a prior)
turns the evidence into a log-strength per listing. The listwise divider separately gives an
in-context acceptable / not-acceptable label.
"""

from __future__ import annotations

import math
from collections import defaultdict

from core.common import HUMAN, OUT, read_jsonl


def _evidence() -> dict[int, list[tuple[str, str, float, float]]]:
    """Per query: (a, b, wins_a, weight) with wins_a in {0, 0.5, 1}."""
    evidence: dict[int, list] = defaultdict(list)
    grades: dict[int, dict[str, int]] = defaultdict(dict)
    for record in read_jsonl(HUMAN):
        if record["grade"] is not None:
            grades[record["query_id"]][record["document_id"]] = record["grade"]
        else:
            grades[record["query_id"]].pop(record["document_id"], None)
    for q, by_doc in grades.items():
        docs = sorted(by_doc)
        for i, a in enumerate(docs):
            for b in docs[i + 1:]:
                ga, gb = by_doc[a], by_doc[b]
                evidence[q].append((a, b, 1.0 if ga > gb else 0.0 if ga < gb else 0.5, 1.0 if ga != gb else 0.5))
    for record in {r["query_id"]: r for r in read_jsonl(OUT / "human_listwise.jsonl")}.values():
        order = record["order"]
        for i, a in enumerate(order):
            for b in order[i + 1:]:
                evidence[record["query_id"]].append((a, b, 1.0, 1.0))
    for record in {r["id"]: r for r in read_jsonl(OUT / "human_item_pairwise.jsonl")}.values():
        a, b = record["left_doc"], record["right_doc"]
        wins = 0.5 if record["winner"] == "tie" else 1.0 if record["winner"] == a else 0.0
        evidence[record["query_id"]].append((a, b, wins, 2.0))
    return evidence


def bradley_terry(comparisons: list[tuple[str, str, float, float]], iterations: int = 200) -> dict[str, float]:
    items = sorted({x for a, b, _, _ in comparisons for x in (a, b)})
    wins: dict[str, float] = defaultdict(float)
    games: dict[tuple[str, str], float] = defaultdict(float)
    for a, b, w, weight in comparisons:
        wins[a] += weight * w
        wins[b] += weight * (1 - w)
        games[(a, b)] += weight
    strength = {item: 1.0 for item in items}
    for _ in range(iterations):
        updated = {}
        for item in items:
            # Prior: one tie (weight 1) against a reference item of strength 1 keeps estimates finite.
            denominator = 1.0 / (strength[item] + 1.0)
            for (a, b), n in games.items():
                if item in (a, b):
                    denominator += n / (strength[a] + strength[b])
            updated[item] = (wins[item] + 0.5) / denominator
        scale = math.exp(sum(math.log(v) for v in updated.values()) / len(updated))
        strength = {item: value / scale for item, value in updated.items()}
    return {item: math.log(value) for item, value in strength.items()}


def unified_gold() -> dict[int, dict[str, float]]:
    return {q: bradley_terry(comparisons) for q, comparisons in _evidence().items() if comparisons}


def divider_labels() -> dict[tuple[int, str], int]:
    """Listwise in-context acceptability: 1 above the divider, 0 below (queries whose divider was moved)."""
    labels = {}
    for record in {r["query_id"]: r for r in read_jsonl(OUT / "human_listwise.jsonl")}.values():
        if record["cut"] < len(record["order"]) or record.get("divider_reviewed"):
            for position, document_id in enumerate(record["order"]):
                labels[(record["query_id"], document_id)] = int(position < record["cut"])
    return labels
