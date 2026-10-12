"""Hand-computed checks for evaluation metrics and agreement statistics."""

from __future__ import annotations

import math

import pytest

from evaluation.judge import pool
from evaluation.metrics import (bootstrap_ci, cohen_kappa, hit_at_k, kendall_tau, ndcg_at_k,
                               paired_permutation_p, precision_at_k, reciprocal_rank)


def test_ranking_metrics_by_hand():
    ranked = [0, 2, 3, 0, 1]
    pool_grades = [3, 2, 1, 0, 0, 3]
    dcg = 3 / math.log2(3) + 7 / math.log2(4) + 1 / math.log2(6)
    ideal = 7 + 7 / math.log2(3) + 3 / math.log2(4) + 1 / math.log2(5)
    assert ndcg_at_k(ranked, pool_grades) == pytest.approx(dcg / ideal)
    assert precision_at_k(ranked) == pytest.approx(0.4)
    assert reciprocal_rank(ranked) == pytest.approx(0.5)
    assert hit_at_k(ranked) == 1.0
    assert ndcg_at_k([0, 0], [0, 0, 1]) == 0.0
    assert ndcg_at_k([1], [0, 1]) == pytest.approx(1.0)


def test_kappa_perfect_chance_and_ordinal_weights():
    labels = [0, 1, 2, 3, 3, 2]
    assert cohen_kappa(labels, labels, [0, 1, 2, 3]) == pytest.approx(1.0)
    assert cohen_kappa(labels, labels, [0, 1, 2, 3], "quadratic") == pytest.approx(1.0)
    near = cohen_kappa([0, 1, 2, 3], [1, 2, 3, 2], [0, 1, 2, 3], "quadratic")
    far = cohen_kappa([0, 1, 2, 3], [3, 3, 0, 0], [0, 1, 2, 3], "quadratic")
    assert near > far
    # Two raters with independent, identical marginals agree at chance level.
    assert cohen_kappa([0, 0, 1, 1], [0, 1, 0, 1], [0, 1]) == pytest.approx(0.0)


def test_statistics_are_reproducible_and_sensible():
    values = [0.0, 1.0] * 50
    low, high = bootstrap_ci(values)
    assert low < 0.5 < high
    assert bootstrap_ci(values) == (low, high)
    assert paired_permutation_p([1.0] * 30, [0.0] * 30) < 0.001
    assert paired_permutation_p(values, values) == pytest.approx(1.0)
    assert kendall_tau([1, 2, 3], [10, 20, 30]) == 1.0
    assert kendall_tau([1, 2, 3], [3, 2, 1]) == -1.0


def test_pool_is_union_of_top5_per_query():
    runs = [
        {"query_id": 1, "system": "a", "document_ids": ["d1", "d2", "d3", "d4", "d5", "d6"]},
        {"query_id": 1, "system": "b", "document_ids": ["d5", "d7", "d1", "d8", "d9", "d10"]},
        {"query_id": 2, "system": "a", "document_ids": ["d1"]},
    ]
    pooled = pool(runs)
    assert pooled[1] == ["d1", "d2", "d3", "d4", "d5", "d7", "d8", "d9"]
    assert pooled[2] == ["d1"]


def test_kendall_tau_b_matches_scipy_with_ties():
    from scipy.stats import kendalltau
    from evaluation.metrics import kendall_tau_b
    ranking, grades = [7, 6, 5, 4, 3, 2, 1, 0], [3, 3, 2, 3, 1, 0, 1, 0]
    assert kendall_tau_b(ranking, grades) == pytest.approx(kendalltau(ranking, grades).statistic)
