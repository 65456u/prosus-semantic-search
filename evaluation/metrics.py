"""Pure ranking, statistics, and agreement functions (no I/O, unit-tested).

Grades are 0-3. A listing is "acceptable" at grade >= 2. Unjudged listings
count as grade 0; with full top-5 pooling, no evaluated listing is unjudged.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

ACCEPTABLE = 2


def dcg(grades: Sequence[float]) -> float:
    return sum((2 ** grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1))


def ndcg_at_k(ranked: Sequence[float], judged_pool: Sequence[float], k: int = 5) -> float:
    """Ideal ranking comes from all judged pool grades for the query.

    A query whose pool has no relevant listing has an undefined nDCG; it scores
    0, consistent with "nothing relevant was returned".
    """
    ideal = dcg(sorted(judged_pool, reverse=True)[:k])
    return dcg(list(ranked)[:k]) / ideal if ideal > 0 else 0.0


def precision_at_k(ranked: Sequence[float], k: int = 5) -> float:
    return sum(grade >= ACCEPTABLE for grade in list(ranked)[:k]) / k


def reciprocal_rank(ranked: Sequence[float], k: int = 5) -> float:
    for rank, grade in enumerate(list(ranked)[:k], start=1):
        if grade >= ACCEPTABLE:
            return 1.0 / rank
    return 0.0


def hit_at_k(ranked: Sequence[float], k: int = 5) -> float:
    return float(any(grade >= ACCEPTABLE for grade in list(ranked)[:k]))


def bootstrap_ci(values: Sequence[float], resamples: int = 10_000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap over queries for the mean."""
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, len(data), size=(resamples, len(data)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_permutation_p(a: Sequence[float], b: Sequence[float], resamples: int = 10_000, seed: int = 0) -> float:
    """Two-sided paired randomization test on per-query differences (sign flips)."""
    differences = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    observed = abs(differences.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(resamples, len(differences)))
    permuted = np.abs((signs * differences).mean(axis=1))
    return float((np.sum(permuted >= observed - 1e-12) + 1) / (resamples + 1))


def cohen_kappa(a: Sequence[int], b: Sequence[int], categories: Sequence[int], weights: str | None = None) -> float:
    """Cohen's kappa; weights=None (nominal), 'linear', or 'quadratic' for ordinal grades."""
    index = {category: position for position, category in enumerate(categories)}
    size = len(categories)
    observed = np.zeros((size, size))
    for x, y in zip(a, b):
        observed[index[x], index[y]] += 1
    total = observed.sum()
    if total == 0:
        return float("nan")
    observed /= total
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    i, j = np.meshgrid(range(size), range(size), indexing="ij")
    if weights == "quadratic":
        disagreement = ((i - j) / max(size - 1, 1)) ** 2
    elif weights == "linear":
        disagreement = np.abs(i - j) / max(size - 1, 1)
    else:
        disagreement = (i != j).astype(float)
    expected_disagreement = (disagreement * expected).sum()
    if expected_disagreement == 0:
        return 1.0
    return float(1 - (disagreement * observed).sum() / expected_disagreement)


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall tau-a between two score vectors over the same systems."""
    concordant = discordant = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            product = (x[i] - x[j]) * (y[i] - y[j])
            concordant += product > 0
            discordant += product < 0
    pairs = len(x) * (len(x) - 1) / 2
    return (concordant - discordant) / pairs if pairs else float("nan")


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall tau-b: tau corrected for ties, for comparing a ranking with coarse 0-3 grades."""
    concordant = discordant = ties_x = ties_y = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    return (concordant - discordant) / denominator if denominator else float("nan")
