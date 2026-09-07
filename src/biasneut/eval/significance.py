"""Statistical rigor (§6.5): "≥3 seeds with mean ± std; paired significance
testing (McNemar for the classifier, bootstrap CIs for generation metrics)
against baselines. Report negative results."
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class McNemarResult:
    statistic: float
    p_value: float
    n_a_only_correct: int
    n_b_only_correct: int


def mcnemar_test(system_a_correct: list[bool], system_b_correct: list[bool], exact_threshold: int = 25) -> McNemarResult:
    """Paired test on binary correctness (e.g. "classifier says neutral")
    between two systems on the same examples. Uses the exact binomial test
    for small discordant-pair counts (recommended when n01+n10 < 25) and the
    chi-square approximation otherwise — both are the standard McNemar
    variants; we don't pull in statsmodels for one test.
    """
    a_only = sum(1 for a, b in zip(system_a_correct, system_b_correct) if a and not b)
    b_only = sum(1 for a, b in zip(system_a_correct, system_b_correct) if b and not a)
    n_discordant = a_only + b_only

    if n_discordant == 0:
        return McNemarResult(statistic=0.0, p_value=1.0, n_a_only_correct=a_only, n_b_only_correct=b_only)

    if n_discordant < exact_threshold:
        p_value = stats.binomtest(min(a_only, b_only), n_discordant, 0.5).pvalue
        statistic = float(min(a_only, b_only))
    else:
        statistic = (abs(a_only - b_only) - 1) ** 2 / n_discordant
        p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return McNemarResult(statistic=statistic, p_value=p_value, n_a_only_correct=a_only, n_b_only_correct=b_only)


def bootstrap_ci(values: list[float], n_samples: int = 1000, ci: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
    """Return (point_estimate, ci_low, ci_high) for the mean of ``values``."""
    values_arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_samples)
    n = len(values_arr)
    for i in range(n_samples):
        sample_idx = rng.integers(0, n, size=n)
        means[i] = values_arr[sample_idx].mean()
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return float(values_arr.mean()), float(lo), float(hi)


@dataclass
class SeedAggregateResult:
    mean: float
    std: float
    per_seed: list[float]


def aggregate_across_seeds(per_seed_values: list[float]) -> SeedAggregateResult:
    arr = np.asarray(per_seed_values, dtype=float)
    return SeedAggregateResult(mean=float(arr.mean()), std=float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
                                per_seed=list(per_seed_values))
