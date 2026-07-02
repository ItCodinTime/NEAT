"""Exact multi-seed aggregation and paired optimizer comparisons."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetricSummary:
    count: int
    mean: float
    std: float
    ci95_low: float
    ci95_high: float
    minimum: float
    maximum: float


def summarize(values: Iterable[float]) -> MetricSummary:
    """Compute sample statistics and a two-sided t-distribution 95% CI."""
    sample = np.asarray(list(values), dtype=np.float64)
    if sample.size < 2:
        raise ValueError("at least two seeds are required for statistical aggregation")
    try:
        from scipy.stats import t
    except ImportError as exc:
        raise RuntimeError(
            "statistics reports require the benchmark extra (scipy)"
        ) from exc
    mean = float(np.mean(sample))
    std = float(np.std(sample, ddof=1))
    margin = float(t.ppf(0.975, sample.size - 1) * std / math.sqrt(sample.size))
    return MetricSummary(
        count=int(sample.size),
        mean=mean,
        std=std,
        ci95_low=mean - margin,
        ci95_high=mean + margin,
        minimum=float(np.min(sample)),
        maximum=float(np.max(sample)),
    )


def require_complete_seeds(
    rows: Iterable[Mapping[str, object]], expected_seeds: Iterable[int]
) -> dict[str, dict[int, Mapping[str, object]]]:
    """Index optimizer rows and reject missing, duplicate, or extra seeds."""
    expected = set(expected_seeds)
    if len(expected) < 2:
        raise ValueError("a report requires at least two expected seeds")
    indexed: dict[str, dict[int, Mapping[str, object]]] = defaultdict(dict)
    for row in rows:
        optimizer = str(row["optimizer"])
        seed = int(row["seed"])
        if seed in indexed[optimizer]:
            raise ValueError(f"duplicate result for {optimizer} seed {seed}")
        indexed[optimizer][seed] = row
    if not indexed:
        raise ValueError("no result rows supplied")
    failures = []
    for optimizer, by_seed in sorted(indexed.items()):
        actual = set(by_seed)
        missing, extra = sorted(expected - actual), sorted(actual - expected)
        if missing or extra:
            failures.append(f"{optimizer}: missing={missing}, extra={extra}")
    if failures:
        raise ValueError("incomplete seed aggregate; " + "; ".join(failures))
    return dict(indexed)


def paired_comparison(
    left: Mapping[int, float], right: Mapping[int, float]
) -> dict[str, float]:
    """Run paired t/Wilcoxon tests and compute paired Cohen's d."""
    if set(left) != set(right):
        raise ValueError("paired comparison requires identical seed sets")
    if len(left) < 2:
        raise ValueError("paired comparison requires at least two seeds")
    try:
        from scipy.stats import ttest_rel, wilcoxon
    except ImportError as exc:
        raise RuntimeError("paired tests require the benchmark extra (scipy)") from exc
    seeds = sorted(left)
    left_values = np.asarray([left[seed] for seed in seeds], dtype=np.float64)
    right_values = np.asarray([right[seed] for seed in seeds], dtype=np.float64)
    differences = left_values - right_values
    std = float(np.std(differences, ddof=1))
    effect = 0.0 if std == 0.0 else float(np.mean(differences) / std)
    t_result = ttest_rel(left_values, right_values)
    if np.all(differences == 0):
        wilcoxon_statistic, wilcoxon_pvalue = 0.0, 1.0
    else:
        w_result = wilcoxon(left_values, right_values)
        wilcoxon_statistic = float(w_result.statistic)
        wilcoxon_pvalue = float(w_result.pvalue)
    return {
        "paired_t_statistic": float(t_result.statistic),
        "paired_t_pvalue": float(t_result.pvalue),
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_pvalue": wilcoxon_pvalue,
        "cohens_d_paired": effect,
    }
