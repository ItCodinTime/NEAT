from __future__ import annotations

import pytest
from benchmarks.stats import paired_comparison, require_complete_seeds, summarize


def test_summary_keeps_full_precision_and_bounds_mean() -> None:
    pytest.importorskip("scipy")
    result = summarize([1.0, 2.0, 4.0, 8.0, 16.0])
    assert result.count == 5
    assert result.std > 0
    assert result.ci95_low < result.mean < result.ci95_high
    assert result.minimum == 1.0
    assert result.maximum == 16.0


def test_completeness_rejects_a_missing_seed() -> None:
    rows = [
        {"optimizer": "neat", "seed": 1},
        {"optimizer": "neat", "seed": 2},
        {"optimizer": "adamw", "seed": 1},
    ]
    with pytest.raises(ValueError, match=r"adamw: missing=\[2\]"):
        require_complete_seeds(rows, [1, 2])


def test_paired_comparison_matches_by_seed_not_input_order() -> None:
    pytest.importorskip("scipy")
    result = paired_comparison({3: 3.0, 1: 1.0, 2: 2.0}, {1: 0.0, 2: 0.5, 3: 1.0})
    assert result["cohens_d_paired"] > 0
    assert 0 <= result["wilcoxon_pvalue"] <= 1
