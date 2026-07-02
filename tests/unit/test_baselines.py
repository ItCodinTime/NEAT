from __future__ import annotations

import numpy as np
from benchmarks.baselines import pcgrad
from benchmarks.tuning import require_all_optimizers_tuned, run_tuning


def test_pcgrad_removes_pairwise_conflict_deterministically() -> None:
    gradients = np.array([[1.0, 0.0], [-1.0, 1.0]], dtype=np.float32)
    first = pcgrad(gradients, seed=7)
    second = pcgrad(gradients, seed=7)
    np.testing.assert_array_equal(first, second)
    assert np.dot(first, gradients[0]) >= 0.0
    assert np.dot(first, gradients[1]) >= 0.0


def test_tuning_records_the_same_grid_for_every_optimizer(tmp_path) -> None:
    report = run_tuning(
        task="toy",
        optimizers=["neat", "adamw"],
        search_space={"lr": [0.1, 0.01], "weight_decay": [0.0, 0.1]},
        evaluate=lambda name, config: config["lr"] - config["weight_decay"],
        output_dir=tmp_path,
    )
    assert len(report["trials"]) == 8
    require_all_optimizers_tuned(report, ["neat", "adamw"])
    assert (tmp_path / "tuning_report.md").exists()
