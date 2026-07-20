from __future__ import annotations

import json

import pytest
from benchmarks.sts.analyze_promotion import CANDIDATES, CONTROL, main


def _write_result(root, dataset: str, variant: str, seed: int, score: float) -> None:
    path = (
        root
        / "runs"
        / dataset
        / "multi"
        / "task_neat"
        / variant
        / f"seed_{seed}"
        / "result.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    correction = 0.0 if variant == CONTROL else 0.1
    path.write_text(
        json.dumps(
            {
                "test": None,
                "history": [
                    {
                        "validation_average_accuracy": score,
                        "correction_ratio": correction,
                    }
                    for _ in range(3)
                ],
            }
        ),
        encoding="utf-8",
    )


def test_promotion_report_applies_declared_validation_gate(tmp_path) -> None:
    datasets = ("multi_mnist", "multi_fashion_mnist")
    seeds = (101, 103)
    for dataset in datasets:
        for seed in seeds:
            _write_result(tmp_path, dataset, CONTROL, seed, 0.80)
            _write_result(tmp_path, dataset, CANDIDATES[0], seed, 0.81)
            second_score = 0.81 if dataset == "multi_mnist" else 0.79
            _write_result(tmp_path, dataset, CANDIDATES[1], seed, second_score)
            _write_result(tmp_path, dataset, CANDIDATES[2], seed, 0.70)

    main([str(tmp_path), "--seeds", "101", "103"])

    report = json.loads(
        (tmp_path / "promotion_assets" / "promotion_decision.json").read_text()
    )
    assert report["test_split_used"] is False
    assert report["decisions"][CANDIDATES[0]]["advance"] is True
    assert report["decisions"][CANDIDATES[1]]["advance"] is False
    assert report["decisions"][CANDIDATES[2]]["advance"] is False


def test_promotion_report_rejects_test_metrics(tmp_path) -> None:
    _write_result(tmp_path, "multi_mnist", CONTROL, 101, 0.80)
    path = next(tmp_path.rglob("result.json"))
    result = json.loads(path.read_text())
    result["test"] = {"average_accuracy": 0.9}
    path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(SystemExit, match="test split"):
        main([str(tmp_path), "--datasets", "multi_mnist", "--seeds", "101"])
