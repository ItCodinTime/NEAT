from __future__ import annotations

import json
from argparse import Namespace

import pytest
from benchmarks.sts.analyze_severity import _load_rows
from benchmarks.sts.multitask_vision import dataset_spec_hash, result_path
from benchmarks.sts.run_suite import _matrix


def test_dataset_hash_changes_with_overlap_offset() -> None:
    assert dataset_spec_hash("multi_mnist", 0) != dataset_spec_hash("multi_mnist", 8)


def test_nondefault_overlap_has_distinct_result_path(tmp_path) -> None:
    shared = {
        "output_dir": str(tmp_path),
        "dataset": "multi_mnist",
        "setting": "multi",
        "method": "task_neat",
        "neat_variant": "adam_alpha_0_5",
        "seed": 101,
    }
    default = result_path(Namespace(**shared, overlap_offset=8))
    severe = result_path(Namespace(**shared, overlap_offset=0))

    assert default != severe
    assert "overlap_offset_0" in str(severe)


def test_severity_matrix_is_paired_and_validation_only() -> None:
    args = Namespace(
        phase="severity",
        datasets=["multi_mnist"],
        overlap_offsets=[0, 4, 8],
        seeds=[101, 103],
        epochs=5,
    )
    rows = list(_matrix(args, {"multi_mnist": {"adam": 0.001}}))

    assert len(rows) == 12
    assert {row[3] for row in rows} == {"adam_alpha_0", "adam_alpha_0_5"}
    assert {row[-1] for row in rows} == {0, 4, 8}


def test_severity_analysis_computes_paired_gain(tmp_path) -> None:
    for variant, accuracy, correction in (
        ("adam_alpha_0", 0.8, 0.0),
        ("adam_alpha_0_5", 0.81, 0.04),
    ):
        path = tmp_path / "runs" / variant / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "config": {
                        "dataset": "multi_mnist",
                        "overlap_offset": 4,
                        "seed": 101,
                        "neat_variant": variant,
                    },
                    "history": [
                        {
                            "validation_average_accuracy": accuracy,
                            "task_gradient_cosine": -0.2,
                            "correction_ratio": correction,
                        }
                    ],
                    "test": None,
                }
            )
        )

    rows = _load_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["paired_validation_gain"] == pytest.approx(0.01)
    assert rows[0]["correction_ratio"] == pytest.approx(0.04)
