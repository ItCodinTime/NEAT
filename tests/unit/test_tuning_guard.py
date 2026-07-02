import pytest
from benchmarks.report import _require_uniform_tuning, _row


def test_report_rows_preserve_tuning_status(tmp_path) -> None:
    manifest = {
        "config": {
            "optimizer": "neat",
            "seed": 1,
            "task": "toy",
            "tuning_status": "validation_selected",
        },
        "history": [{"eval_metric": 1.0}],
        "wall_time_seconds": 1.0,
    }
    row = _row(tmp_path / "result.json", manifest)
    assert row["tuning_status"] == "validation_selected"


def test_tuning_status_mismatch_is_not_silently_coercible() -> None:
    with pytest.raises(ValueError, match="mixes tuning statuses"):
        _require_uniform_tuning(
            "toy",
            [
                {"tuning_status": "validation_selected"},
                {"tuning_status": "untuned_shared"},
            ],
        )
