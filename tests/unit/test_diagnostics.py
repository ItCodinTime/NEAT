from __future__ import annotations

import json

import pytest

from neat_optim.diagnostics import diagnose_snapshot, write_training_report


@pytest.mark.parametrize(
    ("snapshot", "expected"),
    [
        (
            {"mean_conflict_ratio": 0.002, "correction_active_fraction": 0.05},
            "inactive",
        ),
        (
            {
                "mean_conflict_ratio": 0.05,
                "mean_correction_ratio": 0.02,
                "correction_active_fraction": 0.5,
                "mean_update_alignment": 0.99,
            },
            "active",
        ),
        (
            {
                "mean_conflict_ratio": 0.2,
                "mean_correction_ratio": 0.12,
                "correction_active_fraction": 0.8,
                "mean_update_alignment": 0.9,
            },
            "strong",
        ),
    ],
)
def test_diagnose_snapshot_classifies_activity(snapshot, expected) -> None:
    assert diagnose_snapshot(snapshot)["status"] == expected


def test_write_training_report_json_preserves_claim_limit(tmp_path) -> None:
    path = write_training_report(
        [
            {
                "epoch": 1,
                "loss": 0.5,
                "neat_mean_conflict_ratio": 0.02,
                "neat_mean_correction_ratio": 0.01,
                "neat_correction_active_fraction": 0.4,
            }
        ],
        tmp_path / "report.json",
        metadata={"title": "Trial A"},
    )

    payload = json.loads(path.read_text())
    assert payload["metadata"]["title"] == "Trial A"
    assert payload["epochs"][0]["loss"] == pytest.approx(0.5)
    assert payload["interpretation"]["status"] == "active"
    assert "not causal" in payload["claim_limit"]


def test_write_training_report_html_escapes_metadata(tmp_path) -> None:
    path = write_training_report(
        [{"epoch": 1, "neat_mean_conflict_ratio": 0.0}],
        tmp_path / "report.html",
        metadata={"title": "<unsafe>"},
    )

    document = path.read_text()
    assert "&lt;unsafe&gt;" in document
    assert "<unsafe>" not in document


def test_write_training_report_rejects_unknown_extension(tmp_path) -> None:
    with pytest.raises(ValueError, match=".json or .html"):
        write_training_report([], tmp_path / "report.txt")
