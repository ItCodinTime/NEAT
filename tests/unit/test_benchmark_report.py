import json

import pytest
from benchmarks.report import main


def test_report_rejects_manifest_referencing_missing_result(tmp_path) -> None:
    plan = {
        "seeds": [1, 2],
        "optimizers": ["neat"],
        "results": ["neat-seed1/result.json", "neat-seed2/result.json"],
    }
    (tmp_path / "comparison_manifest.json").write_text(
        json.dumps(plan), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="references missing results"):
        main([str(tmp_path), "--output-dir", str(tmp_path / "report")])
