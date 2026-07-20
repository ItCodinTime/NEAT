"""Portable interpretation and reporting for NEAT optimizer diagnostics."""

from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def diagnose_snapshot(snapshot: Mapping[str, float]) -> dict[str, Any]:
    """Classify optimizer activity without claiming a performance effect."""
    conflict = float(snapshot.get("mean_conflict_ratio", 0.0))
    correction = float(snapshot.get("mean_correction_ratio", 0.0))
    active = float(snapshot.get("correction_active_fraction", 0.0))
    alignment = float(snapshot.get("mean_update_alignment", 1.0))

    if conflict < 0.01 and active < 0.1:
        status = "inactive"
        explanation = (
            "The correction was rarely engaged. This workload provides little "
            "evidence for or against conflict correction."
        )
    elif correction >= 0.1 or alignment < 0.95:
        status = "strong"
        explanation = (
            "The correction materially changed the gradient direction or magnitude. "
            "Compare against a correction-off control before attributing performance."
        )
    else:
        status = "active"
        explanation = (
            "The correction engaged but remained bounded. A matched correction-off "
            "control is required to estimate its performance contribution."
        )
    return {
        "status": status,
        "explanation": explanation,
        "signals": {
            "mean_conflict_ratio": conflict,
            "mean_correction_ratio": correction,
            "correction_active_fraction": active,
            "mean_update_alignment": alignment,
        },
    }


def _numeric_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    normalized = []
    for row in rows:
        values = {}
        for key, value in row.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            number = float(value)
            if math.isfinite(number):
                values[str(key)] = number
        normalized.append(values)
    return normalized


def _diagnostic_snapshot(row: Mapping[str, float]) -> dict[str, float]:
    return {
        key.removeprefix("neat_"): value
        for key, value in row.items()
        if key.startswith("neat_")
    }


def write_training_report(
    rows: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write epoch diagnostics as JSON or a self-contained HTML report."""
    destination = Path(path)
    if destination.suffix.lower() not in {".json", ".html"}:
        raise ValueError("training report path must end in .json or .html")
    destination.parent.mkdir(parents=True, exist_ok=True)
    numeric_rows = _numeric_rows(rows)
    latest = _diagnostic_snapshot(numeric_rows[-1]) if numeric_rows else {}
    interpretation = diagnose_snapshot(latest)
    payload = {
        "schema_version": 1,
        "metadata": dict(metadata or {}),
        "epochs": numeric_rows,
        "interpretation": interpretation,
        "claim_limit": (
            "Optimizer diagnostics describe mechanism activity, not causal "
            "performance improvement. Use a matched correction-off control."
        ),
    }
    if destination.suffix.lower() == ".json":
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return destination

    columns = sorted({key for row in numeric_rows for key in row})
    header = "".join(f"<th>{html.escape(key)}</th>" for key in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{row.get(key, float('nan')):.6g}</td>" for key in columns)
        + "</tr>"
        for row in numeric_rows
    )
    title = html.escape(str(payload["metadata"].get("title", "NEAT training report")))
    status = html.escape(str(interpretation["status"]).upper())
    explanation = html.escape(str(interpretation["explanation"]))
    claim_limit = html.escape(str(payload["claim_limit"]))
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body {{ color: #202124; font: 15px system-ui, sans-serif; margin: 0; }}
main {{ margin: 0 auto; max-width: 1120px; padding: 32px 20px; }}
h1 {{ font-size: 28px; margin: 0 0 20px; }}
.status {{ border-left: 4px solid #137333; padding: 2px 16px; }}
.status strong {{ color: #137333; }}
.limit {{ color: #5f6368; margin: 24px 0; }}
.table-wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; font-variant-numeric: tabular-nums; width: 100%; }}
th, td {{ border-bottom: 1px solid #dadce0; padding: 9px 10px; text-align: right; }}
th {{ background: #f8f9fa; position: sticky; top: 0; }}
</style>
</head>
<body><main>
<h1>{title}</h1>
<section class="status"><strong>{status}</strong><p>{explanation}</p></section>
<p class="limit">{claim_limit}</p>
<div class="table-wrap"><table>
<thead><tr>{header}</tr></thead><tbody>{body}</tbody>
</table></div>
</main></body>
</html>
"""
    destination.write_text(document)
    return destination
