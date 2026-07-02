"""Run a declared experiment plan or regenerate paper assets from manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-manifests", action="store_true")
    parser.add_argument("--runs-dir", default="benchmark-runs/paper")
    parser.add_argument("--assets-dir", default="paper_assets")
    parser.add_argument("--plan", default="benchmarks/paper_experiments.json")
    return parser.parse_args(argv)


def _run_plan(path: Path) -> None:
    if not path.exists():
        raise SystemExit(
            "Paper experiment plan is unresolved. The author must create "
            f"{path} after resolving docs/v3_decisions.md."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    for command in payload.get("commands", []):
        if not isinstance(command, list) or not all(
            isinstance(item, str) for item in command
        ):
            raise ValueError("paper experiment commands must be JSON string arrays")
        subprocess.run(command, check=True)


def _checksums(directory: Path) -> None:
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(f"{digest}  {path.relative_to(directory)}")
    (directory / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main(argv=None) -> None:
    args = parse_args(argv)
    if not args.from_manifests:
        _run_plan(Path(args.plan))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.report",
            args.runs_dir,
            "--output-dir",
            args.assets_dir,
            "--paper",
        ],
        check=True,
    )
    _checksums(Path(args.assets_dir))


if __name__ == "__main__":
    main()
