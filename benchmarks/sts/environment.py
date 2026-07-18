"""Capture a secret-free environment and dataset provenance snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

from benchmarks.manifest import package_versions, write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/sts")
    parser.add_argument("--output-dir", default="benchmark-runs/sts/environment")
    return parser.parse_args(argv)


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def file_hashes(root: Path) -> dict[str, str]:
    hashes = {}
    if not root.exists():
        return hashes
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[str(path.relative_to(root))] = digest.hexdigest()
    return hashes


def main(argv=None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    payload = {
        "schema_version": 1,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "packages": package_versions(
            ("neat-optim", "numpy", "torch", "torchvision", "scipy", "matplotlib")
        ),
        "git_sha": command_output(["git", "rev-parse", "HEAD"]).strip(),
        "git_status": command_output(["git", "status", "--porcelain"]),
        "uname": command_output(["uname", "-a"]),
        "cpu": command_output(["lscpu"]),
        "memory": command_output(["free", "-h"]),
        "disk": command_output(["df", "-h", "/"]),
        "pip_freeze": command_output([sys.executable, "-m", "pip", "freeze"]),
        "dataset_sha256": file_hashes(Path(args.data_dir)),
    }
    write_json(output_dir / "environment.json", payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in payload.items():
        if isinstance(value, str):
            lines.extend((f"## {key}", value.rstrip(), ""))
        elif key != "dataset_sha256":
            lines.extend((f"## {key}", json.dumps(value, indent=2, sort_keys=True), ""))
    (output_dir / "environment.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
