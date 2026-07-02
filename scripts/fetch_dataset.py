"""Download a dataset artifact and verify its required SHA-256 checksum."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("destination")
    parser.add_argument("--sha256", required=True)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, destination: Path, expected_sha256: str) -> Path:
    """Download to a temporary file, verify, then atomically install it."""
    if len(expected_sha256) != 64:
        raise ValueError("SHA-256 must contain exactly 64 hexadecimal characters")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=destination.name, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        with urllib.request.urlopen(url) as response:  # noqa: S310
            shutil.copyfileobj(response, temporary)
    actual = sha256(temporary_path)
    if actual.lower() != expected_sha256.lower():
        temporary_path.unlink()
        raise ValueError(f"checksum mismatch: expected {expected_sha256}, got {actual}")
    temporary_path.replace(destination)
    return destination


def main(argv=None) -> None:
    args = parse_args(argv)
    fetch(args.url, Path(args.destination), args.sha256)


if __name__ == "__main__":
    main()
