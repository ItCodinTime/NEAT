from __future__ import annotations

import os
import re
from pathlib import Path


def main() -> None:
    required = [
        "README.md",
        "CHANGELOG.md",
        "CITATION.cff",
        "GOVERNANCE.md",
        "SUPPORT.md",
        "pyproject.toml",
        "src/neat_optim/_version.py",
        ".github/workflows/ci.yml",
        ".github/workflows/release.yml",
    ]
    missing = [path for path in required if not Path(path).exists()]
    if missing:
        raise SystemExit(f"missing required release files: {missing}")

    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject, flags=re.MULTILINE)
    if match is None:
        raise SystemExit("project version is missing from pyproject.toml")
    version = match.group(1)
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise SystemExit(f"release version must be stable SemVer, got {version!r}")

    changelog = Path("CHANGELOG.md").read_text(encoding="utf-8")
    if f"## [{version}]" not in changelog:
        raise SystemExit(f"CHANGELOG.md has no section for {version}")

    citation = Path("CITATION.cff").read_text(encoding="utf-8")
    if f"version: {version}" not in citation:
        raise SystemExit(f"CITATION.cff version does not match {version}")

    release_tag = os.environ.get("NEAT_RELEASE_TAG")
    if release_tag and release_tag != f"v{version}":
        raise SystemExit(
            f"release tag {release_tag!r} does not match project version v{version}"
        )
    print(f"release check passed for {version}")


if __name__ == "__main__":
    main()
