from __future__ import annotations

from pathlib import Path


def main() -> None:
    required = [
        "README.md",
        "CHANGELOG.md",
        "pyproject.toml",
        "src/neat_optim/_version.py",
    ]
    missing = [path for path in required if not Path(path).exists()]
    if missing:
        raise SystemExit(f"missing required release files: {missing}")
    print("release check passed")


if __name__ == "__main__":
    main()
