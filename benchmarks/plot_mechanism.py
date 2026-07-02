"""Plot loss and mechanism traces from a benchmark result manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result")
    parser.add_argument("--output", default="mechanism_traces.pdf")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        import matplotlib.pyplot as plt

        from benchmarks.style import PALETTE, apply_style
    except ImportError as exc:
        raise SystemExit("mechanism plots require the benchmark extra") from exc
    manifest = json.loads(Path(args.result).read_text(encoding="utf-8"))
    history = manifest["history"]
    required = {
        "train_loss",
        "conflict_coefficient",
        "correction_ratio",
        "alignment_ema",
        "gate_active",
    }
    missing = required - set(history[0])
    if missing:
        raise ValueError(f"manifest lacks mechanism traces: {sorted(missing)}")
    apply_style()
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 8))
    x = [row["epoch"] for row in history]
    axes[0].plot(x, [row["train_loss"] for row in history], color=PALETTE[0])
    axes[0].set(ylabel="Mean loss")
    axes[1].plot(
        x,
        [row["conflict_coefficient"] for row in history],
        label="conflict coefficient",
        color=PALETTE[1],
    )
    axes[1].plot(
        x,
        [row["correction_ratio"] for row in history],
        label="correction / gradient norm",
        color=PALETTE[2],
    )
    axes[1].plot(
        x,
        [row["alignment_ema"] for row in history],
        label="alignment EMA",
        color=PALETTE[3],
    )
    axes[1].legend()
    running_frequency = []
    active = 0.0
    for index, row in enumerate(history, 1):
        active += row["gate_active"]
        running_frequency.append(active / index)
    axes[2].plot(x, running_frequency, color=PALETTE[4])
    axes[2].set(xlabel="Step", ylabel="Gate activation frequency")
    figure.tight_layout()
    figure.savefig(args.output)


if __name__ == "__main__":
    main()
