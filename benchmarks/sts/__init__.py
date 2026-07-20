"""Regeneron STS multi-task benchmark and reporting tools."""

from __future__ import annotations

METHODS = ("adam", "pcgrad", "mgda", "neat")
EXPERIMENTAL_METHODS = ("lion", "task_neat")
ALL_METHODS = METHODS + EXPERIMENTAL_METHODS
DATASETS = ("multi_mnist", "multi_fashion_mnist")
PAPER_SEEDS = (7, 11, 19, 23, 29)
