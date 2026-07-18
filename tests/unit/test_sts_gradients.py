from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from benchmarks.sts.gradients import (  # noqa: E402
    gradient_geometry,
    mean_gradients,
    mgda_two_task,
    pcgrad_two_task,
)
from benchmarks.sts.multitask_vision import (  # noqa: E402
    PairedVisionDataset,
    dataset_spec_hash,
    make_model,
)


def test_mgda_returns_exact_minimum_norm_two_task_combination() -> None:
    rows = ((torch.tensor([1.0, 0.0]),), (torch.tensor([0.0, 1.0]),))
    combined = mgda_two_task(rows)
    torch.testing.assert_close(combined[0], torch.tensor([0.5, 0.5]))


def test_pcgrad_changes_only_conflicting_task_gradients() -> None:
    rows = ((torch.tensor([1.0, 0.0]),), (torch.tensor([-1.0, 1.0]),))
    combined = pcgrad_two_task(rows)
    raw = mean_gradients(rows)
    ratio, alignment = gradient_geometry(raw, combined)
    assert float(ratio) > 0.0
    assert 0.0 < float(alignment) <= 1.0


def test_paired_dataset_uses_declared_fixed_overlap() -> None:
    class FakeDataset:
        def __getitem__(self, index: int):
            return torch.full((1, 28, 28), float(index + 1)), index

    dataset = PairedVisionDataset(FakeDataset(), [0, 1], pairing_seed=1)
    image, first, second = dataset[0]
    assert image.shape == (1, 36, 36)
    assert int(first) == 0
    assert int(second) in {0, 1}
    assert image[:, :8, :8].eq(1.0).all()
    assert image[:, 28:, 28:].eq(float(second + 1)).all()


def test_dataset_spec_hash_and_model_shape_are_stable() -> None:
    assert dataset_spec_hash("multi_mnist") == dataset_spec_hash("multi_mnist")
    assert dataset_spec_hash("multi_mnist") != dataset_spec_hash("multi_fashion_mnist")
    outputs = make_model()(torch.zeros((2, 1, 36, 36)))
    assert [tuple(output.shape) for output in outputs] == [(2, 10), (2, 10)]
