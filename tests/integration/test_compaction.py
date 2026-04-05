import numpy as np
import pytest

tensorflow = pytest.importorskip("tensorflow")
keras = pytest.importorskip("keras")

from neat_optim import compact_dense_model, search_compact_dense_model  # noqa: E402


def test_compact_dense_model_removes_dead_hidden_unit_exactly() -> None:
    _ = tensorflow
    model = keras.Sequential(
        [
            keras.layers.Input((2,)),
            keras.layers.Dense(3, activation="relu", use_bias=True, name="hidden"),
            keras.layers.Dense(1, activation="linear", use_bias=True, name="output"),
        ]
    )
    model(np.zeros((1, 2), dtype=np.float32))
    model.layers[0].set_weights(
        [
            np.array([[1.0, 0.0, 0.5], [0.0, 0.0, -0.5]], dtype=np.float32),
            np.array([0.0, 0.0, 0.1], dtype=np.float32),
        ]
    )
    model.layers[1].set_weights(
        [
            np.array([[1.0], [0.0], [0.25]], dtype=np.float32),
            np.array([0.2], dtype=np.float32),
        ]
    )

    sample = np.array([[1.0, 2.0], [0.25, -0.5]], dtype=np.float32)
    baseline = model(sample, training=False).numpy()
    compacted, report = compact_dense_model(model, unit_threshold=0.0)
    compacted_output = compacted(sample, training=False).numpy()

    np.testing.assert_allclose(compacted_output, baseline, atol=1e-6)
    assert report.original_hidden_units == (3,)
    assert report.compacted_hidden_units == (2,)
    assert report.compacted_param_count < report.original_param_count


def test_compact_dense_model_preserves_output_shape() -> None:
    _ = tensorflow
    model = keras.Sequential(
        [
            keras.layers.Input((4,)),
            keras.layers.Dense(4, activation="relu"),
            keras.layers.Dense(3, activation="relu"),
            keras.layers.Dense(2),
        ]
    )
    model(np.zeros((1, 4), dtype=np.float32))
    compacted, _ = compact_dense_model(model, unit_threshold=1e6)

    assert compacted.output_shape == model.output_shape
    assert compacted.input_shape == model.input_shape


def test_search_compact_dense_model_selects_smallest_no_loss_candidate() -> None:
    _ = tensorflow
    model = keras.Sequential(
        [
            keras.layers.Input((2,)),
            keras.layers.Dense(3, activation="relu", use_bias=True, name="hidden"),
            keras.layers.Dense(1, activation="linear", use_bias=True, name="output"),
        ]
    )
    model(np.zeros((1, 2), dtype=np.float32))
    model.layers[0].set_weights(
        [
            np.array([[1.0, 0.0, 0.5], [0.0, 0.0, -0.5]], dtype=np.float32),
            np.array([0.0, 0.0, 0.1], dtype=np.float32),
        ]
    )
    model.layers[1].set_weights(
        [
            np.array([[1.0], [0.0], [0.25]], dtype=np.float32),
            np.array([0.2], dtype=np.float32),
        ]
    )
    sample = np.array([[1.0, 2.0], [0.25, -0.5]], dtype=np.float32)
    baseline = model(sample, training=False).numpy()

    def scorer(candidate) -> float:
        output = candidate(sample, training=False).numpy()
        return float(-np.max(np.abs(output - baseline)))

    compacted, result = search_compact_dense_model(
        model,
        thresholds=(0.0, 1.0),
        scorer=scorer,
    )

    compacted_output = compacted(sample, training=False).numpy()
    np.testing.assert_allclose(compacted_output, baseline, atol=1e-6)
    assert result.accepted is True
    assert result.threshold == pytest.approx(0.0)
    assert result.report is not None
    assert result.report.compacted_param_count < result.report.original_param_count
