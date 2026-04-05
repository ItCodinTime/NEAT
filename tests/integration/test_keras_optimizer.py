import numpy as np
import pytest

tensorflow = pytest.importorskip("tensorflow")
keras = pytest.importorskip("keras")

from neat_optim import NEAT  # noqa: E402
from neat_optim.config import NEATConfig  # noqa: E402
from neat_optim.engine.reference import neat_step_reference  # noqa: E402
from neat_optim.state import ArrayState  # noqa: E402


def _to_numpy(tensor) -> np.ndarray:
    return tensorflow.convert_to_tensor(tensor).numpy()


def _run_reference(
    initial_param: np.ndarray,
    gradients: list[np.ndarray],
    **config_kwargs,
):
    param = initial_param.astype(np.float32, copy=True)
    state = ArrayState.zeros_like(param)
    result = None
    config = NEATConfig(native="never", **config_kwargs)
    for gradient in gradients:
        result = neat_step_reference(param, gradient, state, config)
        param = result.param
        state = result.state
    assert result is not None
    return result


def _run_keras(
    initial_param: np.ndarray,
    gradients: list[np.ndarray],
    **optimizer_kwargs,
):
    variable = keras.Variable(initial_param.tolist(), dtype="float32")
    optimizer = NEAT(**optimizer_kwargs)
    for gradient in gradients:
        optimizer.apply_gradients(
            [(keras.ops.array(gradient, dtype="float32"), variable)]
        )
    return optimizer, _to_numpy(variable)


def test_keras_optimizer_can_apply_gradients() -> None:
    _ = tensorflow
    variable = keras.Variable([1.0, -2.0], dtype="float32")
    gradient = keras.ops.array([0.5, -0.25], dtype="float32")
    optimizer = NEAT(learning_rate=0.1, alpha=0.25, beta=0.9)

    optimizer.apply_gradients([(gradient, variable)])

    values = _to_numpy(variable)
    assert values.shape == (2,)
    assert optimizer.variables


def test_keras_optimizer_config_round_trip() -> None:
    optimizer = NEAT(
        learning_rate=1e-3,
        alpha=0.2,
        beta=0.8,
        nce_mode="cosine",
        weight_decay=1e-4,
        decouple_weight_decay=False,
        nce_clip_ratio=0.5,
        sparsity_l1=1e-4,
        prune_threshold=1e-3,
        opponent_source="gradient_ema",
        opponent_ema_decay=0.8,
        opponent_blend=0.4,
        correction_warmup_steps=2,
        conflict_threshold=0.1,
        adaptive_correction=True,
        adaptive_correction_decay=0.75,
        adaptive_correction_min_scale=1.0,
        adaptive_correction_max_scale=2.0,
    )
    clone = NEAT.from_config(optimizer.get_config())
    assert clone.get_config()["alpha"] == pytest.approx(0.2)
    assert clone.get_config()["beta"] == pytest.approx(0.8)
    assert clone.get_config()["nce_mode"] == "cosine"
    assert clone.get_config()["weight_decay"] == pytest.approx(1e-4)
    assert clone.get_config()["decouple_weight_decay"] is False
    assert clone.get_config()["nce_clip_ratio"] == pytest.approx(0.5)
    assert clone.get_config()["sparsity_l1"] == pytest.approx(1e-4)
    assert clone.get_config()["prune_threshold"] == pytest.approx(1e-3)
    assert clone.get_config()["opponent_source"] == "gradient_ema"
    assert clone.get_config()["opponent_ema_decay"] == pytest.approx(0.8)
    assert clone.get_config()["opponent_blend"] == pytest.approx(0.4)
    assert clone.get_config()["correction_warmup_steps"] == 2
    assert clone.get_config()["conflict_threshold"] == pytest.approx(0.1)
    assert clone.get_config()["adaptive_correction"] is True
    assert clone.get_config()["adaptive_correction_decay"] == pytest.approx(0.75)
    assert clone.get_config()["adaptive_correction_min_scale"] == pytest.approx(1.0)
    assert clone.get_config()["adaptive_correction_max_scale"] == pytest.approx(2.0)


def test_keras_optimizer_matches_reference_projection_mode() -> None:
    initial_param = np.array([1.0, -2.0], dtype=np.float32)
    gradients = [
        np.array([0.5, -0.25], dtype=np.float32),
        np.array([-0.5, 0.25], dtype=np.float32),
    ]

    optimizer, keras_param = _run_keras(
        initial_param,
        gradients,
        learning_rate=0.1,
        alpha=0.25,
        beta=0.9,
        nce_mode="projection",
    )
    reference = _run_reference(
        initial_param,
        gradients,
        learning_rate=0.1,
        alpha=0.25,
        beta=0.9,
        nce_mode="projection",
    )

    np.testing.assert_allclose(keras_param, reference.param, atol=1e-6)
    np.testing.assert_allclose(
        _to_numpy(optimizer.momentums[0]),
        reference.state.momentum,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        _to_numpy(optimizer.nces[0]),
        reference.state.nce,
        atol=1e-6,
    )


def test_keras_optimizer_matches_reference_with_weight_decay_modes() -> None:
    initial_param = np.array([1.0, -2.0], dtype=np.float32)
    gradients = [np.array([0.5, -0.25], dtype=np.float32)]

    decoupled_optimizer, decoupled_param = _run_keras(
        initial_param,
        gradients,
        learning_rate=0.1,
        alpha=0.25,
        beta=0.9,
        weight_decay=1e-2,
        decouple_weight_decay=True,
    )
    decoupled_reference = _run_reference(
        initial_param,
        gradients,
        learning_rate=0.1,
        alpha=0.25,
        beta=0.9,
        weight_decay=1e-2,
        decouple_weight_decay=True,
    )
    np.testing.assert_allclose(decoupled_param, decoupled_reference.param, atol=1e-6)
    np.testing.assert_allclose(
        _to_numpy(decoupled_optimizer.momentums[0]),
        decoupled_reference.state.momentum,
        atol=1e-6,
    )

    coupled_optimizer, coupled_param = _run_keras(
        initial_param,
        gradients,
        learning_rate=0.1,
        alpha=0.25,
        beta=0.9,
        weight_decay=1e-2,
        decouple_weight_decay=False,
    )
    coupled_reference = _run_reference(
        initial_param,
        gradients,
        learning_rate=0.1,
        alpha=0.25,
        beta=0.9,
        weight_decay=1e-2,
        decouple_weight_decay=False,
    )
    np.testing.assert_allclose(coupled_param, coupled_reference.param, atol=1e-6)
    np.testing.assert_allclose(
        _to_numpy(coupled_optimizer.momentums[0]),
        coupled_reference.state.momentum,
        atol=1e-6,
    )


def test_keras_optimizer_matches_reference_when_nce_is_disabled() -> None:
    initial_param = np.array([1.0, -2.0], dtype=np.float32)
    gradients = [
        np.array([0.5, -0.25], dtype=np.float32),
        np.array([-0.5, 0.25], dtype=np.float32),
    ]

    optimizer, keras_param = _run_keras(
        initial_param,
        gradients,
        learning_rate=0.1,
        alpha=0.25,
        beta=0.9,
        nce_mode="off",
    )
    reference = _run_reference(
        initial_param,
        gradients,
        learning_rate=0.1,
        alpha=0.25,
        beta=0.9,
        nce_mode="off",
    )

    np.testing.assert_allclose(keras_param, reference.param, atol=1e-6)
    np.testing.assert_allclose(
        _to_numpy(optimizer.nces[0]),
        np.zeros_like(initial_param),
        atol=1e-6,
    )


def test_keras_optimizer_matches_reference_with_sparsity_controls() -> None:
    initial_param = np.array([0.02, -0.02, 0.5], dtype=np.float32)
    gradients = [np.zeros_like(initial_param)]

    optimizer, keras_param = _run_keras(
        initial_param,
        gradients,
        learning_rate=1.0,
        alpha=0.0,
        beta=0.0,
        nce_mode="off",
        sparsity_l1=0.01,
        prune_threshold=0.03,
    )
    reference = _run_reference(
        initial_param,
        gradients,
        learning_rate=1.0,
        alpha=0.0,
        beta=0.0,
        nce_mode="off",
        sparsity_l1=0.01,
        prune_threshold=0.03,
    )

    np.testing.assert_allclose(keras_param, reference.param, atol=1e-6)
    np.testing.assert_allclose(
        _to_numpy(optimizer.momentums[0]),
        reference.state.momentum,
        atol=1e-6,
    )


def test_keras_optimizer_matches_reference_with_previous_gradient_opponent() -> None:
    initial_param = np.array([1.0, 2.0], dtype=np.float32)
    gradients = [
        np.array([-1.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
    ]

    optimizer, keras_param = _run_keras(
        initial_param,
        gradients,
        learning_rate=0.1,
        alpha=0.5,
        beta=0.0,
        opponent_source="previous_gradient",
    )
    reference = _run_reference(
        initial_param,
        gradients,
        learning_rate=0.1,
        alpha=0.5,
        beta=0.0,
        opponent_source="previous_gradient",
    )

    np.testing.assert_allclose(keras_param, reference.param, atol=1e-6)
    np.testing.assert_allclose(
        _to_numpy(optimizer.nces[0]),
        reference.state.nce,
        atol=1e-6,
    )
    snapshot = optimizer.diagnostic_snapshot()
    assert snapshot["mean_conflict_ratio"] >= 0.0
    assert snapshot["mean_correction_ratio"] >= 0.0


def test_keras_optimizer_matches_reference_with_blended_adaptive_opponent() -> None:
    initial_param = np.array([1.0, 2.0], dtype=np.float32)
    gradients = [
        np.array([0.5, 0.0], dtype=np.float32),
        np.array([-1.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
    ]

    optimizer, keras_param = _run_keras(
        initial_param,
        gradients,
        learning_rate=0.1,
        alpha=0.5,
        beta=0.0,
        opponent_source="blended",
        opponent_blend=0.25,
        adaptive_correction=True,
        adaptive_correction_decay=0.5,
        adaptive_correction_min_scale=1.0,
        adaptive_correction_max_scale=2.5,
    )
    reference = _run_reference(
        initial_param,
        gradients,
        learning_rate=0.1,
        alpha=0.5,
        beta=0.0,
        opponent_source="blended",
        opponent_blend=0.25,
        adaptive_correction=True,
        adaptive_correction_decay=0.5,
        adaptive_correction_min_scale=1.0,
        adaptive_correction_max_scale=2.5,
    )

    np.testing.assert_allclose(keras_param, reference.param, atol=1e-6)
    np.testing.assert_allclose(
        _to_numpy(optimizer.nces[0]),
        reference.state.nce,
        atol=1e-6,
    )
    snapshot = optimizer.diagnostic_snapshot()
    assert snapshot["mean_correction_ratio"] > 0.0


def test_keras_optimizer_reports_zero_correction_when_nce_is_disabled() -> None:
    initial_param = np.array([1.0, -2.0], dtype=np.float32)
    gradients = [
        np.array([0.5, -0.25], dtype=np.float32),
        np.array([-0.5, 0.25], dtype=np.float32),
    ]

    optimizer, _ = _run_keras(
        initial_param,
        gradients,
        learning_rate=0.1,
        alpha=0.25,
        beta=0.9,
        nce_mode="off",
    )
    snapshot = optimizer.diagnostic_snapshot()

    assert snapshot["mean_conflict_ratio"] == pytest.approx(0.0, abs=1e-8)
    assert snapshot["mean_correction_ratio"] == pytest.approx(0.0, abs=1e-8)
    assert snapshot["correction_active_fraction"] == pytest.approx(0.0, abs=1e-8)
