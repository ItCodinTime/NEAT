import numpy as np
import pytest

tensorflow = pytest.importorskip("tensorflow")
keras = pytest.importorskip("keras")

from neat_optim import NEAT  # noqa: E402
from neat_optim.config import NEATConfig  # noqa: E402
from neat_optim.engine.reference import neat_step_reference  # noqa: E402
from neat_optim.state import ArrayState  # noqa: E402


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
    return optimizer, keras.ops.convert_to_numpy(variable)


def test_keras_optimizer_can_apply_gradients() -> None:
    _ = tensorflow
    variable = keras.Variable([1.0, -2.0], dtype="float32")
    gradient = keras.ops.array([0.5, -0.25], dtype="float32")
    optimizer = NEAT(learning_rate=0.1, alpha=0.25, beta=0.9)

    optimizer.apply_gradients([(gradient, variable)])

    values = keras.ops.convert_to_numpy(variable)
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
    )
    clone = NEAT.from_config(optimizer.get_config())
    assert clone.get_config()["alpha"] == pytest.approx(0.2)
    assert clone.get_config()["beta"] == pytest.approx(0.8)
    assert clone.get_config()["nce_mode"] == "cosine"
    assert clone.get_config()["weight_decay"] == pytest.approx(1e-4)
    assert clone.get_config()["decouple_weight_decay"] is False
    assert clone.get_config()["nce_clip_ratio"] == pytest.approx(0.5)


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
        keras.ops.convert_to_numpy(optimizer.momentums[0]),
        reference.state.momentum,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(optimizer.nces[0]),
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
        keras.ops.convert_to_numpy(decoupled_optimizer.momentums[0]),
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
        keras.ops.convert_to_numpy(coupled_optimizer.momentums[0]),
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
        keras.ops.convert_to_numpy(optimizer.nces[0]),
        np.zeros_like(initial_param),
        atol=1e-6,
    )
