import pytest

tensorflow = pytest.importorskip("tensorflow")
keras = pytest.importorskip("keras")

from neat_optim import NEAT  # noqa: E402


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
    optimizer = NEAT(learning_rate=1e-3, alpha=0.2, beta=0.8)
    clone = NEAT.from_config(optimizer.get_config())
    assert clone.get_config()["alpha"] == pytest.approx(0.2)
