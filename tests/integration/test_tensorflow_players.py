import numpy as np
import pytest

tensorflow = pytest.importorskip("tensorflow")
keras = pytest.importorskip("keras")

from neat_optim import PlayerNEATConfig  # noqa: E402
from neat_optim.training import create_player_states, player_train_step  # noqa: E402


def test_player_train_step_updates_a_built_model() -> None:
    _ = tensorflow
    model = keras.Sequential(
        [
            keras.layers.Input((4,)),
            keras.layers.Dense(3),
        ]
    )
    x = tensorflow.constant(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=tensorflow.float32,
    )
    y = tensorflow.constant([0, 1, 2, 1], dtype=tensorflow.int32)
    _ = model(x, training=True)
    before = [variable.numpy().copy() for variable in model.trainable_variables]
    states = create_player_states(model)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction="none",
    )

    result = player_train_step(
        model,
        x,
        y,
        loss_fn,
        states,
        PlayerNEATConfig(
            learning_rate=1e-2,
            alpha=0.25,
            beta=0.9,
            sparsity_l1=1e-4,
            prune_threshold=0.0,
        ),
    )

    after = [variable.numpy() for variable in model.trainable_variables]
    assert result.loss > 0.0
    assert len(result.states) == len(model.trainable_variables)
    assert len(result.metrics) == len(model.trainable_variables)
    assert any(
        not np.allclose(before_value, after_value)
        for before_value, after_value in zip(before, after, strict=True)
    )
