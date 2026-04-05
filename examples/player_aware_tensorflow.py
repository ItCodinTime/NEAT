"""Explicit player-aware NEAT training with per-example TensorFlow gradients.

Requires:
    pip install "neat-optim[keras]" tensorflow
"""

import keras
import numpy as np
import tensorflow as tf

from neat_optim import PlayerNEATConfig
from neat_optim.training import create_player_states, player_train_step


def main() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 8)).astype("float32")
    y = rng.integers(0, 3, size=(32,), endpoint=False)

    model = keras.Sequential(
        [
            keras.layers.Input((8,)),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(3),
        ]
    )
    _ = model(tf.convert_to_tensor(x), training=True)
    states = create_player_states(model)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction="none",
    )
    config = PlayerNEATConfig(
        learning_rate=1e-2,
        alpha=0.25,
        beta=0.9,
        sparsity_l1=1e-4,
        prune_threshold=1e-3,
    )

    for step in range(1, 6):
        result = player_train_step(model, x, y, loss_fn, states, config)
        states = result.states
        mean_conflict = float(
            np.mean([metric.mean_player_conflict for metric in result.metrics])
        )
        mean_active = float(
            np.mean([metric.active_fraction for metric in result.metrics])
        )
        print(
            f"step={step} loss={result.loss:.4f} "
            f"mean_conflict={mean_conflict:.4f} active_fraction={mean_active:.4f}"
        )


if __name__ == "__main__":
    main()
