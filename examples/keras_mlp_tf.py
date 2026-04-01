"""Train a small Keras model with NEAT.

Requires:
    pip install "neat-optim[keras]" tensorflow
"""

import keras
import numpy as np

from neat_optim import NEAT


def main() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(256, 32)).astype("float32")
    y = rng.integers(0, 3, size=(256,), endpoint=False)

    model = keras.Sequential(
        [
            keras.layers.Input((32,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(3),
        ]
    )
    model.compile(
        optimizer=NEAT(learning_rate=1e-3, alpha=0.2, beta=0.9),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(x, y, epochs=2, batch_size=32, verbose=2)


if __name__ == "__main__":
    main()
