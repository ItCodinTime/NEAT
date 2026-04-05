"""Minimal Keras optimizer example.

Requires:
    pip install "neat-optim[keras]" tensorflow
"""

import keras
import tensorflow as tf

from neat_optim import NEAT


def main() -> None:
    variable = keras.Variable([1.0, -2.0], dtype="float32")
    gradient = keras.ops.array([0.5, -0.25], dtype="float32")
    optimizer = NEAT(learning_rate=0.1, alpha=0.25, beta=0.9)
    optimizer.apply_gradients([(gradient, variable)])
    print(tf.convert_to_tensor(variable).numpy())


if __name__ == "__main__":
    main()
