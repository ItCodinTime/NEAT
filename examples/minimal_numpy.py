from __future__ import annotations

import numpy as np

from neat_optim import NEATConfig
from neat_optim.engine.functional import neat_step
from neat_optim.state import ArrayState


def main() -> None:
    param = np.array([1.0, -2.0], dtype=np.float32)
    grad = np.array([0.5, -0.25], dtype=np.float32)
    state = ArrayState.zeros_like(param)
    config = NEATConfig(learning_rate=0.1, alpha=0.25, beta=0.9)

    for step in range(3):
        result = neat_step(param, grad, state, config)
        param = result.param
        state = result.state
        print(step, param, result.metrics)


if __name__ == "__main__":
    main()
