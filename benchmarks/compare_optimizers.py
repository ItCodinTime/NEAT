from __future__ import annotations

import json
import time

import numpy as np

from neat_optim import NEATConfig
from neat_optim.engine.functional import neat_step
from neat_optim.state import ArrayState


def _run_neat(steps: int = 200) -> dict[str, float]:
    rng = np.random.default_rng(7)
    param = rng.normal(size=(1024,)).astype(np.float32)
    target = np.zeros_like(param)
    state = ArrayState.zeros_like(param)
    config = NEATConfig(learning_rate=1e-2, alpha=0.25, beta=0.9)

    start = time.perf_counter()
    for _ in range(steps):
        grad = param - target
        result = neat_step(param, grad, state, config)
        param = result.param
        state = result.state
    elapsed = time.perf_counter() - start
    return {
        "optimizer": "neat",
        "steps": float(steps),
        "final_loss": float(np.mean(np.square(param - target))),
        "seconds": elapsed,
    }


def _run_sgd(steps: int = 200, learning_rate: float = 1e-2) -> dict[str, float]:
    rng = np.random.default_rng(7)
    param = rng.normal(size=(1024,)).astype(np.float32)
    target = np.zeros_like(param)
    velocity = np.zeros_like(param)

    start = time.perf_counter()
    for _ in range(steps):
        grad = param - target
        velocity = 0.9 * velocity + 0.1 * grad
        param = param - (learning_rate * velocity)
    elapsed = time.perf_counter() - start
    return {
        "optimizer": "sgd_momentum",
        "steps": float(steps),
        "final_loss": float(np.mean(np.square(param - target))),
        "seconds": elapsed,
    }


def main() -> None:
    rows = [_run_sgd(), _run_neat()]
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
