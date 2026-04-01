import numpy as np

from neat_optim.config import NEATConfig
from neat_optim.engine.reference import neat_step_reference
from neat_optim.state import ArrayState


def test_worked_example_matches_documented_math() -> None:
    param = np.array([1.0, 2.0], dtype=np.float32)
    grad = np.array([1.0, 0.0], dtype=np.float32)
    state = ArrayState(
        momentum=np.array([-1.0, 0.0], dtype=np.float32),
        nce=np.array([0.0, 0.0], dtype=np.float32),
        step=0,
    )
    config = NEATConfig(
        learning_rate=0.1,
        alpha=0.5,
        beta=0.0,
        nce_mode="projection",
        eps=1e-8,
    )

    result = neat_step_reference(param, grad, state, config)

    np.testing.assert_allclose(
        result.state.nce, np.array([-0.5, 0.0], dtype=np.float32)
    )
    np.testing.assert_allclose(
        result.state.momentum, np.array([0.5, 0.0], dtype=np.float32)
    )
    np.testing.assert_allclose(result.param, np.array([0.95, 2.0], dtype=np.float32))
