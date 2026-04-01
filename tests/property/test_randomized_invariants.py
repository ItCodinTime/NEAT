import numpy as np

from neat_optim.config import NEATConfig
from neat_optim.engine.reference import neat_step_reference
from neat_optim.state import ArrayState


def test_randomized_reference_step_stays_finite() -> None:
    rng = np.random.default_rng(1234)
    config = NEATConfig(learning_rate=1e-2, alpha=0.25, beta=0.9)

    for _ in range(100):
        param = rng.normal(size=(16,)).astype(np.float32)
        grad = rng.normal(size=(16,)).astype(np.float32)
        state = ArrayState(
            momentum=rng.normal(size=(16,)).astype(np.float32),
            nce=np.zeros((16,), dtype=np.float32),
            step=0,
        )
        result = neat_step_reference(param, grad, state, config)
        assert np.isfinite(result.param).all()
        assert np.isfinite(result.state.momentum).all()
        assert np.isfinite(result.state.nce).all()
