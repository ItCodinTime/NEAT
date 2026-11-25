import numpy as np
import pytest

from neat_optim.config import NEATConfig
from neat_optim.engine.functional import neat_step
from neat_optim.exceptions import ConfigurationError
from neat_optim.state import ArrayState


def test_functional_step_falls_back_when_native_is_unavailable(
    sample_param: np.ndarray, sample_grad: np.ndarray
) -> None:
    param = sample_param.copy()
    state = ArrayState.zeros_like(param)
    config = NEATConfig(native="never")

    result = neat_step(param, sample_grad, state, config)

    assert result.state.step == 1
    assert result.metrics.grad_norm > 0.0


def test_native_force_rejects_unsupported_configuration(
    sample_param: np.ndarray, sample_grad: np.ndarray
) -> None:
    with pytest.raises(ConfigurationError, match="native='force'"):
        neat_step(
            sample_param.copy(),
            sample_grad,
            ArrayState.zeros_like(sample_param),
            NEATConfig(native="force", adaptive_alpha=True),
        )
