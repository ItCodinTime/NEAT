import numpy as np
import pytest

from neat_optim import ArrayState, NEATConfig
from neat_optim.engine.reference import neat_step_reference


def _config(**kwargs):
    return NEATConfig(
        spec_version="nce_spec_v3",
        native="never",
        learning_rate=0.01,
        beta=0.0,
        **kwargs,
    )


@pytest.mark.parametrize(
    "option",
    [
        {"conflict_granularity": "per_parameter"},
        {"trust_region_ratio": 0.1},
        {"opponent_ensemble": True},
        {"correction_preconditioning_order": "correction_then_precondition"},
        {"deterministic": True},
    ],
)
def test_v3_options_preserve_zero_conflict_baseline(option) -> None:
    parameter = np.array([1.0, -1.0], dtype=np.float32)
    gradient = np.array([0.5, -0.25], dtype=np.float32)
    state = ArrayState.zeros_like(parameter)
    state.momentum[:] = gradient
    state.gradient_ema[:] = gradient
    result = neat_step_reference(parameter, gradient, state, _config(**option))
    np.testing.assert_array_equal(result.state.nce, np.zeros_like(gradient))
    np.testing.assert_allclose(result.param, parameter - 0.01 * gradient)


def test_trust_region_bounds_correction_relative_to_gradient() -> None:
    parameter = np.ones(4, dtype=np.float32)
    gradient = np.ones(4, dtype=np.float32)
    state = ArrayState.zeros_like(parameter)
    state.momentum[:] = -gradient
    result = neat_step_reference(
        parameter,
        gradient,
        state,
        _config(alpha=10.0, trust_region_ratio=0.05),
    )
    assert result.metrics.correction_ratio <= 0.05 + 1e-6
    corrected = gradient + result.state.nce
    assert float(np.dot(corrected, gradient)) >= 0.0


def test_deterministic_mode_is_bit_exact_across_repeated_runs() -> None:
    parameter = np.linspace(-1.0, 1.0, 257, dtype=np.float32)
    gradient = np.cos(parameter).astype(np.float32)
    first_state = ArrayState.zeros_like(parameter)
    second_state = ArrayState.zeros_like(parameter)
    first_state.momentum[:] = -gradient
    second_state.momentum[:] = -gradient
    config = _config(deterministic=True)
    first = neat_step_reference(parameter, gradient, first_state, config)
    second = neat_step_reference(parameter, gradient, second_state, config)
    np.testing.assert_array_equal(first.param, second.param)
    np.testing.assert_array_equal(first.state.nce, second.state.nce)
