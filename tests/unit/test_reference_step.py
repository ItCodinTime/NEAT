import numpy as np
import pytest

from neat_optim.config import NEATConfig
from neat_optim.engine.reference import neat_step_reference
from neat_optim.state import ArrayState


def test_reference_step_advances_state(
    sample_param: np.ndarray, sample_grad: np.ndarray
) -> None:
    state = ArrayState.zeros_like(sample_param)
    config = NEATConfig(learning_rate=0.1, alpha=0.25, beta=0.9)

    result = neat_step_reference(sample_param, sample_grad, state, config)

    assert result.state.step == 1
    assert result.param.shape == sample_param.shape
    assert result.state.momentum.shape == sample_param.shape
    assert result.metrics.grad_norm > 0.0
    assert np.isfinite(result.param).all()


def test_reference_step_without_prior_conflict_matches_momentum_blend() -> None:
    param = np.array([1.0, -2.0], dtype=np.float32)
    grad = np.array([0.5, -0.25], dtype=np.float32)
    state = ArrayState.zeros_like(param)
    config = NEATConfig(learning_rate=0.1, alpha=0.25, beta=0.9)

    result = neat_step_reference(param, grad, state, config)
    expected_momentum = 0.1 * grad
    expected_param = param - (0.1 * expected_momentum)

    np.testing.assert_allclose(result.state.momentum, expected_momentum, atol=1e-6)
    np.testing.assert_allclose(result.param, expected_param, atol=1e-6)
    assert result.metrics.conflict_ratio == pytest.approx(0.0)


def test_projection_mode_applies_negative_correction_for_conflict() -> None:
    param = np.array([1.0, 2.0], dtype=np.float32)
    grad = np.array([1.0, 0.0], dtype=np.float32)
    state = ArrayState(
        momentum=np.array([-1.0, 0.0], dtype=np.float32),
        nce=np.zeros(2, dtype=np.float32),
        step=3,
    )
    config = NEATConfig(learning_rate=0.1, alpha=0.5, beta=0.0)

    result = neat_step_reference(param, grad, state, config)

    np.testing.assert_allclose(
        result.state.nce, np.array([-0.5, 0.0], dtype=np.float32)
    )
    np.testing.assert_allclose(
        result.state.momentum, np.array([0.5, 0.0], dtype=np.float32)
    )
    np.testing.assert_allclose(result.param, np.array([0.95, 2.0], dtype=np.float32))
    assert result.metrics.conflict_ratio == pytest.approx(1.0)


def test_correction_is_clipped_to_gradient_norm() -> None:
    param = np.array([0.0, 0.0], dtype=np.float32)
    grad = np.array([1.0, 0.0], dtype=np.float32)
    state = ArrayState(
        momentum=np.array([-10.0, 0.0], dtype=np.float32),
        nce=np.zeros(2, dtype=np.float32),
        step=0,
    )
    config = NEATConfig(
        learning_rate=0.1,
        alpha=10.0,
        beta=0.0,
        nce_clip_ratio=0.25,
    )

    result = neat_step_reference(param, grad, state, config)

    assert np.linalg.norm(result.state.nce) == pytest.approx(0.25, abs=1e-6)


def test_reference_step_supports_matrix_parameters() -> None:
    param = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float32)
    grad = np.array([[0.5, -0.25], [0.1, -0.2]], dtype=np.float32)
    state = ArrayState.zeros_like(param)
    config = NEATConfig(learning_rate=0.05, alpha=0.25, beta=0.9)

    result = neat_step_reference(param, grad, state, config)

    assert result.state.step == 1
    assert result.param.shape == param.shape
    assert result.state.momentum.shape == param.shape
    assert result.state.nce.shape == param.shape
    assert np.isfinite(result.param).all()
