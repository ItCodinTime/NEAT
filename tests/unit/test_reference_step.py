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


def test_previous_gradient_opponent_signal_changes_second_step() -> None:
    param = np.array([1.0, 2.0], dtype=np.float32)
    first_grad = np.array([-1.0, 0.0], dtype=np.float32)
    second_grad = np.array([1.0, 0.0], dtype=np.float32)
    state = ArrayState.zeros_like(param)
    config = NEATConfig(
        learning_rate=0.1,
        alpha=0.5,
        beta=0.0,
        opponent_source="previous_gradient",
    )

    first = neat_step_reference(param, first_grad, state, config)
    result = neat_step_reference(first.param, second_grad, first.state, config)

    np.testing.assert_allclose(
        result.state.nce,
        np.array([-0.5, 0.0], dtype=np.float32),
    )
    assert result.metrics.conflict_ratio == pytest.approx(1.0)
    assert result.metrics.correction_ratio == pytest.approx(0.5, abs=1e-6)


def test_adaptive_preconditioning_updates_second_moment_state() -> None:
    param = np.array([1.0, -2.0], dtype=np.float32)
    grad = np.array([0.5, -0.25], dtype=np.float32)
    state = ArrayState.zeros_like(param)
    config = NEATConfig(
        learning_rate=0.01,
        alpha=0.0,
        beta=0.9,
        adaptive_preconditioning=True,
        second_moment_beta=0.99,
        bias_correction=True,
    )

    result = neat_step_reference(param, grad, state, config)

    assert result.state.second_moment is not None
    np.testing.assert_allclose(
        result.state.second_moment,
        0.01 * np.square(grad),
        atol=1e-6,
    )


def test_warmup_and_conflict_threshold_gate_correction() -> None:
    param = np.array([1.0, 2.0], dtype=np.float32)
    grad = np.array([1.0, 0.0], dtype=np.float32)
    state = ArrayState(
        momentum=np.array([-1.0, 0.0], dtype=np.float32),
        nce=np.zeros(2, dtype=np.float32),
        step=0,
    )
    warmup_config = NEATConfig(
        learning_rate=0.1,
        alpha=0.5,
        beta=0.0,
        correction_warmup_steps=1,
    )

    warmup_result = neat_step_reference(param, grad, state, warmup_config)
    np.testing.assert_allclose(warmup_result.state.nce, np.zeros(2, dtype=np.float32))

    state.step = 1
    threshold_result = neat_step_reference(
        param,
        np.array([0.1, 0.0], dtype=np.float32),
        ArrayState(
            momentum=np.array([-1.0, 0.0], dtype=np.float32),
            nce=np.zeros(2, dtype=np.float32),
            step=1,
        ),
        NEATConfig(
            learning_rate=0.1,
            alpha=0.5,
            beta=0.0,
            conflict_threshold=1.0,
        ),
    )
    np.testing.assert_allclose(
        threshold_result.state.nce,
        np.zeros(2, dtype=np.float32),
    )


def test_lion_update_mode_uses_sign_update() -> None:
    param = np.array([1.0, -2.0], dtype=np.float32)
    grad = np.array([0.5, -0.25], dtype=np.float32)
    state = ArrayState.zeros_like(param)
    config = NEATConfig(
        learning_rate=0.1,
        alpha=0.0,
        beta=0.9,
        nce_mode="off",
        update_mode="lion",
    )

    result = neat_step_reference(param, grad, state, config)

    np.testing.assert_allclose(
        result.param,
        np.array([0.9, -1.9], dtype=np.float32),
        atol=1e-6,
    )
    assert result.metrics.update_norm == pytest.approx(np.sqrt(2.0), abs=1e-6)


def test_adaptive_alpha_tracks_conflict_and_gradient_noise() -> None:
    param = np.array([1.0, 2.0], dtype=np.float32)
    grad = np.array([1.0, 0.0], dtype=np.float32)
    state = ArrayState(
        momentum=np.array([-1.0, 0.0], dtype=np.float32),
        nce=np.zeros(2, dtype=np.float32),
        previous_gradient=np.array([-1.0, 0.0], dtype=np.float32),
        gradient_ema=np.zeros(2, dtype=np.float32),
        step=1,
    )
    config = NEATConfig(
        learning_rate=0.1,
        alpha=0.25,
        beta=0.0,
        adaptive_alpha=True,
        adaptive_alpha_min=0.1,
        adaptive_alpha_max=0.6,
        gradient_noise_decay=0.5,
        adaptive_correction_decay=0.5,
    )

    result = neat_step_reference(param, grad, state, config)

    assert result.metrics.effective_alpha > config.alpha
    assert result.metrics.effective_alpha <= config.adaptive_alpha_max
    assert result.metrics.gradient_noise > 0.0
    assert result.state.gradient_noise_ema > 0.0


def test_gradient_centralization_subtracts_feature_mean() -> None:
    param = np.zeros((2, 2), dtype=np.float32)
    grad = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
    state = ArrayState.zeros_like(param)
    config = NEATConfig(
        learning_rate=1.0,
        alpha=0.0,
        beta=0.0,
        nce_mode="off",
        gradient_centralization=True,
    )

    result = neat_step_reference(param, grad, state, config)

    expected_grad = np.array([[-2.0, -2.0], [2.0, 2.0]], dtype=np.float32)
    np.testing.assert_allclose(result.state.momentum, expected_grad, atol=1e-6)
    np.testing.assert_allclose(result.param, -expected_grad, atol=1e-6)


def test_nesterov_uses_lookahead_momentum_update() -> None:
    param = np.array([1.0, -2.0], dtype=np.float32)
    grad = np.array([0.5, -0.25], dtype=np.float32)
    state = ArrayState.zeros_like(param)
    config = NEATConfig(
        learning_rate=0.1,
        alpha=0.0,
        beta=0.9,
        nce_mode="off",
        nesterov=True,
    )

    result = neat_step_reference(param, grad, state, config)

    momentum = 0.1 * grad
    expected_update = (0.9 * momentum) + (0.1 * grad)
    np.testing.assert_allclose(result.param, param - (0.1 * expected_update), atol=1e-6)


def test_lookahead_syncs_slow_parameter_every_k_steps() -> None:
    param = np.array([1.0, 2.0], dtype=np.float32)
    grad = np.array([1.0, 0.0], dtype=np.float32)
    state = ArrayState.zeros_like(param)
    config = NEATConfig(
        learning_rate=0.1,
        alpha=0.0,
        beta=0.0,
        nce_mode="off",
        lookahead_k=2,
        lookahead_alpha=0.5,
    )

    first = neat_step_reference(param, grad, state, config)
    second = neat_step_reference(first.param, grad, first.state, config)

    np.testing.assert_allclose(first.param, np.array([0.9, 2.0], dtype=np.float32))
    np.testing.assert_allclose(second.param, np.array([0.9, 2.0], dtype=np.float32))
    np.testing.assert_allclose(
        second.state.slow_param,
        np.array([0.9, 2.0], dtype=np.float32),
    )
