import numpy as np
import pytest

from neat_optim.config import NEATConfig, PlayerNEATConfig
from neat_optim.engine.multiplayer import neat_player_step
from neat_optim.engine.reference import neat_step_reference
from neat_optim.exceptions import ShapeError
from neat_optim.state import ArrayState


def test_player_step_matches_reference_when_players_agree() -> None:
    param = np.array([1.0, -2.0], dtype=np.float32)
    grad = np.array([0.5, -0.25], dtype=np.float32)
    player_grads = np.stack([grad, grad], axis=0)

    player_state = ArrayState.zeros_like(param)
    reference_state = ArrayState.zeros_like(param)
    player_result = neat_player_step(
        param,
        player_grads,
        player_state,
        PlayerNEATConfig(learning_rate=0.1, alpha=0.25, beta=0.9),
    )
    reference_result = neat_step_reference(
        param,
        grad,
        reference_state,
        NEATConfig(learning_rate=0.1, alpha=0.25, beta=0.9, native="never"),
    )

    np.testing.assert_allclose(player_result.param, reference_result.param, atol=1e-6)
    np.testing.assert_allclose(
        player_result.state.momentum,
        reference_result.state.momentum,
        atol=1e-6,
    )
    assert player_result.metrics.mean_player_conflict == pytest.approx(0.0)


def test_player_step_detects_conflict_between_opposing_players() -> None:
    param = np.array([1.0, 2.0], dtype=np.float32)
    player_grads = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    state = ArrayState.zeros_like(param)

    result = neat_player_step(
        param,
        player_grads,
        state,
        PlayerNEATConfig(learning_rate=0.1, alpha=0.5, beta=0.0),
    )

    np.testing.assert_allclose(result.param, param, atol=1e-6)
    assert result.metrics.mean_player_conflict == pytest.approx(1.0, abs=1e-6)
    assert result.metrics.max_player_conflict == pytest.approx(1.0, abs=1e-6)


def test_player_step_applies_lightweight_sparsity_controls() -> None:
    param = np.array([0.02, -0.02, 0.5], dtype=np.float32)
    player_grads = np.zeros((2, 3), dtype=np.float32)
    state = ArrayState.zeros_like(param)

    result = neat_player_step(
        param,
        player_grads,
        state,
        PlayerNEATConfig(
            learning_rate=1.0,
            alpha=0.0,
            beta=0.0,
            nce_mode="off",
            sparsity_l1=0.01,
            prune_threshold=0.03,
        ),
    )

    np.testing.assert_allclose(
        result.param,
        np.array([0.0, 0.0, 0.49], dtype=np.float32),
        atol=1e-6,
    )
    assert result.metrics.active_fraction == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_player_step_tracks_adaptive_conflict_state() -> None:
    param = np.array([1.0, 2.0], dtype=np.float32)
    player_grads = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    state = ArrayState.zeros_like(param)

    result = neat_player_step(
        param,
        player_grads,
        state,
        PlayerNEATConfig(
            learning_rate=0.1,
            alpha=0.5,
            beta=0.0,
            adaptive_correction=True,
            adaptive_correction_decay=0.5,
            adaptive_correction_max_scale=2.0,
        ),
    )

    assert result.state.conflict_ema > 0.0
    assert result.metrics.mean_correction_ratio >= 0.0


def test_player_step_rejects_invalid_player_gradient_shape() -> None:
    param = np.zeros((2, 2), dtype=np.float32)
    state = ArrayState.zeros_like(param)

    with pytest.raises(ShapeError):
        neat_player_step(
            param,
            np.zeros((3, 4), dtype=np.float32),
            state,
            PlayerNEATConfig(),
        )
