from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neat_optim import ArrayState, NEATConfig, TorchNEAT
from neat_optim.engine.reference import neat_step_reference


def _run_reference_and_torch(
    initial: np.ndarray,
    gradients: list[np.ndarray],
    config: NEATConfig,
) -> tuple[ArrayState, dict[str, object]]:
    """Advance both adapters through the same deterministic gradient stream."""
    reference_param = initial.copy()
    reference_state = ArrayState.zeros_like(initial)
    parameter = torch.nn.Parameter(torch.from_numpy(initial.copy()))
    kwargs = asdict(config)
    kwargs.pop("learning_rate")
    kwargs.pop("native")
    optimizer = TorchNEAT(
        [parameter], learning_rate=config.learning_rate, **kwargs
    )

    for gradient in gradients:
        result = neat_step_reference(
            reference_param, gradient, reference_state, config
        )
        reference_param, reference_state = result.param, result.state
        parameter.grad = torch.from_numpy(gradient.copy())
        optimizer.step()

    np.testing.assert_allclose(
        parameter.detach().numpy(), reference_param, rtol=2e-5, atol=2e-6
    )
    return reference_state, optimizer.state[parameter]


def test_torch_neat_matches_reference_basic_step() -> None:
    initial = np.array([1.0, -2.0], dtype=np.float32)
    gradient = np.array([0.5, -0.25], dtype=np.float32)
    config = NEATConfig(learning_rate=0.1, alpha=0.25, beta=0.9, native="never")
    expected = neat_step_reference(
        initial, gradient, ArrayState.zeros_like(initial), config
    )

    parameter = torch.nn.Parameter(torch.from_numpy(initial.copy()))
    parameter.grad = torch.from_numpy(gradient.copy())
    optimizer = TorchNEAT([parameter], learning_rate=0.1, alpha=0.25, beta=0.9)
    optimizer.step()

    np.testing.assert_allclose(
        parameter.detach().numpy(), expected.param, rtol=1e-6, atol=1e-7
    )
    assert optimizer.diagnostic_snapshot()["mean_effective_alpha"] == pytest.approx(
        0.25
    )


def test_torch_neat_accepts_pytorch_lr_alias() -> None:
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = TorchNEAT([parameter], lr=0.01)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)


def test_torch_neat_samples_diagnostics_without_changing_updates() -> None:
    """Diagnostic sampling must be observational, never algorithmic."""
    full_parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    sampled_parameter = torch.nn.Parameter(full_parameter.detach().clone())
    full = TorchNEAT([full_parameter], lr=0.01, diagnostic_interval=1)
    sampled = TorchNEAT([sampled_parameter], lr=0.01, diagnostic_interval=3)

    for _ in range(4):
        gradient = torch.tensor([0.2, -0.3])
        full_parameter.grad = gradient.clone()
        sampled_parameter.grad = gradient.clone()
        full.step()
        sampled.step()

    torch.testing.assert_close(full_parameter, sampled_parameter)
    assert full._diagnostic_count == 4
    assert sampled._diagnostic_count == 2


def test_torch_neat_rejects_invalid_diagnostic_interval() -> None:
    with pytest.raises(ValueError, match="diagnostic_interval"):
        TorchNEAT([torch.nn.Parameter(torch.ones(1))], diagnostic_interval=0)


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "opponent_source": "previous_gradient",
            "adaptive_correction": True,
            "adaptive_preconditioning": True,
            "bias_correction": True,
            "adaptive_alpha": True,
        },
        {
            "opponent_source": "gradient_ema",
            "nce_mode": "cosine",
            "nesterov": True,
            "conflict_threshold": 0.05,
        },
        {
            "update_mode": "lion",
            "lookahead_k": 2,
            "lookahead_alpha": 0.4,
            "weight_decay": 0.02,
        },
        {
            "decouple_weight_decay": False,
            "weight_decay": 0.02,
            "sparsity_l1": 0.01,
            "prune_threshold": 0.02,
        },
    ],
)
def test_torch_neat_matches_reference_advanced_modes(
    overrides: dict[str, object],
) -> None:
    """Protect framework parity for every major optimizer feature family."""
    initial = np.array([[0.4, -0.2], [0.1, -0.5]], dtype=np.float32)
    gradients = [
        np.array([[0.3, -0.4], [0.2, 0.1]], dtype=np.float32),
        np.array([[-0.5, 0.2], [-0.1, -0.3]], dtype=np.float32),
        np.array([[0.1, 0.3], [0.4, -0.2]], dtype=np.float32),
        np.array([[-0.2, -0.1], [0.3, 0.5]], dtype=np.float32),
    ]
    config = NEATConfig(
        learning_rate=0.03,
        alpha=0.25,
        beta=0.9,
        native="never",
        **overrides,
    )
    reference_state, torch_state = _run_reference_and_torch(
        initial, gradients, config
    )

    for name in (
        "momentum",
        "nce",
        "previous_gradient",
        "gradient_ema",
        "second_moment",
        "slow_param",
    ):
        expected = getattr(reference_state, name)
        actual = torch_state[name].detach().numpy()
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    for name in ("conflict_ema", "gradient_noise_ema", "alignment_ema"):
        assert float(torch_state[name]) == pytest.approx(
            getattr(reference_state, name), rel=2e-5, abs=2e-6
        )


def test_torch_gradient_centralization_uses_pytorch_channel_layout() -> None:
    """Centralize over input/spatial axes while preserving output channels."""
    parameter = torch.nn.Parameter(torch.zeros((2, 3, 2, 2)))
    parameter.grad = torch.arange(24, dtype=torch.float32).reshape_as(parameter)
    optimizer = TorchNEAT(
        [parameter], learning_rate=0.1, beta=0.0, gradient_centralization=True
    )
    optimizer.step()

    # With beta=0 and no conflict on the first step, the update is exactly the
    # centralized gradient. Each output channel should therefore sum to zero.
    updates = -parameter.detach() / 0.1
    torch.testing.assert_close(updates.mean(dim=(1, 2, 3)), torch.zeros(2))


def test_benchmark_parsers_do_not_load_heavy_dependencies() -> None:
    from benchmarks.torch_suite.diffusion import parse_args as diffusion_args
    from benchmarks.torch_suite.language_model import parse_args as language_args
    from benchmarks.torch_suite.reinforcement_learning import parse_args as rl_args
    from benchmarks.torch_suite.vision import parse_args as vision_args

    assert vision_args([]).dataset == "cifar10"
    assert language_args([]).task == "sst2"
    assert rl_args([]).env == "HalfCheetah-v5"
    assert diffusion_args([]).batch_size == 128
