import pytest

from neat_optim.config import NEATConfig, PlayerNEATConfig
from neat_optim.exceptions import ConfigurationError


def test_valid_config_round_trip() -> None:
    config = NEATConfig(
        learning_rate=1e-2,
        alpha=0.2,
        beta=0.8,
        nce_mode="cosine",
        sparsity_l1=1e-4,
        prune_threshold=1e-3,
        opponent_source="gradient_ema",
        opponent_ema_decay=0.8,
        opponent_blend=0.3,
        correction_warmup_steps=2,
        conflict_threshold=0.1,
        adaptive_correction=True,
        adaptive_correction_decay=0.8,
        adaptive_correction_min_scale=1.0,
        adaptive_correction_max_scale=2.5,
    )
    payload = config.as_dict()
    assert payload["learning_rate"] == pytest.approx(1e-2)
    assert payload["alpha"] == pytest.approx(0.2)
    assert payload["beta"] == pytest.approx(0.8)
    assert payload["nce_mode"] == "cosine"
    assert payload["sparsity_l1"] == pytest.approx(1e-4)
    assert payload["prune_threshold"] == pytest.approx(1e-3)
    assert payload["opponent_source"] == "gradient_ema"
    assert payload["opponent_ema_decay"] == pytest.approx(0.8)
    assert payload["opponent_blend"] == pytest.approx(0.3)
    assert payload["correction_warmup_steps"] == 2
    assert payload["conflict_threshold"] == pytest.approx(0.1)
    assert payload["adaptive_correction"] is True
    assert payload["adaptive_correction_decay"] == pytest.approx(0.8)
    assert payload["adaptive_correction_min_scale"] == pytest.approx(1.0)
    assert payload["adaptive_correction_max_scale"] == pytest.approx(2.5)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("learning_rate", 0.0),
        ("alpha", -0.1),
        ("beta", 1.0),
        ("eps", 0.0),
        ("weight_decay", -1.0),
        ("nce_clip_ratio", 0.0),
        ("sparsity_l1", -1.0),
        ("prune_threshold", -1.0),
        ("opponent_ema_decay", 1.0),
        ("opponent_blend", 1.5),
        ("correction_warmup_steps", -1),
        ("conflict_threshold", 1.5),
        ("adaptive_correction_decay", 1.0),
        ("adaptive_correction_min_scale", 0.0),
    ],
)
def test_invalid_config_values(field: str, value: float) -> None:
    kwargs = {field: value}
    with pytest.raises(ConfigurationError):
        NEATConfig(**kwargs)


def test_invalid_opponent_source() -> None:
    with pytest.raises(ConfigurationError):
        NEATConfig(opponent_source="invalid")


def test_invalid_adaptive_scale_range() -> None:
    with pytest.raises(ConfigurationError):
        NEATConfig(
            adaptive_correction_min_scale=2.0,
            adaptive_correction_max_scale=1.5,
        )


def test_valid_player_config_round_trip() -> None:
    config = PlayerNEATConfig(
        learning_rate=1e-2,
        alpha=0.2,
        beta=0.8,
        opponent_mode="batch_mean",
        player_reduction="sum",
        sparsity_l1=1e-4,
        prune_threshold=1e-3,
    )

    payload = config.as_dict()
    assert payload["opponent_mode"] == "batch_mean"
    assert payload["player_reduction"] == "sum"
    assert payload["native"] == "never"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("opponent_mode", "invalid"),
        ("player_reduction", "invalid"),
        ("native", "auto"),
    ],
)
def test_invalid_player_config_values(field: str, value: str) -> None:
    kwargs = {field: value}
    with pytest.raises(ConfigurationError):
        PlayerNEATConfig(**kwargs)
