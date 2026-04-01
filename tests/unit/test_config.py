import pytest

from neat_optim.config import NEATConfig
from neat_optim.exceptions import ConfigurationError


def test_valid_config_round_trip() -> None:
    config = NEATConfig(learning_rate=1e-2, alpha=0.2, beta=0.8, nce_mode="cosine")
    payload = config.as_dict()
    assert payload["learning_rate"] == pytest.approx(1e-2)
    assert payload["alpha"] == pytest.approx(0.2)
    assert payload["beta"] == pytest.approx(0.8)
    assert payload["nce_mode"] == "cosine"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("learning_rate", 0.0),
        ("alpha", -0.1),
        ("beta", 1.0),
        ("eps", 0.0),
        ("weight_decay", -1.0),
        ("nce_clip_ratio", 0.0),
    ],
)
def test_invalid_config_values(field: str, value: float) -> None:
    kwargs = {field: value}
    with pytest.raises(ConfigurationError):
        NEATConfig(**kwargs)
