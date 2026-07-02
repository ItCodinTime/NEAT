import pytest

from neat_optim.config import NEATConfig
from neat_optim.exceptions import ConfigurationError
from neat_optim.spec.equations import NCE_SPEC_V2, NCE_SPEC_V3, SPEC_VERSION


def test_v2_remains_default_and_compatibility_alias() -> None:
    assert NEATConfig().spec_version == NCE_SPEC_V2
    assert SPEC_VERSION == NCE_SPEC_V2


def test_v3_is_additively_selectable() -> None:
    assert NEATConfig(spec_version=NCE_SPEC_V3).spec_version == NCE_SPEC_V3


def test_unknown_spec_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="spec_version"):
        NEATConfig(spec_version="future")
