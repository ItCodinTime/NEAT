from argparse import Namespace

import pytest
from benchmarks.manifest import reproducibility_metadata
from benchmarks.torch_suite.common import orchestrate_seeds


def test_direct_internal_seed_requires_explicit_permission(monkeypatch) -> None:
    monkeypatch.delenv("NEAT_BENCHMARK_ORCHESTRATED", raising=False)
    args = Namespace(seed=7, allow_single_seed=False, seeds=[7])
    with pytest.raises(SystemExit, match="allow-single-seed"):
        orchestrate_seeds("unused", args, [])


def test_orchestrated_single_seed_inherits_preliminary_stamp(monkeypatch) -> None:
    monkeypatch.setenv("NEAT_BENCHMARK_ORCHESTRATED", "1")
    monkeypatch.setenv("NEAT_BENCHMARK_PRELIMINARY", "1")
    args = Namespace(seed=7, allow_single_seed=False, seeds=[7])
    assert orchestrate_seeds("unused", args, []) is False
    assert args.preliminary is True


def test_reproducibility_metadata_redacts_hostname() -> None:
    metadata = reproducibility_metadata(seed=7, invocation=["benchmark"])
    assert metadata["hardware"]["hostname"] == "redacted"
