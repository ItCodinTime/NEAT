from benchmarks.torch_suite.locked_study import STUDIES, _has_completed_result


def test_locked_studies_give_each_optimizer_equal_tuning_budget() -> None:
    for study in STUDIES.values():
        candidates = study["learning_rates"]
        assert set(candidates) == {"neat", "adamw", "sgd", "lion"}
        assert {len(values) for values in candidates.values()} == {4}


def test_completed_result_requires_exactly_one_artifact(tmp_path) -> None:
    trial = tmp_path / "trial"
    assert not _has_completed_result(trial)

    (trial / "first").mkdir(parents=True)
    (trial / "first" / "result.json").write_text("{}", encoding="utf-8")
    assert _has_completed_result(trial)

    (trial / "second").mkdir()
    (trial / "second" / "result.json").write_text("{}", encoding="utf-8")
    assert not _has_completed_result(trial)
