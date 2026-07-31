from benchmarks.torch_suite.locked_study import STUDIES


def test_locked_studies_give_each_optimizer_equal_tuning_budget() -> None:
    for study in STUDIES.values():
        candidates = study["learning_rates"]
        assert set(candidates) == {"neat", "adamw", "sgd", "lion"}
        assert {len(values) for values in candidates.values()} == {4}
