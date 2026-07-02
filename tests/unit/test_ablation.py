from benchmarks.ablate import one_at_a_time


def test_ablation_changes_exactly_one_setting() -> None:
    base = {"learning_rate": 0.1, "adaptive_alpha": False}
    cases = one_at_a_time(base, {"adaptive_alpha": [True]})
    assert cases == [
        ("base", base),
        (
            "adaptive_alpha=True",
            {"learning_rate": 0.1, "adaptive_alpha": True},
        ),
    ]
