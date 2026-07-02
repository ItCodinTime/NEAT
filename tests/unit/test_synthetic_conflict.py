import numpy as np
from benchmarks.tasks.synthetic_conflict import QuadraticConflictProblem


def test_quadratic_optimum_has_zero_mean_gradient() -> None:
    problem = QuadraticConflictProblem.create(dimensionality=8, conflict_intensity=0.75)
    gradients = problem.gradients(problem.optimum)
    np.testing.assert_allclose(gradients.mean(axis=0), 0.0, atol=1e-7)


def test_noise_is_reproducible_by_seed() -> None:
    problem = QuadraticConflictProblem.create(dimensionality=4, conflict_intensity=0.5)
    point = np.zeros(4)
    np.testing.assert_array_equal(
        problem.gradients(point, noise_level=0.2, seed=11),
        problem.gradients(point, noise_level=0.2, seed=11),
    )
