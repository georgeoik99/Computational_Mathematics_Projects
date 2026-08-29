import numpy as np

from garch_mle.estimation import fit_custom_bfgs, fit_scipy_bfgs
from garch_mle.model import GARCHParameters, simulate_garch11
from garch_mle.optimizers import bfgs_minimize


def test_custom_bfgs_solves_positive_definite_quadratic():
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    linear = np.array([-1.0, 2.0])

    def objective(x):
        return 0.5 * x @ matrix @ x + linear @ x, matrix @ x + linear

    result = bfgs_minimize(objective, np.array([3.0, -4.0]))
    expected = -np.linalg.solve(matrix, linear)
    assert result.success
    np.testing.assert_allclose(result.x, expected, atol=1e-7)
    objectives = np.array([item.objective for item in result.history])
    assert np.all(np.diff(objectives) <= 1e-12)


def test_custom_fit_agrees_with_scipy_reference():
    truth = GARCHParameters(0.02, 0.05, 0.08, 0.88)
    returns, _ = simulate_garch11(2200, truth, burn=800, seed=2026)
    custom = fit_custom_bfgs(returns)
    reference = fit_scipy_bfgs(returns)
    assert custom.success
    assert abs(custom.nll - reference.nll) < 1e-5
    assert custom.gradient_inf_norm < 2e-5
    np.testing.assert_allclose(custom.params.as_array(), reference.params.as_array(), atol=2e-3)
