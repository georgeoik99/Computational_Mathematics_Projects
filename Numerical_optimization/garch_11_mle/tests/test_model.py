import numpy as np
import pytest

from garch_mle.model import (
    GARCHParameters,
    filter_variance,
    gaussian_nll_and_gradient,
    natural_to_unconstrained,
    simulate_garch11,
    unconstrained_to_natural,
)


def central_difference(function, point, step=1e-6):
    gradient = np.empty_like(point)
    for index in range(point.size):
        forward = point.copy()
        backward = point.copy()
        forward[index] += step
        backward[index] -= step
        gradient[index] = (function(forward) - function(backward)) / (2.0 * step)
    return gradient


def test_transform_round_trip_and_feasibility():
    parameters = GARCHParameters(0.02, 0.05, 0.08, 0.88)
    recovered = unconstrained_to_natural(natural_to_unconstrained(parameters))
    np.testing.assert_allclose(recovered.as_array(), parameters.as_array(), rtol=1e-12, atol=1e-12)
    for point in [np.zeros(4), np.array([-2.0, -20.0, 20.0, -20.0]), np.array([3.0, 10.0, -15.0, 15.0])]:
        candidate = unconstrained_to_natural(point)
        candidate.validate()
        assert candidate.persistence < 1.0


def test_parameter_validation_rejects_nonstationarity():
    with pytest.raises(ValueError, match="stationarity"):
        GARCHParameters(0.0, 0.1, 0.4, 0.6).validate()


def test_filter_and_simulation_are_reproducible_and_positive():
    parameters = GARCHParameters(0.02, 0.05, 0.08, 0.88)
    first_returns, first_variance = simulate_garch11(500, parameters, seed=4)
    second_returns, second_variance = simulate_garch11(500, parameters, seed=4)
    np.testing.assert_array_equal(first_returns, second_returns)
    np.testing.assert_array_equal(first_variance, second_variance)
    assert np.all(first_variance > 0.0)
    assert np.all(filter_variance(first_returns, parameters) > 0.0)


def test_analytic_gradient_matches_central_differences():
    parameters = GARCHParameters(0.02, 0.05, 0.08, 0.88)
    returns, _ = simulate_garch11(700, parameters, seed=12)
    point = natural_to_unconstrained(GARCHParameters(0.01, 0.07, 0.10, 0.84))
    objective, analytic = gaussian_nll_and_gradient(point, returns)
    numeric = central_difference(lambda x: gaussian_nll_and_gradient(x, returns)[0], point)
    assert np.isfinite(objective)
    relative_error = np.linalg.norm(analytic - numeric, np.inf) / (1.0 + np.linalg.norm(numeric, np.inf))
    assert relative_error < 2e-6
