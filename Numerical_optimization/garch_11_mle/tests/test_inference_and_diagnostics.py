import numpy as np

from garch_mle.diagnostics import information_criteria, ljung_box, standardized_residuals
from garch_mle.estimation import fit_custom_bfgs
from garch_mle.forecasting import variance_forecast
from garch_mle.inference import observed_information, profile_persistence
from garch_mle.model import GARCHParameters, simulate_garch11


def test_inference_diagnostics_and_forecast_are_finite():
    truth = GARCHParameters(0.02, 0.05, 0.08, 0.88)
    returns, _ = simulate_garch11(2500, truth, burn=800, seed=9)
    fit = fit_custom_bfgs(returns)
    inference = observed_information(fit.unconstrained, returns)
    profile = profile_persistence(
        returns,
        fit.params,
        fit.nll,
        grid=np.linspace(max(0.7, fit.params.persistence - 0.08), 0.99, 9),
    )
    residuals = standardized_residuals(returns, fit.params)
    diagnostic = ljung_box(residuals**2, 15)
    forecast = variance_forecast(returns, fit.params, 30)
    assert np.all(inference.hessian_eigenvalues > 0.0)
    assert np.all(inference.standard_errors > 0.0)
    assert profile.confidence_interval[0] < fit.params.persistence < profile.confidence_interval[1]
    assert 0.0 <= diagnostic.p_value <= 1.0
    assert np.all(np.isfinite(forecast)) and np.all(forecast > 0.0)
    assert abs(forecast[-1] - fit.params.unconditional_variance) < abs(forecast[0] - fit.params.unconditional_variance)
    aic, bic = information_criteria(fit.nll, 4, returns.size)
    assert np.isfinite(aic) and np.isfinite(bic)
