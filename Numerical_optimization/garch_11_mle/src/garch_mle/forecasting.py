"""Closed-form conditional-variance forecasts for GARCH(1,1)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .model import GARCHParameters, filter_variance

FloatArray = NDArray[np.float64]


def variance_forecast(
    returns: ArrayLike,
    params: GARCHParameters,
    horizon: int,
) -> FloatArray:
    """Forecast h_{T+k|T} for k=1,...,horizon."""

    if horizon < 1:
        raise ValueError("horizon must be positive.")
    values = np.asarray(returns, dtype=float)
    filtered = filter_variance(values, params)
    last_residual = values[-1] - params.mu
    one_step = params.omega + params.alpha * last_residual**2 + params.beta * filtered[-1]
    forecasts = np.empty(horizon, dtype=float)
    forecasts[0] = one_step
    long_run = params.unconditional_variance
    for step in range(1, horizon):
        forecasts[step] = long_run + params.persistence**step * (one_step - long_run)
    return forecasts
