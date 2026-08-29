"""GARCH(1,1) model, simulation, constraint transform and analytic score."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
STATIONARITY_MARGIN = 1e-8


@dataclass(frozen=True)
class GARCHParameters:
    """Natural GARCH(1,1) parameters."""

    mu: float
    omega: float
    alpha: float
    beta: float

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def unconditional_variance(self) -> float:
        return self.omega / (1.0 - self.persistence)

    def as_array(self) -> FloatArray:
        return np.array([self.mu, self.omega, self.alpha, self.beta], dtype=float)

    def validate(self) -> None:
        values = self.as_array()
        if not np.all(np.isfinite(values)):
            raise ValueError("All parameters must be finite.")
        if self.omega <= 0.0:
            raise ValueError("omega must be strictly positive.")
        if self.alpha < 0.0 or self.beta < 0.0:
            raise ValueError("alpha and beta must be non-negative.")
        if self.persistence >= 1.0:
            raise ValueError("Weak covariance stationarity requires alpha + beta < 1.")


def _as_returns(returns: ArrayLike) -> FloatArray:
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("returns must be a one-dimensional array with at least 2 values.")
    if not np.all(np.isfinite(values)):
        raise ValueError("returns must contain only finite values.")
    return values


def unconstrained_to_natural(
    unconstrained: ArrayLike,
    margin: float = STATIONARITY_MARGIN,
) -> GARCHParameters:
    """Map R^4 smoothly into the strictly stationary admissible set.

    omega uses a log transform. alpha and beta use a three-part softmax;
    the third part is the positive slack 1 - alpha - beta.
    """

    u = np.asarray(unconstrained, dtype=float)
    if u.shape != (4,) or not np.all(np.isfinite(u)):
        raise ValueError("unconstrained must be a finite vector of length four.")
    if not 0.0 < margin < 1.0:
        raise ValueError("margin must lie strictly between zero and one.")

    scale = 1.0 - margin
    logits = np.array([u[2], u[3], 0.0])
    logits -= np.max(logits)
    weights = np.exp(logits)
    weights /= np.sum(weights)
    params = GARCHParameters(
        mu=float(u[0]),
        omega=float(np.exp(np.clip(u[1], -700.0, 700.0))),
        alpha=float(scale * weights[0]),
        beta=float(scale * weights[1]),
    )
    params.validate()
    return params


def natural_to_unconstrained(
    params: GARCHParameters,
    margin: float = STATIONARITY_MARGIN,
) -> FloatArray:
    """Inverse of :func:`unconstrained_to_natural` in the interior."""

    params.validate()
    scale = 1.0 - margin
    slack = scale - params.alpha - params.beta
    if params.alpha <= 0.0 or params.beta <= 0.0 or slack <= 0.0:
        raise ValueError("The inverse transform requires an interior parameter vector.")
    return np.array(
        [
            params.mu,
            np.log(params.omega),
            np.log(params.alpha / slack),
            np.log(params.beta / slack),
        ],
        dtype=float,
    )


def transform_jacobian(
    unconstrained: ArrayLike,
    margin: float = STATIONARITY_MARGIN,
) -> FloatArray:
    """Jacobian d(mu, omega, alpha, beta) / d(unconstrained)."""

    params = unconstrained_to_natural(unconstrained, margin=margin)
    scale = 1.0 - margin
    jacobian = np.zeros((4, 4), dtype=float)
    jacobian[0, 0] = 1.0
    jacobian[1, 1] = params.omega
    jacobian[2, 2] = params.alpha * (1.0 - params.alpha / scale)
    jacobian[2, 3] = -params.alpha * params.beta / scale
    jacobian[3, 2] = -params.alpha * params.beta / scale
    jacobian[3, 3] = params.beta * (1.0 - params.beta / scale)
    return jacobian


def simulate_garch11(
    n: int,
    params: GARCHParameters,
    *,
    burn: int = 500,
    seed: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Simulate Gaussian GARCH(1,1) returns and conditional variances."""

    params.validate()
    if n < 2 or burn < 0:
        raise ValueError("n must be at least 2 and burn must be non-negative.")
    rng = np.random.default_rng(seed)
    total = n + burn
    innovations = rng.standard_normal(total)
    variances = np.empty(total, dtype=float)
    residuals = np.empty(total, dtype=float)
    returns = np.empty(total, dtype=float)

    variances[0] = params.unconditional_variance
    residuals[0] = np.sqrt(variances[0]) * innovations[0]
    returns[0] = params.mu + residuals[0]
    for t in range(1, total):
        variances[t] = (
            params.omega
            + params.alpha * residuals[t - 1] ** 2
            + params.beta * variances[t - 1]
        )
        residuals[t] = np.sqrt(variances[t]) * innovations[t]
        returns[t] = params.mu + residuals[t]
    return returns[burn:], variances[burn:]


def filter_variance(returns: ArrayLike, params: GARCHParameters) -> FloatArray:
    """Apply the variance recursion using unconditional-variance initialization."""

    values = _as_returns(returns)
    params.validate()
    residuals = values - params.mu
    variances = np.empty_like(values)
    variances[0] = params.unconditional_variance
    for t in range(1, values.size):
        variances[t] = (
            params.omega
            + params.alpha * residuals[t - 1] ** 2
            + params.beta * variances[t - 1]
        )
    return variances


def gaussian_nll_and_gradient(
    unconstrained: ArrayLike,
    returns: ArrayLike,
) -> tuple[float, FloatArray]:
    """Gaussian conditional negative log-likelihood and analytic gradient.

    Derivatives of the variance recursion are first formed with respect to the
    natural parameters, then mapped to the unconstrained scale by the chain rule.
    """

    values = _as_returns(returns)
    u = np.asarray(unconstrained, dtype=float)
    params = unconstrained_to_natural(u)
    residuals = values - params.mu
    theta = params.as_array()
    _, omega, alpha, beta = theta
    persistence = alpha + beta

    variances = np.empty_like(values)
    variance_derivatives = np.empty((values.size, 4), dtype=float)
    variances[0] = omega / (1.0 - persistence)
    variance_derivatives[0] = np.array(
        [
            0.0,
            1.0 / (1.0 - persistence),
            omega / (1.0 - persistence) ** 2,
            omega / (1.0 - persistence) ** 2,
        ]
    )

    for t in range(1, values.size):
        previous_residual = residuals[t - 1]
        variances[t] = omega + alpha * previous_residual**2 + beta * variances[t - 1]
        variance_derivatives[t, 0] = (
            -2.0 * alpha * previous_residual
            + beta * variance_derivatives[t - 1, 0]
        )
        variance_derivatives[t, 1] = 1.0 + beta * variance_derivatives[t - 1, 1]
        variance_derivatives[t, 2] = (
            previous_residual**2 + beta * variance_derivatives[t - 1, 2]
        )
        variance_derivatives[t, 3] = (
            variances[t - 1] + beta * variance_derivatives[t - 1, 3]
        )

    if np.any(~np.isfinite(variances)) or np.any(variances <= 0.0):
        return float("inf"), np.full(4, np.nan)

    squared_residuals = residuals**2
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        nll = 0.5 * np.sum(
            np.log(2.0 * np.pi) + np.log(variances) + squared_residuals / variances
        )
        variance_coefficients = 1.0 / variances - squared_residuals / variances**2
        natural_gradient = 0.5 * np.sum(
            variance_coefficients[:, None] * variance_derivatives,
            axis=0,
        )
        natural_gradient[0] += np.sum(-residuals / variances)
        unconstrained_gradient = transform_jacobian(u).T @ natural_gradient
    if not np.isfinite(nll) or not np.all(np.isfinite(unconstrained_gradient)):
        return float("inf"), np.full(4, np.nan)
    return float(nll), unconstrained_gradient
