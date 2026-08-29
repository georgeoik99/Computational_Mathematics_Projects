"""Observed-information inference on the transformed and natural scales."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2
from scipy.stats import norm

from .model import (
    GARCHParameters,
    gaussian_nll_and_gradient,
    natural_to_unconstrained,
    transform_jacobian,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class InferenceResult:
    hessian_unconstrained: FloatArray
    covariance_unconstrained: FloatArray
    covariance_natural: FloatArray
    standard_errors: FloatArray
    confidence_intervals: FloatArray
    hessian_eigenvalues: FloatArray


@dataclass(frozen=True)
class ProfileLikelihoodResult:
    persistence_grid: FloatArray
    profile_nll: FloatArray
    cutoff: float
    confidence_interval: tuple[float, float]


def numerical_hessian_from_gradient(
    gradient,
    x: ArrayLike,
    *,
    relative_step: float = 1e-4,
) -> FloatArray:
    """Central-difference Hessian of an analytic gradient."""

    point = np.asarray(x, dtype=float)
    hessian = np.empty((point.size, point.size), dtype=float)
    for column in range(point.size):
        step = relative_step * max(1.0, abs(float(point[column])))
        forward = point.copy()
        backward = point.copy()
        forward[column] += step
        backward[column] -= step
        hessian[:, column] = (gradient(forward) - gradient(backward)) / (2.0 * step)
    return 0.5 * (hessian + hessian.T)


def observed_information(
    unconstrained_mle: ArrayLike,
    returns: ArrayLike,
    *,
    level: float = 0.95,
) -> InferenceResult:
    """Compute Wald uncertainty from the observed negative-loglikelihood Hessian."""

    x = np.asarray(unconstrained_mle, dtype=float)
    values = np.asarray(returns, dtype=float)

    def gradient(point: FloatArray) -> FloatArray:
        return gaussian_nll_and_gradient(point, values)[1]

    hessian = numerical_hessian_from_gradient(gradient, x)
    eigenvalues = np.linalg.eigvalsh(hessian)
    if np.min(eigenvalues) <= 0.0:
        raise np.linalg.LinAlgError(
            "Observed information is not positive definite; Wald inference is unreliable."
        )
    covariance_u = np.linalg.inv(hessian)
    jacobian = transform_jacobian(x)
    covariance_natural = jacobian @ covariance_u @ jacobian.T
    standard_errors = np.sqrt(np.maximum(np.diag(covariance_natural), 0.0))
    # The Jacobian is already available; reconstruct the natural point robustly.
    from .model import unconstrained_to_natural

    estimate = unconstrained_to_natural(x).as_array()
    critical_value = norm.ppf(0.5 + level / 2.0)
    confidence_intervals = np.column_stack(
        [estimate - critical_value * standard_errors, estimate + critical_value * standard_errors]
    )
    return InferenceResult(
        hessian,
        covariance_u,
        covariance_natural,
        standard_errors,
        confidence_intervals,
        eigenvalues,
    )


def profile_persistence(
    returns: ArrayLike,
    joint_mle: GARCHParameters,
    joint_nll: float,
    *,
    grid: ArrayLike | None = None,
    level: float = 0.95,
) -> ProfileLikelihoodResult:
    """Profile rho=alpha+beta while re-estimating all nuisance parameters."""

    values = np.asarray(returns, dtype=float)
    if grid is None:
        lower = max(0.50, joint_mle.persistence - 0.12)
        upper = min(0.995, joint_mle.persistence + 0.03)
        persistence_grid = np.linspace(lower, upper, 31)
    else:
        persistence_grid = np.asarray(grid, dtype=float)
    if (
        persistence_grid.ndim != 1
        or persistence_grid.size < 5
        or np.any(np.diff(persistence_grid) <= 0.0)
        or np.any((persistence_grid <= 0.0) | (persistence_grid >= 0.999))
    ):
        raise ValueError("grid must be a strictly increasing interior persistence grid.")

    initial_share = np.clip(joint_mle.alpha / joint_mle.persistence, 1e-6, 1.0 - 1e-6)
    initial = np.array(
        [joint_mle.mu, np.log(joint_mle.omega), np.log(initial_share / (1.0 - initial_share))]
    )
    profile = np.empty_like(persistence_grid)
    for index, persistence in enumerate(persistence_grid):
        def objective(nuisance: FloatArray) -> float:
            share = expit(nuisance[2])
            parameters = GARCHParameters(
                mu=float(nuisance[0]),
                omega=float(np.exp(np.clip(nuisance[1], -700.0, 700.0))),
                alpha=float(persistence * share),
                beta=float(persistence * (1.0 - share)),
            )
            return gaussian_nll_and_gradient(natural_to_unconstrained(parameters), values)[0]

        fit = minimize(objective, initial, method="BFGS", options={"gtol": 1e-6, "maxiter": 300})
        if not np.isfinite(fit.fun):
            raise RuntimeError(f"Profile optimization failed at persistence={persistence:.6f}.")
        profile[index] = float(fit.fun)
        initial = np.asarray(fit.x, dtype=float)

    cutoff = float(joint_nll + 0.5 * chi2.ppf(level, df=1))
    inside = profile <= cutoff
    if not np.any(inside):
        raise RuntimeError("The profile grid does not intersect the LR confidence region.")
    first = int(np.flatnonzero(inside)[0])
    last = int(np.flatnonzero(inside)[-1])

    def interpolate(left_index: int, right_index: int) -> float:
        x0, x1 = persistence_grid[[left_index, right_index]]
        y0, y1 = profile[[left_index, right_index]]
        if y1 == y0:
            return float((x0 + x1) / 2.0)
        return float(x0 + (cutoff - y0) * (x1 - x0) / (y1 - y0))

    lower_ci = float(persistence_grid[0]) if first == 0 else interpolate(first - 1, first)
    upper_ci = (
        float(persistence_grid[-1])
        if last == persistence_grid.size - 1
        else interpolate(last, last + 1)
    )
    return ProfileLikelihoodResult(
        persistence_grid,
        profile,
        cutoff,
        (lower_ci, upper_ci),
    )
