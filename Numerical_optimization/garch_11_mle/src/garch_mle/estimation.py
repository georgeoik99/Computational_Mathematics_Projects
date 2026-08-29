"""Starting values and custom/reference estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from .model import (
    GARCHParameters,
    gaussian_nll_and_gradient,
    natural_to_unconstrained,
    unconstrained_to_natural,
)
from .optimizers import OptimizationResult, bfgs_minimize

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GARCHFit:
    method: str
    params: GARCHParameters
    unconstrained: FloatArray
    nll: float
    success: bool
    message: str
    nit: int
    nfev: int
    gradient_inf_norm: float
    optimizer_result: object


def default_start(returns: ArrayLike) -> FloatArray:
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("returns must be a finite one-dimensional sample.")
    variance = max(float(np.var(values)), 1e-6)
    params = GARCHParameters(
        mu=float(np.mean(values)),
        omega=0.05 * variance,
        alpha=0.10,
        beta=0.85,
    )
    return natural_to_unconstrained(params)


def _objective(returns: FloatArray):
    def evaluate(x: FloatArray) -> tuple[float, FloatArray]:
        return gaussian_nll_and_gradient(x, returns)

    return evaluate


def fit_custom_bfgs(
    returns: ArrayLike,
    *,
    start: ArrayLike | None = None,
    gtol: float = 1e-6,
    max_iter: int = 500,
) -> GARCHFit:
    values = np.asarray(returns, dtype=float)
    x0 = default_start(values) if start is None else np.asarray(start, dtype=float)
    result: OptimizationResult = bfgs_minimize(
        _objective(values), x0, gtol=gtol, max_iter=max_iter
    )
    return GARCHFit(
        method="Custom BFGS + Armijo",
        params=unconstrained_to_natural(result.x),
        unconstrained=result.x,
        nll=result.fun,
        success=result.success,
        message=result.message,
        nit=result.nit,
        nfev=result.nfev,
        gradient_inf_norm=float(np.linalg.norm(result.jac, np.inf)),
        optimizer_result=result,
    )

def fit_scipy_bfgs(
    returns: ArrayLike,
    *,
    start: ArrayLike | None = None,
    gtol: float = 1e-6,
    max_iter: int = 1000,
) -> GARCHFit:
    """Reference fit using SciPy's independently implemented BFGS solver."""

    values = np.asarray(returns, dtype=float)
    x0 = default_start(values) if start is None else np.asarray(start, dtype=float)

    def fun(x: FloatArray) -> float:
        return gaussian_nll_and_gradient(x, values)[0]

    def jac(x: FloatArray) -> FloatArray:
        return gaussian_nll_and_gradient(x, values)[1]

    result = minimize(
        fun,
        x0,
        method="BFGS",
        jac=jac,
        options={"gtol": gtol, "maxiter": max_iter},
    )
    return GARCHFit(
        method="SciPy BFGS reference",
        params=unconstrained_to_natural(result.x),
        unconstrained=np.asarray(result.x, dtype=float),
        nll=float(result.fun),
        success=bool(result.success),
        message=str(result.message),
        nit=int(result.nit),
        nfev=int(result.nfev),
        gradient_inf_norm=float(np.linalg.norm(result.jac, np.inf)),
        optimizer_result=result,
    )
