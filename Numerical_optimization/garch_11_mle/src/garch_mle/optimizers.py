"""A transparent BFGS implementation with an Armijo backtracking line search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
ObjectiveWithGradient = Callable[[FloatArray], tuple[float, FloatArray]]


@dataclass(frozen=True)
class OptimizationIteration:
    iteration: int
    objective: float
    gradient_inf_norm: float
    step_norm: float
    step_size: float


@dataclass(frozen=True)
class OptimizationResult:
    x: FloatArray
    fun: float
    jac: FloatArray
    success: bool
    message: str
    nit: int
    nfev: int
    history: tuple[OptimizationIteration, ...]


def bfgs_minimize(
    objective: ObjectiveWithGradient,
    x0: ArrayLike,
    *,
    gtol: float = 1e-6,
    xtol: float = 1e-10,
    ftol: float = 1e-12,
    max_iter: int = 500,
    armijo: float = 1e-4,
    contraction: float = 0.5,
    min_step: float = 1e-12,
) -> OptimizationResult:
    """Minimize a smooth function by inverse-BFGS and Armijo backtracking."""

    x = np.asarray(x0, dtype=float).copy()
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("x0 must be a finite one-dimensional vector.")
    if not (0.0 < armijo < 1.0 and 0.0 < contraction < 1.0):
        raise ValueError("Invalid line-search constants.")

    objective_value, gradient = objective(x)
    nfev = 1
    if not np.isfinite(objective_value) or not np.all(np.isfinite(gradient)):
        raise ValueError("The objective and gradient must be finite at x0.")

    inverse_hessian = np.eye(x.size)
    history: list[OptimizationIteration] = [
        OptimizationIteration(0, objective_value, float(np.linalg.norm(gradient, np.inf)), 0.0, 0.0)
    ]

    for iteration in range(1, max_iter + 1):
        gradient_norm = float(np.linalg.norm(gradient, np.inf))
        if gradient_norm <= gtol:
            return OptimizationResult(
                x, objective_value, gradient, True,
                "Gradient infinity norm satisfies gtol.",
                iteration - 1, nfev, tuple(history),
            )

        direction = -inverse_hessian @ gradient
        directional_derivative = float(gradient @ direction)
        if directional_derivative >= -1e-14 * max(1.0, float(np.linalg.norm(gradient))):
            direction = -gradient
            inverse_hessian = np.eye(x.size)
            directional_derivative = -float(gradient @ gradient)

        step_size = 1.0
        accepted = False
        while step_size >= min_step:
            candidate = x + step_size * direction
            candidate_value, candidate_gradient = objective(candidate)
            nfev += 1
            if (
                np.isfinite(candidate_value)
                and np.all(np.isfinite(candidate_gradient))
                and candidate_value <= objective_value + armijo * step_size * directional_derivative
            ):
                accepted = True
                break
            step_size *= contraction

        if not accepted:
            return OptimizationResult(
                x, objective_value, gradient, False,
                "Armijo line search failed to find an admissible descent step.",
                iteration - 1, nfev, tuple(history),
            )

        step = candidate - x
        gradient_change = candidate_gradient - gradient
        curvature = float(gradient_change @ step)
        scale = float(np.linalg.norm(step) * np.linalg.norm(gradient_change))
        if curvature > 1e-12 * max(1.0, scale):
            rho = 1.0 / curvature
            identity = np.eye(x.size)
            left = identity - rho * np.outer(step, gradient_change)
            inverse_hessian = (
                left @ inverse_hessian @ left.T + rho * np.outer(step, step)
            )

        previous_value = objective_value
        x = candidate
        objective_value = float(candidate_value)
        gradient = np.asarray(candidate_gradient, dtype=float)
        step_norm = float(np.linalg.norm(step))
        history.append(
            OptimizationIteration(
                iteration,
                objective_value,
                float(np.linalg.norm(gradient, np.inf)),
                step_norm,
                step_size,
            )
        )

        small_step = step_norm <= xtol * (1.0 + float(np.linalg.norm(x)))
        small_change = abs(previous_value - objective_value) <= ftol * (1.0 + abs(previous_value))
        if small_step and small_change:
            return OptimizationResult(
                x, objective_value, gradient, True,
                "Step and relative objective change satisfy xtol and ftol.",
                iteration, nfev, tuple(history),
            )

    return OptimizationResult(
        x, objective_value, gradient, False,
        "Maximum number of BFGS iterations reached.",
        max_iter, nfev, tuple(history),
    )
