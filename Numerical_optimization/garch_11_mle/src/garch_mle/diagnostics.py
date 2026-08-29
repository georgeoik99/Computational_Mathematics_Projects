"""Residual and information-criterion diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2

from .model import GARCHParameters, filter_variance

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LjungBoxResult:
    lags: int
    statistic: float
    p_value: float


def autocorrelation(values: ArrayLike, max_lag: int) -> FloatArray:
    sample = np.asarray(values, dtype=float)
    centered = sample - np.mean(sample)
    denominator = float(centered @ centered)
    if denominator <= 0.0:
        raise ValueError("Autocorrelation is undefined for a constant sample.")
    if max_lag < 1 or max_lag >= sample.size:
        raise ValueError("max_lag must be between 1 and n - 1.")
    return np.array(
        [float(centered[lag:] @ centered[:-lag]) / denominator for lag in range(1, max_lag + 1)]
    )


def ljung_box(values: ArrayLike, lags: int = 20) -> LjungBoxResult:
    sample = np.asarray(values, dtype=float)
    correlations = autocorrelation(sample, lags)
    indices = np.arange(1, lags + 1)
    statistic = sample.size * (sample.size + 2.0) * np.sum(
        correlations**2 / (sample.size - indices)
    )
    return LjungBoxResult(lags, float(statistic), float(chi2.sf(statistic, lags)))


def standardized_residuals(returns: ArrayLike, params: GARCHParameters) -> FloatArray:
    values = np.asarray(returns, dtype=float)
    variances = filter_variance(values, params)
    return (values - params.mu) / np.sqrt(variances)


def gaussian_constant_variance_nll(returns: ArrayLike) -> tuple[float, float, float]:
    values = np.asarray(returns, dtype=float)
    mean = float(np.mean(values))
    variance = float(np.mean((values - mean) ** 2))
    nll = 0.5 * values.size * (np.log(2.0 * np.pi) + np.log(variance) + 1.0)
    return float(nll), mean, variance


def information_criteria(nll: float, n_parameters: int, n_observations: int) -> tuple[float, float]:
    aic = 2.0 * n_parameters + 2.0 * nll
    bic = np.log(n_observations) * n_parameters + 2.0 * nll
    return float(aic), float(bic)
