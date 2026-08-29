"""Run the reproducible GARCH(1,1) synthetic-data experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from garch_mle.diagnostics import (  # noqa: E402
    autocorrelation,
    gaussian_constant_variance_nll,
    information_criteria,
    ljung_box,
    standardized_residuals,
)
from garch_mle.estimation import default_start, fit_custom_bfgs, fit_scipy_bfgs  # noqa: E402
from garch_mle.forecasting import variance_forecast  # noqa: E402
from garch_mle.inference import observed_information, profile_persistence  # noqa: E402
from garch_mle.model import (  # noqa: E402
    GARCHParameters,
    gaussian_nll_and_gradient,
    natural_to_unconstrained,
    simulate_garch11,
)

PARAMETER_NAMES = ["mu", "omega", "alpha", "beta"]


def save_figures(
    returns: np.ndarray,
    true_variance: np.ndarray,
    true_params: GARCHParameters,
    custom_fit,
    reference_fit,
    output_dir: Path,
) -> None:
    from garch_mle.model import filter_variance

    output_dir.mkdir(parents=True, exist_ok=True)
    fitted_variance = filter_variance(returns, custom_fit.params)
    window = min(600, returns.size)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(returns[:window], color="#365b8c", linewidth=0.7)
    axes[0].set_ylabel("Return (%)")
    axes[0].set_title("Synthetic GARCH(1,1) returns")
    axes[1].plot(np.sqrt(true_variance[:window]), label="True volatility", linewidth=1.4)
    axes[1].plot(np.sqrt(fitted_variance[:window]), label="Fitted volatility", linewidth=1.0, alpha=0.85)
    axes[1].set_ylabel("Conditional volatility")
    axes[1].set_xlabel("Time")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "returns_and_volatility.png", dpi=180)
    plt.close(fig)

    history = custom_fit.optimizer_result.history
    iterations = [item.iteration for item in history]
    objectives = [item.objective for item in history]
    gradients = [item.gradient_inf_norm for item in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(iterations, objectives, marker="o", markersize=2)
    axes[0].axhline(reference_fit.nll, color="#b2433f", linestyle="--", label="SciPy reference")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Negative log-likelihood")
    axes[0].legend()
    axes[1].semilogy(iterations, np.maximum(gradients, 1e-16), marker="o", markersize=2)
    axes[1].axhline(1e-6, color="#b2433f", linestyle="--", label="gtol")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Gradient infinity norm")
    axes[1].legend()
    fig.suptitle("Custom BFGS convergence trace")
    fig.tight_layout()
    fig.savefig(output_dir / "optimizer_trace.png", dpi=180)
    plt.close(fig)

    estimates = np.vstack(
        [true_params.as_array(), custom_fit.params.as_array(), reference_fit.params.as_array()]
    )
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    labels = ["True", "Custom", "SciPy"]
    for index, name in enumerate(PARAMETER_NAMES):
        axes[index].bar(labels, estimates[:, index], color=["#5a8f5a", "#365b8c", "#b2433f"])
        axes[index].set_title(name)
        axes[index].tick_params(axis="x", rotation=25)
    fig.suptitle("Parameter recovery and reference-solver agreement")
    fig.tight_layout()
    fig.savefig(output_dir / "parameter_comparison.png", dpi=180)
    plt.close(fig)

    residuals = standardized_residuals(returns, custom_fit.params)
    max_lag = 20
    acf_residuals = autocorrelation(residuals, max_lag)
    acf_squares = autocorrelation(residuals**2, max_lag)
    lags = np.arange(1, max_lag + 1)
    band = 1.96 / np.sqrt(returns.size)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].hist(residuals, bins=35, density=True, color="#8da0cb", edgecolor="white")
    x = np.linspace(-4, 4, 300)
    axes[0].plot(x, np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi), color="#b2433f")
    axes[0].set_title("Standardized residuals")
    for axis, values, title in zip(
        axes[1:], [acf_residuals, acf_squares], ["ACF: residuals", "ACF: squared residuals"]
    ):
        axis.bar(lags, values, color="#365b8c")
        axis.axhline(band, color="#b2433f", linestyle="--")
        axis.axhline(-band, color="#b2433f", linestyle="--")
        axis.set_xlabel("Lag")
        axis.set_title(title)
    fig.tight_layout()
    fig.savefig(output_dir / "residual_diagnostics.png", dpi=180)
    plt.close(fig)

    alpha_grid = np.linspace(0.02, 0.20, 28)
    beta_grid = np.linspace(0.72, 0.97, 30)
    surface = np.full((beta_grid.size, alpha_grid.size), np.nan)
    for row, beta in enumerate(beta_grid):
        for column, alpha in enumerate(alpha_grid):
            if alpha + beta < 0.995:
                candidate = GARCHParameters(
                    custom_fit.params.mu, custom_fit.params.omega, float(alpha), float(beta)
                )
                point = natural_to_unconstrained(candidate)
                surface[row, column] = gaussian_nll_and_gradient(point, returns)[0]
    fig, axis = plt.subplots(figsize=(7, 5))
    levels = np.linspace(np.nanmin(surface), np.nanpercentile(surface, 70), 18)
    contour = axis.contourf(alpha_grid, beta_grid, surface, levels=levels, cmap="viridis")
    axis.scatter([custom_fit.params.alpha], [custom_fit.params.beta], c="white", edgecolors="black", label="MLE")
    axis.plot(alpha_grid, 1.0 - alpha_grid, color="red", linestyle="--", label=r"$\alpha+\beta=1$")
    axis.set_xlabel(r"$\alpha$")
    axis.set_ylabel(r"$\beta$")
    axis.set_title("Conditional NLL section at fitted mu and omega")
    axis.legend()
    fig.colorbar(contour, ax=axis, label="Negative log-likelihood")
    fig.tight_layout()
    fig.savefig(output_dir / "likelihood_section.png", dpi=180)
    plt.close(fig)


def monte_carlo_recovery(
    true_params: GARCHParameters,
    replications: int,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    records: list[dict[str, float | int | bool]] = []
    seed_sequence = np.random.SeedSequence(seed)
    for replication, child in enumerate(seed_sequence.spawn(replications), start=1):
        child_seed = int(child.generate_state(1)[0])
        returns, _ = simulate_garch11(sample_size, true_params, burn=500, seed=child_seed)
        fit = fit_custom_bfgs(returns, gtol=2e-6)
        record: dict[str, float | int | bool] = {
            "replication": replication,
            "success": fit.success,
            "nll": fit.nll,
            "iterations": fit.nit,
        }
        record.update(dict(zip(PARAMETER_NAMES, fit.params.as_array(), strict=True)))
        record["persistence"] = fit.params.persistence
        records.append(record)
    return pd.DataFrame.from_records(records)


def run(sample_size: int, replications: int, seed: int) -> dict[str, object]:
    true_params = GARCHParameters(mu=0.02, omega=0.05, alpha=0.08, beta=0.88)
    returns, true_variance = simulate_garch11(sample_size, true_params, burn=1000, seed=seed)
    custom_fit = fit_custom_bfgs(returns)
    reference_fit = fit_scipy_bfgs(returns)
    base_start = default_start(returns)
    start_offsets = [
        np.zeros(4),
        np.array([0.0, 0.7, -0.5, 0.3]),
        np.array([0.0, -0.7, 0.5, -0.3]),
        np.array([0.10, 0.0, 1.0, -1.0]),
        np.array([-0.10, 0.2, -1.0, 1.0]),
    ]
    multistart_fits = [custom_fit] + [
        fit_custom_bfgs(returns, start=base_start + offset)
        for offset in start_offsets[1:]
    ]
    if not custom_fit.success:
        raise RuntimeError(f"Custom BFGS failed: {custom_fit.message}")
    if abs(custom_fit.nll - reference_fit.nll) > 1e-5:
        raise RuntimeError("Custom and reference objectives do not agree within tolerance.")

    inference = observed_information(custom_fit.unconstrained, returns)
    profile = profile_persistence(returns, custom_fit.params, custom_fit.nll)
    result_dir = PROJECT_ROOT / "results"
    table_dir = result_dir / "tables"
    figure_dir = result_dir / "figures"
    data_dir = PROJECT_ROOT / "data"
    table_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"return_percent": returns, "true_variance": true_variance}).to_csv(
        data_dir / "synthetic_garch11_returns.csv", index_label="time"
    )

    comparison = pd.DataFrame(
        [
            {
                "method": fit.method,
                **dict(zip(PARAMETER_NAMES, fit.params.as_array(), strict=True)),
                "persistence": fit.params.persistence,
                "nll": fit.nll,
                "gradient_inf_norm": fit.gradient_inf_norm,
                "iterations": fit.nit,
                "function_evaluations": fit.nfev,
                "success": fit.success,
                "message": fit.message,
            }
            for fit in [custom_fit, reference_fit]
        ]
    )
    comparison.to_csv(table_dir / "optimizer_comparison.csv", index=False)

    multistart_table = pd.DataFrame(
        [
            {
                "start": index,
                **dict(zip(PARAMETER_NAMES, fit.params.as_array(), strict=True)),
                "persistence": fit.params.persistence,
                "nll": fit.nll,
                "gradient_inf_norm": fit.gradient_inf_norm,
                "iterations": fit.nit,
                "success": fit.success,
            }
            for index, fit in enumerate(multistart_fits)
        ]
    )
    multistart_table.to_csv(table_dir / "multistart_comparison.csv", index=False)

    estimates = custom_fit.params.as_array()
    inference_table = pd.DataFrame(
        {
            "parameter": PARAMETER_NAMES,
            "estimate": estimates,
            "standard_error": inference.standard_errors,
            "wald_95_lower": inference.confidence_intervals[:, 0],
            "wald_95_upper": inference.confidence_intervals[:, 1],
            "true_value": true_params.as_array(),
        }
    )
    inference_table.to_csv(table_dir / "wald_inference.csv", index=False)

    pd.DataFrame(
        {"persistence": profile.persistence_grid, "profile_nll": profile.profile_nll}
    ).to_csv(table_dir / "persistence_profile.csv", index=False)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(profile.persistence_grid, profile.profile_nll, color="#365b8c", linewidth=1.6)
    axis.axhline(profile.cutoff, color="#b2433f", linestyle="--", label="95% LR cutoff")
    axis.axvline(custom_fit.params.persistence, color="black", label="MLE")
    axis.axvspan(*profile.confidence_interval, color="#8da0cb", alpha=0.2, label="95% profile interval")
    axis.set_xlabel(r"Persistence $\rho=\alpha+\beta$")
    axis.set_ylabel("Profile negative log-likelihood")
    axis.set_title("Profile likelihood for GARCH persistence")
    axis.legend()
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "persistence_profile.png", dpi=180)
    plt.close(fig)

    residuals = standardized_residuals(returns, custom_fit.params)
    residual_lb = ljung_box(residuals, 20)
    squared_lb = ljung_box(residuals**2, 20)
    diagnostics = pd.DataFrame(
        [
            {"series": "standardized residuals", "lags": 20, "Q": residual_lb.statistic, "p_value": residual_lb.p_value},
            {"series": "squared standardized residuals", "lags": 20, "Q": squared_lb.statistic, "p_value": squared_lb.p_value},
        ]
    )
    diagnostics.to_csv(table_dir / "residual_diagnostics.csv", index=False)

    constant_nll, _, _ = gaussian_constant_variance_nll(returns)
    garch_aic, garch_bic = information_criteria(custom_fit.nll, 4, returns.size)
    constant_aic, constant_bic = information_criteria(constant_nll, 2, returns.size)
    model_comparison = pd.DataFrame(
        [
            {"model": "Gaussian constant variance", "parameters": 2, "nll": constant_nll, "AIC": constant_aic, "BIC": constant_bic},
            {"model": "Gaussian GARCH(1,1)", "parameters": 4, "nll": custom_fit.nll, "AIC": garch_aic, "BIC": garch_bic},
        ]
    )
    model_comparison.to_csv(table_dir / "model_comparison.csv", index=False)

    forecast = variance_forecast(returns, custom_fit.params, 20)
    pd.DataFrame(
        {"horizon": np.arange(1, 21), "variance_forecast": forecast, "volatility_forecast": np.sqrt(forecast)}
    ).to_csv(table_dir / "variance_forecast.csv", index=False)

    mc = monte_carlo_recovery(true_params, replications, min(sample_size, 1500), seed + 1)
    mc.to_csv(table_dir / "monte_carlo_estimates.csv", index=False)
    successful = mc.loc[mc["success"]]
    summary_rows = []
    truth = {**dict(zip(PARAMETER_NAMES, true_params.as_array(), strict=True)), "persistence": true_params.persistence}
    for parameter in [*PARAMETER_NAMES, "persistence"]:
        errors = successful[parameter] - truth[parameter]
        summary_rows.append(
            {
                "parameter": parameter,
                "true_value": truth[parameter],
                "mean_estimate": successful[parameter].mean(),
                "bias": errors.mean(),
                "rmse": np.sqrt(np.mean(errors**2)),
                "successful_replications": len(successful),
            }
        )
    pd.DataFrame(summary_rows).to_csv(table_dir / "monte_carlo_summary.csv", index=False)

    save_figures(returns, true_variance, true_params, custom_fit, reference_fit, figure_dir)

    summary: dict[str, object] = {
        "sample_size": sample_size,
        "seed": seed,
        "monte_carlo_replications": replications,
        "monte_carlo_convergence_rate": float(mc["success"].mean()),
        "multistart_success_rate": float(multistart_table["success"].mean()),
        "multistart_objective_spread": float(
            multistart_table.loc[multistart_table["success"], "nll"].max()
            - multistart_table.loc[multistart_table["success"], "nll"].min()
        ),
        "custom_success": custom_fit.success,
        "reference_success": reference_fit.success,
        "objective_difference": abs(custom_fit.nll - reference_fit.nll),
        "parameter_max_abs_difference": float(
            np.max(np.abs(custom_fit.params.as_array() - reference_fit.params.as_array()))
        ),
        "estimated_persistence": custom_fit.params.persistence,
        "true_persistence": true_params.persistence,
        "observed_information_min_eigenvalue": float(np.min(inference.hessian_eigenvalues)),
        "persistence_profile_95_ci": list(profile.confidence_interval),
        "ljung_box_residuals_p_value": residual_lb.p_value,
        "ljung_box_squared_residuals_p_value": squared_lb.p_value,
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=4000)
    parser.add_argument("--replications", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.sample_size, arguments.replications, arguments.seed), indent=2))


if __name__ == "__main__":
    main()
