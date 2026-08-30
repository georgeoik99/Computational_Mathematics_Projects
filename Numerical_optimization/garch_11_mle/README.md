# GARCH(1,1) Maximum-Likelihood Optimization

Self-contained academic Jupyter notebook on constrained Gaussian/QML estimation of a GARCH(1,1) model.

## Contents

- `garch_11_mle.ipynb`: mathematical formulation, assumptions, likelihood, analytic score recursion, constraint transform, custom BFGS with Armijo backtracking, convergence discussion, inference, diagnostics, plots, and comparison with SciPy BFGS.
- `data/greek_equity_prices.csv`: local real-price dataset used for the OTE daily-return case study.

The notebook also generates a reproducible synthetic GARCH sample with known parameters for gradient verification and parameter-recovery testing. All implementation code is contained directly in the notebook; no local Python package is required.

## Run

Install the dependencies listed in the parent directory's `notebook_requirements.txt`, open `garch_11_mle.ipynb`, and run all cells from this directory so that the relative `data/` path resolves correctly.
