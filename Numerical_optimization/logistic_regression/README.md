# Dose–Response Logistic Regression

Self-contained academic notebook using the classic tobacco-budworm dose–mortality experiment.

## Contents

- `logistic_regression.ipynb`: grouped-binomial likelihood, analytic gradient and Hessian, convexity and separation, custom gradient descent and damped Newton with Armijo backtracking, SciPy reference solvers, inference, likelihood-ratio testing, diagnostics, dose–response plots, and LD50 estimation.
- `data/budworm_dose_mortality.csv`: 12 experimental groups representing 240 binary killed/alive outcomes.

Run the notebook from this directory so that the relative `data/` path resolves correctly. Dependencies are listed in the parent directory's `notebook_requirements.txt`.

## Data source

The values are reproduced from the official R/MASS `dose.p` example and are documented in Venables & Ripley, *Modern Applied Statistics*, and Collett, *Modelling Binary Data*:

- https://stat.ethz.ch/R-manual/R-devel/RHOME/library/MASS/html/dose.p.html
- https://stat.ethz.ch/CRAN/web/packages/GLMsData/GLMsData.pdf
