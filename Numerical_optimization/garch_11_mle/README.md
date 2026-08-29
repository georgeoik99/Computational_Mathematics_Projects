# GARCH(1,1) Maximum-Likelihood Optimization

This project turns Gaussian conditional maximum-likelihood estimation of a GARCH(1,1) model into a transparent numerical-optimization study. The mathematical problem, assumptions, derivatives, constraints and convergence limitations come first; the Python implementation, controlled simulation, diagnostics and solver comparison follow.

The experiment is deliberately synthetic. Its true parameters and latent volatility are known, so parameter recovery and volatility filtering can be tested rather than merely illustrated.

## 1. Mathematical problem

For returns (r_1,\ldots,r_T), let

\[
r_t=\mu+\varepsilon_t, \qquad
\varepsilon_t=\sqrt{h_t}z_t, \qquad
z_t\overset{\mathrm{iid}}{\sim}N(0,1),
\]

with the variance recursion

\[
h_t=\omega+\alpha\varepsilon_{t-1}^2+\beta h_{t-1}.
\]

The natural parameter vector is

\[
\theta=(\mu,\omega,\alpha,\beta)^\top.
\]

The admissible parameter set is

\[
\omega>0, \qquad \alpha\ge 0, \qquad \beta\ge 0,
\qquad \alpha+\beta<1.
\]

Non-negativity makes the conditional variances positive. The last inequality gives finite unconditional variance under the maintained model:

\[
\bar h=\operatorname{Var}(\varepsilon_t)
=\frac{\omega}{1-\alpha-\beta}.
\]

It is useful not to conflate three different conditions:

- (alpha+\beta<1) is the finite-second-moment or weak covariance-stationarity condition used by this project;
- strict stationarity is associated with (E[\log(\alpha z_t^2+\beta)]<0);
- under Gaussian innovations, a finite fourth moment requires (3\alpha^2+2\alpha\beta+\beta^2<1).

The fourth-moment condition is not imposed by the estimator. With non-Gaussian real returns, the same Gaussian objective can be treated as a quasi-likelihood, but robust sandwich inference would then be preferable to the model-based covariance used here.

## 2. Conditional likelihood

The filter is initialized at the parameter-dependent unconditional variance,

\[
h_1=\frac{\omega}{1-\alpha-\beta}.
\]

For a Gaussian innovation, observation (t) contributes

\[
\ell_t(\theta)
=-\frac12\left[
\log(2\pi)+\log h_t+\frac{\varepsilon_t^2}{h_t}
\right].
\]

The optimization problem is therefore

\[
\widehat\theta
=\arg\min_{\theta\in\Theta}
f(\theta),
\qquad
f(\theta)=-\sum_{t=1}^{T}\ell_t(\theta).
\]

Log-densities are summed; products of densities are never formed. The GARCH likelihood is not generally convex, so a successful local optimizer does not prove that a global optimum has been found.

## 3. Analytic score recursion

Let (q=\nabla_\theta\varepsilon_t=(-1,0,0,0)^\top) and (d_t=\nabla_\theta h_t). Because the initialization depends on the parameters,

\[
d_1=
\left(
0,
\frac{1}{1-\alpha-\beta},
\frac{\omega}{(1-\alpha-\beta)^2},
\frac{\omega}{(1-\alpha-\beta)^2}
\right)^\top.
\]

For (t\ge2), differentiation of the filter gives

\[
d_t=
\begin{pmatrix}
0\\1\\\varepsilon_{t-1}^2\\h_{t-1}
\end{pmatrix}
+2\alpha\varepsilon_{t-1}q
+\beta d_{t-1}.
\]

The negative-loglikelihood gradient contribution is

\[
\nabla_\theta f_t
=\frac12\left[
\left(\frac1{h_t}-\frac{\varepsilon_t^2}{h_t^2}\right)d_t
+\frac{2\varepsilon_t}{h_t}q
\right].
\]

The implementation evaluates the variance and derivative recursions in one pass. A central finite-difference test independently checks the resulting analytic gradient.

## 4. Constraint-respecting parameter scale

Optimization is performed over an unconstrained vector (u\in\mathbb R^4). Define (c=1-10^{-8}) and

\[
\mu=u_\mu, \qquad \omega=e^{u_\omega},
\]

\[
\alpha=c\frac{e^{u_\alpha}}{1+e^{u_\alpha}+e^{u_\beta}},
\qquad
\beta=c\frac{e^{u_\beta}}{1+e^{u_\alpha}+e^{u_\beta}}.
\]

This smooth map enforces positivity and leaves a strictly positive stationarity slack. The chain rule gives

\[
\nabla_u f=J_{\theta\leftarrow u}^\top\nabla_\theta f.
\]

The nonlinear stationarity transform is a GARCH-specific extension; it is not presented as material copied from the course notes.

## 5. Optimization algorithm and stopping rules

The primary estimator is inverse-BFGS implemented from scratch. At iteration (k),

\[
p_k=-H_k\nabla f(u_k),
\qquad
u_{k+1}=u_k+a_kp_k,
\]

where (a_k) is found by Armijo backtracking. If the BFGS direction is not a descent direction, the algorithm resets to steepest descent. The rank-two update is skipped when the curvature quantity (y_k^Ts_k) is numerically inadequate.

Termination is explicit:

- (|\nabla f(u_k)|_\infty\le10^{-6}); or
- both the relative step and relative objective change are below tolerance; or
- failure of the line search; or
- 500 iterations.

The trace records objective, gradient norm, step norm and accepted step size. Armijo backtracking supplies monotone sufficient decrease. Classical BFGS superlinear convergence results require stronger assumptions, commonly smoothness, positive curvature and Wolfe-type line searches. They do not provide a global guarantee for this non-convex likelihood. Five dispersed starts and an independent solver are therefore checked explicitly.

SciPy's independently implemented BFGS, started from the same documented default rule but run separately, is the reference solver.

## 6. Inference, profile likelihood and diagnostics

The observed information is the Hessian of the negative log-likelihood at the MLE. It is computed by central differences of the analytic gradient and must be positive definite before it is inverted. Covariance is first obtained on the unconstrained scale and then moved to the natural scale with the delta method:

\[
\widehat{\operatorname{Cov}}(\widehat\theta)
=J\,\widehat{\operatorname{Cov}}(\widehat u)J^\top.
\]

The project reports model-based Wald intervals and a likelihood-ratio profile interval for persistence (ho=\alpha+\beta). For every fixed candidate (ho), (mu), (omega) and the split between (alpha) and (eta) are re-estimated. The 95% profile region satisfies

\[
2\{\ell(\widehat\theta)-\ell_p(\rho)\}
\le\chi^2_{1,0.95}.
\]

Standardized residuals (z_t=\varepsilon_t/\sqrt{h_t}) are assessed with histograms, autocorrelations and Ljung-Box statistics for both (z_t) and (z_t^2). AIC and BIC compare GARCH against a Gaussian constant-variance baseline. Diagnostics assess model adequacy; they are not optimizer convergence tests.

## 7. Reproducible experiment

The fixed data-generating process is

\[
(\mu,\omega,\alpha,\beta)=(0.02,0.05,0.08,0.88),
\qquad \alpha+\beta=0.96.
\]

The experiment uses 4,000 retained observations, a 1,000-observation burn-in and seed 2026. Returns are in percentage units. Twelve smaller Monte Carlo samples measure bias, RMSE and convergence frequency; 12 replications are illustrative rather than a high-precision Monte Carlo study.

Key deterministic-run results:

| Quantity | Result |
|---|---:|
| Custom BFGS iterations / evaluations | 11 / 41 |
| Final gradient infinity norm | (2.77\times10^{-8}) |
| Custom vs SciPy NLL difference | (9.09\times10^{-13}) |
| Maximum parameter difference vs SciPy | (2.72\times10^{-10}) |
| Estimated persistence | 0.96421 |
| True persistence | 0.96000 |
| 95% profile-likelihood interval for persistence | [0.94442, 0.97889] |
| Minimum observed-information eigenvalue | 6.66464 |
| Successful starts / total starts | 5 / 5 |
| Monte Carlo convergence rate | 12 / 12 |

The fitted natural parameters are

| Parameter | True | Estimate | Model-based SE | Wald 95% interval |
|---|---:|---:|---:|---:|
| (mu) | 0.0200 | -0.00082 | 0.01599 | [-0.03216, 0.03053] |
| (omega) | 0.0500 | 0.04459 | 0.01010 | [0.02480, 0.06438] |
| (alpha) | 0.0800 | 0.08220 | 0.01094 | [0.06076, 0.10364] |
| (eta) | 0.8800 | 0.88201 | 0.01639 | [0.84988, 0.91414] |

The Ljung-Box p-values are 0.894 for standardized residuals and 0.996 for squared standardized residuals in this locked synthetic run. These large values are consistent with the correctly specified DGP, but they should not be treated as guaranteed outcomes across random samples.

## 8. Relationship to the MSc material

The project is faithful to the course's numerical workflow, not a claim that GARCH itself appears in the slides.

| Course-derived principle | GARCH-specific application or extension |
|---|---|
| State the objective and derivatives before code | Conditional Gaussian likelihood and recursive score |
| Use a parameter scale that respects constraints | Softmax stationarity transform for (alpha+\beta<1) |
| Implement a transparent iterative method with safeguards | BFGS, Armijo backtracking, trace and curvature checks |
| Inspect convergence code, objective and sensitivity | Gradient norm, iteration trace and independent start |
| Verify custom code with an independent optimizer | SciPy BFGS comparison |
| Use a numerical Hessian for observed-information inference | Central differences of the analytic GARCH score |
| Report Wald and profile-likelihood uncertainty | Wald intervals and profile interval for persistence |
| Use simulation with fixed seeds and known targets | Parameter recovery and Monte Carlo bias/RMSE |
| Compare AIC but retain diagnostics | Constant variance vs GARCH plus residual checks |

GARCH stationarity theory, score recursions, the nonlinear transform, volatility diagnostics and delta-method transformation are domain-specific additions. They are labeled as such because they are absent from the supplied notes.

## 9. Project structure

```text
garch_11_mle/
├── README.md
├── garch_11_mle.ipynb
├── pyproject.toml
├── requirements.txt
├── data/
│   ├── README.md
│   └── synthetic_garch11_returns.csv
├── scripts/
│   └── run_experiment.py
├── src/garch_mle/
│   ├── model.py
│   ├── optimizers.py
│   ├── estimation.py
│   ├── inference.py
│   ├── diagnostics.py
│   └── forecasting.py
├── tests/
└── results/
    ├── figures/
    ├── tables/
    └── summary.json
```

## 10. Reproduce

From this directory:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -e .
.venv/Scripts/python -m pytest
.venv/Scripts/python scripts/run_experiment.py
```

On macOS or Linux, replace `.venv/Scripts/python` by `.venv/bin/python`.

The script regenerates the synthetic CSV, all result tables, six figures and `results/summary.json`. Tests never access the network.
