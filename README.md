# Bayesian Filtering on Infinite-Dimensional Path Spaces

> Posterior inference over stochastic drift and diffusion functions via Gaussian process priors and Girsanov-based likelihoods.

---

## Overview

Standard Bayesian filtering estimates a **finite parameter vector**. This project investigates something fundamentally harder: placing a prior directly over the **infinite-dimensional function space** of drift and diffusion coefficients in a stochastic differential equation, then computing the posterior conditioned on observed paths.

Given a price process:

$$dX_t = \mu_t \, dt + \sigma_t \, dW_t$$

where $\mu_t, \sigma_t$ are **unknown stochastic functions**, we define:

- Prior: $\mu \sim \mathcal{GP}(0, k_\mu)$, $\sigma \sim \mathcal{GP}(0, k_\sigma)$
- Likelihood: Girsanov density $\frac{d\mathbb{P}^{\mu}}{d\mathbb{P}^0} = \exp\!\left(\int_0^T \mu_t \, dX_t - \tfrac{1}{2}\int_0^T \mu_t^2 \, dt\right)$
- Posterior: $p(\mu, \sigma \mid X_{0:T})$

The central challenge is that the unknown lives in $L^2([0,T])$ — there is no Lebesgue measure in infinite dimensions, and standard Bayes' theorem must be replaced by Radon-Nikodym arguments on path space.

---

## Research Problems

### 1. Posterior Existence in Function Space

Show that $p(\mu, \sigma \mid X_{0:T})$ is a well-defined probability measure on $L^2([0,T])$.

Key requirements:
- Novikov's condition: $\mathbb{E}\!\left[\exp\!\left(\tfrac{1}{2}\int_0^T \mu_t^2 \, dt\right)\right] < \infty$ for GP priors
- Absolute continuity of the posterior w.r.t. the prior
- Cameron-Martin space arguments specific to the chosen kernel class

**Reference framework:** Stuart (2010) — Bayesian inverse problems in Banach spaces; Da Prato & Zabczyk — SPDEs in infinite dimensions.

### 2. Posterior Consistency in Infinite Dimension

As observations become dense (or $T \to \infty$), does the posterior concentrate around the true $(\mu^*, \sigma^*)$?

**Proposed theorem direction:**

> *If $\mu^* \in H^s([0,T])$ and the GP prior has a Matérn kernel with smoothness parameter $\nu > s$, then the posterior contracts around $\mu^*$ at rate $n^{-s/(2s+1)}$ in $L^2$.*

This rate is minimax optimal for nonparametric drift estimation. The SDE setting introduces ill-posedness absent in direct regression: $\mu_t$ is observed only through an integral functional (the path $X_t$), slowing contraction relative to the direct-observation case.

**Reference framework:** van der Vaart & van Zanten (2008, 2011) on GP posterior contraction; Schwartz's theorem generalized to infinite-dimensional spaces.

### 3. Stability Under Discretization

Does the posterior computed from discrete observations $X_{t_0}, \ldots, X_{t_n}$ converge to the continuous-time posterior as mesh $\delta = \max|t_{i+1} - t_i| \to 0$?

**Proposed theorem direction:**

> *The Hellinger distance between the continuous-time posterior and the Euler-Maruyama discretized posterior is $\mathcal{O}(\delta^{1/2})$, matching the strong convergence rate of Euler-Maruyama.*

**Reference framework:** Cotter, Roberts, Stuart & White (2013) — MCMC for infinite-dimensional Bayesian problems; preconditioned Crank-Nicolson (pCN) proposals.

---

## The $\sigma$ Problem

Diffusion coefficient estimation is harder than drift estimation for structural reasons:

- $\mu$ is **non-identifiable** from quadratic variation at fixed $T$ — only $\sigma$ is identified from QV
- The joint posterior $p(\mu, \sigma \mid X_{0:T})$ is **not product-structured** — $\mu$ and $\sigma$ are a posteriori dependent even under independent priors
- The Girsanov density involves $\sigma^{-1}$, creating integrability issues when the GP prior assigns mass near zero

Addressing this requires log-normal GP priors (to enforce positivity) and separate treatment via local time or $p$-variation arguments.

---

## Scope and Roadmap

| Milestone | Status |
|-----------|--------|
| Literature review: GP nonparametric Bayes | 🔲 |
| Posterior existence for fixed $\sigma$ (drift-only) | 🔲 |
| Posterior contraction rates (drift, continuous obs.) | 🔲 |
| Discretization stability theorem | 🔲 |
| Joint $(\mu, \sigma)$ posterior: existence | 🔲 |
| Simulation study: GP posterior vs. particle filter | 🔲 |

The recommended near-term scope is **"Posterior contraction rates for GP priors on SDE drift functions under continuous observation"** — tractable, connects cleanly to existing literature, and produces a self-contained theorem.

---

## Key References

- **Stuart, A.M. (2010).** Inverse problems: A Bayesian perspective. *Acta Numerica*, 19, 451–559.
- **van der Vaart, A. & van Zanten, H. (2008).** Rates of contraction of posterior distributions based on Gaussian process priors. *Annals of Statistics*, 36(3), 1435–1463.
- **van der Vaart, A. & van Zanten, H. (2011).** Information rates of nonparametric Gaussian process methods. *Journal of Machine Learning Research*, 12, 2095–2119.
- **Cotter, S.L., Roberts, G.O., Stuart, A.M., & White, D. (2013).** MCMC methods for functions: modifying old algorithms to make them faster. *Statistical Science*, 28(3), 424–446.
- **Da Prato, G. & Zabczyk, J. (1992).** *Stochastic Equations in Infinite Dimensions.* Cambridge University Press.
- **Liptser, R.S. & Shiryaev, A.N. (2001).** *Statistics of Random Processes.* Springer.

---

## Contributing

This is an active research project. If you are working on related problems in:
- Nonparametric Bayesian inference for SDEs
- Gaussian process priors on path spaces
- Infinite-dimensional MCMC

feel free to open an issue or reach out.

---

## License

MIT
