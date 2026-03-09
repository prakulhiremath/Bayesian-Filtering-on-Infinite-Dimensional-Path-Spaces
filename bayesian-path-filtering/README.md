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

## Repository Structure

```
bayesian-path-filtering/
├── README.md
├── theory/
│   ├── 01_posterior_existence.md        # Well-posedness on function space
│   ├── 02_contraction_rates.md          # Posterior consistency theorems
│   ├── 03_discretization_stability.md   # Euler-Maruyama convergence
│   └── 04_sigma_problem.md              # Why diffusion estimation is harder
├── notes/
│   ├── literature_map.md                # Key references and how they connect
│   ├── open_problems.md                 # Unresolved questions and conjectures
│   └── proof_sketches.md                # Informal arguments and intuitions
├── simulations/
│   ├── gp_prior.py                      # GP prior sampling on [0,T]
│   ├── girsanov_likelihood.py           # Girsanov density computation
│   ├── posterior_drift.py               # Full posterior for drift-only case
│   ├── discretization_error.py          # Hellinger distance vs mesh size δ
│   └── utils.py                         # Shared numerical utilities
├── notebooks/
│   ├── 01_gp_priors_on_path_space.ipynb
│   ├── 02_girsanov_likelihood.ipynb
│   └── 03_posterior_contraction.ipynb
├── tests/
│   ├── test_gp_prior.py
│   ├── test_girsanov.py
│   └── test_discretization.py
├── references/
│   └── bibliography.bib
├── requirements.txt
└── .gitignore
```

---

## Research Problems

### 1. Posterior Existence in Function Space

Show that $p(\mu, \sigma \mid X_{0:T})$ is a well-defined probability measure on $L^2([0,T])$.

Key requirements:
- Novikov's condition: $\mathbb{E}\!\left[\exp\!\left(\tfrac{1}{2}\int_0^T \mu_t^2 \, dt\right)\right] < \infty$ for GP priors
- Absolute continuity of the posterior w.r.t. the prior
- Cameron-Martin space arguments specific to the chosen kernel class

**Reference framework:** Stuart (2010); Da Prato & Zabczyk (1992).

### 2. Posterior Consistency in Infinite Dimension

**Proposed theorem direction:**

> *If $\mu^* \in H^s([0,T])$ and the GP prior has a Matérn kernel with smoothness $\nu > s$, then the posterior contracts around $\mu^*$ at rate $n^{-s/(2s+1)}$ in $L^2$.*

**Reference framework:** van der Vaart & van Zanten (2008, 2011).

### 3. Stability Under Discretization

**Proposed theorem direction:**

> *The Hellinger distance between the continuous-time posterior and the Euler-Maruyama discretized posterior is $\mathcal{O}(\delta^{1/2})$.*

**Reference framework:** Cotter, Roberts, Stuart & White (2013).

---

## Quickstart

```bash
git clone https://github.com/your-username/bayesian-path-filtering
cd bayesian-path-filtering
pip install -r requirements.txt

# Run GP prior simulation
python simulations/gp_prior.py

# Run posterior drift estimation
python simulations/posterior_drift.py

# Run all tests
pytest tests/
```

---

## Roadmap

| Milestone | Status |
|-----------|--------|
| Literature review: GP nonparametric Bayes | 🔲 |
| Posterior existence for fixed σ (drift-only) | 🔲 |
| Posterior contraction rates (drift, continuous obs.) | 🔲 |
| Discretization stability theorem | 🔲 |
| Joint (μ, σ) posterior: existence | 🔲 |
| Simulation study: GP posterior vs. particle filter | 🔲 |

---

## Key References

- **Stuart, A.M. (2010).** Inverse problems: A Bayesian perspective. *Acta Numerica*, 19, 451–559.
- **van der Vaart, A. & van Zanten, H. (2008).** Rates of contraction of posterior distributions based on Gaussian process priors. *Annals of Statistics*, 36(3), 1435–1463.
- **van der Vaart, A. & van Zanten, H. (2011).** Information rates of nonparametric Gaussian process methods. *JMLR*, 12, 2095–2119.
- **Cotter, S.L., Roberts, G.O., Stuart, A.M., & White, D. (2013).** MCMC methods for functions. *Statistical Science*, 28(3), 424–446.
- **Da Prato, G. & Zabczyk, J. (1992).** *Stochastic Equations in Infinite Dimensions.* Cambridge University Press.
- **Liptser, R.S. & Shiryaev, A.N. (2001).** *Statistics of Random Processes.* Springer.

---

## License

MIT
