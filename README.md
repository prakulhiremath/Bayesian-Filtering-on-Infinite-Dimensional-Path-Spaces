<div align="center">

<br/>

```
                      ╔══════════════════════════════════════════════════════════════════╗
                      ║   BAYESIAN FILTERING ON INFINITE-DIMENSIONAL PATH SPACES         ║
                      ║   Posterior inference over stochastic operators via              ║
                      ║   Gaussian Process priors & Girsanov-Radon-Nikodym derivatives  ║
                      ╚══════════════════════════════════════════════════════════════════╝
```

[![arXiv](https://img.shields.io/badge/arXiv-2604.XXXXX-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2604.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-0f172a?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3b82f6?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Status: Research](https://img.shields.io/badge/Status-Active%20Research-22c55e?style=flat-square)]()
[![Affiliation](https://img.shields.io/badge/Affiliation-AoE%20Research%20Group-6366f1?style=flat-square)](https://aliensonearth.in)

<br/>

*Where the state is not a vector. It is a path.*

<br/>

</div>

---

## The Problem in One Paragraph

Standard Bayesian filtering — Kalman, EKF, particle filters — estimates a **finite-dimensional vector** $z \in \mathbb{R}^n$. But what if the unknown is a **function**? Given an observed price process

$$dX_t = \mu(X_t)\,dt + \sigma(X_t)\,dW_t$$

the drift $\mu(\cdot)$ and diffusion $\sigma(\cdot)$ are **infinite-dimensional objects** living in $L^2([0,T])$. No Lebesgue measure exists there. Standard Bayes' theorem breaks. Particle weights collapse to a single point regardless of sample size. This repository builds the mathematical and computational infrastructure to solve this — rigorously, from measure theory up.

---

## Table of Contents

- [Mathematical Formulation](#1-mathematical-formulation)
- [Why Classical Filters Fail](#2-why-classical-filters-fail-the-measure-singularity-problem)
- [Our Approach](#3-our-approach)
- [Core Research Results](#4-core-research-results)
- [Repository Architecture](#5-repository-architecture)
- [Quickstart](#6-quickstart)
- [Theoretical Roadmap](#7-theoretical-roadmap)
- [Key References](#8-key-references)
- [Citation](#9-citation)

---

## 1. Mathematical Formulation

### 1.1 The Observation Model

Let $(\Omega, \mathcal{F}, \mathbb{P})$ be a complete probability space and $T = [0, \tau]$ a compact horizon. We observe a continuous semimartingale:

$$dX_t = \mu(X_t)\,dt + \sigma(X_t)\,dW_t, \quad X_0 \sim \pi_0$$

where $W$ is a standard Wiener process, and **both** $\mu : \mathbb{R} \to \mathbb{R}$ and $\sigma : \mathbb{R} \to \mathbb{R}_{>0}$ are unknown functions — the objects of inference.

### 1.2 The Infinite-Dimensional Prior

We place Gaussian Process priors over the functional unknowns:

$$\mu \sim \mathcal{GP}(0,\, \mathcal{K}_\mu), \qquad \log\sigma \sim \mathcal{GP}(0,\, \mathcal{K}_\sigma)$$

The covariance kernels $\mathcal{K}$ belong to the **Matérn class** with smoothness $\nu$, placing $\mu$ in the Sobolev space $H^s([0,T])$ for $s = \nu - \tfrac{1}{2}$ with probability one. The GP prior is a Gaussian measure $\nu_0 = \mathcal{N}(0, \mathcal{C})$ on the Hilbert space $\mathcal{H} = L^2([0,T])$, where $\mathcal{C}$ is a **trace-class, self-adjoint, positive** covariance operator:

$$\operatorname{tr}(\mathcal{C}) = \sum_{j=1}^{\infty} \lambda_j < \infty$$

> **Why trace-class?** This is the necessary and sufficient condition for a Gaussian measure to be $\sigma$-additive on a separable Hilbert space (Da Prato & Zabczyk, 2014, Thm. 2.3.1). Without it, the prior is not a valid probability measure.

### 1.3 The Likelihood via Girsanov's Theorem

In infinite dimensions, there is no density against Lebesgue measure. Instead, the likelihood is a **Radon-Nikodym derivative** — a change of measure on path space $\mathcal{C}([0,T], \mathbb{R})$:

$$\mathcal{L}(\mu;\, X_{0:T}) = \frac{d\mathbb{P}^\mu}{d\mathbb{P}^0}(X) = \exp\!\left(\int_0^T \frac{\mu(X_t)}{\sigma^2(X_t)}\,dX_t - \frac{1}{2}\int_0^T \frac{|\mu(X_t)|^2}{\sigma^2(X_t)}\,dt\right)$$

where $\mathbb{P}^0$ is the reference (zero-drift) Wiener measure. This expression is well-defined if and only if **Novikov's condition** holds:

$$\mathbb{E}\!\left[\exp\!\left(\frac{1}{2}\int_0^T \frac{|\mu(X_t)|^2}{\sigma^2(X_t)}\,dt\right)\right] < \infty$$

We verify this for GP priors via a Fernique-type tail bound on the Cameron-Martin norm $\|\mu\|_\mathcal{E} = \|\mathcal{C}^{-1/2}\mu\|_\mathcal{H}$.

### 1.4 The Posterior

By the abstract Bayes formula on function space (Stuart, 2010, Thm. 4.1):

$$\frac{d\nu^X}{d\nu_0}(\mu) = \frac{1}{Z(X)}\exp\!\Big(-\Phi(\mu;\, X_{0:T})\Big), \quad Z(X) = \int_\mathcal{H} e^{-\Phi(\mu;\,X)}\,\nu_0(d\mu)$$

where the **potential** $\Phi(\mu; X) = -\log \mathcal{L}(\mu; X)$ encodes the Girsanov likelihood. The posterior $\nu^X$ is a well-defined probability measure on $(\mathcal{H}, \mathcal{B}(\mathcal{H}))$, absolutely continuous with respect to the GP prior $\nu_0$.

---

## 2. Why Classical Filters Fail: The Measure Singularity Problem

<table>
<tr>
<th>Issue</th>
<th>Finite Dimensions ($\mathbb{R}^n$)</th>
<th>Infinite Dimensions ($\mathcal{H}$)</th>
</tr>
<tr>
<td><strong>Reference measure</strong></td>
<td>Lebesgue measure $\lambda^n$ exists</td>
<td><strong>No</strong> translation-invariant $\sigma$-finite measure (Bogachev, 1998)</td>
</tr>
<tr>
<td><strong>Bayes' theorem</strong></td>
<td>$p(\theta|x) \propto p(x|\theta)\,p(\theta)$ via densities</td>
<td>Must use Radon-Nikodym derivative; no densities w.r.t. Lebesgue</td>
</tr>
<tr>
<td><strong>Particle filter ESS</strong></td>
<td>Degrades as $O(n)$</td>
<td><strong>Collapses to 1</strong> regardless of $N$ (Bickel et al., 2008)</td>
</tr>
<tr>
<td><strong>Measure equivalence</strong></td>
<td>Gaussian shifts are always equivalent</td>
<td>Feldman-Hájek: two Gaussians are either equivalent <em>or</em> singular — no middle ground</td>
</tr>
<tr>
<td><strong>Diffusion estimation</strong></td>
<td>Standard likelihood</td>
<td><strong>Singular</strong>: changing $\sigma$ changes the topology of path space</td>
</tr>
</table>

### The Feldman-Hájek Dichotomy (the "$\sigma$ problem")

For two Gaussian measures $\mathcal{N}(m_1, \mathcal{C}_1)$ and $\mathcal{N}(m_2, \mathcal{C}_2)$ on a Hilbert space, they are either **equivalent** ($\ll$ each other) or **mutually singular** ($\perp$). They are equivalent if and only if:

1. $m_1 - m_2 \in \mathcal{C}_1^{1/2}(\mathcal{H})$ (mean difference in Cameron-Martin space), **and**
2. $\mathcal{C}_1^{-1/2}\mathcal{C}_2\mathcal{C}_1^{-1/2} - I$ is **Hilbert-Schmidt**

Violating either condition makes the measures singular — importance weights are either 0 or $\infty$. This is why estimating $\sigma$ from path data is fundamentally harder than estimating $\mu$, and requires the log-Normal GP prior in our formulation.

---

## 3. Our Approach

### 3.1 Key Technical Components

```
┌─────────────────────────────────────────────────────────────────┐
│  PRIOR                   LIKELIHOOD              POSTERIOR       │
│                                                                 │
│  μ ~ GP(0, K_μ)    ×    Girsanov density    →   ν^X on L²     │
│                          (Radon-Nikodym)          (well-posed)  │
│                                                                 │
│  ──────────────────────────────────────────────────────────     │
│                                                                 │
│  SAMPLER: Preconditioned Crank-Nicolson (pCN)                  │
│  • Dimension-independent acceptance rate                        │
│  • Stable as discretization δ → 0                              │
│  • Proposals in Cameron-Martin space                            │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 The Preconditioned Crank-Nicolson (pCN) Sampler

Standard Metropolis-Hastings proposes $\mu' = \mu + \sqrt{2\beta}\,\xi$, $\xi \sim \nu_0$. As $\dim \mathcal{H} \to \infty$, acceptance rate $\to 0$. The **pCN proposal** instead is:

$$\mu' = \sqrt{1 - \beta^2}\,\mu + \beta\,\xi, \quad \xi \sim \nu_0, \quad \beta \in (0,1)$$

This is **measure-preserving** with respect to the prior $\nu_0$, so the acceptance probability depends only on the likelihood ratio — it does **not** degenerate with dimension. The acceptance probability is:

$$\alpha(\mu, \mu') = \min\!\left(1,\, \exp\!\big(\Phi(\mu; X) - \Phi(\mu'; X)\big)\right)$$

### 3.3 Karhunen-Loève Discretization

For numerical implementation, we project onto the leading $J$ eigenfunctions of $\mathcal{C}$:

$$\mu^{(J)} = \sum_{j=1}^J \xi_j\,\sqrt{\lambda_j}\,e_j, \quad \xi_j \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0,1)$$

The approximation error satisfies $\|\mu - \mu^{(J)}\|^2_{L^2} = \sum_{j>J}\lambda_j \to 0$. For Matérn-$\nu$ kernels with eigenvalue decay $\lambda_j \sim j^{-2\nu-1}$, the Hellinger distance between the true and projected posteriors decays as $O(J^{-\nu})$.

---

## 4. Core Research Results

### Theorem A — Posterior Well-Posedness

> **Theorem** (Stuart 2010, adapted). *Under Novikov's condition and Assumption 2.1 (trace-class $\mathcal{C}$), the posterior $\nu^X$ is a well-defined probability measure on $(\mathcal{H}, \mathcal{B}(\mathcal{H}))$, absolutely continuous with respect to $\nu_0$, with $Z(X) \in (0, \infty)$ $\mathbb{P}$-almost surely.*

### Theorem B — Posterior Contraction Rate

> **Theorem** (van der Vaart & van Zanten 2008, applied). *Let $\mu^* \in H^s([0,T])$ be the ground truth and let the prior have a Matérn-$\nu$ kernel with $\nu > s$. As the observation window $T \to \infty$, the posterior concentrates at the minimax-optimal rate:*
>
> $$\nu^{X_{0:T}}\!\left(\mu : \|\mu - \mu^*\|_{L^2} > \epsilon_T\right) \xrightarrow{\mathbb{P}} 0, \quad \epsilon_T = T^{-s/(2s+1)}$$

### Theorem C — Discretization Stability

> **Theorem** (proposed). *Let $\nu^X_\delta$ be the posterior under Euler-Maruyama discretization with mesh size $\delta$. Then:*
>
> $$d_{\mathrm{He}}\!\left(\nu^X, \nu^X_\delta\right) \leq C\,\sqrt{\delta}$$
>
> *where $C$ depends on $\|\mu^*\|_{H^1}$ and $\sigma_{\min} = \inf_x \sigma(x) > 0$.*

### Theorem D — pCN Spectral Gap

> **Theorem** (Cotter et al. 2013). *The pCN Markov chain on $(\mathcal{H}, \nu^X)$ admits a spectral gap $\rho(\beta) > 0$ that is independent of the KL truncation order $J$. Hence the mixing time is $O(\rho^{-1})$ uniformly in dimension.*

---

## 5. Repository Architecture

```
bayesian-path-filtering/
│
├── 📂 theory/                         # Rigorous mathematical notes
│   ├── 01_posterior_existence.md      # Well-posedness: Cameron-Martin + Novikov
│   ├── 02_contraction_rates.md        # Posterior consistency, minimax rates
│   ├── 03_discretization_stability.md # Euler-Maruyama Hellinger bound
│   └── 04_sigma_problem.md            # Feldman-Hájek & the diffusion obstruction
│
├── 📂 core/                           # Production-grade implementations
│   ├── operators.py                   # Covariance operators, RKHS projections, KL expansion
│   ├── girsanov.py                    # Path-integral likelihood via Itô-Stratonovich
│   └── samplers/
│       ├── pCN.py                     # Preconditioned Crank-Nicolson (dimension-independent)
│       ├── mala.py                    # Manifold-adjusted Langevin (for comparison)
│       └── smc.py                     # Sequential Monte Carlo on function space
│
├── 📂 simulations/                    # Reproducible experiments
│   ├── gp_prior.py                    # GP prior sampling on [0,T]
│   ├── girsanov_likelihood.py         # Discrete Girsanov density computation
│   ├── posterior_drift.py             # Full posterior for drift-only case
│   ├── fBM_paths.py                   # Fractional Brownian Motion path generators
│   ├── posterior_contraction.py       # Convergence rate verification vs. T
│   └── discretization_error.py        # Hellinger distance vs. mesh size δ
│
├── 📂 notebooks/                      # Interactive research
│   ├── 01_gp_priors_on_path_space.ipynb
│   ├── 02_girsanov_likelihood.ipynb
│   └── 03_posterior_contraction.ipynb
│
├── 📂 notes/
│   ├── literature_map.md              # How the key papers connect
│   ├── open_problems.md               # Conjectures and unresolved questions
│   └── proof_sketches.md              # Informal arguments and intuitions
│
├── 📂 tests/
│   ├── test_gp_prior.py
│   ├── test_girsanov.py
│   └── test_discretization.py
│
├── references/bibliography.bib
├── requirements.txt
└── README.md
```

---

## 6. Quickstart

### Prerequisites

```bash
Python >= 3.10
torch >= 2.0     # or jax>=0.4.1 for functional autodiff
gpytorch         # GP covariance operators
numpy, scipy
matplotlib
```

### Installation

```bash
git clone https://github.com/prakulhiremath/Bayesian-Filtering-on-Infinite-Dimensional-Path-Spaces.git
cd Bayesian-Filtering-on-Infinite-Dimensional-Path-Spaces
pip install -r requirements.txt
```

### Run the pCN Sampler (Drift Posterior)

```bash
# Infer μ(·) from a single observed path via pCN MCMC
python core/samplers/pCN.py \
  --kernel matern52 \
  --lengthscale 0.2 \
  --obs_noise 0.01 \
  --kl_modes 50 \
  --n_samples 5000
```

### Run All Simulations

```bash
python simulations/posterior_drift.py       # GP posterior for μ, fixed σ
python simulations/posterior_contraction.py # Verify T^{-s/(2s+1)} rate
python simulations/discretization_error.py  # Verify O(√δ) Hellinger bound
pytest tests/                               # Full test suite
```

### Interactive Notebooks

```bash
jupyter lab notebooks/
# Start with: 01_gp_priors_on_path_space.ipynb
```

---

## 7. Theoretical Roadmap

| Milestone | Status | Notes |
|---|---|---|
| Literature synthesis: GP nonparametric Bayes | ✅ Complete | See `notes/literature_map.md` |
| Posterior existence (drift-only, fixed σ) | ✅ Complete | See `theory/01_posterior_existence.md` |
| pCN sampler implementation & dimension-independence test | ✅ Complete | See `core/samplers/pCN.py` |
| Posterior contraction rate: drift, continuous obs. | 🔄 In Progress | van der Vaart & van Zanten framework |
| Discretization stability theorem (Theorem C) | 🔄 In Progress | Euler-Maruyama Hellinger bound |
| Joint (μ, σ) posterior: Feldman-Hájek analysis | 🔲 Planned | Log-Normal GP prior for σ |
| fSDE extension: Fractional Brownian Motion | 🔲 Planned | Hurst exponent H ≠ 0.5 |
| Lévy jump-diffusion: Itô-Lévy isometry in ℋ | 🔲 Planned | Companion to arXiv preprint |
| arXiv submission (stat.ML + math.PR + q-fin.ST) | 🔲 Planned | Draft: `docs/preprint.pdf` |

---

## 8. Key References

> Ordered by conceptual dependency — read in this order for maximum clarity.

**Foundations — Measure Theory in Function Space**

- **Bogachev, V.I. (1998).** *Gaussian Measures.* AMS. — Why Lebesgue measure fails in $\infty$-dim.
- **Da Prato, G. & Zabczyk, J. (2014).** *Stochastic Equations in Infinite Dimensions* (2nd ed.). CUP. — SPDEs, trace-class operators, Gaussian measures on Hilbert spaces.
- **Liptser, R.S. & Shiryaev, A.N. (2001).** *Statistics of Random Processes.* Springer. — Girsanov theorem, Novikov's condition, path-space likelihoods.

**Bayesian Inference in Hilbert Spaces**

- **Stuart, A.M. (2010).** Inverse problems: A Bayesian perspective. *Acta Numerica*, 19, 451–559. — **The** foundational reference for this project. Abstract Bayes, well-posedness, MAP estimation.
- **van der Vaart, A. & van Zanten, H. (2008).** Rates of contraction of posterior distributions based on Gaussian process priors. *Annals of Statistics*, 36(3), 1435–1463. — Minimax posterior contraction rates.
- **van der Vaart, A. & van Zanten, H. (2011).** Information rates of nonparametric Gaussian process methods. *JMLR*, 12, 2095–2119. — Extension to continuous-time observations.

**Dimension-Independent MCMC**

- **Cotter, S.L., Roberts, G.O., Stuart, A.M., & White, D. (2013).** MCMC methods for functions: Modifying old algorithms to make them faster. *Statistical Science*, 28(3), 424–446. — pCN, MALA, HMC on function space. Basis for `core/samplers/`.

**Fractional and Jump Dynamics**

- **Mishura, Y. (2008).** *Stochastic Calculus for Fractional Brownian Motion and Related Processes.* Springer. — fSDE theory for $H \neq \tfrac{1}{2}$.
- **Applebaum, D. (2009).** *Lévy Processes and Stochastic Calculus* (2nd ed.). CUP. — Lévy-Itô decomposition, compensated Poisson measures.
- **Peszat, S. & Zabczyk, J. (2007).** *Stochastic Partial Differential Equations with Lévy Noise.* CUP. — Infinite-dimensional Lévy processes.

---

## 9. Citation

If this repository contributes to your research, please cite:

@article{hiremath2026pard,
  title   = {PARD-SSM: Probabilistic Cyber-Attack Regime Detection via
             Variational Switching State-Space Models},
  author  = {Hiremath, Prakul Sunil and Bhekane, Sahil and Bagawan, PeerAhammad M},
  journal = {arXiv preprint arXiv:2604.02299},
  year    = {2026}
}
```

---

<div align="center">

**[Aliens on Earth (AoE) Research Group](https://aliensonearth.in)** &nbsp;·&nbsp; **[VTU Belagavi](https://vtu.ac.in)**

[prakulhiremath03@gmail.com](mailto:prakulhiremath03@gmail.com)

<br/>

*"The curse of dimensionality is not a problem to be optimized around. It is a structure to be understood."*

</div>
