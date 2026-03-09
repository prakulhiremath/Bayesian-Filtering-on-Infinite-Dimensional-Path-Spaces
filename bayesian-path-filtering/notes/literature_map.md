# Literature Map

A structured guide to the key papers and how they connect to this project.

---

## Core Framework Papers

### Stuart (2010) — The Foundational Reference
**"Inverse problems: A Bayesian perspective"** — *Acta Numerica* 19, 451–559

**What it does:** Establishes the general theory of Bayesian inference in infinite-dimensional spaces (Banach spaces). Shows that the posterior is well-defined when the likelihood is continuous and the prior is Gaussian. Introduces the concept of well-posedness for Bayesian inverse problems.

**What we borrow:** The abstract framework (prior measure → likelihood → posterior measure via Radon-Nikodym). Specifically §3 (Gaussian priors) and §4 (Bayesian formulation).

**Key gap for our problem:** Stuart assumes a **deterministic forward map** $G: X \mapsto y + \eta$. Our SDE forward map is stochastic (the path $X_{0:T}$ is itself random given $\mu$). The Girsanov density plays the role of Stuart's likelihood, but it depends on the observed data in a more complex way.

---

### van der Vaart & van Zanten (2008) — Contraction Rates for GPs
**"Rates of contraction of posterior distributions based on Gaussian process priors"** — *Ann. Stat.* 36(3), 1435–1463

**What it does:** Derives minimax-optimal posterior contraction rates for GP priors in the **nonparametric regression** setting. Key theorem: if the true function lies in a Sobolev ball $H^s$ and the GP prior has Matérn-$\nu$ kernel with $\nu > s$, contraction occurs at rate $n^{-s/(2s+1)}$.

**What we borrow:** The proof technique (KL support + test existence). The specific contraction rates as benchmarks.

**Key gap for our problem:** Regression has fixed design points and direct observation. We have **indirect observation** through the SDE path integral, which introduces ill-posedness. The KL divergence calculation in §4 must be replaced with Girsanov-based KL on path space.

---

### van der Vaart & van Zanten (2011) — RKHS and Information Rates
**"Information rates of nonparametric Gaussian process methods"** — *JMLR* 12, 2095–2119

**What it does:** Characterizes GP posterior behavior in terms of the **reproducing kernel Hilbert space (RKHS)**. Shows that priors whose RKHS matches the smoothness of the truth achieve optimal rates.

**What we borrow:** The RKHS machinery for controlling small-ball probabilities and posterior concentration.

---

### Cotter, Roberts, Stuart & White (2013) — Infinite-Dimensional MCMC
**"MCMC methods for functions: modifying old algorithms to make them faster"** — *Stat. Sci.* 28(3), 424–446

**What it does:** Shows that standard MCMC (random walk Metropolis) degenerates as the dimension of the discretization grows. Introduces **preconditioned Crank-Nicolson (pCN)** proposals that are dimension-robust.

**What we borrow:** The pCN algorithm for sampling the GP posterior on path space. The analysis of acceptance rates as a function of mesh size $\delta$.

**Relevance:** This is the computational backbone for our simulation studies.

---

## SDE-Specific References

### Nickl & Söhl (2017) — Bayesian Diffusion Estimation
**"Nonparametric Bayesian posterior contraction rates for discretely observed scalar diffusions"** — *Ann. Stat.* 45(4), 1664–1693

**What it does:** Proves posterior contraction rates for the **drift function** of a scalar diffusion observed at discrete times. Uses wavelets instead of GP priors, but the contraction rate analysis applies.

**Relevance:** The closest existing paper to our setting. **Key reference for situating our work.** The GP prior version of their main theorem is essentially what Theorem 2.1 in `02_contraction_rates.md` aims to prove.

---

### Liptser & Shiryaev (2001) — Statistics of Random Processes
*Statistics of Random Processes I & II.* Springer.

**What it does:** The definitive reference for likelihood-based inference for stochastic processes. Chapter 7 covers the Girsanov formula and its consequences for drift estimation. Chapter 17 covers nonparametric estimation for diffusions.

**What we borrow:** The Girsanov density formula, conditions for it to be a true martingale (Novikov), and the Cramér-Rao lower bounds for drift estimation.

---

### Florens-Zmirou (1993) — Diffusion Coefficient Estimation
**"On estimating the diffusion coefficient from discrete observations"** — *J. Appl. Prob.* 30(4), 790–804

**What it does:** Proposes nonparametric kernel estimators for the diffusion coefficient $\sigma^2(x)$ (state-dependent case) using discrete observations and occupation time arguments.

**Relevance:** Relevant to the $\sigma$ problem (§4 in theory). The local time approach is the classical alternative to quadratic variation methods.

---

### Barndorff-Nielsen & Shephard (2002) — Realized Volatility
**"Econometric analysis of realized volatility and its use in estimating stochastic volatility models"** — *J. Royal Stat. Soc. B* 64(2), 253–280

**What it does:** Establishes the theory of **realized volatility** as a consistent nonparametric estimator of integrated volatility under discrete observation.

**Relevance:** Justifies the two-stage estimation strategy (estimate $\sigma$ via realized vol, then estimate $\mu$ via GP posterior).

---

## Functional Analysis & Measure Theory

### Da Prato & Zabczyk (1992)
*Stochastic Equations in Infinite Dimensions.* Cambridge University Press.

**What it does:** The foundational reference for SPDEs and infinite-dimensional stochastic analysis. Chapter 1 covers Gaussian measures on Hilbert spaces, Chapter 4 covers the Cameron-Martin theorem.

**What we borrow:** Cameron-Martin space theory, trace-class operators, Radon-Nikodym derivatives on function space.

---

### Bogachev (1998)
*Gaussian Measures.* American Mathematical Society.

**What it does:** Comprehensive treatment of Gaussian measures on infinite-dimensional spaces.

**What we borrow:** Small-ball probabilities (Ch. 4), the Cameron-Martin theorem (Ch. 2), absolute continuity and singularity of Gaussian measures (Ch. 3).

---

### Hairer, Stuart & Vollmer (2014)
**"Spectral gaps for a Metropolis-Hastings algorithm in infinite dimensions"** — *Ann. Appl. Probab.* 24(6), 2455–2490

**What it does:** Proves that pCN MCMC has a spectral gap bounded away from zero uniformly in the dimension of the discretization. This is the key result making infinite-dimensional MCMC feasible.

**Relevance:** Underpins the computational feasibility of our simulation approach.

---

## Connections Diagram

```
Bayesian Inverse Problems          Nonparametric Bayes
      (Stuart 2010)           (van der Vaart & van Zanten 08,11)
           |                              |
           |    THIS PROJECT              |
           +-----------> Bayesian GP <---+
                         filtering for
                         SDE path space
                              |
              +---------------+---------------+
              |               |               |
    Posterior          Contraction        Computational
    Existence          Rates              Algorithms
  (Da Prato,         (Nickl & Söhl       (Cotter et al.
   Bogachev)          2017 + GP ext.)     2013, pCN)
              |               |               |
    σ-Problem           Discretization    Simulation
 (Liptser,             Stability         Studies
  Florens-Zmirou,     (Kloeden &
  BN-Shephard)         Platen)
```

---

## What Has Not Been Done

| Result | Status |
|--------|--------|
| Posterior existence (fixed σ, GP prior) | Essentially follows from Stuart + Novikov. Write-up needed. |
| Posterior existence (joint μ, σ) | Open. Log-normal prior required. |
| Contraction rates (GP prior, continuous obs.) | Open. Main theorem of this project. |
| Contraction rates (GP prior, discrete obs.) | Open. Follows from continuous case + discretization stability. |
| Discretization stability (Hellinger bound) | Open. Proof sketch in `03_discretization_stability.md`. |
| Adaptation (unknown smoothness s) | Likely follows from hierarchical GP results. Open. |
| Semiparametric efficiency for σ | Open. Requires Le Cam theory in path space. |
