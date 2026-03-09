# 4. The σ Problem: Why Diffusion Estimation is Harder

## The Fundamental Asymmetry

Drift and diffusion play **structurally different roles** in an SDE. This asymmetry is not a technical inconvenience — it reflects deep information-theoretic differences between the two objects.

| Property | Drift $\mu_t$ | Diffusion $\sigma_t$ |
|----------|--------------|---------------------|
| Identified from | Time averages of $dX_t$ | Quadratic variation of $X$ |
| Obs. at fixed $T$ | Not identified without $\sigma$ | Identified (QV is $O(1)$) |
| As $T \to \infty$ | Identified (consistent estimation) | Already identified at $T=0^+$ |
| Fisher information | $\int_0^T \mu_t^2 / \sigma_t^2 \, dt$ | $\int_0^T 2/\sigma_t^2 \, dt$ |
| Under discrete obs. | Degrades with mesh $\delta$ | Robust to $\delta$ (via realized vol) |

---

## Quadratic Variation and $\sigma$-Identification

The **quadratic variation** of $X$ satisfies:

$$[X]_T = \int_0^T \sigma_t^2 \, dt$$

Under continuous observation, $[X]_T$ is **exactly observable** (it is a pathwise quantity, not a statistical estimate). This means $\int_0^T \sigma_t^2 \, dt$ is known exactly from a single path.

However, this only identifies $\sigma^2$ through its integral — not pointwise. Full pointwise identification requires:

$$[X]_t = \int_0^t \sigma_s^2 \, ds \quad \text{known for all } t \in [0,T]$$

which is also the case under continuous observation (the quadratic variation process is observable). Thus **$\sigma$ is nonparametrically identified** under continuous observation.

Under discrete observation with mesh $\delta$, realized volatility identifies $\sigma^2$ at rate $O(\delta^{1/2})$, which is **independent of $T$** — this is the basis of high-frequency econometrics.

---

## Non-Identifiability at Fixed $T$

A subtle point: at **fixed $T$**, the drift $\mu$ is **not identified** from a single continuous path.

To see why: the mutual information between $X_{0:T}$ and $\mu$ under $\mathbb{P}^\mu$ is:

$$I(\mu; X_{0:T}) = \frac{T}{2} \|\mu\|_{L^2}^2$$

This grows linearly in $T$. At fixed $T$, there is a fundamental limit on how precisely $\mu$ can be estimated — the **Cramér-Rao lower bound** on $L^2$ estimation error is $O(T^{-1})$ in the parametric case.

In contrast, the mutual information between $X_{0:T}$ and $\sigma$ is:

$$I(\sigma; X_{0:T}) \to \infty \quad \text{as } [X]_t \text{ is observable}$$

$\sigma$ is identified "instantaneously" because quadratic variation accumulates at rate $\sigma_t^2 dt$ — every infinitesimal time interval carries $O(dt)$ information about $\sigma_t$.

---

## The Joint Posterior: Non-Product Structure

A key structural result is that the joint posterior $p(\mu, \sigma \mid X_{0:T})$ is **not a product measure**, even when $\mu \perp \sigma$ under the prior.

The joint Girsanov density:

$$\frac{d\mathbb{P}^{\mu,\sigma}}{d\mathbb{P}^0} = \prod_{t} \frac{1}{\sigma_t} \cdot \exp\!\left(\int_0^T \frac{\mu_t}{\sigma_t^2} \, dX_t - \frac{1}{2}\int_0^T \frac{\mu_t^2}{\sigma_t^2} \, dt\right)$$

The terms $\mu_t / \sigma_t^2$ and $\mu_t^2 / \sigma_t^2$ make $\mu$ and $\sigma$ **posterior-dependent** — knowing $\sigma$ better sharpens the posterior on $\mu$, and vice versa.

This coupling creates challenges for:
- **MCMC:** Gibbs sampling between $\mu$ and $\sigma$ mixes slowly when they are strongly correlated
- **Variational inference:** Mean-field approximations miss the coupling, underestimating posterior uncertainty
- **Contraction theory:** Standard proofs that factor through marginals break down

---

## GP Prior Specification for $\sigma$

### Problem with Standard GP Priors

A naive $\sigma \sim \mathcal{GP}(m, k_\sigma)$ is problematic because:

1. GP samples are continuous but **not bounded away from zero** — the prior assigns positive mass to $\{\sigma_t = 0\}$ if $m_\sigma = 0$
2. The Girsanov density involves $\sigma_t^{-1}$ and $\sigma_t^{-2}$, which blow up near zero
3. This makes the normalizing constant potentially infinite: $\mathbb{E}[\sigma_t^{-2}]$ may not exist for a GP with $m_\sigma = 0$

### Log-Normal GP Prior

The standard remedy is to place the GP prior on $\log \sigma$:

$$\log \sigma \sim \mathcal{GP}(m_\sigma, k_\sigma)$$

so that $\sigma_t = e^{f_t}$ with $f \sim \mathcal{GP}$. Then:

- $\sigma_t > 0$ almost surely
- $\mathbb{E}[\sigma_t^{-p}] = e^{-pm_\sigma + p^2 k_\sigma(t,t)/2} < \infty$ for all $p > 0$
- The RKHS of the log-normal process is well-characterized

### Posterior for Log-Normal Prior

Under the log-normal prior $f = \log \sigma \sim \mathcal{GP}(m_f, k_f)$, the posterior on $f$ is:

$$\frac{d\Pi(f \mid X_{0:T})}{d\Pi_0(f)} \propto \exp\!\left(-\int_0^T f_t \, dt + \int_0^T e^{-f_t}\mu_t \, dX_t - \frac{1}{2}\int_0^T e^{-2f_t}\mu_t^2 \, dt\right)$$

This is **not Gaussian** in $f$, even with a GP prior. The posterior must be approximated via:
- Laplace approximation in function space (linearize around the posterior mode)
- MCMC on the function space (pCN proposals for $f$)
- Variational Bayes with structured families

---

## Realized Volatility as a Sufficient Statistic

Under discrete observation with mesh $\delta$, the **realized volatility**:

$$RV_\delta = \sum_{i=0}^{n-1} (X_{t_{i+1}} - X_{t_i})^2$$

satisfies (by the theory of quadratic variation):

$$RV_\delta \xrightarrow{\mathbb{P}} \int_0^T \sigma_t^2 \, dt \quad \text{as } \delta \to 0$$

More precisely, for each sub-interval $[s, s+h]$:

$$\sum_{t_i \in [s, s+h]} (X_{t_{i+1}} - X_{t_i})^2 \xrightarrow{\mathbb{P}} \int_s^{s+h} \sigma_t^2 \, dt$$

This suggests a **two-stage estimation strategy**:
1. Estimate $\sigma^2$ non-parametrically from realized volatility (kernel smoothing of squared increments)
2. Condition on $\hat{\sigma}^2$ and estimate $\mu$ via the GP posterior

The efficiency loss from two-stage vs. joint estimation is an open question.

---

## Local Time and Non-Semimartingale Extensions

For processes with **irregular diffusion** (e.g., $\sigma_t = \sigma(X_t)$ depends on the state), identification of $\sigma$ uses **local time** $L^x_T$ via the occupation times formula:

$$\int_0^T h(X_t) \sigma_t^2 \, dt = \int_{-\infty}^\infty h(x) L^x_T \, dx$$

The local time $L^x_T$ is observable from the path (as the density of the occupation measure), giving a nonparametric estimator of $x \mapsto \sigma^2(x)$.

This approach — using local time instead of quadratic variation for state-dependent $\sigma$ — was developed by Florens-Zmirou (1993) and extends to the Bayesian setting via GP priors on $\sigma(\cdot)$.

---

## Open Problems Specific to $\sigma$

1. **Posterior existence for joint $(\mu, \sigma)$:** Prove the analog of Theorem 1.1 under a log-normal GP prior on $\sigma$.

2. **Contraction rates for $\sigma$:** The quadratic-variation identification means $\sigma$ should be estimable at rate $T^{-1/2}$ in the parametric case. What is the nonparametric rate for $\sigma^* \in H^s$?

3. **Semiparametric efficiency:** Is the two-stage estimator (realized vol then drift posterior) semiparametrically efficient, or does the joint posterior do better?

4. **Leverage effect:** In finance, $\mu_t$ and $\sigma_t$ are negatively correlated (leverage effect). Can a joint GP prior capture this through a **cross-covariance kernel** $k_{\mu\sigma}(s,t) = \text{Cov}(\mu_s, \sigma_t)$?

---

## References

- Florens-Zmirou, D. (1993). On estimating the diffusion coefficient from discrete observations. *J. Appl. Prob.* 30(4).
- Barndorff-Nielsen, O.E. & Shephard, N. (2002). Econometric analysis of realized volatility. *J. Royal Stat. Soc. B* 64(2).
- Jacod, J. & Protter, P. (2012). *Discretization of Processes.* Springer.
- Bibby, B.M. & Sørensen, M. (1995). Martingale estimation functions for discretely observed diffusion processes. *Bernoulli* 1(1-2).
- Nickl, R. (2023). *Bayesian Non-Linear Statistical Inverse Problems.* Zürich Lectures in Advanced Mathematics.
