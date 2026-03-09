# 1. Posterior Existence in Function Space

## Problem Statement

Let $(\Omega, \mathcal{F}, \mathbb{P})$ be a probability space. Consider the SDE

$$dX_t = \mu_t \, dt + \sigma_t \, dW_t, \quad X_0 = x_0, \quad t \in [0,T]$$

where:
- $W_t$ is a standard Brownian motion
- $\mu \in L^2([0,T])$ is the unknown drift function
- $\sigma \in L^2([0,T])$, $\sigma_t > 0$ a.e., is the unknown diffusion

We place Gaussian process priors:

$$\mu \sim \mathcal{GP}(0, k_\mu), \quad \sigma \sim \mathcal{GP}(0, k_\sigma)$$

**Question:** Is the posterior $p(\mu, \sigma \mid X_{0:T})$ a well-defined probability measure on $L^2([0,T]) \times L^2([0,T])$?

---

## Why the Standard Argument Fails

In finite-dimensional Bayesian inference, posterior existence is trivial: if the prior has a density and the likelihood is measurable and integrable, Bayes' theorem gives a valid posterior density.

In infinite dimensions, **there is no Lebesgue measure** on $L^2([0,T])$. We cannot write:

$$p(\mu \mid X_{0:T}) \propto p(X_{0:T} \mid \mu) \cdot p(\mu)$$

as densities against a common reference measure, because no such reference exists.

Instead, we must work with **measures directly** and ask when the posterior measure is absolutely continuous with respect to the prior measure.

---

## The Girsanov Framework

Let $\mathbb{P}^0$ denote Wiener measure (the law of $X$ under $\mu \equiv 0$, $\sigma \equiv 1$). Under the true drift $\mu$, the path measure $\mathbb{P}^\mu$ satisfies:

$$\frac{d\mathbb{P}^\mu}{d\mathbb{P}^0}\bigg|_{\mathcal{F}_T} = \exp\!\left(\int_0^T \mu_t \, dW_t - \frac{1}{2}\int_0^T \mu_t^2 \, dt\right)$$

When $\sigma \not\equiv 1$, the full Girsanov density becomes:

$$\frac{d\mathbb{P}^{\mu,\sigma}}{d\mathbb{P}^0} = \frac{1}{\sigma_t} \exp\!\left(\int_0^T \frac{\mu_t}{\sigma_t} \, dX_t - \frac{1}{2}\int_0^T \frac{\mu_t^2}{\sigma_t^2} \, dt\right)$$

The **posterior measure** $\Pi(\cdot \mid X_{0:T})$ is then formally defined via:

$$\frac{d\Pi(\mu \mid X_{0:T})}{d\Pi_0(\mu)} \propto \frac{d\mathbb{P}^\mu}{d\mathbb{P}^0}(X_{0:T})$$

where $\Pi_0$ is the GP prior measure.

---

## Conditions for Well-Posedness

### Condition 1: Novikov's Condition

For the Girsanov density to be a true martingale (and not just a local martingale), we need:

$$\mathbb{E}_{\Pi_0}\!\left[\exp\!\left(\frac{1}{2}\int_0^T \mu_t^2 \, dt\right)\right] < \infty$$

For a GP prior $\mu \sim \mathcal{GP}(0, k_\mu)$, this becomes a condition on the **trace of the covariance operator**. Specifically, if $\{(\lambda_i, e_i)\}$ are the eigenvalues/functions of $k_\mu$ on $L^2([0,T])$, then:

$$\sum_{i=1}^\infty \lambda_i = \text{tr}(k_\mu) = \int_0^T k_\mu(t,t) \, dt < \infty$$

is **necessary but not sufficient**. The full Novikov condition for GP priors requires:

$$\prod_{i=1}^\infty (1 - \lambda_i)^{-1/2} < \infty \quad \Longleftrightarrow \quad \sum_{i=1}^\infty \lambda_i < \infty$$

*This holds for Matérn and squared-exponential kernels on bounded domains.*

### Condition 2: Measurability of the Likelihood

The map $\mu \mapsto \frac{d\mathbb{P}^\mu}{d\mathbb{P}^0}(X_{0:T})$ must be measurable with respect to the Borel $\sigma$-algebra on $L^2([0,T])$. This follows from:

- Itô integrals $\int_0^T \mu_t \, dX_t$ depend continuously on $\mu$ in $L^2$
- The exponential is a continuous function

### Condition 3: Integrability of the Normalizing Constant

$$Z(X_{0:T}) = \int_{L^2} \frac{d\mathbb{P}^\mu}{d\mathbb{P}^0}(X_{0:T}) \, d\Pi_0(\mu) < \infty \quad \text{a.s.}$$

Under Novikov's condition and continuity of the likelihood, this follows from Fubini's theorem and the $\mathbb{P}^0$-integrability of the Girsanov density.

---

## Main Theorem (Drift-Only Case)

**Theorem 1.1** *(Posterior existence, fixed σ)*

Let $\sigma \equiv 1$ and let $\Pi_0 = \mathcal{GP}(0, k_\mu)$ with continuous, positive-definite kernel $k_\mu$ satisfying:

$$\int_0^T k_\mu(t,t) \, dt < \infty$$

Then for $\mathbb{P}^{\mu^*}$-almost every observed path $X_{0:T}$, the posterior measure

$$\Pi(\cdot \mid X_{0:T}) \ll \Pi_0$$

is a well-defined probability measure on $L^2([0,T])$ with Radon-Nikodym derivative:

$$\frac{d\Pi(\mu \mid X_{0:T})}{d\Pi_0(\mu)} = \frac{1}{Z} \exp\!\left(\int_0^T \mu_t \, dX_t - \frac{1}{2}\int_0^T \mu_t^2 \, dt\right)$$

*Proof sketch:* Verify Novikov → Girsanov density is an $L^1(\Pi_0)$ function of $\mu$ → apply Bayes' theorem on the abstract probability space. $\square$

---

## The Cameron-Martin Space

A central object is the **Cameron-Martin space** (RKHS) of the GP prior:

$$\mathcal{H}_{k_\mu} = \left\{ f \in L^2([0,T]) : \|f\|_{\mathcal{H}}^2 = \langle f, k_\mu^{-1} f \rangle_{L^2} < \infty \right\}$$

The posterior mean $\hat{\mu}$ lives in $\mathcal{H}_{k_\mu}$, and the posterior is a **Gaussian measure** on $L^2([0,T])$ with:

$$\hat{\mu}(\cdot) = \int_0^T k_\mu(\cdot, s) \, dX_s \cdot (I + C_\mu)^{-1}$$

where $C_\mu$ is the covariance operator of the prior. This gives an explicit, computable posterior mean.

---

## Extension to Joint (μ, σ) Case

When $\sigma$ is also unknown with GP prior $\sigma \sim \mathcal{GP}(m_\sigma, k_\sigma)$ (typically $m_\sigma > 0$ to avoid zero-crossing issues):

1. The Girsanov density involves $\sigma_t^{-1}$, so **integrability near $\sigma = 0$** is the critical issue
2. Standard GP priors assign positive mass to $\{\sigma_t < \epsilon\}$ — the $\sigma^{-1}$ term may not be integrable
3. **Remedy:** Use a **log-normal GP** prior: $\log \sigma \sim \mathcal{GP}(0, k_\sigma)$, so $\sigma_t > 0$ a.s. and $\mathbb{E}[\sigma_t^{-p}] < \infty$ for all $p > 0$

**Open problem:** Establish Theorem 1.1 for the joint case under log-normal prior on $\sigma$.

---

## References

- Girsanov, I.V. (1960). On transforming a certain class of stochastic processes by absolutely continuous substitution of measures.
- Da Prato, G. & Zabczyk, J. (1992). *Stochastic Equations in Infinite Dimensions.* §1.3, §4.1.
- Stuart, A.M. (2010). Inverse problems: A Bayesian perspective. §3.2–3.4.
- Bogachev, V.I. (1998). *Gaussian Measures.* Chapter 2 (Cameron-Martin spaces).
