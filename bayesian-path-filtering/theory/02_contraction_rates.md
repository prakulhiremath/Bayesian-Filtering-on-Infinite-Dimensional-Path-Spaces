# 2. Posterior Contraction Rates

## Setup

We work in the **drift-only** setting ($\sigma \equiv 1$) under continuous observation of $X_{0:T}$.

Let:
- $\mu^* \in L^2([0,T])$ be the **true drift function**
- $\Pi_0 = \mathcal{GP}(0, k_\mu)$ be the GP prior
- $\Pi_n(\cdot \mid X_{0:T})$ denote the posterior after observing $n$ discrete (or continuous) observations

**Goal:** Show that as $n \to \infty$ (or $T \to \infty$), the posterior $\Pi_n$ contracts around $\mu^*$.

Formally, we want to find a rate $\varepsilon_n \to 0$ such that:

$$\Pi_n\!\left(\mu : \|\mu - \mu^*\|_{L^2} > M\varepsilon_n \mid X_{0:T}\right) \to 0 \quad \text{in } \mathbb{P}^{\mu^*}\text{-probability}$$

for some large constant $M$.

---

## Background: The van der Vaart–van Zanten Framework

For **nonparametric regression** $Y_i = f(x_i) + \varepsilon_i$, van der Vaart & van Zanten (2008) showed:

If the GP prior has **Matérn-$\nu$ kernel** and $f^* \in C^\beta([0,1])$ (Hölder smoothness $\beta$):

$$\varepsilon_n = n^{-\beta/(2\beta+1)} (\log n)^{1/2}$$

is the posterior contraction rate, which is **minimax optimal** (up to log factors).

The proof requires two conditions:
1. **KL support:** $\Pi_0(\mu : \text{KL}(\mathbb{P}^{\mu^*}, \mathbb{P}^\mu) \leq \varepsilon_n^2) \geq e^{-cn\varepsilon_n^2}$
2. **Test existence:** There exist exponentially consistent tests distinguishing $\mu^*$ from $\{\|\mu - \mu^*\| > 2M\varepsilon_n\}$

---

## Key Difference: SDE vs. Regression

In regression, the observations $Y_i = f(x_i) + \varepsilon_i$ are **direct** (modulo noise). In the SDE setting, we observe $X_t$ which is an **integral functional** of $\mu$:

$$X_t = x_0 + \int_0^t \mu_s \, ds + W_t$$

This introduces **indirect observation** analogous to inverse problems. The effective "signal" in $X_{0:T}$ about $\mu$ is:

$$\int_0^T \mu_t \, dX_t = \int_0^T \mu_t^2 \, dt + \int_0^T \mu_t \, dW_t$$

The Fisher information for $\mu$ under continuous observation on $[0,T]$ is the operator $I(\mu^*) = T \cdot \text{Id}$ (in the iid sense), meaning the problem becomes **well-posed as $T \to \infty$**.

---

## KL Divergence Between Path Measures

The KL divergence between $\mathbb{P}^{\mu^*}$ and $\mathbb{P}^\mu$ (both on $C([0,T])$) is:

$$\text{KL}(\mathbb{P}^{\mu^*} \| \mathbb{P}^\mu) = \frac{1}{2}\int_0^T (\mu^*_t - \mu_t)^2 \, dt = \frac{1}{2}\|\mu^* - \mu\|_{L^2}^2$$

This follows directly from the Girsanov formula. Crucially, **the KL divergence is just the $L^2$ squared distance**, which simplifies the analysis considerably compared to general inverse problems.

---

## Main Theorem (Continuous Observation)

**Theorem 2.1** *(Posterior contraction, continuous observation)*

Let $\mu^* \in H^s([0,1])$ (Sobolev smoothness $s > 0$) and let the prior be $\Pi_0 = \mathcal{GP}(0, k_\mu)$ where $k_\mu$ is the Matérn-$\nu$ kernel with $\nu > s$.

Under continuous observation on $[0,T]$ as $T \to \infty$, the posterior satisfies:

$$\Pi_T\!\left(\mu : \|\mu - \mu^*\|_{L^2} > M T^{-s/(2s+1)} (\log T)^{1/2} \mid X_{0:T}\right) \to 0$$

in $\mathbb{P}^{\mu^*}$-probability, for any $M > 0$ sufficiently large.

**Remark:** The rate $T^{-s/(2s+1)}$ matches the frequentist minimax rate for nonparametric drift estimation (see Korostelev, 1993).

---

## Proof Sketch

**Step 1: KL support condition.**

We need to show:
$$\Pi_0\!\left(\mu : \|\mu - \mu^*\|_{L^2}^2 \leq \varepsilon_T^2\right) \geq e^{-cT\varepsilon_T^2}$$

Since $\text{KL}(\mathbb{P}^{\mu^*} \| \mathbb{P}^\mu) = \frac{1}{2}\|\mu^* - \mu\|_{L^2}^2$, this reduces to a **small-ball probability** for the GP prior centered at $\mu^*$.

For GP priors, small-ball probabilities are controlled by the RKHS norm of $\mu^*$:
$$-\log \Pi_0(\|\mu - \mu^*\|_{L^2} \leq \varepsilon) \asymp \varepsilon^{-1/s}$$

when $\mu^* \in \mathcal{H}_{k_\mu}$ and $k_\mu$ is Matérn-$\nu$ with $\nu > s$.

**Step 2: Test existence.**

We need exponentially consistent tests $\phi_T$ for:
$$H_0: \mu = \mu^* \quad \text{vs.} \quad H_1: \|\mu - \mu^*\|_{L^2} > 2M\varepsilon_T$$

The **likelihood ratio test** works here. By Girsanov:
$$\log \frac{d\mathbb{P}^\mu}{d\mathbb{P}^{\mu^*}} = \int_0^T (\mu_t - \mu^*_t) \, dX_t - \frac{1}{2}\int_0^T (\mu_t^2 - {\mu^*_t}^2) \, dt$$

Under $\mathbb{P}^{\mu^*}$, the log-likelihood ratio is Gaussian with mean $-\frac{T}{2}\|\mu - \mu^*\|_{L^2}^2$ and variance $T\|\mu - \mu^*\|_{L^2}^2$. The likelihood ratio test $\phi_T = \mathbf{1}[\text{LR} > 0]$ satisfies:

$$\mathbb{P}^{\mu^*}[\phi_T = 1] \leq e^{-T\|\mu - \mu^*\|_{L^2}^2/8}$$
$$\mathbb{P}^\mu[\phi_T = 0] \leq e^{-T\|\mu - \mu^*\|_{L^2}^2/8}$$

**Step 3: Apply the general theorem.**

Combining Steps 1–2 via Theorem 2.1 in van der Vaart & van Zanten (2008) (generalized to path-space likelihoods) yields the contraction rate. $\square$

---

## Discrete Observation: Rate Degradation

Under **discrete** observations at times $0 = t_0 < t_1 < \cdots < t_n = T$ with mesh $\delta$:

The effective information about $\mu$ is reduced. The Euler-Maruyama approximation gives:

$$X_{t_{i+1}} - X_{t_i} \approx \mu_{t_i} \delta + \sqrt{\delta} \, Z_i, \quad Z_i \sim \mathcal{N}(0,1)$$

This is a **Gaussian sequence model** with $n$ observations and noise level $\delta^{1/2}$. The posterior contraction rate becomes:

$$\varepsilon_n \asymp n^{-s/(2s+1)}$$

which matches the nonparametric regression rate — **slower than continuous observation** when $T$ is fixed and $n \to \infty$.

---

## Adaptation

A natural question: what if $s$ (the smoothness of $\mu^*$) is unknown?

Standard results (Ghosal & van der Vaart, 2007) show that **hierarchical GP priors** with random length-scale $\ell \sim \Pi_\ell$ achieve adaptive rates:

$$\mu \mid \ell \sim \mathcal{GP}(0, k_\mu(\cdot/\ell)), \quad \ell \sim \text{Gamma}(a, b)$$

The posterior over $\ell$ concentrates on the correct scale, giving near-optimal rates simultaneously over all Sobolev classes $H^s$.

---

## Open Problems

1. **Exact constant:** The theorem gives the rate up to constants. What is the exact Pinsker constant for the GP posterior?

2. **Non-Gaussian likelihoods:** Extension to jump-diffusion processes $dX_t = \mu_t dt + \sigma_t dW_t + dJ_t$ where $J_t$ is a Lévy process.

3. **Nonstationary priors:** Can spatially-varying GP priors (e.g., deep kernel learning) achieve better rates when $\mu^*$ has heterogeneous smoothness?

---

## References

- van der Vaart, A. & van Zanten, H. (2008). Rates of contraction of posterior distributions. *Ann. Stat.* 36(3).
- van der Vaart, A. & van Zanten, H. (2011). Information rates of nonparametric GP methods. *JMLR* 12.
- Ghosal, S. & van der Vaart, A. (2007). Posterior convergence rates. *Ann. Stat.* 35(1).
- Korostelev, A.P. (1993). An asymptotically minimax regression estimator in the uniform norm up to exact constant. *Theory Prob. Appl.* 38(4).
- Nickl, R. & Söhl, J. (2017). Nonparametric Bayesian posterior contraction rates for discretely observed scalar diffusions. *Ann. Stat.* 45(4), 1664–1693.
