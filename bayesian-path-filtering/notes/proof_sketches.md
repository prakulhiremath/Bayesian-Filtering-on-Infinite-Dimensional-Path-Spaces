# Proof Sketches and Informal Arguments

Informal write-ups to capture intuition before formalizing. These are working notes, not polished mathematics.

---

## Sketch 1: Why the Girsanov KL Simplifies

**Claim:** The KL divergence between path measures is just $L^2$ distance.

**Informal argument:**

By Girsanov, under $\mathbb{P}^{\mu^*}$ the log-likelihood ratio for $\mu$ vs $\mu^*$ is:

$$\log \frac{d\mathbb{P}^\mu}{d\mathbb{P}^{\mu^*}} = \int_0^T (\mu_t - \mu^*_t) \, dX_t - \frac{1}{2}\int_0^T (\mu_t^2 - {\mu^*_t}^2) \, dt$$

Under $\mathbb{P}^{\mu^*}$, $X_t = x_0 + \int_0^t \mu^*_s \, ds + W_t$, so:

$$\int_0^T (\mu_t - \mu^*_t) \, dX_t = \int_0^T (\mu_t - \mu^*_t)\mu^*_t \, dt + \int_0^T (\mu_t - \mu^*_t) \, dW_t$$

Taking the $\mathbb{P}^{\mu^*}$-expectation of the log-ratio:

$$\mathbb{E}^{\mu^*}\!\left[\log \frac{d\mathbb{P}^\mu}{d\mathbb{P}^{\mu^*}}\right] = \int_0^T (\mu_t - \mu^*_t)\mu^*_t \, dt - \frac{1}{2}\int_0^T (\mu_t^2 - {\mu^*_t}^2) \, dt$$

$$= \int_0^T \left[(\mu_t - \mu^*_t)\mu^*_t - \frac{1}{2}(\mu_t - \mu^*_t)(\mu_t + \mu^*_t)\right] dt$$

$$= -\frac{1}{2}\int_0^T (\mu_t - \mu^*_t)^2 \, dt = -\frac{1}{2}\|\mu - \mu^*\|_{L^2}^2$$

So: $\text{KL}(\mathbb{P}^{\mu^*} \| \mathbb{P}^\mu) = \frac{1}{2}\|\mu - \mu^*\|_{L^2}^2$.

**Why this is great:** The KL condition in the contraction rate proof reduces to a ball condition in $L^2$ — exactly what we can compute for GP priors using small-ball probability results. The SDE structure "cooperates" with the GP prior in a very clean way.

---

## Sketch 2: Small-Ball Probability for GP Priors

**What we need:** Lower bound on $\Pi_0(\|\mu - \mu^*\|_{L^2} \leq \varepsilon)$ for $\mu^* \in H^s$.

**Informal argument:**

Decompose $\mu = \mu^* + h$ where $h \sim \Pi_0$. We need:

$$\Pi_0(\|h - (\mu^* - 0)\|_{L^2} \leq \varepsilon) = \Pi_0(\|h\|_{L^2} \leq \varepsilon \text{ after centering at } \mu^*)$$

For a centered GP $h \sim \mathcal{GP}(0, k_\mu)$:

$$\Pi_0(\|h + \mu^*\|_{L^2} \leq \varepsilon) \geq \Pi_0(\|h\|_{L^2} \leq \varepsilon/2) \cdot \mathbf{1}[\|\mu^*\|_{L^2} \leq \varepsilon/2]$$

For small $\varepsilon$ (and $\|\mu^*\|_{L^2}$ fixed), this reduces to the centered small-ball probability. By Kuelbs & Li (1993):

$$-\log \Pi_0(\|h\|_{L^2} \leq \varepsilon) \asymp \phi(\varepsilon)$$

where $\phi(\varepsilon)$ is the **entropy function** of the kernel. For Matérn-$\nu$ kernel:

$$\phi(\varepsilon) \asymp \varepsilon^{-1/\nu}$$

**The crucial step:** When $\mu^* \in \mathcal{H}_{k_\mu}$ (the RKHS), the Cameron-Martin formula gives:

$$\Pi_0(\|h + \mu^*\|_{L^2} \leq \varepsilon) \geq e^{-\|\mu^*\|_\mathcal{H}^2 / 2} \cdot \Pi_0(\|h\|_{L^2} \leq \varepsilon)$$

But if $\mu^* \in H^s$ and $\nu > s$, then $\mu^* \notin \mathcal{H}_{k_\mu}$! We need to approximate $\mu^*$ by elements of $\mathcal{H}_{k_\mu}$.

**Resolution:** Choose $\mu^*_\varepsilon \in \mathcal{H}_{k_\mu}$ with $\|\mu^*_\varepsilon - \mu^*\|_{L^2} \leq \varepsilon/2$ and $\|\mu^*_\varepsilon\|_\mathcal{H} \leq C\varepsilon^{-1/s}$. Then:

$$\Pi_0(\|h + \mu^*\|_{L^2} \leq \varepsilon) \geq e^{-C\varepsilon^{-2/s}} \cdot \Pi_0(\|h\|_{L^2} \leq \varepsilon/2)$$

This gives the KL support condition with $e^{-cn\varepsilon_n^2}$ for the right choice of $\varepsilon_n$.

---

## Sketch 3: Why Standard MCMC Fails in Infinite Dimensions

**Intuition:**

Random walk Metropolis proposes $\mu_{\text{prop}} = \mu_{\text{cur}} + \sqrt{s} \, \xi$ where $\xi \sim \mathcal{N}(0, I_d)$ in $\mathbb{R}^d$.

As we refine the discretization ($d = n = T/\delta$ grows), the proposal is a small perturbation in every direction. In infinite dimensions:
- The prior measure $\Pi_0 = \mathcal{GP}(0, k_\mu)$ concentrates on functions of roughness $\approx \nu - 1/2$ (Matérn)
- A Gaussian perturbation $\xi \sim \mathcal{N}(0, I_d)$ is much rougher — it lives in $H^{-1/2}$, not $H^\nu$
- The proposal is **singular** with respect to the prior in infinite dimensions — the acceptance rate goes to zero

**The pCN fix:**

pCN proposes $\mu_{\text{prop}} = \sqrt{1 - \beta^2} \, \mu_{\text{cur}} + \beta \xi$ where $\xi \sim \Pi_0$.

Now the proposal noise $\beta\xi$ has the same covariance structure as the prior. The acceptance ratio becomes:

$$\alpha = \min\!\left(1, \frac{p(X_{0:T} \mid \mu_{\text{prop}})}{p(X_{0:T} \mid \mu_{\text{cur}})}\right) = \min\!\left(1, \exp(\ell(\mu_{\text{prop}}) - \ell(\mu_{\text{cur}}))\right)$$

This depends only on the **likelihood**, not the prior. As $d \to \infty$, the likelihood ratio stays $O(1)$ because the likelihood difference is approximately:

$$\ell(\mu_{\text{prop}}) - \ell(\mu_{\text{cur}}) \approx \nabla_\mu \ell(\mu_{\text{cur}}) \cdot (\mu_{\text{prop}} - \mu_{\text{cur}})$$

which is $O(\beta)$ for small $\beta$, regardless of $d$. So the acceptance rate is bounded away from zero uniformly in $d$.

---

## Sketch 4: Posterior Mean as a Wiener Filter

**Observation:** For the drift-only case, the posterior mean has an explicit form that looks like a Wiener filter.

Under $\Pi_0 = \mathcal{GP}(0, k_\mu)$, the posterior is Gaussian with:

- **Mean:** $\hat{\mu} = k_\mu \cdot (k_\mu + C_\text{noise})^{-1} \cdot m_\text{data}$
- **Covariance:** $\hat{k} = k_\mu - k_\mu \cdot (k_\mu + C_\text{noise})^{-1} \cdot k_\mu$

where $m_\text{data}(t) = dX_t/dt$ (formal derivative of $X$) and $C_\text{noise} = \text{Id}$ (unit noise from the Brownian motion).

In Fourier space (if $k_\mu$ is stationary with spectral density $S_\mu(\omega)$):

$$\hat{\mu}_\omega = \frac{S_\mu(\omega)}{S_\mu(\omega) + 1} \cdot \tilde{X}_\omega$$

This is exactly the **Wiener filter** for signal $\mu$ in white noise, where $S_\mu(\omega)$ is the signal power spectrum and 1 is the noise power. The posterior mean is the minimum mean-square-error linear estimator of $\mu$ from $X$.

**Implication:** The GP posterior mean can be computed efficiently via FFT when the kernel is stationary, making the posterior computable in $O(n \log n)$ time.

---

## Sketch 5: Leverage Effect as Cross-Covariance

In equity markets, $\mu_t$ and $\sigma_t$ are empirically negatively correlated (the "leverage effect"): volatility rises when returns fall.

A joint GP prior capturing this:

$$\begin{pmatrix} \mu \\ f \end{pmatrix} \sim \mathcal{GP}\!\left(\begin{pmatrix} 0 \\ m_f \end{pmatrix}, \begin{pmatrix} k_\mu & k_{\mu f} \\ k_{\mu f}^\top & k_f \end{pmatrix}\right)$$

where $\sigma = e^f$ and $k_{\mu f}(s, t) < 0$ (negative cross-covariance) encodes the leverage effect.

The posterior under this prior would automatically "borrow strength" — observing high volatility would shift the posterior on $\mu$ downward. This is a **statistically natural** way to model leverage that doesn't require specifying a parametric model.

**Open question:** Under what conditions on the cross-covariance kernel does this prior maintain positive definiteness of the joint covariance matrix operator?

---

*These sketches should be formalized into full proofs as the project develops. Flag any errors or gaps immediately.*
