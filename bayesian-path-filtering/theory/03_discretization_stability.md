# 3. Discretization Stability

## Problem Statement

In practice, we never observe a continuous path $X_{0:T}$. We observe discrete snapshots:

$$X_{t_0}, X_{t_1}, \ldots, X_{t_n}, \quad 0 = t_0 < t_1 < \cdots < t_n = T$$

with mesh $\delta = \max_i |t_{i+1} - t_i|$.

Let:
- $\Pi^\infty(\cdot \mid X_{0:T})$ — the ideal **continuous-time posterior**
- $\Pi^\delta(\cdot \mid X_{t_0:\ldots:t_n})$ — the **discretized posterior** using Euler-Maruyama likelihood

**Question:** Does $\Pi^\delta \to \Pi^\infty$ as $\delta \to 0$, and at what rate?

---

## The Discretized Likelihood

### Euler-Maruyama Approximation

Under Euler-Maruyama, the SDE increments are approximated as:

$$X_{t_{i+1}} - X_{t_i} \approx \mu_{t_i}(t_{i+1} - t_i) + \sigma_{t_i}(W_{t_{i+1}} - W_{t_i})$$

The discrete likelihood is:

$$p^\delta(X_{t_0:\ldots:t_n} \mid \mu, \sigma) = \prod_{i=0}^{n-1} \frac{1}{\sigma_{t_i}\sqrt{2\pi\delta_i}} \exp\!\left(-\frac{(X_{t_{i+1}} - X_{t_i} - \mu_{t_i}\delta_i)^2}{2\sigma_{t_i}^2 \delta_i}\right)$$

where $\delta_i = t_{i+1} - t_i$.

### Log-Likelihood Comparison

The continuous-time log-likelihood (from Girsanov, with $\sigma \equiv 1$) is:

$$\ell^\infty(\mu) = \int_0^T \mu_t \, dX_t - \frac{1}{2}\int_0^T \mu_t^2 \, dt$$

The discrete Euler-Maruyama log-likelihood is:

$$\ell^\delta(\mu) = \sum_{i=0}^{n-1} \frac{\mu_{t_i}(X_{t_{i+1}} - X_{t_i})}{\delta_i} - \frac{1}{2}\sum_{i=0}^{n-1} \mu_{t_i}^2 \delta_i$$

This is a **Riemann-Itô approximation** of $\ell^\infty$. The approximation error is:

$$\ell^\infty(\mu) - \ell^\delta(\mu) = \sum_{i=0}^{n-1} \int_{t_i}^{t_{i+1}} (\mu_t - \mu_{t_i}) \, dX_t + O(\delta)$$

The first term is $O(\delta^{1/2})$ in $L^2$ by Itô isometry, giving **overall error $O(\delta^{1/2})$**.

---

## Measuring Posterior Discrepancy

We measure the discrepancy between $\Pi^\infty$ and $\Pi^\delta$ using the **Hellinger distance**:

$$d_H^2(\Pi^\infty, \Pi^\delta) = \frac{1}{2}\int_{L^2} \left(\sqrt{\frac{d\Pi^\infty}{d\Pi_0}} - \sqrt{\frac{d\Pi^\delta}{d\Pi_0}}\right)^2 d\Pi_0$$

The Hellinger distance is preferred over total variation or KL divergence because:
- It is symmetric
- It satisfies $d_H \leq d_{TV} \leq \sqrt{2} d_H$ (comparable to TV)
- Square roots dampen the effect of the normalizing constant

---

## Main Theorem

**Theorem 3.1** *(Discretization stability)*

Let $\sigma \equiv 1$, $\Pi_0 = \mathcal{GP}(0, k_\mu)$ with Matérn kernel, and suppose $\mu^* \in H^1([0,T])$.

Under uniform mesh $\delta = T/n$, as $\delta \to 0$:

$$d_H(\Pi^\infty(\cdot \mid X_{0:T}),\ \Pi^\delta(\cdot \mid X_{t_0:\ldots:t_n})) = O(\delta^{1/2})$$

in $\mathbb{P}^{\mu^*}$-probability.

**Remark:** The rate $\delta^{1/2}$ matches the **strong convergence rate of Euler-Maruyama** for SDEs with Lipschitz coefficients. This is sharp in the sense that higher-order schemes (Milstein) do not improve the rate when $\mu^*$ has only $H^1$ regularity.

---

## Proof Sketch

**Step 1: Bound the Hellinger distance in terms of log-likelihood differences.**

Using the Cauchy-Schwarz inequality and properties of the Hellinger distance:

$$d_H^2(\Pi^\infty, \Pi^\delta) \leq C \cdot \mathbb{E}_{\Pi_0}\!\left[(\ell^\infty(\mu) - \ell^\delta(\mu))^2\right]^{1/2}$$

**Step 2: Bound the log-likelihood approximation error.**

$$\mathbb{E}_{\Pi_0}\!\left[(\ell^\infty(\mu) - \ell^\delta(\mu))^2\right] = \mathbb{E}_{\Pi_0}\!\left[\left(\sum_i \int_{t_i}^{t_{i+1}} (\mu_t - \mu_{t_i}) \, dW_t\right)^2\right]$$

By Itô isometry and the Lipschitz/Hölder continuity of $\mu$ (implied by $\mu^* \in H^1$):

$$= \mathbb{E}_{\Pi_0}\!\left[\sum_i \int_{t_i}^{t_{i+1}} (\mu_t - \mu_{t_i})^2 \, dt\right] \leq C \delta \|\mu\|_{H^1}^2$$

Taking the square root gives $O(\delta^{1/2})$.

**Step 3: Control the normalizing constants.**

The normalizing constants $Z^\infty$ and $Z^\delta$ are both bounded away from zero and infinity under $\mathbb{P}^{\mu^*}$, uniformly in $\delta$. This follows from Novikov's condition and the bound in Step 2. $\square$

---

## Implications for MCMC

The main consequence for computation: standard MCMC algorithms degenerate as $\delta \to 0$ (the acceptance rate goes to zero as the dimension of the discretized problem grows). This is the **curse of dimensionality in MCMC**.

### Preconditioned Crank-Nicolson (pCN)

Cotter et al. (2013) showed that the **pCN proposal**:

$$\mu_{\text{prop}} = \sqrt{1-\beta^2} \, \mu_{\text{cur}} + \beta \xi, \quad \xi \sim \Pi_0$$

achieves an acceptance rate bounded away from zero uniformly in $\delta$. The key insight is that pCN proposes from the **prior-scaled ball** around the current state, not from a Euclidean ball.

The optimal scaling is $\beta \asymp n^{-1/4}$ (Roberts & Rosenthal, 2001, adapted to infinite dimensions).

### Dimension-Robust Convergence

For the preconditioned algorithm, the spectral gap $\gamma_n$ of the Markov chain satisfies:

$$\gamma_n \geq \gamma_\infty > 0 \quad \text{uniformly in } n = T/\delta$$

This is the key theorem making infinite-dimensional MCMC computationally tractable.

---

## Higher-Order Schemes

For $\mu^* \in H^2([0,T])$, the **Milstein scheme** gives:

$$X_{t_{i+1}} - X_{t_i} \approx \mu_{t_i}\delta + \sigma_{t_i}(W_{t_{i+1}} - W_{t_i}) + \frac{1}{2}\sigma_{t_i}\sigma'_{t_i}\left((W_{t_{i+1}} - W_{t_i})^2 - \delta\right)$$

The corresponding likelihood approximation error is $O(\delta)$, giving:

$$d_H(\Pi^\infty, \Pi^{\text{Milstein}}) = O(\delta)$$

However, the Milstein scheme requires $\sigma'_t$, which introduces additional estimation complexity.

---

## Numerical Illustration

The following experiment (implemented in `simulations/discretization_error.py`) verifies Theorem 3.1:

1. Fix $T = 1$, $\mu^*(t) = \sin(2\pi t)$, $\sigma \equiv 1$
2. Simulate $X_{0:T}$ from $\mathbb{P}^{\mu^*}$
3. Compute GP posteriors for $\delta \in \{0.5, 0.25, 0.1, 0.05, 0.01\}$
4. Approximate $d_H(\Pi^\infty, \Pi^\delta)$ via Monte Carlo
5. Fit $\log d_H$ vs $\log \delta$ — expected slope $\approx 0.5$

---

## Open Problems

1. **Milstein without $\sigma'$:** Can the $O(\delta)$ rate be achieved without computing $\sigma'$, using e.g. Lévy area approximations?

2. **Non-uniform meshes:** Theorem 3.1 assumes uniform mesh. For adaptive meshes that concentrate points where $|\mu^*|$ is large, can the constant be improved?

3. **Joint stability:** Extend Theorem 3.1 to the joint $(\mu, \sigma)$ case. The $\sigma^{-1}$ in the Girsanov density creates additional sensitivity.

---

## References

- Cotter, S.L., Roberts, G.O., Stuart, A.M., & White, D. (2013). MCMC methods for functions. *Stat. Sci.* 28(3).
- Kloeden, P.E. & Platen, E. (1992). *Numerical Solution of Stochastic Differential Equations.* Springer.
- Hairer, M., Stuart, A.M., & Vollmer, S.J. (2014). Spectral gaps for a Metropolis-Hastings algorithm in infinite dimensions. *Ann. Appl. Probab.* 24(6).
- Roberts, G.O. & Rosenthal, J.S. (2001). Optimal scaling for various Metropolis-Hastings algorithms. *Stat. Sci.* 16(4).
