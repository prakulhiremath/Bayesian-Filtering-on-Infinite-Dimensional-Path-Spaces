# Open Problems and Conjectures

A running list of unresolved questions, ranked roughly by tractability.

---

## Tier 1: Tractable (Paper-Ready with Effort)

### OP-1: Posterior Contraction Rate Theorem (GP Prior, Continuous Observation)

**Statement:** Let $\mu^* \in H^s([0,T])$, $\Pi_0 = \mathcal{GP}(0, k_\mu)$ with Matérn-$\nu$ kernel, $\nu > s$, $\sigma \equiv 1$, continuous observation on $[0,T]$ as $T \to \infty$. Then:

$$\Pi_T\!\left(\|\mu - \mu^*\|_{L^2} > M T^{-s/(2s+1)} (\log T)^{1/2}\right) \to 0 \quad \mathbb{P}^{\mu^*}\text{-prob.}$$

**Why tractable:** The proof technique is known (van der Vaart & van Zanten). The main new ingredient is handling the Girsanov likelihood instead of a Gaussian regression likelihood. The KL divergence simplifies to $\frac{1}{2}\|\mu - \mu^*\|_{L^2}^2$ (Girsanov formula), making the KL support condition easier than in general inverse problems.

**Key step:** Verifying that the prior puts sufficient mass near $\mu^*$ (small-ball probability for GP around a Sobolev function).

**Estimated effort:** 3–6 months for a complete proof.

---

### OP-2: Hellinger Bound for Euler-Maruyama Discretization

**Statement:** Under conditions of OP-1, with discrete observation at mesh $\delta = T/n$:

$$d_H(\Pi^\infty, \Pi^\delta) = O(\delta^{1/2})$$

in $\mathbb{P}^{\mu^*}$-probability.

**Why tractable:** The proof reduces to bounding the $L^2(\Pi_0)$-norm of the log-likelihood approximation error, which is controlled by the Itô isometry and Lipschitz properties of $\mu$.

**Key step:** Uniform control of the normalizing constants $Z^\infty$ and $Z^\delta$.

**Estimated effort:** 2–4 months.

---

## Tier 2: Moderately Hard (Dissertation-Level)

### OP-3: Posterior Existence for Joint $(\mu, \sigma)$

**Statement:** Let $\log \sigma \sim \mathcal{GP}(m_f, k_f)$ and $\mu \sim \mathcal{GP}(0, k_\mu)$ independent. The joint posterior $\Pi(\mu, \sigma \mid X_{0:T})$ is a well-defined probability measure on $L^2([0,T])^2$.

**Main difficulty:** The Girsanov density involves $\sigma_t^{-2}$, so integrability requires $\mathbb{E}_{\Pi_0}[\sigma_t^{-2p}] < \infty$ for some $p > 1$. For the log-normal prior $\sigma = e^f$:

$$\mathbb{E}[e^{-2pf_t}] = e^{-2pm_f(t) + 2p^2 k_f(t,t)}$$

This is finite for all $p$ — so the log-normal prior resolves the integrability issue. The challenge is making this work uniformly in $t$ and controlling the interaction between $\mu$ and $\sigma$.

**Key technique:** Use the fact that under the log-normal prior, $\sigma$ is bounded below in probability: $\Pi_0(\inf_t \sigma_t > \epsilon) \to 1$ as $m_f \to \infty$. Then condition on $\sigma > \epsilon$ and take $\epsilon \to 0$.

---

### OP-4: Adaptive Rates Under Unknown Smoothness

**Statement:** Under the hierarchical prior:

$$\mu \mid \ell \sim \mathcal{GP}(0, k_\ell), \quad \ell \sim \text{Gamma}(a, b)$$

the marginal posterior contracts at the rate $T^{-s/(2s+1)}$ simultaneously for all $\mu^* \in H^s$, without knowing $s$.

**Main difficulty:** The marginalization over $\ell$ introduces a mixture of Gaussian process priors. Standard adaptation results (Ghosal & van der Vaart, 2007) apply to i.i.d. data — extending to the SDE likelihood requires controlling the marginal small-ball probability.

**Conjecture:** The result holds, with the length-scale posterior concentrating on $\ell^* \asymp T^{-1/(2s+1)}$.

---

### OP-5: Posterior Contraction Under Misspecification

**Statement:** What happens when $\mu^* \notin H^s$ (the prior smoothness class)?

If $\mu^*$ has weaker smoothness $\mu^* \in H^r$ with $r < s < \nu$, the posterior should still contract, but at the slower rate $T^{-r/(2r+1)}$.

**Main difficulty:** Standard contraction proofs require $\mu^* \in \mathcal{H}_{k_\mu}$ (the RKHS). When $r < s$, $\mu^* \notin \mathcal{H}_{k_\mu}$, and the KL support condition fails in its standard form.

**Technique:** Use the **approximation theory** approach: replace $\mu^*$ with a sequence $\mu^*_T \in \mathcal{H}_{k_\mu}$ that approximates $\mu^*$ at rate $T^{r/(2r+1)}$ in $L^2$.

---

## Tier 3: Hard (Long-Term / Collaborative)

### OP-6: Bernstein-von Mises Theorem for Path Space

**Statement:** Under continuous observation with large $T$, the rescaled posterior:

$$T^{s/(2s+1)} (\mu - \hat{\mu}_T)$$

converges weakly to a Gaussian process in $L^2([0,T])$.

**Why hard:** Bernstein-von Mises in infinite dimensions generally fails unless the problem is "LAN" (locally asymptotically normal) in the right sense. For SDE drift estimation, LAN holds in the parametric case (Ibragimov & Has'minskii, 1981), but the nonparametric extension requires controlling the bias of the posterior mean.

**Recent work:** Castillo & Nickl (2014) proved BvM for Gaussian white noise models. Extension to SDE drift is open.

---

### OP-7: Semiparametric Efficiency for $\sigma$

**Statement:** The two-stage estimator (realized vol → GP posterior for $\mu$) achieves the **semiparametric efficiency bound** for estimating smooth functionals $\phi(\mu)$ when $\sigma$ is a nuisance parameter.

**Why hard:** Semiparametric efficiency in path space requires:
1. Computing the tangent space of the model at $(\mu^*, \sigma^*)$
2. Finding the efficient influence function
3. Showing the two-stage procedure achieves the Cramér-Rao bound for the tangent space

This is Le Cam theory in infinite dimensions — technically demanding and with few precedents.

---

### OP-8: Non-Stationary GP Priors and Locally Adaptive Rates

If $\mu^*$ has **spatially varying smoothness** (smooth in some regions, rough in others), stationary GP priors (Matérn, squared-exponential) are suboptimal. 

**Conjecture:** Deep GP priors (layers of GP compositions) or spatially-varying Matérn priors achieve locally adaptive rates $\varepsilon(t)$ that depend on the local smoothness of $\mu^*$ at $t$.

**Why hard:** There is no general theory of posterior contraction for deep GPs. The RKHS of a composition of GPs is not well-characterized.

---

## Meta-Problem: Right Topology for the Posterior

All results above use $L^2([0,T])$ as the function space. But there are reasons to prefer other topologies:

- **$C([0,T])$ (sup-norm):** More relevant for pointwise inference about $\mu_t$ at specific times $t$
- **$H^{-1}([0,T])$ (negative Sobolev):** May give faster rates by exploiting smoothing properties of the integral operator
- **Weak topology on $C([0,T])^*$:** Most natural for the Girsanov setup but hardest to work with

**Open question:** What is the natural topology for posterior contraction in this problem, and is $L^2$ the right choice?

---

*Last updated: 2025. Add new problems here as they arise during the research.*
