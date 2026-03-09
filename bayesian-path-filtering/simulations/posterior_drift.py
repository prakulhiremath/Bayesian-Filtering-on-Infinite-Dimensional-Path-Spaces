"""
posterior_drift.py — Full GP posterior for the drift function μ given observed path X_{0:T}.

The posterior is Gaussian with:
    Posterior mean:  μ̂ = K (K + C_noise)^{-1} m_data
    Posterior cov:   Σ̂ = K − K (K + C_noise)^{-1} K

where m_data[i] = ΔX_i / Δt_i (local increment estimate of μ) and
C_noise = diag(1/Δt_i) (heteroskedastic noise from Brownian motion).

This implements the discretized posterior for the model:
    ΔX_i = μ_{t_i} Δt_i + ΔW_i,  ΔW_i ~ N(0, Δt_i)

which gives:
    ΔX_i / √Δt_i = μ_{t_i} √Δt_i + Z_i,  Z_i ~ N(0, 1)

a heteroskedastic regression of μ on noisy slope estimates.

Usage
-----
    python simulations/posterior_drift.py
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import (
    build_kernel_matrix, matern_kernel, simulate_sde, sample_gp,
    log_sum_exp
)


# ---------------------------------------------------------------------------
# GP posterior (closed form)
# ---------------------------------------------------------------------------

def gp_posterior_drift(t_grid, dX, K_prior, sigma_obs=None, jitter=1e-8):
    """
    Compute the closed-form GP posterior for drift μ.

    Model:  ΔX_i ~ N(μ_{t_i} Δt_i,  σ² Δt_i)

    The design matrix is the identity (we observe μ directly, corrupted by noise).

    Parameters
    ----------
    t_grid : np.ndarray, shape (n+1,)
        Time grid.
    dX : np.ndarray, shape (n,)
        Path increments X_{t_{i+1}} − X_{t_i}.
    K_prior : np.ndarray, shape (n, n)
        Prior kernel matrix evaluated at left endpoints t_grid[:-1].
    sigma_obs : float or None
        Observation noise std. If None, uses σ = 1 (unit diffusion).
    jitter : float
        Numerical jitter for Cholesky stability.

    Returns
    -------
    mu_post : np.ndarray, shape (n,)
        Posterior mean.
    var_post : np.ndarray, shape (n,)
        Posterior marginal variances (diagonal of posterior covariance).
    Sigma_post : np.ndarray, shape (n, n)
        Full posterior covariance matrix.
    """
    n = len(dX)
    dt = np.diff(t_grid)

    # Noise variance at each point: Var(ΔX_i / Δt_i) = σ² / Δt_i
    if sigma_obs is None:
        sigma_obs = 1.0
    noise_var = (sigma_obs ** 2) / dt  # shape (n,)
    C_noise = np.diag(noise_var)

    # "Observed" values: local slope estimates
    y = dX / dt  # shape (n,)  — these are noisy observations of μ

    # Posterior: standard GP regression formulas
    # K_post = K − K (K + C_noise)^{-1} K
    # mu_post = K (K + C_noise)^{-1} y
    A = K_prior + C_noise  # shape (n, n)
    A += jitter * np.eye(n)

    # Solve via Cholesky for numerical stability
    L = np.linalg.cholesky(A)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))   # A^{-1} y
    V = np.linalg.solve(L, K_prior)                         # L^{-1} K

    mu_post = K_prior @ alpha
    Sigma_post = K_prior - V.T @ V
    var_post = np.diag(Sigma_post)

    return mu_post, var_post, Sigma_post


def posterior_samples(mu_post, Sigma_post, n_samples=20, seed=None):
    """
    Draw samples from the GP posterior N(μ̂, Σ̂).

    Parameters
    ----------
    mu_post : np.ndarray, shape (n,)
    Sigma_post : np.ndarray, shape (n, n)
    n_samples : int
    seed : int or None

    Returns
    -------
    np.ndarray, shape (n_samples, n)
    """
    rng = np.random.default_rng(seed)
    n = len(mu_post)
    # Add small jitter for positive definiteness
    Sigma_jitter = Sigma_post + 1e-8 * np.eye(n)
    L = np.linalg.cholesky(Sigma_jitter)
    Z = rng.standard_normal((n, n_samples))
    return (mu_post[:, None] + L @ Z).T


# ---------------------------------------------------------------------------
# Posterior contraction experiment
# ---------------------------------------------------------------------------

def contraction_experiment(mu_star_fn, T_values, kernel_fn, kernel_kwargs,
                            n_grid_per_unit=200, n_paths=30, seed=0):
    """
    Measure posterior L² contraction as T increases.

    For each T in T_values:
    1. Simulate n_paths paths from μ*.
    2. Compute GP posterior for each path.
    3. Record ‖μ̂ − μ*‖_{L²} and posterior std.

    Parameters
    ----------
    mu_star_fn : callable, t ↦ μ*(t)
    T_values : list of float
    kernel_fn : callable
    kernel_kwargs : dict
    n_grid_per_unit : int
    n_paths : int
    seed : int

    Returns
    -------
    dict with keys 'T_values', 'mean_errors', 'std_errors', 'posterior_stds'
    """
    rng = np.random.default_rng(seed)
    mean_errors = []
    std_errors = []
    post_stds = []

    for T in T_values:
        n_grid = int(n_grid_per_unit * T)
        t_grid = np.linspace(0, T, n_grid + 1)
        dt = np.diff(t_grid)
        mu_star = mu_star_fn(t_grid[:-1])

        K = build_kernel_matrix(t_grid[:-1], kernel_fn, **kernel_kwargs)

        errors = []
        avg_post_std = []

        for _ in range(n_paths):
            seed_i = rng.integers(0, 2**31)
            X, _ = simulate_sde(mu_star_fn, lambda t: 1.0, t_grid, x0=0.0, seed=seed_i)
            dX = np.diff(X)

            mu_post, var_post, _ = gp_posterior_drift(t_grid, dX, K)

            # L² error: ‖μ̂ − μ*‖²_{L²} ≈ Σ (μ̂_i − μ*_i)² Δt_i
            l2_error = np.sqrt(np.sum((mu_post - mu_star)**2 * dt))
            errors.append(l2_error)
            avg_post_std.append(np.sqrt(np.mean(var_post)))

        mean_errors.append(np.mean(errors))
        std_errors.append(np.std(errors))
        post_stds.append(np.mean(avg_post_std))
        print(f"  T={T:.1f}: mean L² error = {np.mean(errors):.4f} ± {np.std(errors):.4f}")

    return {
        'T_values': T_values,
        'mean_errors': np.array(mean_errors),
        'std_errors': np.array(std_errors),
        'posterior_stds': np.array(post_stds),
    }


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main():
    import os
    os.makedirs("plots", exist_ok=True)

    # Setup
    T = 1.0
    n_grid = 300
    t_grid = np.linspace(0, T, n_grid + 1)
    dt = np.diff(t_grid)

    mu_star_fn = lambda t: np.sin(2 * np.pi * t)
    sigma_true = lambda t: 1.0

    # Simulate one path
    X, _ = simulate_sde(mu_star_fn, sigma_true, t_grid, x0=0.0, seed=42)
    dX = np.diff(X)

    # Build prior kernel matrices for different smoothness levels
    kernel_configs = [
        {"name": "Matérn-0.5",  "kwargs": {"nu": 0.5, "length_scale": 0.3}},
        {"name": "Matérn-1.5",  "kwargs": {"nu": 1.5, "length_scale": 0.3}},
        {"name": "Matérn-2.5",  "kwargs": {"nu": 2.5, "length_scale": 0.3}},
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    t_left = t_grid[:-1]
    mu_star_vals = mu_star_fn(t_left)

    for ax, config in zip(axes, kernel_configs):
        K = build_kernel_matrix(t_left, matern_kernel, **config["kwargs"])
        mu_post, var_post, Sigma_post = gp_posterior_drift(t_grid, dX, K)
        post_samp = posterior_samples(mu_post, Sigma_post, n_samples=10, seed=0)

        # 95% credible band
        ci_upper = mu_post + 1.96 * np.sqrt(var_post)
        ci_lower = mu_post - 1.96 * np.sqrt(var_post)

        # Plot
        for s in post_samp[:5]:
            ax.plot(t_left, s, alpha=0.25, linewidth=0.8, color='steelblue')
        ax.fill_between(t_left, ci_lower, ci_upper, alpha=0.25, color='steelblue', label='95% CI')
        ax.plot(t_left, mu_post, 'b-', linewidth=2.0, label='Posterior mean')
        ax.plot(t_left, mu_star_vals, 'r--', linewidth=1.5, label='True μ*')

        l2_err = np.sqrt(np.sum((mu_post - mu_star_vals)**2 * dt))
        ax.set_title(f"{config['name']}\nL² error: {l2_err:.4f}")
        ax.set_xlabel("t")
        ax.set_ylabel("μ(t)")
        ax.legend(fontsize=8)

    plt.suptitle("GP Posterior for Drift Function (T=1, σ=1)", fontsize=13)
    plt.tight_layout()
    plt.savefig("plots/posterior_drift.png", dpi=150, bbox_inches='tight')
    print("Saved: plots/posterior_drift.png")

    # Contraction experiment
    print("\nRunning posterior contraction experiment...")
    T_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    results = contraction_experiment(
        mu_star_fn, T_values,
        kernel_fn=matern_kernel,
        kernel_kwargs={"nu": 1.5, "length_scale": 0.3},
        n_paths=20, seed=7
    )

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.errorbar(results['T_values'], results['mean_errors'],
                 yerr=results['std_errors'], fmt='o-', label='Posterior mean error')
    ax2.plot(results['T_values'], results['posterior_stds'], 's--', label='Avg. posterior std')

    # Reference line: T^{-s/(2s+1)} for s=1 (Matérn-1.5), rate = T^{-1/3}
    T_arr = np.array(T_values, dtype=float)
    ref = results['mean_errors'][0] * (T_arr / T_arr[0]) ** (-1/3)
    ax2.plot(T_arr, ref, 'k:', label='T^{-1/3} reference')

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel("T (observation window)")
    ax2.set_ylabel("L² error")
    ax2.set_title("Posterior Contraction Rate (Matérn-1.5 prior)")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("plots/contraction_rate.png", dpi=150, bbox_inches='tight')
    print("Saved: plots/contraction_rate.png")

    plt.show()


if __name__ == "__main__":
    main()
