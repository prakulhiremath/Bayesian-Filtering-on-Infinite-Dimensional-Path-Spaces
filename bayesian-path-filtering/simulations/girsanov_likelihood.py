"""
girsanov_likelihood.py — Girsanov density computation and log-likelihood evaluation.

Implements the Girsanov formula for the log-likelihood of a drift function μ
given an observed path X_{0:T}:

    ℓ(μ) = ∫₀ᵀ μ_t dX_t − ½ ∫₀ᵀ μ_t² dt

and the discrete (Euler-Maruyama) approximation:

    ℓ_δ(μ) = Σ_i μ_{t_i} ΔX_i − ½ Σ_i μ_{t_i}² Δt_i

Usage
-----
    python simulations/girsanov_likelihood.py
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import simulate_sde, matern_kernel, build_kernel_matrix, sample_gp


# ---------------------------------------------------------------------------
# Log-likelihood functions
# ---------------------------------------------------------------------------

def log_likelihood_continuous(mu_vals, dX, dt):
    """
    Approximate the continuous-time Girsanov log-likelihood:
        ℓ(μ) = ∫₀ᵀ μ_t dX_t − ½ ∫₀ᵀ μ_t² dt

    Uses left-endpoint Riemann sums (Euler-Maruyama approximation of the
    Itô integral).

    Parameters
    ----------
    mu_vals : np.ndarray, shape (n,)
        Drift values at left endpoints of each interval.
    dX : np.ndarray, shape (n,)
        Path increments X_{t_{i+1}} − X_{t_i}.
    dt : np.ndarray, shape (n,)
        Time increments t_{i+1} − t_i.

    Returns
    -------
    float
        Log-likelihood value.
    """
    ito_integral = np.sum(mu_vals * dX)
    quadratic_term = 0.5 * np.sum(mu_vals ** 2 * dt)
    return ito_integral - quadratic_term


def log_likelihood_euler(mu_vals, dX, dt):
    """
    Euler-Maruyama discrete log-likelihood (equivalent form):
        ℓ_δ(μ) = Σ_i [μ_{t_i} ΔX_i/Δt_i − ½ μ_{t_i}²] Δt_i

    This is the same as log_likelihood_continuous — both are left-endpoint
    Riemann-Itô sums. Kept separate for conceptual clarity.
    """
    return log_likelihood_continuous(mu_vals, dX, dt)


def log_likelihood_milstein(mu_vals, sigma_vals, dX, dt, dW=None):
    """
    Milstein-corrected log-likelihood for non-unit diffusion σ.

    ℓ(μ, σ) = ∫₀ᵀ (μ_t/σ_t²) dX_t − ½ ∫₀ᵀ (μ_t/σ_t)² dt − ∫₀ᵀ log σ_t dt

    Parameters
    ----------
    mu_vals : np.ndarray, shape (n,)
    sigma_vals : np.ndarray, shape (n,)
    dX : np.ndarray, shape (n,)
    dt : np.ndarray, shape (n,)
    dW : np.ndarray or None
        Brownian increments (needed for Milstein correction; if None, uses dX).

    Returns
    -------
    float
    """
    ito_term = np.sum((mu_vals / sigma_vals**2) * dX)
    quadratic_term = 0.5 * np.sum((mu_vals / sigma_vals)**2 * dt)
    log_sigma_term = np.sum(np.log(sigma_vals) * dt)
    return ito_term - quadratic_term - log_sigma_term


# ---------------------------------------------------------------------------
# Novikov condition check
# ---------------------------------------------------------------------------

def novikov_condition(K, t_grid, n_samples=2000, seed=42):
    """
    Empirically verify Novikov's condition:
        E[exp(½ ∫₀ᵀ μ_t² dt)] < ∞

    Approximates via Monte Carlo under the GP prior.

    Parameters
    ----------
    K : np.ndarray, shape (n, n)
        GP prior kernel matrix.
    t_grid : np.ndarray, shape (n,)
    n_samples : int
    seed : int

    Returns
    -------
    dict with keys: 'mean', 'std', 'max', 'finite_frac'
    """
    dt = np.diff(t_grid)
    samples = sample_gp(t_grid[:-1], K[:-1, :-1], n_samples=n_samples, seed=seed)

    # Compute ½ ∫ μ_t² dt for each sample
    half_quad = 0.5 * np.sum(samples**2 * dt[np.newaxis, :], axis=1)

    novikov_vals = np.exp(half_quad)
    return {
        'mean': np.mean(novikov_vals),
        'std': np.std(novikov_vals),
        'max': np.max(novikov_vals),
        'finite_frac': np.mean(np.isfinite(novikov_vals)),
        'log_half_quad_mean': np.mean(half_quad),
        'log_half_quad_std': np.std(half_quad),
    }


# ---------------------------------------------------------------------------
# Likelihood landscape visualization
# ---------------------------------------------------------------------------

def likelihood_surface_1d(mu_star, t_grid, X, dX, dt,
                           n_perturbations=50, perturbation_scale=1.0, seed=42):
    """
    Visualize the likelihood surface along random directions in function space.

    Generates paths through function space:
        μ(α) = μ* + α · v, α ∈ [-2, 2]

    where v is a random direction drawn from the GP prior.

    Parameters
    ----------
    mu_star : np.ndarray, shape (n,)
        True drift (center of exploration).
    t_grid : np.ndarray, shape (n+1,)
    X : np.ndarray, shape (n+1,)
    dX : np.ndarray, shape (n,)
    dt : np.ndarray, shape (n,)
    n_perturbations : int
        Number of random directions.
    perturbation_scale : float
    seed : int

    Returns
    -------
    alphas : np.ndarray
    log_likelihoods : np.ndarray, shape (n_perturbations, len(alphas))
    """
    rng = np.random.default_rng(seed)
    n = len(t_grid) - 1
    alphas = np.linspace(-3, 3, 60)
    log_likelihoods = np.zeros((n_perturbations, len(alphas)))

    for i in range(n_perturbations):
        direction = rng.standard_normal(n)
        direction /= np.sqrt(np.sum(direction**2 * dt)) + 1e-10  # L² normalize

        for j, alpha in enumerate(alphas):
            mu_test = mu_star + alpha * perturbation_scale * direction
            log_likelihoods[i, j] = log_likelihood_continuous(mu_test, dX, dt)

    return alphas, log_likelihoods


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main():
    import os
    os.makedirs("plots", exist_ok=True)

    # Setup
    T = 1.0
    n_grid = 500
    t_grid = np.linspace(0, T, n_grid + 1)
    dt = np.diff(t_grid)

    # True drift and diffusion
    mu_true = lambda t: np.sin(2 * np.pi * t)
    sigma_true = lambda t: 1.0  # unit diffusion for now

    # Simulate path
    X, dW = simulate_sde(mu_true, sigma_true, t_grid, x0=0.0, seed=42)
    dX = np.diff(X)

    mu_true_vals = mu_true(t_grid[:-1])

    # Evaluate log-likelihood at the true drift
    ll_true = log_likelihood_continuous(mu_true_vals, dX, dt)
    print(f"Log-likelihood at true μ*: {ll_true:.4f}")

    # Evaluate at zero drift
    ll_zero = log_likelihood_continuous(np.zeros_like(mu_true_vals), dX, dt)
    print(f"Log-likelihood at μ ≡ 0:   {ll_zero:.4f}")

    # Novikov condition check
    print("\nNovikov condition check:")
    K = build_kernel_matrix(t_grid[:-1], matern_kernel, nu=1.5, length_scale=0.2)
    novikov = novikov_condition(K, t_grid, n_samples=1000)
    for key, val in novikov.items():
        print(f"  {key:30s}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")

    # Likelihood surface along random directions
    print("\nComputing likelihood surface...")
    alphas, log_liks = likelihood_surface_1d(mu_true_vals, t_grid, X, dX, dt,
                                              n_perturbations=20, seed=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: observed path
    axes[0].plot(t_grid, X, color='steelblue', linewidth=1.0, label='Observed path')
    axes[0].plot(t_grid[:-1], mu_true_vals, 'r--', linewidth=1.5, label='True drift μ*(t)')
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("X_t / μ_t")
    axes[0].set_title("Simulated SDE Path")
    axes[0].legend()

    # Panel 2: log-likelihood along random directions
    ll_normalized = log_liks - log_liks[:, [np.argmin(np.abs(alphas))]]  # center at α=0
    for i in range(min(10, log_liks.shape[0])):
        axes[1].plot(alphas, ll_normalized[i], alpha=0.5, linewidth=1.0)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=1.5, label='α=0 (true μ*)')
    axes[1].set_xlabel("α (perturbation magnitude)")
    axes[1].set_ylabel("ℓ(μ* + αv) − ℓ(μ*)")
    axes[1].set_title("Log-Likelihood Along Random Directions")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("plots/girsanov_likelihood.png", dpi=150, bbox_inches='tight')
    print("\nSaved: plots/girsanov_likelihood.png")
    plt.show()


if __name__ == "__main__":
    main()
