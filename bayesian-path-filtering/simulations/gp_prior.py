"""
gp_prior.py — Gaussian process prior sampling on [0, T].

Demonstrates the Matérn GP prior on drift functions μ ~ GP(0, k_μ),
including prior samples, RKHS norm computation, and small-ball probabilities.

Usage
-----
    python simulations/gp_prior.py
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import build_kernel_matrix, matern_kernel, squared_exp_kernel


def sample_gp(t_grid, K, n_samples=5, seed=None):
    """
    Draw samples from GP(0, K) using Cholesky decomposition.

    Parameters
    ----------
    t_grid : np.ndarray, shape (n,)
    K : np.ndarray, shape (n, n)
        Kernel matrix.
    n_samples : int
    seed : int or None

    Returns
    -------
    samples : np.ndarray, shape (n_samples, n)
    """
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(K)
    Z = rng.standard_normal((len(t_grid), n_samples))
    return (L @ Z).T


def rkhs_norm(f, K_inv):
    """
    Compute the RKHS norm ‖f‖²_H = f^T K^{-1} f.

    Parameters
    ----------
    f : np.ndarray, shape (n,)
    K_inv : np.ndarray, shape (n, n)

    Returns
    -------
    float
    """
    return float(f @ K_inv @ f)


def empirical_small_ball_probability(t_grid, K, epsilon_values, n_monte_carlo=5000, seed=42):
    """
    Estimate Π₀(‖μ‖_{L²} ≤ ε) via Monte Carlo for a range of ε values.

    Parameters
    ----------
    t_grid : np.ndarray, shape (n,)
    K : np.ndarray, shape (n, n)
    epsilon_values : np.ndarray
    n_monte_carlo : int
    seed : int

    Returns
    -------
    probs : np.ndarray, shape (len(epsilon_values),)
    """
    samples = sample_gp(t_grid, K, n_samples=n_monte_carlo, seed=seed)
    dt = t_grid[1] - t_grid[0]  # assumes uniform grid
    l2_norms = np.sqrt(np.sum(samples**2 * dt, axis=1))

    probs = np.array([np.mean(l2_norms <= eps) for eps in epsilon_values])
    return probs


def plot_prior_samples(t_grid, samples, title="GP Prior Samples", ax=None):
    """Plot GP prior sample paths."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))

    for i, s in enumerate(samples):
        ax.plot(t_grid, s, alpha=0.7, linewidth=1.2, label=f"Sample {i+1}" if i < 3 else None)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel("t")
    ax.set_ylabel("μ(t)")
    ax.set_title(title)
    ax.legend(loc='upper right')
    return ax


def plot_small_ball(epsilon_values, probs, kernel_labels, ax=None):
    """Plot small-ball probability curves on log-log scale."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    for probs_i, label in zip(probs, kernel_labels):
        mask = probs_i > 0
        ax.loglog(epsilon_values[mask], probs_i[mask], marker='o', markersize=4, label=label)

    # Reference line: expected slope −1/ν for Matérn-ν
    eps_ref = epsilon_values[epsilon_values > 0]
    ax.loglog(eps_ref, 0.5 * eps_ref**2, 'k--', alpha=0.5, label='~ε² reference')

    ax.set_xlabel("ε")
    ax.set_ylabel("Π₀(‖μ‖_{L²} ≤ ε)")
    ax.set_title("Small-Ball Probability vs. ε (log-log)")
    ax.legend()
    return ax


def main():
    # Setup
    T = 1.0
    n_grid = 200
    t_grid = np.linspace(0, T, n_grid)

    kernel_configs = [
        {"name": "Matérn-0.5 (ℓ=0.2)", "fn": matern_kernel, "kwargs": {"nu": 0.5,  "length_scale": 0.2}},
        {"name": "Matérn-1.5 (ℓ=0.2)", "fn": matern_kernel, "kwargs": {"nu": 1.5,  "length_scale": 0.2}},
        {"name": "Matérn-2.5 (ℓ=0.2)", "fn": matern_kernel, "kwargs": {"nu": 2.5,  "length_scale": 0.2}},
        {"name": "Sq.Exp. (ℓ=0.2)",    "fn": squared_exp_kernel, "kwargs": {"length_scale": 0.2}},
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for i, config in enumerate(kernel_configs):
        K = build_kernel_matrix(t_grid, config["fn"], **config["kwargs"])
        samples = sample_gp(t_grid, K, n_samples=6, seed=i)
        plot_prior_samples(t_grid, samples, title=config["name"], ax=axes[i])

    plt.suptitle("GP Prior Sample Paths for Different Kernels", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("plots/gp_prior_samples.png", dpi=150, bbox_inches='tight')
    print("Saved: plots/gp_prior_samples.png")

    # Small-ball probability experiment
    print("\nComputing small-ball probabilities...")
    epsilon_values = np.logspace(-2, 0, 30)
    all_probs = []
    labels = []

    for config in kernel_configs[:3]:  # Matérn kernels only
        K = build_kernel_matrix(t_grid, config["fn"], **config["kwargs"])
        probs = empirical_small_ball_probability(t_grid, K, epsilon_values, n_monte_carlo=3000)
        all_probs.append(probs)
        labels.append(config["name"])

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    plot_small_ball(epsilon_values, all_probs, labels, ax=ax2)
    plt.tight_layout()
    plt.savefig("plots/small_ball_probability.png", dpi=150, bbox_inches='tight')
    print("Saved: plots/small_ball_probability.png")

    # RKHS norm of a target function
    mu_star = lambda t: np.sin(2 * np.pi * t)
    mu_star_vals = mu_star(t_grid)

    print("\nRKHS norms of μ*(t) = sin(2πt):")
    for config in kernel_configs:
        K = build_kernel_matrix(t_grid, config["fn"], **config["kwargs"])
        K_inv = np.linalg.inv(K)
        norm = rkhs_norm(mu_star_vals, K_inv)
        print(f"  {config['name']:30s}: ‖μ*‖²_H = {norm:.4f}")

    plt.show()


if __name__ == "__main__":
    import os
    os.makedirs("plots", exist_ok=True)
    main()
