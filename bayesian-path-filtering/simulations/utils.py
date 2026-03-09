"""
utils.py — Shared numerical utilities for Bayesian path filtering simulations.
"""

import numpy as np
from scipy.linalg import cholesky, solve_triangular


# ---------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------

def matern_kernel(s, t, nu=1.5, length_scale=1.0, variance=1.0):
    """
    Matérn covariance kernel k(s, t).

    Parameters
    ----------
    s, t : float or np.ndarray
        Input points.
    nu : float
        Smoothness parameter. Common values: 0.5, 1.5, 2.5.
    length_scale : float
        Length scale ℓ.
    variance : float
        Signal variance σ².

    Returns
    -------
    float or np.ndarray
        Kernel value(s).
    """
    from scipy.special import kv, gamma as gamma_fn
    r = np.abs(s - t) / length_scale
    r = np.where(r == 0, 1e-10, r)  # avoid division by zero

    if nu == 0.5:
        return variance * np.exp(-r)
    elif nu == 1.5:
        return variance * (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
    elif nu == 2.5:
        return variance * (1 + np.sqrt(5) * r + 5 * r**2 / 3) * np.exp(-np.sqrt(5) * r)
    else:
        # General case via modified Bessel function
        factor = (2 ** (1 - nu)) / gamma_fn(nu)
        x = np.sqrt(2 * nu) * r
        return variance * factor * (x ** nu) * kv(nu, x)


def squared_exp_kernel(s, t, length_scale=1.0, variance=1.0):
    """
    Squared exponential (RBF) kernel: k(s,t) = σ² exp(-|s-t|²/(2ℓ²)).
    """
    r2 = ((s - t) / length_scale) ** 2
    return variance * np.exp(-0.5 * r2)


def build_kernel_matrix(t_grid, kernel_fn, jitter=1e-8, **kernel_kwargs):
    """
    Build the n×n kernel matrix K with entries K[i,j] = k(t_i, t_j).

    Parameters
    ----------
    t_grid : np.ndarray, shape (n,)
        Time grid.
    kernel_fn : callable
        Kernel function with signature kernel_fn(s, t, **kwargs).
    jitter : float
        Small diagonal regularization for numerical stability.
    **kernel_kwargs
        Additional keyword arguments passed to kernel_fn.

    Returns
    -------
    K : np.ndarray, shape (n, n)
        Kernel matrix.
    """
    n = len(t_grid)
    s_grid, t_grid_2d = np.meshgrid(t_grid, t_grid, indexing='ij')
    K = kernel_fn(s_grid, t_grid_2d, **kernel_kwargs)
    K += jitter * np.eye(n)
    return K


# ---------------------------------------------------------------------------
# SDE simulation
# ---------------------------------------------------------------------------

def simulate_sde(mu_fn, sigma_fn, t_grid, x0=0.0, seed=None):
    """
    Simulate dXt = μ(t) dt + σ(t) dWt using Euler-Maruyama.

    Parameters
    ----------
    mu_fn : callable
        Drift function t ↦ μ(t).
    sigma_fn : callable
        Diffusion function t ↦ σ(t).
    t_grid : np.ndarray, shape (n+1,)
        Time grid including t=0.
    x0 : float
        Initial condition X₀.
    seed : int or None
        Random seed.

    Returns
    -------
    X : np.ndarray, shape (n+1,)
        Simulated path.
    dW : np.ndarray, shape (n,)
        Brownian increments.
    """
    rng = np.random.default_rng(seed)
    n = len(t_grid) - 1
    dt = np.diff(t_grid)

    dW = rng.normal(0, 1, size=n) * np.sqrt(dt)
    X = np.zeros(n + 1)
    X[0] = x0

    for i in range(n):
        X[i + 1] = X[i] + mu_fn(t_grid[i]) * dt[i] + sigma_fn(t_grid[i]) * dW[i]

    return X, dW


# ---------------------------------------------------------------------------
# Itô integral approximation
# ---------------------------------------------------------------------------

def ito_integral(integrand, dX, dt=None):
    """
    Approximate the Itô integral ∫ f(t) dX_t using left-endpoint Riemann sums.

    Parameters
    ----------
    integrand : np.ndarray, shape (n,)
        Values f(t_i) at left endpoints.
    dX : np.ndarray, shape (n,)
        Increments X_{t_{i+1}} - X_{t_i}.
    dt : np.ndarray or None
        Time increments (not used here, included for interface consistency).

    Returns
    -------
    float
        Approximation of ∫ f(t) dX_t.
    """
    return np.sum(integrand * dX)


def quadratic_variation(X, t_grid=None):
    """
    Compute the realized quadratic variation [X]_T = Σ (X_{i+1} - X_i)².

    Parameters
    ----------
    X : np.ndarray, shape (n+1,)
        Observed path.
    t_grid : np.ndarray or None
        Time grid (unused, for interface consistency).

    Returns
    -------
    float
        Realized quadratic variation.
    """
    return np.sum(np.diff(X) ** 2)


def realized_volatility(X, t_grid):
    """
    Nonparametric estimator of σ²(t) via kernel-smoothed squared increments.

    Parameters
    ----------
    X : np.ndarray, shape (n+1,)
        Observed path.
    t_grid : np.ndarray, shape (n+1,)
        Time grid.

    Returns
    -------
    t_mid : np.ndarray, shape (n,)
        Midpoints of time intervals.
    rv_local : np.ndarray, shape (n,)
        Local realized variance estimates at midpoints.
    """
    dX = np.diff(X)
    dt = np.diff(t_grid)
    t_mid = 0.5 * (t_grid[:-1] + t_grid[1:])
    rv_local = dX ** 2 / dt  # estimates σ²(t_i) * dt_i / dt_i = σ²(t_i)
    return t_mid, rv_local


# ---------------------------------------------------------------------------
# Hellinger distance (Monte Carlo approximation)
# ---------------------------------------------------------------------------

def hellinger_distance_mc(log_w1, log_w2, n_samples=10000, seed=None):
    """
    Estimate Hellinger distance between two distributions via their
    unnormalized log-weights against a common base measure (Monte Carlo).

    d_H²(P, Q) = 1 - ∫ √(dP/dR · dQ/dR) dR

    Parameters
    ----------
    log_w1 : np.ndarray, shape (n_samples,)
        Log unnormalized weights for distribution P.
    log_w2 : np.ndarray, shape (n_samples,)
        Log unnormalized weights for distribution Q.
    n_samples : int
        Number of Monte Carlo samples.
    seed : int or None
        Unused (samples assumed pre-drawn).

    Returns
    -------
    float
        Estimated Hellinger distance d_H(P, Q) ∈ [0, 1].
    """
    # Normalize log weights for numerical stability
    log_w1 = log_w1 - np.max(log_w1)
    log_w2 = log_w2 - np.max(log_w2)

    w1 = np.exp(log_w1)
    w2 = np.exp(log_w2)

    w1 /= w1.sum()
    w2 /= w2.sum()

    # Hellinger: d_H² = 1 - Σ √(w1_i · w2_i) / √(Σ w1_i · Σ w2_i)
    overlap = np.sum(np.sqrt(w1 * w2))
    d_H_sq = max(0.0, 1.0 - overlap)
    return np.sqrt(d_H_sq)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def log_sum_exp(log_weights):
    """Numerically stable log-sum-exp."""
    max_lw = np.max(log_weights)
    return max_lw + np.log(np.sum(np.exp(log_weights - max_lw)))


def effective_sample_size(log_weights):
    """
    Effective sample size (ESS) for importance-weighted samples.
    ESS = (Σ w_i)² / Σ w_i²
    """
    log_weights = log_weights - log_sum_exp(log_weights)
    weights = np.exp(log_weights)
    return 1.0 / np.sum(weights ** 2)
