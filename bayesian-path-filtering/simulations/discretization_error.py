"""
discretization_error.py — Empirical verification of Theorem 3.1.

Measures the Hellinger distance between:
    Π^∞  — "continuous-time" posterior (fine grid proxy)
    Π^δ  — coarse-grid posterior

as a function of mesh size δ = T/n.

Expected: d_H(Π^∞, Π^δ) = O(δ^{1/2})

Usage
-----
    python simulations/discretization_error.py
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import build_kernel_matrix, matern_kernel, simulate_sde
from posterior_drift import gp_posterior_drift


# ---------------------------------------------------------------------------
# Hellinger distance between two Gaussians
# ---------------------------------------------------------------------------

def hellinger_gaussian(mu1, Sigma1, mu2, Sigma2, jitter=1e-8):
    """
    Exact Hellinger distance between two Gaussian distributions N(μ₁, Σ₁) and N(μ₂, Σ₂).

    d_H²(P, Q) = 1 − |Σ₁|^{1/4} |Σ₂|^{1/4} / |½(Σ₁+Σ₂)|^{1/2}
                    × exp(−⅛ (μ₁−μ₂)ᵀ [½(Σ₁+Σ₂)]^{-1} (μ₁−μ₂))

    Parameters
    ----------
    mu1, mu2 : np.ndarray, shape (n,)
    Sigma1, Sigma2 : np.ndarray, shape (n, n)
    jitter : float

    Returns
    -------
    float in [0, 1]
    """
    n = len(mu1)
    I = jitter * np.eye(n)

    Sigma1 = Sigma1 + I
    Sigma2 = Sigma2 + I
    Sigma_avg = 0.5 * (Sigma1 + Sigma2) + I

    sign1, logdet1 = np.linalg.slogdet(Sigma1)
    sign2, logdet2 = np.linalg.slogdet(Sigma2)
    sign_avg, logdet_avg = np.linalg.slogdet(Sigma_avg)

    if sign1 <= 0 or sign2 <= 0 or sign_avg <= 0:
        return 1.0  # degenerate case

    log_det_term = 0.25 * logdet1 + 0.25 * logdet2 - 0.5 * logdet_avg

    diff = mu1 - mu2
    Sigma_avg_inv = np.linalg.solve(Sigma_avg, np.eye(n))
    quad_form = diff @ Sigma_avg_inv @ diff

    log_bc = log_det_term - 0.125 * quad_form  # log Bhattacharyya coefficient
    bc = np.exp(np.clip(log_bc, -50, 0))

    d_H_sq = max(0.0, 1.0 - bc)
    return np.sqrt(d_H_sq)


# ---------------------------------------------------------------------------
# Discretization experiment
# ---------------------------------------------------------------------------

def discretization_experiment(mu_star_fn, T, delta_values,
                               kernel_fn, kernel_kwargs,
                               n_fine=2000, n_paths=20, seed=0):
    """
    Measure d_H(Π^∞, Π^δ) as δ varies.

    Uses n_fine grid as proxy for continuous-time posterior.

    Parameters
    ----------
    mu_star_fn : callable
    T : float
    delta_values : list of float
        Coarse mesh sizes to test.
    kernel_fn : callable
    kernel_kwargs : dict
    n_fine : int
        Number of intervals for the "continuous-time" reference.
    n_paths : int
    seed : int

    Returns
    -------
    dict with keys 'delta_values', 'mean_hellinger', 'std_hellinger'
    """
    rng = np.random.default_rng(seed)

    # Fine (reference) grid
    t_fine = np.linspace(0, T, n_fine + 1)
    K_fine = build_kernel_matrix(t_fine[:-1], kernel_fn, **kernel_kwargs)

    mean_hd = []
    std_hd = []

    for delta in delta_values:
        n_coarse = max(2, int(T / delta))
        t_coarse = np.linspace(0, T, n_coarse + 1)
        K_coarse = build_kernel_matrix(t_coarse[:-1], kernel_fn, **kernel_kwargs)

        hd_list = []

        for path_idx in range(n_paths):
            sim_seed = rng.integers(0, 2**31)

            # Simulate on fine grid
            X_fine, _ = simulate_sde(mu_star_fn, lambda t: 1.0, t_fine, x0=0.0, seed=sim_seed)
            dX_fine = np.diff(X_fine)

            # "Continuous" posterior on fine grid
            mu_inf, var_inf, Sigma_inf = gp_posterior_drift(t_fine, dX_fine, K_fine)

            # Coarsen: sub-sample the fine path at coarse grid points
            fine_idx = np.round(np.linspace(0, n_fine, n_coarse + 1)).astype(int)
            X_coarse = X_fine[fine_idx]
            dX_coarse = np.diff(X_coarse)

            # Coarse posterior
            mu_delta, var_delta, Sigma_delta = gp_posterior_drift(t_coarse, dX_coarse, K_coarse)

            # Interpolate coarse posterior mean to fine grid for comparison
            # (compare at coarse grid points for fair Hellinger computation)
            mu_inf_coarse = mu_inf[fine_idx[:-1]]
            var_inf_coarse = var_inf[fine_idx[:-1]]
            Sigma_inf_coarse = Sigma_inf[np.ix_(fine_idx[:-1], fine_idx[:-1])]

            hd = hellinger_gaussian(mu_inf_coarse, Sigma_inf_coarse,
                                     mu_delta, Sigma_delta)
            hd_list.append(hd)

        mean_hd.append(np.mean(hd_list))
        std_hd.append(np.std(hd_list))
        print(f"  δ={delta:.4f} (n={n_coarse:4d}): d_H = {np.mean(hd_list):.4f} ± {np.std(hd_list):.4f}")

    return {
        'delta_values': np.array(delta_values),
        'mean_hellinger': np.array(mean_hd),
        'std_hellinger': np.array(std_hd),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import os
    os.makedirs("plots", exist_ok=True)

    T = 1.0
    mu_star_fn = lambda t: np.sin(2 * np.pi * t)

    kernel_kwargs = {"nu": 1.5, "length_scale": 0.3}

    delta_values = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]

    print(f"Discretization experiment: T={T}, Matérn-1.5 prior")
    print(f"Testing δ ∈ {delta_values}\n")

    results = discretization_experiment(
        mu_star_fn, T, delta_values,
        kernel_fn=matern_kernel,
        kernel_kwargs=kernel_kwargs,
        n_fine=1000, n_paths=15, seed=42
    )

    # Fit power law: d_H ~ C δ^α
    log_delta = np.log(results['delta_values'])
    log_hd = np.log(results['mean_hellinger'] + 1e-12)
    coeffs = np.polyfit(log_delta, log_hd, 1)
    alpha_fit = coeffs[0]
    print(f"\nFitted exponent: α = {alpha_fit:.3f}  (theoretical: 0.5)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Log-log plot of Hellinger vs δ
    ax = axes[0]
    ax.errorbar(results['delta_values'], results['mean_hellinger'],
                yerr=results['std_hellinger'], fmt='o-', capsize=4,
                color='steelblue', label='Empirical d_H')

    # Reference lines
    delta_arr = results['delta_values']
    C = results['mean_hellinger'][0] / delta_arr[0]**0.5
    ax.plot(delta_arr, C * delta_arr**0.5, 'r--', label='O(δ^{1/2}) reference')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Mesh size δ")
    ax.set_ylabel("d_H(Π^∞, Π^δ)")
    ax.set_title(f"Hellinger Distance vs. Mesh Size\nFitted exponent: α = {alpha_fit:.3f}")
    ax.legend()

    # Posterior comparison at a specific δ
    ax2 = axes[1]
    t_fine = np.linspace(0, T, 1001)
    K_fine = build_kernel_matrix(t_fine[:-1], matern_kernel, **kernel_kwargs)
    X_fine, _ = simulate_sde(mu_star_fn, lambda t: 1.0, t_fine, x0=0.0, seed=99)
    dX_fine = np.diff(X_fine)
    mu_inf, var_inf, _ = gp_posterior_drift(t_fine, dX_fine, K_fine)

    for delta_plot in [0.1, 0.02, 0.005]:
        n_c = int(T / delta_plot)
        t_c = np.linspace(0, T, n_c + 1)
        K_c = build_kernel_matrix(t_c[:-1], matern_kernel, **kernel_kwargs)
        fine_idx = np.round(np.linspace(0, 1000, n_c + 1)).astype(int)
        X_c = X_fine[fine_idx]
        dX_c = np.diff(X_c)
        mu_d, var_d, _ = gp_posterior_drift(t_c, dX_c, K_c)
        ax2.plot(t_c[:-1], mu_d, linewidth=1.2, label=f"δ={delta_plot}")

    ax2.plot(t_fine[:-1], mu_inf, 'k-', linewidth=2.0, label='Reference (δ→0)', zorder=10)
    ax2.plot(t_fine[:-1], mu_star_fn(t_fine[:-1]), 'r--', linewidth=1.5, label='True μ*')
    ax2.set_xlabel("t")
    ax2.set_ylabel("Posterior mean μ̂(t)")
    ax2.set_title("Posterior Mean vs. Discretization")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("plots/discretization_error.png", dpi=150, bbox_inches='tight')
    print("\nSaved: plots/discretization_error.png")
    plt.show()


if __name__ == "__main__":
    main()
