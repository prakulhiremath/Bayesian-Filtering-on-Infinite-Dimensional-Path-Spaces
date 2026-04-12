"""
baselines.py — Experiment 4: Path-space likelihood vs naive baselines.

Compares three methods for drift estimation from SDE observations:

METHOD 1 (Ours):
    GP regression on finite differences with HETEROSKEDASTIC noise
    derived from the Girsanov/SDE structure:
        y_i = DX_i / dt_i,   noise_var_i = sigma^2 / dt_i
    This is the correct SDE likelihood.

METHOD 2 (Baseline 1 — Naive GP on path values):
    GP regression directly on (t_i, X_ti).
    Treats path values as i.i.d. noisy observations of mu.
    WRONG: conflates Brownian fluctuations with drift signal.
    Misspecified at the likelihood level.

METHOD 3 (Baseline 2 — Homoskedastic finite differences):
    GP regression on finite differences DX_i/dt_i but with a
    SINGLE noise variance fit by MLE (ignores heteroskedastic structure).
    Partially correct structure but wrong noise model.

EXPECTED RESULT:
    At low sigma (high information): our method significantly
    outperforms both baselines in RMS and coverage.
    At high sigma (low information): all methods perform similarly
    since the prior dominates.

Usage
-----
    python simulations/baselines.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from utils import build_kernel_matrix, matern_kernel, simulate_sde
from posterior_drift import gp_posterior_drift


# ---------------------------------------------------------------------------
# Baseline 1 — Naive GP on path values X_t
# ---------------------------------------------------------------------------

def baseline_naive_path(t_grid, X, K_prior_t, sigma_noise=0.1, jitter=1e-8):
    """
    Naive GP regression: treat X_ti as noisy observations of mu(t).

    Model (WRONG for SDEs):
        X_ti ~ N(mu(t_i), sigma_noise^2)

    This ignores:
    - The cumulative nature of X_t (it integrates mu, not evaluates it)
    - The Brownian motion covariance structure
    - The heteroskedastic noise from discretization

    Parameters
    ----------
    t_grid      : np.ndarray (n+1,)
    X           : np.ndarray (n+1,)   observed path values
    K_prior_t   : np.ndarray (n,n)    kernel matrix at t_grid[:-1]
    sigma_noise : float               fixed isotropic noise (tuned by MLE)
    jitter      : float

    Returns
    -------
    mu_post  : np.ndarray (n,)
    var_post : np.ndarray (n,)
    """
    n  = len(t_grid) - 1
    y  = X[:-1]                                  # use X values (not increments)
    noise_var = sigma_noise ** 2

    A     = K_prior_t + noise_var * np.eye(n) + jitter * np.eye(n)
    L     = np.linalg.cholesky(A)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    V     = np.linalg.solve(L, K_prior_t)

    mu_post  = K_prior_t @ alpha
    var_post = np.clip(np.diag(K_prior_t - V.T @ V), 0.0, None)
    return mu_post, var_post


# ---------------------------------------------------------------------------
# Baseline 2 — GP on finite differences, homoskedastic noise
# ---------------------------------------------------------------------------

def _neg_log_marginal_likelihood(log_sigma_noise, y, K, jitter=1e-8):
    """Negative log marginal likelihood for homoskedastic GP."""
    sigma_noise = np.exp(log_sigma_noise)
    n   = len(y)
    A   = K + sigma_noise**2 * np.eye(n) + jitter * np.eye(n)
    try:
        L   = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        return 1e10
    alpha   = np.linalg.solve(L.T, np.linalg.solve(L, y))
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return 0.5 * (y @ alpha + log_det + n * np.log(2 * np.pi))


def baseline_homoskedastic_fd(t_grid, dX, K_prior, jitter=1e-8):
    """
    GP on finite differences with homoskedastic noise fit by MLE.

    Model (partially correct):
        DX_i / dt_i ~ N(mu(t_i), sigma_fixed^2)

    This gets the observation type right (finite differences) but
    uses a single scalar noise variance instead of the correct
    heteroskedastic sigma^2/dt_i. The scalar sigma_fixed is fit
    by maximising the GP marginal likelihood.

    Why it's weaker:
        The correct noise is sigma^2/dt_i — it's not a free parameter,
        it's derived from the SDE physics. Fitting it by MLE gives a
        single compromise value that's wrong for all grid points.

    Parameters
    ----------
    t_grid   : np.ndarray (n+1,)
    dX       : np.ndarray (n,)
    K_prior  : np.ndarray (n,n)
    jitter   : float

    Returns
    -------
    mu_post       : np.ndarray (n,)
    var_post      : np.ndarray (n,)
    sigma_fit     : float           MLE noise estimate
    """
    dt = np.diff(t_grid)
    y  = dX / dt

    # Fit scalar noise by MLE over log(sigma) in [-5, 3]
    result = minimize_scalar(
        _neg_log_marginal_likelihood,
        bounds=(-5, 3),
        method='bounded',
        args=(y, K_prior, jitter),
    )
    sigma_fit = float(np.exp(result.x))

    n  = len(y)
    A  = K_prior + sigma_fit**2 * np.eye(n) + jitter * np.eye(n)
    L  = np.linalg.cholesky(A)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    V     = np.linalg.solve(L, K_prior)

    mu_post  = K_prior @ alpha
    var_post = np.clip(np.diag(K_prior - V.T @ V), 0.0, None)
    return mu_post, var_post, sigma_fit


# ---------------------------------------------------------------------------
# Comparison experiment
# ---------------------------------------------------------------------------

def comparison_experiment(mu_star_fn, sigma_values, T, dt,
                           kernel_fn, kernel_kwargs,
                           n_paths=30, seed=0):
    """
    Compare all three methods across sigma values.

    Parameters
    ----------
    mu_star_fn    : callable
    sigma_values  : list of float
    T             : float
    dt            : float
    kernel_fn     : callable
    kernel_kwargs : dict
    n_paths       : int
    seed          : int

    Returns
    -------
    dict with keys: sigma_values,
                    ours_{mean,std}, bl1_{mean,std}, bl2_{mean,std}
                    ours_cov, bl1_cov, bl2_cov
    """
    rng    = np.random.default_rng(seed)
    n      = int(round(T / dt))
    t_grid = np.linspace(0, T, n + 1)
    t_left = t_grid[:-1]
    mu_star = mu_star_fn(t_left)
    K = build_kernel_matrix(t_left, kernel_fn, **kernel_kwargs)

    results = {k: [] for k in [
        'ours_mean', 'ours_std', 'ours_cov',
        'bl1_mean',  'bl1_std',  'bl1_cov',
        'bl2_mean',  'bl2_std',  'bl2_cov',
    ]}

    for sigma in sigma_values:
        ours_e, bl1_e, bl2_e   = [], [], []
        ours_cv, bl1_cv, bl2_cv = [], [], []

        for _ in range(n_paths):
            seed_i = int(rng.integers(0, 2**31))
            rng_i  = np.random.default_rng(seed_i)

            # Simulate path
            dW = rng_i.normal(0, sigma * np.sqrt(dt), n)
            dX = mu_star * dt + dW
            X  = np.concatenate([[0.0], np.cumsum(dX)])

            # --- Our method ---
            mu_o, var_o, _ = gp_posterior_drift(
                t_grid, dX, K, sigma_obs=sigma)
            ci_lo_o = mu_o - 1.96 * np.sqrt(var_o)
            ci_hi_o = mu_o + 1.96 * np.sqrt(var_o)
            ours_e.append(float(np.sqrt(np.mean((mu_o - mu_star)**2))))
            ours_cv.append(float(np.mean(
                (mu_star >= ci_lo_o) & (mu_star <= ci_hi_o))))

            # --- Baseline 1: naive GP on X values ---
            # Tune sigma_noise by a rough heuristic: std of X
            sigma_b1 = float(np.std(X[:-1])) + 1e-4
            mu_b1, var_b1 = baseline_naive_path(
                t_grid, X, K, sigma_noise=sigma_b1)
            ci_lo_b1 = mu_b1 - 1.96 * np.sqrt(var_b1)
            ci_hi_b1 = mu_b1 + 1.96 * np.sqrt(var_b1)
            bl1_e.append(float(np.sqrt(np.mean((mu_b1 - mu_star)**2))))
            bl1_cv.append(float(np.mean(
                (mu_star >= ci_lo_b1) & (mu_star <= ci_hi_b1))))

            # --- Baseline 2: homoskedastic finite differences ---
            mu_b2, var_b2, _ = baseline_homoskedastic_fd(
                t_grid, dX, K)
            ci_lo_b2 = mu_b2 - 1.96 * np.sqrt(var_b2)
            ci_hi_b2 = mu_b2 + 1.96 * np.sqrt(var_b2)
            bl2_e.append(float(np.sqrt(np.mean((mu_b2 - mu_star)**2))))
            bl2_cv.append(float(np.mean(
                (mu_star >= ci_lo_b2) & (mu_star <= ci_hi_b2))))

        results['ours_mean'].append(float(np.mean(ours_e)))
        results['ours_std'].append(float(np.std(ours_e)))
        results['ours_cov'].append(float(np.mean(ours_cv)))
        results['bl1_mean'].append(float(np.mean(bl1_e)))
        results['bl1_std'].append(float(np.std(bl1_e)))
        results['bl1_cov'].append(float(np.mean(bl1_cv)))
        results['bl2_mean'].append(float(np.mean(bl2_e)))
        results['bl2_std'].append(float(np.std(bl2_e)))
        results['bl2_cov'].append(float(np.mean(bl2_cv)))

        print(f"  sigma={sigma:.3f} | "
              f"Ours: RMS={np.mean(ours_e):.4f} Cov={np.mean(ours_cv):.2f} | "
              f"BL1:  RMS={np.mean(bl1_e):.4f} Cov={np.mean(bl1_cv):.2f} | "
              f"BL2:  RMS={np.mean(bl2_e):.4f} Cov={np.mean(bl2_cv):.2f}")

    for k in results:
        results[k] = np.array(results[k])
    results['sigma_values'] = sigma_values
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import os
    os.makedirs("plots", exist_ok=True)

    mu_star_fn = lambda t: np.sin(2 * np.pi * t)

    # Two regimes: high noise (prior dominates) and low noise (data dominates)
    sigma_values = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05]

    print("=" * 70)
    print("Experiment 4: Our method vs baselines")
    print("Columns: sigma | Ours RMS/Coverage | BL1 RMS/Cov | BL2 RMS/Cov")
    print("=" * 70)

    results = comparison_experiment(
        mu_star_fn,
        sigma_values,
        T=1.0,
        dt=0.005,
        kernel_fn=matern_kernel,
        kernel_kwargs={"nu": 1.5, "length_scale": 0.3, "variance": 1.0},
        n_paths=30,
        seed=42,
    )

    sigma_arr = np.array(sigma_values)

    # ------------------------------------------------------------------
    # Figure 1 — RMS comparison across sigma
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.errorbar(sigma_arr, results['ours_mean'], yerr=results['ours_std'],
                fmt='o-', capsize=4, color='steelblue',
                label='Ours (Girsanov heteroskedastic)', zorder=3)
    ax.errorbar(sigma_arr, results['bl1_mean'], yerr=results['bl1_std'],
                fmt='s--', capsize=4, color='tomato',
                label='BL1: Naive GP on X_t', zorder=2)
    ax.errorbar(sigma_arr, results['bl2_mean'], yerr=results['bl2_std'],
                fmt='^:', capsize=4, color='darkorange',
                label='BL2: Homoskedastic FD', zorder=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Noise level sigma")
    ax.set_ylabel("RMS error")
    ax.set_title("RMS Error vs Sigma\n(lower is better)")
    ax.legend(fontsize=9)
    ax.invert_xaxis()   # left = low noise = high information

    # Panel 2 — Coverage
    ax2 = axes[1]
    ax2.plot(sigma_arr, results['ours_cov'], 'o-', color='steelblue',
             label='Ours', zorder=3)
    ax2.plot(sigma_arr, results['bl1_cov'],  's--', color='tomato',
             label='BL1: Naive GP on X_t', zorder=2)
    ax2.plot(sigma_arr, results['bl2_cov'],  '^:', color='darkorange',
             label='BL2: Homoskedastic FD', zorder=1)
    ax2.axhline(0.95, color='black', linestyle=':', lw=1.5,
                label='Nominal 95%')
    ax2.set_xscale('log')
    ax2.set_xlabel("Noise level sigma")
    ax2.set_ylabel("95% CI Coverage")
    ax2.set_title("Posterior Coverage vs Sigma\n(target: 0.95)")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9)
    ax2.invert_xaxis()

    plt.suptitle("Experiment 4: Our Method vs Baselines", fontsize=13)
    plt.tight_layout()
    plt.savefig("plots/baseline_comparison.png", dpi=150, bbox_inches='tight')
    print("\nSaved: plots/baseline_comparison.png")

    # ------------------------------------------------------------------
    # Figure 2 — Visual comparison on one path at low sigma
    # ------------------------------------------------------------------
    sigma_demo = 0.1
    T_demo, dt_demo = 1.0, 0.005
    n_demo = int(T_demo / dt_demo)
    t_grid_d = np.linspace(0, T_demo, n_demo + 1)
    t_left_d = t_grid_d[:-1]
    mu_star_d = mu_star_fn(t_left_d)

    rng_d = np.random.default_rng(123)
    dW_d  = rng_d.normal(0, sigma_demo * np.sqrt(dt_demo), n_demo)
    dX_d  = mu_star_d * dt_demo + dW_d
    X_d   = np.concatenate([[0.0], np.cumsum(dX_d)])

    K_d = build_kernel_matrix(t_left_d, matern_kernel,
                               nu=1.5, length_scale=0.3, variance=1.0)

    mu_o, var_o, _  = gp_posterior_drift(t_grid_d, dX_d, K_d,
                                          sigma_obs=sigma_demo)
    mu_b1, var_b1   = baseline_naive_path(t_grid_d, X_d, K_d,
                                           sigma_noise=float(np.std(X_d[:-1]))+1e-4)
    mu_b2, var_b2, sigma_b2_fit = baseline_homoskedastic_fd(t_grid_d, dX_d, K_d)

    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    method_data = [
        (mu_o,  var_o,  "Ours (Girsanov)",         'steelblue'),
        (mu_b1, var_b1, "BL1: Naive GP on X_t",    'tomato'),
        (mu_b2, var_b2, f"BL2: Homoskedastic FD\n(sigma_fit={sigma_b2_fit:.3f})", 'darkorange'),
    ]

    for ax3, (mu_m, var_m, title, col) in zip(axes2, method_data):
        ci_lo = mu_m - 1.96 * np.sqrt(var_m)
        ci_hi = mu_m + 1.96 * np.sqrt(var_m)
        rms   = np.sqrt(np.mean((mu_m - mu_star_d)**2))
        cov   = np.mean((mu_star_d >= ci_lo) & (mu_star_d <= ci_hi))

        ax3.fill_between(t_left_d, ci_lo, ci_hi,
                         alpha=0.25, color=col, label='95% CI')
        ax3.plot(t_left_d, mu_m,      color=col, lw=2.0, label='Posterior mean')
        ax3.plot(t_left_d, mu_star_d, 'r--',     lw=1.5, label='True mu*')
        ax3.set_title(f"{title}\nRMS={rms:.4f}  Coverage={cov:.2f}")
        ax3.set_xlabel("t")
        ax3.set_ylabel("mu(t)")
        ax3.legend(fontsize=8)

    plt.suptitle(f"Visual Comparison at sigma={sigma_demo}", fontsize=13)
    plt.tight_layout()
    plt.savefig("plots/baseline_visual.png", dpi=150, bbox_inches='tight')
    print("Saved: plots/baseline_visual.png")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print(f"{'sigma':>8} | {'Ours RMS':>10} {'BL1 RMS':>10} {'BL2 RMS':>10} |"
          f" {'Ours Cov':>10} {'BL1 Cov':>10} {'BL2 Cov':>10}")
    print("-" * 70)
    for i, sigma in enumerate(sigma_values):
        print(f"  {sigma:6.3f} | "
              f"{results['ours_mean'][i]:10.4f} "
              f"{results['bl1_mean'][i]:10.4f} "
              f"{results['bl2_mean'][i]:10.4f} | "
              f"{results['ours_cov'][i]:10.3f} "
              f"{results['bl1_cov'][i]:10.3f} "
              f"{results['bl2_cov'][i]:10.3f}")
    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()
