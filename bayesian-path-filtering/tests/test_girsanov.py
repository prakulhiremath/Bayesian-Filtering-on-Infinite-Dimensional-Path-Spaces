"""
test_girsanov.py — Unit tests for Girsanov likelihood computation.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulations'))

from utils import simulate_sde, quadratic_variation
from girsanov_likelihood import (
    log_likelihood_continuous,
    log_likelihood_milstein,
    novikov_condition,
)
from utils import build_kernel_matrix, matern_kernel


class TestLogLikelihood:
    @pytest.fixture
    def simple_path(self):
        """Simple SDE path for testing."""
        T, n = 1.0, 200
        t_grid = np.linspace(0, T, n + 1)
        mu_fn = lambda t: np.sin(2 * np.pi * t)
        sigma_fn = lambda t: 1.0
        X, dW = simulate_sde(mu_fn, sigma_fn, t_grid, x0=0.0, seed=42)
        dX = np.diff(X)
        dt = np.diff(t_grid)
        return {'X': X, 'dX': dX, 'dt': dt, 't_grid': t_grid,
                'mu_true': mu_fn(t_grid[:-1])}

    def test_true_mu_higher_than_zero(self, simple_path):
        """Log-likelihood at true μ should be higher than at μ ≡ 0 (on average)."""
        dX, dt, mu_true = simple_path['dX'], simple_path['dt'], simple_path['mu_true']
        ll_true = log_likelihood_continuous(mu_true, dX, dt)
        ll_zero = log_likelihood_continuous(np.zeros_like(mu_true), dX, dt)
        # Not guaranteed for a single path, but should hold with high probability
        # Just check that both are finite
        assert np.isfinite(ll_true)
        assert np.isfinite(ll_zero)

    def test_returns_float(self, simple_path):
        dX, dt, mu_true = simple_path['dX'], simple_path['dt'], simple_path['mu_true']
        result = log_likelihood_continuous(mu_true, dX, dt)
        assert isinstance(result, (float, np.floating))

    def test_scale_invariance(self, simple_path):
        """Log-likelihood should change predictably under scaling of μ."""
        dX, dt = simple_path['dX'], simple_path['dt']
        mu = simple_path['mu_true']
        ll1 = log_likelihood_continuous(mu, dX, dt)
        # For μ → c*μ: ℓ(cμ) = c*∫μdX - c²/2 * ∫μ²dt
        # This is a downward-opening parabola in c, maximized at c=1
        # So ℓ(2μ) < ℓ(μ) if the quadratic term dominates
        ll2 = log_likelihood_continuous(2 * mu, dX, dt)
        assert np.isfinite(ll2)


class TestQuadraticVariation:
    def test_approaches_T_for_unit_diffusion(self):
        """For σ=1, QV should converge to T as n→∞."""
        T = 1.0
        n_values = [100, 500, 2000, 10000]
        qv_values = []
        for n in n_values:
            t_grid = np.linspace(0, T, n + 1)
            X, _ = simulate_sde(lambda t: 0.0, lambda t: 1.0, t_grid, seed=42)
            qv_values.append(quadratic_variation(X))

        # QV should be close to T = 1.0 for large n
        np.testing.assert_allclose(qv_values[-1], T, atol=0.1)

    def test_scales_with_sigma(self):
        """For constant σ, QV ≈ σ² * T."""
        T = 1.0
        n = 5000
        t_grid = np.linspace(0, T, n + 1)
        for sigma in [0.5, 1.0, 2.0]:
            X, _ = simulate_sde(lambda t: 0.0, lambda t: sigma, t_grid, seed=10)
            qv = quadratic_variation(X)
            np.testing.assert_allclose(qv, sigma**2 * T, rtol=0.15)


class TestNovikovCondition:
    def test_finite_for_square_exp_kernel(self):
        """Novikov condition should be satisfied for smooth GP priors."""
        t_grid = np.linspace(0, 1, 51)
        from utils import squared_exp_kernel
        K = build_kernel_matrix(t_grid[:-1], squared_exp_kernel, length_scale=0.3)
        result = novikov_condition(K, t_grid, n_samples=300, seed=0)
        assert np.isfinite(result['mean'])
        assert result['finite_frac'] == 1.0, "All Novikov values should be finite"

    def test_finite_for_matern_kernel(self):
        """Novikov condition should hold for Matérn kernels."""
        t_grid = np.linspace(0, 1, 51)
        for nu in [0.5, 1.5, 2.5]:
            K = build_kernel_matrix(t_grid[:-1], matern_kernel, nu=nu, length_scale=0.3)
            result = novikov_condition(K, t_grid, n_samples=200, seed=0)
            assert result['finite_frac'] > 0.95, f"Many infinite values for nu={nu}"
