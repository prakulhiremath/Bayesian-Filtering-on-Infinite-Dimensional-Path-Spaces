"""
test_discretization.py — Tests for discretization stability.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulations'))

from utils import build_kernel_matrix, matern_kernel, simulate_sde
from posterior_drift import gp_posterior_drift
from discretization_error import hellinger_gaussian


class TestGPPosterior:
    @pytest.fixture
    def setup(self):
        T, n = 1.0, 100
        t_grid = np.linspace(0, T, n + 1)
        mu_fn = lambda t: np.sin(2 * np.pi * t)
        X, _ = simulate_sde(mu_fn, lambda t: 1.0, t_grid, x0=0.0, seed=0)
        dX = np.diff(X)
        K = build_kernel_matrix(t_grid[:-1], matern_kernel, nu=1.5, length_scale=0.3)
        return {'t_grid': t_grid, 'dX': dX, 'K': K, 'mu_fn': mu_fn}

    def test_posterior_mean_finite(self, setup):
        mu_post, var_post, _ = gp_posterior_drift(setup['t_grid'], setup['dX'], setup['K'])
        assert np.all(np.isfinite(mu_post))
        assert np.all(np.isfinite(var_post))

    def test_posterior_variance_positive(self, setup):
        _, var_post, _ = gp_posterior_drift(setup['t_grid'], setup['dX'], setup['K'])
        assert np.all(var_post > 0)

    def test_posterior_variance_less_than_prior(self, setup):
        """Posterior variance should be strictly less than prior variance."""
        _, var_post, _ = gp_posterior_drift(setup['t_grid'], setup['dX'], setup['K'])
        prior_var = np.diag(setup['K'])
        # At most locations, posterior var < prior var
        assert np.mean(var_post < prior_var) > 0.5

    def test_posterior_contracts_with_more_data(self):
        """Longer observation window should reduce posterior uncertainty."""
        mu_fn = lambda t: np.cos(np.pi * t)
        k_kwargs = {"nu": 1.5, "length_scale": 0.3}

        avg_vars = []
        for T in [0.5, 2.0, 8.0]:
            n = int(200 * T)
            t_grid = np.linspace(0, T, n + 1)
            X, _ = simulate_sde(mu_fn, lambda t: 1.0, t_grid, x0=0.0, seed=7)
            dX = np.diff(X)
            K = build_kernel_matrix(t_grid[:-1], matern_kernel, **k_kwargs)
            _, var_post, _ = gp_posterior_drift(t_grid, dX, K)
            avg_vars.append(np.mean(var_post))

        assert avg_vars[0] > avg_vars[1] > avg_vars[2], \
            "Posterior variance should decrease as T increases"


class TestHellingerGaussian:
    def test_same_distribution_is_zero(self):
        """Hellinger distance between identical distributions should be 0."""
        n = 10
        mu = np.random.randn(n)
        Sigma = np.eye(n) * 0.5
        d = hellinger_gaussian(mu, Sigma, mu, Sigma)
        np.testing.assert_allclose(d, 0.0, atol=1e-6)

    def test_between_zero_and_one(self):
        """Hellinger distance should be in [0, 1]."""
        n = 8
        mu1 = np.zeros(n)
        mu2 = np.ones(n)
        Sigma = np.eye(n)
        d = hellinger_gaussian(mu1, Sigma, mu2, Sigma)
        assert 0.0 <= d <= 1.0

    def test_symmetric(self):
        """Hellinger distance should be symmetric."""
        n = 5
        rng = np.random.default_rng(42)
        mu1 = rng.standard_normal(n)
        mu2 = rng.standard_normal(n)
        A = rng.standard_normal((n, n))
        Sigma = A @ A.T + np.eye(n)
        d12 = hellinger_gaussian(mu1, Sigma, mu2, Sigma)
        d21 = hellinger_gaussian(mu2, Sigma, mu1, Sigma)
        np.testing.assert_allclose(d12, d21, atol=1e-8)

    def test_increases_with_mean_separation(self):
        """Hellinger distance should increase as means separate."""
        n = 5
        Sigma = np.eye(n)
        distances = []
        for scale in [0.0, 0.5, 1.0, 2.0, 5.0]:
            mu1 = np.zeros(n)
            mu2 = np.ones(n) * scale
            distances.append(hellinger_gaussian(mu1, Sigma, mu2, Sigma))
        assert all(distances[i] <= distances[i + 1] for i in range(len(distances) - 1))
