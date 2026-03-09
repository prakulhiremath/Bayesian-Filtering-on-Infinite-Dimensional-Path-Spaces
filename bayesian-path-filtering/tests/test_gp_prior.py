"""
test_gp_prior.py — Unit tests for GP prior utilities.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulations'))

from utils import matern_kernel, build_kernel_matrix, squared_exp_kernel
from gp_prior import sample_gp, rkhs_norm, empirical_small_ball_probability


class TestMaternKernel:
    def test_positive_definite(self):
        """Kernel matrix should be positive definite."""
        t = np.linspace(0, 1, 20)
        for nu in [0.5, 1.5, 2.5]:
            K = build_kernel_matrix(t, matern_kernel, nu=nu, length_scale=0.3)
            eigvals = np.linalg.eigvalsh(K)
            assert np.all(eigvals > 0), f"K not positive definite for nu={nu}"

    def test_symmetry(self):
        """Kernel matrix should be symmetric."""
        t = np.linspace(0, 1, 15)
        K = build_kernel_matrix(t, matern_kernel, nu=1.5, length_scale=0.3)
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_diagonal_is_variance(self):
        """Diagonal of kernel matrix equals the variance parameter."""
        t = np.linspace(0, 1, 10)
        variance = 2.0
        K = build_kernel_matrix(t, matern_kernel, nu=1.5, length_scale=0.3,
                                 variance=variance, jitter=0)
        np.testing.assert_allclose(np.diag(K), variance, atol=1e-8)

    def test_self_covariance(self):
        """k(t, t) = variance for all t."""
        t_vals = np.array([0.0, 0.3, 0.7, 1.0])
        for nu in [0.5, 1.5, 2.5]:
            k_self = matern_kernel(t_vals, t_vals, nu=nu)
            np.testing.assert_allclose(k_self, 1.0, atol=1e-10)

    def test_decreasing_with_distance(self):
        """Kernel should decay as |s - t| increases."""
        t0 = 0.5
        distances = np.array([0.0, 0.1, 0.3, 0.5, 1.0])
        k_vals = matern_kernel(t0, t0 + distances, nu=1.5)
        assert np.all(np.diff(k_vals) <= 0), "Kernel not monotone decreasing"


class TestGPSampling:
    def test_sample_shape(self):
        t = np.linspace(0, 1, 50)
        K = build_kernel_matrix(t, matern_kernel, nu=1.5, length_scale=0.3)
        samples = sample_gp(t, K, n_samples=10, seed=0)
        assert samples.shape == (10, 50)

    def test_zero_mean(self):
        """Sample mean should be approximately zero."""
        t = np.linspace(0, 1, 100)
        K = build_kernel_matrix(t, matern_kernel, nu=1.5, length_scale=0.3)
        samples = sample_gp(t, K, n_samples=500, seed=42)
        sample_mean = samples.mean(axis=0)
        np.testing.assert_allclose(sample_mean, 0, atol=0.2)

    def test_covariance_structure(self):
        """Sample covariance should match kernel matrix."""
        t = np.linspace(0, 1, 20)
        K = build_kernel_matrix(t, matern_kernel, nu=1.5, length_scale=0.3, jitter=0)
        samples = sample_gp(t, K, n_samples=5000, seed=0)
        sample_cov = np.cov(samples.T)
        # Allow generous tolerance for Monte Carlo
        np.testing.assert_allclose(sample_cov, K, atol=0.3)


class TestRKHSNorm:
    def test_positive(self):
        """RKHS norm should be positive for non-zero function."""
        t = np.linspace(0, 1, 30)
        K = build_kernel_matrix(t, matern_kernel, nu=1.5, length_scale=0.3)
        K_inv = np.linalg.inv(K)
        f = np.sin(2 * np.pi * t)
        norm = rkhs_norm(f, K_inv)
        assert norm > 0

    def test_zero_function(self):
        """RKHS norm of zero function should be zero."""
        t = np.linspace(0, 1, 20)
        K = build_kernel_matrix(t, matern_kernel, nu=1.5, length_scale=0.3)
        K_inv = np.linalg.inv(K)
        f = np.zeros(len(t))
        norm = rkhs_norm(f, K_inv)
        np.testing.assert_allclose(norm, 0, atol=1e-10)


class TestSmallBallProbability:
    def test_decreasing(self):
        """Small-ball probability should increase with ε."""
        t = np.linspace(0, 1, 50)
        K = build_kernel_matrix(t, matern_kernel, nu=1.5, length_scale=0.3)
        eps_vals = np.array([0.1, 0.3, 0.5, 1.0])
        probs = empirical_small_ball_probability(t, K, eps_vals, n_monte_carlo=500, seed=0)
        assert np.all(np.diff(probs) >= 0), "Probabilities should be non-decreasing in ε"

    def test_all_within_unit_interval(self):
        """Probabilities should be in [0, 1]."""
        t = np.linspace(0, 1, 40)
        K = build_kernel_matrix(t, matern_kernel, nu=1.5, length_scale=0.3)
        eps_vals = np.linspace(0.1, 2.0, 10)
        probs = empirical_small_ball_probability(t, K, eps_vals, n_monte_carlo=200, seed=1)
        assert np.all(probs >= 0) and np.all(probs <= 1)
