"""Tests for src/quantiles.py.

Validates:
- portfolio_returns: correctness for boundary weights
- quantile_curve: monotonicity, boundary behavior
- finite_diff_gradient_1d / finite_diff_hessian_1d: linear/quadratic cases
- conditional_mean_narrow_band: correct conditioning
- adaptive_bandwidth: positive output, scales with IQR
- check_homogeneity: q_alpha(c*u) = c * q_alpha(u)
"""

import numpy as np
import pytest

from src.quantiles import (
    portfolio_returns,
    portfolio_returns_grid,
    empirical_quantile,
    quantile_curve,
    finite_diff_gradient_1d,
    finite_diff_hessian_1d,
    conditional_mean_narrow_band,
    adaptive_bandwidth,
    check_homogeneity,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def simple_X():
    """Small synthetic X for deterministic checks."""
    rng = np.random.default_rng(0)
    return np.exp(rng.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=50_000))


# ---------------------------------------------------------------------------
# portfolio_returns
# ---------------------------------------------------------------------------


def test_portfolio_returns_t1(simple_X):
    """t=1 => Y = X[:,0]."""
    Y = portfolio_returns(simple_X, 1.0)
    np.testing.assert_array_equal(Y, simple_X[:, 0])


def test_portfolio_returns_t0(simple_X):
    """t=0 => Y = X[:,1]."""
    Y = portfolio_returns(simple_X, 0.0)
    np.testing.assert_array_equal(Y, simple_X[:, 1])


def test_portfolio_returns_equal_weights(simple_X):
    """t=0.5 => Y = 0.5*(X[:,0] + X[:,1])."""
    Y = portfolio_returns(simple_X, 0.5)
    np.testing.assert_allclose(Y, 0.5 * (simple_X[:, 0] + simple_X[:, 1]))


def test_portfolio_returns_shape(simple_X):
    Y = portfolio_returns(simple_X, 0.3)
    assert Y.shape == (simple_X.shape[0],)


# ---------------------------------------------------------------------------
# portfolio_returns_grid
# ---------------------------------------------------------------------------


def test_portfolio_returns_grid_shape(simple_X):
    t_grid = np.linspace(0, 1, 11)
    Y_grid = portfolio_returns_grid(simple_X, t_grid)
    assert Y_grid.shape == (11, simple_X.shape[0])


def test_portfolio_returns_grid_boundary_rows(simple_X):
    t_grid = np.array([0.0, 1.0])
    Y_grid = portfolio_returns_grid(simple_X, t_grid)
    np.testing.assert_allclose(Y_grid[1], simple_X[:, 0])
    np.testing.assert_allclose(Y_grid[0], simple_X[:, 1])


# ---------------------------------------------------------------------------
# empirical_quantile
# ---------------------------------------------------------------------------


def test_empirical_quantile_median():
    Y = np.arange(1001, dtype=float)
    q = empirical_quantile(Y, 0.5)
    assert abs(q - 500.0) < 1.0


def test_empirical_quantile_extreme_low():
    Y = np.arange(100, dtype=float)
    q = empirical_quantile(Y, 0.01)
    assert 0 <= q <= 2


def test_empirical_quantile_extreme_high():
    Y = np.arange(100, dtype=float)
    q = empirical_quantile(Y, 0.99)
    assert q >= 97


# ---------------------------------------------------------------------------
# quantile_curve
# ---------------------------------------------------------------------------


def test_quantile_curve_shape(simple_X):
    t_grid = np.linspace(0.1, 0.9, 17)
    q = quantile_curve(simple_X, t_grid, 0.1)
    assert q.shape == (17,)


def test_quantile_curve_positive_lognormal(simple_X):
    """All quantiles should be positive for lognormal samples."""
    t_grid = np.linspace(0.05, 0.95, 10)
    q = quantile_curve(simple_X, t_grid, 0.05)
    assert np.all(q > 0)


def test_quantile_curve_monotone_alpha(simple_X):
    """Higher alpha => higher quantile at every t."""
    t_grid = np.linspace(0.1, 0.9, 9)
    q_lo = quantile_curve(simple_X, t_grid, 0.10)
    q_hi = quantile_curve(simple_X, t_grid, 0.90)
    assert np.all(q_hi > q_lo)


def test_quantile_curve_boundary_t1(simple_X):
    """q_alpha at t=1 ≈ empirical_quantile(X[:,0], alpha)."""
    alpha = 0.15
    t_grid = np.array([1.0])
    q = quantile_curve(simple_X, t_grid, alpha)
    q_direct = empirical_quantile(simple_X[:, 0], alpha)
    assert abs(q[0] - q_direct) < 1e-10


# ---------------------------------------------------------------------------
# finite_diff_gradient_1d
# ---------------------------------------------------------------------------


def test_fd_gradient_linear():
    """For q = a*t + b, gradient should be constant = a."""
    t = np.linspace(0, 1, 50)
    q = 3.0 * t + 1.0
    grad = finite_diff_gradient_1d(q, t)
    # Central difference is exact for linear functions
    np.testing.assert_allclose(grad[1:-1], 3.0, atol=1e-8)


def test_fd_gradient_shape():
    t = np.linspace(0, 1, 30)
    q = np.sin(t)
    grad = finite_diff_gradient_1d(q, t)
    assert grad.shape == (30,)


def test_fd_gradient_quadratic():
    """For q = t^2, gradient ≈ 2t at interior points."""
    t = np.linspace(0, 1, 200)
    q = t ** 2
    grad = finite_diff_gradient_1d(q, t)
    np.testing.assert_allclose(grad[10:-10], 2.0 * t[10:-10], atol=1e-4)


# ---------------------------------------------------------------------------
# finite_diff_hessian_1d
# ---------------------------------------------------------------------------


def test_fd_hessian_quadratic():
    """For q = t^2, d^2q/dt^2 = 2 everywhere."""
    t = np.linspace(0, 1, 300)
    q = t ** 2
    hess = finite_diff_hessian_1d(q, t)
    # Interior points should be very close to 2
    np.testing.assert_allclose(hess[10:-10], 2.0, atol=1e-3)


def test_fd_hessian_linear():
    """For q = a*t, d^2q/dt^2 = 0."""
    t = np.linspace(0, 1, 100)
    q = 5.0 * t
    hess = finite_diff_hessian_1d(q, t)
    np.testing.assert_allclose(hess[5:-5], 0.0, atol=1e-8)


def test_fd_hessian_shape():
    t = np.linspace(0, 1, 50)
    q = np.cos(t)
    hess = finite_diff_hessian_1d(q, t)
    assert hess.shape == (50,)


def test_fd_hessian_cosine():
    """For q = cos(t), d^2q/dt^2 = -cos(t)."""
    t = np.linspace(0, np.pi, 500)
    q = np.cos(t)
    hess = finite_diff_hessian_1d(q, t)
    np.testing.assert_allclose(hess[20:-20], -np.cos(t[20:-20]), atol=1e-4)


# ---------------------------------------------------------------------------
# conditional_mean_narrow_band
# ---------------------------------------------------------------------------


def test_cond_mean_at_median():
    """For X = (Y, Y), E[X | Y ≈ q] = (q, q)."""
    Y = np.linspace(0.1, 5.0, 10_000)
    X = np.column_stack([Y, Y])
    q = float(np.median(Y))
    h = 0.1
    cond = conditional_mean_narrow_band(X, Y, q, h)
    assert abs(cond[0] - q) < 0.15
    assert abs(cond[1] - q) < 0.15


def test_cond_mean_error_too_narrow():
    """Raises ValueError when band is too narrow."""
    Y = np.arange(1000, dtype=float)
    X = np.column_stack([Y, Y])
    with pytest.raises(ValueError):
        conditional_mean_narrow_band(X, Y, 500.0, h_band=0.0)


def test_cond_mean_shape():
    """Should return (n_dim,) array."""
    rng = np.random.default_rng(42)
    Y = rng.standard_normal(5000)
    X = np.column_stack([Y, Y * 2])
    cond = conditional_mean_narrow_band(X, Y, 0.0, h_band=0.3)
    assert cond.shape == (2,)


# ---------------------------------------------------------------------------
# adaptive_bandwidth
# ---------------------------------------------------------------------------


def test_adaptive_bandwidth_positive():
    Y = np.random.default_rng(0).standard_normal(1000)
    h = adaptive_bandwidth(Y, 0.0)
    assert h > 0


def test_adaptive_bandwidth_scales_with_iqr():
    """Wider distribution -> larger bandwidth."""
    rng = np.random.default_rng(1)
    Y_narrow = rng.normal(0, 0.1, 1000)
    Y_wide = rng.normal(0, 10.0, 1000)
    h_narrow = adaptive_bandwidth(Y_narrow, 0.0)
    h_wide = adaptive_bandwidth(Y_wide, 0.0)
    assert h_wide > h_narrow


def test_adaptive_bandwidth_floor():
    """When IQR ≈ 0, falls back to floor based on q_alpha."""
    Y = np.full(1000, 5.0)   # zero IQR
    h = adaptive_bandwidth(Y, 5.0)
    assert h > 0


# ---------------------------------------------------------------------------
# check_homogeneity
# ---------------------------------------------------------------------------


def test_homogeneity_exact():
    """q_alpha(c*u) = c * q_alpha(u) analytically for lognormal."""
    rng = np.random.default_rng(5)
    X = np.exp(rng.multivariate_normal([0, 0], np.eye(2), size=50_000))
    u = np.array([0.4, 0.6])
    c = 3.0
    alpha = 0.10
    q_cu, c_q_u = check_homogeneity(X, u, alpha, c)
    assert abs(q_cu - c_q_u) / max(abs(c_q_u), 1e-10) < 1e-10


def test_homogeneity_c_one():
    """c=1 should return equal values trivially."""
    rng = np.random.default_rng(6)
    X = np.exp(rng.standard_normal((20_000, 2)))
    u = np.array([0.5, 0.5])
    q_cu, c_q_u = check_homogeneity(X, u, 0.5, c=1.0)
    assert abs(q_cu - c_q_u) < 1e-14


def test_homogeneity_returns_tuple():
    rng = np.random.default_rng(7)
    X = np.exp(rng.standard_normal((5_000, 2)))
    u = np.array([0.6, 0.4])
    result = check_homogeneity(X, u, 0.25, c=2.0)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)
