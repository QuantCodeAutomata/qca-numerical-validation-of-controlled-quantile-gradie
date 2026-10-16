"""Tests for src/surface_measure.py.

Validates:
- w_y_func: correct values, boundary handling
- log_gy_func: shape, sign, vectorized vs scalar agreement
- quadrature_moments_2d: normalization, positivity, variance positivity
- eq_327_curvature_term: sign behavior in known regimes
- kde_density: plausible output for simple distributions
"""

import numpy as np
import pytest
from scipy.stats import norm

from src.distributions import get_scenarios, build_scenario
from src.surface_measure import (
    w_y_func,
    log_gy_func,
    quadrature_moments_2d,
    eq_327_curvature_term,
    kde_density,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scenarios():
    return get_scenarios()


@pytest.fixture(scope="module")
def sc_s1(scenarios):
    return scenarios["S1"]


@pytest.fixture(scope="module")
def sc_s3(scenarios):
    return scenarios["S3"]


# ---------------------------------------------------------------------------
# w_y_func
# ---------------------------------------------------------------------------


def test_wy_finite_at_interior(sc_s1):
    """w_y should be finite at an interior point."""
    u = np.array([0.5, 0.5])
    y = 2.0
    x1 = 0.5 * (1.0 / u[0])  # midpoint of integration domain
    val = w_y_func(x1, u, y, sc_s1)
    assert np.isfinite(val)


def test_wy_inf_at_boundary(sc_s1):
    """w_y should be +inf when x1 = 0 (x not in positive orthant)."""
    u = np.array([0.5, 0.5])
    y = 1.0
    val = w_y_func(0.0, u, y, sc_s1)
    assert val == np.inf


def test_wy_inf_at_x2_zero(sc_s1):
    """w_y should be +inf when x1 = 1/u1 (x2 = 0)."""
    u = np.array([0.5, 0.5])
    y = 1.0
    val = w_y_func(1.0 / u[0], u, y, sc_s1)
    assert val == np.inf


def test_wy_positive_y_effect(sc_s1):
    """Larger y shifts w_y depending on sign of a(x)."""
    u = np.array([0.5, 0.5])
    x1 = 0.7
    w_small = w_y_func(x1, u, 0.5, sc_s1)
    w_large = w_y_func(x1, u, 5.0, sc_s1)
    # They should differ (y appears via log(y)*a(x))
    assert w_small != w_large


# ---------------------------------------------------------------------------
# log_gy_func
# ---------------------------------------------------------------------------


def test_log_gy_shape(sc_s1):
    u = np.array([0.6, 0.4])
    x1_arr = np.linspace(1e-5, 1.0 / u[0] - 1e-5, 50)
    result = log_gy_func(x1_arr, u, 1.0, sc_s1)
    assert result.shape == (50,)


def test_log_gy_all_finite_interior(sc_s1):
    """All interior x1 values should give finite log g_y."""
    u = np.array([0.5, 0.5])
    x1_arr = np.linspace(0.05, 1.95, 100)  # within (0, 1/u1)=2
    result = log_gy_func(x1_arr, u, 1.0, sc_s1)
    assert np.all(np.isfinite(result))


def test_log_gy_minus_inf_at_boundary(sc_s1):
    """x1 = 0 should give -inf in log g_y (invalid point)."""
    u = np.array([0.5, 0.5])
    x1_arr = np.array([0.0, 0.5, 1.0])
    result = log_gy_func(x1_arr, u, 1.0, sc_s1)
    assert result[0] == -np.inf


def test_log_gy_matches_wy(sc_s1):
    """log g_y = -w_y should hold at interior points."""
    u = np.array([0.5, 0.5])
    y = 2.0
    x1_arr = np.array([0.5, 1.0, 1.5])  # interior of (0, 2)
    log_gy = log_gy_func(x1_arr, u, y, sc_s1)
    for i, x1 in enumerate(x1_arr):
        w = w_y_func(x1, u, y, sc_s1)
        assert abs(log_gy[i] - (-w)) < 1e-12


# ---------------------------------------------------------------------------
# quadrature_moments_2d
# ---------------------------------------------------------------------------


def test_quad_moments_returns_dict(sc_s1):
    u = np.array([0.5, 0.5])
    result = quadrature_moments_2d(u, 2.0, sc_s1, n_quad=100)
    assert isinstance(result, dict)
    for key in ["E_x1", "E_x2", "Var_gamma", "E_a", "Z_y_unnorm"]:
        assert key in result


def test_quad_moments_E_x_positive(sc_s1):
    """Conditional expectations of x_i should be positive (lognormal support)."""
    u = np.array([0.4, 0.6])
    result = quadrature_moments_2d(u, 1.5, sc_s1, n_quad=200)
    assert result["E_x1"] > 0
    assert result["E_x2"] > 0


def test_quad_moments_Var_gamma_nonneg(sc_s1):
    """Variance should be non-negative."""
    u = np.array([0.6, 0.4])
    result = quadrature_moments_2d(u, 1.0, sc_s1, n_quad=200)
    assert result["Var_gamma"] >= 0


def test_quad_moments_Z_y_positive(sc_s1):
    """Partition function should be positive (non-trivial measure)."""
    u = np.array([0.5, 0.5])
    result = quadrature_moments_2d(u, 1.0, sc_s1, n_quad=200)
    assert result["Z_y_unnorm"] > 0


def test_quad_moments_constraint_satisfied(sc_s1):
    """E[u^T x] under mu_y should equal 1 (by construction of S_1)."""
    u = np.array([0.5, 0.5])
    result = quadrature_moments_2d(u, 1.0, sc_s1, n_quad=500)
    u_dot_Ex = u[0] * result["E_x1"] + u[1] * result["E_x2"]
    # On S_1: u^T x = 1 exactly, so the expectation under mu_y is 1
    assert abs(u_dot_Ex - 1.0) < 0.01


def test_quad_moments_different_t(sc_s1):
    """Moments change meaningfully with different t values."""
    r1 = quadrature_moments_2d(np.array([0.2, 0.8]), 1.0, sc_s1, n_quad=200)
    r2 = quadrature_moments_2d(np.array([0.8, 0.2]), 1.0, sc_s1, n_quad=200)
    # E_x1 and E_x2 should swap approximately
    assert r1["E_x1"] != r2["E_x1"]


def test_quad_moments_convergence(sc_s1):
    """Increasing n_quad should give converging E_x1."""
    u = np.array([0.5, 0.5])
    r100 = quadrature_moments_2d(u, 2.0, sc_s1, n_quad=100)
    r500 = quadrature_moments_2d(u, 2.0, sc_s1, n_quad=500)
    assert abs(r100["E_x1"] - r500["E_x1"]) < 0.05


# ---------------------------------------------------------------------------
# eq_327_curvature_term
# ---------------------------------------------------------------------------


def test_eq327_returns_finite(sc_s1):
    u = np.array([0.5, 0.5])
    y = 1.5
    sum_Bmu = float(np.sum(sc_s1.B @ sc_s1.mu))
    moments = quadrature_moments_2d(u, y, sc_s1, n_quad=200)
    T_y = eq_327_curvature_term(u, y, sc_s1, moments, sum_Bmu)
    assert np.isfinite(T_y)


def test_eq327_curvature_sign_s1_left_tail():
    """For S1 interior scenario at small y (left tail), T_y > 0 indicates
    d^2q/dt^2 = -T_y < 0 (concavity), consistent with paper's claim."""
    sc = get_scenarios()["S1"]
    u = np.array([0.5, 0.5])
    y = 0.05  # deep left tail
    sum_Bmu = float(np.sum(sc.B @ sc.mu))
    moments = quadrature_moments_2d(u, y, sc, n_quad=400)
    T_y = eq_327_curvature_term(u, y, sc, moments, sum_Bmu)
    # We expect T_y > 0 for left tail in interior regime (d^2q/dt^2 = -T_y < 0)
    assert T_y > 0, f"Expected T_y > 0 for left-tail concavity in S1, got {T_y}"


# ---------------------------------------------------------------------------
# kde_density
# ---------------------------------------------------------------------------


def test_kde_density_positive():
    """KDE estimate should be positive at the sample center."""
    Y = np.random.default_rng(42).standard_normal(5000)
    d = kde_density(Y, 0.0)
    assert d > 0


def test_kde_density_peak_near_mean():
    """KDE of standard normal should have higher density at 0 than at 3."""
    Y = np.random.default_rng(5).standard_normal(10_000)
    d0 = kde_density(Y, 0.0)
    d3 = kde_density(Y, 3.0)
    assert d0 > d3


def test_kde_density_float_output():
    Y = np.linspace(0, 10, 1000)
    d = kde_density(Y, 5.0)
    assert isinstance(d, float)
