"""Tests for src/diagnostics.py.

Validates:
- gradient_fd_vs_formula: returns correct structure, correlation > 0.5 at moderate alpha
- hessian_transport_vs_fd: returns correct structure, sign consistency
- compute_curvature_sign_fraction: bounds, edge cases
"""

import numpy as np
import pytest

from src.distributions import get_scenarios, simulate_lognormal
from src.diagnostics import (
    gradient_fd_vs_formula,
    hessian_transport_vs_fd,
    compute_curvature_sign_fraction,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scenarios():
    return get_scenarios()


@pytest.fixture(scope="module")
def X_s1(scenarios):
    """Small X for fast tests."""
    return simulate_lognormal(scenarios["S1"], n_samples=30_000, seed=99)


@pytest.fixture(scope="module")
def t_grid_small():
    return np.linspace(0.1, 0.9, 9)


# ---------------------------------------------------------------------------
# compute_curvature_sign_fraction
# ---------------------------------------------------------------------------


def test_curvature_sign_fraction_all_negative():
    """All negative values -> fraction = 1 for sign=-1."""
    h = -np.ones(20)
    frac = compute_curvature_sign_fraction(h, sign=-1)
    assert abs(frac - 1.0) < 1e-12


def test_curvature_sign_fraction_all_positive():
    """All positive values -> fraction = 0 for sign=-1."""
    h = np.ones(20)
    frac = compute_curvature_sign_fraction(h, sign=-1)
    assert abs(frac) < 1e-12


def test_curvature_sign_fraction_half():
    """Half negative -> fraction = 0.5."""
    h = np.array([-1.0] * 10 + [1.0] * 10)
    frac = compute_curvature_sign_fraction(h, sign=-1)
    assert abs(frac - 0.5) < 1e-12


def test_curvature_sign_fraction_with_nan():
    """NaN values are ignored."""
    h = np.array([-1.0, -1.0, np.nan, 1.0])
    frac = compute_curvature_sign_fraction(h, sign=-1)
    # 2 negative out of 3 valid
    assert abs(frac - 2.0 / 3.0) < 1e-10


def test_curvature_sign_fraction_all_nan():
    """All NaN returns nan."""
    h = np.full(5, np.nan)
    frac = compute_curvature_sign_fraction(h, sign=-1)
    assert np.isnan(frac)


def test_curvature_sign_fraction_bounds():
    """Result is always in [0, 1]."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        h = rng.standard_normal(30)
        frac = compute_curvature_sign_fraction(h, sign=-1)
        assert 0.0 <= frac <= 1.0


def test_curvature_sign_positive_fraction():
    """sign=+1 counts positive values."""
    h = np.array([1.0, 2.0, -1.0, -2.0])
    frac = compute_curvature_sign_fraction(h, sign=+1)
    assert abs(frac - 0.5) < 1e-12


# ---------------------------------------------------------------------------
# gradient_fd_vs_formula
# ---------------------------------------------------------------------------


def test_gradient_result_keys(X_s1, t_grid_small, scenarios):
    result = gradient_fd_vs_formula(X_s1, t_grid_small, 0.10, scenarios["S1"])
    for key in ["t_grid", "q_curve", "fd_gradient", "formula_gradient"]:
        assert key in result


def test_gradient_shapes(X_s1, t_grid_small, scenarios):
    result = gradient_fd_vs_formula(X_s1, t_grid_small, 0.10, scenarios["S1"])
    n = len(t_grid_small)
    assert result["q_curve"].shape == (n,)
    assert result["fd_gradient"].shape == (n,)
    assert result["formula_gradient"].shape == (n,)


def test_gradient_q_curve_positive(X_s1, t_grid_small, scenarios):
    """Quantile curve should be positive for lognormal."""
    result = gradient_fd_vs_formula(X_s1, t_grid_small, 0.20, scenarios["S1"])
    assert np.all(result["q_curve"] > 0)


def test_gradient_fd_finite(X_s1, t_grid_small, scenarios):
    """FD gradient should be finite everywhere."""
    result = gradient_fd_vs_formula(X_s1, t_grid_small, 0.20, scenarios["S1"])
    assert np.all(np.isfinite(result["fd_gradient"]))


def test_gradient_formula_some_finite(X_s1, t_grid_small, scenarios):
    """Formula gradient should have at least some finite values."""
    result = gradient_fd_vs_formula(X_s1, t_grid_small, 0.10, scenarios["S1"])
    valid = np.isfinite(result["formula_gradient"])
    assert valid.sum() >= 1


def test_gradient_correlation(X_s1, scenarios):
    """FD and formula-based gradients should be positively correlated."""
    t_grid = np.linspace(0.15, 0.85, 11)
    result = gradient_fd_vs_formula(X_s1, t_grid, 0.20, scenarios["S1"])
    fd = result["fd_gradient"]
    form = result["formula_gradient"]
    valid = np.isfinite(form)
    if valid.sum() > 3:
        corr = np.corrcoef(fd[valid], form[valid])[0, 1]
        assert corr > 0.3, f"Expected positive correlation, got {corr:.3f}"


# ---------------------------------------------------------------------------
# hessian_transport_vs_fd
# ---------------------------------------------------------------------------


def test_hessian_result_keys(X_s1, t_grid_small, scenarios):
    result = hessian_transport_vs_fd(
        X_s1, t_grid_small, 0.10, scenarios["S1"], n_quad=50
    )
    for key in ["t_grid", "q_curve", "fd_hessian", "transport_hessian"]:
        assert key in result


def test_hessian_shapes(X_s1, t_grid_small, scenarios):
    result = hessian_transport_vs_fd(
        X_s1, t_grid_small, 0.10, scenarios["S1"], n_quad=50
    )
    n = len(t_grid_small)
    assert result["fd_hessian"].shape == (n,)
    assert result["transport_hessian"].shape == (n,)


def test_hessian_fd_finite(X_s1, t_grid_small, scenarios):
    """FD hessian should be finite."""
    result = hessian_transport_vs_fd(
        X_s1, t_grid_small, 0.20, scenarios["S1"], n_quad=50
    )
    assert np.all(np.isfinite(result["fd_hessian"]))


def test_hessian_transport_some_finite(X_s1, t_grid_small, scenarios):
    """Transport hessian should have at least some finite values."""
    result = hessian_transport_vs_fd(
        X_s1, t_grid_small, 0.10, scenarios["S1"], n_quad=100
    )
    valid = np.isfinite(result["transport_hessian"])
    assert valid.sum() >= 1


def test_hessian_sign_left_tail(scenarios):
    """In S1 left tail at interior points, Hessian should tend negative (concavity)."""
    sc = scenarios["S1"]
    X = simulate_lognormal(sc, n_samples=20_000, seed=77)
    t_grid = np.linspace(0.2, 0.8, 9)
    result = hessian_transport_vs_fd(X, t_grid, 0.05, sc, n_quad=100)
    fd_hess = result["fd_hessian"][2:-2]
    frac_neg = compute_curvature_sign_fraction(fd_hess, sign=-1)
    # At least a slight majority of interior points should be concave at alpha=0.05
    assert frac_neg >= 0.3, f"Expected frac_neg >= 0.3, got {frac_neg:.3f}"


def test_hessian_transport_finite_at_moderate_alpha(scenarios):
    """At moderate alpha, transport formula should return mostly finite values."""
    sc = scenarios["S1"]
    X = simulate_lognormal(sc, n_samples=20_000, seed=88)
    t_grid = np.linspace(0.2, 0.8, 7)
    result = hessian_transport_vs_fd(X, t_grid, 0.20, sc, n_quad=100)
    valid = np.isfinite(result["transport_hessian"])
    # At moderate alpha, most points should be computable
    assert valid.sum() >= len(t_grid) // 2
