"""Tests for src/distributions.py.

Validates:
- Scenario construction and SPD checks
- Correct b, beta, regime classification, and t* values
- Lognormal simulation: positivity, shape, marginal distribution
- phi_X and a_func evaluate correctly
"""

import numpy as np
import pytest
from scipy import linalg

from src.distributions import (
    Scenario,
    build_scenario,
    get_scenarios,
    simulate_lognormal,
    phi_X,
    a_func,
    check_spd,
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
def sc_s2(scenarios):
    return scenarios["S2"]


@pytest.fixture(scope="module")
def sc_s3(scenarios):
    return scenarios["S3"]


# ---------------------------------------------------------------------------
# check_spd
# ---------------------------------------------------------------------------


def test_check_spd_identity():
    """Identity matrix is SPD."""
    assert check_spd(np.eye(2))


def test_check_spd_non_symmetric():
    """Non-symmetric matrix is not SPD."""
    assert not check_spd(np.array([[1.0, 2.0], [0.0, 1.0]]))


def test_check_spd_non_pd():
    """Singular/non-PD matrix fails."""
    assert not check_spd(np.array([[1.0, 1.0], [1.0, 1.0]]))


def test_check_spd_s1(sc_s1):
    assert check_spd(sc_s1.B)


def test_check_spd_s3(sc_s3):
    """S3 B is still SPD despite negative off-diagonal."""
    assert check_spd(sc_s3.B)


# ---------------------------------------------------------------------------
# Scenario construction: B values and b / beta
# ---------------------------------------------------------------------------


def test_s1_b_and_beta(sc_s1):
    """S1: b=(0.6,0.35), beta=0.95."""
    np.testing.assert_allclose(sc_s1.b, [0.6, 0.35], atol=1e-10)
    assert abs(sc_s1.beta - 0.95) < 1e-10


def test_s2_b_and_beta(sc_s2):
    """S2: b=(5.1,0.35), beta=5.45."""
    np.testing.assert_allclose(sc_s2.b, [5.1, 0.35], atol=1e-10)
    assert abs(sc_s2.beta - 5.45) < 1e-10


def test_s3_b_and_beta(sc_s3):
    """S3: b=(0.2,-0.05), beta=0.15."""
    np.testing.assert_allclose(sc_s3.b, [0.2, -0.05], atol=1e-10)
    assert abs(sc_s3.beta - 0.15) < 1e-10


def test_s1_regime(sc_s1):
    assert sc_s1.regime == "interior"


def test_s2_regime(sc_s2):
    assert sc_s2.regime == "interior"


def test_s3_regime(sc_s3):
    assert sc_s3.regime == "corner"


def test_s1_t_star(sc_s1):
    """S1 left-tail optimizer t* = b[0]/beta = 0.6315789474."""
    expected = 0.6 / 0.95
    assert abs(sc_s1.t_star_left - expected) < 1e-10


def test_s2_t_star(sc_s2):
    """S2 left-tail optimizer t* = b[0]/beta = 0.9357798165."""
    expected = 5.1 / 5.45
    assert abs(sc_s2.t_star_left - expected) < 1e-10


def test_s3_t_star_left(sc_s3):
    """S3 left-tail optimizer t* = 1 (corner)."""
    assert abs(sc_s3.t_star_left - 1.0) < 1e-14


def test_s3_t_star_right(sc_s3):
    """S3 right-tail optimizer t* = 0 (corner)."""
    assert abs(sc_s3.t_star_right - 0.0) < 1e-14


def test_A_is_inverse_of_B(sc_s1):
    """A = B^{-1} must satisfy A @ B ≈ I."""
    np.testing.assert_allclose(sc_s1.A @ sc_s1.B, np.eye(2), atol=1e-12)


def test_A_spd(sc_s1):
    """A must also be SPD (covariance matrix)."""
    assert check_spd(sc_s1.A)


# ---------------------------------------------------------------------------
# Lognormal simulation
# ---------------------------------------------------------------------------


def test_simulate_shape(sc_s1):
    X = simulate_lognormal(sc_s1, n_samples=1000, seed=0)
    assert X.shape == (1000, 2)


def test_simulate_positive(sc_s1):
    """All samples must be strictly positive."""
    X = simulate_lognormal(sc_s1, n_samples=5000, seed=1)
    assert np.all(X > 0)


def test_simulate_log_normal_distribution(sc_s1):
    """log(X) should be approximately N(mu, A); check marginal mean."""
    n = 100_000
    X = simulate_lognormal(sc_s1, n_samples=n, seed=7)
    log_X = np.log(X)
    # log(X_1) ~ N(mu_1, A_11); sample mean should be close to mu_1
    assert abs(log_X[:, 0].mean() - sc_s1.mu[0]) < 0.01
    assert abs(log_X[:, 1].mean() - sc_s1.mu[1]) < 0.01


def test_simulate_reproducible(sc_s1):
    """Same seed produces identical samples."""
    X1 = simulate_lognormal(sc_s1, n_samples=500, seed=99)
    X2 = simulate_lognormal(sc_s1, n_samples=500, seed=99)
    np.testing.assert_array_equal(X1, X2)


def test_simulate_different_seeds(sc_s1):
    """Different seeds produce different samples."""
    X1 = simulate_lognormal(sc_s1, n_samples=500, seed=0)
    X2 = simulate_lognormal(sc_s1, n_samples=500, seed=1)
    assert not np.array_equal(X1, X2)


# ---------------------------------------------------------------------------
# phi_X: negative log-density
# ---------------------------------------------------------------------------


def test_phi_X_scalar_single_point(sc_s1):
    """phi_X should return a finite scalar for a valid point."""
    x = np.array([[1.0, 1.0]])
    phi = phi_X(x, sc_s1)
    assert phi.shape == (1,)
    assert np.isfinite(phi[0])


def test_phi_X_batch(sc_s1):
    """phi_X on a batch of points returns correct shape."""
    x = np.ones((20, 2)) * 1.5
    phi = phi_X(x, sc_s1)
    assert phi.shape == (20,)
    assert np.all(np.isfinite(phi))


def test_phi_X_monotone_in_distance(sc_s1):
    """phi_X should be larger (less dense) farther from mean in log-space."""
    x_near = np.array([[np.exp(0.0), np.exp(0.0)]])  # at mu
    x_far = np.array([[np.exp(3.0), np.exp(3.0)]])
    phi_near = phi_X(x_near, sc_s1)
    phi_far = phi_X(x_far, sc_s1)
    # phi_X = -log f_X, so higher phi = lower density
    assert phi_far[0] > phi_near[0]


# ---------------------------------------------------------------------------
# a_func
# ---------------------------------------------------------------------------


def test_a_func_scalar(sc_s1):
    """a(x) = sum_i b_i * log(x_i)."""
    x = np.array([[np.e, np.e]])  # log(x_i) = 1
    result = a_func(x, sc_s1.b)
    expected = float(np.sum(sc_s1.b))
    assert abs(result[0] - expected) < 1e-12


def test_a_func_ones(sc_s1):
    """a(1) = 0 since log(1) = 0."""
    x = np.ones((1, 2))
    result = a_func(x, sc_s1.b)
    assert abs(result[0]) < 1e-14


def test_a_func_batch(sc_s1):
    """a_func should handle batched input."""
    x = np.exp(np.ones((50, 2)))  # each row = (e, e)
    result = a_func(x, sc_s1.b)
    assert result.shape == (50,)
    np.testing.assert_allclose(result, float(np.sum(sc_s1.b)), atol=1e-12)


# ---------------------------------------------------------------------------
# build_scenario edge cases
# ---------------------------------------------------------------------------


def test_build_scenario_small_positive():
    """Build a 2x2 interior scenario with all-positive b."""
    sc = build_scenario("test", (0, 0), [[1.0, 0.2], [0.2, 0.5]])
    assert sc.regime == "interior"
    assert sc.beta > 0


def test_build_scenario_corner():
    """Build a 2x2 corner scenario."""
    sc = build_scenario("corner", (0, 0), [[0.1, -0.2], [-0.2, 0.3]])
    # b = B@1 = [0.1-0.2, -0.2+0.3] = [-0.1, 0.1]; check regime
    # Actually B.sum(axis=0) for B = [[0.1,-0.2],[-0.2,0.3]]
    # B[col_sum]: [0.1+(-0.2), -0.2+0.3] = [-0.1, 0.1]
    # One b_i is negative -> corner
    assert sc.regime == "corner"
