"""Surface measure on S_1 = {x > 0 : u^T x = 1} and quadrature-based moments.

The paper's pulled-back conditional measure mu_y on S_1 has density proportional
to g_y(x) = exp(-w_y(x)) where
    w_y(x) = log(y) * a(x) + phi_X(x),
    a(x)   = sum_i b_i * log(x_i),
    phi_X  = -log f_X  (negative log-density of the lognormal).

For the 2D simplex parameterization with u = (u1, u2):
    S_1 = {(x1, x2) : x1 > 0, x2 > 0, u1*x1 + u2*x2 = 1}
    x2 = (1 - u1*x1) / u2,  x1 in (0, 1/u1).

The Jacobian factor dvol^{S_1} / dx1 = ||u|| / u2 cancels in all ratio
expectations (E[h(x)] under mu_y), so it is omitted from the quadrature.
"""

import numpy as np
from scipy import linalg
from scipy.stats import gaussian_kde
from typing import Dict, Tuple

from src.distributions import phi_X, a_func


# ---------------------------------------------------------------------------
# Point-wise helpers (scalar x vectors)
# ---------------------------------------------------------------------------

def phi_X_scalar(x: np.ndarray, scenario) -> float:
    """phi_X for a single point x, shape (n,)."""
    return float(phi_X(x, scenario))


def a_func_scalar(x: np.ndarray, b: np.ndarray) -> float:
    """a(x) for a single point x, shape (n,)."""
    return float(a_func(x, b))


# ---------------------------------------------------------------------------
# w_y and log g_y
# ---------------------------------------------------------------------------

def w_y_func(x1: float, u: np.ndarray, y: float, scenario) -> float:
    """w_y(x) = log(y)*a(x) + phi_X(x) evaluated on S_1 at x1.

    Parameters
    ----------
    x1       : float, first coordinate (x2 determined by the constraint)
    u        : np.ndarray, shape (2,), portfolio weights
    y        : float, portfolio value (level set)
    scenario : Scenario

    Returns
    -------
    float (np.inf if x is not in the positive orthant)
    """
    x2 = (1.0 - u[0] * x1) / u[1]
    if x1 <= 0 or x2 <= 0:
        return np.inf
    x = np.array([x1, x2])
    return np.log(y) * a_func_scalar(x, scenario.b) + phi_X_scalar(x, scenario)


def log_gy_func(x1_arr: np.ndarray, u: np.ndarray, y: float, scenario) -> np.ndarray:
    """Compute log g_y(x) = -w_y(x) for an array of x1 values.

    Uses direct vectorised computation to avoid Python-level loops.

    Parameters
    ----------
    x1_arr   : np.ndarray, shape (M,)
    u        : np.ndarray, shape (2,)
    y        : float
    scenario : Scenario

    Returns
    -------
    log_gy : np.ndarray, shape (M,)  (-inf where x is invalid)
    """
    u1, u2 = u[0], u[1]
    x2_arr = (1.0 - u1 * x1_arr) / u2
    valid = (x1_arr > 0) & (x2_arr > 0)

    result = np.full_like(x1_arr, -np.inf, dtype=float)
    if not valid.any():
        return result

    x1v = x1_arr[valid]
    x2v = x2_arr[valid]

    log_x1 = np.log(x1v)
    log_x2 = np.log(x2v)

    # a(x) = b[0]*log(x1) + b[1]*log(x2)
    a_vals = scenario.b[0] * log_x1 + scenario.b[1] * log_x2

    # phi_X(x) = 0.5*(log x - mu)^T B (log x - mu) + log(x1) + log(x2)
    #            + log(2*pi) + 0.5*log det(A)
    diff1 = log_x1 - scenario.mu[0]
    diff2 = log_x2 - scenario.mu[1]
    B = scenario.B
    mahal = (B[0, 0] * diff1 ** 2
             + (B[0, 1] + B[1, 0]) * diff1 * diff2
             + B[1, 1] * diff2 ** 2)
    log_det_A = np.log(linalg.det(scenario.A))
    phi_vals = (0.5 * mahal
                + log_x1 + log_x2
                + np.log(2 * np.pi)
                + 0.5 * log_det_A)

    w_vals = np.log(y) * a_vals + phi_vals
    result[valid] = -w_vals
    return result


# ---------------------------------------------------------------------------
# Gauss-Legendre quadrature for moments under mu_y
# ---------------------------------------------------------------------------

def quadrature_moments_2d(
    u: np.ndarray, y: float, scenario, n_quad: int = 2000
) -> Dict:
    """Compute moments of x under mu_y on S_1 via Gauss-Legendre quadrature.

    All expectations are with respect to the conditional measure mu_y whose
    density is proportional to g_y(x) = exp(-w_y(x)).

    Parameters
    ----------
    u        : np.ndarray, shape (2,), portfolio weights (both positive)
    y        : float, portfolio value (quantile level)
    scenario : Scenario
    n_quad   : int, number of Gauss-Legendre nodes

    Returns
    -------
    dict with keys:
        E_x, E_x1, E_x2      : conditional expectations of x1, x2
        E_gamma, Var_gamma   : mean and variance of gamma = x1 - x2
        E_a                  : E[a(x)] under mu_y
        E_a_centered_gamma2  : E[a(x) * (gamma - E_gamma)^2]  (paper's eq 3.27)
        Cov_a_gamma2         : Cov(a, (gamma-E_gamma)^2) = E_a_centered_gamma2 - E_a*Var_gamma
        Z_y_unnorm           : unnormalized partition function integral
        log_gy_max           : stabilisation shift used in log-sum-exp
    """
    u1, u2 = float(u[0]), float(u[1])

    # Integration domain: x1 in (eps, 1/u1 - eps)
    eps = 1e-10
    x1_min = eps
    x1_max = 1.0 / u1 - eps

    # Gauss-Legendre nodes/weights on [-1, 1] -> transform to [x1_min, x1_max]
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    half_len = 0.5 * (x1_max - x1_min)
    mid = 0.5 * (x1_max + x1_min)
    x1_vals = half_len * nodes + mid     # shape (n_quad,)
    x2_vals = (1.0 - u1 * x1_vals) / u2

    # log g_y evaluated at quadrature nodes
    log_gy = log_gy_func(x1_vals, u, y, scenario)

    # Log-sum-exp stabilisation
    log_gy_max = float(np.max(log_gy[np.isfinite(log_gy)], initial=-np.inf))
    gy_shifted = np.exp(log_gy - log_gy_max)   # g_y / max(g_y)

    # Weighted quadrature: integral ≈ sum(weights * f) * half_len
    def _quad(f_vals: np.ndarray) -> float:
        return float(np.sum(weights * f_vals) * half_len)

    Z_y_unnorm = _quad(gy_shifted)

    # Conditional expectations (divide by Z_y_unnorm)
    E_x1 = _quad(x1_vals * gy_shifted) / Z_y_unnorm
    E_x2 = _quad(x2_vals * gy_shifted) / Z_y_unnorm
    E_x = np.array([E_x1, E_x2])

    # gamma = x1 - x2  (= v^T x for v = (1, -1))
    gamma_vals = x1_vals - x2_vals
    E_gamma = _quad(gamma_vals * gy_shifted) / Z_y_unnorm
    E_gamma2 = _quad(gamma_vals ** 2 * gy_shifted) / Z_y_unnorm
    Var_gamma = E_gamma2 - E_gamma ** 2

    # a(x) values at quadrature nodes
    log_x1_vals = np.log(x1_vals)
    log_x2_vals = np.log(x2_vals)
    a_vals = scenario.b[0] * log_x1_vals + scenario.b[1] * log_x2_vals

    E_a = _quad(a_vals * gy_shifted) / Z_y_unnorm

    # E[a * gamma]  and  E[a * gamma^2]
    E_a_gamma = _quad(a_vals * gamma_vals * gy_shifted) / Z_y_unnorm
    E_a_gamma2 = _quad(a_vals * gamma_vals ** 2 * gy_shifted) / Z_y_unnorm

    # E[a * (gamma - E_gamma)^2]  (used directly in eq 3.27)
    E_a_centered_gamma2 = E_a_gamma2 - 2.0 * E_gamma * E_a_gamma + E_gamma ** 2 * E_a

    # Cov(a, (gamma - E_gamma)^2) = E[a*(gamma-Egamma)^2] - E[a]*Var[gamma]
    Cov_a_gamma2 = E_a_centered_gamma2 - E_a * Var_gamma

    return {
        'E_x': E_x,
        'E_x1': E_x1,
        'E_x2': E_x2,
        'E_gamma': E_gamma,
        'Var_gamma': Var_gamma,
        'E_a': E_a,
        'E_a_gamma': E_a_gamma,
        'E_a_gamma2': E_a_gamma2,
        'E_a_centered_gamma2': E_a_centered_gamma2,
        'Cov_a_gamma2': Cov_a_gamma2,
        'Z_y_unnorm': Z_y_unnorm,
        'log_gy_max': log_gy_max,
    }


# ---------------------------------------------------------------------------
# Density estimation (KDE fallback)
# ---------------------------------------------------------------------------

def density_f_Y(
    u: np.ndarray, y: float, scenario, n_quad: int = 2000
) -> float:
    """Placeholder: f_Y estimation via quadrature on S_1.

    In practice, use kde_density on a Monte Carlo sample of Y = u^T X.
    The quadrature-based approach requires an additional outer integral over
    y which is expensive; KDE is preferred for diagnostics.
    """
    pass  # KDE on MC sample is used in practice; see kde_density


def kde_density(Y_samples: np.ndarray, y_eval: float) -> float:
    """Gaussian KDE estimate of f_Y at y_eval.

    Parameters
    ----------
    Y_samples : np.ndarray, shape (n_samples,)
    y_eval    : float

    Returns
    -------
    float
    """
    kde = gaussian_kde(Y_samples)
    return float(kde(y_eval)[0])


# ---------------------------------------------------------------------------
# Equation (3.27): curvature / transport term for d^2 q / dt^2
# ---------------------------------------------------------------------------

def eq_327_curvature_term(
    u: np.ndarray,
    y: float,
    scenario,
    moments: Dict,
    beta_sum_Bmu: float,
) -> float:
    """Evaluate the eq (3.27) transport term for d^2 q_alpha / dt^2.

    The paper gives:
        d_y [f_Y(y) * Var_{mu_y}[gamma]] = f_Y(y) * T(y)
    where
        T(y) = (1/y) * (coeff * Var_gamma - E[a*(gamma - E_gamma)^2])
        coeff = -beta * log(y) + sum_j (B @ mu)_j + 1

    And the second derivative of the quantile is:
        d^2 q / dt^2 = -(1/f_Y) * d_y[f_Y * Var_y[gamma]] = -T(y)

    This function returns T(y), so the caller should negate it to get d^2q/dt^2.

    Parameters
    ----------
    u              : np.ndarray, shape (2,)
    y              : float, quantile level
    scenario       : Scenario
    moments        : dict returned by quadrature_moments_2d
    beta_sum_Bmu   : float, sum_j (B @ mu)_j  (= 1^T B mu = b^T mu)

    Returns
    -------
    T_y : float   (negate to obtain d^2q/dt^2)
    """
    Var_g = moments['Var_gamma']
    E_a_cg2 = moments['E_a_centered_gamma2']

    coeff = -scenario.beta * np.log(y) + beta_sum_Bmu + 1.0
    T_y = (1.0 / y) * (coeff * Var_g - E_a_cg2)
    return float(T_y)
