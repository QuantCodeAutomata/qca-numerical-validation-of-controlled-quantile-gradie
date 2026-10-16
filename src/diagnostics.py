"""Diagnostic comparisons between finite-difference estimates and analytical
formulae for the gradient and Hessian of the quantile function q_alpha(t).

Gradient formula (first derivative):
    dq_alpha/dt = (E[X_1 | Y = q] - E[X_2 | Y = q])
               = E[X | Y = q] . v,  v = (1, -1)

Hessian formula (second derivative, from eq 3.27):
    d^2 q_alpha / dt^2 = -T(y)
where
    T(y) = (1/y) * (coeff * Var_{mu_y}[gamma] - E_{mu_y}[a*(gamma - Egamma)^2])
    coeff = -beta * log(y) + sum_j (B @ mu)_j + 1
    gamma = x_1 - x_2  (=v^T x, v=(1,-1))
"""

import numpy as np
from scipy.stats import gaussian_kde
from typing import Dict

from src.quantiles import (
    portfolio_returns,
    empirical_quantile,
    quantile_curve,
    finite_diff_gradient_1d,
    finite_diff_hessian_1d,
    conditional_mean_narrow_band,
    adaptive_bandwidth,
)
from src.surface_measure import quadrature_moments_2d, eq_327_curvature_term


def gradient_fd_vs_formula(
    X: np.ndarray,
    t_grid: np.ndarray,
    alpha: float,
    scenario,
) -> Dict:
    """Compare finite-difference gradient with narrow-band conditional-mean formula.

    Parameters
    ----------
    X        : np.ndarray, shape (n_samples, 2)
    t_grid   : np.ndarray, shape (T,)
    alpha    : float in (0, 1)
    scenario : Scenario

    Returns
    -------
    dict with keys:
        t_grid           : np.ndarray, shape (T,)
        q_curve          : np.ndarray, shape (T,)  empirical quantile curve
        fd_gradient      : np.ndarray, shape (T,)  central FD derivative
        formula_gradient : np.ndarray, shape (T,)  narrow-band MC estimate
    """
    q_curve = quantile_curve(X, t_grid, alpha)
    fd_grad = finite_diff_gradient_1d(q_curve, t_grid)

    formula_grad = np.full(len(t_grid), np.nan)
    for i, t in enumerate(t_grid):
        u = np.array([t, 1.0 - t])
        Y = X @ u
        q_a = q_curve[i]
        h = adaptive_bandwidth(Y, q_a)
        try:
            cond_mean = conditional_mean_narrow_band(X, Y, q_a, h)
            # dq/dt = E[X | Y = q] . v,  v = (1, -1)
            formula_grad[i] = cond_mean[0] - cond_mean[1]
        except ValueError:
            pass   # leave as nan

    return {
        't_grid': t_grid,
        'q_curve': q_curve,
        'fd_gradient': fd_grad,
        'formula_gradient': formula_grad,
    }


def hessian_transport_vs_fd(
    X: np.ndarray,
    t_grid: np.ndarray,
    alpha: float,
    scenario,
    n_quad: int = 500,
) -> Dict:
    """Compare finite-difference second derivative with the transport formula.

    The analytical formula (eq 3.27) gives:
        d^2 q / dt^2 = -T(y)
    where T(y) is computed via Gauss-Legendre quadrature on the level set S_1.

    Parameters
    ----------
    X        : np.ndarray, shape (n_samples, 2)
    t_grid   : np.ndarray, shape (T,)
    alpha    : float in (0, 1)
    scenario : Scenario
    n_quad   : int, number of quadrature nodes per t-point

    Returns
    -------
    dict with keys:
        t_grid            : np.ndarray, shape (T,)
        q_curve           : np.ndarray, shape (T,)
        fd_hessian        : np.ndarray, shape (T,)  second FD derivative
        transport_hessian : np.ndarray, shape (T,)  eq 3.27 formula
    """
    q_curve = quantile_curve(X, t_grid, alpha)
    fd_hess = finite_diff_hessian_1d(q_curve, t_grid)

    # sum_j (B @ mu)_j = b^T mu  (= 0 when mu = 0)
    sum_Bmu = float(np.sum(scenario.B @ scenario.mu))

    transport_hessian = np.full(len(t_grid), np.nan)
    for i, t in enumerate(t_grid):
        y = q_curve[i]
        if y <= 0:
            continue
        u = np.array([t, 1.0 - t])
        try:
            moments = quadrature_moments_2d(u, y, scenario, n_quad=n_quad)
            T_y = eq_327_curvature_term(u, y, scenario, moments, sum_Bmu)
            # d^2 q / dt^2 = -T(y)  (f_Y cancels in the ratio)
            transport_hessian[i] = -T_y
        except Exception:
            pass   # leave as nan

    return {
        't_grid': t_grid,
        'q_curve': q_curve,
        'fd_hessian': fd_hess,
        'transport_hessian': transport_hessian,
    }


def compute_curvature_sign_fraction(
    hessian: np.ndarray, sign: int = -1
) -> float:
    """Fraction of grid points where ``hessian * sign > 0``.

    With sign=-1 this measures the fraction of points exhibiting concavity
    (d^2q/dt^2 < 0).

    Parameters
    ----------
    hessian : np.ndarray
    sign    : int, +1 or -1

    Returns
    -------
    float in [0, 1] or nan if no valid points
    """
    valid = ~np.isnan(hessian)
    if valid.sum() == 0:
        return float('nan')
    return float(np.mean((hessian[valid] * sign) > 0))
