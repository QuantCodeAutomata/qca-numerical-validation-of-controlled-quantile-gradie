import numpy as np
from scipy import linalg
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Scenario:
    name: str
    mu: np.ndarray       # mean of Z where X = exp(Z)
    B: np.ndarray        # precision-like matrix (B = A^{-1})
    A: np.ndarray        # covariance of Z (A = B^{-1})
    b: np.ndarray        # column sums of B: b_i = sum_j B_{ji}
    beta: float          # sum of b_i
    regime: str          # 'interior' or 'corner'
    t_star_left: float   # asymptotic left-tail optimizer t*
    t_star_right: float  # asymptotic right-tail optimizer t*


def check_spd(B: np.ndarray) -> bool:
    """Return True if B is symmetric positive definite (verified via Cholesky)."""
    if not np.allclose(B, B.T):
        return False
    try:
        np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def build_scenario(name: str, mu, B_matrix) -> Scenario:
    """Build a Scenario from name, mu vector, and B matrix."""
    B = np.array(B_matrix, dtype=float)
    A = linalg.inv(B)
    b = B.sum(axis=0)          # column sums: b_i = sum_j B_{ji}
    beta = float(b.sum())
    regime = 'interior' if np.all(b > 0) else 'corner'

    if regime == 'interior':
        t_star_left = float(b[0] / beta)
        t_star_right = float(b[0] / beta)
    else:
        # Corner regime: left-tail optimizer sits at the corner of the dominant asset.
        # For S3: b[0]=0.2 > 0, b[1]=-0.05 < 0 => left-tail -> t=1, right-tail -> t=0
        t_star_left = 1.0 if b[0] > 0 else 0.0
        t_star_right = 0.0 if b[0] > 0 else 1.0

    return Scenario(
        name=name,
        mu=np.array(mu, dtype=float),
        B=B,
        A=A,
        b=b,
        beta=beta,
        regime=regime,
        t_star_left=t_star_left,
        t_star_right=t_star_right,
    )


def get_scenarios() -> Dict[str, Scenario]:
    """Return the three canonical lognormal scenarios."""
    scenarios = {}

    # S1: moderate variance, interior regime
    scenarios['S1'] = build_scenario(
        name='S1',
        mu=(0, 0),
        B_matrix=[[0.5, 0.1], [0.1, 0.25]],
    )

    # S2: high variance in first asset, interior regime
    scenarios['S2'] = build_scenario(
        name='S2',
        mu=(0, 0),
        B_matrix=[[5.0, 0.1], [0.1, 0.25]],
    )

    # S3: negative off-diagonal, corner regime
    scenarios['S3'] = build_scenario(
        name='S3',
        mu=(0, 0),
        B_matrix=[[0.5, -0.3], [-0.3, 0.25]],
    )

    return scenarios


# Alias for backward compatibility
GET_SCENARIOS = get_scenarios


def simulate_lognormal(scenario: Scenario, n_samples: int, seed: int = 42) -> np.ndarray:
    """Simulate X = exp(Z), Z ~ N(mu, A).

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_dim)
    """
    rng = np.random.default_rng(seed)
    n = len(scenario.mu)
    Z = rng.multivariate_normal(scenario.mu, scenario.A, size=n_samples)
    return np.exp(Z)


def phi_X(x: np.ndarray, scenario: Scenario) -> np.ndarray:
    """Negative log-density of the lognormal distribution (up to additive constant).

    phi_X(x) = 0.5*(log x - mu)^T B (log x - mu) + sum_j log(x_j)
               + (n/2)*log(2*pi) + 0.5*log det(A)

    where B = A^{-1}.  Satisfies f_X(x) = exp(-phi_X(x)).

    Parameters
    ----------
    x : np.ndarray, shape (..., n)
    scenario : Scenario

    Returns
    -------
    phi : np.ndarray, shape (...)
    """
    n = len(scenario.mu)
    log_x = np.log(x)                          # (..., n)
    diff = log_x - scenario.mu                  # (..., n)
    mahal = np.einsum('...i,ij,...j->...', diff, scenario.B, diff)  # B = A^{-1}
    log_det_A = np.log(linalg.det(scenario.A))
    return 0.5 * mahal + np.sum(log_x, axis=-1) + (n / 2) * np.log(2 * np.pi) + 0.5 * log_det_A


def a_func(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a(x) = sum_i b_i * log(x_i).

    Parameters
    ----------
    x : np.ndarray, shape (..., n)
    b : np.ndarray, shape (n,)

    Returns
    -------
    a : np.ndarray, shape (...)
    """
    return np.sum(b * np.log(x), axis=-1)
