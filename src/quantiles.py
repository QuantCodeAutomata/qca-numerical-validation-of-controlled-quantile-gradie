import numpy as np
from typing import Tuple, Optional


def portfolio_returns(X: np.ndarray, t: float) -> np.ndarray:
    """Y = t*X[:,0] + (1-t)*X[:,1] for 2D simplex weights.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, 2)
    t : float in [0, 1]

    Returns
    -------
    Y : np.ndarray, shape (n_samples,)
    """
    return t * X[:, 0] + (1 - t) * X[:, 1]


def portfolio_returns_grid(X: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Compute portfolio returns for every t in t_grid, vectorized.

    Parameters
    ----------
    X      : np.ndarray, shape (n_samples, 2)
    t_grid : np.ndarray, shape (T,)

    Returns
    -------
    Y_grid : np.ndarray, shape (T, n_samples)
    """
    return np.outer(t_grid, X[:, 0]) + np.outer(1 - t_grid, X[:, 1])


def empirical_quantile(Y: np.ndarray, alpha: float) -> float:
    """Empirical alpha-quantile of Y."""
    return float(np.quantile(Y, alpha))


def quantile_curve(
    X: np.ndarray,
    t_grid: np.ndarray,
    alpha: float,
    max_bytes: int = 400_000_000,
) -> np.ndarray:
    """q_alpha(t) for each t in t_grid.

    Processes t_grid in chunks so that the intermediate (chunk × N) float64
    matrix never exceeds *max_bytes* bytes (default 400 MB).

    Parameters
    ----------
    X         : np.ndarray, shape (n_samples, 2)
    t_grid    : np.ndarray, shape (T,)
    alpha     : float in (0, 1)
    max_bytes : int, memory ceiling per chunk in bytes

    Returns
    -------
    q : np.ndarray, shape (T,)
    """
    n_samples = X.shape[0]
    bytes_per_row = n_samples * 8          # float64
    chunk_size = max(1, max_bytes // bytes_per_row)

    q = np.empty(len(t_grid))
    for start in range(0, len(t_grid), chunk_size):
        chunk = t_grid[start : start + chunk_size]
        Y_chunk = portfolio_returns_grid(X, chunk)   # (chunk, N)
        q[start : start + chunk_size] = np.quantile(Y_chunk, alpha, axis=1)
    return q


def finite_diff_gradient_1d(q_curve: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Central finite-difference first derivative dq/dt.

    Uses np.gradient which applies central differences at interior points
    and one-sided differences at the endpoints.

    Parameters
    ----------
    q_curve : np.ndarray, shape (T,)
    t_grid  : np.ndarray, shape (T,), assumed uniformly spaced

    Returns
    -------
    grad : np.ndarray, shape (T,)
    """
    dt = t_grid[1] - t_grid[0]
    return np.gradient(q_curve, dt)


def finite_diff_hessian_1d(q_curve: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Central finite-difference second derivative d^2q/dt^2.

    Parameters
    ----------
    q_curve : np.ndarray, shape (T,)
    t_grid  : np.ndarray, shape (T,), assumed uniformly spaced

    Returns
    -------
    hess : np.ndarray, shape (T,)
    """
    dt = t_grid[1] - t_grid[0]
    return np.gradient(np.gradient(q_curve, dt), dt)


def conditional_mean_narrow_band(
    X: np.ndarray, Y: np.ndarray, q_alpha: float, h_band: float
) -> np.ndarray:
    """Estimate E[X | Y ≈ q_alpha] by narrow-band conditioning.

    Parameters
    ----------
    X       : np.ndarray, shape (n_samples, n_dim)
    Y       : np.ndarray, shape (n_samples,)
    q_alpha : float, the quantile level
    h_band  : float, half-width of the conditioning band

    Returns
    -------
    cond_mean : np.ndarray, shape (n_dim,)
    """
    mask = np.abs(Y - q_alpha) <= h_band
    if mask.sum() < 10:
        raise ValueError(f'Too few samples in band: {mask.sum()}')
    return X[mask].mean(axis=0)


def adaptive_bandwidth(Y: np.ndarray, q_alpha: float, frac: float = 0.005) -> float:
    """Adaptive half-bandwidth for narrow-band conditioning.

    Bandwidth = frac * IQR, with a floor of 1e-6 * |q_alpha|.

    Parameters
    ----------
    Y       : np.ndarray, shape (n_samples,)
    q_alpha : float
    frac    : float, fraction of IQR to use

    Returns
    -------
    h : float
    """
    iqr = float(np.quantile(Y, 0.75) - np.quantile(Y, 0.25))
    return max(frac * iqr, 1e-6 * abs(q_alpha))


def check_homogeneity(
    X: np.ndarray, u: np.ndarray, alpha: float, c: float = 2.0
) -> Tuple[float, float]:
    """Verify positive homogeneity: q_alpha(c*u^T X) = c * q_alpha(u^T X).

    For a lognormal portfolio Y = u^T X, c*Y = (c*u)^T X, so this holds
    exactly for any distribution.

    Parameters
    ----------
    X     : np.ndarray, shape (n_samples, n_dim)
    u     : np.ndarray, shape (n_dim,)  portfolio weight vector
    alpha : float
    c     : float, positive scalar

    Returns
    -------
    (q_cu, c_times_q_u) : Tuple[float, float]
        q_cu = q_alpha((c*u)^T X),  c_times_q_u = c * q_alpha(u^T X)
    """
    Y_u = X @ u
    Y_cu = X @ (c * u)        # = c * Y_u
    q_u = empirical_quantile(Y_u, alpha)
    q_cu = empirical_quantile(Y_cu, alpha)
    return q_cu, c * q_u
