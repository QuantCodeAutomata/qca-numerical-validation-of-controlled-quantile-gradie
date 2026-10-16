"""Experiment 4: Mass-flow and critical-point diagnostics under the pulled-back conditional measure.

Studies how mu_y on S_1 evolves with y (the quantile level):
- Profiles of g_y = exp(-w_y) along S_1
- Critical points (roots of d/dx1 [-w_y] = 0)
- Transport term sign from eq (3.27) and (3.28)
- Mass concentration pattern as y varies (left -> right)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from src.distributions import get_scenarios
from src.surface_measure import (
    quadrature_moments_2d,
    log_gy_func,
    eq_327_curvature_term,
)


# ---------------------------------------------------------------------------
# Critical-point equation on S_1
# ---------------------------------------------------------------------------


def critical_point_deriv(
    x1: float, u: np.ndarray, y: float, scenario
) -> float:
    """d/dx1 of [-w_y] along S_1 at the point x1.

    On S_1: x2 = (1 - u1*x1)/u2.
    Evaluates the directional derivative of -w_y with respect to x1,
    which equals zero at critical points of g_y on S_1.

    Parameters
    ----------
    x1       : float
    u        : np.ndarray, shape (2,)
    y        : float
    scenario : Scenario

    Returns
    -------
    float  (returns np.nan when x is not in the positive orthant)
    """
    u1, u2 = float(u[0]), float(u[1])
    x2 = (1.0 - u1 * x1) / u2
    if x1 <= 0.0 or x2 <= 0.0:
        return np.nan

    log_x1, log_x2 = np.log(x1), np.log(x2)
    diff = np.array([log_x1 - scenario.mu[0], log_x2 - scenario.mu[1]])
    B_diff = scenario.B @ diff  # shape (2,)

    # d a(x)/dx1 = b1/x1 + b2/x2 * (-u1/u2)
    da_dx1 = scenario.b[0] / x1 + scenario.b[1] / x2 * (-u1 / u2)

    # d phi_X(x)/dx1 = (B diff)_1/x1 + (B diff)_2*(-u1/u2)/x2
    #                  + 1/x1 + 1/x2*(-u1/u2)   [from the log-Jacobian sum log x_j]
    dphi_dx1 = (
        B_diff[0] / x1
        + B_diff[1] * (-u1 / u2) / x2
        + 1.0 / x1
        + 1.0 / x2 * (-u1 / u2)
    )

    # d [-w_y]/dx1 = -log(y)*da_dx1 - dphi_dx1
    return -np.log(y) * da_dx1 - dphi_dx1


def find_critical_points(
    u: np.ndarray, y: float, scenario, n_scan: int = 200
) -> list:
    """Find all critical points of g_y on S_1 by scanning + Brent root-finding.

    Parameters
    ----------
    u        : np.ndarray, shape (2,)
    y        : float
    scenario : Scenario
    n_scan   : int, number of scan intervals

    Returns
    -------
    list of float, sorted x1 values at critical points
    """
    u1 = float(u[0])
    x1_max = 1.0 / u1
    eps = 1e-8
    x1_scan = np.linspace(eps, x1_max - eps, n_scan + 1)

    f_vals = np.array(
        [critical_point_deriv(x1, u, y, scenario) for x1 in x1_scan]
    )
    roots = []
    for i in range(n_scan):
        fa, fb = f_vals[i], f_vals[i + 1]
        if np.isnan(fa) or np.isnan(fb):
            continue
        if fa * fb < 0.0:
            try:
                root = brentq(
                    critical_point_deriv,
                    x1_scan[i], x1_scan[i + 1],
                    args=(u, y, scenario),
                    xtol=1e-12,
                    maxiter=200,
                )
                roots.append(float(root))
            except Exception:
                pass
    return sorted(set(roots))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_exp4() -> pd.DataFrame:
    """Run mass-flow and critical-point diagnostics.

    Returns
    -------
    pd.DataFrame
    """
    os.makedirs("results", exist_ok=True)
    scenarios = get_scenarios()

    # Representative t values and y grid
    t_values = [0.3, 0.5, 0.7]
    y_grid = np.logspace(-2, 2, 40)   # y from 0.01 to 100

    # y values used for profile visualisation
    y_profile_vals = [0.02, 0.1, 0.5, 2.0, 20.0]

    records = []

    for sc_name, sc in scenarios.items():
        sum_Bmu = float(np.sum(sc.B @ sc.mu))
        print(f"\n=== Scenario {sc_name} | regime={sc.regime} | beta={sc.beta:.4f} ===")

        n_t = len(t_values)
        fig, axes = plt.subplots(n_t, 1, figsize=(10, 4 * n_t), squeeze=False)
        fig.suptitle(f"{sc_name}: g_y profiles on S_1 (mass concentration)")

        for ax_row, t in enumerate(t_values):
            u = np.array([t, 1.0 - t])
            u1, u2 = u[0], u[1]
            x1_max = 1.0 / u1
            x1_plot = np.linspace(1e-6, x1_max - 1e-6, 600)

            ax = axes[ax_row, 0]
            colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(y_profile_vals)))
            for y_p, col in zip(y_profile_vals, colors):
                log_gy = log_gy_func(x1_plot, u, y_p, sc)
                finite = np.isfinite(log_gy)
                if not finite.any():
                    continue
                log_gy_max = log_gy[finite].max()
                gy_norm = np.where(finite, np.exp(log_gy - log_gy_max), 0.0)
                ax.plot(x1_plot, gy_norm, color=col, label=f"y={y_p}")

            ax.set_xlabel("x₁")
            ax.set_ylabel("g_y / max(g_y)")
            ax.set_title(f"t={t}")
            ax.legend(fontsize=7)

            # Analytical & moment diagnostics over y_grid
            for y in y_grid:
                try:
                    moments = quadrature_moments_2d(u, y, sc, n_quad=400)
                    T_y = eq_327_curvature_term(u, y, sc, moments, sum_Bmu)
                    roots = find_critical_points(u, y, sc, n_scan=100)
                    coeff = -sc.beta * np.log(y) + sum_Bmu + 1.0
                    records.append(
                        {
                            "scenario": sc_name,
                            "t": t,
                            "y": y,
                            "n_critical_pts": len(roots),
                            "E_x1": moments["E_x1"],
                            "E_x2": moments["E_x2"],
                            "Var_gamma": moments["Var_gamma"],
                            "transport_T_y": T_y,
                            "transport_sign": int(np.sign(T_y)),
                            "hessian_sign": int(np.sign(-T_y)),
                            "coeff_eq327": coeff,
                        }
                    )
                except Exception as exc:
                    records.append(
                        {
                            "scenario": sc_name,
                            "t": t,
                            "y": y,
                            "n_critical_pts": np.nan,
                            "transport_T_y": np.nan,
                            "error": str(exc),
                        }
                    )

        plt.tight_layout()
        plt.savefig(
            f"results/exp4_mass_flow_{sc_name}.png", dpi=100, bbox_inches="tight"
        )
        plt.close()

        # Transport sign vs y plot
        sc_recs = [r for r in records if r.get("scenario") == sc_name]
        for t_val in t_values:
            t_recs = [
                r for r in sc_recs if r.get("t") == t_val and "transport_T_y" in r
            ]
            if not t_recs:
                continue
            ys = [r["y"] for r in t_recs]
            T_vals = [r.get("transport_T_y", np.nan) for r in t_recs]
            print(
                f"  t={t_val}: transport T_y at y=[{min(ys):.3f}…{max(ys):.3f}]"
                f"  sign changes: {_count_sign_changes(T_vals)}"
            )

        # Critical points vs y (branch tracking)
        fig_cp, ax_cp = plt.subplots(figsize=(9, 5))
        ax_cp.set_title(f"{sc_name}: Critical-point locations vs y")
        ax_cp.set_xlabel("y (log scale)")
        ax_cp.set_ylabel("x₁* (critical point on S₁, t=0.5)")
        t_fixed = 0.5
        u_fixed = np.array([t_fixed, 1.0 - t_fixed])
        for y in y_grid:
            cps = find_critical_points(u_fixed, y, sc, n_scan=100)
            for cp in cps:
                ax_cp.plot(y, cp, "ko", markersize=3)
        ax_cp.set_xscale("log")
        plt.tight_layout()
        plt.savefig(
            f"results/exp4_critical_pts_{sc_name}.png", dpi=100, bbox_inches="tight"
        )
        plt.close()

    df = pd.DataFrame(records)
    df.to_csv("results/exp4_results.csv", index=False)
    print("\nExp4 results saved to results/exp4_results.csv")
    return df


def _count_sign_changes(vals: list) -> int:
    """Count sign changes in a sequence, ignoring NaN."""
    valid = [v for v in vals if not (np.isnan(v) if isinstance(v, float) else False)]
    changes = 0
    for i in range(len(valid) - 1):
        if np.sign(valid[i]) != np.sign(valid[i + 1]) and valid[i] != 0:
            changes += 1
    return changes


if __name__ == "__main__":
    df = run_exp4()
    print(df.head(20).to_string())
    print("Experiment 4 complete.")
