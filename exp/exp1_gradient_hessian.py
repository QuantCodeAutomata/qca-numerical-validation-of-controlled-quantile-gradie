"""Experiment 1: Numerical validation of quantile gradient and Hessian formulas.

Validates Theorem 1.1 for linear-payoff Y = u^T X with X multivariate lognormal.
- Gradient: dq_alpha/du = E[X | u^T X = q_alpha] (narrow-band MC + quadrature)
- Hessian: d^2q/dt^2 explained by transport term from eq (3.27) (no direct h term)
- Homogeneity: q_alpha(c*u) = c*q_alpha(u)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.distributions import get_scenarios, simulate_lognormal
from src.quantiles import (
    quantile_curve,
    finite_diff_gradient_1d,
    finite_diff_hessian_1d,
    check_homogeneity,
)
from src.diagnostics import (
    gradient_fd_vs_formula,
    hessian_transport_vs_fd,
    compute_curvature_sign_fraction,
)


def run_exp1(n_samples: int = 300_000, seed: int = 42) -> pd.DataFrame:
    """Run gradient/Hessian validation experiment.

    Parameters
    ----------
    n_samples : int
        Number of Monte Carlo draws.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with per-scenario / per-alpha results.
    """
    os.makedirs("results", exist_ok=True)
    scenarios = get_scenarios()

    alpha_grid_left = [0.05, 0.10, 0.20]
    alpha_grid_right = [0.80, 0.90, 0.95]
    alpha_all = alpha_grid_left + alpha_grid_right
    t_grid = np.linspace(0.05, 0.95, 19)

    records = []

    for sc_name, sc in scenarios.items():
        print(f"\n=== Scenario {sc_name} | regime={sc.regime} | beta={sc.beta:.4f} ===")
        X = simulate_lognormal(sc, n_samples, seed)

        # ---- Gradient comparison ------------------------------------------------
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f"{sc_name}: FD gradient vs formula-based gradient", fontsize=12)

        for ax, alpha in zip(axes.flat, alpha_all):
            res = gradient_fd_vs_formula(X, t_grid, alpha, sc)
            fd_grad = res["fd_gradient"]
            formula_grad = res["formula_gradient"]

            valid = ~np.isnan(formula_grad)
            if valid.sum() > 2:
                mae = float(np.nanmean(np.abs(fd_grad[valid] - formula_grad[valid])))
                corr = float(np.corrcoef(fd_grad[valid], formula_grad[valid])[0, 1])
            else:
                mae, corr = float("nan"), float("nan")

            records.append(
                {
                    "experiment": "exp1_gradient",
                    "scenario": sc_name,
                    "alpha": alpha,
                    "grad_mae": mae,
                    "grad_corr": corr,
                }
            )
            print(
                f"  alpha={alpha:.2f}  gradient MAE={mae:.4f}  corr={corr:.4f}"
            )

            ax.plot(t_grid, fd_grad, "b-o", markersize=3, label="FD")
            ax.plot(t_grid, formula_grad, "r--s", markersize=3, label="Formula (MC band)")
            ax.set_xlabel("t")
            ax.set_ylabel("dq/dt")
            ax.set_title(f"α={alpha}")
            ax.legend(fontsize=7)

        plt.tight_layout()
        plt.savefig(f"results/exp1_gradient_{sc_name}.png", dpi=100, bbox_inches="tight")
        plt.close()

        # ---- Hessian sign diagnostic -------------------------------------------
        fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
        fig2.suptitle(f"{sc_name}: FD Hessian vs transport formula", fontsize=12)
        hess_alphas = [0.05, 0.10, 0.90, 0.95]

        for ax, alpha in zip(axes2.flat, hess_alphas):
            res_h = hessian_transport_vs_fd(X, t_grid, alpha, sc, n_quad=300)
            fd_hess = res_h["fd_hessian"]
            transport_hess = res_h["transport_hessian"]

            frac_neg_fd = compute_curvature_sign_fraction(fd_hess[2:-2], sign=-1)
            frac_neg_tr = compute_curvature_sign_fraction(transport_hess[2:-2], sign=-1)

            records.append(
                {
                    "experiment": "exp1_hessian",
                    "scenario": sc_name,
                    "alpha": alpha,
                    "frac_neg_fd": frac_neg_fd,
                    "frac_neg_transport": frac_neg_tr,
                }
            )
            print(
                f"  alpha={alpha:.2f}  frac_neg_fd={frac_neg_fd:.2f}"
                f"  frac_neg_transport={frac_neg_tr:.2f}"
            )

            ax.plot(t_grid, fd_hess, "b-", label="FD hessian")
            ax.plot(t_grid, transport_hess, "r--", label="Transport formula")
            ax.axhline(0, color="k", linewidth=0.5)
            ax.set_xlabel("t")
            ax.set_ylabel("d²q/dt²")
            ax.set_title(f"α={alpha}")
            ax.legend(fontsize=7)

        plt.tight_layout()
        plt.savefig(f"results/exp1_hessian_{sc_name}.png", dpi=100, bbox_inches="tight")
        plt.close()

        # ---- Homogeneity check -------------------------------------------------
        for alpha in [0.05, 0.50, 0.95]:
            u = np.array([0.5, 0.5])
            q_cu, c_q_u = check_homogeneity(X, u, alpha, c=2.0)
            rel_err = abs(q_cu - c_q_u) / (abs(c_q_u) + 1e-15)
            records.append(
                {
                    "experiment": "exp1_homogeneity",
                    "scenario": sc_name,
                    "alpha": alpha,
                    "q_cu": q_cu,
                    "c_q_u": c_q_u,
                    "rel_error": rel_err,
                }
            )
            print(f"  Homogeneity alpha={alpha:.2f}  rel_error={rel_err:.2e}")

    df = pd.DataFrame(records)
    df.to_csv("results/exp1_results.csv", index=False)
    print("\nExp1 results saved to results/exp1_results.csv")
    return df


if __name__ == "__main__":
    df = run_exp1()
    print(df.to_string())
    print("Experiment 1 complete.")
