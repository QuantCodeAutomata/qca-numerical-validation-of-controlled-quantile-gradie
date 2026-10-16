"""Experiment 5: Asymptotic portfolio optimization and optimizer convergence.

Numerically solves u_alpha^* = argmax_{u in simplex} q_alpha(u) and compares
the finite-alpha optimizers to the theoretical asymptotic optimizers.

For interior scenarios (S1, S2): t* = b[0]/beta
For corner scenario (S3): left-tail t*=1, right-tail t*=0

Also cross-checks the A-matrix optimizer formula:
    u1* = (A22 - A12) / (A11 + A22 - 2*A12)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

from src.distributions import get_scenarios, simulate_lognormal, build_scenario
from src.quantiles import quantile_curve


def optimizer_a_formula(A: np.ndarray) -> tuple:
    """Interior 2D asymptotic optimizer from the A matrix.

    u1* = (A22 - A12) / (A11 + A22 - 2*A12)
    u2* = (A11 - A12) / (A11 + A22 - 2*A12)

    Parameters
    ----------
    A : np.ndarray, shape (2, 2)

    Returns
    -------
    (u1_star, u2_star) : tuple of float
    """
    denom = A[0, 0] + A[1, 1] - 2.0 * A[0, 1]
    u1 = (A[1, 1] - A[0, 1]) / denom
    u2 = (A[0, 0] - A[0, 1]) / denom
    return float(u1), float(u2)


def find_optimizer(q_curve_vals: np.ndarray, t_grid: np.ndarray) -> float:
    """Coarse grid search + PCHIP refinement for argmax.

    Parameters
    ----------
    q_curve_vals : np.ndarray, shape (T,)
    t_grid       : np.ndarray, shape (T,)

    Returns
    -------
    float
    """
    best_idx = int(np.argmax(q_curve_vals))
    t_star = float(t_grid[best_idx])

    if 3 <= best_idx <= len(t_grid) - 4:
        lo = max(0, best_idx - 7)
        hi = min(len(t_grid), best_idx + 8)
        try:
            interp = PchipInterpolator(t_grid[lo:hi], q_curve_vals[lo:hi])
            t_fine = np.linspace(t_grid[lo], t_grid[hi - 1], 2000)
            q_fine = interp(t_fine)
            t_star = float(t_fine[np.argmax(q_fine)])
        except Exception:
            pass  # fall back to grid result

    return t_star


def run_exp5(n_samples: int = 500_000, seed: int = 42) -> pd.DataFrame:
    """Run optimizer convergence experiment.

    Parameters
    ----------
    n_samples : int
    seed      : int

    Returns
    -------
    pd.DataFrame
    """
    os.makedirs("results", exist_ok=True)
    scenarios = get_scenarios()

    t_grid = np.linspace(0.001, 0.999, 501)
    left_alphas = [0.20, 0.10, 0.05, 0.02, 0.01]
    right_alphas = [0.80, 0.90, 0.95, 0.98, 0.99]

    records = []

    for sc_name, sc in scenarios.items():
        print(
            f"\n=== Scenario {sc_name} | regime={sc.regime}"
            f" | t*_left={sc.t_star_left:.6f}  t*_right={sc.t_star_right:.6f} ==="
        )

        X = simulate_lognormal(sc, n_samples, seed)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{sc_name}: Quantile curves and optimizer convergence")

        # ---- A-formula cross-check for interior regimes --------------------
        if sc.regime == "interior":
            u1_A, u2_A = optimizer_a_formula(sc.A)
            u1_b, u2_b = float(sc.b[0] / sc.beta), float(sc.b[1] / sc.beta)
            formula_err = abs(u1_A - u1_b)
            records.append(
                {
                    "scenario": sc_name,
                    "tail": "formula_check",
                    "alpha": np.nan,
                    "t_alpha_star": u1_A,
                    "t_pred": u1_b,
                    "opt_error": formula_err,
                    "note": "A-formula vs b/beta",
                }
            )
            print(
                f"  A-formula: u1={u1_A:.10f}  b/beta: u1={u1_b:.10f}"
                f"  err={formula_err:.2e}"
            )

        # ---- Left tail ------------------------------------------------------
        t_stars_left = []
        for alpha in left_alphas:
            q_c = quantile_curve(X, t_grid, alpha)
            t_a_star = find_optimizer(q_c, t_grid)
            opt_err = abs(t_a_star - sc.t_star_left)
            t_stars_left.append(t_a_star)

            records.append(
                {
                    "scenario": sc_name,
                    "tail": "left",
                    "alpha": alpha,
                    "t_alpha_star": t_a_star,
                    "t_pred": sc.t_star_left,
                    "opt_error": opt_err,
                }
            )
            print(
                f"  LEFT  alpha={alpha:.3f}  t*_alpha={t_a_star:.6f}"
                f"  t*_pred={sc.t_star_left:.6f}  err={opt_err:.6f}"
            )
            axes[0].plot(t_grid, q_c, alpha=0.7, label=f"α={alpha}")

        axes[0].axvline(
            sc.t_star_left, color="k", linestyle="--", linewidth=1.5,
            label=f"t*={sc.t_star_left:.4f}"
        )
        axes[0].set_xlabel("t")
        axes[0].set_title("Left-tail curves")
        axes[0].legend(fontsize=7)

        # ---- Right tail -----------------------------------------------------
        t_stars_right = []
        for alpha in right_alphas:
            q_c = quantile_curve(X, t_grid, alpha)
            t_a_star = find_optimizer(q_c, t_grid)
            opt_err = abs(t_a_star - sc.t_star_right)
            t_stars_right.append(t_a_star)

            records.append(
                {
                    "scenario": sc_name,
                    "tail": "right",
                    "alpha": alpha,
                    "t_alpha_star": t_a_star,
                    "t_pred": sc.t_star_right,
                    "opt_error": opt_err,
                }
            )
            print(
                f"  RIGHT alpha={alpha:.3f}  t*_alpha={t_a_star:.6f}"
                f"  t*_pred={sc.t_star_right:.6f}  err={opt_err:.6f}"
            )
            axes[1].plot(t_grid, q_c, alpha=0.7, label=f"α={alpha}")

        axes[1].axvline(
            sc.t_star_right, color="k", linestyle="--", linewidth=1.5,
            label=f"t*={sc.t_star_right:.4f}"
        )
        axes[1].set_xlabel("t")
        axes[1].set_title("Right-tail curves")
        axes[1].legend(fontsize=7)

        plt.tight_layout()
        plt.savefig(
            f"results/exp5_optimizer_{sc_name}.png", dpi=100, bbox_inches="tight"
        )
        plt.close()

    # ---- mu-robustness for S1 -----------------------------------------------
    sc1 = scenarios["S1"]
    fig_mu, ax_mu = plt.subplots(figsize=(8, 5))
    ax_mu.set_title("S1 left-tail optimizer: mu robustness (fixed A)")

    for mu_alt, ls in [((0, 0), "-"), ((0.5, -0.5), "--"), ((-0.5, 0.5), ":")]:
        sc_alt = build_scenario(f"S1_mu={mu_alt}", mu_alt, sc1.B)
        X_alt = simulate_lognormal(sc_alt, n_samples, seed + 10)
        t_stars_mu = []
        for alpha in left_alphas:
            q_c = quantile_curve(X_alt, t_grid, alpha)
            t_stars_mu.append(find_optimizer(q_c, t_grid))
        ax_mu.plot(left_alphas, t_stars_mu, ls + "o", label=f"mu={mu_alt}")
        for alpha, t_s in zip(left_alphas, t_stars_mu):
            records.append(
                {
                    "scenario": f"S1_mu_robustness",
                    "tail": "left",
                    "alpha": alpha,
                    "mu": str(mu_alt),
                    "t_alpha_star": t_s,
                    "t_pred": sc1.t_star_left,
                    "opt_error": abs(t_s - sc1.t_star_left),
                }
            )

    ax_mu.axhline(
        sc1.t_star_left, color="k", linestyle="--",
        label=f"t*={sc1.t_star_left:.4f}"
    )
    ax_mu.set_xlabel("alpha")
    ax_mu.set_ylabel("t*_alpha")
    ax_mu.legend()
    plt.tight_layout()
    plt.savefig("results/exp5_mu_robustness.png", dpi=100, bbox_inches="tight")
    plt.close()

    # ---- Global trajectory plot --------------------------------------------
    fig_traj, ax_traj = plt.subplots(figsize=(10, 5))
    ax_traj.set_title("Optimizer convergence (left tail) across scenarios")
    for sc_name, sc in scenarios.items():
        sc_left = [r for r in records if r.get("scenario") == sc_name
                   and r.get("tail") == "left" and "note" not in r]
        if not sc_left:
            continue
        alphas_l = [r["alpha"] for r in sc_left]
        t_stars_l = [r["t_alpha_star"] for r in sc_left]
        ax_traj.plot(alphas_l, t_stars_l, "o-", label=sc_name)
        ax_traj.axhline(sc.t_star_left, linestyle="--", alpha=0.4)
    ax_traj.set_xlabel("alpha")
    ax_traj.set_ylabel("t*_alpha")
    ax_traj.legend()
    plt.tight_layout()
    plt.savefig("results/exp5_optimizer_trajectory.png", dpi=100, bbox_inches="tight")
    plt.close()

    df = pd.DataFrame(records)
    df.to_csv("results/exp5_results.csv", index=False)
    print("\nExp5 results saved to results/exp5_results.csv")
    return df


if __name__ == "__main__":
    df = run_exp5()
    print(df.to_string())
    print("Experiment 5 complete.")
