"""Experiment 3: Corner-regime left-tail and right-tail asymptotics (S3).

Validates for scenario S3 (b2 < 0, corner regime):
- Left tail: q_alpha(t) ~ k * t  (linear, optimizer at t=1)
- Right tail: q_alpha(t) ~ k * (1-t)  (linear, optimizer at t=0)
- Left-tail concavity; right-tail convexity
- Corner dominance diagnostic
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress

from src.distributions import get_scenarios, simulate_lognormal
from src.quantiles import quantile_curve, finite_diff_hessian_1d
from src.diagnostics import compute_curvature_sign_fraction


def find_optimizer(q_curve_vals: np.ndarray, t_grid: np.ndarray) -> float:
    """Return t that maximises q_curve_vals (grid-based, includes boundaries)."""
    return float(t_grid[np.argmax(q_curve_vals)])


def run_exp3(n_samples: int = 500_000, seed: int = 42) -> pd.DataFrame:
    """Run corner-regime asymptotics experiment.

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
    sc = scenarios["S3"]

    print(
        f"=== Scenario S3 | b={sc.b} | beta={sc.beta:.4f}"
        f" | regime={sc.regime} ==="
    )
    print(f"  A =\n{sc.A}")
    print(f"  Corner dominance: A[1,1]={sc.A[1,1]:.4f} vs A[0,0]={sc.A[0,0]:.4f}")

    # Check corner dominance condition: b1^2/B11 > b2^2/B22 ?
    cond_val = sc.b[0] ** 2 / sc.B[0, 0] - sc.b[1] ** 2 / sc.B[1, 1]
    print(f"  Corner dominance b1^2/B11 - b2^2/B22 = {cond_val:.4f}")

    X = simulate_lognormal(sc, n_samples, seed)

    # Dense simplex grid including endpoints
    t_full = np.linspace(0.0, 1.0, 201)
    trim_mask = (t_full >= 0.02) & (t_full <= 0.98)

    left_alphas = [0.20, 0.10, 0.05, 0.02, 0.01, 0.005]
    right_alphas = [0.80, 0.90, 0.95, 0.98, 0.99, 0.995]

    records = []

    # ---- Left tail --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("S3 Corner regime: left-tail (→ t) and right-tail (→ 1-t) shapes")
    colors_l = plt.cm.Blues(np.linspace(0.35, 0.9, len(left_alphas)))

    for alpha, color in zip(left_alphas, colors_l):
        q_vals = quantile_curve(X, t_full, alpha)
        q_max = q_vals.max() + 1e-15
        q_norm = q_vals / q_max
        t_star_alpha = find_optimizer(q_vals, t_full)

        # Shape comparison: theoretical left-tail shape is t (normalized)
        # Target normalized: t / max(t on grid) = t/1.0 = t
        target = t_full
        shape_err = float(np.max(np.abs(q_norm[trim_mask] - target[trim_mask])))

        # Regression on t
        slope, intercept, r, _, _ = linregress(
            t_full[trim_mask], q_norm[trim_mask]
        )
        r2 = r ** 2

        # Curvature
        fd_hess = finite_diff_hessian_1d(q_vals, t_full)
        frac_neg = compute_curvature_sign_fraction(fd_hess[5:-5], sign=-1)

        record = {
            "tail": "left",
            "alpha": alpha,
            "t_star_alpha": t_star_alpha,
            "t_star_pred": 1.0,
            "opt_error": abs(t_star_alpha - 1.0),
            "shape_error_sup": shape_err,
            "regression_r2": r2,
            "regression_slope": slope,
            "regression_intercept": intercept,
            "frac_neg_curvature": frac_neg,
        }
        records.append(record)

        print(
            f"  LEFT  alpha={alpha:.3f}  t*_alpha={t_star_alpha:.3f}"
            f"  opt_err={abs(t_star_alpha-1.0):.3f}"
            f"  shape_err={shape_err:.4f}  R²={r2:.4f}  frac_neg={frac_neg:.2f}"
        )

        axes[0].plot(t_full, q_norm, color=color, label=f"α={alpha}")

    axes[0].plot(t_full, t_full, "k--", linewidth=2, label="q₀=t (theory)")
    axes[0].axvline(1.0, color="red", linestyle=":", linewidth=1, label="t*=1")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("q_α(t) / max")
    axes[0].set_title("Left-tail normalized curves")
    axes[0].legend(fontsize=7)

    # ---- Right tail -------------------------------------------------------
    colors_r = plt.cm.Reds(np.linspace(0.35, 0.9, len(right_alphas)))

    for alpha, color in zip(right_alphas, colors_r):
        q_vals = quantile_curve(X, t_full, alpha)
        q_max = q_vals.max() + 1e-15
        q_norm = q_vals / q_max
        t_star_alpha = find_optimizer(q_vals, t_full)

        # Shape comparison: theoretical right-tail shape is (1-t)
        target = 1.0 - t_full
        shape_err = float(np.max(np.abs(q_norm[trim_mask] - target[trim_mask])))

        # Regression on (1-t)
        slope, intercept, r, _, _ = linregress(
            1.0 - t_full[trim_mask], q_norm[trim_mask]
        )
        r2 = r ** 2

        fd_hess = finite_diff_hessian_1d(q_vals, t_full)
        frac_pos = compute_curvature_sign_fraction(fd_hess[5:-5], sign=+1)

        record = {
            "tail": "right",
            "alpha": alpha,
            "t_star_alpha": t_star_alpha,
            "t_star_pred": 0.0,
            "opt_error": abs(t_star_alpha - 0.0),
            "shape_error_sup": shape_err,
            "regression_r2": r2,
            "regression_slope": slope,
            "regression_intercept": intercept,
            "frac_neg_curvature": frac_pos,  # frac positive curvature for right tail
        }
        records.append(record)

        print(
            f"  RIGHT alpha={alpha:.3f}  t*_alpha={t_star_alpha:.3f}"
            f"  opt_err={abs(t_star_alpha-0.0):.3f}"
            f"  shape_err={shape_err:.4f}  R²={r2:.4f}  frac_pos={frac_pos:.2f}"
        )

        axes[1].plot(t_full, q_norm, color=color, label=f"α={alpha}")

    axes[1].plot(
        t_full, 1.0 - t_full, "k--", linewidth=2, label="q₀=1-t (theory)"
    )
    axes[1].axvline(0.0, color="blue", linestyle=":", linewidth=1, label="t*=0")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("q_α(t) / max")
    axes[1].set_title("Right-tail normalized curves")
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig("results/exp3_corner_shapes.png", dpi=100, bbox_inches="tight")
    plt.close()

    # ---- Optimizer trajectory plot -----------------------------------------
    fig_traj, ax_traj = plt.subplots(figsize=(8, 5))
    left_recs = [r for r in records if r["tail"] == "left"]
    right_recs = [r for r in records if r["tail"] == "right"]

    ax_traj.plot(
        [r["alpha"] for r in left_recs],
        [r["t_star_alpha"] for r in left_recs],
        "b-o", label="Left-tail optimizer"
    )
    ax_traj.plot(
        [r["alpha"] for r in right_recs],
        [r["t_star_alpha"] for r in right_recs],
        "r-s", label="Right-tail optimizer"
    )
    ax_traj.axhline(1.0, color="blue", linestyle="--", alpha=0.5, label="t*=1 (left)")
    ax_traj.axhline(0.0, color="red", linestyle="--", alpha=0.5, label="t*=0 (right)")
    ax_traj.set_xlabel("alpha")
    ax_traj.set_ylabel("t*_alpha")
    ax_traj.set_title("S3 corner-regime: optimizer trajectory")
    ax_traj.legend()
    plt.tight_layout()
    plt.savefig("results/exp3_optimizer_trajectory.png", dpi=100, bbox_inches="tight")
    plt.close()

    df = pd.DataFrame(records)
    df.to_csv("results/exp3_results.csv", index=False)
    print("\nExp3 results saved to results/exp3_results.csv")
    return df


if __name__ == "__main__":
    df = run_exp3()
    print(df.to_string())
    print("Experiment 3 complete.")
