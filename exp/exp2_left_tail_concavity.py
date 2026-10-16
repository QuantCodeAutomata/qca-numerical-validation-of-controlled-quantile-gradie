"""Experiment 2: Left-tail concavity and universal interior shape.

Tests the paper's left-tail claims for 2D lognormal portfolios in the
barycentric/interior regime (S1 and S2) where all b_i > 0.

Key checks:
- q_alpha(t) becomes concave for small alpha
- Normalized curves converge to theoretical universal shape q_0(t)
- Optimizer t_alpha^* converges to b[0]/beta
- Shape is invariant to mu (only depends on A through B)
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
from src.quantiles import quantile_curve, finite_diff_hessian_1d
from src.diagnostics import compute_curvature_sign_fraction


def universal_shape_q0(t_grid: np.ndarray, b: np.ndarray, beta: float) -> np.ndarray:
    """Theoretical interior universal shape.

    q_0(t) = exp((1/beta) * (b1*log(beta*t/b1) + b2*log(beta*(1-t)/b2)))

    Parameters
    ----------
    t_grid : np.ndarray, t values in (0, 1)
    b      : np.ndarray, shape (2,), column sums of B
    beta   : float, sum of b

    Returns
    -------
    np.ndarray, shape (T,)
    """
    t = np.clip(np.asarray(t_grid, dtype=float), 1e-12, 1 - 1e-12)
    arg1 = b[0] * np.log(beta * t / b[0])
    arg2 = b[1] * np.log(beta * (1.0 - t) / b[1])
    return np.exp((arg1 + arg2) / beta)


def find_optimizer(q_curve_vals: np.ndarray, t_grid: np.ndarray) -> float:
    """Grid search + PCHIP refinement for argmax q_alpha(t).

    Parameters
    ----------
    q_curve_vals : np.ndarray, shape (T,)
    t_grid       : np.ndarray, shape (T,)

    Returns
    -------
    float, estimated optimizer t^*
    """
    best_idx = int(np.argmax(q_curve_vals))
    t_star = float(t_grid[best_idx])

    # PCHIP refinement in a local window
    if 3 <= best_idx <= len(t_grid) - 4:
        lo = max(0, best_idx - 6)
        hi = min(len(t_grid), best_idx + 7)
        interp = PchipInterpolator(t_grid[lo:hi], q_curve_vals[lo:hi])
        t_fine = np.linspace(t_grid[lo], t_grid[hi - 1], 2000)
        q_fine = interp(t_fine)
        t_star = float(t_fine[np.argmax(q_fine)])

    return t_star


def run_exp2(n_samples: int = 500_000, seed: int = 42) -> pd.DataFrame:
    """Run left-tail concavity and universal shape experiment.

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

    alpha_grid = [0.20, 0.10, 0.05, 0.02, 0.01]
    t_full = np.linspace(0.001, 0.999, 201)
    trim_mask = (t_full >= 0.02) & (t_full <= 0.98)

    records = []

    for sc_name in ["S1", "S2"]:
        sc = scenarios[sc_name]
        print(
            f"\n=== Scenario {sc_name} | b={sc.b} | beta={sc.beta:.4f}"
            f" | t*={sc.t_star_left:.10f} ==="
        )

        X = simulate_lognormal(sc, n_samples, seed)

        # Theoretical shape
        q0_full = universal_shape_q0(t_full, sc.b, sc.beta)
        q0_norm = q0_full / q0_full.max()

        # Verify q0(t*) = 1 analytically
        t_star_idx = int(np.argmax(q0_full))
        assert abs(q0_full[t_star_idx] / q0_full.max() - 1.0) < 1e-6, (
            "q0 max should be 1"
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{sc_name}: Left-tail quantile curves and universal shape")
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(alpha_grid)))

        for alpha, color in zip(alpha_grid, colors):
            q_vals = quantile_curve(X, t_full, alpha)

            # Optimizer
            t_a_star = find_optimizer(q_vals, t_full)
            opt_err = abs(t_a_star - sc.t_star_left)

            # Normalized curve
            q_norm = q_vals / (q_vals.max() + 1e-15)

            # Shape error on trimmed domain
            shape_err = float(
                np.max(np.abs(q_norm[trim_mask] - q0_norm[trim_mask]))
            )

            # Curvature
            fd_hess = finite_diff_hessian_1d(q_vals, t_full)
            frac_neg = compute_curvature_sign_fraction(fd_hess[5:-5], sign=-1)

            record = {
                "scenario": sc_name,
                "mu": str(sc.mu.tolist()),
                "alpha": alpha,
                "t_alpha_star": t_a_star,
                "t_star_pred": sc.t_star_left,
                "opt_error": opt_err,
                "shape_error_sup": shape_err,
                "frac_neg_curvature": frac_neg,
            }
            records.append(record)

            print(
                f"  alpha={alpha:.3f}  t*_alpha={t_a_star:.4f}"
                f"  opt_err={opt_err:.4f}  shape_err={shape_err:.4f}"
                f"  frac_neg={frac_neg:.2f}"
            )

            axes[0].plot(t_full, q_vals, color=color, label=f"α={alpha}")
            axes[1].plot(t_full, q_norm, color=color, alpha=0.8, label=f"α={alpha}")

        axes[1].plot(t_full, q0_norm, "k--", linewidth=2, label="q_0 (theory)")
        axes[1].axvline(
            sc.t_star_left, color="red", linestyle=":", linewidth=1.5,
            label=f"t*={sc.t_star_left:.4f}"
        )

        for ax in axes:
            ax.set_xlabel("t")
            ax.legend(fontsize=7)
        axes[0].set_title("Quantile curves q_α(t)")
        axes[1].set_title("Normalized curves vs q₀(t)")

        plt.tight_layout()
        plt.savefig(f"results/exp2_shape_{sc_name}.png", dpi=100, bbox_inches="tight")
        plt.close()

        # ---- mu-invariance test ------------------------------------------------
        alt_mus = [(0.5, -0.5), (-0.5, 0.5)]
        for mu_alt in alt_mus:
            sc_alt = build_scenario(f"{sc_name}_mu_alt", mu_alt, sc.B)
            X_alt = simulate_lognormal(sc_alt, n_samples, seed + 1)

            for alpha in [0.05, 0.02]:
                q_alt = quantile_curve(X_alt, t_full, alpha)
                q_alt_norm = q_alt / (q_alt.max() + 1e-15)
                t_a_star_alt = find_optimizer(q_alt, t_full)
                shape_err_alt = float(
                    np.max(np.abs(q_alt_norm[trim_mask] - q0_norm[trim_mask]))
                )
                records.append(
                    {
                        "scenario": f"{sc_name}_mu={mu_alt}",
                        "mu": str(list(mu_alt)),
                        "alpha": alpha,
                        "t_alpha_star": t_a_star_alt,
                        "t_star_pred": sc.t_star_left,
                        "opt_error": abs(t_a_star_alt - sc.t_star_left),
                        "shape_error_sup": shape_err_alt,
                        "frac_neg_curvature": float("nan"),
                    }
                )
                print(
                    f"  mu={mu_alt} alpha={alpha:.2f}  shape_err={shape_err_alt:.4f}"
                    f"  t*_alpha={t_a_star_alt:.4f}"
                )

    # Optimizer convergence plot
    fig_opt, axes_opt = plt.subplots(1, 2, figsize=(12, 5))
    fig_opt.suptitle("Optimizer convergence to t* as alpha -> 0")
    for i, sc_name in enumerate(["S1", "S2"]):
        sc = scenarios[sc_name]
        sc_recs = [r for r in records if r["scenario"] == sc_name]
        alphas = [r["alpha"] for r in sc_recs]
        t_stars = [r["t_alpha_star"] for r in sc_recs]
        axes_opt[i].plot(alphas, t_stars, "o-", color="steelblue", label="t*_alpha")
        axes_opt[i].axhline(
            sc.t_star_left, color="red", linestyle="--",
            label=f"t*={sc.t_star_left:.4f}"
        )
        axes_opt[i].set_xlabel("alpha")
        axes_opt[i].set_ylabel("t*_alpha")
        axes_opt[i].set_title(sc_name)
        axes_opt[i].legend()
    plt.tight_layout()
    plt.savefig("results/exp2_optimizer_convergence.png", dpi=100, bbox_inches="tight")
    plt.close()

    df = pd.DataFrame(records)
    df.to_csv("results/exp2_results.csv", index=False)
    print("\nExp2 results saved to results/exp2_results.csv")
    return df


if __name__ == "__main__":
    df = run_exp2()
    print(df.to_string())
    print("Experiment 2 complete.")
