"""Experiment 6: Subadditivity diagnostics for quantiles/VaR.

Tests portfolio quantile subadditivity using synthetic lognormal variables.

Gains-tail gap: Delta_alpha = q_alpha(X1 + X2) - q_alpha(X1) - q_alpha(X2)
Loss-tail gap (optional): VaR_alpha(L_sum) - VaR_alpha(L1) - VaR_alpha(L2)
  where L_i = X0_i - X_i, X0_i = E[X_i] = exp(mu_i + 0.5*A_ii)

The paper predicts gains-tail subadditivity (Delta <= 0) for large alpha in
regimes where the right-tail portfolio quantile is maximized at interior points,
consistent with the right-tail shape and optimizer analysis.
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


def run_exp6(n_samples: int = 500_000, seed: int = 42) -> pd.DataFrame:
    """Run subadditivity diagnostic experiment.

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

    alpha_grid = [0.90, 0.95, 0.975, 0.99, 0.995]
    records = []

    fig_main, axes_main = plt.subplots(
        2, len(scenarios), figsize=(5 * len(scenarios), 10)
    )
    fig_main.suptitle(
        "Gains-tail and loss-tail subadditivity gaps (Delta_alpha)", fontsize=12
    )

    for col_idx, (sc_name, sc) in enumerate(scenarios.items()):
        X = simulate_lognormal(sc, n_samples, seed)
        X1, X2 = X[:, 0], X[:, 1]
        X_sum = X1 + X2

        # Loss benchmark: E[Xi] = exp(mu_i + 0.5 * A_ii)
        X0_1 = float(np.exp(sc.mu[0] + 0.5 * sc.A[0, 0]))
        X0_2 = float(np.exp(sc.mu[1] + 0.5 * sc.A[1, 1]))
        L1 = X0_1 - X1
        L2 = X0_2 - X2
        L_sum = L1 + L2

        print(f"\n=== Scenario {sc_name} | X0=({X0_1:.4f},{X0_2:.4f}) ===")

        gains_deltas = []
        loss_deltas = []

        for alpha in alpha_grid:
            # Gains-tail
            q1 = float(np.quantile(X1, alpha))
            q2 = float(np.quantile(X2, alpha))
            q_sum = float(np.quantile(X_sum, alpha))
            delta_gains = q_sum - q1 - q2

            # Loss-tail VaR
            var_l1 = float(np.quantile(L1, alpha))
            var_l2 = float(np.quantile(L2, alpha))
            var_lsum = float(np.quantile(L_sum, alpha))
            delta_loss = var_lsum - var_l1 - var_l2

            gains_deltas.append(delta_gains)
            loss_deltas.append(delta_loss)

            record = {
                "scenario": sc_name,
                "alpha": alpha,
                "q_X1": q1,
                "q_X2": q2,
                "q_Xsum": q_sum,
                "delta_gains": delta_gains,
                "delta_gains_subadditive": delta_gains <= 0,
                "var_L1": var_l1,
                "var_L2": var_l2,
                "var_Lsum": var_lsum,
                "delta_loss": delta_loss,
                "delta_loss_subadditive": delta_loss <= 0,
                "X0_1": X0_1,
                "X0_2": X0_2,
            }
            records.append(record)

            sign_g = "≤0 ✓" if delta_gains <= 0 else ">0 ✗"
            sign_l = "≤0 ✓" if delta_loss <= 0 else ">0 ✗"
            print(
                f"  alpha={alpha:.3f}  gains_gap={delta_gains:.4f} {sign_g}"
                f"  loss_gap={delta_loss:.4f} {sign_l}"
            )

        # Gains plot
        ax_g = axes_main[0, col_idx]
        bar_colors = ["green" if d <= 0 else "red" for d in gains_deltas]
        ax_g.bar(
            [str(a) for a in alpha_grid],
            gains_deltas,
            color=bar_colors,
            alpha=0.75,
            edgecolor="k",
            linewidth=0.5,
        )
        ax_g.axhline(0, color="k", linewidth=0.8)
        ax_g.set_title(f"{sc_name}\nGains-tail gap Δ_α")
        ax_g.set_xlabel("α")
        ax_g.set_ylabel("q(X₁+X₂) − q(X₁) − q(X₂)")
        ax_g.tick_params(axis="x", labelsize=8)

        # Loss plot
        ax_l = axes_main[1, col_idx]
        bar_colors_l = ["green" if d <= 0 else "red" for d in loss_deltas]
        ax_l.bar(
            [str(a) for a in alpha_grid],
            loss_deltas,
            color=bar_colors_l,
            alpha=0.75,
            edgecolor="k",
            linewidth=0.5,
        )
        ax_l.axhline(0, color="k", linewidth=0.8)
        ax_l.set_title(f"{sc_name}\nLoss-tail gap Δ^loss_α")
        ax_l.set_xlabel("α")
        ax_l.set_ylabel("VaR(L₁+L₂) − VaR(L₁) − VaR(L₂)")
        ax_l.tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    plt.savefig("results/exp6_subadditivity.png", dpi=100, bbox_inches="tight")
    plt.close()

    # Summary per scenario
    fig_sum, ax_sum = plt.subplots(figsize=(10, 5))
    ax_sum.set_title("Gains-tail gap by scenario and alpha")
    width = 0.25
    sc_names = list(scenarios.keys())
    x_pos = np.arange(len(alpha_grid))

    for i, sc_name in enumerate(sc_names):
        sc_recs = [r for r in records if r["scenario"] == sc_name]
        deltas = [r["delta_gains"] for r in sorted(sc_recs, key=lambda r: r["alpha"])]
        ax_sum.bar(
            x_pos + i * width, deltas, width, label=sc_name, alpha=0.75
        )

    ax_sum.set_xticks(x_pos + width)
    ax_sum.set_xticklabels([str(a) for a in alpha_grid])
    ax_sum.axhline(0, color="k", linewidth=0.8)
    ax_sum.set_xlabel("alpha")
    ax_sum.set_ylabel("Gains-tail gap")
    ax_sum.legend()
    plt.tight_layout()
    plt.savefig("results/exp6_summary.png", dpi=100, bbox_inches="tight")
    plt.close()

    df = pd.DataFrame(records)
    df.to_csv("results/exp6_results.csv", index=False)
    print("\nExp6 results saved to results/exp6_results.csv")
    return df


if __name__ == "__main__":
    df = run_exp6()
    print(df.to_string())
    print("Experiment 6 complete.")
