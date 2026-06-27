"""
fig_cold_start_sweep.py — Slide 6 cold-start line plot.

Shows AUROC as a function of training set size for HPS, HPS-Euclidean, and C4.
This is the regime where geometric priors should help most (low N).

Key takeaway: HPS > HPS-Euclidean (geometric prior helps over flat) at all N,
but C4 dominates throughout (linear probe wins via simplicity).

Usage:
    python fig_cold_start_sweep.py
    python fig_cold_start_sweep.py --output figures_for_meeting/fig_cold_start.png
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


# Default values from hyperbolic_vs_euclidean_diverse.json (Part B sweep)
DEFAULT_SWEEP = {
    "N": [45, 90, 225, 450, 900, 2250],
    "HPS":            [0.7012, 0.8232, 0.9459, 0.9004, 0.9792, 0.9722],
    "HPS-Euclidean":  [0.6093, 0.6139, 0.8865, 0.8293, 0.9436, 0.9266],
    "C4":             [0.9614, 0.9784, 0.9807, 0.9892, 0.9907, 0.9931],
}


def load_from_json(json_path):
    """Try to load actual sweep values from JSON. Returns dict or None."""
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path) as f:
            d = json.load(f)
        if "part_b_sweep" in d:
            sweep = d["part_b_sweep"]
            return {
                "N":             [r["n"] for r in sweep],
                "HPS":           [r["hps_auroc"] for r in sweep],
                "HPS-Euclidean": [r["euc_auroc"] for r in sweep],
                "C4":            [r["c4_auroc"] for r in sweep],
            }
    except Exception as e:
        print(f"  [warn] Could not load JSON ({e}); using defaults.")
    return None


def make_figure(data, output_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    N = data["N"]
    methods = [
        ("HPS", "#9b59b6", "o", "-"),
        ("HPS-Euclidean", "#95a5a6", "s", "--"),
        ("C4", "#2ecc71", "^", "-"),
    ]

    for name, color, marker, linestyle in methods:
        ax.plot(N, data[name], marker=marker, linestyle=linestyle, color=color,
                linewidth=2.5, markersize=9, label=name, markeredgecolor="black",
                markeredgewidth=0.8)

    ax.set_xscale("log")
    ax.set_xlabel("Training attacks (N)", fontsize=12, fontweight="bold")
    ax.set_ylabel("AUROC (test set)", fontsize=12, fontweight="bold")
    ax.set_title("Cold-start sweep: low-N is where geometric priors should help most\n"
                 "Result: HPS > HPS-Euclidean, but C4 dominates throughout",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0.55, 1.02)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=11, loc="lower right", framealpha=0.95)

    # Annotate the largest gap (N=45)
    ax.annotate(
        f"At N=45:\nC4 - HPS = +{data['C4'][0] - data['HPS'][0]:.2f} AUROC\n"
        f"HPS - Euclidean = +{data['HPS'][0] - data['HPS-Euclidean'][0]:.2f}",
        xy=(45, 0.62), xytext=(120, 0.65),
        fontsize=9, ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef9e7", edgecolor="#f39c12"),
        arrowprops=dict(arrowstyle="->", color="#f39c12", lw=1.0),
    )

    # Tick labels at actual N values
    ax.set_xticks(N)
    ax.set_xticklabels([str(n) for n in N])

    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"  saved -> {output_path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results/hyperbolic_vs_euclidean_diverse.json",
                   help="Path to JSON with cold-start sweep (optional, will use defaults)")
    p.add_argument("--output", default="figures_for_meeting/fig_cold_start_sweep.png")
    args = p.parse_args()

    loaded = load_from_json(args.input)
    data = loaded if loaded else DEFAULT_SWEEP

    print("\n  Cold-start values used:")
    print(f"    {'N':>6}  {'HPS':>8}  {'HPS-Euc':>8}  {'C4':>8}")
    for i, n in enumerate(data["N"]):
        print(f"    {n:>6}  {data['HPS'][i]:>8.4f}  {data['HPS-Euclidean'][i]:>8.4f}  {data['C4'][i]:>8.4f}")
    print()

    make_figure(data, args.output)


if __name__ == "__main__":
    main()
