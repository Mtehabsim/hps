"""
fig_4method_comparison.py — Slide 6 main bar chart.

Generates a 2-panel bar chart comparing HPS, HPS-Euclidean, C4, and Anthropic MTP.
Left panel:  AUROC.
Right panel: TPR @ 5% FPR.

Colors follow slide style guide:
  HPS:         purple  (#9b59b6)
  HPS-Euc:     gray    (#95a5a6)
  C4:          green   (#2ecc71)
  MTP:         dark green (#27ae60)

Usage:
    python fig_4method_comparison.py
    python fig_4method_comparison.py --output figures_for_meeting/fig_4method.png
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


# Default values from your experimental results (Llama-3-8B-Instruct, fixed cache)
# Source: hyperbolic_vs_euclidean_diverse.json + anthropic_mtp_llama3.json
DEFAULT_DATA = {
    "HPS\n(Lorentz)": {
        "auroc": 0.9905,
        "auroc_std": 0.0011,
        "tpr": 0.9802,
        "tpr_std": 0.003,
        "params": "262K",
        "color": "#9b59b6",
    },
    "HPS-Euclidean\n(matched)": {
        "auroc": 0.9680,
        "auroc_std": 0.0022,
        "tpr": 0.9311,
        "tpr_std": 0.005,
        "params": "262K",
        "color": "#95a5a6",
    },
    "C4\n(linear probe)": {
        "auroc": 0.9984,
        "auroc_std": 0.0000,
        "tpr": 0.9954,
        "tpr_std": 0.000,
        "params": "4097",
        "color": "#2ecc71",
    },
    "MTP\n(Anthropic)": {
        "auroc": 0.9988,
        "auroc_std": 0.0001,
        "tpr": 0.9946,
        "tpr_std": 0.001,
        "params": "4097",
        "color": "#27ae60",
    },
}


def load_from_json(json_paths):
    """Try to load actual values from JSON results files. Returns updated dict or None."""
    data = {k: dict(v) for k, v in DEFAULT_DATA.items()}
    try:
        # Try hyperbolic_vs_euclidean_diverse.json (HPS, HPS-Euc, C4)
        if "hyp_vs_euc" in json_paths and os.path.exists(json_paths["hyp_vs_euc"]):
            with open(json_paths["hyp_vs_euc"]) as f:
                d = json.load(f)
            # Expected: d["part_a"]["HPS (Lorentz)"]["auroc_mean"], etc.
            # Fall back gracefully if structure differs
            if "part_a" in d:
                pa = d["part_a"]
                for our_key, file_key in [
                    ("HPS\n(Lorentz)", "HPS (Lorentz)"),
                    ("HPS-Euclidean\n(matched)", "HPS-Euclidean (matched)"),
                    ("C4\n(linear probe)", "C4 (linear probe)"),
                ]:
                    if file_key in pa:
                        data[our_key]["auroc"] = pa[file_key].get("auroc_mean", data[our_key]["auroc"])
                        data[our_key]["auroc_std"] = pa[file_key].get("auroc_std", data[our_key]["auroc_std"])
                        data[our_key]["tpr"] = pa[file_key].get("tpr_mean", data[our_key]["tpr"])
        # Try anthropic_mtp_llama3.json
        if "mtp" in json_paths and os.path.exists(json_paths["mtp"]):
            with open(json_paths["mtp"]) as f:
                d = json.load(f)
            if "mtp_per_layer" in d:
                best = max(d["mtp_per_layer"], key=lambda r: r.get("auroc", 0))
                data["MTP\n(Anthropic)"]["auroc"] = best.get("auroc", data["MTP\n(Anthropic)"]["auroc"])
                data["MTP\n(Anthropic)"]["tpr"] = best.get("tpr_at_5pct_fpr", data["MTP\n(Anthropic)"]["tpr"])
    except Exception as e:
        print(f"  [warn] Could not load JSON ({e}); using defaults.")
    return data


def make_figure(data, output_path):
    methods = list(data.keys())
    aurocs = [data[m]["auroc"] for m in methods]
    auroc_stds = [data[m]["auroc_std"] for m in methods]
    tprs = [data[m]["tpr"] for m in methods]
    tpr_stds = [data[m]["tpr_std"] for m in methods]
    colors = [data[m]["color"] for m in methods]
    params = [data[m]["params"] for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    x = np.arange(len(methods))

    # AUROC panel — zoomed to make differences visible
    bars1 = ax1.bar(x, aurocs, yerr=auroc_stds, capsize=4,
                    color=colors, edgecolor="black", linewidth=1.0)
    ax1.set_ylabel("AUROC", fontsize=12, fontweight="bold")
    ax1.set_title("AUROC (Llama-3-8B-Instruct, fixed cache)", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=9)
    ax1.set_ylim(0.94, 1.005)
    ax1.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    for i, (bar, val) in enumerate(zip(bars1, aurocs)):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
                 f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")
    # Annotate parameter count under each bar
    for i, p in enumerate(params):
        ax1.text(i, 0.943, f"{p} params", ha="center", fontsize=8, color="dimgray")

    # TPR @ 5% FPR panel
    bars2 = ax2.bar(x, tprs, yerr=tpr_stds, capsize=4,
                    color=colors, edgecolor="black", linewidth=1.0)
    ax2.set_ylabel("TPR @ 5% FPR", fontsize=12, fontweight="bold")
    ax2.set_title("TPR @ 5% FPR", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=9)
    ax2.set_ylim(0.88, 1.01)
    ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    for i, (bar, val) in enumerate(zip(bars2, tprs)):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                 f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("Four-method comparison: HPS does not beat linear probes",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"  saved -> {output_path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hyp_vs_euc", default="results/hyperbolic_vs_euclidean_diverse.json",
                   help="Path to hyperbolic_vs_euclidean_diverse.json (optional)")
    p.add_argument("--mtp", default="results/anthropic_mtp_llama3.json",
                   help="Path to anthropic_mtp_llama3.json (optional)")
    p.add_argument("--output", default="figures_for_meeting/fig_4method_comparison.png")
    args = p.parse_args()

    data = load_from_json({"hyp_vs_euc": args.hyp_vs_euc, "mtp": args.mtp})

    print("\n  Method values used:")
    for m, d in data.items():
        m_clean = m.replace("\n", " ")
        print(f"    {m_clean:30s}  AUROC={d['auroc']:.4f}  TPR={d['tpr']:.4f}  ({d['params']} params)")
    print()

    make_figure(data, args.output)


if __name__ == "__main__":
    main()
