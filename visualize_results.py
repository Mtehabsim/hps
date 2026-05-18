"""
HPS Visualization — Results Summary for Mentor Presentation
═════════════════════════════════════════════════════════════
Generates a publication-quality 4-panel figure summarizing the HPS findings.

Usage:
  python visualize_results.py
Output:
  figures/hps_summary.png
  figures/hps_concept.png
  figures/hps_per_attack.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Color palette
COLOR_HPS = "#2E86AB"      # blue (highlight)
COLOR_RTV = "#A23B72"      # magenta
COLOR_EUC = "#888888"      # grey (ablation)
COLOR_BENIGN = "#5BA659"   # green
COLOR_ATTACK = "#D62828"   # red


# ═══════════════════════════════════════════════════════════════════════════
#  Results data (from experiment8 + experiment10)
# ═══════════════════════════════════════════════════════════════════════════

methods = ["Euclidean\n(ablation)", "RTV\n(Derya & Sunar 2026)", "HPS-Full\n(ours)"]
colors = [COLOR_EUC, COLOR_RTV, COLOR_HPS]

# Cross-attack mean AUROC
auroc_mean = [0.513, 0.769, 0.815]
tpr_mean = [0.000, 0.047, 0.236]

# Per-method AUROC
attack_methods = ["GCG", "JBC", "PAIR", "Random Search"]
auroc_per = {
    "Euclidean": [0.769, 0.548, 0.445, 0.291],
    "RTV":       [0.833, 0.583, 0.801, 0.860],
    "HPS-Full":  [0.943, 0.753, 0.635, 0.930],
}
tpr_per = {
    "Euclidean": [0.000, 0.000, 0.000, 0.000],
    "RTV":       [0.000, 0.000, 0.015, 0.173],
    "HPS-Full":  [0.603, 0.167, 0.000, 0.173],
}


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 1: Headline result — 4 panels
# ═══════════════════════════════════════════════════════════════════════════

def make_summary_figure():
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.30)

    # ── Panel 1: Cross-Attack AUROC (top-left) ──
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(methods, auroc_mean, color=colors, edgecolor="black", linewidth=0.8)
    ax1.set_ylabel("Cross-Attack AUROC")
    ax1.set_title("(a) Cross-Attack Generalization (AUROC)", fontweight="bold")
    ax1.set_ylim(0, 1.0)
    ax1.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    for bar, v in zip(bars, auroc_mean):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                 f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax1.text(0.02, 0.55, "random", color="gray", fontsize=8, transform=ax1.get_yaxis_transform())

    # ── Panel 2: Cross-Attack TPR@FPR=1% (top-right) ──
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(methods, tpr_mean, color=colors, edgecolor="black", linewidth=0.8)
    ax2.set_ylabel("TPR @ FPR = 1%")
    ax2.set_title("(b) Deployment-Threshold Performance", fontweight="bold")
    ax2.set_ylim(0, 0.30)
    for bar, v in zip(bars, tpr_mean):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                 f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    # Highlight HPS advantage
    ax2.annotate(f"5×\nadvantage",
                 xy=(2, 0.236), xytext=(1.5, 0.27),
                 fontsize=9, ha="center", color="black",
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    # ── Panel 3: Per-Attack AUROC heatmap (bottom-left) ──
    ax3 = fig.add_subplot(gs[1, 0])
    data = np.array([
        auroc_per["Euclidean"],
        auroc_per["RTV"],
        auroc_per["HPS-Full"],
    ])
    im = ax3.imshow(data, cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect="auto")
    ax3.set_xticks(range(len(attack_methods)))
    ax3.set_xticklabels(attack_methods, rotation=20, ha="right")
    ax3.set_yticks(range(3))
    ax3.set_yticklabels(["Euclidean", "RTV", "HPS-Full"])
    ax3.set_title("(c) AUROC per Held-Out Attack Method", fontweight="bold")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax3.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center",
                     color="black", fontsize=10, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax3, fraction=0.04, pad=0.04)
    cbar.set_label("AUROC", fontsize=9)

    # ── Panel 4: TPR@FPR=1% per attack (bottom-right) ──
    ax4 = fig.add_subplot(gs[1, 1])
    width = 0.27
    x = np.arange(len(attack_methods))
    ax4.bar(x - width, tpr_per["Euclidean"], width, label="Euclidean", color=COLOR_EUC, edgecolor="black", linewidth=0.6)
    ax4.bar(x,         tpr_per["RTV"],       width, label="RTV",       color=COLOR_RTV, edgecolor="black", linewidth=0.6)
    ax4.bar(x + width, tpr_per["HPS-Full"],  width, label="HPS-Full",  color=COLOR_HPS, edgecolor="black", linewidth=0.6)
    ax4.set_xticks(x)
    ax4.set_xticklabels(attack_methods, rotation=20, ha="right")
    ax4.set_ylabel("TPR @ FPR = 1%")
    ax4.set_title("(d) Deployment Performance per Attack", fontweight="bold")
    ax4.legend(loc="upper right", framealpha=0.95)
    ax4.set_ylim(0, 0.7)

    fig.suptitle("Hyperbolic Physiological Sentinel (HPS) — Cross-Attack Generalization on Vicuna-13B",
                 fontweight="bold", y=0.995)
    out_path = os.path.join(OUT_DIR, "hps_summary.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 2: Conceptual diagram — Hyperbolic vs Euclidean separation
# ═══════════════════════════════════════════════════════════════════════════

def make_concept_figure():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    np.random.seed(42)
    n = 60

    # Generate fake data: benign near origin, attacks toward boundary
    # Euclidean view: linear separation (overfits direction-specific)
    benign_e = np.random.randn(n, 2) * 0.4
    # Three "attack types" come from different directions
    angles = [0.4, 1.6, 2.8]  # different angular positions
    attacks_e = []
    attack_labels = ["GCG", "JBC", "PAIR"]
    for ang in angles:
        cluster = np.array([[np.cos(ang), np.sin(ang)]]) * np.random.uniform(2.0, 3.0, (n // 3, 1))
        cluster += np.random.randn(n // 3, 2) * 0.3
        attacks_e.append(cluster)

    # ── Panel 1: Euclidean ──
    ax1 = axes[0]
    ax1.scatter(benign_e[:, 0], benign_e[:, 1], c=COLOR_BENIGN, s=40, alpha=0.7,
                edgecolor="darkgreen", linewidth=0.5, label="Benign")
    attack_colors_eu = ["#D62828", "#F77F00", "#FCBF49"]
    for i, (atk, lbl) in enumerate(zip(attacks_e, attack_labels)):
        ax1.scatter(atk[:, 0], atk[:, 1], c=attack_colors_eu[i], s=40, alpha=0.7,
                    edgecolor="darkred", linewidth=0.5, label=f"{lbl} attacks")
    # Draw a Euclidean "decision boundary" that catches GCG but misses PAIR
    boundary_x = np.linspace(-3.5, 3.5, 100)
    boundary_y = -0.4 * boundary_x + 0.5  # arbitrary linear boundary fit to GCG
    ax1.plot(boundary_x, boundary_y, "k--", linewidth=2, label="Trained decision boundary")
    # Annotate the failure
    ax1.annotate("PAIR attacks\nfall on benign side\n→ negative transfer",
                 xy=(attacks_e[2][:, 0].mean(), attacks_e[2][:, 1].mean()),
                 xytext=(-3.5, -2),
                 fontsize=10, color="darkred",
                 arrowprops=dict(arrowstyle="->", color="darkred", lw=1.2),
                 ha="left")
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-3, 3.5)
    ax1.set_aspect("equal")
    ax1.set_title("Euclidean Projection (catastrophic cross-attack failure)",
                  fontweight="bold")
    ax1.set_xlabel("Projected dim 1")
    ax1.set_ylabel("Projected dim 2")
    ax1.legend(loc="upper right", framealpha=0.95, fontsize=8)
    ax1.grid(alpha=0.3)

    # ── Panel 2: Hyperbolic ──
    ax2 = axes[1]
    # Poincaré disk visualization
    theta = np.linspace(0, 2 * np.pi, 200)
    ax2.plot(np.cos(theta), np.sin(theta), "k-", linewidth=2)  # boundary

    # Benign at small radius (interior), attacks at large radius (near boundary)
    benign_r = np.random.uniform(0, 0.3, n)
    benign_theta = np.random.uniform(0, 2 * np.pi, n)
    benign_h = np.column_stack([benign_r * np.cos(benign_theta),
                                 benign_r * np.sin(benign_theta)])

    # Attacks at high radius BUT distributed in different angles
    attacks_h = []
    for ang in angles:
        atk_r = np.random.uniform(0.75, 0.92, n // 3)
        atk_theta = ang + np.random.randn(n // 3) * 0.25
        atk = np.column_stack([atk_r * np.cos(atk_theta),
                                atk_r * np.sin(atk_theta)])
        attacks_h.append(atk)

    ax2.scatter(benign_h[:, 0], benign_h[:, 1], c=COLOR_BENIGN, s=40, alpha=0.7,
                edgecolor="darkgreen", linewidth=0.5, label="Benign (near origin)")
    for i, (atk, lbl) in enumerate(zip(attacks_h, attack_labels)):
        ax2.scatter(atk[:, 0], atk[:, 1], c=attack_colors_eu[i], s=40, alpha=0.7,
                    edgecolor="darkred", linewidth=0.5, label=f"{lbl} attacks")
    # Radial boundary
    boundary = plt.Circle((0, 0), 0.55, color="black", fill=False,
                          linestyle="--", linewidth=2)
    ax2.add_patch(boundary)
    ax2.text(0, -0.65, "Radial threshold\n(direction-agnostic)",
             ha="center", fontsize=9, fontweight="bold")
    # Annotate the success
    ax2.annotate("All attacks at\nhigh radius regardless\nof direction\n→ generalizes",
                 xy=(0.7, 0.7), xytext=(1.05, 0.3),
                 fontsize=9, color="darkblue",
                 arrowprops=dict(arrowstyle="->", color="darkblue", lw=1.2),
                 ha="left")
    ax2.set_xlim(-1.4, 2.0)
    ax2.set_ylim(-1.3, 1.5)
    ax2.set_aspect("equal")
    ax2.set_title("Hyperbolic Projection (radial separation generalizes)",
                  fontweight="bold")
    ax2.set_xlabel("x₁")
    ax2.set_ylabel("x₂")
    ax2.legend(loc="upper right", framealpha=0.95, fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle("Why Hyperbolic Geometry Helps Cross-Attack Generalization",
                 fontweight="bold", y=1.02)
    out_path = os.path.join(OUT_DIR, "hps_concept.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 3: Pipeline diagram
# ═══════════════════════════════════════════════════════════════════════════

def make_pipeline_figure():
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.set_aspect("equal")
    ax.axis("off")

    boxes = [
        (0.5, 1.5, 2.0, 1.0, "Input prompt", "#F4F1DE"),
        (3.0, 1.5, 2.0, 1.0, "Vicuna-13B\nforward pass", "#E8E1D5"),
        (5.5, 1.5, 2.2, 1.0, "Layer activations\n(8 layers × 5120)", "#DBE7E4"),
        (8.2, 1.5, 2.2, 1.0, "Lorentz lift\n+ contrastive\nprojection", COLOR_HPS),
        (10.9, 1.5, 1.6, 1.0, "12 trajectory\nfeatures", "#B8D4E3"),
        (12.7, 1.5, 1.0, 1.0, "Logistic\nregression", "#C8B8DB"),
    ]
    for x, y, w, h, txt, color in boxes:
        face = color
        text_color = "white" if color == COLOR_HPS else "black"
        rect = plt.Rectangle((x, y), w, h, facecolor=face, edgecolor="black",
                             linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center",
                fontsize=10, color=text_color, fontweight="bold" if color == COLOR_HPS else "normal")

    arrow_y = 2.0
    arrow_xs = [(2.5, 3.0), (5.0, 5.5), (7.7, 8.2), (10.4, 10.9), (12.5, 12.7)]
    for x1, x2 in arrow_xs:
        ax.annotate("", xy=(x2, arrow_y), xytext=(x1, arrow_y),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    ax.text(9.3, 0.7, "Hyperbolic geometry\n(novel)", ha="center", color=COLOR_HPS,
            fontweight="bold", fontsize=10)
    ax.annotate("", xy=(9.3, 1.4), xytext=(9.3, 1.0),
                arrowprops=dict(arrowstyle="->", color=COLOR_HPS, lw=2))

    # Output
    ax.text(13.2, 0.5, "BLOCK / ALLOW", fontsize=11, fontweight="bold", ha="center")
    ax.annotate("", xy=(13.2, 1.0), xytext=(13.2, 1.4),
                arrowprops=dict(arrowstyle="<-", color="black", lw=1.5))

    ax.set_title("HPS Pipeline: Activation Trajectory Detection in Hyperbolic Space",
                 fontweight="bold", fontsize=13)

    out_path = os.path.join(OUT_DIR, "hps_pipeline.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    print(f"Generating figures in {OUT_DIR}/")
    make_summary_figure()
    make_concept_figure()
    make_pipeline_figure()
    print("\nAll figures generated.")
    print(f"  • {OUT_DIR}/hps_summary.png   — 4-panel results comparison")
    print(f"  • {OUT_DIR}/hps_concept.png   — Why hyperbolic helps (geometric intuition)")
    print(f"  • {OUT_DIR}/hps_pipeline.png  — Method pipeline diagram")
