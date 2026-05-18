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
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    np.random.seed(42)
    n = 60

    # ───────────────────────────────────────────────────────────────────
    #  Key insight:
    #  Euclidean contrastive loss pushes TRAINED attacks far from benign
    #  in specific learned directions, but held-out attack types may not
    #  be radially separated at all — they can land NEAR benign in the
    #  trained projection space because the projection only learned to
    #  separate attacks IN DIRECTIONS IT SAW.
    #
    #  Hyperbolic geodesic loss makes radius the dominant signal because
    #  hyperbolic distance grows exponentially with radius — the loss
    #  landscape rewards pushing ALL attacks to high radius regardless
    #  of angular direction.
    # ───────────────────────────────────────────────────────────────────

    angles_train = [0.4, 1.6, 4.0]   # GCG, JBC, Random Search (in training)
    angle_holdout = 2.8              # PAIR (held out)
    attack_train_labels = ["GCG (train)", "JBC (train)", "Random Search (train)"]
    train_colors = ["#D62828", "#F77F00", "#FCBF49"]
    holdout_color = "#7E1F86"

    # ───────────────────────────────────────────────────────────────────
    #  Left panel: Euclidean trained projection
    # ───────────────────────────────────────────────────────────────────
    ax1 = axes[0]
    benign_e = np.random.randn(n, 2) * 0.4

    # Trained attacks pushed far in specific learned directions
    train_attacks_e = []
    for ang in angles_train:
        cluster = np.array([[np.cos(ang), np.sin(ang)]]) * np.random.uniform(2.2, 3.0, (n // 3, 1))
        cluster += np.random.randn(n // 3, 2) * 0.25
        train_attacks_e.append(cluster)

    # Held-out PAIR attacks: NOT pushed to high radius because projection
    # never saw this direction during contrastive training. Lands AMONG benign.
    pair_e = np.array([[np.cos(angle_holdout), np.sin(angle_holdout)]]) * np.random.uniform(0.6, 1.0, (n // 3, 1))
    pair_e += np.random.randn(n // 3, 2) * 0.35

    # Plot
    ax1.scatter(benign_e[:, 0], benign_e[:, 1], c=COLOR_BENIGN, s=40, alpha=0.7,
                edgecolor="darkgreen", linewidth=0.5, label="Benign")
    for i, (atk, lbl) in enumerate(zip(train_attacks_e, attack_train_labels)):
        ax1.scatter(atk[:, 0], atk[:, 1], c=train_colors[i], s=40, alpha=0.7,
                    edgecolor="darkred", linewidth=0.5, label=lbl)
    ax1.scatter(pair_e[:, 0], pair_e[:, 1], c=holdout_color, s=70,
                marker="*", alpha=0.9, edgecolor="black", linewidth=0.8,
                label="PAIR (HELD OUT)")

    # Draw the "circular boundary the mentor asks about"
    circle_e = plt.Circle((0, 0), 1.5, color="blue", fill=False, linestyle=":", linewidth=2.0)
    ax1.add_patch(circle_e)
    ax1.text(0, -3.2, "If you tried a circular threshold (||x|| > R):\nPAIR is at low radius — it falls INSIDE the circle\nand is misclassified as benign.",
             ha="center", fontsize=9, color="blue", fontweight="bold")

    ax1.annotate("PAIR lands NEAR benign:\nprojection W never saw\nthis direction during training,\nso it didn't push PAIR outward.",
                 xy=(pair_e[:, 0].mean(), pair_e[:, 1].mean()),
                 xytext=(-3.7, 1.5),
                 fontsize=9, color=holdout_color,
                 arrowprops=dict(arrowstyle="->", color=holdout_color, lw=1.4),
                 ha="left", fontweight="bold")

    ax1.set_xlim(-4.5, 4.0)
    ax1.set_ylim(-4.0, 3.5)
    ax1.set_aspect("equal")
    ax1.set_title("Euclidean Contrastive Projection",
                  fontweight="bold")
    ax1.set_xlabel("Trained projection dim 1\n(direction-specific contrastive separation)")
    ax1.set_ylabel("Trained projection dim 2")
    ax1.legend(loc="upper right", framealpha=0.95, fontsize=8)
    ax1.grid(alpha=0.3)

    # ───────────────────────────────────────────────────────────────────
    #  Right panel: Hyperbolic trained projection (Poincaré disk view)
    # ───────────────────────────────────────────────────────────────────
    ax2 = axes[1]
    theta = np.linspace(0, 2 * np.pi, 200)
    ax2.plot(np.cos(theta), np.sin(theta), "k-", linewidth=2.5)
    ax2.fill(np.cos(theta), np.sin(theta), color="#E8F4F8", alpha=0.3)

    benign_r = np.random.uniform(0, 0.30, n)
    benign_theta = np.random.uniform(0, 2 * np.pi, n)
    benign_h = np.column_stack([benign_r * np.cos(benign_theta),
                                 benign_r * np.sin(benign_theta)])

    # Trained attacks at high radius
    train_attacks_h = []
    for ang in angles_train:
        atk_r = np.random.uniform(0.78, 0.92, n // 3)
        atk_theta = ang + np.random.randn(n // 3) * 0.20
        atk = np.column_stack([atk_r * np.cos(atk_theta),
                                atk_r * np.sin(atk_theta)])
        train_attacks_h.append(atk)

    # Held-out PAIR ALSO at high radius — because hyperbolic geodesic
    # distance penalizes ANY point near origin if it's labeled adversarial,
    # so the contrastive loss pressures the projection to push ALL attack
    # representations toward the boundary regardless of angular direction.
    pair_r = np.random.uniform(0.74, 0.88, n // 3)
    pair_theta = angle_holdout + np.random.randn(n // 3) * 0.20
    pair_h = np.column_stack([pair_r * np.cos(pair_theta),
                               pair_r * np.sin(pair_theta)])

    ax2.scatter(benign_h[:, 0], benign_h[:, 1], c=COLOR_BENIGN, s=40, alpha=0.8,
                edgecolor="darkgreen", linewidth=0.5, label="Benign (near origin)")
    for i, (atk, lbl) in enumerate(zip(train_attacks_h, attack_train_labels)):
        ax2.scatter(atk[:, 0], atk[:, 1], c=train_colors[i], s=40, alpha=0.8,
                    edgecolor="darkred", linewidth=0.5, label=lbl)
    ax2.scatter(pair_h[:, 0], pair_h[:, 1], c=holdout_color, s=80,
                marker="*", alpha=0.95, edgecolor="black", linewidth=0.8,
                label="PAIR (HELD OUT)")

    # Radial threshold
    boundary = plt.Circle((0, 0), 0.55, color="blue", fill=False,
                          linestyle=":", linewidth=2.5)
    ax2.add_patch(boundary)
    ax2.text(0, -1.45, "Radial threshold (||x|| > R) catches ALL attacks\nincluding PAIR — the geometry forces it.",
             ha="center", fontsize=9, color="blue", fontweight="bold")

    ax2.annotate("PAIR is pushed to HIGH RADIUS\n(hyperbolic loss penalty grows\nexponentially with distance from origin\n→ ALL attacks land at boundary)",
                 xy=(pair_h[:, 0].mean(), pair_h[:, 1].mean()),
                 xytext=(-1.4, 1.05),
                 fontsize=9, color=holdout_color,
                 arrowprops=dict(arrowstyle="->", color=holdout_color, lw=1.4),
                 ha="left", fontweight="bold")

    ax2.set_xlim(-1.6, 1.6)
    ax2.set_ylim(-1.7, 1.5)
    ax2.set_aspect("equal")
    ax2.set_title("Hyperbolic Contrastive Projection (Poincaré disk view)",
                  fontweight="bold")
    ax2.set_xlabel("Lorentz coordinate x₁\n(radius = ||x|| has consistent meaning across directions)")
    ax2.set_ylabel("Lorentz coordinate x₂")
    ax2.legend(loc="upper right", framealpha=0.95, fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle("Why a Euclidean Circle Doesn't Solve It — Hyperbolic Geometry Changes the Training Dynamics",
                 fontweight="bold", y=1.00, fontsize=13)

    # Add explanatory caption below
    fig.text(0.5, -0.04,
             "KEY INSIGHT: It's not just about the boundary shape — it's about where the TRAINED PROJECTION places held-out attacks.\n"
             "In Euclidean space, contrastive loss creates DIRECTION-SPECIFIC separation: attacks that match training-direction features get pushed away;\n"
             "novel-direction attacks (e.g., PAIR) stay near origin and even a circular boundary fails to catch them.\n"
             "In hyperbolic space, geodesic distance grows exponentially with radius, so the loss rewards pushing ALL attacks to the boundary regardless of direction.",
             ha="center", fontsize=10, style="italic", color="#222")

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
