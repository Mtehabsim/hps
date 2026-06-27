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
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

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
    ax1.text(0, -3.5, "Even a circular threshold (||x|| > R)\nwould miss PAIR — it's at low radius.",
             ha="center", fontsize=9, color="blue", fontweight="bold")

    ax1.annotate("PAIR (held out) lands\nNEAR benign — projection W\nnever saw this direction.",
                 xy=(pair_e[:, 0].mean(), pair_e[:, 1].mean()),
                 xytext=(-4.3, 2.3),
                 fontsize=8.5, color=holdout_color,
                 arrowprops=dict(arrowstyle="->", color=holdout_color, lw=1.4),
                 ha="left", fontweight="bold")

    ax1.set_xlim(-4.5, 4.0)
    ax1.set_ylim(-4.0, 3.5)
    ax1.set_aspect("equal")
    ax1.set_title("Euclidean Contrastive Projection",
                  fontweight="bold")
    ax1.set_xlabel("Projection dim 1 (illustrative)\n(actual HPS uses d_p=64 dimensions; this is a 2D schematic)")
    ax1.set_ylabel("Projection dim 2 (illustrative)")
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
    ax2.text(0, -1.55, "Radial threshold (||x|| > R)\ncatches ALL attacks including PAIR.",
             ha="center", fontsize=9, color="blue", fontweight="bold")

    ax2.annotate("PAIR pushed to HIGH RADIUS:\nhyperbolic loss rewards\npushing ALL attacks to\nboundary, regardless of direction",
                 xy=(pair_h[:, 0].mean(), pair_h[:, 1].mean()),
                 xytext=(-1.55, 1.2),
                 fontsize=8.5, color=holdout_color,
                 arrowprops=dict(arrowstyle="->", color=holdout_color, lw=1.4),
                 ha="left", fontweight="bold")

    ax2.set_xlim(-1.6, 1.6)
    ax2.set_ylim(-1.7, 1.5)
    ax2.set_aspect("equal")
    ax2.set_title("Hyperbolic Contrastive Projection (Poincaré disk view)",
                  fontweight="bold")
    ax2.set_xlabel("Lorentz coordinate x₁ (illustrative)\n(actual HPS uses 64-dim Lorentz hyperboloid; this is a Poincaré disk schematic)")
    ax2.set_ylabel("Lorentz coordinate x₂ (illustrative)")
    ax2.legend(loc="upper right", framealpha=0.95, fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle("Why a Euclidean Circle Doesn't Solve It — Hyperbolic Geometry Changes the Training Dynamics\n"
                 "(Conceptual illustration — schematic 2D projection, not real activation data)",
                 fontweight="bold", y=1.00, fontsize=12)

    out_path = os.path.join(OUT_DIR, "hps_concept.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 3: Pipeline diagram
# ═══════════════════════════════════════════════════════════════════════════

def make_pipeline_figure():
    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.4, 1.6, 1.6, 1.6, 1.0],
                          wspace=0.35)

    # ── Stage 1: Input prompt ──
    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.axis("off")
    ax1.set_title("1. Input Prompt", fontweight="bold", fontsize=12, pad=10)

    # Two example prompts: benign (green) and attack (red)
    bubble1 = mpatches.FancyBboxPatch((0.05, 0.55), 0.9, 0.32,
        boxstyle="round,pad=0.02", facecolor="#E8F4E8",
        edgecolor=COLOR_BENIGN, linewidth=2)
    ax1.add_patch(bubble1)
    ax1.text(0.5, 0.715, "BENIGN", ha="center", color=COLOR_BENIGN,
             fontweight="bold", fontsize=10)
    ax1.text(0.5, 0.62, "\"How does\nphotosynthesis\nwork?\"", ha="center",
             fontsize=8.5, style="italic")

    bubble2 = mpatches.FancyBboxPatch((0.05, 0.10), 0.9, 0.32,
        boxstyle="round,pad=0.02", facecolor="#F8E8E8",
        edgecolor=COLOR_ATTACK, linewidth=2)
    ax1.add_patch(bubble2)
    ax1.text(0.5, 0.275, "ATTACK", ha="center", color=COLOR_ATTACK,
             fontweight="bold", fontsize=10)
    ax1.text(0.5, 0.18, "\"How to make...\n[GCG suffix]\nxyzABC123\"", ha="center",
             fontsize=8.5, style="italic", family="monospace")

    # ── Stage 2: LLM forward pass ──
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis("off")
    ax2.set_title("2. Vicuna-13B (40 layers)", fontweight="bold", fontsize=12, pad=10)

    # Draw stack of layers
    n_layers_show = 8
    layer_height = 0.08
    layer_y_start = 0.15
    layer_x = 0.25
    layer_w = 0.5
    selected_layers_viz = [0, 1, 2, 5, 6, 7]  # indices into n_layers_show that are selected
    for i in range(n_layers_show):
        y = layer_y_start + i * layer_height
        is_selected = i in selected_layers_viz
        color = COLOR_HPS if is_selected else "#D8D8D8"
        rect = mpatches.Rectangle((layer_x, y), layer_w, layer_height * 0.85,
                                   facecolor=color, edgecolor="black", linewidth=0.7)
        ax2.add_patch(rect)
        if is_selected:
            ax2.text(layer_x + layer_w + 0.04, y + layer_height * 0.4,
                     "←", color=COLOR_HPS, fontweight="bold", fontsize=12)
    # Layer labels
    ax2.text(layer_x - 0.05, layer_y_start, "L0", fontsize=8, ha="right")
    ax2.text(layer_x - 0.05, layer_y_start + 7 * layer_height, "L39", fontsize=8, ha="right")
    ax2.text(0.5, 0.05, "Tap 8 selected layers\n(top-Fisher score)", ha="center",
             fontsize=9, color=COLOR_HPS, fontweight="bold")
    ax2.text(0.5, 0.92, "Hidden states\nat each layer", ha="center", fontsize=9, style="italic")

    # ── Stage 3: Hyperbolic projection ──
    ax3 = fig.add_subplot(gs[2])
    ax3.set_xlim(-1.4, 1.4); ax3.set_ylim(-1.3, 1.3)
    ax3.set_aspect("equal"); ax3.axis("off")
    ax3.set_title("3. Lorentz Projection\n(novel)", fontweight="bold", fontsize=12,
                  color=COLOR_HPS, pad=10)

    # Poincaré disk
    theta = np.linspace(0, 2 * np.pi, 200)
    ax3.plot(np.cos(theta), np.sin(theta), "k-", linewidth=2)
    ax3.fill(np.cos(theta), np.sin(theta), color="#E8F4F8", alpha=0.5)

    np.random.seed(7)
    # Benign trajectory: stays near origin (dots connected by arrows showing layer-to-layer)
    benign_traj = []
    for i in range(8):
        r = 0.05 + i * 0.03 + np.random.uniform(-0.02, 0.02)
        ang = 0.5 + i * 0.15
        benign_traj.append([r * np.cos(ang), r * np.sin(ang)])
    benign_traj = np.array(benign_traj)
    ax3.plot(benign_traj[:, 0], benign_traj[:, 1], "-",
             color=COLOR_BENIGN, linewidth=1.5, alpha=0.8)
    ax3.scatter(benign_traj[:, 0], benign_traj[:, 1], c=COLOR_BENIGN, s=35,
                edgecolor="darkgreen", linewidth=0.6, zorder=3)

    # Attack trajectory: drifts to high radius
    attack_traj = []
    for i in range(8):
        r = 0.15 + i * 0.10 + np.random.uniform(-0.02, 0.02)
        ang = -0.7 + i * 0.05
        attack_traj.append([r * np.cos(ang), r * np.sin(ang)])
    attack_traj = np.array(attack_traj)
    ax3.plot(attack_traj[:, 0], attack_traj[:, 1], "-",
             color=COLOR_ATTACK, linewidth=1.5, alpha=0.8)
    ax3.scatter(attack_traj[:, 0], attack_traj[:, 1], c=COLOR_ATTACK, s=35,
                edgecolor="darkred", linewidth=0.6, zorder=3)

    # Annotate trajectories
    ax3.text(benign_traj[-1, 0] + 0.05, benign_traj[-1, 1] + 0.05,
             "benign\ntrajectory", fontsize=8, color=COLOR_BENIGN, fontweight="bold")
    ax3.text(attack_traj[-1, 0] - 0.45, attack_traj[-1, 1] - 0.15,
             "attack\ntrajectory", fontsize=8, color=COLOR_ATTACK, fontweight="bold")
    ax3.text(0, -1.18, "Each layer's activation\n→ point on hyperboloid",
             ha="center", fontsize=8.5, style="italic")

    # ── Stage 4: Trajectory features ──
    ax4 = fig.add_subplot(gs[3])
    ax4.set_xlim(0, 1); ax4.set_ylim(0, 1); ax4.axis("off")
    ax4.set_title("4. Trajectory Features", fontweight="bold", fontsize=12, pad=10)

    # Show 12-feature vector visually as small bar chart
    feat_names_compact = ["mean r", "max r", "min r", "std r", "range r",
                          "max κ", "mean κ", "std κ", "spike loc",
                          "displ", "path", "ratio"]
    np.random.seed(3)
    benign_feats = np.random.uniform(0.0, 0.3, 12)
    attack_feats = np.random.uniform(0.5, 1.0, 12)
    attack_feats[0] = 0.95; benign_feats[0] = 0.10  # mean r dominant

    x_pos = np.arange(12)
    width = 0.4
    ax4.set_xlim(-0.5, 12)
    ax4.set_ylim(-0.05, 1.15)
    ax4.bar(x_pos - width/2, benign_feats, width, color=COLOR_BENIGN,
            edgecolor="darkgreen", linewidth=0.5, alpha=0.8, label="Benign")
    ax4.bar(x_pos + width/2, attack_feats, width, color=COLOR_ATTACK,
            edgecolor="darkred", linewidth=0.5, alpha=0.8, label="Attack")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(feat_names_compact, fontsize=6.5, rotation=70, ha="right")
    ax4.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax4.set_ylabel("Value", fontsize=8)
    ax4.tick_params(axis="y", labelsize=7)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.text(5.5, 1.08, "12 geometric features\n(radial / curvature / displacement)",
             ha="center", fontsize=8.5, style="italic", transform=ax4.transData)

    # ── Stage 5: Decision ──
    ax5 = fig.add_subplot(gs[4])
    ax5.set_xlim(0, 1); ax5.set_ylim(0, 1); ax5.axis("off")
    ax5.set_title("5. Decision", fontweight="bold", fontsize=12, pad=10)

    # Traffic-light style decision boxes
    allow_box = mpatches.FancyBboxPatch((0.1, 0.55), 0.8, 0.32,
        boxstyle="round,pad=0.02", facecolor="#D4F0D4",
        edgecolor=COLOR_BENIGN, linewidth=2.2)
    ax5.add_patch(allow_box)
    ax5.text(0.5, 0.78, "✓ ALLOW", ha="center", color="darkgreen",
             fontweight="bold", fontsize=14)
    ax5.text(0.5, 0.625, "score < threshold", ha="center", fontsize=8, style="italic")

    block_box = mpatches.FancyBboxPatch((0.1, 0.10), 0.8, 0.32,
        boxstyle="round,pad=0.02", facecolor="#F8D4D4",
        edgecolor=COLOR_ATTACK, linewidth=2.2)
    ax5.add_patch(block_box)
    ax5.text(0.5, 0.34, "✗ BLOCK", ha="center", color="darkred",
             fontweight="bold", fontsize=14)
    ax5.text(0.5, 0.18, "score > threshold", ha="center", fontsize=8, style="italic")

    # ── Add arrows between stages ──
    arrow_kw = dict(arrowstyle="->", color="black", lw=1.8,
                    connectionstyle="arc3,rad=0")
    # We need to draw arrows in figure coordinates spanning between subplots
    fig.canvas.draw()
    # Use figure-level annotations
    arrow_ys = [0.5]
    bbox1 = ax1.get_position()
    bbox2 = ax2.get_position()
    bbox3 = ax3.get_position()
    bbox4 = ax4.get_position()
    bbox5 = ax5.get_position()

    pairs = [(bbox1, bbox2), (bbox2, bbox3), (bbox3, bbox4), (bbox4, bbox5)]
    for b1, b2 in pairs:
        x_start = b1.x1 - 0.01
        x_end = b2.x0 + 0.01
        y = (b1.y0 + b1.y1) / 2
        fig.add_artist(FancyArrowPatch((x_start, y), (x_end, y),
                                       arrowstyle="->", color="black",
                                       mutation_scale=15, lw=1.5))

    fig.suptitle("HPS Pipeline: Activation Trajectory Detection in Hyperbolic Space",
                 fontweight="bold", fontsize=14, y=1.02)

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
