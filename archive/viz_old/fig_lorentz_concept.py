"""
fig_lorentz_concept.py — Slide 1 theoretical motivation.

Generates a 2-panel conceptual figure:
  Left:  Lorentz hyperboloid in 3D — benign vs attack at different radial positions
  Right: Poincaré disk projection — same data, easier to visualize hierarchy

Both panels show the hypothesis: harmful (specific, deeper hierarchy) sits at higher
radial position, benign (general) near origin.

NOT real data — illustrative. Uses synthetic samples drawn to resemble the
empirical pattern (benign median ≈ 3.20, attack median ≈ 3.50 in Lorentz coords).

Usage:
    python fig_lorentz_concept.py
    python fig_lorentz_concept.py --output figures_for_meeting/fig_concept.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def lorentz_to_poincare(x_time, x_spatial, kappa=0.1):
    """Map Lorentz coords (t, s) to Poincaré disk."""
    return x_spatial / (x_time + 1.0 / np.sqrt(kappa))


def make_figure(output_path, seed=42):
    rng = np.random.default_rng(seed)

    fig = plt.figure(figsize=(13, 5.5))

    # ---- LEFT PANEL: 3D Lorentz hyperboloid ----
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")

    # Draw hyperboloid surface t² - x² - y² = 1/κ
    kappa = 1.0
    u = np.linspace(-1.5, 1.5, 30)
    v = np.linspace(0, 2 * np.pi, 30)
    U, V = np.meshgrid(u, v)
    X = np.sinh(U) * np.cos(V)
    Y = np.sinh(U) * np.sin(V)
    T = np.cosh(U)
    ax1.plot_surface(X, Y, T, alpha=0.18, color="#5dade2", edgecolor="#aed6f1", linewidth=0.3)

    # Benign points: small radial position (close to apex)
    n_ben = 50
    r_ben = rng.normal(0.5, 0.2, n_ben).clip(0.05, 1.2)
    theta_ben = rng.uniform(0, 2 * np.pi, n_ben)
    bx = r_ben * np.cos(theta_ben) * 0.8
    by = r_ben * np.sin(theta_ben) * 0.8
    bt = np.sqrt(1 + bx ** 2 + by ** 2)
    ax1.scatter(bx, by, bt, c="#3498db", s=35, alpha=0.85, label="Benign", edgecolor="black", linewidth=0.4)

    # Attack points: larger radial position (further from apex)
    n_atk = 50
    r_atk = rng.normal(1.4, 0.25, n_atk).clip(0.7, 2.2)
    theta_atk = rng.uniform(0, 2 * np.pi, n_atk)
    ax_ = r_atk * np.cos(theta_atk) * 0.8
    ay_ = r_atk * np.sin(theta_atk) * 0.8
    at_ = np.sqrt(1 + ax_ ** 2 + ay_ ** 2)
    ax1.scatter(ax_, ay_, at_, c="#e74c3c", s=35, alpha=0.85, label="Attack", edgecolor="black", linewidth=0.4)

    # Origin / apex marker
    ax1.scatter([0], [0], [1], c="black", s=60, marker="*", label="Apex (origin)", zorder=10)

    ax1.set_title("Lorentz hyperboloid\n(harm features at higher radial position)",
                  fontsize=11, fontweight="bold")
    ax1.set_xlabel("$x_1$", fontsize=10)
    ax1.set_ylabel("$x_2$", fontsize=10)
    ax1.set_zlabel("$x_0$ (time)", fontsize=10)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.view_init(elev=15, azim=45)

    # ---- RIGHT PANEL: Poincaré disk projection ----
    ax2 = fig.add_subplot(1, 2, 2)

    # Disk boundary
    boundary = Circle((0, 0), 1.0, fill=False, color="black", linewidth=2)
    ax2.add_patch(boundary)
    ax2.add_patch(Circle((0, 0), 1.0, alpha=0.05, color="#5dade2"))

    # Project benign / attack to Poincaré disk
    bp_x = lorentz_to_poincare(bt, bx)
    bp_y = lorentz_to_poincare(bt, by)
    ap_x = lorentz_to_poincare(at_, ax_)
    ap_y = lorentz_to_poincare(at_, ay_)

    ax2.scatter(bp_x, bp_y, c="#3498db", s=45, alpha=0.85, label="Benign (general, near origin)",
                edgecolor="black", linewidth=0.5)
    ax2.scatter(ap_x, ap_y, c="#e74c3c", s=45, alpha=0.85, label="Attack (specific, near boundary)",
                edgecolor="black", linewidth=0.5)
    ax2.scatter([0], [0], c="black", s=80, marker="*", label="Origin (apex)", zorder=10)

    # Concentric circles to indicate hierarchy levels
    for r in [0.3, 0.6, 0.9]:
        ax2.add_patch(Circle((0, 0), r, fill=False, color="gray", linestyle=":", linewidth=0.6, alpha=0.5))

    # Annotation arrow showing direction of harm
    ax2.annotate("", xy=(0.85, 0.35), xytext=(0.05, 0.0),
                 arrowprops=dict(arrowstyle="->", color="#7d3c98", lw=2.0))
    ax2.text(0.45, 0.45, "increasing\nradial position\n(harm hypothesis)",
             fontsize=9, color="#7d3c98", ha="center", fontweight="bold")

    ax2.set_xlim(-1.15, 1.15)
    ax2.set_ylim(-1.15, 1.15)
    ax2.set_aspect("equal")
    ax2.set_title("Poincaré disk view\n(hierarchy: general → specific)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("$x$", fontsize=10)
    ax2.set_ylabel("$y$", fontsize=10)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax2.grid(alpha=0.2)

    fig.suptitle("Hypothesis: hyperbolic geometry encodes hierarchical safety structure",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"  saved -> {output_path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="figures_for_meeting/fig_lorentz_concept.png")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print("\n  Generating conceptual figure (synthetic illustrative data)...")
    print("  This is for pedagogical motivation only, NOT real experimental data.\n")

    make_figure(args.output, args.seed)


if __name__ == "__main__":
    main()
