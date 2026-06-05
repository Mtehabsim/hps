"""
generate_methodology_diagram.py — Side-by-side flowchart of 4 methods.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os


def draw_method_flow(ax, x_offset, method_name, color, steps):
    """Draw a vertical pipeline for one method."""
    box_width = 2.2
    box_height = 0.55
    spacing = 0.7
    y_top = 7.5

    # Title
    ax.text(x_offset + box_width / 2, y_top + 0.5, method_name,
            ha="center", va="center", fontsize=12, fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.85,
                      edgecolor="black", linewidth=1.2))

    # Step boxes
    for i, step in enumerate(steps):
        y = y_top - 0.5 - i * spacing - 0.5
        box = FancyBboxPatch(
            (x_offset, y - box_height / 2), box_width, box_height,
            boxstyle="round,pad=0.05", facecolor="white",
            edgecolor=color, linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(x_offset + box_width / 2, y, step,
                ha="center", va="center", fontsize=8.5, color="black")

        # Arrow to next step
        if i < len(steps) - 1:
            arrow = FancyArrowPatch(
                (x_offset + box_width / 2, y - box_height / 2),
                (x_offset + box_width / 2, y - spacing + box_height / 2),
                arrowstyle="-|>", mutation_scale=15,
                color=color, linewidth=1.3,
            )
            ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(figsize=(14, 9))

    # Common input box at top
    ax.text(7.5, 9.0, "Input prompt → LLM forward pass → Hidden states h ∈ ℝ^(T × n_layers × d)",
            ha="center", va="center", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecf0f1",
                      edgecolor="black", linewidth=1.5))

    # Arrows from common input to each method
    for x in [1.4, 4.4, 8.5, 11.6]:
        arrow = FancyArrowPatch((7.5, 8.7), (x + 1.1, 8.1),
                                  arrowstyle="-|>", mutation_scale=15,
                                  color="gray", linewidth=1.2,
                                  connectionstyle="arc3,rad=0.0")
        ax.add_patch(arrow)

    # HPS pipeline
    draw_method_flow(ax, x_offset=1.4, method_name="HPS\n(262K params)",
                     color="#9b59b6",
                     steps=[
                         "Select 6 layers\n(last token)",
                         "Linear proj W ∈ ℝ^(d×64)\nz_spatial = W·h_l",
                         "Lift to Lorentz hyperboloid\nz_0 = √(1/κ + ‖z_spatial‖²)",
                         "Extract 12 trajectory feats:\n5 radial + 4 curvature + 3 disp",
                         "StandardScaler",
                         "Logistic Regression",
                     ])

    # HPS-Euclidean pipeline
    draw_method_flow(ax, x_offset=4.4, method_name="HPS-Euclidean\n(262K params, matched)",
                     color="#f39c12",
                     steps=[
                         "Select 6 layers\n(last token)",
                         "Linear proj W ∈ ℝ^(d×64)\n(no Lorentz lift)",
                         "Per-layer scale + margin\n(parameter-matched)",
                         "Extract 12 trajectory feats:\nflat-space variants",
                         "StandardScaler",
                         "Logistic Regression",
                     ])

    # C4 pipeline
    draw_method_flow(ax, x_offset=8.5, method_name="C4\n(4,097 params)",
                     color="#2ecc71",
                     steps=[
                         "Select 6 layers\n(last token)",
                         "Mean across layers\nf = (1/6) Σ h_l[-1]",
                         "(no projection)",
                         "(no trajectory features)",
                         "StandardScaler",
                         "Logistic Regression",
                     ])

    # MTP pipeline (Anthropic exact)
    draw_method_flow(ax, x_offset=11.6, method_name="MTP\n(4,097 params, Anthropic)",
                     color="#27ae60",
                     steps=[
                         "Select 1 layer\n(L17 best for Llama-3)",
                         "Mean across tokens\nf = (1/T) Σ h_L[t]",
                         "(no projection)",
                         "(no trajectory features)",
                         "L2 reg λ=1e4\n(StandardScaler)",
                         "Logistic Regression",
                     ])

    # Common output at bottom
    ax.text(7.5, 0.7, "Score → benign vs attack classification",
            ha="center", va="center", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecf0f1",
                      edgecolor="black", linewidth=1.5))

    # Arrows from each method to common output
    for x in [1.4, 4.4, 8.5, 11.6]:
        arrow = FancyArrowPatch((x + 1.1, 1.6), (7.5, 1.0),
                                  arrowstyle="-|>", mutation_scale=15,
                                  color="gray", linewidth=1.2)
        ax.add_patch(arrow)

    # Key differences callout box
    ax.text(7.5, -0.3,
            "Key differences:  HPS=Lorentz curvature  |  "
            "HPS-Euc=flat (matched params)  |  C4=cross-layer pool  |  MTP=cross-token pool",
            ha="center", va="center", fontsize=9, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff8dc",
                      edgecolor="gray"))

    ax.set_xlim(0, 15)
    ax.set_ylim(-1, 10)
    ax.axis("off")
    ax.set_title("Methodology: Four Activation-Based Jailbreak Detection Pipelines\n"
                 "(same input, different processing, same output type)",
                 fontsize=13, fontweight="bold", pad=15)

    output_dir = "figures_for_meeting"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "fig_methodology_pipelines.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
