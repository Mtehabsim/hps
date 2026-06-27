"""
HPS Real-Data Visualization
═══════════════════════════
Generates the ACTUAL projection scatter plots from real Vicuna-13B activations.
Run this on the DGX after experiment6/8 has trained the HPS projection.

Computes:
  1. PCA of Euclidean projections (256-dim → 2D)
  2. PCA of Hyperbolic projections (256-dim → 2D)

Both shown side-by-side so the mentor can see the actual data structure,
not just a conceptual illustration.

Usage:
  python visualize_real.py
Output:
  figures/hps_real_projections.png
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import config
from utils import load_model, save_json
from experiment7 import extract_all_layers, LorentzProjection, contrastive_loss
from dataset import BENIGN

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "figure.titlesize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOR_BENIGN = "#5BA659"
ATTACK_COLORS = {
    "GCG":                       "#D62828",
    "JBC":                       "#F77F00",
    "PAIR":                      "#7E1F86",
    "prompt_with_random_search": "#FCBF49",
}

HPS_LAYERS = [0, 1, 2, 35, 36, 37, 38, 39]


def main():
    import json
    cat_path = os.path.join(config.RESULTS_DIR, "validated_attacks_categorized.json")
    with open(cat_path) as f:
        categorized = json.load(f)

    attack_prompts = []
    attack_methods = []
    for method, prompts in categorized.items():
        for p in prompts:
            attack_prompts.append(p)
            attack_methods.append(method)

    benign_prompts = list(BENIGN)

    # Load model
    model, tokenizer = load_model(config.MODEL_NAME, config.DEVICE, config.DTYPE)
    device = config.DEVICE

    # Extract activations
    print(f"Extracting activations for {len(benign_prompts)} benign + {len(attack_prompts)} attacks...")
    all_prompts = benign_prompts + attack_prompts
    labels = np.array([0] * len(benign_prompts) + [1] * len(attack_prompts))
    methods_arr = np.array(["benign"] * len(benign_prompts) + attack_methods)

    all_acts = []
    for i, p in enumerate(all_prompts):
        d = extract_all_layers(model, tokenizer, p, device, "last")
        all_acts.append(d)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(all_prompts)}")

    d_hidden = all_acts[0][HPS_LAYERS[0]].shape[0]
    n_layers = len(HPS_LAYERS)
    X = np.zeros((len(all_prompts), n_layers, d_hidden))
    for i, ad in enumerate(all_acts):
        for j, l in enumerate(HPS_LAYERS):
            if l in ad:
                X[i, j] = ad[l]

    # Train HPS projection (same as exp8)
    print("\nTraining HPS-Full (Lorentz contrastive)...")
    torch.manual_seed(42); np.random.seed(42)
    proj_h = LorentzProjection(d_hidden, config.PROJECTION_DIM, 1.0, n_layers=n_layers).to(device)
    opt = optim.Adam(proj_h.parameters(), lr=1e-3, weight_decay=1e-5)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(labels, dtype=torch.long, device=device)
    for _ in range(120):
        loss = torch.tensor(0.0, device=device)
        for l in range(n_layers):
            h = proj_h(X_t[:, l, :])
            loss = loss + contrastive_loss(h, y_t, k=proj_h.k, tau=proj_h.tau(l))
        loss = loss / n_layers
        opt.zero_grad(); loss.backward(); opt.step()
    proj_h.eval()

    # Train Euclidean projection (matching architecture)
    print("Training Euclidean projection (matched architecture)...")
    torch.manual_seed(42); np.random.seed(42)
    proj_e = nn.Linear(d_hidden, config.PROJECTION_DIM, bias=False).to(device)
    nn.init.xavier_uniform_(proj_e.weight)
    scale_e = nn.Parameter(torch.tensor(1.0 / np.sqrt(config.PROJECTION_DIM), dtype=torch.float32, device=device))
    opt_e = optim.Adam(list(proj_e.parameters()) + [scale_e], lr=1e-3, weight_decay=1e-5)
    for _ in range(120):
        loss = torch.tensor(0.0, device=device)
        for l in range(n_layers):
            h = (proj_e(X_t[:, l, :]) * scale_e).float()
            dists = torch.cdist(h, h)
            sm = (y_t.unsqueeze(0) == y_t.unsqueeze(1)).float()
            dm = 1.0 - sm
            tr = torch.triu(torch.ones(h.shape[0], h.shape[0], device=device), diagonal=1)
            ns = (sm * tr).sum().clamp(min=1)
            nd = (dm * tr).sum().clamp(min=1)
            same_loss = (dists ** 2 * sm * tr).sum() / ns
            diff_loss = (torch.clamp(2.0 - dists, min=0) ** 2 * dm * tr).sum() / nd
            loss = loss + (same_loss + diff_loss) / 2.0
        loss = loss / n_layers
        opt_e.zero_grad(); loss.backward(); opt_e.step()
    proj_e.eval()

    # Project all prompts at the LAST selected layer (most discriminative)
    last_layer_idx = -1
    print(f"\nProjecting all prompts at layer {HPS_LAYERS[last_layer_idx]}...")

    with torch.no_grad():
        # Hyperbolic: full Lorentz point (1+d_p), use spatial part
        h_lorentz = proj_h(X_t[:, last_layer_idx, :])  # (N, d_p+1)
        h_lorentz_np = h_lorentz.cpu().numpy()

        # Euclidean projection
        h_euclidean = (proj_e(X_t[:, last_layer_idx, :]) * scale_e).cpu().numpy()

    # Compute PCA to 2D for visualization
    print("Computing PCA to 2D...")
    pca_h = PCA(n_components=2).fit(h_lorentz_np)
    pca_e = PCA(n_components=2).fit(h_euclidean)

    h_2d = pca_h.transform(h_lorentz_np)
    e_2d = pca_e.transform(h_euclidean)

    print(f"  Hyperbolic PCA explained variance: {pca_h.explained_variance_ratio_}")
    print(f"  Euclidean PCA explained variance:  {pca_e.explained_variance_ratio_}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Euclidean
    ax1 = axes[0]
    benign_mask = methods_arr == "benign"
    ax1.scatter(e_2d[benign_mask, 0], e_2d[benign_mask, 1], c=COLOR_BENIGN, s=20,
                alpha=0.5, edgecolor="darkgreen", linewidth=0.3, label="Benign", zorder=2)
    for method, color in ATTACK_COLORS.items():
        mask = methods_arr == method
        ax1.scatter(e_2d[mask, 0], e_2d[mask, 1], c=color, s=25, alpha=0.7,
                    edgecolor="black", linewidth=0.3, label=method, zorder=3)
    ax1.set_xlabel(f"PC 1 ({pca_e.explained_variance_ratio_[0]*100:.1f}% var)")
    ax1.set_ylabel(f"PC 2 ({pca_e.explained_variance_ratio_[1]*100:.1f}% var)")
    ax1.set_title("Euclidean Projection (PCA of 256-dim → 2D)\nReal Vicuna-13B activations",
                  fontweight="bold")
    ax1.legend(loc="best", framealpha=0.9, fontsize=9)
    ax1.grid(alpha=0.3)

    # Right: Hyperbolic
    ax2 = axes[1]
    ax2.scatter(h_2d[benign_mask, 0], h_2d[benign_mask, 1], c=COLOR_BENIGN, s=20,
                alpha=0.5, edgecolor="darkgreen", linewidth=0.3, label="Benign", zorder=2)
    for method, color in ATTACK_COLORS.items():
        mask = methods_arr == method
        ax2.scatter(h_2d[mask, 0], h_2d[mask, 1], c=color, s=25, alpha=0.7,
                    edgecolor="black", linewidth=0.3, label=method, zorder=3)
    ax2.set_xlabel(f"PC 1 ({pca_h.explained_variance_ratio_[0]*100:.1f}% var)")
    ax2.set_ylabel(f"PC 2 ({pca_h.explained_variance_ratio_[1]*100:.1f}% var)")
    ax2.set_title("Hyperbolic Projection (PCA of Lorentz point → 2D)\nReal Vicuna-13B activations",
                  fontweight="bold")
    ax2.legend(loc="best", framealpha=0.9, fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle(f"Real-Data Projection: Vicuna-13B Activations at Layer {HPS_LAYERS[last_layer_idx]}",
                 fontweight="bold", y=1.02)

    out_path = os.path.join(OUT_DIR, "hps_real_projections.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
