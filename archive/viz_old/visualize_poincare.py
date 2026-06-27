"""
Visualization — Poincaré Disk Projection of HPS Detector Space
═══════════════════════════════════════════════════════════════
Visualizes where the trained HPS LorentzProjection places benign vs attack
samples in hyperbolic space, projected onto the Poincaré disk.

This shows the ACTUAL geometric separation your detector operates on,
not an independent UMAP re-embedding.

Usage:
  python visualize_poincare.py
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.decomposition import PCA

import config
from utils import load_model
from experiment7 import extract_all_layers, LorentzProjection

HPS_LAYERS = [0, 1, 2, 35, 36, 37, 38, 39]


def hyperboloid_to_poincare_2d(lorentz_pts, spatial_2d):
    """Project to Poincaré disk using time coordinate and 2D spatial reduction.
    lorentz_pts: (N, d_proj+1) — full Lorentz points
    spatial_2d: (N, 2) — PCA-reduced spatial coordinates
    Returns (N, 2) Poincaré disk coordinates.
    """
    t = lorentz_pts[:, 0]
    denom = 1.0 + t
    u = spatial_2d[:, 0] / denom
    v = spatial_2d[:, 1] / denom
    return np.column_stack((u, v))


def main():
    print(f"\n{'═'*60}")
    print(f"  Poincaré Disk Visualization of HPS Projection")
    print(f"  Model: {config.MODEL_NAME}")
    print(f"{'═'*60}\n")

    device = config.DEVICE

    # ── Load trained projection ──
    adv_path = os.path.join(config.RESULTS_DIR, "hps_adv_projection.pt")
    std_path = os.path.join(config.RESULTS_DIR, "hps_projection_head.pt")

    if os.path.exists(adv_path):
        ckpt = torch.load(adv_path, map_location=device, weights_only=False)
        tag = "HPS-Adv"
    elif os.path.exists(std_path):
        ckpt = torch.load(std_path, map_location=device, weights_only=False)
        tag = "HPS"
    else:
        print("ERROR: No trained projection found. Run experiment7 or experiment12 first.")
        return

    d_in = ckpt["d_in"]
    d_proj = ckpt["d_proj"]
    n_layers = ckpt["n_layers"]
    proj_h = LorentzProjection(d_in, d_proj, 1.0, n_layers=n_layers).to(device)
    proj_h.load_state_dict(ckpt["state_dict"])
    proj_h.eval()
    print(f"  Loaded {tag} projection (d_in={d_in}, d_proj={d_proj})")

    # ── Load data ──
    cat_path = os.path.join(config.RESULTS_DIR, "validated_attacks_categorized.json")
    with open(cat_path) as f:
        categorized = json.load(f)

    from dataset import BENIGN
    benign_prompts = list(BENIGN)

    # Sample for visualization (keep it manageable)
    rng = np.random.RandomState(42)
    n_ben_viz = min(80, len(benign_prompts))
    ben_viz = [benign_prompts[i] for i in rng.permutation(len(benign_prompts))[:n_ben_viz]]

    # Sample attacks by method
    method_samples = {}
    for method, prompts in categorized.items():
        n_viz = min(40, len(prompts))
        method_samples[method] = [prompts[i] for i in rng.permutation(len(prompts))[:n_viz]]

    # ── Load model and extract activations ──
    model, tokenizer = load_model(config.MODEL_NAME, device, config.DTYPE)

    def extract_hps(prompts, label):
        acts = []
        for i, p in enumerate(prompts):
            d = extract_all_layers(model, tokenizer, p, device, "last")
            vec = np.array([d[l] for l in HPS_LAYERS if l in d])
            acts.append(vec)
            if (i + 1) % 30 == 0:
                print(f"    {label}: {i+1}/{len(prompts)}")
        return np.array(acts)

    print(f"\n  Extracting activations...")
    X_ben = extract_hps(ben_viz, "benign")

    X_by_method = {}
    for method, prompts in method_samples.items():
        X_by_method[method] = extract_hps(prompts, method)

    # ── Project through trained HPS ──
    print(f"\n  Projecting through {tag}...")

    def project_to_lorentz(X):
        """X: (N, n_layers, d_hidden) → (N, n_layers, d_proj+1) Lorentz points."""
        pts = []
        with torch.no_grad():
            for i in range(len(X)):
                layer_pts = []
                for l in range(X.shape[1]):
                    h = torch.tensor(X[i, l], dtype=torch.float32, device=device).unsqueeze(0)
                    lp = proj_h(h).cpu().numpy()[0]
                    layer_pts.append(lp)
                pts.append(layer_pts)
        return np.array(pts)  # (N, n_layers, d_proj+1)

    lorentz_ben = project_to_lorentz(X_ben)
    lorentz_methods = {m: project_to_lorentz(X) for m, X in X_by_method.items()}

    # Use the FINAL selected layer's projection as the representative point
    final_layer_idx = -1  # last of the selected layers (layer 39)

    pts_ben = lorentz_ben[:, final_layer_idx, :]  # (N, d_proj+1)
    pts_methods = {m: lp[:, final_layer_idx, :] for m, lp in lorentz_methods.items()}

    # ── PCA on spatial part → 2D ──
    all_pts = np.vstack([pts_ben] + list(pts_methods.values()))
    all_spatial = all_pts[:, 1:]  # drop time coordinate
    pca = PCA(n_components=2, random_state=42)
    all_spatial_2d = pca.fit_transform(all_spatial)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # ── Poincaré projection ──
    poincare_all = hyperboloid_to_poincare_2d(all_pts, all_spatial_2d)

    # Split back
    idx = 0
    poincare_ben = poincare_all[idx:idx+len(pts_ben)]; idx += len(pts_ben)
    poincare_methods = {}
    for m in pts_methods:
        n = len(pts_methods[m])
        poincare_methods[m] = poincare_all[idx:idx+n]; idx += n

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(10, 10))

    # Poincaré disk boundary
    circle = patches.Circle((0, 0), radius=1.0, edgecolor='black',
                            facecolor='#f8f8f8', lw=1.5, zorder=0)
    ax.add_patch(circle)

    # Colors for attack methods
    method_colors = plt.cm.Set1(np.linspace(0, 1, len(poincare_methods)))

    # Plot benign
    ax.scatter(poincare_ben[:, 0], poincare_ben[:, 1],
               c='#2ecc71', label='Benign', alpha=0.7, s=40,
               edgecolors='w', linewidth=0.5, zorder=2)

    # Plot each attack method
    for i, (method, pts) in enumerate(poincare_methods.items()):
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=[method_colors[i]], label=method, alpha=0.7, s=40,
                   edgecolors='w', linewidth=0.5, zorder=2)

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"Poincaré Disk — {tag} Projection (Layer 39)\n"
                 f"Trained detector's view of benign vs attack geometry",
                 fontsize=13, pad=15)
    ax.legend(loc='upper left', bbox_to_anchor=(-0.15, 1.05), fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(config.PLOTS_DIR, "poincare_disk_hps.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved → {out_path}")

    # ── Also plot trajectory view (all layers for a few samples) ──
    fig, ax = plt.subplots(figsize=(10, 10))
    circle = patches.Circle((0, 0), radius=1.0, edgecolor='black',
                            facecolor='#f8f8f8', lw=1.5, zorder=0)
    ax.add_patch(circle)

    # Project ALL layers for a few samples to show trajectories
    n_traj = 8
    all_traj_pts = np.vstack([lorentz_ben[:n_traj].reshape(-1, d_proj+1)] +
                             [lp[:n_traj].reshape(-1, d_proj+1) for lp in lorentz_methods.values()])
    all_traj_spatial = all_traj_pts[:, 1:]
    pca_traj = PCA(n_components=2, random_state=42)
    traj_2d = pca_traj.fit_transform(all_traj_spatial)
    traj_poincare = hyperboloid_to_poincare_2d(all_traj_pts, traj_2d)

    # Draw benign trajectories
    n_layers_sel = lorentz_ben.shape[1]
    for i in range(n_traj):
        start = i * n_layers_sel
        end = start + n_layers_sel
        traj = traj_poincare[start:end]
        ax.plot(traj[:, 0], traj[:, 1], 'g-', alpha=0.4, linewidth=1)
        ax.scatter(traj[-1, 0], traj[-1, 1], c='#2ecc71', s=50,
                   edgecolors='w', linewidth=0.5, zorder=3)

    # Draw attack trajectories
    offset = n_traj * n_layers_sel
    colors_iter = iter(method_colors)
    for method, lp in lorentz_methods.items():
        c = next(colors_iter)
        for i in range(min(n_traj, len(lp))):
            start = offset + i * n_layers_sel
            end = start + n_layers_sel
            if end > len(traj_poincare):
                break
            traj = traj_poincare[start:end]
            ax.plot(traj[:, 0], traj[:, 1], color=c, alpha=0.4, linewidth=1)
            ax.scatter(traj[-1, 0], traj[-1, 1], c=[c], s=50,
                       edgecolors='w', linewidth=0.5, zorder=3)
        offset += min(n_traj, len(lp)) * n_layers_sel

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"Poincaré Disk — {tag} Layer Trajectories\n"
                 f"Lines show activation path through layers (→ = deeper)",
                 fontsize=13, pad=15)

    # Manual legend
    ax.plot([], [], 'g-', label='Benign trajectory')
    ax.plot([], [], 'r-', label='Attack trajectory')
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    out_path2 = os.path.join(config.PLOTS_DIR, "poincare_disk_trajectories.png")
    plt.savefig(out_path2, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path2}")

    print(f"\n{'═'*60}")
    print(f"  VISUALIZATION COMPLETE")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
