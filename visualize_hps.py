"""
HPS Visualization — How the hyperbolic projection works.
Loads saved projection + cached activations. No model needed.

Generates:
  1. Poincaré disk: where benign vs attacks land in hyperbolic space
  2. Radial distribution: histogram of x0 (time coordinate) per class
  3. Layer-by-layer trajectory: how activations move through layers
  4. Feature importance: which of the 12 features matter most
  5. 3-panel comparison: HPS vs RTV vs Ensemble feature spaces

Usage:
  python visualize_hps.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from experiment7 import LorentzProjection, extract_trajectory_features
from rtv_standalone import compute_fingerprint, TOKEN_POSITIONS

HPS_LAYERS = [0, 1, 2, 28, 29, 30, 31]
RTV_LAYERS = [17, 24, 31]


def main():
    print("Loading saved data...")

    # Load projection
    proj_path = "results/hps_llama3_projection.pt"
    if not os.path.exists(proj_path):
        print(f"ERROR: {proj_path} not found. Run hps_llama3.py first.")
        return
    ckpt = torch.load(proj_path, map_location="cpu", weights_only=False)
    proj = LorentzProjection(ckpt["d_in"], ckpt["d_proj"], 1.0, n_layers=ckpt["n_layers"])
    proj.load_state_dict(ckpt["state_dict"])
    proj.eval()
    print(f"  Projection loaded (d_in={ckpt['d_in']}, d_proj={ckpt['d_proj']})")

    # Load cached activations
    cache_path = "results/llama3_activations_cache.npz"
    if not os.path.exists(cache_path):
        print(f"ERROR: {cache_path} not found. Run hps_llama3.py first.")
        return
    cache = np.load(cache_path, allow_pickle=True)
    hs_test_ben = cache["hs_test_ben"].tolist()
    hs_test_atk = cache["hs_test_atk"].tolist()
    print(f"  Activations loaded: {len(hs_test_ben)} benign, {len(hs_test_atk)} attacks")

    # Load attack methods from JSON
    with open("llama3_attacks.json") as f:
        categorized = json.load(f)
    # Reconstruct test methods (same split as hps_llama3.py)
    attack_methods = []
    for method, prompts in categorized.items():
        for p in prompts:
            if p:
                attack_methods.append(method)
    rng = np.random.RandomState(42)
    atk_idx = rng.permutation(len(attack_methods))
    n_atk_tr = int(0.8 * len(atk_idx))
    test_methods = [attack_methods[i] for i in atk_idx[n_atk_tr:]]

    # Convert to HPS arrays
    def to_hps(hs_list):
        return np.array([[hs[l][-1] for l in HPS_LAYERS] for hs in hs_list])

    X_te_ben = to_hps(hs_test_ben)
    X_te_atk = to_hps(hs_test_atk)
    n_ben = len(X_te_ben)
    n_atk = len(X_te_atk)

    # Project through Lorentz
    with torch.no_grad():
        # Project each layer for each sample
        def project_all(X):
            """X: (N, n_layers, d_hidden) → lorentz points per layer"""
            N, L, D = X.shape
            all_pts = []
            for i in range(N):
                layer_pts = []
                for l in range(L):
                    h = torch.tensor(X[i, l], dtype=torch.float32).unsqueeze(0)
                    pt = proj(h).squeeze(0).numpy()
                    layer_pts.append(pt)
                all_pts.append(layer_pts)
            return np.array(all_pts)  # (N, n_layers, d_proj+1)

        pts_ben = project_all(X_te_ben)
        pts_atk = project_all(X_te_atk)

    # Extract trajectory features
    feats_ben = extract_trajectory_features(proj, X_te_ben)
    feats_atk = extract_trajectory_features(proj, X_te_atk)

    print("  Generating plots...")

    # ═══════════════════════════════════════════════════════════════════
    #  PLOT 1: Poincaré Disk (final layer projection)
    # ═══════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(9, 9))
    circle = patches.Circle((0, 0), 1.0, edgecolor='black', facecolor='#f9f9f9', lw=1.5)
    ax.add_patch(circle)

    # Use final layer, PCA spatial to 2D, then Poincaré map
    final_ben = pts_ben[:, -1, :]  # (N, d_proj+1)
    final_atk = pts_atk[:, -1, :]
    all_final = np.vstack([final_ben, final_atk])
    spatial = all_final[:, 1:]  # drop time coordinate
    pca = PCA(n_components=2, random_state=42)
    spatial_2d = pca.fit_transform(spatial)
    t_coord = all_final[:, 0]
    # Poincaré: u = spatial / (1 + t)
    poincare = spatial_2d / (1 + t_coord[:, None])

    ax.scatter(poincare[:n_ben, 0], poincare[:n_ben, 1],
               c='#2ecc71', label='Benign', alpha=0.7, s=40, edgecolors='w', linewidth=0.3)
    ax.scatter(poincare[n_ben:, 0], poincare[n_ben:, 1],
               c='#9b59b6', label='Attacks', alpha=0.4, s=20, edgecolors='w', linewidth=0.3)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title("Poincaré Disk — HPS Projection (Final Layer)\nBenign near center, attacks toward boundary", fontsize=12)
    ax.legend(fontsize=11, loc='upper left')
    plt.tight_layout()
    plt.savefig("results/viz_poincare_disk.png", dpi=150)
    plt.close()

    # ═══════════════════════════════════════════════════════════════════
    #  PLOT 2: Radial Distribution (x0 histogram)
    # ═══════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 5))
    # Mean x0 across layers for each sample
    radii_ben = pts_ben[:, :, 0].mean(axis=1)
    radii_atk = pts_atk[:, :, 0].mean(axis=1)
    ax.hist(radii_ben, bins=40, alpha=0.6, color='#2ecc71', label='Benign', density=True)
    ax.hist(radii_atk, bins=40, alpha=0.6, color='#9b59b6', label='Attacks', density=True)
    ax.axvline(np.median(radii_ben), color='#2ecc71', linestyle='--', lw=2, label=f'Benign median={np.median(radii_ben):.2f}')
    ax.axvline(np.median(radii_atk), color='#9b59b6', linestyle='--', lw=2, label=f'Attack median={np.median(radii_atk):.2f}')
    ax.set_xlabel("Radial Position (x₀, time coordinate)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Radial Distribution in Hyperbolic Space\n(Higher x₀ = further from origin = more 'extreme')", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("results/viz_radial_distribution.png", dpi=150)
    plt.close()

    # ═══════════════════════════════════════════════════════════════════
    #  PLOT 3: Layer-by-Layer Trajectory (radial position across layers)
    # ═══════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot mean trajectory for benign and attacks
    mean_traj_ben = pts_ben[:, :, 0].mean(axis=0)  # (n_layers,)
    mean_traj_atk = pts_atk[:, :, 0].mean(axis=0)
    std_traj_ben = pts_ben[:, :, 0].std(axis=0)
    std_traj_atk = pts_atk[:, :, 0].std(axis=0)

    layers_x = range(len(HPS_LAYERS))
    ax.plot(layers_x, mean_traj_ben, 'g-o', label='Benign (mean)', linewidth=2, markersize=8)
    ax.fill_between(layers_x, mean_traj_ben - std_traj_ben, mean_traj_ben + std_traj_ben,
                    alpha=0.2, color='green')
    ax.plot(layers_x, mean_traj_atk, 'purple', marker='^', label='Attacks (mean)', linewidth=2, markersize=8)
    ax.fill_between(layers_x, mean_traj_atk - std_traj_atk, mean_traj_atk + std_traj_atk,
                    alpha=0.2, color='purple')
    ax.set_xticks(layers_x)
    ax.set_xticklabels([str(l) for l in HPS_LAYERS])
    ax.set_xlabel("Layer Index", fontsize=11)
    ax.set_ylabel("Radial Position (x₀)", fontsize=11)
    ax.set_title("Activation Trajectory Through Layers\n(How radial position evolves from early to late layers)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/viz_trajectory.png", dpi=150)
    plt.close()

    # ═══════════════════════════════════════════════════════════════════
    #  PLOT 4: Feature Importance (permutation)
    # ═══════════════════════════════════════════════════════════════════
    X_all = np.vstack([feats_ben, feats_atk])
    y_all = np.array([0]*n_ben + [1]*n_atk)
    sc = StandardScaler()
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(sc.fit_transform(X_all), y_all)

    # Coefficient magnitude as importance proxy
    importance = np.abs(clf.coef_[0])
    feature_names = ['mean_r', 'max_r', 'min_r', 'std_r', 'range_r',
                     'max_curv', 'mean_curv', 'std_curv', 'spike_loc',
                     'displacement', 'path_len', 'progress']

    fig, ax = plt.subplots(figsize=(10, 5))
    idx_sorted = np.argsort(importance)[::-1]
    ax.barh(range(12), importance[idx_sorted], color='#3498db')
    ax.set_yticks(range(12))
    ax.set_yticklabels([feature_names[i] for i in idx_sorted])
    ax.set_xlabel("Absolute Coefficient (Importance)", fontsize=11)
    ax.set_title("HPS Feature Importance\n(Which trajectory features drive detection)", fontsize=12)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("results/viz_feature_importance.png", dpi=150)
    plt.close()

    # ═══════════════════════════════════════════════════════════════════
    #  PLOT 5: Per-method radial separation
    # ═══════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(12, 5))
    methods_unique = sorted(set(test_methods))
    positions = []
    labels = []
    data = [radii_ben]
    labels.append('Benign')
    for m in methods_unique:
        idx = [i for i, x in enumerate(test_methods) if x == m]
        if len(idx) > 0:
            data.append(pts_atk[idx, :, 0].mean(axis=1))
            labels.append(m)

    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors_box = ['#2ecc71'] + ['#9b59b6'] * len(methods_unique)
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Mean Radial Position (x₀)", fontsize=11)
    ax.set_title("Radial Position by Attack Type\n(Higher = easier to detect)", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig("results/viz_per_method_radial.png", dpi=150)
    plt.close()

    print("\n  Plots saved:")
    print("    results/viz_poincare_disk.png")
    print("    results/viz_radial_distribution.png")
    print("    results/viz_trajectory.png")
    print("    results/viz_feature_importance.png")
    print("    results/viz_per_method_radial.png")
    print("\n  Done!")


if __name__ == "__main__":
    main()
