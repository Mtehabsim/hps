"""
visualize_activation_space.py — Visualize HOW the model separates benign
from attack in activation space.

This produces 8 diagnostic visualizations to answer the question:
"Is the classifier learning semantic harm, or just norm-based shortcut?"

  Plot 1: Per-layer NORM distribution (benign vs attack histograms)
          → If norms are perfectly separable, that's the confound directly
  Plot 2: PCA 2D of raw last-token activations at deep layer
          → If classes form 2 distinct clusters, linearly separable
  Plot 3: PCA 2D AFTER L2-normalization (removes norm shortcut)
          → If still separable: real semantic signal exists
          → If collapses to mixed: classifier was using norm
  Plot 4: t-SNE 2D of raw activations (non-linear projection)
  Plot 5: t-SNE 2D after L2-normalization
  Plot 6: LR coefficient magnitudes (which dimensions matter)
  Plot 7: HPS radial distribution histogram (Lorentz x_0)
  Plot 8: Decision-boundary visualization (PCA + LR boundary)

Usage:
  # On any cache:
  python visualize_activation_space.py \\
      --cache results/llama3_activations_cache_diverse_fixed.npz \\
      --hidden_dim 4096 \\
      --layers 0 2 17 24 28 31 \\
      --output_dir results/figs/viz_llama3

  # Quick run (subsample for speed):
  python visualize_activation_space.py \\
      --cache results/llama3_activations_cache_diverse_fixed.npz \\
      --max_samples 1000

Output:
  results/figs/viz_<name>/01_norm_dist.png
  results/figs/viz_<name>/02_pca_raw.png
  results/figs/viz_<name>/03_pca_normalized.png
  results/figs/viz_<name>/04_tsne_raw.png
  results/figs/viz_<name>/05_tsne_normalized.png
  results/figs/viz_<name>/06_lr_coefficients.png
  results/figs/viz_<name>/07_hps_radial_distribution.png
  results/figs/viz_<name>/08_decision_boundary.png
  results/figs/viz_<name>/SUMMARY.png  (composite of all 8)
"""

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from hps_core import (
    LorentzProjection,
    contrastive_loss,
    extract_trajectory_features,
)


# ---------------------------------------------------------------------------
# Cache loading (handles both formats)
# ---------------------------------------------------------------------------

def to_lasttoken(arr):
    out = []
    for l in sorted(arr.keys()):
        t = arr[l]
        out.append(t[-1] if t.ndim == 2 else t)
    return np.stack(out, axis=0)


def load_cache(cache_path, layers):
    cache = np.load(cache_path, allow_pickle=True)
    keys = list(cache.keys())

    if "X_benign" in keys:
        X_ben = cache["X_benign"]
        X_atk = cache["X_attack"]
        rng = np.random.RandomState(42)
        ben_idx = rng.permutation(len(X_ben))
        atk_idx = rng.permutation(len(X_atk))
        n_ben_train = int(0.8 * len(ben_idx))
        n_atk_train = int(0.8 * len(atk_idx))
        return {
            "X_tr_ben": X_ben[ben_idx[:n_ben_train]],
            "X_tr_atk": X_atk[atk_idx[:n_atk_train]],
            "X_te_ben": X_ben[ben_idx[n_ben_train:]],
            "X_te_atk": X_atk[atk_idx[n_atk_train:]],
        }

    if "hs_train_ben" in keys:
        def to_arr(hs_list):
            arrs = [to_lasttoken(h) for h in hs_list]
            if not arrs:
                return np.empty((0, len(layers), 0), dtype=np.float32)
            return np.stack(arrs, axis=0)

        return {
            "X_tr_ben": to_arr(cache["hs_train_ben"]),
            "X_tr_atk": to_arr(cache["hs_train_atk"]),
            "X_te_ben": to_arr(cache["hs_test_ben"]),
            "X_te_atk": to_arr(cache["hs_test_atk"]),
        }

    raise ValueError(f"Unknown cache format. Keys: {keys}")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

CB = "#3498db"   # benign blue
CA = "#e74c3c"   # attack red


def add_metric_text(ax, text, loc="upper left"):
    """Add a small metric box to a subplot."""
    locs = {
        "upper left": (0.02, 0.98, "left", "top"),
        "upper right": (0.98, 0.98, "right", "top"),
        "lower right": (0.98, 0.02, "right", "bottom"),
        "lower left": (0.02, 0.02, "left", "bottom"),
    }
    x, y, ha, va = locs[loc]
    ax.text(x, y, text,
            transform=ax.transAxes, ha=ha, va=va, fontsize=8,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))


# ---------------------------------------------------------------------------
# Plot 1: Per-layer norm distributions
# ---------------------------------------------------------------------------

def plot_norm_distributions(X_ben, X_atk, layers, output_path):
    n_layers = X_ben.shape[1]
    n_cols = 3
    n_rows = (n_layers + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3))
    axes = axes.flatten() if n_layers > 1 else [axes]

    norms_ben = np.linalg.norm(X_ben, axis=2)  # (N, n_layers)
    norms_atk = np.linalg.norm(X_atk, axis=2)

    from sklearn.metrics import roc_auc_score

    for i, layer in enumerate(layers):
        ax = axes[i]
        b_norms = norms_ben[:, i]
        a_norms = norms_atk[:, i]

        # AUROC for this layer using only norm
        labels = np.concatenate([np.zeros(len(b_norms)), np.ones(len(a_norms))])
        scores = np.concatenate([b_norms, a_norms])
        try:
            auroc = roc_auc_score(labels, scores)
        except Exception:
            auroc = 0.5

        # Histograms
        all_min = min(b_norms.min(), a_norms.min())
        all_max = max(b_norms.max(), a_norms.max())
        bins = np.linspace(all_min, all_max, 50)

        ax.hist(b_norms, bins=bins, alpha=0.6, label=f"Benign (n={len(b_norms)})",
                color=CB, density=True)
        ax.hist(a_norms, bins=bins, alpha=0.6, label=f"Attack (n={len(a_norms)})",
                color=CA, density=True)
        ax.set_title(f"Layer {layer}  |  norm-only AUROC = {auroc:.3f}",
                     fontsize=10)
        ax.set_xlabel("L2 norm")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for i in range(n_layers, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Per-layer activation norm distributions\n"
                 "(if benign and attack are linearly separable in norm, "
                 "that's the confound directly)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved {output_path}")


# ---------------------------------------------------------------------------
# Plot 2 & 3: PCA 2D (raw and L2-normalized)
# ---------------------------------------------------------------------------

def plot_pca_2d(X_ben, X_atk, layer_idx, output_path, normalize=False,
                 title_suffix="raw"):
    Xb = X_ben[:, layer_idx, :].astype(np.float32)
    Xa = X_atk[:, layer_idx, :].astype(np.float32)

    if normalize:
        Xb = Xb / (np.linalg.norm(Xb, axis=1, keepdims=True) + 1e-7)
        Xa = Xa / (np.linalg.norm(Xa, axis=1, keepdims=True) + 1e-7)

    X_all = np.concatenate([Xb, Xa])
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_all)

    Xb_pca = X_pca[:len(Xb)]
    Xa_pca = X_pca[len(Xb):]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(Xb_pca[:, 0], Xb_pca[:, 1], alpha=0.5, s=20,
               color=CB, label=f"Benign (n={len(Xb)})", edgecolor="white",
               linewidth=0.3)
    ax.scatter(Xa_pca[:, 0], Xa_pca[:, 1], alpha=0.5, s=20,
               color=CA, label=f"Attack (n={len(Xa)})", edgecolor="white",
               linewidth=0.3)

    ev_pct = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({ev_pct[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({ev_pct[1]:.1f}% variance)")
    ax.set_title(f"PCA 2D — layer {layer_idx} ({title_suffix})\n"
                 f"Total var explained: {sum(ev_pct):.1f}%")
    ax.legend()
    ax.grid(alpha=0.3)
    add_metric_text(ax, f"PC1 var: {ev_pct[0]:.1f}%\n"
                         f"PC2 var: {ev_pct[1]:.1f}%")
    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved {output_path}")


# ---------------------------------------------------------------------------
# Plot 4 & 5: t-SNE 2D
# ---------------------------------------------------------------------------

def plot_tsne_2d(X_ben, X_atk, layer_idx, output_path, normalize=False,
                  perplexity=30, max_samples=2000, title_suffix="raw"):
    rng = np.random.RandomState(42)
    n_ben = min(len(X_ben), max_samples // 2)
    n_atk = min(len(X_atk), max_samples // 2)
    ben_idx = rng.choice(len(X_ben), n_ben, replace=False)
    atk_idx = rng.choice(len(X_atk), n_atk, replace=False)

    Xb = X_ben[ben_idx, layer_idx, :].astype(np.float32)
    Xa = X_atk[atk_idx, layer_idx, :].astype(np.float32)

    if normalize:
        Xb = Xb / (np.linalg.norm(Xb, axis=1, keepdims=True) + 1e-7)
        Xa = Xa / (np.linalg.norm(Xa, axis=1, keepdims=True) + 1e-7)

    X_all = np.concatenate([Xb, Xa])
    print(f"    Running t-SNE on {len(X_all)} samples ({title_suffix}) "
          f"— may take 30-90 seconds...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                init="random", learning_rate="auto", max_iter=1000)
    X_tsne = tsne.fit_transform(X_all)

    Xb_tsne = X_tsne[:len(Xb)]
    Xa_tsne = X_tsne[len(Xb):]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(Xb_tsne[:, 0], Xb_tsne[:, 1], alpha=0.5, s=20,
               color=CB, label=f"Benign (n={len(Xb)})", edgecolor="white",
               linewidth=0.3)
    ax.scatter(Xa_tsne[:, 0], Xa_tsne[:, 1], alpha=0.5, s=20,
               color=CA, label=f"Attack (n={len(Xa)})", edgecolor="white",
               linewidth=0.3)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_title(f"t-SNE 2D — layer {layer_idx} ({title_suffix})\n"
                 f"perplexity={perplexity}, n={len(X_all)}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved {output_path}")


# ---------------------------------------------------------------------------
# Plot 6: LR coefficient magnitudes
# ---------------------------------------------------------------------------

def plot_lr_coefficients(X_tr_ben, X_tr_atk, layers, output_path):
    """C4-style: mean across layers, then LR. Show coefficient magnitudes."""
    feats_ben = X_tr_ben.mean(axis=1)  # (N, d)
    feats_atk = X_tr_atk.mean(axis=1)
    X = np.concatenate([feats_ben, feats_atk])
    y = np.concatenate([np.zeros(len(feats_ben)), np.ones(len(feats_atk))])

    sc = StandardScaler()
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(sc.fit_transform(X), y)
    coefs = np.abs(clf.coef_[0])

    # Sort and take top-50
    n_show = min(50, len(coefs))
    top_idx = np.argsort(coefs)[-n_show:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution of all coefficients
    axes[0].hist(coefs, bins=80, alpha=0.7, color="#34495e")
    axes[0].set_xlabel("|coefficient|")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"LR coefficient magnitude distribution\n"
                      f"({len(coefs)} features, top-50 highlighted)")
    axes[0].axvline(coefs[top_idx[-1]], color="red", linestyle="--",
                    label=f"top-50 cutoff: {coefs[top_idx[-1]]:.4f}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_yscale("log")

    # Top-50 coefficients
    axes[1].bar(range(n_show), coefs[top_idx], color="#34495e", alpha=0.85)
    axes[1].set_xlabel("Feature rank (top-50 by |coef|)")
    axes[1].set_ylabel("|coefficient|")
    axes[1].set_title(f"Top-{n_show} most discriminative features\n"
                      f"(if a few features dominate, classifier is using "
                      f"shortcuts)")
    axes[1].grid(alpha=0.3)

    # Concentration metric
    total = coefs.sum()
    top10_share = coefs[np.argsort(coefs)[-10:]].sum() / total * 100
    add_metric_text(axes[1],
                     f"Top-10 features:\n"
                     f"  share of total |coefs|: {top10_share:.1f}%\n"
                     f"  (high → shortcut-based)",
                     loc="upper right")

    plt.suptitle("LR coefficient analysis (C4-style mean-pool)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved {output_path}")


# ---------------------------------------------------------------------------
# Plot 7: HPS radial distribution
# ---------------------------------------------------------------------------

def train_hps_for_viz(X_tr_ben, X_tr_atk, hidden_dim, n_layers,
                        epochs=50, kappa=0.1, device="cuda"):
    proj = LorentzProjection(
        hidden_dim=hidden_dim, n_layers=n_layers,
        proj_dim=64, kappa=kappa,
    ).to(device)
    opt = torch.optim.Adam(proj.parameters(), lr=1e-3, weight_decay=1e-5)
    Xb = torch.from_numpy(np.asarray(X_tr_ben, dtype=np.float32)).to(device)
    Xa = torch.from_numpy(np.asarray(X_tr_atk, dtype=np.float32)).to(device)

    proj.train()
    for ep in range(epochs):
        opt.zero_grad()
        loss = contrastive_loss(proj(Xb), proj(Xa), kappa=kappa)
        loss.backward()
        opt.step()
    proj.eval()
    return proj


def plot_hps_radial(X_tr_ben, X_tr_atk, X_te_ben, X_te_atk, hidden_dim,
                     n_layers, output_path, kappa=0.1, epochs=50, device="cuda"):
    proj = train_hps_for_viz(X_tr_ben, X_tr_atk, hidden_dim, n_layers,
                              epochs=epochs, kappa=kappa, device=device)

    def get_radial(X):
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
        with torch.no_grad():
            z = proj(Xt).cpu().numpy()  # (N, n_layers, 65)
        return z[:, :, 0].mean(axis=1)  # mean radial across layers

    r_te_ben = get_radial(X_te_ben)
    r_te_atk = get_radial(X_te_atk)

    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(
        np.concatenate([np.zeros(len(r_te_ben)), np.ones(len(r_te_atk))]),
        np.concatenate([r_te_ben, r_te_atk]),
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    all_min = min(r_te_ben.min(), r_te_atk.min())
    all_max = max(r_te_ben.max(), r_te_atk.max())
    bins = np.linspace(all_min, all_max, 50)
    ax.hist(r_te_ben, bins=bins, alpha=0.6, label=f"Benign (n={len(r_te_ben)})",
            color=CB, density=True)
    ax.hist(r_te_atk, bins=bins, alpha=0.6, label=f"Attack (n={len(r_te_atk)})",
            color=CA, density=True)
    med_b = float(np.median(r_te_ben))
    med_a = float(np.median(r_te_atk))
    ax.axvline(med_b, color=CB, linestyle="--",
                label=f"benign median: {med_b:.3f}")
    ax.axvline(med_a, color=CA, linestyle="--",
                label=f"attack median: {med_a:.3f}")
    direction = "as expected (atk > ben)" if med_a > med_b \
                else "INVERTED (ben > atk)"
    ax.set_xlabel("Mean Lorentz radial position (x_0)")
    ax.set_ylabel("Density")
    ax.set_title(f"HPS Lorentz radial distribution (kappa={kappa})\n"
                 f"radial-only AUROC = {auroc:.4f} | {direction}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved {output_path}")


# ---------------------------------------------------------------------------
# Plot 8: Decision boundary visualization
# ---------------------------------------------------------------------------

def plot_decision_boundary(X_ben, X_atk, layer_idx, output_path,
                            normalize=False):
    Xb = X_ben[:, layer_idx, :].astype(np.float32)
    Xa = X_atk[:, layer_idx, :].astype(np.float32)
    if normalize:
        Xb = Xb / (np.linalg.norm(Xb, axis=1, keepdims=True) + 1e-7)
        Xa = Xa / (np.linalg.norm(Xa, axis=1, keepdims=True) + 1e-7)

    X = np.concatenate([Xb, Xa])
    y = np.concatenate([np.zeros(len(Xb)), np.ones(len(Xa))])

    # PCA project to 2D
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # Train LR on PCA features
    clf = LogisticRegression(random_state=42, max_iter=2000)
    clf.fit(X_pca, y)

    # Compute mesh
    pad = 1.0
    x_min, x_max = X_pca[:, 0].min() - pad, X_pca[:, 0].max() + pad
    y_min, y_max = X_pca[:, 1].min() - pad, X_pca[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                          np.linspace(y_min, y_max, 200))
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9, 7))
    contour = ax.contourf(xx, yy, Z, levels=20, cmap="RdBu_r", alpha=0.5)
    plt.colorbar(contour, ax=ax, label="P(attack)")
    ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2,
                linestyles="--")

    Xb_pca = X_pca[:len(Xb)]
    Xa_pca = X_pca[len(Xb):]
    ax.scatter(Xb_pca[:, 0], Xb_pca[:, 1], alpha=0.6, s=20, color=CB,
                label="Benign", edgecolor="white", linewidth=0.3)
    ax.scatter(Xa_pca[:, 0], Xa_pca[:, 1], alpha=0.6, s=20, color=CA,
                label="Attack", edgecolor="white", linewidth=0.3)

    # Compute LR accuracy on PCA features
    train_acc = clf.score(X_pca, y)
    ax.set_title(f"LR decision boundary on PCA-2D — layer {layer_idx} "
                 f"({'normalized' if normalize else 'raw'})\n"
                 f"In-sample LR accuracy on 2D PCA: {train_acc:.3f}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved {output_path}")


# ---------------------------------------------------------------------------
# Composite summary
# ---------------------------------------------------------------------------

def make_composite(individual_paths, output_path, title="Activation Space Summary"):
    """Combine all 8 plots into one summary image."""
    n_plots = len(individual_paths)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(n_cols * 6, n_rows * 5))

    for i, p in enumerate(individual_paths):
        if not os.path.exists(p):
            continue
        img = plt.imread(p)
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(os.path.basename(p), fontsize=9)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--layers", type=int, nargs="+",
                        default=[0, 2, 17, 24, 28, 31])
    parser.add_argument("--hidden_dim", type=int, default=4096)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_samples", type=int, default=2000,
                        help="Subsample for t-SNE and decision boundary "
                             "(speed)")
    parser.add_argument("--deep_layer_idx", type=int, default=-1,
                        help="Which layer index to use for PCA/t-SNE plots "
                             "(-1 = last/deepest)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--kappa", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip_tsne", action="store_true",
                        help="Skip t-SNE (slowest plot)")
    parser.add_argument("--skip_hps", action="store_true",
                        help="Skip HPS training")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Output dir
    if args.output_dir is None:
        cache_name = os.path.basename(args.cache).replace(".npz", "")
        args.output_dir = os.path.join("results/figs",
                                         f"viz_{cache_name}")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("ACTIVATION SPACE VISUALIZATION")
    print("=" * 70)
    print(f"  Cache:       {args.cache}")
    print(f"  Layers:      {args.layers}")
    print(f"  Hidden dim:  {args.hidden_dim}")
    print(f"  Deep layer:  {args.deep_layer_idx}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Device:      {device}")
    print()

    # Load
    print("Loading cache...")
    data = load_cache(args.cache, args.layers)
    n_train_ben = len(data["X_tr_ben"])
    n_train_atk = len(data["X_tr_atk"])
    n_test_ben = len(data["X_te_ben"])
    n_test_atk = len(data["X_te_atk"])
    print(f"  Train: {n_train_ben} benign + {n_train_atk} attack")
    print(f"  Test:  {n_test_ben} benign + {n_test_atk} attack")

    # Concatenate train+test for visualization (more samples)
    X_ben = np.concatenate([data["X_tr_ben"], data["X_te_ben"]])
    X_atk = np.concatenate([data["X_tr_atk"], data["X_te_atk"]])

    # Subsample for speed
    if len(X_ben) > args.max_samples or len(X_atk) > args.max_samples:
        rng = np.random.RandomState(42)
        n_ben_keep = min(args.max_samples, len(X_ben))
        n_atk_keep = min(args.max_samples, len(X_atk))
        X_ben_s = X_ben[rng.choice(len(X_ben), n_ben_keep, replace=False)]
        X_atk_s = X_atk[rng.choice(len(X_atk), n_atk_keep, replace=False)]
    else:
        X_ben_s = X_ben
        X_atk_s = X_atk
    print(f"  Subsampled for viz: {len(X_ben_s)} benign + "
          f"{len(X_atk_s)} attack")

    deep_idx = args.deep_layer_idx
    if deep_idx < 0:
        deep_idx = len(args.layers) + deep_idx
    deep_layer = args.layers[deep_idx]

    paths = []

    # ---------- Plot 1: Per-layer norm distributions ----------
    print("\n[1/8] Plot 1: per-layer norm distributions")
    p1 = os.path.join(args.output_dir, "01_norm_dist.png")
    plot_norm_distributions(X_ben_s, X_atk_s, args.layers, p1)
    paths.append(p1)

    # ---------- Plot 2: PCA raw ----------
    print(f"\n[2/8] Plot 2: PCA 2D raw (layer {deep_layer})")
    p2 = os.path.join(args.output_dir, "02_pca_raw.png")
    plot_pca_2d(X_ben_s, X_atk_s, deep_idx, p2,
                 normalize=False, title_suffix="raw activations")
    paths.append(p2)

    # ---------- Plot 3: PCA L2-normalized ----------
    print(f"\n[3/8] Plot 3: PCA 2D after L2-normalization "
          f"(layer {deep_layer})")
    p3 = os.path.join(args.output_dir, "03_pca_normalized.png")
    plot_pca_2d(X_ben_s, X_atk_s, deep_idx, p3,
                 normalize=True, title_suffix="L2-normalized")
    paths.append(p3)

    # ---------- Plot 4: t-SNE raw ----------
    if not args.skip_tsne:
        print(f"\n[4/8] Plot 4: t-SNE 2D raw (layer {deep_layer})")
        p4 = os.path.join(args.output_dir, "04_tsne_raw.png")
        plot_tsne_2d(X_ben_s, X_atk_s, deep_idx, p4,
                      normalize=False, max_samples=args.max_samples,
                      title_suffix="raw activations")
        paths.append(p4)

        # ---------- Plot 5: t-SNE normalized ----------
        print(f"\n[5/8] Plot 5: t-SNE 2D L2-normalized "
              f"(layer {deep_layer})")
        p5 = os.path.join(args.output_dir, "05_tsne_normalized.png")
        plot_tsne_2d(X_ben_s, X_atk_s, deep_idx, p5,
                      normalize=True, max_samples=args.max_samples,
                      title_suffix="L2-normalized")
        paths.append(p5)
    else:
        print("\n[4-5/8] Skipping t-SNE (--skip_tsne)")

    # ---------- Plot 6: LR coefficients ----------
    print("\n[6/8] Plot 6: LR coefficient magnitudes")
    p6 = os.path.join(args.output_dir, "06_lr_coefficients.png")
    plot_lr_coefficients(data["X_tr_ben"], data["X_tr_atk"], args.layers, p6)
    paths.append(p6)

    # ---------- Plot 7: HPS radial distribution ----------
    if not args.skip_hps:
        print("\n[7/8] Plot 7: HPS radial distribution")
        p7 = os.path.join(args.output_dir, "07_hps_radial_distribution.png")
        plot_hps_radial(
            data["X_tr_ben"], data["X_tr_atk"],
            data["X_te_ben"], data["X_te_atk"],
            args.hidden_dim, len(args.layers), p7,
            kappa=args.kappa, epochs=args.epochs, device=device,
        )
        paths.append(p7)
    else:
        print("\n[7/8] Skipping HPS (--skip_hps)")

    # ---------- Plot 8: Decision boundary ----------
    print(f"\n[8/8] Plot 8: LR decision boundary (layer {deep_layer})")
    p8 = os.path.join(args.output_dir, "08_decision_boundary.png")
    plot_decision_boundary(X_ben_s, X_atk_s, deep_idx, p8, normalize=False)
    paths.append(p8)

    # ---------- Composite summary ----------
    print("\nComposite summary...")
    summary_path = os.path.join(args.output_dir, "SUMMARY.png")
    make_composite(paths, summary_path,
                    title=f"Activation space visualization "
                          f"({os.path.basename(args.cache)})")

    # ---------- Diagnosis ----------
    print("\n" + "=" * 70)
    print("HOW TO INTERPRET")
    print("=" * 70)
    print()
    print("  Plot 1 (norm distributions): If benign/attack histograms are")
    print("         clearly separated, that's the norm confound. Look at the")
    print("         per-layer AUROC — if any layer shows AUROC > 0.95,")
    print("         classifier can use norm alone.")
    print()
    print("  Plot 2 vs Plot 3 (PCA raw vs L2-normalized): If raw shows")
    print("         clear separation but normalized shows mixing, the")
    print("         classifier is using norm. If both show separation,")
    print("         real semantic signal exists.")
    print()
    print("  Plot 4 vs Plot 5 (t-SNE raw vs normalized): Same logic for")
    print("         non-linear projection.")
    print()
    print("  Plot 6 (LR coefficients): If top-10 features account for a")
    print("         large share of total |coefs| (>50%), classifier is")
    print("         using a few shortcut dimensions.")
    print()
    print("  Plot 7 (HPS radial): The geometric hypothesis says attacks")
    print("         should be at higher radial. If we see ben > atk:")
    print("         INVERSION (geometric hypothesis fails on this data).")
    print()
    print("  Plot 8 (decision boundary): If the LR boundary cleanly")
    print("         separates two clusters in PCA-2D, the data is")
    print("         linearly separable in the first few PCA components.")
    print()
    print(f"  Full summary: {summary_path}")


if __name__ == "__main__":
    main()
