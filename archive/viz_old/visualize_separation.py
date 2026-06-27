#!/usr/bin/env python3
"""
Visualize harmful-vs-benign separation in activation space (PCA + t-SNE).

NOTE: PCA/t-SNE show *class separability*, NOT curvature. They cannot confirm
hyperbolicity (t-SNE distorts distances). Use the controlled delta-hyperbolicity
(analyze_curvature.py) for the geometry question. This script answers a different
question: are harmful/benign linearly separable? (If yes -> explains why a linear
probe like C4 suffices and a hyperbolic prior adds nothing.)

Usage:
  python visualize_separation.py --cache results/llama3_activations_cache_alllayers.npz \
      --layer 7 --n 600 --output results/viz_sep_layer7
"""
import argparse, os
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.environ.get("MPLCONFIGDIR", "/tmp/mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from analyze_curvature import load_cache, extract_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/llama3_activations_cache_alllayers.npz")
    ap.add_argument("--layer", type=int, default=7, help="cache layer index (0..31)")
    ap.add_argument("--n", type=int, default=600, help="points per class")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--output", default="results/viz_sep")
    args = ap.parse_args()

    d = load_cache(args.cache)
    ben = extract_layer(d, args.layer, "ben").astype(np.float64)
    atk = extract_layer(d, args.layer, "atk").astype(np.float64)
    rng = np.random.default_rng(0)
    if len(ben) > args.n: ben = ben[rng.choice(len(ben), args.n, replace=False)]
    if len(atk) > args.n: atk = atk[rng.choice(len(atk), args.n, replace=False)]

    X = np.concatenate([ben, atk], 0)
    y = np.concatenate([np.zeros(len(ben)), np.ones(len(atk))])
    mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True); sd[sd < 1e-6] = 1.0
    Xs = (X - mu) / sd
    radial = np.sqrt(1.0 / 0.1 + (Xs ** 2).sum(1))  # WS5 radial coordinate

    pca2 = PCA(n_components=2).fit_transform(Xs)
    # linear separability proxy: AUROC of best single PCA axis already ~ C4; report PC1 spread
    print(f"[sep] layer {args.layer}: benign={len(ben)} harmful={len(atk)} | "
          f"radial mean ben={radial[y==0].mean():.1f} atk={radial[y==1].mean():.1f}", flush=True)
    tsne2 = TSNE(n_components=2, perplexity=args.perplexity, init="pca",
                 random_state=0).fit_transform(Xs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, emb, name in [(axes[0], pca2, "PCA (linear)"), (axes[1], tsne2, "t-SNE (local, not distance-faithful)")]:
        ax.scatter(emb[y == 0, 0], emb[y == 0, 1], s=12, c="tab:green", alpha=0.5, label="benign")
        ax.scatter(emb[y == 1, 0], emb[y == 1, 1], s=12, c="tab:red", alpha=0.5, label="harmful")
        ax.set_title(name); ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2"); ax.legend()
    fig.suptitle(f"Harmful vs benign separation @ cache layer {args.layer} "
                 f"(separation => linear probe suffices; not a curvature claim)")
    fig.tight_layout(); fig.savefig(args.output + ".png", dpi=140)

    # bonus: PCA colored by radial coordinate (ties WS5 to the geometry)
    fig2, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(pca2[:, 0], pca2[:, 1], s=14, c=radial, cmap="viridis")
    plt.colorbar(sc, label="radial x0 (WS5)")
    ax.set_title(f"PCA colored by radial coordinate @ layer {args.layer}")
    fig2.tight_layout(); fig2.savefig(args.output + "_radial.png", dpi=140)
    print(f"[sep] wrote {args.output}.png and {args.output}_radial.png", flush=True)


if __name__ == "__main__":
    main()
