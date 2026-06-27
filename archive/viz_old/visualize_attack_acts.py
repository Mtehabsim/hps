#!/usr/bin/env python3
"""
Unified attack visualization: benign + harmful(clean) + adaptive(obfuscated).

Shows, in ONE consistent 2-D projection (PCA and t-SNE), all three groups:
  - benign            (clean, from the activation cache)            [green]
  - harmful / "attacks" (clean harmful prompt, from the attack dump) [blue]
  - adaptive          (obfuscated harmful, from the attack dump)     [red X]
with arrows harmful->adaptive so you can see whether the soft-prompt attack drags
harmful activations into the benign region.

A faint overlay of cache-harmful [purple] is included as a PIPELINE-ALIGNMENT CHECK:
if cache-harmful overlaps the dump's clean-harmful, the cache (benign) and the live
attack pipeline are comparable and the benign overlay is trustworthy.

Caveat: t-SNE is for qualitative viewing only (not distance-faithful).

Usage:
  python visualize_attack_acts.py --acts runs/acts_HPSMetric_<PID>.npz \
      --cache results/llama3_activations_cache_alllayers.npz --output results/viz_attack_HPS
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
    ap.add_argument("--acts", required=True)
    ap.add_argument("--cache", default="results/llama3_activations_cache_alllayers.npz")
    ap.add_argument("--layer_pos", type=int, default=None,
                    help="which dumped layer position (0..L-1); default=max-movement layer")
    ap.add_argument("--n_benign", type=int, default=400)
    ap.add_argument("--perplexity", type=float, default=20.0)
    ap.add_argument("--output", default="results/viz_attack")
    args = ap.parse_args()

    z = np.load(args.acts, allow_pickle=True)
    layers = list(z["layers"]); attacked = z["attacked"].astype(np.float64)
    clean = z["clean"].astype(np.float64); score = z["score"]
    N, L, H = attacked.shape
    move = np.linalg.norm(attacked - clean, axis=2).mean(0)
    pos = args.layer_pos if args.layer_pos is not None else int(np.argmax(move))
    live_layer = int(layers[pos]); cache_layer = max(live_layer - 1, 0)
    print(f"[viz] {args.acts}: N={N}, layers={layers}, mean jailbreak={score.mean():.3f}", flush=True)
    print(f"[viz] layer pos {pos} (live {live_layer} / cache {cache_layer}); "
          f"per-layer clean->attacked L2={np.round(move,2)}", flush=True)

    cl = clean[:, pos, :]; at = attacked[:, pos, :]

    # cache groups at the matched transformer layer
    d = load_cache(args.cache)
    rng = np.random.default_rng(0)
    ben_c = extract_layer(d, cache_layer, "ben").astype(np.float64)
    har_c = extract_layer(d, cache_layer, "atk").astype(np.float64)
    if len(ben_c) > args.n_benign: ben_c = ben_c[rng.choice(len(ben_c), args.n_benign, replace=False)]
    if len(har_c) > args.n_benign: har_c = har_c[rng.choice(len(har_c), args.n_benign, replace=False)]

    allpts = np.concatenate([ben_c, har_c, cl, at], 0)
    mu = allpts.mean(0, keepdims=True); sd = allpts.std(0, keepdims=True); sd[sd < 1e-6] = 1.0
    Xs = (allpts - mu) / sd
    nb, nh = len(ben_c), len(har_c)
    sl = {"ben": slice(0, nb), "har": slice(nb, nb + nh),
          "cl": slice(nb + nh, nb + nh + N), "at": slice(nb + nh + N, nb + nh + 2 * N)}

    pca = PCA(n_components=2).fit_transform(Xs)
    tsne = TSNE(n_components=2, perplexity=args.perplexity, init="pca", random_state=0).fit_transform(Xs)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, emb, name in [(axes[0], pca, "PCA (linear)"), (axes[1], tsne, "t-SNE (qualitative)")]:
        ax.scatter(emb[sl["ben"], 0], emb[sl["ben"], 1], s=10, c="tab:green", alpha=0.25, label="benign (clean)")
        ax.scatter(emb[sl["har"], 0], emb[sl["har"], 1], s=10, c="tab:purple", alpha=0.18, label="harmful cache (align-check)")
        ax.scatter(emb[sl["cl"], 0], emb[sl["cl"], 1], s=70, c="tab:blue", label="harmful clean (attacks)", zorder=3)
        ax.scatter(emb[sl["at"], 0], emb[sl["at"], 1], s=70, c="tab:red", marker="X", label="adaptive (obfuscated)", zorder=3)
        cl2, at2 = emb[sl["cl"]], emb[sl["at"]]
        for i in range(N):
            ax.annotate("", xy=at2[i], xytext=cl2[i], arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5))
        ax.set_title(name); ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2"); ax.legend(fontsize=8)
    fig.suptitle(f"benign + harmful(clean) + adaptive @ live layer {live_layer} | mean jailbreak={score.mean():.2f}")
    fig.tight_layout(); fig.savefig(args.output + ".png", dpi=140)
    print(f"[viz] wrote {args.output}.png  (check: does 'harmful cache' overlap 'harmful clean'? "
          f"if yes, benign overlay is aligned)", flush=True)


if __name__ == "__main__":
    main()
