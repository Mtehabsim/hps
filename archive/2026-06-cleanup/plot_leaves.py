#!/usr/bin/env python3
"""
Visualize the harm-taxonomy LEAVES (categories) as colored clusters in representation space:
  Panel 1: 2D PCA of harm reps, colored by leaf
  Panel 2: 2D t-SNE of harm reps, colored by leaf  (clusters clearer)
  Panel 3: learned HYPERBOLIC embedding on the Poincaré disk, colored by leaf  (the "tree" layout;
           prototypes shown as stars) -- this is literally where each leaf lands in hyperbolic space.
Optionally overlays benign as gray.

Usage:
  python plot_leaves.py --harmful_npz results/harm_taxonomy_llm_reps.npz \
    --benign_reps_npz results/hier_detector_benign.npz --epochs 300 --output results/leaves
"""
import argparse, os, sys
import numpy as np
import torch
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
from hierarchical_detector import ProtoNet, expmap0


def pca2(X):
    Z = X - X.mean(0); _, _, Vt = np.linalg.svd(Z, full_matrices=False); return Z @ Vt[:2].T


def train_hyp_2d(X, y, n_class, epochs, seed=0):
    torch.manual_seed(seed); dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = ProtoNet(X.shape[1], 2, n_class, hyperbolic=True).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-4)
    Xt = torch.tensor(X, dtype=torch.float32, device=dev); yt = torch.tensor(y, device=dev)
    ce = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad(); loss = ce(net.logits(Xt), yt); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
    net.eval()
    with torch.no_grad():
        enc = net.encode(Xt)                       # [N,2] tangent
        L = expmap0(enc).cpu().numpy()             # [N,3] Lorentz (x0,x1,x2)
        Lp = expmap0(net.proto).cpu().numpy()
    def poincare(Lz): return Lz[:, 1:] / (1.0 + Lz[:, :1])   # -> unit disk
    return poincare(L), poincare(Lp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harmful_npz", required=True); ap.add_argument("--benign_reps_npz")
    ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--output", default="results/leaves")
    args = ap.parse_args()
    d = np.load(args.harmful_npz, allow_pickle=True)
    Xh = d["reps"].astype(np.float32); labels = d["labels"]
    leaves = sorted(set(labels.tolist())); ix = {l: i for i, l in enumerate(leaves)}
    y = np.array([ix[l] for l in labels]); nL = len(leaves)
    Xb = np.load(args.benign_reps_npz)["reps"].astype(np.float32) if (args.benign_reps_npz and os.path.exists(args.benign_reps_npz)) else None
    mu, sd = Xh.mean(0), Xh.std(0) + 1e-6; Xhs = (Xh - mu) / sd
    Xbs = (Xb - mu) / sd if Xb is not None else None

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(nL)]
    # projections
    P_all = pca2(np.concatenate([Xhs, Xbs], 0) if Xbs is not None else Xhs)
    P_pca = P_all[:len(Xhs)]; P_pca_b = P_all[len(Xhs):] if Xbs is not None else None
    try:
        from sklearn.manifold import TSNE
        P_tsne = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(Xhs)
    except Exception as e:
        print(f"[leaves] tSNE failed ({e}); using PCA", flush=True); P_tsne = P_pca
    poin, poin_proto = train_hyp_2d(Xhs, y, nL, args.epochs)

    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    def scatter_by_leaf(a, P, title, disk=False):
        if disk:
            th = np.linspace(0, 2 * np.pi, 200); a.plot(np.cos(th), np.sin(th), c="lightgray", lw=1)
        for i, lf in enumerate(leaves):
            m = y == i
            a.scatter(P[m, 0], P[m, 1], s=12, color=colors[i], label=lf, alpha=0.7)
        a.set_title(title)
    if P_pca_b is not None:
        ax[0].scatter(P_pca_b[:, 0], P_pca_b[:, 1], s=8, color="lightgray", alpha=0.4, label="benign")
    scatter_by_leaf(ax[0], P_pca, "PCA (harm reps, by leaf)")
    scatter_by_leaf(ax[1], P_tsne, "t-SNE (harm reps, by leaf)")
    scatter_by_leaf(ax[2], poin, "Hyperbolic Poincaré disk (learned, by leaf)", disk=True)
    ax[2].scatter(poin_proto[:, 0], poin_proto[:, 1], marker="*", s=200, c=colors, edgecolor="k", linewidth=0.5)
    ax[1].legend(fontsize=7, ncol=2, loc="best")
    fig.suptitle("Harm-taxonomy leaves in representation space (each color = one of 14 categories; ★ = hyperbolic prototype)")
    fig.tight_layout(); os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output + ".png", dpi=140); print(f"[leaves] wrote {args.output}.png", flush=True)


if __name__ == "__main__":
    main()
