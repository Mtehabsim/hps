#!/usr/bin/env python3
"""
Data-driven hierarchy test: does hyperbolic help when the tree is DERIVED FROM THE DATA
(not hand-designed) and includes MULTIPLE benign + MULTIPLE harm clusters?

Addresses the "our hand-taxonomy is wrong/shallow + benign forced into one blob" concern:
  1. extract a LARGE benign subset (+ the harmful reps)
  2. cluster benign+harm together (Ward) -> K data-driven clusters; each tagged harm- or
     benign-majority. The dendrogram gives the tree distances (cophenetic).
  3. train the SAME prototype detector in Hyperbolic vs Euclidean geometry (+ linear C4) to
     classify into the K data clusters; evaluate held-out BINARY harm detection (harm-cluster
     vs benign-cluster) -> AUROC + TPR@1%FPR, H vs E vs C4. Sweep embed_dims x seeds + viz.

If hyperbolic STILL doesn't beat Euclidean/C4 with the data's OWN (deeper, multi-benign)
hierarchy -> the negative is robust to hierarchy choice. If it flips -> our tree was the cause.

Usage:
  python data_driven_hierarchy.py --harmful_npz results/harm_taxonomy_llm_reps.npz \
    --benign_csv obfuscated-activations/inference_time_experiments/datasets/harmful_dataset/benign_train_no_spec_tokens.csv \
    --model_path $MP --layer 24 --n_benign 2000 --n_clusters 30 \
    --embed_dims 8 16 32 --seeds 5 --output results/data_driven
"""
import argparse, json, os, sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
from hierarchical_detector import ProtoNet, train_eval, tpr_at_fpr, extract_benign


def run_once(X, origin, K, d_emb, epochs, beta, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X)); c = int(0.7 * len(X)); tr, te = idx[:c], idx[c:]
    Xtr, Xte = X[tr], X[te]; otr, ote = origin[tr], origin[te]
    # cluster TRAIN into K data-driven clusters (Ward); tag harm/benign by majority
    cl = AgglomerativeClustering(n_clusters=K, linkage="ward").fit(Xtr)
    ytr = cl.labels_
    cent = np.stack([Xtr[ytr == k].mean(0) if (ytr == k).any() else Xtr.mean(0) for k in range(K)])
    harm_frac = np.array([otr[ytr == k].mean() if (ytr == k).any() else 0.0 for k in range(K)])
    harm_cl = np.where(harm_frac > 0.5)[0]; ben_cl = np.where(harm_frac <= 0.5)[0]
    Z = linkage(cent, method="ward"); tree_D = squareform(cophenet(Z)[1])  # KxK data-tree distances
    purity = float(np.mean(np.maximum(harm_frac, 1 - harm_frac)))
    res, viz = {}, {}
    for geo, hyp in [("hyperbolic", True), ("euclidean", False)]:
        lg, enc = train_eval(Xtr, ytr, Xte, K, d_emb, hyp, tree_D, epochs, beta, seed)
        hs = lg[:, harm_cl].max(1) - lg[:, ben_cl].max(1)        # harm-cluster vs benign-cluster
        res[geo] = {"binary_auroc": float(roc_auc_score(ote, hs)), "binary_tpr": tpr_at_fpr(ote, hs)}
        viz[geo] = {"score": hs, "radius": np.linalg.norm(enc, axis=1), "enc": enc}
    clf = LogisticRegression(max_iter=2000).fit(Xtr, otr)
    s = clf.decision_function(Xte)
    res["c4_linear"] = {"binary_auroc": float(roc_auc_score(ote, s)), "binary_tpr": tpr_at_fpr(ote, s)}
    return res, viz, ote, {"n_harm_cl": len(harm_cl), "n_ben_cl": len(ben_cl), "purity": purity}


def make_viz(viz, y_bin, output):
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    for r, geo in enumerate(["hyperbolic", "euclidean"]):
        sc, rad, enc = viz[geo]["score"], viz[geo]["radius"], viz[geo]["enc"]
        hm, bn = y_bin == 1, y_bin == 0
        for col, (data, title) in enumerate([(sc, "harm-score"), (rad, "embedding radius ‖enc‖")]):
            ax[r, col].hist(data[bn], 30, alpha=0.6, label="benign", color="tab:blue", density=True)
            ax[r, col].hist(data[hm], 30, alpha=0.6, label="harmful", color="tab:red", density=True)
            ax[r, col].set_title(f"{geo}: {title}"); ax[r, col].legend(fontsize=7)
        Z = enc - enc.mean(0); _, _, Vt = np.linalg.svd(Z, full_matrices=False); P = Z @ Vt[:2].T
        ax[r, 2].scatter(P[bn, 0], P[bn, 1], s=6, alpha=0.5, label="benign", color="tab:blue")
        ax[r, 2].scatter(P[hm, 0], P[hm, 1], s=6, alpha=0.5, label="harmful", color="tab:red")
        ax[r, 2].set_title(f"{geo}: 2D PCA"); ax[r, 2].legend(fontsize=7)
    fig.suptitle("Data-driven hierarchy: H vs E (multi-cluster benign+harm)")
    fig.tight_layout(); fig.savefig(output + "_viz.png", dpi=140); print(f"[dd] wrote {output}_viz.png", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harmful_npz", required=True)
    ap.add_argument("--benign_reps_npz"); ap.add_argument("--benign_csv"); ap.add_argument("--model_path")
    ap.add_argument("--layer", type=int, default=24); ap.add_argument("--n_benign", type=int, default=2000)
    ap.add_argument("--n_clusters", type=int, default=30)
    ap.add_argument("--embed_dim", type=int, default=16); ap.add_argument("--embed_dims", type=int, nargs="*")
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--beta", type=float, default=1.0); ap.add_argument("--output", default="results/data_driven")
    args = ap.parse_args()
    dims = args.embed_dims if args.embed_dims else [args.embed_dim]

    Xh = np.load(args.harmful_npz, allow_pickle=True)["reps"].astype(np.float32)
    if args.benign_reps_npz and os.path.exists(args.benign_reps_npz):
        Xb = np.load(args.benign_reps_npz)["reps"].astype(np.float32)
    else:
        Xb = np.array([])
    if len(Xb) < args.n_benign and args.benign_csv and args.model_path:
        Xb = extract_benign(args.benign_csv, args.model_path, args.layer, args.n_benign)
        np.savez(args.output + "_benign.npz", reps=Xb)
    X = np.concatenate([Xh, Xb], 0).astype(np.float32)
    origin = np.concatenate([np.ones(len(Xh)), np.zeros(len(Xb))])
    mu, sd = X.mean(0), X.std(0) + 1e-6; X = (X - mu) / sd
    print(f"[dd] {len(Xh)} harmful + {len(Xb)} benign; K={args.n_clusters} data clusters", flush=True)

    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list)); viz0 = None; meta = []
    for D in dims:
        for s in range(args.seeds):
            res, viz, y_bin, info = run_once(X, origin, args.n_clusters, D, args.epochs, args.beta, s)
            meta.append(info)
            for m in res:
                for k, v in res[m].items(): agg[m][k].append(v)
            if viz0 is None: viz0 = (viz, y_bin)
    nr = len(dims) * args.seeds
    print(f"[dd] clusters: ~{np.mean([m['n_harm_cl'] for m in meta]):.0f} harm-majority + ~{np.mean([m['n_ben_cl'] for m in meta]):.0f} benign-majority; mean purity {np.mean([m['purity'] for m in meta]):.2f}", flush=True)
    print(f"\n[dd] {nr} runs (dims {dims} x {args.seeds} seeds)", flush=True)
    print(f"{'model':12s}  binary_AUROC      binary_TPR@1%FPR", flush=True)
    out = {}
    for m in ["hyperbolic", "euclidean", "c4_linear"]:
        a = agg[m]; out[m] = {k: [float(np.mean(v)), float(np.std(v))] for k, v in a.items()}
        print(f"{m:12s}  {np.mean(a['binary_auroc']):.3f}±{np.std(a['binary_auroc']):.3f}   {np.mean(a['binary_tpr']):.3f}±{np.std(a['binary_tpr']):.3f}", flush=True)
    h, e, c = agg["hyperbolic"], agg["euclidean"], agg["c4_linear"]
    print(f"\n[dd] binary TPR: Δ(H−E)={np.mean(h['binary_tpr'])-np.mean(e['binary_tpr']):+.3f}  Δ(H−C4)={np.mean(h['binary_tpr'])-np.mean(c['binary_tpr']):+.3f}", flush=True)
    print(f"[dd] hyperbolic HELPS only if H > E AND H ≥ C4 (data's own hierarchy, multi-benign).", flush=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump(out, open(args.output + ".json", "w"), indent=2)
    if viz0: make_viz(viz0[0], viz0[1], args.output)
    print(f"[dd] wrote {args.output}.json", flush=True)


if __name__ == "__main__":
    main()
