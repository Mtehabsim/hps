#!/usr/bin/env python3
"""
HELM-style curvature replication (arXiv:2505.24722, Appendix A.1).

HELM motivates hyperbolic LLMs by computing Ollivier-Ricci curvature (ORC) on a
kNN graph of TOKEN embeddings and observing broadly NEGATIVE curvature. Crucially,
that is the token/vocabulary subspace -- NOT the last-token "is-this-harmful"
representation that the HPS probe (and our delta analysis) operate on.

This script computes graph ORC for:
  (A) the model's token-embedding matrix         (HELM's object  -> expect negative)
  (B) last-token harmful/benign reps from cache  (our/HPS object -> expect ~flat)
  (C) a dimension-matched random Gaussian        (baseline)
and overlays the curvature distributions, so we can show WHERE hyperbolicity lives
vs. where the jailbreak-detection signal lives.

Graph ORC (faithful to HELM A.1): for each directed kNN edge (i,j),
    kappa(i,j) = 1 - W1(mu_i, mu_j) / d_G(i,j),
mu_i = uniform over i's k nearest neighbours, ground metric = graph shortest-path
distance, W1 = optimal assignment (equal-size uniform measures). kappa<0 => neighbours
diverge (tree-like / hyperbolic); kappa>0 => converge (spherical/clustered).

Usage:
  # validate the estimator's sign convention
  python helm_token_curvature.py --selftest
  # run on Llama-3 token embeddings + our last-token cache
  python helm_token_curvature.py --model_path $MP \
      --cache results/llama3_activations_cache_alllayers.npz \
      --n_sample 3000 --k 10 --output results/helm_token_curvature
"""
import argparse, glob, json, os, sys
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.optimize import linear_sum_assignment
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def graph_orc(X, k=10, n_sample=3000, standardize=True, seed=0, max_edges=20000):
    """Per-edge Ollivier-Ricci curvature on the kNN graph of point cloud X."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    if standardize:
        mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True)
        sd[sd < 1e-6] = 1.0
        X = (X - mu) / sd
    if X.shape[0] > n_sample:
        X = X[rng.choice(X.shape[0], n_sample, replace=False)]
    n = X.shape[0]
    D = cdist(X, X)
    nbr = np.argsort(D, axis=1)[:, 1:k + 1]               # k nearest (exclude self)
    rows = np.repeat(np.arange(n), k); cols = nbr.ravel()
    G = csr_matrix((np.ones(n * k), (rows, cols)), shape=(n, n))
    G = G.maximum(G.T)                                     # symmetric, unweighted
    Dg = shortest_path(G, method="D", directed=False, unweighted=True)
    if not np.isfinite(Dg).all():
        Dg[~np.isfinite(Dg)] = Dg[np.isfinite(Dg)].max() + 1.0
    # sample edges to bound cost
    edges = [(i, int(j)) for i in range(n) for j in nbr[i]]
    if len(edges) > max_edges:
        idx = rng.choice(len(edges), max_edges, replace=False)
        edges = [edges[t] for t in idx]
    kappas = []
    for i, j in edges:
        Ni, Nj = nbr[i], nbr[j]
        C = Dg[np.ix_(Ni, Nj)]                             # k x k graph-distance cost
        r, c = linear_sum_assignment(C)
        W1 = C[r, c].mean()                                # uniform 1/k masses
        d = Dg[i, j]
        if d > 0:
            kappas.append(1.0 - W1 / d)
    return np.array(kappas)


def selftest():
    rng = np.random.default_rng(0)
    # tree / ultrametric points (hierarchical) -> expect NEGATIVE
    def tree_pts(nbits=10, n=1500):
        P = []
        for _ in range(n):
            b = rng.integers(0, 2, nbits)
            P.append([(2 * b[i] - 1) * (0.5 ** i) for i in range(nbits)])
        return np.array(P)
    # 2D grid (flat) -> expect ~0
    g = np.array([[x, y] for x in range(40) for y in range(40)], float)
    # tight gaussian cluster (dense) -> expect POSITIVE
    clust = rng.standard_normal((1500, 5)) * 0.05
    # high-D random gaussian baseline
    rnd = rng.standard_normal((1500, 4096))
    for name, X, std in [("tree/ultrametric", tree_pts(), True),
                         ("2D grid (flat)", g, False),
                         ("dense gaussian cluster", clust, True),
                         ("random 4096-D (baseline)", rnd, True)]:
        kap = graph_orc(X, k=10, standardize=std)
        print(f"  {name:28s} median ORC = {np.median(kap):+.3f}  (frac<0 = {np.mean(kap < 0):.2f})", flush=True)


def load_token_embeddings(model_path):
    """Read the token-embedding matrix without loading the whole model if possible."""
    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    key = "model.embed_tokens.weight"
    try:
        from safetensors import safe_open
        if os.path.exists(idx_path):
            shard = json.load(open(idx_path))["weight_map"][key]
        else:
            shard = os.path.basename(glob.glob(os.path.join(model_path, "*.safetensors"))[0])
        with safe_open(os.path.join(model_path, shard), framework="pt") as f:
            return f.get_tensor(key).float().numpy()
    except Exception as e:
        print(f"[helm] safetensors path failed ({e}); loading full model", flush=True)
        import torch
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,
                                                 low_cpu_mem_usage=True, device_map="cpu")
        return m.get_input_embeddings().weight.detach().float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model_path"); ap.add_argument("--cache")
    ap.add_argument("--n_sample", type=int, default=3000); ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--output", default="results/helm_token_curvature")
    args = ap.parse_args()

    if args.selftest:
        print("[helm] graph-ORC sign validation (tree<0, grid~0, cluster>0):", flush=True)
        selftest(); return

    series = {}
    if args.model_path:
        E = load_token_embeddings(args.model_path)
        print(f"[helm] token-embedding matrix: {E.shape}", flush=True)
        series["token embeddings (HELM object)"] = ("tab:red", graph_orc(E, k=args.k, n_sample=args.n_sample))
    if args.cache:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
        from analyze_curvature import load_cache, extract_layer, n_layers_in_cache
        d = load_cache(args.cache); L = n_layers_in_cache(d) - 1   # final layer
        last = np.concatenate([extract_layer(d, L, "atk"), extract_layer(d, L, "ben")], 0)
        print(f"[helm] last-token reps (final layer {L}): {last.shape}", flush=True)
        series["last-token harm reps (HPS object)"] = ("tab:blue", graph_orc(last, k=args.k, n_sample=args.n_sample))
        dim = last.shape[1]
    else:
        dim = 4096
    series["random Gaussian (baseline)"] = ("gray", graph_orc(
        np.random.default_rng(1).standard_normal((args.n_sample, dim)), k=args.k, n_sample=args.n_sample))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    summary = {}
    fig, ax = plt.subplots(figsize=(10, 5))
    print("\n[helm] === Ollivier-Ricci curvature (negative = hyperbolic) ===", flush=True)
    for name, (color, kap) in series.items():
        med, frac = float(np.median(kap)), float(np.mean(kap < 0))
        summary[name] = {"median": med, "mean": float(kap.mean()), "frac_negative": frac, "n_edges": int(kap.size)}
        print(f"[helm] {name:38s} median={med:+.3f}  mean={kap.mean():+.3f}  frac(kappa<0)={frac:.2f}", flush=True)
        ax.hist(kap, bins=60, density=True, alpha=0.45, color=color, label=f"{name} (med {med:+.2f})")
    ax.axvline(0, ls="--", c="black", lw=1)
    ax.set_xlabel("Ollivier-Ricci curvature  (negative = hyperbolic / tree-like)")
    ax.set_ylabel("density"); ax.set_title("Where hyperbolicity lives: token embeddings vs last-token harm reps")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(args.output + ".png", dpi=140)
    json.dump(summary, open(args.output + ".json", "w"), indent=2)
    print(f"[helm] wrote {args.output}.png/.json", flush=True)


if __name__ == "__main__":
    main()
