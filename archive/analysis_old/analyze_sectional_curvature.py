#!/usr/bin/env python3
"""
Signed sectional curvature of Llama-3 activations via the geodesic-triangle
(Toponogov / Apollonius-median) estimator.

NOTE: Ollivier-Ricci on a Euclidean point cloud was tried first and DISCARDED --
its sign failed validation (clustered/hierarchical data came out positive), because
it measures clustering/density, not curvature. This estimator was validated on
synthetic data: flat -> ~0, sphere -> +, hyperbolic -> -.

Method per layer:
  build kNN graph on the standardized activations, use graph shortest-paths as
  geodesics, sample triangles (a,b,c), find the geodesic midpoint m of (b,c), and
  compare the median length to the flat (Apollonius) prediction:
      gamma = d_g(a,m)^2 - [ d_g(a,b)^2/2 + d_g(a,c)^2/2 - d_g(b,c)^2/4 ]   (/ d_g(b,c)^2)
  gamma < 0 => hyperbolic ; ~0 => flat ; > 0 => spherical.
Compared to a dimension-matched random-Gaussian baseline (high-D concentration
makes random data slightly negative, so the baseline is the reference, as with delta).

Usage:
  python analyze_sectional_curvature.py --cache results/llama3_activations_cache_alllayers.npz \
      --n_sample 400 --k 12 --n_tri 4000 --output results/sectional_curvature
"""
import argparse, json, os
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from analyze_curvature import load_cache, extract_layer, n_layers_in_cache


def sectional_curvature(X, k=12, n_sample=400, n_tri=4000, standardize=True, seed=0):
    """Mean normalized geodesic-triangle curvature. <0 hyperbolic, ~0 flat, >0 spherical."""
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
    nbr = np.argsort(D, axis=1)[:, 1:k + 1]
    rows = np.repeat(np.arange(n), k); cols = nbr.ravel(); w = D[rows, cols]
    G = csr_matrix((w, (rows, cols)), shape=(n, n)); G = G.maximum(G.T)
    Dg = shortest_path(G, method="D", directed=False)
    if not np.isfinite(Dg).all():            # disconnected graph -> cap infinities
        Dg[~np.isfinite(Dg)] = Dg[np.isfinite(Dg)].max() * 2.0
    vals = []
    for _ in range(n_tri):
        a, b, c = rng.choice(n, 3, replace=False)
        dbc = Dg[b, c]
        if dbc < 1e-9:
            continue
        m = int(np.argmin(np.abs(Dg[b] - dbc / 2) + np.abs(Dg[c] - dbc / 2)))  # geodesic midpoint
        flat = Dg[a, b] ** 2 / 2 + Dg[a, c] ** 2 / 2 - dbc ** 2 / 4
        vals.append((Dg[a, m] ** 2 - flat) / (dbc ** 2 + 1e-9))
    v = np.array(vals)
    return float(v.mean()), float(v.std() / np.sqrt(len(v)))   # mean, standard error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--n_sample", type=int, default=400)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--n_tri", type=int, default=4000)
    ap.add_argument("--output", default="results/sectional_curvature")
    args = ap.parse_args()

    d = load_cache(args.cache)
    nL = n_layers_in_cache(d)
    Dh = extract_layer(d, 0, "atk").shape[1]
    bmean, bse = sectional_curvature(np.random.default_rng(1).standard_normal((args.n_sample, Dh)),
                                     k=args.k, n_sample=args.n_sample, n_tri=args.n_tri)
    print(f"[curv] layers={nL} k={args.k} n_tri={args.n_tri}", flush=True)
    print(f"[curv] dimension-matched random baseline = {bmean:+.4f} (se {bse:.4f})", flush=True)
    print("layer |    atk        ben     | baseline   (neg & below baseline => hyperbolic)", flush=True)

    res = {"baseline": bmean, "baseline_se": bse, "k": args.k, "layers": {}}
    for L in range(nL):
        am, ase = sectional_curvature(extract_layer(d, L, "atk"), k=args.k, n_sample=args.n_sample, n_tri=args.n_tri)
        bm, bse2 = sectional_curvature(extract_layer(d, L, "ben"), k=args.k, n_sample=args.n_sample, n_tri=args.n_tri)
        res["layers"][str(L)] = {"atk": am, "atk_se": ase, "ben": bm, "ben_se": bse2}
        flag = "  <-- below baseline" if (am < bmean - 2 * bse or bm < bmean - 2 * bse) else ""
        print(f"{L:5d} | {am:+.4f}±{ase:.3f}  {bm:+.4f}±{bse2:.3f} | {bmean:+.4f}{flag}", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump(res, open(args.output + ".json", "w"), indent=2)
    Ls = list(range(nL))
    fig, ax = plt.subplots(figsize=(10, 5))
    am = [res["layers"][str(L)]["atk"] for L in Ls]; ae = [res["layers"][str(L)]["atk_se"] for L in Ls]
    bm = [res["layers"][str(L)]["ben"] for L in Ls]; be = [res["layers"][str(L)]["ben_se"] for L in Ls]
    ax.errorbar(Ls, am, yerr=ae, fmt="o-", c="tab:red", label="harmful", capsize=2)
    ax.errorbar(Ls, bm, yerr=be, fmt="o-", c="tab:green", label="benign", capsize=2)
    ax.axhline(bmean, ls=":", c="black", label=f"random baseline ({bmean:+.3f})")
    ax.axhline(0.0, ls="--", c="gray", lw=0.8)
    lo = min(min(am), min(bm), bmean) - 0.02
    ax.fill_between(Ls, lo, min(0, bmean), color="tab:blue", alpha=0.06)
    ax.text(0.5, lo + 0.005, "hyperbolic (curv<0, below baseline)", fontsize=8, color="tab:blue")
    ax.set_xlabel("layer"); ax.set_ylabel("sectional curvature (signed)")
    ax.set_title("Signed sectional curvature vs depth (validated estimator)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(args.output + ".png", dpi=140)
    print(f"[curv] wrote {args.output}.json/.png", flush=True)


if __name__ == "__main__":
    main()
