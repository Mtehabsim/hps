#!/usr/bin/env python3
"""
The CORRECT way to ask "is this representation hyperbolic": constant-curvature
embedding distortion (Nickel & Kiela 2017; Sala et al. ICML 2018; Gu et al. ICLR 2019).

Instead of reading curvature off a raw point cloud (delta / Ollivier-Ricci -- which we
showed FAIL the known-geometry calibration at high dim), we ask the well-posed question:
    does the data's intrinsic metric embed with LOWER distortion into hyperbolic (H),
    Euclidean (E), or spherical (S) space of the same dimension?
Lowest-distortion geometry wins. This is metric-aware and -- crucially -- it must PASS
a calibration gate: tree-metric -> H wins, sphere -> S wins, flat -> E wins. (delta/ORC
did not.) Only if it passes do we trust it on activations.

Intrinsic metric for a point cloud = graph-geodesic distance on a kNN graph (captures
manifold structure, not raw Euclidean).

Usage:
  python embedding_distortion.py --selftest        # validate the gate
  python embedding_distortion.py --cache results/llama3_activations_cache_alllayers.npz \
      --model_path $MP --n_sample 600 --embed_dim 10 --output results/embedding_distortion
"""
import argparse, json, os, sys
import numpy as np
from scipy.spatial.distance import cdist, squareform, pdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def _distortion(D_emb, D_tgt):
    iu = np.triu_indices_from(D_tgt, k=1)
    a, b = D_emb[iu], D_tgt[iu]
    m = b > 1e-9
    return float(np.mean(np.abs(a[m] - b[m]) / b[m]))


def euclidean_fit(D, n):
    P = D.shape[0]
    J = np.eye(P) - np.ones((P, P)) / P
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    o = np.argsort(w)[::-1]
    wp = np.clip(w[o][:n], 0, None)
    X = V[:, o][:, :n] * np.sqrt(wp + 1e-12)
    return _distortion(cdist(X, X), D)


def spherical_fit(D, n, scales):
    best = np.inf
    for s in scales:
        th = np.clip(D * s, 0, np.pi)
        M = np.cos(th)
        w, V = np.linalg.eigh(M)
        o = np.argsort(w)[::-1]
        wp = np.clip(w[o][:n + 1], 0, None)
        X = V[:, o][:, :n + 1] * np.sqrt(wp + 1e-12)
        nrm = np.linalg.norm(X, axis=1, keepdims=True); nrm[nrm < 1e-9] = 1.0
        X = X / nrm
        G = np.clip(X @ X.T, -1, 1)
        De = np.arccos(G) / s
        best = min(best, _distortion(De, D))
    return best


def hyperbolic_fit(D, n, scales):
    """Sala h-MDS: Lorentzian Gram M_ij = -cosh(s*d_ij); rank-(1 timelike + n spacelike)."""
    best = np.inf
    for s in scales:
        Dc = np.minimum(D * s, 30.0)            # cap to avoid cosh overflow
        M = -np.cosh(Dc)
        w, V = np.linalg.eigh(M)
        o = np.argsort(w)                       # ascending: most-negative first (timelike)
        wt, Vt = w[o[0]], V[:, o[0]]            # timelike: one negative eigenvalue
        op = o[::-1][:n]                        # top-n positive (spacelike)
        wp = np.clip(w[op], 0, None)
        Xs = V[:, op] * np.sqrt(wp + 1e-12)     # spacelike coords
        xt = np.sqrt(np.clip(-wt, 0, None)) * Vt  # timelike coord
        # Lorentzian inner product <xi,xj>_L = -xt_i xt_j + <Xs_i,Xs_j>
        G = -np.outer(xt, xt) + Xs @ Xs.T
        De = np.arccosh(np.clip(-G, 1.0, None)) / s
        best = min(best, _distortion(De, D))
    return best


def fit_all(D, n):
    scales = np.geomspace(0.05, 3.0, 14)
    return {"H": hyperbolic_fit(D, n, scales),
            "E": euclidean_fit(D, n),
            "S": spherical_fit(D, n, scales)}


def graph_geodesic(X, k=10, n_sample=600, standardize=True, seed=0):
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    if standardize:
        mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True); sd[sd < 1e-6] = 1.0
        X = (X - mu) / sd
    if X.shape[0] > n_sample:
        X = X[rng.choice(X.shape[0], n_sample, replace=False)]
    Dpair = cdist(X, X); P = X.shape[0]
    nbr = np.argsort(Dpair, 1)[:, 1:k + 1]
    rows = np.repeat(np.arange(P), k); cols = nbr.ravel()
    G = csr_matrix((Dpair[rows, cols], (rows, cols)), shape=(P, P)); G = G.maximum(G.T)
    Dg = shortest_path(G, directed=False)
    if not np.isfinite(Dg).all():
        Dg[~np.isfinite(Dg)] = Dg[np.isfinite(Dg)].max() * 1.5
    return Dg / np.median(Dg[Dg > 0])           # scale-normalize


def selftest(dims=(2,)):
    rng = np.random.default_rng(0)
    # TREE metric (balanced b-ary tree, true graph distances) -> H should win
    import collections
    b, depth = 3, 6
    edges = []; nodes = [0]; nxt = 1
    q = collections.deque([(0, 0)])
    while q:
        u, dep = q.popleft()
        if dep < depth:
            for _ in range(b):
                edges.append((u, nxt)); q.append((nxt, dep + 1)); nodes.append(nxt); nxt += 1
    Pn = nxt; A = np.zeros((Pn, Pn))
    for u, v in edges: A[u, v] = A[v, u] = 1
    Dtree = shortest_path(csr_matrix(A), directed=False)
    idx = rng.choice(Pn, min(400, Pn), replace=False); Dtree = Dtree[np.ix_(idx, idx)]
    Dtree /= np.median(Dtree[Dtree > 0])
    # SPHERE (geodesic) -> S should win
    Xs = rng.standard_normal((400, 3)); Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
    Dsph = np.arccos(np.clip(Xs @ Xs.T, -1, 1)); Dsph /= np.median(Dsph[Dsph > 0])
    # FLAT (Euclidean plane) -> E should win
    Xe = rng.standard_normal((400, 2)); Dflat = cdist(Xe, Xe); Dflat /= np.median(Dflat[Dflat > 0])
    print("[gate] distortion (lower=better fit); winner should match the shape AT EVERY DIM:", flush=True)
    for n in dims:
        print(f"  -- embed_dim={n} --", flush=True)
        for name, D, exp in [("TREE metric", Dtree, "H"), ("SPHERE geodesic", Dsph, "S"), ("FLAT plane", Dflat, "E")]:
            f = fit_all(D, n=n); win = min(f, key=f.get)
            ok = "OK" if win == exp else "**MISMATCH (method biased at this dim!)**"
            print(f"    {name:16s} H={f['H']:.3f} E={f['E']:.3f} S={f['S']:.3f}  -> {win} (expect {exp}) {ok}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cache"); ap.add_argument("--model_path")
    ap.add_argument("--per_layer", action="store_true", help="scan every cache layer to locate where hyperbolic structure lives")
    ap.add_argument("--n_sample", type=int, default=600); ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--embed_dim", type=int, default=10); ap.add_argument("--output", default="results/embedding_distortion")
    ap.add_argument("--embed_dims", type=int, nargs="*", default=None, help="sweep multiple embedding dims to test winner stability")
    args = ap.parse_args()
    if args.selftest:
        selftest(dims=(args.embed_dims if args.embed_dims else [2, 5, 10, 20, 50])); return

    if args.per_layer and args.cache:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
        from analyze_curvature import load_cache, extract_layer, n_layers_in_cache
        d = load_cache(args.cache); nL = n_layers_in_cache(d)
        dims = args.embed_dims if args.embed_dims else [args.embed_dim]
        # geodesic metric per layer computed ONCE (independent of embed_dim)
        Dgs = {}
        for L in range(nL):
            reps = np.concatenate([extract_layer(d, L, "atk"), extract_layer(d, L, "ben")], 0)
            Dgs[L] = graph_geodesic(reps, args.k, args.n_sample, seed=L)
        res = {str(D): {} for D in dims}
        for D in dims:
            print(f"\n[dist] per-layer distortion at embed_dim={D} (lower=better; E=flat):", flush=True)
            print("layer |   H       E       S    | winner", flush=True)
            for L in range(nL):
                f = fit_all(Dgs[L], D); w = min(f, key=f.get)
                res[str(D)][str(L)] = {**f, "winner": w}
                print(f"{L:5d} | {f['H']:.3f}  {f['E']:.3f}  {f['S']:.3f} |  {w}", flush=True)
        # stability summary: per layer, winners across dims
        print(f"\n[dist] WINNER STABILITY across embed_dims {dims} (E never winning = curved; H stable = hyperbolic lean):", flush=True)
        nflat = 0; nstable_h = 0
        for L in range(nL):
            wins = [res[str(D)][str(L)]["winner"] for D in dims]
            from collections import Counter
            c = Counter(wins); top, cnt = c.most_common(1)[0]
            stable = "stable" if cnt == len(dims) else "MIXED"
            if "E" not in wins: nflat += 1
            if top == "H" and stable == "stable": nstable_h += 1
            print(f"  layer {L:2d}: {wins}  -> {top} ({stable})", flush=True)
        print(f"\n[dist] layers where Euclidean NEVER wins (curved, not flat): {nflat}/{nL}", flush=True)
        print(f"[dist] layers with a STABLE hyperbolic win across all dims: {nstable_h}/{nL}", flush=True)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        json.dump(res, open(args.output + ".json", "w"), indent=2)
        Ls = list(range(nL)); D0 = str(dims[0])
        fig, ax = plt.subplots(figsize=(10, 5))
        for kk, c in [("H", "tab:red"), ("E", "tab:blue"), ("S", "tab:green")]:
            ax.plot(Ls, [res[D0][str(L)][kk] for L in Ls], "o-", c=c, label=f"{kk} (dim {D0})")
        ax.set_xlabel("layer (last-token reps)"); ax.set_ylabel("embedding distortion (lower=better fit)")
        ax.set_title("Per-layer H/E/S fit (E worst = curved; H vs S margin is small)")
        ax.legend(); fig.tight_layout(); fig.savefig(args.output + ".png", dpi=140)
        print(f"[dist] wrote {args.output}.json/.png", flush=True)
        return

    series = {}
    ambient = 4096
    if args.cache:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
        from analyze_curvature import load_cache, extract_layer, n_layers_in_cache
        d = load_cache(args.cache); L = n_layers_in_cache(d) - 1
        reps = np.concatenate([extract_layer(d, L, "atk"), extract_layer(d, L, "ben")], 0)
        ambient = reps.shape[1]
        series["harm-decision reps (last-token)"] = graph_geodesic(reps, args.k, args.n_sample)
    if args.model_path:
        from helm_token_curvature import load_token_embeddings
        E = load_token_embeddings(args.model_path); ambient = E.shape[1]
        series["token embeddings (HELM subspace)"] = graph_geodesic(E, args.k, args.n_sample)
    # MANDATORY matched-dimension random baseline through the SAME pipeline:
    # high-D kNN-geodesics look hyperbolic for ANY cloud, so only the margin BELOW
    # this floor counts as real hyperbolic structure.
    series["random Gaussian (matched-dim FLOOR)"] = graph_geodesic(
        np.random.default_rng(0).standard_normal((max(args.n_sample, 600), ambient)), args.k, args.n_sample)

    out = {}; base_H = None
    print(f"\n[dist] embedding distortion at dim={args.embed_dim} (lower=better fit):", flush=True)
    for name, D in series.items():
        f = fit_all(D, args.embed_dim); win = min(f, key=f.get)
        if "FLOOR" in name: base_H = f["H"]
        out[name] = {**f, "winner": win}
    for name, r in out.items():
        rel = "" if base_H is None or "FLOOR" in name else f"  | H/floor={r['H']/base_H:.2f} ({'HYPERBOLIC' if r['H'] < 0.6*base_H else 'NOT below floor'})"
        print(f"  {name:36s} H={r['H']:.3f} E={r['E']:.3f} S={r['S']:.3f}  -> {r['winner']}{rel}", flush=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump(out, open(args.output + ".json", "w"), indent=2)
    print(f"[dist] wrote {args.output}.json  (genuinely hyperbolic only if H is well BELOW the random floor)", flush=True)


if __name__ == "__main__":
    main()
