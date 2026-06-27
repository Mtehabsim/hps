#!/usr/bin/env python3
"""
WS1 — Curvature / geometry diagnosis of Llama-3 activations.

Measures Gromov delta-hyperbolicity per layer for harmful vs benign activations.
delta_rel in [0,1]:  ~0 => tree-like / strongly hyperbolic (hyperbolic prior matched);
                     ~1 => flat / not hyperbolic (hyperbolic prior MISmatched).

This directly tests whether HPS's (forced, single-curvature) hyperbolic prior is
appropriate for the activations it operates on — i.e., WHY hyperbolic did/didn't help.

Usage:
  python analyze_curvature.py --cache results/llama3_activations_cache_alllayers.npz \
      --n_sample 800 --n_quad 300000 --output results/ws1_curvature
"""
import argparse, json, os
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.environ.get("MPLCONFIGDIR", "/tmp/mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_cache(path):
    d = np.load(path, allow_pickle=True)
    keys = list(d.keys())
    print(f"[cache] {path}\n[cache] keys: {keys}", flush=True)
    for k in keys:
        a = d[k]
        try:
            print(f"[cache]   {k}: shape={a.shape} dtype={a.dtype}", flush=True)
        except Exception as e:
            print(f"[cache]   {k}: <{e}>", flush=True)
    return d


def _example_layer_lasttoken(ex, layer):
    """ex indexable by layer; ex[layer] -> [seq, hidden] (or [hidden]). Return last-token [hidden]."""
    v = np.asarray(ex[layer], dtype=np.float32)
    if v.ndim == 2:
        return v[-1]
    return v.reshape(-1)


def extract_layer(d, layer, which):
    """which in {'ben','atk'}. Combines train+test. Returns [N, hidden] (last-token)."""
    suffixes = {"ben": ["hs_train_ben", "hs_test_ben"], "atk": ["hs_train_atk", "hs_test_atk"]}
    rows = []
    found = False
    for key in suffixes[which]:
        if key in d:
            found = True
            for ex in d[key].tolist():
                rows.append(_example_layer_lasttoken(ex, layer))
    if not found:
        raise KeyError(
            f"Expected keys {suffixes[which]} not in cache. Keys present: {list(d.keys())}. "
            f"Paste the cache-inspect output and I'll adapt the extractor."
        )
    return np.stack(rows, axis=0)


def n_layers_in_cache(d):
    for key in ["hs_train_ben", "hs_train_atk", "hs_test_ben"]:
        if key in d:
            ex = d[key].tolist()[0]
            return len(ex)
    raise KeyError(f"Could not infer #layers; cache keys: {list(d.keys())}")


def delta_rel(X, n_sample=800, n_quad=300000, standardize=True, seed=0):
    """Relative Gromov delta-hyperbolicity via Monte-Carlo 4-point condition.
    Lower => more tree-like / hyperbolic."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    if standardize:
        mu = X.mean(0, keepdims=True)
        sd = X.std(0, keepdims=True)
        sd[sd < 1e-6] = 1.0
        X = (X - mu) / sd
    n = X.shape[0]
    if n > n_sample:
        sel = rng.choice(n, n_sample, replace=False)
        X = X[sel]
        n = n_sample
    # pairwise Euclidean distance matrix
    sq = (X * X).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * X @ X.T, 0.0))
    diam = D.max()
    if diam <= 0:
        return dict(delta_rel_max=float("nan"), delta_rel_p999=float("nan"), diam=0.0, n=n)
    a = rng.integers(0, n, n_quad); b = rng.integers(0, n, n_quad)
    c = rng.integers(0, n, n_quad); e = rng.integers(0, n, n_quad)
    S = np.stack([D[a, b] + D[c, e], D[a, c] + D[b, e], D[a, e] + D[b, c]], axis=1)
    S.sort(axis=1)  # ascending: S[:,2]=max, S[:,1]=mid
    d4 = (S[:, 2] - S[:, 1]) / 2.0
    return dict(
        delta_rel_max=float(2.0 * d4.max() / diam),
        delta_rel_p999=float(2.0 * np.percentile(d4, 99.9) / diam),
        diam=float(diam), n=int(n),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/llama3_activations_cache_alllayers.npz")
    ap.add_argument("--layers", default=None, help="comma list; default = all layers in cache")
    ap.add_argument("--n_sample", type=int, default=800)
    ap.add_argument("--n_quad", type=int, default=300000)
    ap.add_argument("--no_standardize", action="store_true")
    ap.add_argument("--output", default="results/ws1_curvature")
    args = ap.parse_args()

    d = load_cache(args.cache)
    nL = n_layers_in_cache(d)
    layers = list(range(nL)) if args.layers is None else [int(x) for x in args.layers.split(",")]
    print(f"[ws1] analyzing {len(layers)} layers (cache has {nL})", flush=True)

    out = {"cache": args.cache, "n_sample": args.n_sample, "layers": {}}
    for L in layers:
        rec = {}
        for which in ["ben", "atk"]:
            X = extract_layer(d, L, which)
            rec[which] = delta_rel(X, args.n_sample, args.n_quad, not args.no_standardize)
        # combined (harmful + benign together)
        Xall = np.concatenate([extract_layer(d, L, "ben"), extract_layer(d, L, "atk")], 0)
        rec["all"] = delta_rel(Xall, args.n_sample, args.n_quad, not args.no_standardize)
        out["layers"][str(L)] = rec
        print(
            f"[ws1] layer {L:2d}  delta_rel(p99.9): "
            f"ben={rec['ben']['delta_rel_p999']:.3f}  atk={rec['atk']['delta_rel_p999']:.3f}  "
            f"all={rec['all']['delta_rel_p999']:.3f}",
            flush=True,
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output + ".json", "w") as f:
        json.dump(out, f, indent=2)

    Ls = [int(k) for k in out["layers"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    for which, c in [("ben", "tab:blue"), ("atk", "tab:red"), ("all", "tab:green")]:
        ax.plot(Ls, [out["layers"][str(L)][which]["delta_rel_p999"] for L in Ls],
                marker="o", label=which, color=c)
    ax.axhline(0.0, ls=":", c="gray"); ax.set_ylim(0, 1)
    ax.set_xlabel("layer index"); ax.set_ylabel("relative δ-hyperbolicity (p99.9)")
    ax.set_title("Curvature diagnosis: lower = more tree-like/hyperbolic; ~1 = flat (prior mismatched)")
    ax.legend()
    fig.tight_layout(); fig.savefig(args.output + ".png", dpi=140)
    print(f"[ws1] wrote {args.output}.json and {args.output}.png", flush=True)


if __name__ == "__main__":
    main()
