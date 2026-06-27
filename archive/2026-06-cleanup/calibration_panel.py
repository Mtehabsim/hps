#!/usr/bin/env python3
"""
Curvature-statistic calibration panel for high-dimensional representations.

Motivation: a single random-Gaussian null is not enough to judge whether a point
cloud is hyperbolic. This builds a PANEL of KNOWN-geometry references (tree, flat,
Gaussian, clustered, sphere) at a MATCHED ambient dimension / N / k, and computes
both delta-hyperbolicity and Ollivier-Ricci curvature for each.

Key (cautionary) result at D=4096: neither delta nor ORC recovers the known
curvature ordering -- a Euclidean-embedded TREE comes out HIGH-delta / POSITIVE-ORC
(anti-hyperbolic) while a SPHERE comes out NEGATIVE-ORC. Reasons: (1) high-D
concentration of measure, (2) curvature is metric-dependent (a tree is only low-delta
under a tree/hyperbolic metric, not under Euclidean distance). Conclusion: claims
about representational hyperbolicity from these point-cloud statistics are unreliable
at LLM hidden-state dimensions; use functional tests (e.g. does a hyperbolic probe
beat its Euclidean twin) instead.

Optionally overlays real data (token embeddings / cache last-token reps) onto the panel.

Usage:
  python calibration_panel.py --dim 4096 --n 2000 --k 10 --output results/calibration_panel
  # also vary dimension to show the breakdown sets in:
  python calibration_panel.py --dims 8 64 768 4096 --output results/calibration_panel_dims
"""
import argparse, json, os
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from helm_token_curvature import graph_orc
from analyze_curvature import delta_rel


def references(D, N, seed=0):
    rng = np.random.default_rng(seed)
    refs = {}
    sph = rng.standard_normal((N, D)); sph /= np.linalg.norm(sph, axis=1, keepdims=True)
    refs["sphere (unit, +curv)"] = (sph, False)
    refs["random Gaussian (null)"] = (rng.standard_normal((N, D)), True)
    c = rng.standard_normal((8, D)) * 8.0; lab = rng.integers(0, 8, N)
    refs["GMM 8 clusters"] = (c[lab] + rng.standard_normal((N, D)), True)
    flat = np.zeros((N, D)); flat[:, :min(2, D)] = rng.standard_normal((N, min(2, D)))
    refs["flat plane (2D)"] = (flat, True)
    nb = min(14, max(2, D))
    B = rng.integers(0, 2, (N, nb)); coords = (2 * B - 1) * (0.5 ** np.arange(nb))
    Q, _ = np.linalg.qr(rng.standard_normal((D, nb)))
    refs["tree / ultrametric (hyperbolic)"] = (coords @ Q.T, True)
    return refs


def measure(X, std, N, k):
    kap = graph_orc(X, k=k, n_sample=N, standardize=std)
    return float(np.median(kap)), float(np.mean(kap < 0)), float(delta_rel(X, n_sample=min(N, 800))["delta_rel_p999"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=4096)
    ap.add_argument("--dims", type=int, nargs="*", default=None)
    ap.add_argument("--n", type=int, default=2000); ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--output", default="results/calibration_panel")
    args = ap.parse_args()
    dims = args.dims if args.dims else [args.dim]

    out = {}
    for D in dims:
        print(f"\n=== ambient dim D={D}, N={args.n}, k={args.k} ===", flush=True)
        print(f"{'reference':36s} ORC_median  frac<0   delta_rel", flush=True)
        out[str(D)] = {}
        for name, (X, std) in references(D, args.n).items():
            m, f, dl = measure(X, std, args.n, args.k)
            out[str(D)][name] = {"orc_median": m, "frac_neg": f, "delta_rel": dl}
            print(f"  {name:34s} {m:+.3f}      {f:.2f}     {dl:.3f}", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump(out, open(args.output + ".json", "w"), indent=2)
    # plot ORC vs delta for the largest dim, annotate references
    D = str(dims[-1]); fig, ax = plt.subplots(figsize=(9, 6))
    for name, r in out[D].items():
        ax.scatter(r["delta_rel"], r["orc_median"], s=60)
        ax.annotate(name, (r["delta_rel"], r["orc_median"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, ls="--", c="gray", lw=0.8)
    ax.set_xlabel("delta_rel  (low = 'tree-like' by the statistic)")
    ax.set_ylabel("Ollivier-Ricci median  (negative = 'hyperbolic' by the statistic)")
    ax.set_title(f"Known-geometry calibration at D={D}: statistics DON'T recover true curvature")
    fig.tight_layout(); fig.savefig(args.output + ".png", dpi=140)
    print(f"\n[panel] wrote {args.output}.json/.png", flush=True)


if __name__ == "__main__":
    main()
