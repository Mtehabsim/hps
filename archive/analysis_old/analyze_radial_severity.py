#!/usr/bin/env python3
"""
WS5 — Explainability: does the hyperbolic radial coordinate encode harmfulness / "depth"?

For each layer we compute the Lorentz radial coordinate  x0 = sqrt(1/kappa + ||x||^2)
of the (standardized) activation, then ask:
  (a) does radial separate harmful from benign?  -> radial-only AUROC per layer
  (b) how strong is the separation?              -> Cohen's d per layer
  (c) at what DEPTH does it emerge?              -> first layer with AUROC >= threshold
This supports (or refutes) the "distance-from-origin = severity / attack depth" story.

Note: x0 is monotonic in ||x||, so this is the norm-based radial proxy (no learned
projection). It matches HPS's lift up to the projection matrix; we can refine with the
trained projection later if the signal is there.

Usage:
  python analyze_radial_severity.py --cache results/llama3_activations_cache_alllayers.npz \
      --kappa 0.1 --auroc_thresh 0.8 --output results/ws5_radial
"""
import argparse, json, os
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.environ.get("MPLCONFIGDIR", "/tmp/mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from analyze_curvature import load_cache, extract_layer, n_layers_in_cache


def radial_for_layer(ben, atk, kappa):
    """Standardize using combined stats; return radial x0 for benign and harmful."""
    Xall = np.concatenate([ben, atk], 0).astype(np.float64)
    mu = Xall.mean(0, keepdims=True)
    sd = Xall.std(0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    rb = np.sqrt(1.0 / kappa + (((ben - mu) / sd) ** 2).sum(1))
    ra = np.sqrt(1.0 / kappa + (((atk - mu) / sd) ** 2).sum(1))
    return rb, ra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/llama3_activations_cache_alllayers.npz")
    ap.add_argument("--layers", default=None)
    ap.add_argument("--kappa", type=float, default=0.1)
    ap.add_argument("--auroc_thresh", type=float, default=0.8)
    ap.add_argument("--output", default="results/ws5_radial")
    args = ap.parse_args()

    d = load_cache(args.cache)
    nL = n_layers_in_cache(d)
    layers = list(range(nL)) if args.layers is None else [int(x) for x in args.layers.split(",")]

    out = {"cache": args.cache, "kappa": args.kappa, "layers": {}}
    aucs, dsep, mb, ma = [], [], [], []
    for L in layers:
        ben = extract_layer(d, L, "ben")
        atk = extract_layer(d, L, "atk")
        rb, ra = radial_for_layer(ben, atk, args.kappa)
        labels = np.concatenate([np.zeros(len(rb)), np.ones(len(ra))])
        scores = np.concatenate([rb, ra])
        auc = float(roc_auc_score(labels, scores))
        # symmetric AUROC (radial may be higher OR lower for harmful)
        auc = max(auc, 1.0 - auc)
        d_eff = float((ra.mean() - rb.mean()) / np.sqrt(0.5 * (ra.var() + rb.var()) + 1e-12))
        out["layers"][str(L)] = {
            "radial_auroc": auc, "cohens_d": d_eff,
            "mean_ben": float(rb.mean()), "mean_atk": float(ra.mean()),
            "n_ben": int(len(rb)), "n_atk": int(len(ra)),
        }
        aucs.append(auc); dsep.append(d_eff); mb.append(rb.mean()); ma.append(ra.mean())
        print(f"[ws5] layer {L:2d}  radial-AUROC={auc:.3f}  Cohen_d={d_eff:+.2f}  "
              f"mean(ben)={rb.mean():.2f} mean(atk)={ra.mean():.2f}", flush=True)

    # depth: first layer reaching the AUROC threshold
    depth = next((L for L, a in zip(layers, aucs) if a >= args.auroc_thresh), None)
    best = int(layers[int(np.argmax(aucs))])
    out["emergence_depth_layer"] = depth
    out["best_layer"] = best
    out["best_radial_auroc"] = float(max(aucs))
    print(f"[ws5] emergence depth (AUROC>={args.auroc_thresh}): layer {depth} | "
          f"best layer {best} (AUROC {max(aucs):.3f})", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output + ".json", "w") as f:
        json.dump(out, f, indent=2)

    # Plot 1: radial-AUROC and Cohen's d vs layer  +  Plot 2: mean radial vs layer
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(layers, aucs, "o-", color="tab:purple", label="radial-only AUROC")
    axes[0].axhline(0.5, ls=":", c="gray"); axes[0].axhline(args.auroc_thresh, ls="--", c="tab:orange")
    if depth is not None:
        axes[0].axvline(depth, ls="--", c="tab:green", alpha=0.6, label=f"emergence depth (L{depth})")
    axes[0].set_ylim(0.4, 1.0); axes[0].set_xlabel("layer"); axes[0].set_ylabel("AUROC")
    axes[0].set_title("Does radial distance alone separate harmful vs benign?"); axes[0].legend()

    axes[1].plot(layers, mb, "o-", color="tab:blue", label="benign mean radial")
    axes[1].plot(layers, ma, "o-", color="tab:red", label="harmful mean radial")
    axes[1].set_xlabel("layer"); axes[1].set_ylabel("radial x0 (standardized)")
    axes[1].set_title("Mean radial vs depth (separation = 'how deep attack shows up')"); axes[1].legend()
    fig.tight_layout(); fig.savefig(args.output + ".png", dpi=140)

    # Plot 3: radial distribution at the best layer (the "severity axis")
    ben = extract_layer(d, best, "ben"); atk = extract_layer(d, best, "atk")
    rb, ra = radial_for_layer(ben, atk, args.kappa)
    fig2, ax = plt.subplots(figsize=(7, 5))
    ax.hist(rb, bins=40, alpha=0.6, label="benign", color="tab:blue", density=True)
    ax.hist(ra, bins=40, alpha=0.6, label="harmful", color="tab:red", density=True)
    ax.set_xlabel(f"radial x0 at layer {best}"); ax.set_ylabel("density")
    ax.set_title(f"Radial 'severity axis' at best layer {best} (AUROC {max(aucs):.3f})"); ax.legend()
    fig2.tight_layout(); fig2.savefig(args.output + "_bestlayer_hist.png", dpi=140)
    print(f"[ws5] wrote {args.output}.json/.png and _bestlayer_hist.png", flush=True)


if __name__ == "__main__":
    main()
