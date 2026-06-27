#!/usr/bin/env python3
"""
WS6 controls: re-plot the token-curvature figure with reference lines:
  - random Gaussian baseline (point-count-matched: N=800 for all-token, N=120 for last-token)
  - clustered-but-flat (Gaussian-mixture) band -> shows flat clustering gives LOW delta,
    so our above-baseline values mean 'linearly organized', not hyperbolic.
CPU only; no model. Reads results/ws6_token_curvature.json.
"""
import argparse, json, os
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from analyze_curvature import delta_rel


def controls(D=4096, seed=1):
    rng = np.random.default_rng(seed)
    out = {}
    for n in [120, 800]:
        out[f"rand_{n}"] = delta_rel(rng.standard_normal((n, D)), n_sample=n)["delta_rel_p999"]
    gmm = []
    for k in [8, 20]:
        c = rng.standard_normal((k, D)) * 6.0
        lab = rng.integers(0, k, 800)
        X = c[lab] + rng.standard_normal((800, D))
        gmm.append(delta_rel(X, n_sample=800)["delta_rel_p999"])
    out["gmm_lo"], out["gmm_hi"] = min(gmm), max(gmm)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws6", default="results/ws6_token_curvature.json")
    ap.add_argument("--output", default="results/ws6_token_curvature_controls")
    args = ap.parse_args()
    d = json.load(open(args.ws6))
    c = controls()
    print("[controls]", {k: round(v, 3) for k, v in c.items()}, flush=True)
    Ls = sorted((int(k) for k in d["layers"]))
    g = lambda key: [d["layers"][str(L)][key] for L in Ls]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(Ls, g("alltok_atk"), "o-", c="tab:red", label="all-token (harmful)")
    ax.plot(Ls, g("alltok_ben"), "o-", c="tab:green", label="all-token (benign)")
    ax.plot(Ls, g("lasttok_atk"), "s--", c="darkred", alpha=0.6, label="last-token (harmful)")
    ax.plot(Ls, g("lasttok_ben"), "s--", c="darkgreen", alpha=0.6, label="last-token (benign)")
    ax.axhline(c["rand_800"], ls=":", c="black", label=f"random baseline ({c['rand_800']:.3f})")
    ax.axhspan(c["gmm_lo"], c["gmm_hi"], color="gray", alpha=0.18,
               label=f"clustered-flat (GMM) {c['gmm_lo']:.3f}-{c['gmm_hi']:.3f}")
    ax.axhspan(0.0, c["gmm_lo"], color="tab:blue", alpha=0.06)
    ax.text(0.5, c["gmm_lo"] * 0.5, "hyperbolic regime (tree-like)", fontsize=8, color="tab:blue")
    ax.set_xlabel("layer (0 = embedding)"); ax.set_ylabel("relative δ-hyperbolicity")
    ax.set_ylim(0, max(0.27, max(g("lasttok_atk")) * 1.05))
    ax.set_title("Curvature vs depth: representations are linearly organized (high δ), not hyperbolic\n"
                 "below the gray band = more tree-like than flat clusters; below dotted = below random")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(args.output + ".png", dpi=140)
    json.dump({**d, "controls": c}, open(args.output + ".json", "w"), indent=2)
    print(f"[controls] wrote {args.output}.png/.json", flush=True)


if __name__ == "__main__":
    main()
