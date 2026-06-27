#!/usr/bin/env python3
"""
Visualize that harm/benign OVERLAP is adversarially INDUCED, not natural.

Parses the attack framework's stored PCA coordinates (clean benign, clean harmful,
obfuscated/attacked harmful) from one or more run logs and plots them. Shows:
  - clean harmful (red) cleanly SEPARATED from benign (blue)  [natural data is separable]
  - obfuscated-harmful (orange ×) dragged INTO the benign cluster [overlap is attack-induced]

Usage:
  python plot_obfuscation.py --logs runs/c4_lam0.5.log runs/hps_lam0.5.log \
      --labels "C4 (evaded)" "HPS (evaded)" --output results/obfuscation_pca
"""
import argparse, json, re
import numpy as np
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def parse_pca(logpath):
    t = open(logpath, errors="ignore").read()
    m = re.search(r"'pca_plotting_data': '([^']*)'", t)
    if not m:
        return None
    d = json.loads(m.group(1))
    ben = np.array(d["standard_benign"]); harm = np.array(d["standard_harmful"])
    obf_key = next((k for k in d if k.startswith("obf_harmful")), None)
    obf = np.array(d[obf_key]) if obf_key else None
    # also pull the obfuscated AUROC if present (attacker inning)
    auc = re.findall(r"\[RESULT\] who=attacker .*?obfuscated_auc=([0-9.]+)", t)
    return ben, harm, obf, (float(auc[-1]) if auc else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--output", default="results/obfuscation_pca")
    args = ap.parse_args()
    labels = args.labels if args.labels else [os.path.basename(l) for l in args.logs]

    n = len(args.logs)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5.6), squeeze=False)
    for i, (lp, lab) in enumerate(zip(args.logs, labels)):
        ax = axes[0, i]; parsed = parse_pca(lp)
        if parsed is None:
            ax.set_title(f"{lab}: no pca data"); continue
        ben, harm, obf, auc = parsed
        ax.scatter(ben[:, 0], ben[:, 1], s=10, alpha=0.5, c="tab:blue", label="benign (clean)")
        ax.scatter(harm[:, 0], harm[:, 1], s=10, alpha=0.5, c="tab:red", label="harmful (clean)")
        if obf is not None:
            ax.scatter(obf[:, 0], obf[:, 1], s=22, alpha=0.7, c="orange", marker="x",
                       label="harmful (obfuscated/attacked)")
        sub = f"  obf_auc={auc:.3f}" if auc is not None else ""
        ax.set_title(f"{lab}{sub}"); ax.legend(fontsize=8); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.suptitle("Clean harmful/benign are separable; the attack DRAGS harmful into the benign region\n"
                 "(overlap is adversarially induced, not a natural property of the data)")
    fig.tight_layout(); os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output + ".png", dpi=140)
    print(f"[obf-viz] wrote {args.output}.png", flush=True)


if __name__ == "__main__":
    main()
