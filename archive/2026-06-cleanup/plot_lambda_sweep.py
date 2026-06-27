#!/usr/bin/env python3
"""Figure 1: adaptive λ-sweep (obf AUROC + obf TPR@1%FPR) for C4 / rpf / rpfnox0.
Numbers are the measured soft-prompt results (see hps_project_reference.md §3).
The point: rpf (hyperbolic) overlaps rpfnox0 (Euclidean twin) -> hyperbolic lift inert;
by TPR all probes are bypassed at attacker-chosen λ. -> results/lambda_sweep.png
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

lam = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
obf_auroc = {"C4": [0.001, 0.001, 0.001, 0.009, 0.033, 0.086],
             "rpf (hyperbolic)": [0.344, 0.465, 0.825, 0.682, 0.916, 0.985],
             "rpfnox0 (Euclidean twin)": [0.576, 0.842, 0.545, 0.869, 0.872, 0.984]}
obf_tpr = {"C4": [0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
           "rpf (hyperbolic)": [0.005, 0.000, 0.055, 0.010, 0.100, 0.645],
           "rpfnox0 (Euclidean twin)": [0.000, 0.045, 0.015, 0.040, 0.085, 0.620]}
styles = {"C4": dict(marker="s", ls="-", color="k"),
          "rpf (hyperbolic)": dict(marker="o", ls="-", color="tab:red"),
          "rpfnox0 (Euclidean twin)": dict(marker="x", ls="--", color="tab:blue")}

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for k in obf_auroc:
    ax[0].plot(lam, obf_auroc[k], label=k, **styles[k])
    ax[1].plot(lam, obf_tpr[k], label=k, **styles[k])
ax[0].set_title("Obfuscation AUROC vs λ"); ax[0].set_ylabel("obf AUROC")
ax[1].set_title("Obfuscation TPR@1%FPR vs λ (operational)"); ax[1].set_ylabel("obf TPR@1%FPR")
for a in ax:
    a.set_xlabel("attacker behavior weight λ (smaller = more evasion pressure)")
    a.legend(fontsize=8); a.grid(alpha=0.3)
fig.suptitle("Adaptive robustness: hyperbolic lift is inert (rpf ≈ rpfnox0); all probes bypassed by TPR")
fig.tight_layout(); os.makedirs("results", exist_ok=True)
fig.savefig("results/lambda_sweep.png", dpi=140)
print("[fig] wrote results/lambda_sweep.png")
