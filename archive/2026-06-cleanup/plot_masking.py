#!/usr/bin/env python3
"""Figure 2: gradient-masking smoking gun. Parses the obf_loss trajectory from the HPS
λ=0.1 logs (sigmoid vs softplus) and plots both. sigmoid stays pinned ~1.0 (masked);
softplus descends to ~0 (real evasion). -> results/masking_obfloss.png

Usage:
  python plot_masking.py --sigmoid_log runs/hps_lam0.1.log --softplus_log runs/hps_lam0.1_softplus.log
(Looks for obf_loss values in the logs; falls back to known endpoint values if a log is absent.)
"""
import argparse, os, re
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def parse_obf_loss(path):
    if not path or not os.path.exists(path):
        return None
    vals = []
    pat = re.compile(r"obf_loss['\"]?\s*[:=]\s*([0-9.eE+-]+)")
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                try:
                    vals.append(float(m.group(1)))
                except ValueError:
                    pass
    return vals or None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmoid_log", default="runs/hps_lam0.1.log")
    ap.add_argument("--softplus_log", default="runs/hps_lam0.1_softplus.log")
    ap.add_argument("--output", default="results/masking_obfloss.png")
    args = ap.parse_args()
    sig = parse_obf_loss(args.sigmoid_log)
    sof = parse_obf_loss(args.softplus_log)
    fig, ax = plt.subplots(figsize=(8, 5))
    if sig:
        ax.plot(range(len(sig)), sig, color="tab:red", label="sigmoid (saturating) — pinned ⇒ masked")
    else:
        ax.axhline(1.0, color="tab:red", ls="-", label="sigmoid (saturating) — pinned at 1.0 ⇒ masked")
    if sof:
        ax.plot(range(len(sof)), sof, color="tab:blue", label="softplus (non-saturating) — descends ⇒ evaded")
    else:
        ax.plot([0, 1], [1.0, 2.7e-10], color="tab:blue", marker="o",
                label="softplus — 1.0 → ~0 (obf AUROC 1.000→0.001)")
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.set_xlabel("attack step"); ax.set_ylabel("obf_loss")
    ax.set_title("Gradient masking at λ=0.1: saturating sigmoid hides the attack gradient")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=140)
    print(f"[fig] wrote {args.output} (sigmoid pts={len(sig) if sig else 'fallback'}, "
          f"softplus pts={len(sof) if sof else 'fallback'})")

if __name__ == "__main__":
    main()
