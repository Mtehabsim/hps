#!/usr/bin/env python3
"""
Attacker-cost-curve harness: evasion vs attacker BUDGET, per probe.

The robustness question isn't "final obf_auc" but "how much budget must the attacker
spend to evade?". This aggregates attacker-inning [RESULT] lines across runs and plots
evasion (obf_recall@1%FPR and obf_auc) vs budget (lambda for soft, or num_steps for GCG)
for each probe -- so we can see whether a hyperbolic probe (rpf) raises the budget vs its
Euclidean twin (rpfnox0) and vs C4.

GRADIENT-MASKING CONTROL: pass both a soft (gradient) sweep and a GCG (gradient-free,
discrete) sweep. If a probe resists the soft attack but a GCG attack evades it cheaply,
the "robustness" is gradient masking (Athalye et al.), not real.

Usage:
  python attack_cost_curve.py --logdir obfuscated-activations/inference_time_experiments/runs \
      --groups c4:c4_lam rpf:rpf_lam rpfnox0:rpfnox0_lam hps:hps_lam \
      --budget lambda --output results/attack_cost_curve
"""
import argparse, glob, json, os, re
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

RES = re.compile(r"\[RESULT\] who=attacker .*?obfuscated_auc=([0-9.]+).*?"
                 r"obfuscated_recall@1%FPR=([0-9.]+).*?jailbreak_behavior=([0-9.nan]+)")
LAM = re.compile(r"promoted root overrides into experiment: \[[^\]]*lambda_=([0-9.]+)")
STEPS = re.compile(r"num_steps[=: ]+(\d+)")


def parse_log(path, budget):
    txt = open(path, errors="ignore").read()
    res = None
    for m in RES.finditer(txt):
        res = m  # last attacker result
    if not res:
        return None
    b = None
    if budget == "lambda":
        lm = LAM.findall(txt)
        if lm: b = float(lm[-1])
        else:
            mm = re.search(r"lam([0-9.]+)", os.path.basename(path))
            b = float(mm.group(1)) if mm else None
    else:
        sm = STEPS.findall(txt); b = int(sm[-1]) if sm else None
    return dict(budget=b, obf_auc=float(res.group(1)),
                obf_recall=float(res.group(2)),
                jailbreak=float("nan") if "nan" in res.group(3) else float(res.group(3)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--groups", nargs="+", required=True, help="name:fileprefix ...")
    ap.add_argument("--budget", choices=["lambda", "steps"], default="lambda")
    ap.add_argument("--output", default="results/attack_cost_curve")
    args = ap.parse_args()

    data = {}
    for g in args.groups:
        name, pref = g.split(":", 1)
        rows = []
        for f in sorted(glob.glob(os.path.join(args.logdir, pref + "*.log"))):
            r = parse_log(f, args.budget)
            if r and r["budget"] is not None:
                rows.append(r)
        rows.sort(key=lambda r: r["budget"])
        data[name] = rows
        print(f"[cost] {name}: {[(r['budget'], round(r['obf_recall'],3), round(r['obf_auc'],3), round(r['jailbreak'],3)) for r in rows]}", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump(data, open(args.output + ".json", "w"), indent=2)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for name, rows in data.items():
        if not rows: continue
        xb = [r["budget"] for r in rows]
        ax[0].plot(xb, [r["obf_recall"] for r in rows], "o-", label=name)
        ax[1].plot(xb, [r["obf_auc"] for r in rows], "o-", label=name)
    xlab = "attacker budget (lambda = evasion weight)" if args.budget == "lambda" else "attacker budget (GCG steps)"
    for a, t in [(ax[0], "obf recall@1%FPR (high = probe still flags attack)"),
                 (ax[1], "obf AUROC (high = probe not fooled)")]:
        a.set_xlabel(xlab); a.set_ylabel(t); a.set_ylim(-0.02, 1.02); a.axhline(0.5, ls=":", c="gray"); a.legend(fontsize=8)
    ax[0].set_title("Attacker-cost curve: evasion vs budget")
    fig.suptitle("Higher/flatter curve = costlier to evade. Compare rpf (hyperbolic) vs rpfnox0 (Euclidean twin).")
    fig.tight_layout(); fig.savefig(args.output + ".png", dpi=140)
    print(f"[cost] wrote {args.output}.json/.png", flush=True)


if __name__ == "__main__":
    main()
