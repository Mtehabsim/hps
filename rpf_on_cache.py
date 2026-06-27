#!/usr/bin/env python3
"""
rpf_on_cache.py — test the CLEAN hyperbolic lift (rpf vs rpfnox0) on a NON-saturated cache.

Motivation: on the de-confounded diverse_fixed cache, detection is ~0.97-0.998 (not 1.0), and the
HPS contrastive projection beats its matched Euclidean twin (0.99 vs 0.968). But that is a noisy
comparison (two separately-trained projections). The CLEAN isometry-matched control is rpf vs
rpfnox0 — identical pipeline, differ ONLY by the Lorentz radial coordinate x0 = sqrt(1/k + ||x||^2).
rpf/rpfnox0 were only tested on saturated (1.0) data before. Here we test them where there's headroom.

Reports AUROC + TPR@5%FPR + TPR@1%FPR for: C4 (per-layer LR, late fusion), rpfnox0 (full 4096
mean-pooled), rpf (full 4096 + x0, mean-pooled). The decisive number is the rpf - rpfnox0 gap:
  ~0  -> the hyperbolic lift is inert even with headroom (negative holds for clean detection)
  >0  -> a real clean hyperbolic gap appears off-saturation (would nuance the clean story)

Usage:
  python rpf_on_cache.py --cache results/llama3_activations_cache_diverse_fixed.npz \
    --layers 0 2 17 24 28 31 --kappa 0.1 --seeds 5
"""
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def _last(ex, L):
    v = np.asarray(ex[L], dtype=np.float32); return v[-1] if v.ndim == 2 else v

def extract(d, which, layers):
    suff = {"ben": ["hs_train_ben", "hs_test_ben"], "atk": ["hs_train_atk", "hs_test_atk"]}[which]
    out = {s: [] for s in ["train", "test"]}
    names = {"hs_train_ben": "train", "hs_test_ben": "test", "hs_train_atk": "train", "hs_test_atk": "test"}
    for key in suff:
        if key not in d: continue
        split = names[key]
        for ex in d[key].tolist():
            out[split].append(np.stack([_last(ex, L) for L in layers]))   # [n_layers, hidden]
    return {k: (np.stack(v) if v else None) for k, v in out.items()}       # [N, n_layers, hidden]

def tpr_at(y, s, fpr):
    neg = np.sort(s[y == 0]); thr = neg[max(0, int(np.ceil((1 - fpr) * len(neg))) - 1)]
    return float((s[y == 1] > thr).mean())

def fit_eval(Xtr, ytr, Xte, yte, seed=0):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, random_state=seed).fit(sc.transform(Xtr), ytr)
    s = clf.decision_function(sc.transform(Xte))
    return roc_auc_score(yte, s), tpr_at(yte, s, 0.05), tpr_at(yte, s, 0.01)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/llama3_activations_cache_diverse_fixed.npz")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 2, 17, 24, 28, 31])
    ap.add_argument("--kappa", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    d = np.load(args.cache, allow_pickle=True)
    L = args.layers; k = args.kappa
    ben = extract(d, "ben", L); atk = extract(d, "atk", L)
    def XY(split):
        X = np.concatenate([atk[split], ben[split]], 0)          # [N, n_layers, hidden]
        y = np.concatenate([np.ones(len(atk[split])), np.zeros(len(ben[split]))])
        return X.astype(np.float32), y
    Xtr, ytr = XY("train"); Xte, yte = XY("test")
    # per-(layer,dim) standardization from train (for the lift)
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    Xtr_s = (Xtr - mu) / sd; Xte_s = (Xte - mu) / sd

    def feats(mode, Xs):
        if mode == "rpfnox0":
            return Xs.mean(1)                                    # [N, hidden]
        if mode == "rpf":
            x0 = np.sqrt(1.0 / k + (Xs ** 2).sum(-1, keepdims=True))   # [N, n_layers, 1]
            return np.concatenate([x0, Xs], -1).mean(1)          # [N, 1+hidden]
        raise ValueError(mode)

    print(f"[rpf-on-cache] {args.cache}\n[rpf-on-cache] layers={L} kappa={k} | "
          f"train N={len(ytr)} test N={len(yte)} (atk {int(yte.sum())}/ben {int((yte==0).sum())})", flush=True)
    res = {}
    # C4: per-layer LR late fusion
    fused = np.zeros(len(yte)); 
    for i in range(len(L)):
        sc = StandardScaler().fit(Xtr[:, i, :]); clf = LogisticRegression(max_iter=3000).fit(sc.transform(Xtr[:, i, :]), ytr)
        fused += clf.decision_function(sc.transform(Xte[:, i, :]))
    res["C4 (per-layer LR)"] = (roc_auc_score(yte, fused), tpr_at(yte, fused, 0.05), tpr_at(yte, fused, 0.01))
    # rpfnox0 / rpf
    for mode in ["rpfnox0", "rpf"]:
        res[mode] = fit_eval(feats(mode, Xtr_s), ytr, feats(mode, Xte_s), yte)

    print(f"\n{'method':22s}  AUROC    TPR@5%FPR  TPR@1%FPR")
    for m, (a, t5, t1) in res.items():
        print(f"{m:22s}  {a:.4f}   {t5:.4f}     {t1:.4f}")
    ga = res["rpf"][0] - res["rpfnox0"][0]; gt = res["rpf"][1] - res["rpfnox0"][1]
    print(f"\nrpf - rpfnox0 gap:  AUROC {ga:+.4f}   TPR@5% {gt:+.4f}")
    if abs(ga) < 0.005 and abs(gt) < 0.01:
        print("VERDICT: hyperbolic lift INERT even off-saturation -> clean negative holds.")
    elif ga > 0.005:
        print("VERDICT: rpf > rpfnox0 off-saturation -> a real clean hyperbolic gap appears. Investigate!")
    else:
        print("VERDICT: rpf < rpfnox0 -> the lift hurts.")

if __name__ == "__main__":
    main()
