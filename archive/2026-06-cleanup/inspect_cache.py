#!/usr/bin/env python3
"""
Inspect a project activation cache (suspicion #2 + norm-confound check).

Prints keys/shapes/N-per-class, and at a chosen layer computes the per-class activation-NORM
distribution + the AUROC of ||last-token|| alone for harm-vs-benign.

  norm-only AUROC ≈ 1.0  -> the cache carries the NORM CONFOUND (probe can separate by magnitude
                            alone); NOT a confound-fixed cache. Do NOT use for geometry claims.
  norm-only AUROC ≈ 0.5  -> norm is uninformative; cache is norm-clean.

Use to pick the de-confounded canonical cache before re-running embedding-distortion / curvature /
hierarchical analyses.

Usage:
  python inspect_cache.py --cache results/llama3_activations_cache_alllayers.npz --layer 24
  python inspect_cache.py --cache results/llama3_activations_cache_diverse_fixed.npz --layer 24
"""
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score


def _last(ex, layer):
    v = np.asarray(ex[layer], dtype=np.float32)
    return v[-1] if v.ndim == 2 else v


def collect(d, which, layer):
    suff = {"ben": ["hs_train_ben", "hs_test_ben"], "atk": ["hs_train_atk", "hs_test_atk"]}[which]
    rows = []
    for key in suff:
        if key in d:
            for ex in d[key].tolist():
                rows.append(_last(ex, layer))
    return np.stack(rows) if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--layer", type=int, default=24)
    args = ap.parse_args()
    d = np.load(args.cache, allow_pickle=True)
    keys = list(d.keys())
    print(f"[inspect] {args.cache}\n[inspect] keys: {keys}", flush=True)
    for k in keys:
        try:
            a = d[k]
            n = len(a.tolist()) if a.dtype == object else a.shape
            print(f"[inspect]   {k}: n/shape={n}", flush=True)
        except Exception as e:
            print(f"[inspect]   {k}: <{e}>", flush=True)

    ben = collect(d, "ben", args.layer); atk = collect(d, "atk", args.layer)
    if ben is None or atk is None:
        print("[inspect] missing ben/atk keys; cannot run norm check."); return
    nb = np.linalg.norm(ben, axis=1); na = np.linalg.norm(atk, axis=1)
    y = np.r_[np.ones(len(na)), np.zeros(len(nb))]; s = np.r_[na, nb]
    auc = roc_auc_score(y, s); auc = max(auc, 1 - auc)
    print(f"\n=== norm confound check @ layer {args.layer} ===")
    print(f"benign  ||x||: mean={nb.mean():.2f} std={nb.std():.2f} median={np.median(nb):.2f}  (N={len(nb)})")
    print(f"harmful ||x||: mean={na.mean():.2f} std={na.std():.2f} median={np.median(na):.2f}  (N={len(na)})")
    print(f"NORM-ONLY AUROC (harm vs benign) = {auc:.3f}")
    if auc >= 0.85:
        print("VERDICT: STRONG norm confound -> NOT a clean cache. Do not base geometry claims on it.")
    elif auc >= 0.65:
        print("VERDICT: moderate norm signal -> partially confounded; prefer a cleaner cache.")
    else:
        print("VERDICT: norm uninformative -> cache is norm-clean. OK for geometry analyses.")


if __name__ == "__main__":
    main()
