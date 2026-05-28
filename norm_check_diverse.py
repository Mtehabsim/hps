"""
norm_check_diverse.py — Re-check the activation norm confound on the
diverse-benign cache.

Phase B's CHECK 6 found norm-only AUROC=0.917 on the OLD cache. That cache
used narrow Alpaca benign. With diverse benign matched to attack lengths,
do attack activations still have systematically different norms than benign?

If norm-only AUROC drops:  saturation is genuine harm-detection
If norm-only AUROC stays:  there's a magnitude confound beyond length
"""

import os, sys, argparse, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

LLAMA_LAYERS = [0, 2, 17, 24, 28, 31]
VICUNA_LAYERS = [0, 2, 22, 31, 35, 39]


def evaluate_norm_only(X_tr_ben, X_tr_atk, X_te_ben, X_te_atk):
    """Train LR on per-layer L2 norms (6-dim feature) and report AUROC/TPR."""
    def norms(X):
        # X: (N, n_layers, d). Returns (N, n_layers).
        return np.linalg.norm(X, axis=2)

    n_tr = np.concatenate([norms(X_tr_ben), norms(X_tr_atk)])
    y_tr = np.array([0]*len(X_tr_ben) + [1]*len(X_tr_atk))
    n_te_ben = norms(X_te_ben)
    n_te_atk = norms(X_te_atk)

    sc = StandardScaler()
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(sc.fit_transform(n_tr), y_tr)

    # Held-out calibration split
    n_calib = len(n_te_ben) // 2
    s_calib = clf.predict_proba(sc.transform(n_te_ben[:n_calib]))[:, 1]
    s_ben = clf.predict_proba(sc.transform(n_te_ben[n_calib:]))[:, 1]
    s_atk = clf.predict_proba(sc.transform(n_te_atk))[:, 1]

    thr5 = float(np.quantile(s_calib, 0.95))
    thr1 = float(np.quantile(s_calib, 0.99))
    tpr5 = float((s_atk > thr5).mean())
    tpr1 = float((s_atk > thr1).mean())
    auroc = roc_auc_score(
        np.array([0]*len(s_ben) + [1]*len(s_atk)),
        np.concatenate([s_ben, s_atk])
    )

    # Per-layer mean norms
    mean_ben = norms(X_tr_ben).mean(axis=0)
    mean_atk = norms(X_tr_atk).mean(axis=0)

    return {
        "auroc": float(auroc), "tpr5": tpr5, "tpr1": tpr1,
        "mean_ben_per_layer": mean_ben.tolist(),
        "mean_atk_per_layer": mean_atk.tolist(),
        "diff_per_layer": (mean_atk - mean_ben).tolist(),
    }


def load_llama_cache(path, layers):
    cache = np.load(path, allow_pickle=True)
    def to_arr(hs_list):
        return np.array([[hs[l][-1] for l in layers] for hs in hs_list])
    return {
        "X_tr_ben": to_arr(cache["hs_train_ben"].tolist()),
        "X_tr_atk": to_arr(cache["hs_train_atk"].tolist()),
        "X_te_ben": to_arr(cache["hs_test_ben"].tolist()),
        "X_te_atk": to_arr(cache["hs_test_atk"].tolist()),
    }


def load_vicuna_cache(path):
    cache = np.load(path, allow_pickle=True)
    X_be = cache["X_benign"]
    X_at = cache["X_attack"]
    rng = np.random.RandomState(42)
    bi = rng.permutation(len(X_be))
    ai = rng.permutation(len(X_at))
    n_tr_b = int(0.8 * len(X_be))
    n_tr_a = int(0.8 * len(X_at))
    return {
        "X_tr_ben": X_be[bi[:n_tr_b]],
        "X_tr_atk": X_at[ai[:n_tr_a]],
        "X_te_ben": X_be[bi[n_tr_b:]],
        "X_te_atk": X_at[ai[n_tr_a:]],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llama_cache", default="results/llama3_activations_cache_diverse.npz")
    p.add_argument("--vicuna_cache", default="results/vicuna_activations_cache_diverse.npz")
    p.add_argument("--output", default="results/norm_check_diverse.json")
    args = p.parse_args()

    print("\n" + "=" * 70)
    print(" NORM-ONLY CHECK ON DIVERSE CACHE")
    print("=" * 70)

    results = {}
    for name, loader, path, layers in [
        ("Llama-3", load_llama_cache, args.llama_cache, LLAMA_LAYERS),
        ("Vicuna",  load_vicuna_cache, args.vicuna_cache, None),
    ]:
        if not os.path.exists(path):
            print(f"\n  [{name}] cache not found at {path}, skipping")
            continue

        print(f"\n  [{name}] loading {path}...")
        if name == "Llama-3":
            data = loader(path, layers)
        else:
            data = loader(path)

        print(f"    train: {len(data['X_tr_ben'])} ben + "
              f"{len(data['X_tr_atk'])} atk")
        print(f"    test:  {len(data['X_te_ben'])} ben + "
              f"{len(data['X_te_atk'])} atk")

        r = evaluate_norm_only(**data)
        results[name] = r

        print(f"\n  [{name}] Per-layer mean norms:")
        print(f"    Layer | Benign | Attack |  Diff")
        for i, (b, a, d) in enumerate(zip(r["mean_ben_per_layer"],
                                            r["mean_atk_per_layer"],
                                            r["diff_per_layer"])):
            print(f"    {i:>5} | {b:>6.2f} | {a:>6.2f} | {d:>+6.2f}")

        print(f"\n  [{name}] Norm-only classifier:")
        print(f"    AUROC:    {r['auroc']:.4f}")
        print(f"    TPR @ 5%: {r['tpr5']:.4f}")
        print(f"    TPR @ 1%: {r['tpr1']:.4f}")

        if r["auroc"] >= 0.95:
            verdict = "STRONG norm confound"
        elif r["auroc"] >= 0.85:
            verdict = "MODERATE norm confound"
        elif r["auroc"] >= 0.70:
            verdict = "WEAK norm confound"
        else:
            verdict = "norm is not a primary signal"
        print(f"    -> {verdict}")

    # Compare to old (Alpaca-only) baseline
    print("\n" + "=" * 70)
    print(" COMPARISON")
    print("=" * 70)
    print("\n  Norm-only AUROC:")
    print(f"    Old cache (Alpaca benign):   0.9171")
    if "Llama-3" in results:
        print(f"    Diverse cache (Llama-3):     {results['Llama-3']['auroc']:.4f}")
        delta = results["Llama-3"]["auroc"] - 0.9171
        print(f"    Change: {delta:+.4f}")
        if abs(delta) < 0.05:
            print(f"    -> Norm confound is consistent across benign sets.")
            print(f"       The activation magnitudes really do differ between")
            print(f"       attack and benign, regardless of length.")
        elif delta < -0.10:
            print(f"    -> Norm confound substantially reduced.")
            print(f"       Diverse benign captures more of the magnitude variation.")
        else:
            print(f"    -> Norm confound surprisingly increased.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved -> {args.output}")


if __name__ == "__main__":
    main()
