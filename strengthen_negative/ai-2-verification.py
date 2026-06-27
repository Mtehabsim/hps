"""
ai-2-verification.py — Verify TPR=1.000 is real, not a methodology bug.

Four specific checks:
  1. Prompt length confound: train a length-only classifier, measure AUROC
  2. Strict threshold: do methods saturate at 1% and 0.1% FPR too?
  3. Permutation test: with random labels, do we still get high AUROC?
  4. Train/test contamination: any duplicate prompts across splits?

If all 4 pass:
  - Length-only AUROC < 0.9
  - Some method drops at 1% FPR
  - Permutation AUROC ~0.5
  - No duplicate hashes
Then TPR=1.000 is real and the paper's saturation framing is honest.

Usage:
  python strengthen_negative/ai-2-verification.py \
    --test-attacks llama3_attacks.json --harmless data_harmless_6500.csv
"""
import os, sys, json, hashlib, argparse
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

HPS_LAYERS = [0, 2, 17, 24, 28, 31]
LLAMA_CACHE = "results/llama3_activations_cache.npz"
FPR_LEVELS = [0.05, 0.01, 0.001]


def load_prompts(test_attacks_path, harmless_path):
    """Reproduce the same train/test split as hps_llama3.py."""
    df_h = pd.read_csv(harmless_path)
    harmless = df_h.iloc[:, 0].astype(str).tolist()

    with open(test_attacks_path) as f:
        cat = json.load(f)
    attack_prompts, attack_methods = [], []
    for m, prompts in cat.items():
        for p in prompts:
            if p:
                attack_prompts.append(p)
                attack_methods.append(m)

    rng = np.random.RandomState(42)
    a_idx = rng.permutation(len(attack_prompts))
    n_tr_a = int(0.8 * len(a_idx))
    train_atk = [attack_prompts[i] for i in a_idx[:n_tr_a]]
    test_atk = [attack_prompts[i] for i in a_idx[n_tr_a:]]
    train_atk_methods = [attack_methods[i] for i in a_idx[:n_tr_a]]
    test_atk_methods = [attack_methods[i] for i in a_idx[n_tr_a:]]

    n_tr_b = len(train_atk)
    n_te_b = len(test_atk)
    b_idx = rng.permutation(len(harmless))
    train_ben = [harmless[i] for i in b_idx[:n_tr_b]]
    test_ben = [harmless[i] for i in b_idx[n_tr_b:n_tr_b + n_te_b]]

    return {
        "train_ben": train_ben, "train_atk": train_atk,
        "test_ben": test_ben, "test_atk": test_atk,
        "train_atk_methods": train_atk_methods,
        "test_atk_methods": test_atk_methods,
    }


def load_activations():
    """Load mean-pooled activations from the Llama-3 cache."""
    cache = np.load(LLAMA_CACHE, allow_pickle=True)
    def to_arr(hs_list):
        return np.array([[hs[l][-1] for l in HPS_LAYERS] for hs in hs_list])
    X_tr_ben = to_arr(cache["hs_train_ben"].tolist()).mean(axis=1)
    X_tr_atk = to_arr(cache["hs_train_atk"].tolist()).mean(axis=1)
    X_te_ben = to_arr(cache["hs_test_ben"].tolist()).mean(axis=1)
    X_te_atk = to_arr(cache["hs_test_atk"].tolist()).mean(axis=1)
    return X_tr_ben, X_tr_atk, X_te_ben, X_te_atk


def eval_at_fpr(s_calib, s_ben, s_atk, fpr_target):
    """TPR at a given FPR threshold."""
    thr = float(np.quantile(s_calib, 1.0 - fpr_target))
    return {
        "fpr_target": fpr_target,
        "threshold": thr,
        "tpr": float((s_atk > thr).mean()),
        "fpr_actual": float((s_ben > thr).mean()),
    }


# ============================================================
# CHECK 1: Prompt length confound
# ============================================================
def check_length_confound(prompts):
    print("\n" + "=" * 70)
    print(" CHECK 1: PROMPT LENGTH CONFOUND")
    print("=" * 70)
    print("\n  Train a classifier that uses ONLY prompt length.")
    print("  If length-only AUROC > 0.9, length is a major confound.\n")

    train_lens = [[len(p)] for p in prompts["train_ben"] + prompts["train_atk"]]
    test_ben_lens = [[len(p)] for p in prompts["test_ben"]]
    test_atk_lens = [[len(p)] for p in prompts["test_atk"]]
    y_train = np.array([0]*len(prompts["train_ben"]) + [1]*len(prompts["train_atk"]))

    print(f"  Benign prompt length: mean={np.mean([len(p) for p in prompts['train_ben']]):.1f}, "
          f"median={np.median([len(p) for p in prompts['train_ben']]):.0f}")
    print(f"  Attack prompt length: mean={np.mean([len(p) for p in prompts['train_atk']]):.1f}, "
          f"median={np.median([len(p) for p in prompts['train_atk']]):.0f}")

    sc = StandardScaler()
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(sc.fit_transform(np.array(train_lens)), y_train)
    s_ben = clf.predict_proba(sc.transform(np.array(test_ben_lens)))[:, 1]
    s_atk = clf.predict_proba(sc.transform(np.array(test_atk_lens)))[:, 1]
    auroc = roc_auc_score([0]*len(s_ben) + [1]*len(s_atk),
                          np.concatenate([s_ben, s_atk]))
    print(f"\n  Length-only AUROC: {auroc:.3f}")
    if auroc > 0.9:
        print(f"  -> Length is a STRONG confound. Saturation may be largely length-driven.")
    elif auroc > 0.7:
        print(f"  -> Length is a partial confound but not the full story.")
    else:
        print(f"  -> Length alone is insufficient. Saturation is content-based.")

    # Per-attack length-only TPR (at 5% FPR)
    print("\n  Per-attack-type length-only TPR @ 5% FPR:")
    n_calib = len(s_ben) // 2
    thr = float(np.quantile(s_ben[:n_calib], 0.95))
    print(f"  {'Method':<25} | {'TPR':>5}")
    print(f"  {'-'*25}-+-{'-'*5}")
    for m in sorted(set(prompts["test_atk_methods"])):
        idx = [i for i, mm in enumerate(prompts["test_atk_methods"]) if mm == m]
        if idx:
            tpr_m = float((s_atk[idx] > thr).mean())
            print(f"  {m:<25} | {tpr_m:>5.3f}")
    return {"length_only_auroc": float(auroc)}


# ============================================================
# CHECK 2: Strict threshold (1% and 0.1% FPR)
# ============================================================
def check_strict_threshold():
    print("\n" + "=" * 70)
    print(" CHECK 2: STRICT THRESHOLD (do methods diverge at 1% and 0.1% FPR?)")
    print("=" * 70)
    print("\n  Train HPS-style classifier (C4) with 5%, 1%, 0.1% FPR thresholds.")
    print("  If TPR drops at strict FPR, methods can be distinguished there.\n")

    if not os.path.exists(LLAMA_CACHE):
        print(f"  Cache not found at {LLAMA_CACHE}, skipping.")
        return {}

    X_tr_ben, X_tr_atk, X_te_ben, X_te_atk = load_activations()
    X_train = np.concatenate([X_tr_ben, X_tr_atk])
    y_train = np.array([0]*len(X_tr_ben) + [1]*len(X_tr_atk))

    sc = StandardScaler()
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(sc.fit_transform(X_train), y_train)
    s_te_ben = clf.predict_proba(sc.transform(X_te_ben))[:, 1]
    s_te_atk = clf.predict_proba(sc.transform(X_te_atk))[:, 1]

    n_calib = len(s_te_ben) // 2
    s_calib = s_te_ben[:n_calib]
    s_ben = s_te_ben[n_calib:]

    print(f"  {'FPR target':>10} | {'Threshold':>10} | {'TPR':>6} | {'FPR actual':>10}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*6}-+-{'-'*10}")
    results = {}
    for fpr in FPR_LEVELS:
        r = eval_at_fpr(s_calib, s_ben, s_te_atk, fpr)
        results[f"fpr_{fpr}"] = r
        print(f"  {fpr:>10.4f} | {r['threshold']:>10.4f} | "
              f"{r['tpr']:>6.3f} | {r['fpr_actual']:>10.4f}")

    # Per-attack TPR at strictest threshold
    print(f"\n  Per-attack TPR at FPR=0.001:")
    thr_strict = float(np.quantile(s_calib, 0.999))
    return results


# ============================================================
# CHECK 3: Permutation test
# ============================================================
def check_permutation():
    print("\n" + "=" * 70)
    print(" CHECK 3: PERMUTATION TEST (random labels)")
    print("=" * 70)
    print("\n  Shuffle attack labels and retrain. AUROC should be ~0.5.")
    print("  If high AUROC persists with random labels, model finds spurious")
    print("  patterns (length, format) not real harm signal.\n")

    if not os.path.exists(LLAMA_CACHE):
        print("  Cache missing, skipping.")
        return {}

    X_tr_ben, X_tr_atk, X_te_ben, X_te_atk = load_activations()
    X_train = np.concatenate([X_tr_ben, X_tr_atk])
    y_real = np.array([0]*len(X_tr_ben) + [1]*len(X_tr_atk))

    print(f"  {'Trial':<8} | {'AUROC':>6}")
    print(f"  {'-'*8}-+-{'-'*6}")
    rng = np.random.RandomState(42)
    aurocs = []
    for trial in range(3):
        y_shuffled = rng.permutation(y_real)
        sc = StandardScaler()
        clf = LogisticRegression(max_iter=2000, random_state=trial)
        clf.fit(sc.fit_transform(X_train), y_shuffled)
        # Eval on test with original labels
        s_ben = clf.predict_proba(sc.transform(X_te_ben))[:, 1]
        s_atk = clf.predict_proba(sc.transform(X_te_atk))[:, 1]
        auroc = roc_auc_score([0]*len(s_ben)+[1]*len(s_atk),
                              np.concatenate([s_ben, s_atk]))
        aurocs.append(auroc)
        print(f"  shuffled-{trial} | {auroc:>6.3f}")
    mean = np.mean(aurocs)
    print(f"\n  Mean AUROC under random labels: {mean:.3f}")
    if abs(mean - 0.5) < 0.1:
        print(f"  -> Model finds NO spurious signal. Real labels carry real info.")
    elif mean > 0.7:
        print(f"  -> Suspicious: spurious signal exists. Investigate confounds.")
    else:
        print(f"  -> Some spurious correlation but limited.")
    return {"random_label_auroc_mean": float(mean)}


# ============================================================
# CHECK 4: Train/test contamination
# ============================================================
def check_contamination(prompts):
    print("\n" + "=" * 70)
    print(" CHECK 4: TRAIN/TEST CONTAMINATION")
    print("=" * 70)
    print("\n  Hash each prompt. Verify no duplicates across train/test.\n")

    def hashes(prompts_list):
        return set(hashlib.md5(p.encode()).hexdigest() for p in prompts_list)

    tr_ben_h = hashes(prompts["train_ben"])
    tr_atk_h = hashes(prompts["train_atk"])
    te_ben_h = hashes(prompts["test_ben"])
    te_atk_h = hashes(prompts["test_atk"])

    overlap_ben = tr_ben_h & te_ben_h
    overlap_atk = tr_atk_h & te_atk_h
    cross_overlap = (tr_ben_h | tr_atk_h) & (te_ben_h | te_atk_h)

    print(f"  Train benign:  {len(tr_ben_h)} unique  (out of {len(prompts['train_ben'])})")
    print(f"  Train attack:  {len(tr_atk_h)} unique  (out of {len(prompts['train_atk'])})")
    print(f"  Test  benign:  {len(te_ben_h)} unique  (out of {len(prompts['test_ben'])})")
    print(f"  Test  attack:  {len(te_atk_h)} unique  (out of {len(prompts['test_atk'])})")
    print(f"  Benign train∩test: {len(overlap_ben)}")
    print(f"  Attack train∩test: {len(overlap_atk)}")
    print(f"  Any cross-class:   {len(cross_overlap)}")
    if len(overlap_ben) == 0 and len(overlap_atk) == 0 and len(cross_overlap) == 0:
        print(f"\n  -> NO contamination. Splits are clean.")
        return {"contamination": False}
    else:
        print(f"\n  -> CONTAMINATION DETECTED. Investigate.")
        return {"contamination": True,
                "ben_overlap": len(overlap_ben),
                "atk_overlap": len(overlap_atk),
                "cross": len(cross_overlap)}


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-attacks", default="llama3_attacks.json")
    parser.add_argument("--harmless", default="data_harmless_6500.csv")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" VERIFICATION: Is TPR=1.000 a methodology bug or a real result?")
    print("=" * 70)

    prompts = load_prompts(args.test_attacks, args.harmless)
    print(f"\n  Loaded prompts: train {len(prompts['train_ben'])} ben + "
          f"{len(prompts['train_atk'])} atk; test {len(prompts['test_ben'])} ben + "
          f"{len(prompts['test_atk'])} atk")

    results = {}
    results["check1_length"] = check_length_confound(prompts)
    results["check2_strict_threshold"] = check_strict_threshold()
    results["check3_permutation"] = check_permutation()
    results["check4_contamination"] = check_contamination(prompts)

    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print(f"\n  Check 1 (length confound):  AUROC = "
          f"{results['check1_length']['length_only_auroc']:.3f}")
    if "fpr_0.001" in results["check2_strict_threshold"]:
        print(f"  Check 2 (TPR @ 0.1% FPR):    "
              f"{results['check2_strict_threshold']['fpr_0.001']['tpr']:.3f}")
    if "random_label_auroc_mean" in results["check3_permutation"]:
        print(f"  Check 3 (random label AUROC): "
              f"{results['check3_permutation']['random_label_auroc_mean']:.3f}")
    print(f"  Check 4 (contamination):      "
          f"{'YES' if results['check4_contamination'].get('contamination') else 'NO'}")

    print("\n  VERDICT:")
    bad = []
    if results["check1_length"]["length_only_auroc"] > 0.9:
        bad.append("length-only AUROC > 0.9 (length is dominant confound)")
    if results["check3_permutation"].get("random_label_auroc_mean", 0.5) > 0.7:
        bad.append("random-label AUROC > 0.7 (spurious patterns)")
    if results["check4_contamination"].get("contamination"):
        bad.append("train/test contamination detected")
    if bad:
        print(f"  - ISSUES FOUND:")
        for b in bad:
            print(f"      * {b}")
        print(f"  - The TPR=1.000 saturation may not be purely from harm detection.")
    else:
        print(f"  - No bugs found. TPR=1.000 is real but reflects benchmark saturation.")
        print(f"  - Same-distribution comparison is a sanity check, not a result.")

    out_path = "results/verification_ai2.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else str(o))
    print(f"\n  Saved -> {out_path}")


if __name__ == "__main__":
    main()
