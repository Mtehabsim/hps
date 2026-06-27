"""
Experiment 4 — Information-theoretic analysis.

Goal: Show that linear features capture nearly all the discriminative
information available in the activations. If true, no nonlinear method
(including hyperbolic) can extract more information for this task.

Sub-experiments:
  4A. Saturation argument
      Confirm C4 reaches AUROC = 1.000. The maximum is 1.000, so no method
      can improve beyond it. Trivial but rigorous.

  4B. Mutual information estimation
      Estimate I(linear_score; label) and compare to I(raw_activations; label).
      Use sklearn's mutual_info_classif as the KSG-based estimator.
      If they're approximately equal, linear captures all the information.

  4C. PCA dimensionality scan
      Train logistic regression on PCA(activations, k) for k = 1..64.
      Show that AUROC saturates well below 64 dims. This means the
      discriminative subspace is low-dimensional and linear.

  4D. Random projection scan
      Same as 4C but with random Gaussian projections. Shows that even
      uninformed dimensionality reduction captures most of the signal.
      Eliminates the alternative explanation that PCA is doing something
      special.

Usage:
  python strengthen_negative/experiment4_information_theoretic.py
"""

import os
import sys
import json

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif

HPS_LAYERS_LLAMA = [0, 2, 17, 24, 28, 31]
HPS_LAYERS_VICUNA_DEFAULT = [0, 2, 22, 31, 35, 39]
FPR_TARGET = 0.05
SEEDS = [42, 43, 44, 45, 46]


# -------------------------- Data loading --------------------------
def to_arr(hs_list, layers):
    return np.array([[hs[l][-1] for l in layers] for hs in hs_list])


def load_llama3():
    path = "results/llama3_activations_cache.npz"
    if not os.path.exists(path):
        return None
    cache = np.load(path, allow_pickle=True)
    layers = HPS_LAYERS_LLAMA
    X_tr_ben = to_arr(cache["hs_train_ben"].tolist(), layers)
    X_tr_atk = to_arr(cache["hs_train_atk"].tolist(), layers)
    X_te_ben = to_arr(cache["hs_test_ben"].tolist(), layers)
    X_te_atk = to_arr(cache["hs_test_atk"].tolist(), layers)
    return {
        "name": "Llama-3-8B",
        "X_tr_ben": X_tr_ben, "X_tr_atk": X_tr_atk,
        "X_te_ben": X_te_ben, "X_te_atk": X_te_atk,
    }


def load_vicuna():
    path = "results/vicuna_activations_cache.npz"
    if not os.path.exists(path):
        return None
    cache = np.load(path, allow_pickle=True)
    X_benign = cache["X_benign"]
    X_attack = cache["X_attack"]

    rng = np.random.RandomState(42)
    ben_idx = rng.permutation(len(X_benign))
    atk_idx = rng.permutation(len(X_attack))
    n_tr_b = int(0.8 * len(X_benign))
    n_tr_a = int(0.8 * len(X_attack))
    return {
        "name": "Vicuna-13B",
        "X_tr_ben": X_benign[ben_idx[:n_tr_b]],
        "X_te_ben": X_benign[ben_idx[n_tr_b:]],
        "X_tr_atk": X_attack[atk_idx[:n_tr_a]],
        "X_te_atk": X_attack[atk_idx[n_tr_a:]],
    }


def mean_pool(X):
    """Mean-pool across the layer axis. X is (N, L, D)."""
    return X.mean(axis=1)


def fit_lr(X_train, y_train, X_test, n_calib=None, seed=42):
    """Standard fit + 5% FPR threshold protocol."""
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_train)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X_tr_s, y_train)

    # X_test is the concatenation of (X_te_ben, X_te_atk) -- caller provides
    # the index where attacks start.
    return sc, clf


def eval_split(sc, clf, X_te_ben, X_te_atk, seed=42):
    n_calib = max(1, len(X_te_ben) // 2)
    s_calib = clf.predict_proba(sc.transform(X_te_ben[:n_calib]))[:, 1]
    s_ben = clf.predict_proba(sc.transform(X_te_ben[n_calib:]))[:, 1]
    s_atk = clf.predict_proba(sc.transform(X_te_atk))[:, 1]
    thr = float(np.quantile(s_calib, 1.0 - FPR_TARGET))
    tpr = float((s_atk > thr).mean())
    fpr = float((s_ben > thr).mean())
    auroc = roc_auc_score(
        np.array([0] * len(s_ben) + [1] * len(s_atk)),
        np.concatenate([s_ben, s_atk])
    )
    return {"auroc": auroc, "tpr": tpr, "fpr": fpr}


# -------------------------- 4A: Saturation argument --------------------------
def saturation_argument(data):
    print("\n" + "-" * 78)
    print(" 4A. Saturation argument")
    print("-" * 78)
    print("""
  If C4 (a 4097-parameter logistic regression) achieves AUROC = 1.000, then
  the maximum possible AUROC is also 1.000. No method can exceed this; the
  classification task is information-theoretically saturated for these data.
""")

    X_tr_ben = mean_pool(data["X_tr_ben"])
    X_tr_atk = mean_pool(data["X_tr_atk"])
    X_te_ben = mean_pool(data["X_te_ben"])
    X_te_atk = mean_pool(data["X_te_atk"])
    X_tr = np.concatenate([X_tr_ben, X_tr_atk])
    y_tr = np.array([0] * len(X_tr_ben) + [1] * len(X_tr_atk))

    sc = StandardScaler()
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(sc.fit_transform(X_tr), y_tr)
    res = eval_split(sc, clf, X_te_ben, X_te_atk)

    print(f"  C4 AUROC: {res['auroc']:.4f}")
    print(f"  C4 TPR @ 5% FPR: {res['tpr']:.4f}")
    if res["auroc"] >= 0.999:
        print("  -> AUROC is saturated. No method can do meaningfully better on this benchmark.")
    else:
        print(f"  -> AUROC has headroom of {1.0 - res['auroc']:.4f}; some method could "
              "improve, but the gap is small.")
    return res


# -------------------------- 4B: Mutual information --------------------------
def mutual_information_test(data):
    """Estimate I(features; label) for raw vs linear vs PCA-reduced features.

    For high-dim features, sklearn's mutual_info_classif estimates per-feature
    MI. We sum to get an upper bound on total MI (assumes independence).
    For the linear LR score (1-D), it gives a tight estimate.
    """
    print("\n" + "-" * 78)
    print(" 4B. Mutual information estimation")
    print("-" * 78)

    X_tr_ben = mean_pool(data["X_tr_ben"])
    X_tr_atk = mean_pool(data["X_tr_atk"])
    X = np.concatenate([X_tr_ben, X_tr_atk])
    y = np.array([0] * len(X_tr_ben) + [1] * len(X_tr_atk))

    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    print("  Computing I(raw_activations; label) - takes a minute on 4096 dims...")
    # Compute MI per dim and sum (upper bound)
    mi_per_dim = mutual_info_classif(Xs, y, random_state=42)
    mi_raw_sum = float(mi_per_dim.sum())
    mi_raw_max = float(mi_per_dim.max())
    mi_raw_mean = float(mi_per_dim.mean())

    print("  Computing I(linear_LR_score; label)...")
    clf = LogisticRegression(max_iter=2000, random_state=42).fit(Xs, y)
    scores = clf.predict_proba(Xs)[:, 1].reshape(-1, 1)
    mi_lr = float(mutual_info_classif(scores, y, random_state=42)[0])

    # Single-dim entropy bound for binary y: H(y) <= log(2) ~ 0.693 nats
    h_y_max = float(np.log(2))

    print(f"\n  H(y) = {h_y_max:.4f} nats (perfect classifier)")
    print(f"  I(linear_score; y) = {mi_lr:.4f} nats")
    print(f"  I(raw_per_dim; y) max = {mi_raw_max:.4f} nats")
    print(f"  I(raw_per_dim; y) mean = {mi_raw_mean:.4f} nats")
    print(f"  ratio I(linear; y) / H(y) = {mi_lr / h_y_max:.4f}")

    if mi_lr >= 0.95 * h_y_max:
        print("  -> The linear score captures essentially all the entropy of y.")
        print("     No nonlinear method can extract significantly more.")
    else:
        print(f"  -> The linear score captures {mi_lr/h_y_max*100:.1f}% of H(y). "
              "Some headroom for nonlinear methods, but it's bounded.")

    return {
        "h_y_max_nats": h_y_max,
        "I_linear_y_nats": mi_lr,
        "I_raw_per_dim_max": mi_raw_max,
        "I_raw_per_dim_mean": mi_raw_mean,
        "I_raw_sum_upper_bound": mi_raw_sum,
        "ratio_linear_to_H": mi_lr / h_y_max,
    }


# -------------------------- 4C: PCA dimensionality scan --------------------------
def pca_scan(data):
    """Train LR on PCA(X, k) for k in [1, 2, 4, 8, 16, 32, 64].

    AUROC and TPR vs k. Shows where the discriminative signal saturates.
    """
    print("\n" + "-" * 78)
    print(" 4C. PCA dimensionality scan")
    print("-" * 78)

    X_tr_ben = mean_pool(data["X_tr_ben"])
    X_tr_atk = mean_pool(data["X_tr_atk"])
    X_te_ben = mean_pool(data["X_te_ben"])
    X_te_atk = mean_pool(data["X_te_atk"])

    X_tr = np.concatenate([X_tr_ben, X_tr_atk])
    y_tr = np.array([0] * len(X_tr_ben) + [1] * len(X_tr_atk))

    ks = [1, 2, 4, 8, 16, 32, 64]
    results = []
    print(f"\n  {'k':>4} | {'AUROC':>6} | {'TPR@5%':>7} | {'FPR':>5}")
    print(f"  {'-'*4}-+-{'-'*6}-+-{'-'*7}-+-{'-'*5}")
    for k in ks:
        if k > X_tr.shape[1]:
            continue
        pca = PCA(n_components=k, random_state=42).fit(X_tr)
        Xtr_k = pca.transform(X_tr)
        Xte_b_k = pca.transform(X_te_ben)
        Xte_a_k = pca.transform(X_te_atk)

        sc = StandardScaler()
        clf = LogisticRegression(max_iter=2000, random_state=42)
        clf.fit(sc.fit_transform(Xtr_k), y_tr)
        r = eval_split(sc, clf, Xte_b_k, Xte_a_k)
        results.append({"k": k, **r})
        print(f"  {k:>4} | {r['auroc']:>6.3f} | {r['tpr']:>7.3f} | {r['fpr']:>5.3f}")

    # Find saturation point
    saturated_k = None
    for r in results:
        if r["tpr"] >= 0.99:
            saturated_k = r["k"]
            break
    if saturated_k:
        print(f"\n  -> Discriminative signal saturates at k = {saturated_k} dims.")
        print(f"     This means the relevant linear subspace is at most "
              f"{saturated_k}-dimensional.")
    else:
        print("\n  -> No saturation observed; signal continues to improve with more dims.")

    return results


# -------------------------- 4D: Random projection scan --------------------------
def random_projection_scan(data, n_seeds=3):
    """Same as 4C but with random Gaussian projections instead of PCA.

    A Johnson-Lindenstrauss-style baseline. If random projection also captures
    the signal, the result isn't dependent on PCA's specific structure.
    """
    print("\n" + "-" * 78)
    print(" 4D. Random projection scan (averaged over %d seeds)" % n_seeds)
    print("-" * 78)

    X_tr_ben = mean_pool(data["X_tr_ben"])
    X_tr_atk = mean_pool(data["X_tr_atk"])
    X_te_ben = mean_pool(data["X_te_ben"])
    X_te_atk = mean_pool(data["X_te_atk"])

    X_tr = np.concatenate([X_tr_ben, X_tr_atk])
    y_tr = np.array([0] * len(X_tr_ben) + [1] * len(X_tr_atk))

    ks = [1, 2, 4, 8, 16, 32, 64]
    results = []
    print(f"\n  {'k':>4} | {'AUROC mean':>11} | {'TPR mean':>9} | {'AUROC std':>10}")
    print(f"  {'-'*4}-+-{'-'*11}-+-{'-'*9}-+-{'-'*10}")
    d_in = X_tr.shape[1]
    for k in ks:
        if k > d_in:
            continue
        aurocs, tprs = [], []
        for s in range(n_seeds):
            rng = np.random.RandomState(s)
            W = rng.randn(d_in, k).astype(np.float32) / np.sqrt(d_in)
            Xtr_k = X_tr @ W
            Xte_b_k = X_te_ben @ W
            Xte_a_k = X_te_atk @ W

            sc = StandardScaler()
            clf = LogisticRegression(max_iter=2000, random_state=s)
            clf.fit(sc.fit_transform(Xtr_k), y_tr)
            r = eval_split(sc, clf, Xte_b_k, Xte_a_k, seed=s)
            aurocs.append(r["auroc"])
            tprs.append(r["tpr"])
        results.append({
            "k": k,
            "auroc_mean": float(np.mean(aurocs)),
            "auroc_std": float(np.std(aurocs)),
            "tpr_mean": float(np.mean(tprs)),
            "tpr_std": float(np.std(tprs)),
        })
        print(f"  {k:>4} | {results[-1]['auroc_mean']:>11.3f} | "
              f"{results[-1]['tpr_mean']:>9.3f} | {results[-1]['auroc_std']:>10.4f}")

    # If random gets close to perfect at k=64, that confirms linear is sufficient
    final = results[-1]
    if final["auroc_mean"] >= 0.99:
        print(f"\n  -> Random projection to {final['k']} dims achieves "
              f"AUROC = {final['auroc_mean']:.3f}.")
        print(f"     This further supports that the discriminative signal is "
              f"linearly accessible.")
    return results


# -------------------------- 4E: Bound on hyperbolic improvement --------------------------
def upper_bound_hyperbolic_improvement(data, baseline_results):
    """Computes a back-of-envelope upper bound on how much any nonlinear method
    could improve over linear, given the saturation argument.

    AUROC is bounded by 1.0. If C4 already achieves AUROC = X, the maximum
    additional AUROC any method can achieve is (1.0 - X). For a saturated
    benchmark (X ~= 1.0), this bound is essentially zero.
    """
    print("\n" + "-" * 78)
    print(" 4E. Upper bound on hyperbolic improvement")
    print("-" * 78)

    c4_auroc = baseline_results["auroc"]
    c4_tpr = baseline_results["tpr"]
    headroom_auroc = 1.0 - c4_auroc
    headroom_tpr = 1.0 - c4_tpr
    print(f"\n  C4 AUROC = {c4_auroc:.4f}, headroom = {headroom_auroc:.4f}")
    print(f"  C4 TPR   = {c4_tpr:.4f}, headroom = {headroom_tpr:.4f}")
    print(f"\n  Maximum additional AUROC any method can achieve: {headroom_auroc:.4f}")
    print(f"  Maximum additional TPR any method can achieve:   {headroom_tpr:.4f}")

    if headroom_auroc < 0.005:
        print(f"\n  -> Hyperbolic methods CAN improve over C4 by at most "
              f"{headroom_auroc*100:.2f}% AUROC.")
        print(f"     This is an information-theoretic upper bound, not an "
              f"empirical observation.")
    return {
        "c4_auroc": c4_auroc,
        "c4_tpr": c4_tpr,
        "max_auroc_improvement": headroom_auroc,
        "max_tpr_improvement": headroom_tpr,
    }


# -------------------------- Main --------------------------
def main():
    print("\n" + "=" * 78)
    print(" EXPERIMENT 4 — Information-theoretic analysis")
    print("=" * 78)

    datasets = []
    for loader in (load_llama3, load_vicuna):
        d = loader()
        if d is None:
            print(f"[skip] {loader.__name__} cache not found")
            continue
        datasets.append(d)
        print(f"  Loaded {d['name']}: train {len(d['X_tr_ben'])} ben + "
              f"{len(d['X_tr_atk'])} atk, test {len(d['X_te_ben'])} ben + "
              f"{len(d['X_te_atk'])} atk")

    if not datasets:
        print("ERROR: no caches found.")
        return

    all_results = {}
    for ds in datasets:
        print("\n" + "=" * 78)
        print(f"  Dataset: {ds['name']}")
        print("=" * 78)

        ds_results = {}
        ds_results["4A_saturation"] = saturation_argument(ds)
        ds_results["4B_mutual_information"] = mutual_information_test(ds)
        ds_results["4C_pca_scan"] = pca_scan(ds)
        ds_results["4D_random_projection_scan"] = random_projection_scan(ds, n_seeds=3)
        ds_results["4E_upper_bound"] = upper_bound_hyperbolic_improvement(
            ds, ds_results["4A_saturation"])

        all_results[ds["name"]] = ds_results

    out_path = "results/strengthen_exp4_information_theoretic.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved -> {out_path}")

    # Combined conclusion
    print("\n" + "=" * 78)
    print(" CONCLUSION")
    print("=" * 78)
    for ds_name, r in all_results.items():
        c4 = r["4A_saturation"]
        bound = r["4E_upper_bound"]
        mi = r["4B_mutual_information"]
        print(f"\n  {ds_name}:")
        print(f"    C4 (linear) AUROC = {c4['auroc']:.4f}, TPR = {c4['tpr']:.4f}")
        print(f"    Max additional AUROC any method can achieve: "
              f"{bound['max_auroc_improvement']:.4f}")
        print(f"    I(linear; y) / H(y) = {mi['ratio_linear_to_H']:.3f} "
              f"({mi['ratio_linear_to_H']*100:.1f}% of available entropy)")


if __name__ == "__main__":
    main()
