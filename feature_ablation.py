"""
Feature ablation — determine which of the 12 trajectory features matter.

Tests subsets of the 12 features across multiple regimes:
  1. Same-distribution full data (where same-dist permutation importance is measured)
  2. Cold-start cross-attack (N per method = 5, 25, 100)
  3. Vicuna-like regime (4 methods × 25 attacks)

Subsets tested (always with new HPS config: spread layers, κ=0.1 frozen, 50 epochs):
  - all_12              — baseline (radial 5 + curvature 4 + displacement 3)
  - radial_5            — radial features only
  - curvature_4         — curvature features only
  - displacement_3      — displacement features only
  - radial_disp_8       — radial + displacement (drop curvature)
  - top6_by_importance  — top 6 by permutation importance on same-dist
  - top4_by_importance  — top 4
  - top1_by_importance  — single best feature

Decision rule:
  - If top6 ≥ all_12 across ALL regimes (cold-start included): drop to 6
  - If top6 < all_12 in any regime by ≥ 0.05 TPR: keep all 12
  - In between: report and let author decide

Usage:
  python feature_ablation.py \
    --test-attacks llama3_attacks.json \
    --harmless data_harmless_6500.csv

Runtime: ~20-30 minutes on cached activations.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from experiment7 import LorentzProjection, contrastive_loss, extract_trajectory_features
from rtv_standalone import FPR_TARGET

# ── Config (matches new defaults) ──
HPS_LAYERS = [0, 2, 17, 24, 28, 31]
KAPPA_INIT = 0.1
FREEZE_KAPPA = True
EPOCHS = 50
device = "cuda" if torch.cuda.is_available() else "cpu"

FEAT_NAMES = ["mean_r", "max_r", "min_r", "std_r", "range_r",
              "max_κ", "mean_κ", "std_κ", "spike_loc",
              "displacement", "path_len", "progress"]
RADIAL_IDX = [0, 1, 2, 3, 4]
CURVATURE_IDX = [5, 6, 7, 8]
DISPLACEMENT_IDX = [9, 10, 11]


def train_hps_proj(X_train, y_train, seed=42):
    """Train Lorentz projection with new config."""
    n_layers = X_train.shape[1]
    d_hidden = X_train.shape[2]
    torch.manual_seed(seed)
    np.random.seed(seed)
    proj = LorentzProjection(d_hidden, 64, KAPPA_INIT, n_layers=n_layers).to(device)
    if FREEZE_KAPPA:
        proj.log_k.requires_grad = False
    opt = optim.Adam([p for p in proj.parameters() if p.requires_grad],
                     lr=1e-3, weight_decay=1e-5)
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)
    for _ in range(EPOCHS):
        loss = torch.tensor(0.0, device=device)
        for l in range(n_layers):
            h = proj(X_t[:, l, :])
            loss = loss + contrastive_loss(h, y_t, k=proj.k, tau=proj.tau(l))
        loss = loss / n_layers
        opt.zero_grad(); loss.backward(); opt.step()
    proj.eval()
    return proj


def evaluate_subset(feats_train, y_train, feats_te_ben, feats_te_atk, idx, seed=42):
    """Evaluate classifier using only the specified feature indices.
    Returns (auroc, tpr, fpr_actual)."""
    f_tr = feats_train[:, idx]
    f_be = feats_te_ben[:, idx]
    f_at = feats_te_atk[:, idx]
    sc = StandardScaler()
    f_tr_s = sc.fit_transform(f_tr)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(f_tr_s, y_train)
    n_calib = len(f_be) // 2
    if n_calib < 5:
        s_calib = clf.predict_proba(sc.transform(f_be))[:, 1]
        s_ben = s_calib
    else:
        s_calib = clf.predict_proba(sc.transform(f_be[:n_calib]))[:, 1]
        s_ben = clf.predict_proba(sc.transform(f_be[n_calib:]))[:, 1]
    s_atk = clf.predict_proba(sc.transform(f_at))[:, 1]
    thr = float(np.quantile(s_calib, 1.0 - FPR_TARGET))
    tpr = float((s_atk > thr).mean())
    fpr = float((s_ben > thr).mean())
    auroc = roc_auc_score(np.array([0]*len(s_ben) + [1]*len(s_atk)),
                          np.concatenate([s_ben, s_atk]))
    return auroc, tpr, fpr


def to_arr(hs_list, layers):
    return np.array([[hs[l][-1] for l in layers] for hs in hs_list])


def get_top_n_by_importance(feats_train, y_train, n_top, n_repeats=5):
    """Determine top-n features by permutation importance on training data."""
    sc = StandardScaler()
    f_s = sc.fit_transform(feats_train)
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(f_s, y_train)
    perm = permutation_importance(clf, f_s, y_train, n_repeats=n_repeats,
                                   random_state=42, scoring="roc_auc")
    order = np.argsort(perm.importances_mean)[::-1]
    return list(order[:n_top]), perm.importances_mean


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-attacks", required=True)
    parser.add_argument("--harmless", required=True)
    parser.add_argument("--cache", default="results/llama3_activations_cache.npz")
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"  FEATURE ABLATION — Which trajectory features matter?")
    print(f"  Layers:  {HPS_LAYERS}")
    print(f"  κ:       {KAPPA_INIT} ({'frozen' if FREEZE_KAPPA else 'learnable'})")
    print(f"  Epochs:  {EPOCHS}")
    print(f"{'═'*60}\n")

    if not os.path.exists(args.cache):
        print(f"ERROR: Cache not found at {args.cache}")
        return

    cache = np.load(args.cache, allow_pickle=True)
    hs_train_ben = cache["hs_train_ben"].tolist()
    hs_train_atk = cache["hs_train_atk"].tolist()
    hs_test_ben = cache["hs_test_ben"].tolist()
    hs_test_atk = cache["hs_test_atk"].tolist()

    X_tr_ben = to_arr(hs_train_ben, HPS_LAYERS)
    X_tr_atk = to_arr(hs_train_atk, HPS_LAYERS)
    X_te_ben = to_arr(hs_test_ben, HPS_LAYERS)
    X_te_atk = to_arr(hs_test_atk, HPS_LAYERS)

    with open(args.test_attacks) as f:
        categorized = json.load(f)
    attack_methods = []
    for method, prompts in categorized.items():
        for p in prompts:
            if p:
                attack_methods.append(method)
    rng = np.random.RandomState(42)
    atk_idx = rng.permutation(len(attack_methods))
    all_methods = [attack_methods[i] for i in atk_idx]
    methods_unique = sorted(set(attack_methods))

    X_all_atk = np.concatenate([X_tr_atk, X_te_atk])
    X_all_ben = np.concatenate([X_tr_ben, X_te_ben])
    hs_by_method = {m: [] for m in methods_unique}
    for act, method in zip(X_all_atk, all_methods):
        hs_by_method[method].append(act)
    for m in methods_unique:
        hs_by_method[m] = np.array(hs_by_method[m])

    # ══════════════════════════════════════════════════════════════
    #  PART 1: Train projection on full data, compute features once
    # ══════════════════════════════════════════════════════════════
    print(f"  Training HPS projection on full data (5216 attacks)...")
    X_train = np.concatenate([X_tr_ben, X_tr_atk])
    y_train = np.array([0]*len(X_tr_ben) + [1]*len(X_tr_atk))
    proj_full = train_hps_proj(X_train, y_train, seed=42)
    feats_train_full = extract_trajectory_features(proj_full, X_train)
    feats_te_ben_full = extract_trajectory_features(proj_full, X_te_ben)
    feats_te_atk_full = extract_trajectory_features(proj_full, X_te_atk)

    # Determine top features by importance (on same-dist data)
    print(f"  Computing same-distribution permutation importance (n_repeats=5)...")
    top6, importances = get_top_n_by_importance(feats_train_full, y_train, 6)
    top4 = top6[:4]
    top1 = top6[:1]

    print(f"\n  Importance ranking (same-distribution):")
    sorted_by_imp = sorted(enumerate(importances), key=lambda x: -x[1])
    for rank, (idx, val) in enumerate(sorted_by_imp, 1):
        flag = " ← top6" if idx in top6 else ""
        cat = "radial" if idx < 5 else ("curvature" if idx < 9 else "displacement")
        print(f"    {rank:>2}. {FEAT_NAMES[idx]:<12} ({cat:<13})  importance={val:>7.4f}{flag}")

    # Define feature subsets
    SUBSETS = {
        "all_12":            list(range(12)),
        "radial_5":          RADIAL_IDX,
        "curvature_4":       CURVATURE_IDX,
        "displacement_3":    DISPLACEMENT_IDX,
        "radial_disp_8":     RADIAL_IDX + DISPLACEMENT_IDX,
        "top6_byimp":        top6,
        "top4_byimp":        top4,
        "top1_byimp":        top1,
    }

    # ══════════════════════════════════════════════════════════════
    #  PART 2: Same-distribution evaluation across subsets
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  PART A: Same-distribution full data (5216 attacks, 9 methods)")
    print(f"{'─'*60}\n")
    print(f"  {'Subset':<18} | {'#feats':>6} | {'AUROC':>6} | {'TPR@5%':>7} | {'FPR':>5}")
    print(f"  {'─'*18}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*7}─┼─{'─'*5}")
    same_dist_results = {}
    for name, idx in SUBSETS.items():
        a, t, f = evaluate_subset(feats_train_full, y_train, feats_te_ben_full,
                                   feats_te_atk_full, idx)
        same_dist_results[name] = {"auroc": a, "tpr": t, "fpr": f, "n_features": len(idx)}
        print(f"  {name:<18} | {len(idx):>6} | {a:>6.3f} | {t:>7.3f} | {f:>5.3f}")

    # ══════════════════════════════════════════════════════════════
    #  PART 3: Cold-start cross-attack (N per method = 5, 25, 100)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  PART B: Cold-start cross-attack (TPR @ 5% FPR)")
    print(f"  Leave-one-out across 9 methods, varying N per method")
    print(f"{'─'*60}\n")

    ben_split = int(0.8 * len(X_all_ben))
    cv_ben_tr = X_all_ben[:ben_split]
    cv_ben_te = X_all_ben[ben_split:]

    cold_start_results = {}

    for n_per in [5, 25, 100]:
        print(f"\n  Cold-start N={n_per} per method:")
        print(f"  {'Subset':<18} | {'#feats':>6} | {'TPR':>6}")
        print(f"  {'─'*18}─┼─{'─'*6}─┼─{'─'*6}")

        # For each subset, accumulate TPRs across leave-one-out folds
        subset_tprs = {name: [] for name in SUBSETS.keys()}
        for held_out in methods_unique:
            sub_atk = []
            for m in methods_unique:
                if m != held_out:
                    available = hs_by_method[m]
                    take = min(n_per, len(available))
                    sub_atk.append(available[:take])
            train_atk = np.concatenate(sub_atk)
            test_atk = hs_by_method[held_out]
            if len(test_atk) < 5:
                continue
            X_tr = np.concatenate([cv_ben_tr, train_atk])
            y_tr = np.array([0]*len(cv_ben_tr) + [1]*len(train_atk))

            # Train fresh projection on this fold's training data
            proj_fold = train_hps_proj(X_tr, y_tr, seed=42)
            f_tr = extract_trajectory_features(proj_fold, X_tr)
            f_be = extract_trajectory_features(proj_fold, cv_ben_te)
            f_at = extract_trajectory_features(proj_fold, test_atk)

            # Evaluate each subset on this fold
            for name, idx in SUBSETS.items():
                _, t, _ = evaluate_subset(f_tr, y_tr, f_be, f_at, idx)
                subset_tprs[name].append(t)

        # Average across folds
        cold_start_results[n_per] = {}
        for name in SUBSETS.keys():
            mean_tpr = float(np.mean(subset_tprs[name])) if subset_tprs[name] else 0.0
            cold_start_results[n_per][name] = {"tpr": mean_tpr, "n_features": len(SUBSETS[name])}
            print(f"  {name:<18} | {len(SUBSETS[name]):>6} | {mean_tpr:>6.3f}")

    # ══════════════════════════════════════════════════════════════
    #  PART 4: Vicuna-like regime (4 methods × 25 attacks)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  PART C: Vicuna-like regime (4 methods × 25 attacks)")
    print(f"{'─'*60}\n")

    vicuna_methods = methods_unique[:4]
    print(f"  Methods: {vicuna_methods}")
    n_per = 25

    sub_atk = []
    for m in vicuna_methods:
        available = hs_by_method[m]
        take = min(n_per, len(available))
        sub_atk.append(available[:take])

    subset_tprs_vic = {name: [] for name in SUBSETS.keys()}
    for held_out in vicuna_methods:
        train_atk = np.concatenate([sub_atk[i] for i, m in enumerate(vicuna_methods)
                                     if m != held_out])
        idx_held = vicuna_methods.index(held_out)
        test_atk = sub_atk[idx_held]
        if len(test_atk) < 5:
            continue
        X_tr = np.concatenate([cv_ben_tr, train_atk])
        y_tr = np.array([0]*len(cv_ben_tr) + [1]*len(train_atk))
        proj_fold = train_hps_proj(X_tr, y_tr, seed=42)
        f_tr = extract_trajectory_features(proj_fold, X_tr)
        f_be = extract_trajectory_features(proj_fold, cv_ben_te)
        f_at = extract_trajectory_features(proj_fold, test_atk)
        for name, idx in SUBSETS.items():
            _, t, _ = evaluate_subset(f_tr, y_tr, f_be, f_at, idx)
            subset_tprs_vic[name].append(t)

    print(f"  {'Subset':<18} | {'#feats':>6} | {'TPR':>6}")
    print(f"  {'─'*18}─┼─{'─'*6}─┼─{'─'*6}")
    vicuna_results = {}
    for name in SUBSETS.keys():
        mean_tpr = float(np.mean(subset_tprs_vic[name])) if subset_tprs_vic[name] else 0.0
        vicuna_results[name] = {"tpr": mean_tpr, "n_features": len(SUBSETS[name])}
        print(f"  {name:<18} | {len(SUBSETS[name]):>6} | {mean_tpr:>6.3f}")

    # ══════════════════════════════════════════════════════════════
    #  Decision Summary
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  DECISION SUMMARY")
    print(f"{'═'*60}\n")

    a12_sd = same_dist_results["all_12"]["tpr"]
    a12_cs5 = cold_start_results[5]["all_12"]["tpr"]
    a12_cs25 = cold_start_results[25]["all_12"]["tpr"]
    a12_cs100 = cold_start_results[100]["all_12"]["tpr"]
    a12_vic = vicuna_results["all_12"]["tpr"]

    print(f"  all_12 baseline:")
    print(f"    Same-dist:        TPR = {a12_sd:.3f}")
    print(f"    Cold-start N=5:   TPR = {a12_cs5:.3f}")
    print(f"    Cold-start N=25:  TPR = {a12_cs25:.3f}")
    print(f"    Cold-start N=100: TPR = {a12_cs100:.3f}")
    print(f"    Vicuna-like:      TPR = {a12_vic:.3f}")

    print(f"\n  Comparison vs all_12 (Δ TPR; negative = subset worse):")
    print(f"  {'Subset':<18} | {'Same-dist':>10} | {'CS N=5':>7} | {'CS N=25':>7} | {'CS N=100':>8} | {'Vicuna':>7}")
    print(f"  {'─'*18}─┼─{'─'*10}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*8}─┼─{'─'*7}")
    for name in SUBSETS.keys():
        if name == "all_12":
            continue
        d_sd = same_dist_results[name]["tpr"] - a12_sd
        d_cs5 = cold_start_results[5][name]["tpr"] - a12_cs5
        d_cs25 = cold_start_results[25][name]["tpr"] - a12_cs25
        d_cs100 = cold_start_results[100][name]["tpr"] - a12_cs100
        d_vic = vicuna_results[name]["tpr"] - a12_vic
        print(f"  {name:<18} | {d_sd:>+10.3f} | {d_cs5:>+7.3f} | {d_cs25:>+7.3f} | {d_cs100:>+8.3f} | {d_vic:>+7.3f}")

    # Decision rule
    print(f"\n  ── Decision ──")
    top6_deltas = [
        same_dist_results["top6_byimp"]["tpr"] - a12_sd,
        cold_start_results[5]["top6_byimp"]["tpr"] - a12_cs5,
        cold_start_results[25]["top6_byimp"]["tpr"] - a12_cs25,
        cold_start_results[100]["top6_byimp"]["tpr"] - a12_cs100,
        vicuna_results["top6_byimp"]["tpr"] - a12_vic,
    ]
    worst_top6 = min(top6_deltas)
    print(f"  top6_byimp worst Δ across regimes: {worst_top6:+.3f}")

    if worst_top6 >= -0.02:
        print(f"  ✓ DROP TO 6 FEATURES — top6_byimp matches all_12 within 0.02 TPR everywhere")
        print(f"    Recommended subset: {[FEAT_NAMES[i] for i in top6]}")
    elif worst_top6 >= -0.05:
        print(f"  ⚠ MARGINAL — top6_byimp loses {-worst_top6:.3f} TPR in worst regime")
        print(f"    Author's call: keep 12 (safer) or drop to 6 (cleaner)")
    else:
        print(f"  ✗ KEEP ALL 12 — top6_byimp loses {-worst_top6:.3f} TPR (>0.05) in some regime")
        print(f"    The full 12 features matter for the cold-start regime")

    # Save results
    out = {
        "config": {
            "layers": HPS_LAYERS,
            "kappa_init": KAPPA_INIT,
            "freeze_kappa": FREEZE_KAPPA,
            "epochs": EPOCHS,
        },
        "feature_names": FEAT_NAMES,
        "subsets": {name: idx for name, idx in SUBSETS.items()},
        "importance_ranking": [
            {"feature": FEAT_NAMES[i], "importance": float(v)}
            for i, v in sorted(enumerate(importances), key=lambda x: -x[1])
        ],
        "same_dist": same_dist_results,
        "cold_start": cold_start_results,
        "vicuna_like": vicuna_results,
    }
    out_path = "results/feature_ablation.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved → {out_path}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
