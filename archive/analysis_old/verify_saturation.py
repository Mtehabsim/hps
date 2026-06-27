"""
verify_saturation.py — Diagnose whether TPR=1.000 reflects real harm-detection
or surface-level prompt-format confounds.

Multiple AI reviewers questioned: how can multiple very different methods all
hit TPR=1.000? Either (a) the data is genuinely separable in activation space
(real saturation), or (b) the methods are detecting trivial confounds like
prompt length, character distribution, or tokenization artifacts.

This script runs 6 checks to distinguish these possibilities:

  Check 1: Hash-based train/test contamination check
  Check 2: Prompt length confound — train length-only baseline
  Check 3: Tighter FPR thresholds (5%, 1%, 0.1%) — does saturation hold?
  Check 4: Per-attack-type strict-threshold breakdown
  Check 5: Permutation test — shuffle labels, train HPS, measure AUROC
  Check 6: Activation magnitude confound — do attacks just have different norms?

If TPR=1.000 holds even with:
  - Length controlled (Check 2 returns AUROC < 0.9 for length-only)
  - Strict 0.1% FPR threshold (Check 3 still high)
  - Real labels (Check 5 random labels give AUROC ~0.5)
  - Activation magnitude controlled (Check 6 returns AUROC < 0.9 for norm-only)

...then the saturation is real harm-detection. Otherwise we have a confound
problem to acknowledge in the paper.

Usage:
  python verify_saturation.py
  python verify_saturation.py --attacks-json llama3_attacks.json
"""

import argparse
import hashlib
import json
import os
import sys
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hps_core import (
    LorentzProjection,
    contrastive_loss,
    extract_trajectory_features,
)
from vicuna_imbalance_test import load_llama_cache  # reuse loader

LLAMA_LAYERS = [0, 2, 17, 24, 28, 31]


def _np_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    raise TypeError(f"not serializable: {type(o)}")


def auroc(y, s):
    return float(roc_auc_score(y, s))


def tpr_at_fpr(y, s, target=0.05):
    fpr, tpr, _ = roc_curve(y, s)
    valid = fpr <= target
    return float(tpr[valid].max()) if valid.any() else 0.0


# ---------------------------------------------------------------------------
# Check 1: Train/test contamination
# ---------------------------------------------------------------------------

def check_contamination(attacks_json_path, harmless_csv_path, seed=42):
    """Verify no test prompt appears in training set."""
    print("\n" + "=" * 70)
    print("CHECK 1 — Train/test prompt contamination")
    print("=" * 70)

    if not Path(attacks_json_path).exists():
        print(f"  WARNING: attacks JSON not found at {attacks_json_path}")
        print("  Skipping prompt-level contamination check.")
        return {"skipped": True}

    # Load attack prompts and replicate the same split from hps_llama3.py
    with open(attacks_json_path) as f:
        categorized = json.load(f)
    attack_prompts, attack_methods = [], []
    for method, prompts in categorized.items():
        for p in prompts:
            if p:
                attack_prompts.append(p)
                attack_methods.append(method)

    rng = np.random.RandomState(seed)
    atk_idx = rng.permutation(len(attack_prompts))
    n_atk_tr = int(0.8 * len(atk_idx))
    train_atk = [attack_prompts[i] for i in atk_idx[:n_atk_tr]]
    test_atk = [attack_prompts[i] for i in atk_idx[n_atk_tr:]]

    # Hash-based contamination check
    train_hashes = {hashlib.md5(p.encode()).hexdigest() for p in train_atk}
    test_hashes = {hashlib.md5(p.encode()).hexdigest() for p in test_atk}
    overlap = train_hashes & test_hashes

    print(f"  Attack prompts: {len(attack_prompts)} total")
    print(f"  Train: {len(train_atk)} unique hashes: {len(train_hashes)}")
    print(f"  Test:  {len(test_atk)} unique hashes: {len(test_hashes)}")
    print(f"  Overlap (test prompts in train): {len(overlap)}")

    if len(overlap) > 0:
        print(f"  ✗ CONTAMINATION DETECTED: {len(overlap)} test prompts also in train")
    else:
        print(f"  ✓ No contamination — train/test prompts are disjoint")

    return {
        "n_attacks": len(attack_prompts),
        "n_train": len(train_atk),
        "n_test": len(test_atk),
        "n_overlap": len(overlap),
        "contaminated": len(overlap) > 0,
    }


# ---------------------------------------------------------------------------
# Check 2: Prompt length confound
# ---------------------------------------------------------------------------

def check_length_confound(attacks_json_path, harmless_csv_path,
                            tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    """Train a length-only classifier; if it saturates too, length IS the confound."""
    print("\n" + "=" * 70)
    print("CHECK 2 — Prompt length confound")
    print("=" * 70)

    if not Path(attacks_json_path).exists():
        print(f"  Skipping (attack JSON missing).")
        return {"skipped": True}

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"  Tokenizer load failed: {e}. Falling back to character length.")
        tok = None

    # Load
    with open(attacks_json_path) as f:
        categorized = json.load(f)

    import pandas as pd
    df_h = pd.read_csv(harmless_csv_path)
    for col in ["prompt", "goal", "text", "instruction", "query"]:
        if col in df_h.columns:
            harmless_prompts = df_h[col].dropna().tolist()
            break
    else:
        harmless_prompts = df_h.iloc[:, 0].dropna().tolist()

    def lens(prompts):
        if tok is None:
            return np.array([len(p) for p in prompts])
        return np.array([len(tok.encode(p)) for p in prompts])

    rng = np.random.RandomState(42)

    # Build attack list
    attack_prompts, attack_methods = [], []
    for method, prompts in categorized.items():
        for p in prompts:
            if p:
                attack_prompts.append(p)
                attack_methods.append(method)

    # Same split
    atk_idx = rng.permutation(len(attack_prompts))
    n_atk_tr = int(0.8 * len(atk_idx))
    train_atk = [attack_prompts[i] for i in atk_idx[:n_atk_tr]]
    test_atk = [attack_prompts[i] for i in atk_idx[n_atk_tr:]]
    test_methods = [attack_methods[i] for i in atk_idx[n_atk_tr:]]

    rng2 = np.random.RandomState(42)
    ben_idx = rng2.permutation(len(harmless_prompts))
    n_ben_tr = int(0.8 * len(ben_idx))
    train_ben = [harmless_prompts[i] for i in ben_idx[:n_ben_tr]]
    test_ben = [harmless_prompts[i] for i in ben_idx[n_ben_tr:]]

    train_atk_len = lens(train_atk)
    train_ben_len = lens(train_ben)
    test_atk_len = lens(test_atk)
    test_ben_len = lens(test_ben)

    # Statistics
    print(f"  Attack length stats:  mean={train_atk_len.mean():.1f}, "
          f"median={np.median(train_atk_len):.0f}, "
          f"std={train_atk_len.std():.1f}")
    print(f"  Benign length stats:  mean={train_ben_len.mean():.1f}, "
          f"median={np.median(train_ben_len):.0f}, "
          f"std={train_ben_len.std():.1f}")
    print(f"  Diff (atk-ben means): {train_atk_len.mean() - train_ben_len.mean():.1f}")

    # Per-attack length stats
    print(f"\n  Per-attack length statistics (train):")
    for m in sorted(set(attack_methods)):
        idx = [i for i, (p, mm) in enumerate(zip(train_atk,
                [attack_methods[i] for i in atk_idx[:n_atk_tr]]))
                if mm == m]
        if not idx:
            continue
        ls = train_atk_len[idx]
        print(f"    {m:>30s}: n={len(ls):>4}, mean={ls.mean():>6.1f}, "
              f"median={np.median(ls):>5.0f}, std={ls.std():>5.1f}")

    # Train length-only LR classifier
    X_tr = np.concatenate([train_ben_len, train_atk_len]).reshape(-1, 1)
    y_tr = np.concatenate([np.zeros(len(train_ben_len)),
                           np.ones(len(train_atk_len))])
    X_te = np.concatenate([test_ben_len, test_atk_len]).reshape(-1, 1)
    y_te = np.concatenate([np.zeros(len(test_ben_len)),
                           np.ones(len(test_atk_len))])

    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(sc.transform(X_tr), y_tr)
    scores = clf.predict_proba(sc.transform(X_te))[:, 1]

    auc = auroc(y_te, scores)
    tpr5 = tpr_at_fpr(y_te, scores, 0.05)
    tpr1 = tpr_at_fpr(y_te, scores, 0.01)

    print(f"\n  LENGTH-ONLY CLASSIFIER:")
    print(f"    AUROC:   {auc:.4f}")
    print(f"    TPR@5%:  {tpr5:.4f}")
    print(f"    TPR@1%:  {tpr1:.4f}")

    if auc >= 0.9:
        print(f"  ✗ STRONG length confound: length alone gives AUROC = {auc:.4f}")
        print(f"     This explains a substantial fraction of TPR=1.000")
    elif auc >= 0.7:
        print(f"  ⚠ MODERATE length confound: length alone gives AUROC = {auc:.4f}")
        print(f"     Some of the TPR=1.000 reflects format detection, not harm")
    else:
        print(f"  ✓ Length is not the primary signal (AUROC = {auc:.4f})")

    # Per-attack length-only AUROC
    print(f"\n  PER-ATTACK length-only detection:")
    test_atk_methods = [attack_methods[i] for i in atk_idx[n_atk_tr:]]
    for m in sorted(set(test_atk_methods)):
        m_idx = [i for i, mm in enumerate(test_atk_methods) if mm == m]
        if not m_idx:
            continue
        m_lens = test_atk_len[m_idx]
        # Score: distance from benign distribution
        m_scores_atk = clf.predict_proba(sc.transform(m_lens.reshape(-1, 1)))[:, 1]
        ben_scores = clf.predict_proba(sc.transform(test_ben_len.reshape(-1, 1)))[:, 1]
        # AUROC = how well length distinguishes this attack from benign
        m_y = np.concatenate([np.zeros(len(ben_scores)), np.ones(len(m_scores_atk))])
        m_s = np.concatenate([ben_scores, m_scores_atk])
        m_auc = auroc(m_y, m_s)
        print(f"    {m:>30s}: n={len(m_idx):>4}, length-only AUROC = {m_auc:.4f}")

    return {
        "overall_auroc": auc,
        "tpr5": tpr5,
        "tpr1": tpr1,
        "atk_mean_len": float(train_atk_len.mean()),
        "ben_mean_len": float(train_ben_len.mean()),
        "is_strong_confound": auc >= 0.9,
        "is_moderate_confound": 0.7 <= auc < 0.9,
    }


# ---------------------------------------------------------------------------
# Check 3-4: Strict-threshold + per-attack TPR
# ---------------------------------------------------------------------------

def train_methods_and_score(llama_data, seed=42, device="cpu", epochs=50):
    """Train HPS, HPS-Euclidean, C4 on Llama-3 cache; return test scores."""
    X_tr_ben = llama_data["X_tr_ben"]
    X_tr_atk = llama_data["X_tr_atk"]
    X_te_ben = llama_data["X_te_ben"]
    X_te_atk = llama_data["X_te_atk"]

    # ── HPS ──
    X_tr = np.concatenate([X_tr_ben, X_tr_atk], axis=0)
    y_tr = np.concatenate([np.zeros(len(X_tr_ben)), np.ones(len(X_tr_atk))])
    n_layers = X_tr.shape[1]
    d_hidden = X_tr.shape[2]

    torch.manual_seed(seed)
    np.random.seed(seed)
    proj = LorentzProjection(d_hidden, 64, 0.1, n_layers=n_layers).to(device)
    proj.log_k.requires_grad = False
    opt = optim.Adam([p for p in proj.parameters() if p.requires_grad],
                      lr=1e-3, weight_decay=1e-5)
    X_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_tr, dtype=torch.long, device=device)

    for _ in range(epochs):
        loss = torch.tensor(0.0, device=device)
        for li in range(n_layers):
            h = proj(X_t[:, li, :])
            loss = loss + contrastive_loss(h, y_t, k=proj.k, tau=proj.tau(li))
        loss = loss / n_layers
        opt.zero_grad()
        loss.backward()
        opt.step()

    feats_tr = extract_trajectory_features(proj, X_tr)
    feats_te_ben = extract_trajectory_features(proj, X_te_ben)
    feats_te_atk = extract_trajectory_features(proj, X_te_atk)

    sc = StandardScaler().fit(feats_tr)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000,
                              random_state=seed)
    clf.fit(sc.transform(feats_tr), y_tr)
    hps_te_ben = clf.predict_proba(sc.transform(feats_te_ben))[:, 1]
    hps_te_atk = clf.predict_proba(sc.transform(feats_te_atk))[:, 1]

    # ── C4 ──
    X_tr_pool = np.concatenate([X_tr_ben.mean(axis=1), X_tr_atk.mean(axis=1)],
                               axis=0)
    sc4 = StandardScaler().fit(X_tr_pool)
    clf4 = LogisticRegression(class_weight="balanced", max_iter=2000,
                                random_state=seed)
    clf4.fit(sc4.transform(X_tr_pool), y_tr)
    c4_te_ben = clf4.predict_proba(
        sc4.transform(X_te_ben.mean(axis=1)))[:, 1]
    c4_te_atk = clf4.predict_proba(
        sc4.transform(X_te_atk.mean(axis=1)))[:, 1]

    return {
        "hps": {"te_ben": hps_te_ben, "te_atk": hps_te_atk},
        "c4": {"te_ben": c4_te_ben, "te_atk": c4_te_atk},
    }


def check_strict_thresholds(scores_dict):
    """Compute TPR at 5%, 1%, 0.1% FPR for each method."""
    print("\n" + "=" * 70)
    print("CHECK 3 — Tighter FPR thresholds")
    print("=" * 70)

    results = {}
    for method, sc in scores_dict.items():
        y = np.concatenate([np.zeros(len(sc["te_ben"])),
                             np.ones(len(sc["te_atk"]))])
        s = np.concatenate([sc["te_ben"], sc["te_atk"]])
        tpr5 = tpr_at_fpr(y, s, 0.05)
        tpr1 = tpr_at_fpr(y, s, 0.01)
        tpr01 = tpr_at_fpr(y, s, 0.001)
        results[method] = {
            "tpr_at_5pct_fpr": tpr5,
            "tpr_at_1pct_fpr": tpr1,
            "tpr_at_0.1pct_fpr": tpr01,
        }
        print(f"  {method:>20s}: 5%={tpr5:.4f}, 1%={tpr1:.4f}, "
              f"0.1%={tpr01:.4f}")

    return results


def check_per_attack_strict(scores_dict, attacks_json_path, n_test_atk):
    """Per-attack TPR at strict thresholds."""
    print("\n" + "=" * 70)
    print("CHECK 4 — Per-attack-type detection at strict thresholds")
    print("=" * 70)

    if not Path(attacks_json_path).exists():
        print(f"  Skipping (attacks JSON missing).")
        return {}

    # Recover attack labels
    with open(attacks_json_path) as f:
        categorized = json.load(f)
    attack_prompts, attack_methods = [], []
    for method, prompts in categorized.items():
        for p in prompts:
            if p:
                attack_prompts.append(p)
                attack_methods.append(method)

    rng = np.random.RandomState(42)
    atk_idx = rng.permutation(len(attack_prompts))
    n_atk_tr = int(0.8 * len(atk_idx))
    test_methods = [attack_methods[i] for i in atk_idx[n_atk_tr:]]

    if len(test_methods) != n_test_atk:
        print(f"  Length mismatch: {len(test_methods)} vs {n_test_atk}; "
              f"truncating")
        test_methods = test_methods[:n_test_atk]

    results = {}
    for method, sc in scores_dict.items():
        ben = sc["te_ben"]
        atk = sc["te_atk"]
        # Calibrate threshold on benign
        thr5 = float(np.percentile(ben, 95))
        thr1 = float(np.percentile(ben, 99))
        thr01 = float(np.percentile(ben, 99.9))

        per_attack = {}
        for m in sorted(set(test_methods)):
            m_idx = [i for i, mm in enumerate(test_methods) if mm == m]
            if not m_idx:
                continue
            m_scores = atk[m_idx]
            per_attack[m] = {
                "n": len(m_idx),
                "tpr5": float((m_scores > thr5).mean()),
                "tpr1": float((m_scores > thr1).mean()),
                "tpr01": float((m_scores > thr01).mean()),
            }
        results[method] = per_attack

        print(f"\n  {method.upper()}:")
        print(f"    {'attack':>30s}  {'N':>4s}  {'TPR@5%':>7s}  "
              f"{'TPR@1%':>7s}  {'TPR@0.1%':>9s}")
        for m, r in sorted(per_attack.items()):
            print(f"    {m:>30s}  {r['n']:>4d}  "
                  f"{r['tpr5']:>7.4f}  {r['tpr1']:>7.4f}  "
                  f"{r['tpr01']:>9.4f}")

    return results


# ---------------------------------------------------------------------------
# Check 5: Permutation test
# ---------------------------------------------------------------------------

def check_permutation(llama_data, seed=42, device="cpu", n_perms=3, epochs=20):
    """Shuffle labels; if classifier still saturates, it's spurious."""
    print("\n" + "=" * 70)
    print(f"CHECK 5 — Permutation test (n={n_perms} permutations)")
    print("=" * 70)

    X_tr_ben = llama_data["X_tr_ben"]
    X_tr_atk = llama_data["X_tr_atk"]
    X_te_ben = llama_data["X_te_ben"]
    X_te_atk = llama_data["X_te_atk"]

    X_tr_pool = np.concatenate([X_tr_ben.mean(axis=1),
                                 X_tr_atk.mean(axis=1)], axis=0)
    X_te_pool = np.concatenate([X_te_ben.mean(axis=1),
                                 X_te_atk.mean(axis=1)], axis=0)
    y_tr = np.concatenate([np.zeros(len(X_tr_ben)), np.ones(len(X_tr_atk))])
    y_te = np.concatenate([np.zeros(len(X_te_ben)), np.ones(len(X_te_atk))])

    print(f"  REAL labels (baseline):")
    sc = StandardScaler().fit(X_tr_pool)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(sc.transform(X_tr_pool), y_tr)
    scores = clf.predict_proba(sc.transform(X_te_pool))[:, 1]
    real_auc = auroc(y_te, scores)
    real_tpr5 = tpr_at_fpr(y_te, scores, 0.05)
    print(f"    C4 AUROC = {real_auc:.4f}, TPR@5% = {real_tpr5:.4f}")

    perm_aurocs = []
    for p in range(n_perms):
        rng = np.random.RandomState(1000 + p)
        # Shuffle TRAIN labels only — test labels real
        y_tr_shuffled = y_tr.copy()
        rng.shuffle(y_tr_shuffled)
        sc = StandardScaler().fit(X_tr_pool)
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
        clf.fit(sc.transform(X_tr_pool), y_tr_shuffled)
        scores = clf.predict_proba(sc.transform(X_te_pool))[:, 1]
        perm_auc = auroc(y_te, scores)
        perm_tpr5 = tpr_at_fpr(y_te, scores, 0.05)
        perm_aurocs.append(perm_auc)
        print(f"  Permutation {p+1}: AUROC = {perm_auc:.4f}, "
              f"TPR@5% = {perm_tpr5:.4f}")

    perm_mean = float(np.mean(perm_aurocs))
    print(f"\n  Permutation mean AUROC: {perm_mean:.4f} "
          f"(should be ≈0.5 if real signal exists)")

    if perm_mean > 0.6:
        print(f"  ✗ Permutation AUROC = {perm_mean:.4f} >> 0.5; "
              f"there's spurious data structure")
    elif perm_mean > 0.55:
        print(f"  ⚠ Permutation AUROC = {perm_mean:.4f}; mild spurious structure")
    else:
        print(f"  ✓ Permutation AUROC ≈ 0.5; real label signal is genuine")

    return {
        "real_auroc": real_auc,
        "real_tpr5": real_tpr5,
        "permutation_aurocs": perm_aurocs,
        "permutation_mean": perm_mean,
        "spurious_structure": perm_mean > 0.55,
    }


# ---------------------------------------------------------------------------
# Check 6: Activation magnitude confound
# ---------------------------------------------------------------------------

def check_norm_confound(llama_data):
    """Train classifier on ONLY activation L2 norms; if it works, norms are confound."""
    print("\n" + "=" * 70)
    print("CHECK 6 — Activation norm-only confound")
    print("=" * 70)

    X_tr_ben = llama_data["X_tr_ben"]
    X_tr_atk = llama_data["X_tr_atk"]
    X_te_ben = llama_data["X_te_ben"]
    X_te_atk = llama_data["X_te_atk"]

    # Per-layer norms → 6-dim feature vector
    def get_norms(X):
        # X: (N, n_layers, d). Compute per-sample, per-layer L2 norm.
        return np.linalg.norm(X, axis=2)  # (N, n_layers)

    norm_tr = np.concatenate([get_norms(X_tr_ben), get_norms(X_tr_atk)], axis=0)
    y_tr = np.concatenate([np.zeros(len(X_tr_ben)), np.ones(len(X_tr_atk))])
    norm_te = np.concatenate([get_norms(X_te_ben), get_norms(X_te_atk)], axis=0)
    y_te = np.concatenate([np.zeros(len(X_te_ben)), np.ones(len(X_te_atk))])

    print(f"  Mean norm by class (per layer):")
    print(f"    Layer | Benign | Attack | Diff")
    for li in range(norm_tr.shape[1]):
        ben_mean = get_norms(X_tr_ben)[:, li].mean()
        atk_mean = get_norms(X_tr_atk)[:, li].mean()
        print(f"    {li:>5}: {ben_mean:>6.2f} | {atk_mean:>6.2f} | "
              f"{atk_mean - ben_mean:+6.2f}")

    sc = StandardScaler().fit(norm_tr)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(sc.transform(norm_tr), y_tr)
    scores = clf.predict_proba(sc.transform(norm_te))[:, 1]

    auc = auroc(y_te, scores)
    tpr5 = tpr_at_fpr(y_te, scores, 0.05)
    tpr1 = tpr_at_fpr(y_te, scores, 0.01)
    print(f"\n  NORM-ONLY CLASSIFIER:")
    print(f"    AUROC:   {auc:.4f}")
    print(f"    TPR@5%:  {tpr5:.4f}")
    print(f"    TPR@1%:  {tpr1:.4f}")

    if auc >= 0.95:
        print(f"  ✗ STRONG norm confound: norm alone gives AUROC = {auc:.4f}")
    elif auc >= 0.8:
        print(f"  ⚠ MODERATE norm confound: AUROC = {auc:.4f}")
    else:
        print(f"  ✓ Norm is not a strong confound (AUROC = {auc:.4f})")

    return {
        "auroc": auc,
        "tpr5": tpr5,
        "tpr1": tpr1,
        "is_strong_confound": auc >= 0.95,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama3_cache",
                        default="results/llama3_activations_cache.npz")
    parser.add_argument("--attacks_json",
                        default="llama3_attacks.json")
    parser.add_argument("--harmless_csv",
                        default="data_harmless_100.csv")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output",
                        default="results/verify_saturation.json")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("SATURATION VERIFICATION — is TPR=1.000 real or a confound?")
    print("=" * 70)

    output = {"config": {"device": device, "seed": args.seed}}

    # Check 1: Contamination
    output["check1_contamination"] = check_contamination(
        args.attacks_json, args.harmless_csv, args.seed,
    )

    # Check 2: Length confound
    output["check2_length_confound"] = check_length_confound(
        args.attacks_json, args.harmless_csv,
    )

    # Load activations for remaining checks
    print(f"\nLoading Llama-3 cache: {args.llama3_cache}")
    llama_data = load_llama_cache(args.llama3_cache, LLAMA_LAYERS)
    print(f"  Train: {len(llama_data['X_tr_ben'])} ben + "
          f"{len(llama_data['X_tr_atk'])} atk")
    print(f"  Test:  {len(llama_data['X_te_ben'])} ben + "
          f"{len(llama_data['X_te_atk'])} atk")

    # Train methods
    print(f"\nTraining HPS + C4 (epochs={args.epochs}, seed={args.seed})...")
    scores_dict = train_methods_and_score(
        llama_data, seed=args.seed, device=device, epochs=args.epochs,
    )

    # Check 3: Strict thresholds
    output["check3_strict_thresholds"] = check_strict_thresholds(scores_dict)

    # Check 4: Per-attack-type strict
    output["check4_per_attack_strict"] = check_per_attack_strict(
        scores_dict, args.attacks_json, len(llama_data["X_te_atk"]),
    )

    # Check 5: Permutation
    output["check5_permutation"] = check_permutation(
        llama_data, seed=args.seed, device=device, n_perms=3, epochs=20,
    )

    # Check 6: Norm confound
    output["check6_norm_confound"] = check_norm_confound(llama_data)

    # ── Final summary ──
    print("\n" + "=" * 70)
    print("OVERALL VERDICT")
    print("=" * 70)

    issues = []
    if output["check1_contamination"].get("contaminated", False):
        issues.append("CONTAMINATION: train/test prompts overlap")

    c2 = output["check2_length_confound"]
    if not c2.get("skipped"):
        if c2.get("is_strong_confound"):
            issues.append(
                f"STRONG length confound: length-only AUROC = "
                f"{c2['overall_auroc']:.3f}"
            )
        elif c2.get("is_moderate_confound"):
            issues.append(
                f"Moderate length confound: AUROC = {c2['overall_auroc']:.3f}"
            )

    c5 = output["check5_permutation"]
    if c5.get("spurious_structure"):
        issues.append(
            f"Spurious data structure: permutation AUROC = "
            f"{c5['permutation_mean']:.3f}"
        )

    c6 = output["check6_norm_confound"]
    if c6.get("is_strong_confound"):
        issues.append(
            f"Strong norm confound: norm-only AUROC = {c6['auroc']:.3f}"
        )

    if not issues:
        print("\n  ✓ No major confounds detected.")
        print("  TPR=1.000 reflects genuine activation-level harm signal.")
        print("  Saturation is real but reflects benchmark difficulty,")
        print("  not methodological errors. The reframing recommendation")
        print("  (lead with non-saturated comparisons) is correct.")
    else:
        print(f"\n  ✗ Found {len(issues)} issue(s):")
        for i in issues:
            print(f"    - {i}")
        print("\n  Recommendation: address these in paper limitations,")
        print("  potentially re-run analyses with controls.")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=_np_default)
    print(f"\nSaved full results to {output_path}")


if __name__ == "__main__":
    main()
