"""
vicuna_imbalance_test.py — Three remaining hypotheses for the HPS-Vicuna gap.

After vicuna_overfitting_test.py refuted data scarcity (Llama-3 with 668
balanced samples → TPR=0.999 vs Vicuna with 669 imbalanced samples → TPR=0.810),
this script tests three remaining hypotheses:

  Hypothesis I (CLASS IMBALANCE):
    The Vicuna training data is 416 benign + 253 attacks (1.64:1 imbalanced).
    The Llama-3 subsample test used balanced 334+334. Test if forcing Vicuna
    to balance OR forcing Llama-3 to imbalanced 416+253 changes the picture.

  Hypothesis J (ATTACK DIVERSITY):
    Vicuna cache contains a specific set of attack methods. If it has fewer
    distinct attack types than Llama-3 (which has 9), HPS may overfit to
    specific attack signatures. Print the actual attack counts.

  Hypothesis K (PER-ATTACK FAILURE PATTERN):
    HPS may fail on specific Vicuna attack categories while succeeding on
    others, while C4 generalizes uniformly. Per-attack breakdown reveals this.

Tests run:
  T1.  Vicuna BALANCED (sub to 253 ben + 253 atk)
  T2.  Vicuna BALANCED + drop shallow (best fix from previous test)
  T3.  Llama-3 with Vicuna-STYLE imbalance (416 ben + 253 atk, all 9 attacks)
  T4.  Llama-3 with EXTREME imbalance (matching Vicuna ratio + lower count)
  T5.  Per-attack breakdown on Vicuna test set: HPS vs C4
  T6.  Attack composition statistics

Usage:
  python vicuna_imbalance_test.py
  python vicuna_imbalance_test.py --epochs 50
"""

import argparse
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

LLAMA_LAYERS = [0, 2, 17, 24, 28, 31]
VICUNA_LAYERS = [0, 2, 22, 31, 35, 39]
VICUNA_LAYERS_DEEP = [22, 31, 35, 39]
KAPPA_INIT = 0.1
PROJ_DIM = 64


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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


def to_hps_array(hs_list, layers):
    """Convert list-of-dicts (Llama-3 format) to (N, n_layers, d)."""
    n = len(hs_list)
    if n == 0:
        return np.empty((0, len(layers), 0), dtype=np.float32)
    sample = hs_list[0][layers[0]]
    d = sample.shape[1] if sample.ndim == 2 else sample.shape[0]
    out = np.empty((n, len(layers), d), dtype=np.float32)
    for i, hs in enumerate(hs_list):
        for li, L in enumerate(layers):
            t = hs[L]
            out[i, li, :] = t[-1] if t.ndim == 2 else t
    return out


def _hs_array_to_hps(np_object_array, layers):
    """Stream through numpy object array (memory-efficient)."""
    n = len(np_object_array)
    if n == 0:
        return np.empty((0, len(layers), 0), dtype=np.float32)
    first = np_object_array[0]
    sample = first[layers[0]]
    d = sample.shape[1] if sample.ndim == 2 else sample.shape[0]
    out = np.empty((n, len(layers), d), dtype=np.float32)
    for i in range(n):
        hs = np_object_array[i]
        for li, L in enumerate(layers):
            t = hs[L]
            out[i, li, :] = t[-1] if t.ndim == 2 else t
    return out


def load_llama_cache(path, layers):
    """Load Llama-3 dict-format cache."""
    cache = np.load(path, allow_pickle=True)
    return {
        "X_tr_ben": _hs_array_to_hps(cache["hs_train_ben"], layers),
        "X_tr_atk": _hs_array_to_hps(cache["hs_train_atk"], layers),
        "X_te_ben": _hs_array_to_hps(cache["hs_test_ben"], layers),
        "X_te_atk": _hs_array_to_hps(cache["hs_test_atk"], layers),
    }


def load_vicuna_cache(path, layers):
    """Load Vicuna pre-extracted cache, keep attack_methods alignment.

    Supports two formats:
      - Vicuna real format: X_benign/X_attack/attack_methods/layers
      - Dict format (for local synthetic testing): hs_train_ben/atk + hs_test_ben/atk
        (in this case, attack_methods will be None placeholders)
    """
    cache = np.load(path, allow_pickle=True)
    keys = list(cache.keys())

    if "X_benign" in keys:
        # Real Vicuna cache
        X_ben = np.array(cache["X_benign"])
        X_atk = np.array(cache["X_attack"])
        attack_methods = cache["attack_methods"].tolist() \
            if "attack_methods" in cache else [None] * len(X_atk)
        cached_layers = cache["layers"].tolist() if "layers" in cache else None

        if cached_layers is not None and set(layers) <= set(cached_layers):
            idx = [cached_layers.index(L) for L in layers]
            X_ben = X_ben[:, idx, :]
            X_atk = X_atk[:, idx, :]

        rng = np.random.RandomState(42)
        ben_perm = rng.permutation(len(X_ben))
        rng = np.random.RandomState(43)
        atk_perm = rng.permutation(len(X_atk))

        n_ben_te = max(1, int(0.2 * len(X_ben)))
        n_atk_te = max(1, int(0.2 * len(X_atk)))

        tr_ben_idx = ben_perm[n_ben_te:]
        te_ben_idx = ben_perm[:n_ben_te]
        tr_atk_idx = atk_perm[n_atk_te:]
        te_atk_idx = atk_perm[:n_atk_te]

        return {
            "X_tr_ben": X_ben[tr_ben_idx],
            "X_tr_atk": X_atk[tr_atk_idx],
            "X_te_ben": X_ben[te_ben_idx],
            "X_te_atk": X_atk[te_atk_idx],
            "atk_methods_train": [attack_methods[i] for i in tr_atk_idx],
            "atk_methods_test":  [attack_methods[i] for i in te_atk_idx],
        }

    if "hs_train_ben" in keys:
        # Dict format fallback (local testing)
        X_tr_ben = _hs_array_to_hps(cache["hs_train_ben"], layers)
        X_tr_atk = _hs_array_to_hps(cache["hs_train_atk"], layers)
        X_te_ben = _hs_array_to_hps(cache["hs_test_ben"], layers)
        X_te_atk = _hs_array_to_hps(cache["hs_test_atk"], layers)
        return {
            "X_tr_ben": X_tr_ben,
            "X_tr_atk": X_tr_atk,
            "X_te_ben": X_te_ben,
            "X_te_atk": X_te_atk,
            "atk_methods_train": ["unknown"] * len(X_tr_atk),
            "atk_methods_test":  ["unknown"] * len(X_te_atk),
        }

    raise ValueError(
        f"Unknown cache format. Keys: {keys}. "
        f"Expected 'X_benign' (Vicuna) or 'hs_train_ben' (dict format)."
    )


def subsample_balanced(X_ben, X_atk, n_total, seed=42):
    """Equal-class subsample."""
    rng = np.random.RandomState(seed)
    n_per = n_total // 2
    n_ben = min(n_per, len(X_ben))
    n_atk = min(n_per, len(X_atk))
    return (
        X_ben[rng.choice(len(X_ben), size=n_ben, replace=False)],
        X_atk[rng.choice(len(X_atk), size=n_atk, replace=False)],
    )


def subsample_imbalanced(X_ben, X_atk, n_ben, n_atk, seed=42):
    """Specific count per class."""
    rng = np.random.RandomState(seed)
    return (
        X_ben[rng.choice(len(X_ben), size=min(n_ben, len(X_ben)), replace=False)],
        X_atk[rng.choice(len(X_atk), size=min(n_atk, len(X_atk)), replace=False)],
    )


# ---------------------------------------------------------------------------
# Training / scoring
# ---------------------------------------------------------------------------

def train_hps(X_tr, y_tr, n_layers, d, proj_dim=PROJ_DIM, kappa_init=KAPPA_INIT,
              epochs=50, weight_decay=1e-5, seed=42, device="cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    proj = LorentzProjection(d, proj_dim, kappa_init,
                              n_layers=n_layers).to(device)
    proj.log_k.requires_grad = False
    opt = optim.Adam(
        [p for p in proj.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=weight_decay,
    )
    X_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    losses = []
    for _ in range(epochs):
        loss = torch.tensor(0.0, device=device)
        for li in range(n_layers):
            h = proj(X_t[:, li, :])
            loss = loss + contrastive_loss(h, y_t, k=proj.k, tau=proj.tau(li))
        loss = loss / n_layers
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return proj, losses


def score_via_hps(proj, X_train, y_train, X_eval, seed=42):
    feats_train = extract_trajectory_features(proj, X_train)
    feats_eval = extract_trajectory_features(proj, X_eval)
    sc = StandardScaler().fit(feats_train)
    clf = LogisticRegression(max_iter=2000, random_state=seed,
                             class_weight="balanced")
    clf.fit(sc.transform(feats_train), y_train)
    return clf.predict_proba(sc.transform(feats_eval))[:, 1]


def score_via_c4(X_tr_ben, X_tr_atk, X_eval, seed=42):
    X_tr = np.concatenate([X_tr_ben.mean(axis=1),
                           X_tr_atk.mean(axis=1)], axis=0)
    y_tr = np.concatenate([
        np.zeros(len(X_tr_ben)),
        np.ones(len(X_tr_atk)),
    ])
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(max_iter=2000, random_state=seed,
                             class_weight="balanced")
    clf.fit(sc.transform(X_tr), y_tr)
    return clf.predict_proba(sc.transform(X_eval.mean(axis=1)))[:, 1]


def evaluate_config(name, X_tr_ben, X_tr_atk, X_te_ben, X_te_atk,
                     proj_dim=PROJ_DIM, kappa_init=KAPPA_INIT,
                     epochs=50, weight_decay=1e-5,
                     seed=42, device="cpu"):
    X_train = np.concatenate([X_tr_ben, X_tr_atk], axis=0)
    y_train = np.concatenate([
        np.zeros(len(X_tr_ben)),
        np.ones(len(X_tr_atk)),
    ])
    n_layers = X_train.shape[1]
    d = X_train.shape[2]
    ratio = (len(X_tr_ben) / max(len(X_tr_atk), 1))

    print(f"\n  [{name}]")
    print(f"    train: {len(X_tr_ben)} ben + {len(X_tr_atk)} atk "
          f"(ratio {ratio:.2f}:1), n_layers={n_layers}, d={d}, "
          f"epochs={epochs}, wd={weight_decay}")

    proj, losses = train_hps(
        X_train, y_train, n_layers, d,
        proj_dim=proj_dim, epochs=epochs,
        weight_decay=weight_decay, seed=seed, device=device,
    )

    test_eval = np.concatenate([X_te_ben, X_te_atk], axis=0)
    test_y = np.concatenate([
        np.zeros(len(X_te_ben)),
        np.ones(len(X_te_atk)),
    ])
    train_scores_hps = score_via_hps(proj, X_train, y_train, X_train, seed)
    test_scores_hps = score_via_hps(proj, X_train, y_train, test_eval, seed)
    test_scores_c4 = score_via_c4(X_tr_ben, X_tr_atk, test_eval, seed)

    hps_train_a = auroc(y_train, train_scores_hps)
    hps_test_a = auroc(test_y, test_scores_hps)
    hps_test_t = tpr_at_fpr(test_y, test_scores_hps)
    c4_test_a = auroc(test_y, test_scores_c4)
    c4_test_t = tpr_at_fpr(test_y, test_scores_c4)

    print(f"    HPS  loss={losses[-1]:.4f}  "
          f"trAUC={hps_train_a:.4f}  teAUC={hps_test_a:.4f}  "
          f"teTPR5={hps_test_t:.4f}  gap={hps_train_a - hps_test_a:+.4f}")
    print(f"    C4                              "
          f"          teAUC={c4_test_a:.4f}  "
          f"teTPR5={c4_test_t:.4f}")

    return {
        "name": name,
        "n_train_ben": int(len(X_tr_ben)),
        "n_train_atk": int(len(X_tr_atk)),
        "ratio": float(ratio),
        "n_layers": int(n_layers),
        "d_hidden": int(d),
        "final_loss": float(losses[-1]),
        "hps_train_auroc": hps_train_a,
        "hps_test_auroc": hps_test_a,
        "hps_test_tpr5": hps_test_t,
        "hps_train_test_gap": hps_train_a - hps_test_a,
        "c4_test_auroc": c4_test_a,
        "c4_test_tpr5": c4_test_t,
        "hps_minus_c4_tpr5": hps_test_t - c4_test_t,
        "test_scores_hps": test_scores_hps.tolist(),
        "test_scores_c4": test_scores_c4.tolist(),
        "test_y": test_y.tolist(),
    }


# ---------------------------------------------------------------------------
# Per-attack breakdown
# ---------------------------------------------------------------------------

def per_attack_breakdown(test_scores_hps, test_scores_c4, test_y,
                          atk_methods_test, threshold_hps, threshold_c4):
    """For each attack method, compute detection rate."""
    n_ben = int((np.array(test_y) == 0).sum())
    # Test scores: first n_ben are benign, rest are attacks
    atk_scores_hps = np.array(test_scores_hps[n_ben:])
    atk_scores_c4 = np.array(test_scores_c4[n_ben:])
    methods = list(atk_methods_test)

    if len(methods) != len(atk_scores_hps):
        print(f"  WARNING: attack_methods len ({len(methods)}) != "
              f"atk_scores len ({len(atk_scores_hps)}). Skipping breakdown.")
        return {}

    breakdown = {}
    method_set = sorted(set(methods))
    for m in method_set:
        idx = [i for i, x in enumerate(methods) if x == m]
        if not idx:
            continue
        n = len(idx)
        hps_detected = int((atk_scores_hps[idx] > threshold_hps).sum())
        c4_detected = int((atk_scores_c4[idx] > threshold_c4).sum())
        breakdown[m] = {
            "n_total": n,
            "hps_detection_rate": hps_detected / n,
            "c4_detection_rate": c4_detected / n,
            "hps_mean_score": float(atk_scores_hps[idx].mean()),
            "c4_mean_score": float(atk_scores_c4[idx].mean()),
        }
    return breakdown


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama3_cache",
                        default="results/llama3_activations_cache.npz")
    parser.add_argument("--vicuna_cache",
                        default="results/vicuna_activations_cache.npz")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output",
                        default="results/vicuna_imbalance_test.json")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print("VICUNA IMBALANCE + DIVERSITY TEST")
    print("=" * 78)
    print(f"  Llama-3 cache: {args.llama3_cache}")
    print(f"  Vicuna cache:  {args.vicuna_cache}")
    print(f"  Device:        {device}")
    print(f"  Epochs:        {args.epochs}")
    print()

    # Load
    llama = load_llama_cache(args.llama3_cache, LLAMA_LAYERS)
    vicuna = load_vicuna_cache(args.vicuna_cache, VICUNA_LAYERS)

    print(f"  Llama-3: train {len(llama['X_tr_ben'])}+{len(llama['X_tr_atk'])} "
          f"= {len(llama['X_tr_ben']) + len(llama['X_tr_atk'])}, "
          f"test {len(llama['X_te_ben'])}+{len(llama['X_te_atk'])}")
    print(f"  Vicuna:  train {len(vicuna['X_tr_ben'])}+{len(vicuna['X_tr_atk'])} "
          f"= {len(vicuna['X_tr_ben']) + len(vicuna['X_tr_atk'])}, "
          f"test {len(vicuna['X_te_ben'])}+{len(vicuna['X_te_atk'])}")

    # ─────────────────────────────────────────────────────────────────────
    # T6: Attack composition
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("T6 — Attack composition stats")
    print("=" * 78)

    train_methods = vicuna["atk_methods_train"]
    test_methods = vicuna["atk_methods_test"]
    train_counts = Counter(train_methods)
    test_counts = Counter(test_methods)
    print(f"  Vicuna attack methods (train): {dict(train_counts)}")
    print(f"  Vicuna attack methods (test):  {dict(test_counts)}")
    print(f"  Number of unique attack methods on Vicuna: "
          f"{len(set(train_methods + test_methods))}")
    print(f"  (Llama-3 cache has 9 attack methods by construction in "
          f"hps_llama3.py)")

    composition = {
        "vicuna_train_attack_counts": dict(train_counts),
        "vicuna_test_attack_counts": dict(test_counts),
        "vicuna_n_unique_attack_methods": len(set(train_methods + test_methods)),
    }

    # ─────────────────────────────────────────────────────────────────────
    # T1: Vicuna BALANCED
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("T1 — Vicuna BALANCED (subsample to 253 ben + 253 atk)")
    print("=" * 78)

    vic_bal_ben, vic_bal_atk = subsample_imbalanced(
        vicuna["X_tr_ben"], vicuna["X_tr_atk"],
        n_ben=253, n_atk=253, seed=args.seed,
    )
    t1 = evaluate_config(
        "T1. Vicuna BALANCED 253+253",
        vic_bal_ben, vic_bal_atk,
        vicuna["X_te_ben"], vicuna["X_te_atk"],
        epochs=args.epochs, device=device, seed=args.seed,
    )

    # ─────────────────────────────────────────────────────────────────────
    # T2: Vicuna BALANCED + drop shallow layers
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("T2 — Vicuna BALANCED + drop shallow layers")
    print("=" * 78)

    deep_idx = [VICUNA_LAYERS.index(L) for L in VICUNA_LAYERS_DEEP]
    t2 = evaluate_config(
        "T2. Vicuna BALANCED 253+253, deep layers only",
        vic_bal_ben[:, deep_idx, :], vic_bal_atk[:, deep_idx, :],
        vicuna["X_te_ben"][:, deep_idx, :],
        vicuna["X_te_atk"][:, deep_idx, :],
        epochs=args.epochs, device=device, seed=args.seed,
    )

    # ─────────────────────────────────────────────────────────────────────
    # T3: Llama-3 with VICUNA-STYLE imbalance
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("T3 — Llama-3 with VICUNA-STYLE imbalance (416 ben + 253 atk)")
    print("=" * 78)

    llama_im_ben, llama_im_atk = subsample_imbalanced(
        llama["X_tr_ben"], llama["X_tr_atk"],
        n_ben=416, n_atk=253, seed=args.seed,
    )
    t3 = evaluate_config(
        "T3. Llama-3 416+253 (Vicuna-style imbalance)",
        llama_im_ben, llama_im_atk,
        llama["X_te_ben"], llama["X_te_atk"],
        epochs=args.epochs, device=device, seed=args.seed,
    )

    # ─────────────────────────────────────────────────────────────────────
    # T4: Llama-3 with EXTREME imbalance
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("T4 — Llama-3 with EXTREME imbalance (500 ben + 100 atk)")
    print("=" * 78)

    llama_ex_ben, llama_ex_atk = subsample_imbalanced(
        llama["X_tr_ben"], llama["X_tr_atk"],
        n_ben=500, n_atk=100, seed=args.seed,
    )
    t4 = evaluate_config(
        "T4. Llama-3 500+100 (extreme imbalance, ~5:1)",
        llama_ex_ben, llama_ex_atk,
        llama["X_te_ben"], llama["X_te_atk"],
        epochs=args.epochs, device=device, seed=args.seed,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Vicuna baseline (for reference)
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("REF — Vicuna BASELINE (full imbalanced 416+253) for reference")
    print("=" * 78)
    t_ref = evaluate_config(
        "REF. Vicuna BASELINE 416+253",
        vicuna["X_tr_ben"], vicuna["X_tr_atk"],
        vicuna["X_te_ben"], vicuna["X_te_atk"],
        epochs=args.epochs, device=device, seed=args.seed,
    )

    # ─────────────────────────────────────────────────────────────────────
    # T5: Per-attack breakdown on Vicuna baseline
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("T5 — Per-attack breakdown on Vicuna test set")
    print("=" * 78)

    # Use thresholds calibrated on training benign at 5% FPR
    train_eval_ben = vicuna["X_tr_ben"]
    test_y_ref = np.array(t_ref["test_y"])
    n_ben_te = int((test_y_ref == 0).sum())

    train_scores_hps_ref = score_via_hps(
        *([None] * 0),  # type: ignore[arg-type]
        # Need to retrain to get fresh scores; reuse t_ref's scores instead.
    ) if False else None
    # We use the test scores already produced
    test_scores_hps_ref = np.array(t_ref["test_scores_hps"])
    test_scores_c4_ref = np.array(t_ref["test_scores_c4"])
    benign_scores_hps = test_scores_hps_ref[:n_ben_te]
    benign_scores_c4 = test_scores_c4_ref[:n_ben_te]

    # Threshold at 5% FPR on test benign (proxy)
    thr_hps = float(np.percentile(benign_scores_hps, 95))
    thr_c4 = float(np.percentile(benign_scores_c4, 95))

    breakdown = per_attack_breakdown(
        t_ref["test_scores_hps"], t_ref["test_scores_c4"],
        t_ref["test_y"], vicuna["atk_methods_test"],
        threshold_hps=thr_hps, threshold_c4=thr_c4,
    )

    print(f"\n  Test thresholds at 5% FPR:")
    print(f"    HPS threshold: {thr_hps:.4f}")
    print(f"    C4  threshold: {thr_c4:.4f}")
    print()
    print(f"  {'Attack':<25s}  {'N':>4s}  {'HPS rate':>9s}  "
          f"{'C4 rate':>9s}  {'gap':>7s}")
    print("  " + "-" * 65)
    for m in sorted(breakdown.keys()):
        b = breakdown[m]
        gap = b["c4_detection_rate"] - b["hps_detection_rate"]
        print(f"  {str(m):<25s}  {b['n_total']:>4d}  "
              f"{b['hps_detection_rate']:>9.4f}  "
              f"{b['c4_detection_rate']:>9.4f}  "
              f"{gap:>+7.4f}")

    # ─────────────────────────────────────────────────────────────────────
    # SUMMARY + DIAGNOSIS
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"\n{'Config':<55s}  {'TrAUC':>7s}  {'TeAUC':>7s}  "
          f"{'TeTPR5':>7s}  {'C4TPR5':>7s}  {'Δ':>7s}")
    print("-" * 100)
    for r in [t_ref, t1, t2, t3, t4]:
        delta = r["hps_test_tpr5"] - r["c4_test_tpr5"]
        print(
            f"{r['name']:<55s}  "
            f"{r['hps_train_auroc']:>7.4f}  "
            f"{r['hps_test_auroc']:>7.4f}  "
            f"{r['hps_test_tpr5']:>7.4f}  "
            f"{r['c4_test_tpr5']:>7.4f}  "
            f"{delta:>+7.4f}"
        )

    print("\n" + "=" * 78)
    print("DIAGNOSIS")
    print("=" * 78)

    diagnoses = []

    # H_I: class imbalance hypothesis
    delta_balance = t1["hps_test_tpr5"] - t_ref["hps_test_tpr5"]
    if delta_balance > 0.05:
        diagnoses.append(
            f"IMBALANCE_MATTERS: Balanced Vicuna 253+253 improves HPS TPR5 "
            f"from {t_ref['hps_test_tpr5']:.3f} → {t1['hps_test_tpr5']:.3f} "
            f"(Δ +{delta_balance:.3f}). Class imbalance is part of the issue."
        )
    elif delta_balance < -0.05:
        diagnoses.append(
            f"BALANCE_HURTS: Balancing Vicuna actually decreases HPS TPR5 "
            f"({t_ref['hps_test_tpr5']:.3f} → {t1['hps_test_tpr5']:.3f}). "
            f"The extra benign samples were helping HPS."
        )
    else:
        diagnoses.append(
            f"IMBALANCE_NEUTRAL: Balancing Vicuna leaves HPS TPR5 within "
            f"{delta_balance:+.3f} of baseline. Class imbalance is NOT the "
            f"primary issue."
        )

    # H_K: imbalance is the cause vs Llama-3 robustness
    if t3["hps_test_tpr5"] > 0.95:
        diagnoses.append(
            f"LLAMA_ROBUST_TO_IMBALANCE: Llama-3 with Vicuna-style imbalance "
            f"(416+253) achieves HPS TPR5 = {t3['hps_test_tpr5']:.3f}. "
            f"Imbalance alone does NOT break HPS — it's something Vicuna-"
            f"specific."
        )
    else:
        diagnoses.append(
            f"LLAMA_ALSO_HURT: Llama-3 with imbalance drops to "
            f"{t3['hps_test_tpr5']:.3f}. Imbalance is genuinely a problem for "
            f"HPS in general."
        )

    if t4["hps_test_tpr5"] < 0.85:
        diagnoses.append(
            f"EXTREME_IMBALANCE_BREAKS: Llama-3 5:1 imbalance drops HPS to "
            f"{t4['hps_test_tpr5']:.3f}. Severe imbalance breaks HPS."
        )

    # Per-attack: are any attacks specifically failing?
    if breakdown:
        worst_attack = min(breakdown.keys(),
                            key=lambda m: breakdown[m]["hps_detection_rate"])
        worst_rate = breakdown[worst_attack]["hps_detection_rate"]
        c4_on_worst = breakdown[worst_attack]["c4_detection_rate"]
        if worst_rate < 0.7 and c4_on_worst > 0.9:
            diagnoses.append(
                f"PER_ATTACK_FAILURE: HPS specifically fails on attack "
                f"'{worst_attack}' (HPS rate {worst_rate:.3f}, C4 rate "
                f"{c4_on_worst:.3f}). HPS may have an attack-specific blind "
                f"spot."
            )

        # Count attacks where HPS is much worse than C4
        bad_attacks = [m for m in breakdown
                       if breakdown[m]["c4_detection_rate"] -
                          breakdown[m]["hps_detection_rate"] > 0.2]
        if len(bad_attacks) > 1:
            diagnoses.append(
                f"MULTIPLE_HARD_ATTACKS: HPS is >0.2 worse than C4 on "
                f"{len(bad_attacks)} attacks: {bad_attacks}"
            )

    # Final
    if t1["hps_test_tpr5"] >= t_ref["c4_test_tpr5"] - 0.02:
        diagnoses.append(
            f"BALANCING_CLOSES_GAP: With balanced Vicuna data, HPS reaches "
            f"{t1['hps_test_tpr5']:.3f}, comparable to C4 "
            f"({t_ref['c4_test_tpr5']:.3f}). The Vicuna failure was primarily "
            f"a class-imbalance artifact."
        )
    elif t2["hps_test_tpr5"] >= t_ref["c4_test_tpr5"] - 0.02:
        diagnoses.append(
            f"BALANCE_PLUS_DEEP_CLOSES_GAP: Balanced + deep layers gives HPS "
            f"{t2['hps_test_tpr5']:.3f}, near C4 "
            f"({t_ref['c4_test_tpr5']:.3f}). Combined fix works."
        )
    else:
        diagnoses.append(
            f"GAP_PERSISTS: Even with balancing and best layers, HPS only "
            f"reaches {max(t1['hps_test_tpr5'], t2['hps_test_tpr5']):.3f}, "
            f"vs C4's {t_ref['c4_test_tpr5']:.3f}. Some Vicuna-specific "
            f"property continues to disadvantage HPS — investigation should "
            f"focus on per-attack failures or mechanism beyond what we've "
            f"tested."
        )

    print()
    for d in diagnoses:
        print(f"  • {d}")
    print()

    # ─────────────────────────────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────────────────────────────
    output = {
        "config": {
            "device": device,
            "epochs": args.epochs,
            "seed": args.seed,
            "vicuna_layers": VICUNA_LAYERS,
            "vicuna_layers_deep": VICUNA_LAYERS_DEEP,
            "llama_layers": LLAMA_LAYERS,
        },
        "composition": composition,
        "results": {
            "REF_vicuna_baseline": t_ref,
            "T1_vicuna_balanced": t1,
            "T2_vicuna_balanced_deep": t2,
            "T3_llama_imbalanced_vicuna_style": t3,
            "T4_llama_extreme_imbalance": t4,
        },
        "per_attack_breakdown": breakdown,
        "diagnoses": diagnoses,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        # Strip large arrays before saving (test_scores_hps/c4/y are bulky)
        slim = json.loads(json.dumps(output, default=_np_default))
        for r_key in slim["results"]:
            for big_key in ["test_scores_hps", "test_scores_c4", "test_y"]:
                if big_key in slim["results"][r_key]:
                    del slim["results"][r_key][big_key]
        json.dump(slim, f, indent=2)
    print(f"Saved results to {output_path}")

    # ─────────────────────────────────────────────────────────────────────
    # Plots
    # ─────────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    figdir = Path("results/figs")
    figdir.mkdir(parents=True, exist_ok=True)

    # Plot 1: bar comparison of all configs
    configs = [t_ref, t1, t2, t3, t4]
    labels = [r["name"][:38] for r in configs]
    hps_tpr = [r["hps_test_tpr5"] for r in configs]
    c4_tpr = [r["c4_test_tpr5"] for r in configs]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5))
    w = 0.35
    ax.bar(x - w/2, hps_tpr, width=w, label="HPS TPR5", color="tab:blue")
    ax.bar(x + w/2, c4_tpr, width=w, label="C4 TPR5", color="tab:orange",
           alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Test TPR @ 5% FPR")
    ax.set_title("Imbalance + diversity test: HPS vs C4")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim([0, 1.1])
    for xi, v in enumerate(hps_tpr):
        ax.text(xi - w/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
    for xi, v in enumerate(c4_tpr):
        ax.text(xi + w/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=7,
                alpha=0.7)
    plt.tight_layout()
    fig.savefig(figdir / "vicuna_imbalance_test.png", dpi=120)
    plt.close(fig)
    print(f"  saved {figdir}/vicuna_imbalance_test.png")

    # Plot 2: per-attack breakdown
    if breakdown:
        attacks = sorted(breakdown.keys())
        hps_rates = [breakdown[m]["hps_detection_rate"] for m in attacks]
        c4_rates = [breakdown[m]["c4_detection_rate"] for m in attacks]
        x = np.arange(len(attacks))
        fig, ax = plt.subplots(figsize=(10, 4.5))
        w = 0.35
        ax.bar(x - w/2, hps_rates, width=w, label="HPS detection rate",
               color="tab:blue")
        ax.bar(x + w/2, c4_rates, width=w, label="C4 detection rate",
               color="tab:orange", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([str(a) for a in attacks],
                           rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Detection rate (>5%FPR threshold)")
        ax.set_title("Per-attack detection rate on Vicuna test set")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        ax.set_ylim([0, 1.1])
        for xi, v in enumerate(hps_rates):
            ax.text(xi - w/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
        for xi, v in enumerate(c4_rates):
            ax.text(xi + w/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=7,
                    alpha=0.7)
        plt.tight_layout()
        fig.savefig(figdir / "vicuna_per_attack.png", dpi=120)
        plt.close(fig)
        print(f"  saved {figdir}/vicuna_per_attack.png")


if __name__ == "__main__":
    main()
