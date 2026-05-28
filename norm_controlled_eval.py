"""
norm_controlled_eval.py — Re-evaluate HPS, HPS-Euclidean, and C4 after
explicitly removing the activation-norm shortcut.

If diagnose_norm_confound.py shows that the norm confound is fundamental
(not just a chat-template bug), we need a way to evaluate methods AFTER
the shortcut is removed. This script applies several norm-controls and
re-runs the comparison:

  Control 1: Per-prompt L2 normalization     — remove activation magnitude
  Control 2: Standardize by training mean    — remove additive bias
  Control 3: Standardize + unit-norm          — both removed

For each control, train HPS/HPS-Euclidean/C4 on the controlled features,
report AUROC, TPR, and per-attack rates. Compare to uncontrolled baseline.

If saturation persists after norm control: real signal exists.
If saturation evaporates: the entire benchmark was norm-driven.

Outputs:
  results/norm_controlled_eval.json
  results/figs/norm_controlled_*.png

Usage:
  python norm_controlled_eval.py \\
      --cache results/llama3_activations_cache_diverse_clean.npz
  python norm_controlled_eval.py \\
      --cache results/vicuna_activations_cache_diverse.npz \\
      --hidden_dim 5120 --layers 0 2 22 31 35 39
"""

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from hps_core import (
    LorentzProjection,
    contrastive_loss,
    extract_trajectory_features,
)


# ---------------------------------------------------------------------------
# Cache loader (handles both formats)
# ---------------------------------------------------------------------------

def to_lasttoken(arr):
    out = []
    for l in sorted(arr.keys()):
        t = arr[l]
        out.append(t[-1] if t.ndim == 2 else t)
    return np.stack(out, axis=0)


def load_cache(cache_path, layers):
    cache = np.load(cache_path, allow_pickle=True)
    keys = list(cache.keys())

    if "X_benign" in keys:
        X_ben = cache["X_benign"]
        X_atk = cache["X_attack"]
        atk_methods = (cache["attack_methods"].tolist()
                        if "attack_methods" in keys else None)

        if "train_idx_benign" in keys:
            tri_b = cache["train_idx_benign"]
            tei_b = cache["test_idx_benign"]
            tri_a = cache["train_idx_attack"]
            tei_a = cache["test_idx_attack"]
        else:
            rng = np.random.RandomState(42)
            ben_idx = rng.permutation(len(X_ben))
            atk_idx = rng.permutation(len(X_atk))
            tri_b = ben_idx[:int(0.8 * len(ben_idx))]
            tei_b = ben_idx[int(0.8 * len(ben_idx)):]
            tri_a = atk_idx[:int(0.8 * len(atk_idx))]
            tei_a = atk_idx[int(0.8 * len(atk_idx)):]

        return {
            "X_tr_ben": X_ben[tri_b], "X_tr_atk": X_atk[tri_a],
            "X_te_ben": X_ben[tei_b], "X_te_atk": X_atk[tei_a],
            "test_atk_methods": ([atk_methods[i] for i in tei_a]
                                  if atk_methods else None),
        }

    if "hs_train_ben" in keys:
        hs_tr_ben = cache["hs_train_ben"]
        hs_tr_atk = cache["hs_train_atk"]
        hs_te_ben = cache["hs_test_ben"]
        hs_te_atk = cache["hs_test_atk"]

        def to_arr(hs_list):
            arrs = [to_lasttoken(h) for h in hs_list]
            if not arrs:
                return np.empty((0, len(layers), 0), dtype=np.float32)
            return np.stack(arrs, axis=0)

        side = str(cache_path).replace(".npz", "_test_methods.json")
        test_methods = None
        if os.path.exists(side):
            with open(side) as f:
                test_methods = json.load(f).get("test_atk_methods")

        return {
            "X_tr_ben": to_arr(hs_tr_ben), "X_tr_atk": to_arr(hs_tr_atk),
            "X_te_ben": to_arr(hs_te_ben), "X_te_atk": to_arr(hs_te_atk),
            "test_atk_methods": test_methods,
        }

    raise ValueError(f"Unknown cache format. Keys: {keys}")


# ---------------------------------------------------------------------------
# Norm-control transforms
# ---------------------------------------------------------------------------

def l2_normalize(X, eps=1e-7):
    """Per-prompt L2 normalize: each (n_layers, d) becomes unit norm per
       layer."""
    norms = np.linalg.norm(X, axis=-1, keepdims=True) + eps
    return X / norms


def standardize_per_layer(X_tr, X_te_ben, X_te_atk):
    """Standardize per layer using training-set statistics. Returns
       transformed (X_tr, X_te_ben, X_te_atk)."""
    mean = X_tr.mean(axis=0)  # (n_layers, d)
    std = X_tr.std(axis=0) + 1e-7
    return (
        (X_tr - mean) / std,
        (X_te_ben - mean) / std,
        (X_te_atk - mean) / std,
    )


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def train_c4(X_tr_ben, X_tr_atk, scaler=True):
    f_tr = np.concatenate([X_tr_ben.mean(axis=1), X_tr_atk.mean(axis=1)])
    y_tr = np.array([0] * len(X_tr_ben) + [1] * len(X_tr_atk))
    sc = StandardScaler() if scaler else None
    if sc is not None:
        f_tr = sc.fit_transform(f_tr)
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(f_tr, y_tr)
    return sc, clf


def score_c4(sc, clf, X):
    feats = X.mean(axis=1)
    if sc is not None:
        feats = sc.transform(feats)
    return clf.predict_proba(feats)[:, 1]


def train_hps(X_tr_ben, X_tr_atk, hidden_dim, n_layers, kappa=0.1,
               proj_dim=64, epochs=50, lr=1e-3, weight_decay=1e-5,
               device="cuda"):
    proj = LorentzProjection(
        d_in=hidden_dim, n_layers=n_layers,
        d_proj=proj_dim, k=kappa,
    ).to(device)
    optimizer = torch.optim.Adam(
        proj.parameters(), lr=lr, weight_decay=weight_decay,
    )
    X_ben_t = torch.from_numpy(np.asarray(X_tr_ben, dtype=np.float32)).to(device)
    X_atk_t = torch.from_numpy(np.asarray(X_tr_atk, dtype=np.float32)).to(device)

    proj.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        z_ben = proj(X_ben_t)
        z_atk = proj(X_atk_t)
        loss = contrastive_loss(z_ben, z_atk, kappa=kappa)
        loss.backward()
        optimizer.step()

    proj.eval()
    with torch.no_grad():
        z_tr_ben = proj(X_ben_t).cpu().numpy()
        z_tr_atk = proj(X_atk_t).cpu().numpy()
    f_tr_ben = np.array([extract_trajectory_features(z, kappa=kappa)
                          for z in z_tr_ben])
    f_tr_atk = np.array([extract_trajectory_features(z, kappa=kappa)
                          for z in z_tr_atk])
    f_tr = np.concatenate([f_tr_ben, f_tr_atk])
    y_tr = np.array([0] * len(f_tr_ben) + [1] * len(f_tr_atk))
    sc = StandardScaler()
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(sc.fit_transform(f_tr), y_tr)
    return proj, sc, clf


def score_hps(proj, sc, clf, X, kappa=0.1, device="cuda"):
    proj.eval()
    X_t = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
    with torch.no_grad():
        z = proj(X_t).cpu().numpy()
    f = np.array([extract_trajectory_features(zi, kappa=kappa) for zi in z])
    return clf.predict_proba(sc.transform(f))[:, 1]


# ---------------------------------------------------------------------------
# Norm-only baseline
# ---------------------------------------------------------------------------

def train_norm_only(X_tr_ben, X_tr_atk):
    """Trivial classifier using ONLY per-layer L2 norms (6 features)."""
    n_tr = np.concatenate([
        np.linalg.norm(X_tr_ben, axis=2),
        np.linalg.norm(X_tr_atk, axis=2),
    ])
    y_tr = np.array([0] * len(X_tr_ben) + [1] * len(X_tr_atk))
    sc = StandardScaler()
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(sc.fit_transform(n_tr), y_tr)
    return sc, clf


def score_norm_only(sc, clf, X):
    n = np.linalg.norm(X, axis=2)
    return clf.predict_proba(sc.transform(n))[:, 1]


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def eval_metrics(s_ben, s_atk, target_fpr=0.05):
    threshold = float(np.quantile(s_ben, 1 - target_fpr))
    return {
        "auroc": float(roc_auc_score(
            np.array([0] * len(s_ben) + [1] * len(s_atk)),
            np.concatenate([s_ben, s_atk]),
        )),
        "tpr": float((s_atk > threshold).mean()),
        "fpr": float((s_ben > threshold).mean()),
        "threshold": threshold,
    }


def run_condition(condition_name, X_tr_ben, X_tr_atk, X_te_ben, X_te_atk,
                   hidden_dim, n_layers, kappa=0.1, epochs=50, device="cuda",
                   target_fpr=0.05):
    """Train and evaluate HPS, C4, and norm-only on the given features."""
    print(f"\n  Condition: {condition_name}")
    print(f"    Train: {len(X_tr_ben)} ben + {len(X_tr_atk)} atk")
    print(f"    Test:  {len(X_te_ben)} ben + {len(X_te_atk)} atk")

    # C4
    sc4, c4 = train_c4(X_tr_ben, X_tr_atk)
    s_te_ben_c4 = score_c4(sc4, c4, X_te_ben)
    s_te_atk_c4 = score_c4(sc4, c4, X_te_atk)
    m_c4 = eval_metrics(s_te_ben_c4, s_te_atk_c4, target_fpr=target_fpr)

    # Norm-only
    sn, cn = train_norm_only(X_tr_ben, X_tr_atk)
    s_te_ben_n = score_norm_only(sn, cn, X_te_ben)
    s_te_atk_n = score_norm_only(sn, cn, X_te_atk)
    m_norm = eval_metrics(s_te_ben_n, s_te_atk_n, target_fpr=target_fpr)

    # HPS (with try/except in case Lorentz constraint fails on normalized)
    try:
        proj, sh, ch = train_hps(
            X_tr_ben, X_tr_atk, hidden_dim, n_layers,
            kappa=kappa, epochs=epochs, device=device,
        )
        s_te_ben_h = score_hps(proj, sh, ch, X_te_ben, kappa=kappa, device=device)
        s_te_atk_h = score_hps(proj, sh, ch, X_te_atk, kappa=kappa, device=device)
        m_hps = eval_metrics(s_te_ben_h, s_te_atk_h, target_fpr=target_fpr)
    except Exception as e:
        print(f"    HPS training failed: {e}")
        m_hps = {"auroc": float("nan"), "tpr": float("nan"),
                  "fpr": float("nan"), "threshold": float("nan")}

    print(f"    Method      AUROC      TPR@5%FPR")
    print(f"    norm-only   {m_norm['auroc']:.4f}   {m_norm['tpr']:.4f}")
    print(f"    C4          {m_c4['auroc']:.4f}   {m_c4['tpr']:.4f}")
    print(f"    HPS         {m_hps['auroc']:.4f}   {m_hps['tpr']:.4f}")

    return {
        "condition": condition_name,
        "norm_only": m_norm,
        "C4": m_c4,
        "HPS": m_hps,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results, output_path, target_fpr=0.05):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    conditions = [r["condition"] for r in results]
    methods = ["norm_only", "C4", "HPS"]
    colors = {"norm_only": "#95a5a6", "C4": "#2ecc71", "HPS": "#9b59b6"}

    aurocs = {m: [r[m]["auroc"] for r in results] for m in methods}
    tprs = {m: [r[m]["tpr"] for r in results] for m in methods}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(conditions))
    width = 0.27

    for ax, vals, ylabel in [
        (axes[0], aurocs, "AUROC"),
        (axes[1], tprs, f"TPR @ {target_fpr*100:.0f}% FPR"),
    ]:
        for i, m in enumerate(methods):
            offset = (i - 1) * width
            bars = ax.bar(x + offset, vals[m], width, label=m,
                           color=colors[m], alpha=0.85)
            for b, v in zip(bars, vals[m]):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                            f"{v:.3f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(y=0.5, color="black", linewidth=0.5, linestyle="--",
                    alpha=0.5, label="chance")

    plt.suptitle("Effect of norm-control on classifier performance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output",
                        default="results/norm_controlled_eval.json")
    parser.add_argument("--layers", type=int, nargs="+",
                        default=[0, 2, 17, 24, 28, 31])
    parser.add_argument("--hidden_dim", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--target_fpr", type=float, default=0.05)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    figs_dir = "results/figs"
    os.makedirs(figs_dir, exist_ok=True)

    print("=" * 70)
    print("NORM-CONTROLLED EVALUATION")
    print("=" * 70)
    print(f"  Cache:      {args.cache}")
    print(f"  Layers:     {args.layers}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Device:     {device}")
    print()

    cache = load_cache(args.cache, args.layers)
    n_layers = len(args.layers)

    print(f"  Train: {len(cache['X_tr_ben'])} ben + "
          f"{len(cache['X_tr_atk'])} atk")
    print(f"  Test:  {len(cache['X_te_ben'])} ben + "
          f"{len(cache['X_te_atk'])} atk")

    all_results = []

    # ---- BASELINE: no norm control ----
    print("\n" + "=" * 70)
    print("BASELINE: raw activations (no norm control)")
    print("=" * 70)
    res = run_condition(
        "baseline_raw",
        cache["X_tr_ben"], cache["X_tr_atk"],
        cache["X_te_ben"], cache["X_te_atk"],
        hidden_dim=args.hidden_dim, n_layers=n_layers,
        epochs=args.epochs, device=device, target_fpr=args.target_fpr,
    )
    all_results.append(res)

    # ---- CONTROL 1: per-prompt L2 normalization ----
    print("\n" + "=" * 70)
    print("CONTROL 1: per-prompt L2 normalization")
    print("=" * 70)
    print("  Each (n_layers, d) → unit norm per layer. Removes magnitude.")
    Xtr_ben_n = l2_normalize(cache["X_tr_ben"])
    Xtr_atk_n = l2_normalize(cache["X_tr_atk"])
    Xte_ben_n = l2_normalize(cache["X_te_ben"])
    Xte_atk_n = l2_normalize(cache["X_te_atk"])

    res = run_condition(
        "L2_normalized",
        Xtr_ben_n, Xtr_atk_n, Xte_ben_n, Xte_atk_n,
        hidden_dim=args.hidden_dim, n_layers=n_layers,
        epochs=args.epochs, device=device, target_fpr=args.target_fpr,
    )
    all_results.append(res)

    # ---- CONTROL 2: standardize per layer ----
    print("\n" + "=" * 70)
    print("CONTROL 2: standardize per-layer using train-mean/std")
    print("=" * 70)
    print("  Removes additive bias but keeps directional signal.")
    Xtr_full = np.concatenate([cache["X_tr_ben"], cache["X_tr_atk"]], axis=0)
    Xtr_ben_s, Xte_ben_s, _ = standardize_per_layer(
        Xtr_full, cache["X_te_ben"], cache["X_tr_atk"][:1],
    )
    # Apply same transform to all
    mean = Xtr_full.mean(axis=0)
    std = Xtr_full.std(axis=0) + 1e-7
    Xtr_ben_s = (cache["X_tr_ben"] - mean) / std
    Xtr_atk_s = (cache["X_tr_atk"] - mean) / std
    Xte_ben_s = (cache["X_te_ben"] - mean) / std
    Xte_atk_s = (cache["X_te_atk"] - mean) / std

    res = run_condition(
        "standardized",
        Xtr_ben_s, Xtr_atk_s, Xte_ben_s, Xte_atk_s,
        hidden_dim=args.hidden_dim, n_layers=n_layers,
        epochs=args.epochs, device=device, target_fpr=args.target_fpr,
    )
    all_results.append(res)

    # ---- CONTROL 3: standardize THEN unit-norm ----
    print("\n" + "=" * 70)
    print("CONTROL 3: standardize + unit-norm")
    print("=" * 70)
    print("  Most aggressive control: removes both bias and magnitude.")
    Xtr_ben_sn = l2_normalize(Xtr_ben_s)
    Xtr_atk_sn = l2_normalize(Xtr_atk_s)
    Xte_ben_sn = l2_normalize(Xte_ben_s)
    Xte_atk_sn = l2_normalize(Xte_atk_s)

    res = run_condition(
        "std+L2norm",
        Xtr_ben_sn, Xtr_atk_sn, Xte_ben_sn, Xte_atk_sn,
        hidden_dim=args.hidden_dim, n_layers=n_layers,
        epochs=args.epochs, device=device, target_fpr=args.target_fpr,
    )
    all_results.append(res)

    # ---- Diagnosis ----
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    baseline_norm_auroc = all_results[0]["norm_only"]["auroc"]
    l2_norm_auroc = all_results[1]["norm_only"]["auroc"]
    std_norm_auroc = all_results[2]["norm_only"]["auroc"]

    baseline_c4 = all_results[0]["C4"]["tpr"]
    l2_c4 = all_results[1]["C4"]["tpr"]
    std_c4 = all_results[2]["C4"]["tpr"]
    sn_c4 = all_results[3]["C4"]["tpr"]

    print(f"\n  Norm-only AUROC across conditions:")
    print(f"    baseline:        {baseline_norm_auroc:.4f}")
    print(f"    L2-normalized:   {l2_norm_auroc:.4f}")
    print(f"    standardized:    {std_norm_auroc:.4f}")
    print(f"    std + L2-norm:   "
          f"{all_results[3]['norm_only']['auroc']:.4f}")

    print(f"\n  C4 TPR @ 5%FPR across conditions:")
    print(f"    baseline:        {baseline_c4:.4f}")
    print(f"    L2-normalized:   {l2_c4:.4f}")
    print(f"    standardized:    {std_c4:.4f}")
    print(f"    std + L2-norm:   {sn_c4:.4f}")

    if l2_norm_auroc < 0.6 and l2_c4 > 0.85:
        verdict = "REAL_SIGNAL"
        explanation = (
            "Norm confound disappears with L2-normalization, but C4 still "
            "achieves high TPR. Real semantic signal exists beyond norm."
        )
    elif l2_c4 < 0.6:
        verdict = "FULLY_NORM_DRIVEN"
        explanation = (
            "C4 detection collapses when norm is removed. The benchmark is "
            "fully driven by activation magnitudes — there is no semantic "
            "signal beyond norm."
        )
    elif l2_c4 > 0.7:
        verdict = "PARTIALLY_NORM_DRIVEN"
        explanation = (
            "C4 detection drops but stays above chance after norm removal. "
            "Some semantic signal remains; norm explains a large fraction "
            "of the saturation."
        )
    else:
        verdict = "AMBIGUOUS"
        explanation = "Mixed results — inspect per-condition numbers."

    print(f"\n  Verdict: {verdict}")
    print(f"  {explanation}")

    output = {
        "config": vars(args),
        "results": all_results,
        "verdict": verdict,
        "verdict_explanation": explanation,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {args.output}")

    plot_results(all_results,
                  os.path.join(figs_dir, "norm_controlled_eval.png"),
                  target_fpr=args.target_fpr)


if __name__ == "__main__":
    main()
