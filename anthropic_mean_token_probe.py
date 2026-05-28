"""
anthropic_mean_token_probe.py — Faithful reproduction of Anthropic's
mean-token probe (Cunningham et al., 2025) for jailbreak detection.

This is the PUBLISHED method, not a variant. It serves as our principal
linear-probe baseline because:
  1. It's directly cited as published prior art
  2. Reviewers can't object to a faithful reproduction
  3. It enables a 3-way comparison: C4 (cross-layer) vs MTP (cross-token)
  4. The aggregation-axis ablation becomes a real finding

Anthropic's exact specification (from blog post and paper):
  - Single layer linear probe
  - Mean-pool across ALL token positions in the sequence
  - LogisticRegression with L2 regularization, lambda=1e4 for mean-token
  - Standardize activations (mean=0, std=1) before regression
  - "Linear probe quality remained consistently high right up until
    the final layer" — they don't specify a single best layer

Output:
  results/anthropic_mtp_eval.json
  results/figs/mtp_vs_c4_vs_hps.png

Usage:
  python anthropic_mean_token_probe.py \\
      --cache results/llama3_activations_cache_diverse_fixed.npz \\
      --layers 0 2 17 24 28 31

Note: Requires the cache to contain FULL SEQUENCE activations
(Llama-3-style dict cache). Vicuna-style array cache only has
last-token activations and will not work with this script.
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
# Anthropic's exact recipe
# ---------------------------------------------------------------------------

ANTHROPIC_L2_LAMBDA_MEAN_TOKEN = 1e4  # from Appendix D
ANTHROPIC_L2_LAMBDA_LAST_TOKEN = 1e2  # from Appendix D


def mean_token_features(hs_list, layer_idx):
    """
    Anthropic's mean-token probe input: mean over all token positions
    at a single layer.

    Args:
        hs_list: list of dicts {layer_idx -> (T, d) array}
        layer_idx: which layer to probe

    Returns:
        (N, d) array of mean-pooled features
    """
    out = []
    for hs in hs_list:
        if layer_idx not in hs:
            raise KeyError(f"Layer {layer_idx} not in cache "
                            f"(have {list(hs.keys())})")
        t = hs[layer_idx]
        if t.ndim == 1:
            # Already last-token only — can't compute mean-over-tokens
            raise ValueError(
                f"Cache stores only last-token activations (shape {t.shape}). "
                f"Mean-token probe requires FULL SEQUENCE storage.")
        out.append(t.mean(axis=0))  # mean over T → (d,)
    return np.stack(out, axis=0)


def last_token_features(hs_list, layer_idx):
    """C4-style features at a single layer (last-token only)."""
    out = []
    for hs in hs_list:
        t = hs[layer_idx]
        if t.ndim == 2:
            out.append(t[-1])
        else:
            out.append(t)
    return np.stack(out, axis=0)


def cross_layer_last_token(hs_list, layers):
    """C4 features: mean over layers' last tokens."""
    out = []
    for hs in hs_list:
        per_layer = []
        for l in layers:
            t = hs[l]
            per_layer.append(t[-1] if t.ndim == 2 else t)
        out.append(np.stack(per_layer, axis=0).mean(axis=0))  # (d,)
    return np.stack(out, axis=0)


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def load_dict_cache(cache_path):
    """Load Llama-3 style dict cache. Required for mean-token probe."""
    cache = np.load(cache_path, allow_pickle=True)
    keys = list(cache.keys())
    if "hs_train_ben" not in keys:
        raise ValueError(
            f"Cache does not contain full-sequence activations. "
            f"Need Llama-3 dict format (hs_train_ben, hs_train_atk, ...). "
            f"Got keys: {keys}"
        )
    return {
        "hs_train_ben": list(cache["hs_train_ben"]),
        "hs_train_atk": list(cache["hs_train_atk"]),
        "hs_test_ben": list(cache["hs_test_ben"]),
        "hs_test_atk": list(cache["hs_test_atk"]),
    }


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def train_lr_probe(X_tr, y_tr, l2_lambda):
    """
    Anthropic's recipe: standardize, then LR with specified L2 regularization.
    L2 lambda corresponds to sklearn C = 1/lambda.
    """
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    clf = LogisticRegression(
        C=1.0 / l2_lambda,  # sklearn C is inverse of L2 strength
        max_iter=2000,
        random_state=42,
    )
    clf.fit(X_tr_s, y_tr)
    return sc, clf


def score_probe(sc, clf, X):
    return clf.predict_proba(sc.transform(X))[:, 1]


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


# ---------------------------------------------------------------------------
# HPS (for direct comparison) — uses last-token features across all layers
# ---------------------------------------------------------------------------

def train_hps(X_tr_ben, X_tr_atk, hidden_dim, n_layers, kappa=0.1,
               proj_dim=64, epochs=50, lr=1e-3, weight_decay=1e-5,
               device="cuda"):
    proj = LorentzProjection(
        hidden_dim=hidden_dim, n_layers=n_layers,
        proj_dim=proj_dim, kappa=kappa,
    ).to(device)
    optimizer = torch.optim.Adam(
        proj.parameters(), lr=lr, weight_decay=weight_decay,
    )
    Xb = torch.from_numpy(np.asarray(X_tr_ben, dtype=np.float32)).to(device)
    Xa = torch.from_numpy(np.asarray(X_tr_atk, dtype=np.float32)).to(device)

    proj.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        loss = contrastive_loss(proj(Xb), proj(Xa), kappa=kappa)
        loss.backward()
        optimizer.step()

    proj.eval()
    with torch.no_grad():
        z_b = proj(Xb).cpu().numpy()
        z_a = proj(Xa).cpu().numpy()

    f_b = np.array([extract_trajectory_features(z, kappa=kappa) for z in z_b])
    f_a = np.array([extract_trajectory_features(z, kappa=kappa) for z in z_a])

    sc = StandardScaler()
    clf = LogisticRegression(max_iter=2000, random_state=42)
    f = np.concatenate([f_b, f_a])
    y = np.array([0] * len(f_b) + [1] * len(f_a))
    clf.fit(sc.fit_transform(f), y)
    return proj, sc, clf


def score_hps(proj, sc, clf, X, kappa=0.1, device="cuda"):
    Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
    proj.eval()
    with torch.no_grad():
        z = proj(Xt).cpu().numpy()
    f = np.array([extract_trajectory_features(zi, kappa=kappa) for zi in z])
    return clf.predict_proba(sc.transform(f))[:, 1]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(results, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    methods = []
    aurocs = []
    tprs = []

    # Per-layer MTP
    for entry in results["mtp_per_layer"]:
        methods.append(f"MTP-L{entry['layer']}")
        aurocs.append(entry["auroc"])
        tprs.append(entry["tpr"])

    # C4
    methods.append("C4 (cross-layer)")
    aurocs.append(results["c4"]["auroc"])
    tprs.append(results["c4"]["tpr"])

    # HPS
    methods.append("HPS (262K params)")
    aurocs.append(results["hps"]["auroc"])
    tprs.append(results["hps"]["tpr"])

    x = np.arange(len(methods))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # AUROC
    bars = axes[0].bar(x, aurocs, color=["#3498db"] * (len(methods) - 2)
                        + ["#2ecc71", "#9b59b6"], alpha=0.85)
    for b, v in zip(bars, aurocs):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.005,
                      f"{v:.3f}", ha="center", fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    axes[0].set_ylabel("AUROC")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_title("AUROC across linear-probe variants and HPS")

    # TPR
    bars = axes[1].bar(x, tprs, color=["#3498db"] * (len(methods) - 2)
                        + ["#2ecc71", "#9b59b6"], alpha=0.85)
    for b, v in zip(bars, tprs):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.005,
                      f"{v:.3f}", ha="center", fontsize=9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    axes[1].set_ylabel("TPR @ 5% FPR")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_title("TPR @ 5% FPR")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True,
                        help="Llama-3-style dict cache with full sequences")
    parser.add_argument("--output",
                        default="results/anthropic_mtp_eval.json")
    parser.add_argument("--layers", type=int, nargs="+",
                        default=[0, 2, 17, 24, 28, 31],
                        help="Layers to test for mean-token probe + use as "
                             "C4 input")
    parser.add_argument("--hidden_dim", type=int, default=4096)
    parser.add_argument("--target_fpr", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    figs_dir = "results/figs"
    os.makedirs(figs_dir, exist_ok=True)

    print("=" * 70)
    print("ANTHROPIC MEAN-TOKEN PROBE — faithful reproduction")
    print("=" * 70)
    print(f"  Cache:       {args.cache}")
    print(f"  Layers:      {args.layers}")
    print(f"  Hidden dim:  {args.hidden_dim}")
    print(f"  Target FPR:  {args.target_fpr}")
    print(f"  L2 lambda:   {ANTHROPIC_L2_LAMBDA_MEAN_TOKEN} (mean-token)")
    print(f"  Device:      {device}")
    print()

    # Load cache
    print("Loading cache (must be full-sequence dict format)...")
    cache = load_dict_cache(args.cache)
    print(f"  Train: {len(cache['hs_train_ben'])} ben + "
          f"{len(cache['hs_train_atk'])} atk")
    print(f"  Test:  {len(cache['hs_test_ben'])} ben + "
          f"{len(cache['hs_test_atk'])} atk")

    results = {
        "config": {
            "cache": args.cache,
            "layers": args.layers,
            "anthropic_l2_lambda": ANTHROPIC_L2_LAMBDA_MEAN_TOKEN,
            "target_fpr": args.target_fpr,
        },
        "mtp_per_layer": [],
        "c4": None,
        "hps": None,
    }

    y_tr = np.array([0] * len(cache["hs_train_ben"])
                    + [1] * len(cache["hs_train_atk"]))

    # -------------------------------------------------------------------
    # 1. Anthropic's MEAN-TOKEN PROBE — at each layer
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("1. Mean-Token Probe (Anthropic 2025) — per-layer evaluation")
    print("=" * 70)
    print(f"  {'Layer':>5}  {'AUROC':>8}  {'TPR@5%FPR':>12}  "
          f"{'TPR@1%FPR':>12}")

    for layer in args.layers:
        try:
            X_tr_ben = mean_token_features(cache["hs_train_ben"], layer)
            X_tr_atk = mean_token_features(cache["hs_train_atk"], layer)
            X_te_ben = mean_token_features(cache["hs_test_ben"], layer)
            X_te_atk = mean_token_features(cache["hs_test_atk"], layer)

            X_tr = np.concatenate([X_tr_ben, X_tr_atk])
            sc, clf = train_lr_probe(X_tr, y_tr,
                                      l2_lambda=ANTHROPIC_L2_LAMBDA_MEAN_TOKEN)
            s_te_ben = score_probe(sc, clf, X_te_ben)
            s_te_atk = score_probe(sc, clf, X_te_atk)

            m = eval_metrics(s_te_ben, s_te_atk, target_fpr=args.target_fpr)

            # Also TPR @ 1% FPR
            thr1 = float(np.quantile(s_te_ben, 0.99))
            tpr1 = float((s_te_atk > thr1).mean())
            m["tpr1"] = tpr1

            print(f"  {layer:>5}  {m['auroc']:>8.4f}  "
                  f"{m['tpr']:>12.4f}  {m['tpr1']:>12.4f}")

            entry = {"layer": layer, **m}
            results["mtp_per_layer"].append(entry)
        except Exception as e:
            print(f"  Layer {layer}: ERROR — {e}")

    # Pick best layer
    best = max(results["mtp_per_layer"], key=lambda r: r["auroc"])
    print(f"\n  Best mean-token probe layer: {best['layer']} "
          f"(AUROC={best['auroc']:.4f})")
    results["mtp_best_layer"] = best

    # -------------------------------------------------------------------
    # 2. C4 — cross-layer mean-pool last-token (our variant)
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. C4 — cross-layer mean-pool last-token (adaptation, our variant)")
    print("=" * 70)

    X_tr_ben_c4 = cross_layer_last_token(cache["hs_train_ben"], args.layers)
    X_tr_atk_c4 = cross_layer_last_token(cache["hs_train_atk"], args.layers)
    X_te_ben_c4 = cross_layer_last_token(cache["hs_test_ben"], args.layers)
    X_te_atk_c4 = cross_layer_last_token(cache["hs_test_atk"], args.layers)

    X_tr_c4 = np.concatenate([X_tr_ben_c4, X_tr_atk_c4])
    sc_c4, clf_c4 = train_lr_probe(X_tr_c4, y_tr,
                                     l2_lambda=ANTHROPIC_L2_LAMBDA_LAST_TOKEN)
    s_te_ben_c4 = score_probe(sc_c4, clf_c4, X_te_ben_c4)
    s_te_atk_c4 = score_probe(sc_c4, clf_c4, X_te_atk_c4)
    m_c4 = eval_metrics(s_te_ben_c4, s_te_atk_c4, target_fpr=args.target_fpr)
    thr1 = float(np.quantile(s_te_ben_c4, 0.99))
    m_c4["tpr1"] = float((s_te_atk_c4 > thr1).mean())

    print(f"  AUROC = {m_c4['auroc']:.4f}, TPR@5%FPR = {m_c4['tpr']:.4f}, "
          f"TPR@1%FPR = {m_c4['tpr1']:.4f}")
    results["c4"] = m_c4

    # -------------------------------------------------------------------
    # 3. HPS — full hyperbolic projection (for context)
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. HPS (full hyperbolic projection)")
    print("=" * 70)

    # Prepare last-token activations (N, n_layers, d)
    def lt_arr(hs_list, layers):
        out = []
        for hs in hs_list:
            per = [hs[l][-1] if hs[l].ndim == 2 else hs[l] for l in layers]
            out.append(np.stack(per, axis=0))
        return np.stack(out, axis=0)

    X_tr_ben_lt = lt_arr(cache["hs_train_ben"], args.layers)
    X_tr_atk_lt = lt_arr(cache["hs_train_atk"], args.layers)
    X_te_ben_lt = lt_arr(cache["hs_test_ben"], args.layers)
    X_te_atk_lt = lt_arr(cache["hs_test_atk"], args.layers)

    print(f"  Training HPS for {args.epochs} epochs...")
    proj, sh, ch = train_hps(
        X_tr_ben_lt, X_tr_atk_lt,
        hidden_dim=args.hidden_dim, n_layers=len(args.layers),
        epochs=args.epochs, device=device,
    )

    s_te_ben_h = score_hps(proj, sh, ch, X_te_ben_lt, device=device)
    s_te_atk_h = score_hps(proj, sh, ch, X_te_atk_lt, device=device)
    m_hps = eval_metrics(s_te_ben_h, s_te_atk_h, target_fpr=args.target_fpr)
    thr1 = float(np.quantile(s_te_ben_h, 0.99))
    m_hps["tpr1"] = float((s_te_atk_h > thr1).mean())

    print(f"  AUROC = {m_hps['auroc']:.4f}, TPR@5%FPR = {m_hps['tpr']:.4f}, "
          f"TPR@1%FPR = {m_hps['tpr1']:.4f}")
    results["hps"] = m_hps

    # -------------------------------------------------------------------
    # 4. Comparison summary
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Method':<30s} {'Params':>8s} {'AUROC':>8s} "
          f"{'TPR@5%':>8s} {'TPR@1%':>8s}")
    print("-" * 70)
    print(f"{'MTP — best layer (Anthropic)':<30s} "
          f"{args.hidden_dim + 1:>8d} "
          f"{best['auroc']:>8.4f} {best['tpr']:>8.4f} "
          f"{best['tpr1']:>8.4f}")
    print(f"{'C4 — cross-layer (our variant)':<30s} "
          f"{args.hidden_dim + 1:>8d} "
          f"{m_c4['auroc']:>8.4f} {m_c4['tpr']:>8.4f} "
          f"{m_c4['tpr1']:>8.4f}")
    print(f"{'HPS — Lorentz contrastive':<30s} "
          f"{262213:>8d} "
          f"{m_hps['auroc']:>8.4f} {m_hps['tpr']:>8.4f} "
          f"{m_hps['tpr1']:>8.4f}")

    # Diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    mtp_vs_c4 = best["auroc"] - m_c4["auroc"]
    if abs(mtp_vs_c4) < 0.005:
        print(f"  MTP ≈ C4: aggregation axis (tokens vs layers) doesn't matter "
              f"(Δ AUROC = {mtp_vs_c4:+.4f}).")
        print("  → The simple linear probe is robust to design choices.")
    elif mtp_vs_c4 > 0.005:
        print(f"  MTP > C4: Anthropic's exact recipe outperforms our variant "
              f"(Δ AUROC = {mtp_vs_c4:+.4f}).")
        print("  → Our paper should use MTP as the primary baseline.")
    else:
        print(f"  C4 > MTP: cross-layer pooling helps "
              f"(Δ AUROC = {mtp_vs_c4:+.4f}).")
        print("  → Modest contribution: C4 outperforms Anthropic's mean-token "
              "by Δ.")

    mtp_vs_hps = best["auroc"] - m_hps["auroc"]
    if abs(mtp_vs_hps) < 0.005:
        print(f"\n  MTP ≈ HPS: 4097-param Anthropic probe matches 262K-param "
              f"hyperbolic method.")
        print("  → No advantage from learned hyperbolic geometry.")

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {args.output}")

    plot_comparison(results, os.path.join(figs_dir, "mtp_vs_c4_vs_hps.png"))


if __name__ == "__main__":
    main()
