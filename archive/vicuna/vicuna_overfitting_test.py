"""
vicuna_overfitting_test.py — Distinguish overfitting vs intrinsic failure.

Following the vicuna_diagnostic.py finding that HPS converges to LOWER loss
on Vicuna (0.60) but achieves WORSE test performance (TPR=0.81) than on
Llama-3 — classic overfitting — this script tests three concrete fixes
and one definitive control to attribute the failure correctly:

  Fix 1: DROP_SHALLOW       Vicuna shallow layers (0, 2) have eff_dim=1
                            and d′<0.8 — they are essentially noise. Drop
                            them. Test layers = [22, 31, 35, 39].

  Fix 2: REGULARIZATION     HPS overfits → add stronger weight decay,
                            smaller projection dim, and shorter training.

  Fix 3: SUBSAMPLE_LLAMA    The smoking-gun control: subsample Llama-3 to
                            match Vicuna's 669 training samples. If HPS
                            also overfits on subsampled Llama-3 → pure
                            data-scarcity issue. If HPS still works on
                            subsampled Llama-3 → there is a Vicuna-specific
                            issue.

  Combined: DROP_SHALLOW + REGULARIZATION

Each configuration trains HPS and reports:
  - Final training loss
  - Train AUROC, test AUROC, test TPR @ 5% FPR
  - Train-test gap (sign of overfitting)
  - C4 baseline on the same data (for reference)

Usage:
  python vicuna_overfitting_test.py
  python vicuna_overfitting_test.py --epochs 50
  python vicuna_overfitting_test.py --skip_llama_full   # skip slow full-data run
"""

import argparse
import json
import os
import sys
import warnings
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
from vicuna_diagnostic import load_cache  # reuse the dual-format loader

LLAMA_LAYERS_FULL = [0, 2, 17, 24, 28, 31]
VICUNA_LAYERS_FULL = [0, 2, 22, 31, 35, 39]
VICUNA_LAYERS_DEEP = [22, 31, 35, 39]


# ---------------------------------------------------------------------------
# Utils
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


def subsample_balanced(X_ben, X_atk, n_total, seed=42):
    """Return balanced subsample of n_total examples (n_total/2 each class)."""
    rng = np.random.RandomState(seed)
    n_per_class = n_total // 2
    n_ben = min(n_per_class, len(X_ben))
    n_atk = min(n_per_class, len(X_atk))
    ben_idx = rng.choice(len(X_ben), size=n_ben, replace=False)
    atk_idx = rng.choice(len(X_atk), size=n_atk, replace=False)
    return X_ben[ben_idx], X_atk[atk_idx]


def subset_layers(X, source_layers, target_layers):
    """Subset (N, n_source, d) array along the layer axis to keep target_layers."""
    layer_idx = [source_layers.index(L) for L in target_layers]
    return X[:, layer_idx, :]


# ---------------------------------------------------------------------------
# Training / scoring
# ---------------------------------------------------------------------------

def train_hps(X_train, y_train, n_layers, d_hidden,
              proj_dim=64, kappa_init=0.1,
              epochs=50, weight_decay=1e-5,
              seed=42, device="cpu"):
    """Train HPS Lorentz projection. Returns proj + final loss + loss history."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    proj = LorentzProjection(d_hidden, proj_dim, kappa_init,
                              n_layers=n_layers).to(device)
    proj.log_k.requires_grad = False
    opt = optim.Adam(
        [p for p in proj.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=weight_decay,
    )
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)
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
    """Train LR on train trajectory features; score X_eval."""
    feats_train = extract_trajectory_features(proj, X_train)
    feats_eval = extract_trajectory_features(proj, X_eval)
    sc = StandardScaler().fit(feats_train)
    clf = LogisticRegression(max_iter=2000, random_state=seed,
                             class_weight="balanced")
    clf.fit(sc.transform(feats_train), y_train)
    return clf.predict_proba(sc.transform(feats_eval))[:, 1]


def score_via_c4(X_tr_ben, X_tr_atk, X_te, seed=42):
    """C4 = mean-pool layers + LR. Returns scores on X_te."""
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
    return clf.predict_proba(sc.transform(X_te.mean(axis=1)))[:, 1]


def evaluate_config(name, X_tr_ben, X_tr_atk, X_te_ben, X_te_atk,
                     proj_dim=64, kappa_init=0.1,
                     epochs=50, weight_decay=1e-5,
                     seed=42, device="cpu"):
    """Full evaluate: train HPS + C4, report all metrics."""
    X_train = np.concatenate([X_tr_ben, X_tr_atk], axis=0)
    y_train = np.concatenate([
        np.zeros(len(X_tr_ben)),
        np.ones(len(X_tr_atk)),
    ])
    n_layers = X_train.shape[1]
    d_hidden = X_train.shape[2]

    print(f"\n  [{name}]  train: {len(X_tr_ben)}+{len(X_tr_atk)}={len(X_train)},  "
          f"n_layers={n_layers}, d={d_hidden}, "
          f"proj_dim={proj_dim}, epochs={epochs}, wd={weight_decay}")

    # HPS
    proj, losses = train_hps(
        X_train, y_train, n_layers, d_hidden,
        proj_dim=proj_dim, kappa_init=kappa_init,
        epochs=epochs, weight_decay=weight_decay,
        seed=seed, device=device,
    )

    # Train scores (for overfitting diagnosis)
    train_scores_hps = score_via_hps(proj, X_train, y_train, X_train, seed)
    test_eval = np.concatenate([X_te_ben, X_te_atk], axis=0)
    test_y = np.concatenate([
        np.zeros(len(X_te_ben)),
        np.ones(len(X_te_atk)),
    ])
    test_scores_hps = score_via_hps(proj, X_train, y_train, test_eval, seed)

    hps_train_auroc = auroc(y_train, train_scores_hps)
    hps_test_auroc = auroc(test_y, test_scores_hps)
    hps_test_tpr5 = tpr_at_fpr(test_y, test_scores_hps)

    # C4 reference
    test_scores_c4 = score_via_c4(X_tr_ben, X_tr_atk, test_eval, seed)
    c4_test_auroc = auroc(test_y, test_scores_c4)
    c4_test_tpr5 = tpr_at_fpr(test_y, test_scores_c4)

    print(f"    HPS  final_loss={losses[-1]:.4f}  "
          f"train_AUROC={hps_train_auroc:.4f}  "
          f"test_AUROC={hps_test_auroc:.4f}  "
          f"test_TPR={hps_test_tpr5:.4f}  "
          f"gap={hps_train_auroc - hps_test_auroc:+.4f}")
    print(f"    C4                         "
          f"                       test_AUROC={c4_test_auroc:.4f}  "
          f"test_TPR={c4_test_tpr5:.4f}")

    return {
        "name": name,
        "n_train": int(len(X_train)),
        "n_train_ben": int(len(X_tr_ben)),
        "n_train_atk": int(len(X_tr_atk)),
        "n_test": int(len(test_eval)),
        "n_layers": int(n_layers),
        "d_hidden": int(d_hidden),
        "proj_dim": int(proj_dim),
        "epochs": int(epochs),
        "weight_decay": float(weight_decay),
        "kappa_init": float(kappa_init),
        "final_loss": float(losses[-1]),
        "loss_history": losses,
        "hps_train_auroc": hps_train_auroc,
        "hps_test_auroc": hps_test_auroc,
        "hps_test_tpr5": hps_test_tpr5,
        "hps_train_test_gap": hps_train_auroc - hps_test_auroc,
        "c4_test_auroc": c4_test_auroc,
        "c4_test_tpr5": c4_test_tpr5,
        "hps_minus_c4_tpr5": hps_test_tpr5 - c4_test_tpr5,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama3_cache",
                        default="results/llama3_activations_cache.npz")
    parser.add_argument("--vicuna_cache",
                        default="results/vicuna_activations_cache.npz")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=50,
                        help="Default training epochs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output",
                        default="results/vicuna_overfitting_test.json")
    parser.add_argument("--skip_llama_full", action="store_true",
                        help="Skip the full-data Llama-3 baseline (slow)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print("VICUNA OVERFITTING TEST — three fixes + matched-data control")
    print("=" * 78)
    print(f"  Llama-3 cache: {args.llama3_cache}")
    print(f"  Vicuna cache:  {args.vicuna_cache}")
    print(f"  Device:        {device}")
    print(f"  Epochs:        {args.epochs}")
    print()

    # ── Load data ──
    print("Loading caches...")
    llama = load_cache(args.llama3_cache, LLAMA_LAYERS_FULL)
    vicuna = load_cache(args.vicuna_cache, VICUNA_LAYERS_FULL)
    print(f"  Llama-3 train: {llama['X_tr_ben'].shape[0]} ben + "
          f"{llama['X_tr_atk'].shape[0]} atk")
    print(f"  Vicuna  train: {vicuna['X_tr_ben'].shape[0]} ben + "
          f"{vicuna['X_tr_atk'].shape[0]} atk")
    print()

    # Vicuna data sizes (used for matched-N control)
    n_vic_ben = len(vicuna["X_tr_ben"])
    n_vic_atk = len(vicuna["X_tr_atk"])
    n_vic_total = n_vic_ben + n_vic_atk

    results = {}

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 1: Llama-3 controls (establishes how HPS behaves with matched N)
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("PHASE 1 — Llama-3 controls")
    print("=" * 78)

    if not args.skip_llama_full:
        results["A_llama_full_default"] = evaluate_config(
            "A. Llama-3 FULL data (default config)",
            llama["X_tr_ben"], llama["X_tr_atk"],
            llama["X_te_ben"], llama["X_te_atk"],
            proj_dim=64, epochs=args.epochs, weight_decay=1e-5,
            device=device,
        )

    # Subsample Llama-3 to match Vicuna size
    sub_ben, sub_atk = subsample_balanced(
        llama["X_tr_ben"], llama["X_tr_atk"],
        n_total=n_vic_total, seed=args.seed,
    )
    print(f"\n  Subsampled Llama-3: {len(sub_ben)} ben + {len(sub_atk)} atk "
          f"(matched to Vicuna ~669)")

    results["B_llama_matched_default"] = evaluate_config(
        "B. Llama-3 SUBSAMPLED to Vicuna size, default config",
        sub_ben, sub_atk,
        llama["X_te_ben"], llama["X_te_atk"],
        proj_dim=64, epochs=args.epochs, weight_decay=1e-5,
        device=device,
    )

    results["C_llama_matched_regularized"] = evaluate_config(
        "C. Llama-3 SUBSAMPLED + strong regularization",
        sub_ben, sub_atk,
        llama["X_te_ben"], llama["X_te_atk"],
        proj_dim=32, epochs=20, weight_decay=1e-3,
        device=device,
    )

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 2: Vicuna baseline + three fixes
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("PHASE 2 — Vicuna baseline and the three fixes")
    print("=" * 78)

    results["D_vicuna_baseline"] = evaluate_config(
        "D. Vicuna BASELINE (default config: 6 layers, 50 epochs, wd=1e-5)",
        vicuna["X_tr_ben"], vicuna["X_tr_atk"],
        vicuna["X_te_ben"], vicuna["X_te_atk"],
        proj_dim=64, epochs=args.epochs, weight_decay=1e-5,
        device=device,
    )

    # Fix 1: Drop shallow Vicuna layers
    deep_idx = [VICUNA_LAYERS_FULL.index(L) for L in VICUNA_LAYERS_DEEP]
    vic_tr_ben_deep = vicuna["X_tr_ben"][:, deep_idx, :]
    vic_tr_atk_deep = vicuna["X_tr_atk"][:, deep_idx, :]
    vic_te_ben_deep = vicuna["X_te_ben"][:, deep_idx, :]
    vic_te_atk_deep = vicuna["X_te_atk"][:, deep_idx, :]

    results["E_vicuna_drop_shallow"] = evaluate_config(
        "E. Vicuna FIX 1: drop shallow layers (use [22,31,35,39] only)",
        vic_tr_ben_deep, vic_tr_atk_deep,
        vic_te_ben_deep, vic_te_atk_deep,
        proj_dim=64, epochs=args.epochs, weight_decay=1e-5,
        device=device,
    )

    # Fix 2: Regularization (smaller proj, shorter training, more wd)
    results["F_vicuna_regularized"] = evaluate_config(
        "F. Vicuna FIX 2: regularize (proj=32, epochs=20, wd=1e-3)",
        vicuna["X_tr_ben"], vicuna["X_tr_atk"],
        vicuna["X_te_ben"], vicuna["X_te_atk"],
        proj_dim=32, epochs=20, weight_decay=1e-3,
        device=device,
    )

    # Fix 3: Combined (drop shallow + regularization)
    results["G_vicuna_combined"] = evaluate_config(
        "G. Vicuna FIX 1+2 COMBINED: drop shallow + regularize",
        vic_tr_ben_deep, vic_tr_atk_deep,
        vic_te_ben_deep, vic_te_atk_deep,
        proj_dim=32, epochs=20, weight_decay=1e-3,
        device=device,
    )

    # Bonus: very early stopping (epoch=10) on default Vicuna
    results["H_vicuna_early_stop"] = evaluate_config(
        "H. Vicuna BONUS: aggressive early stopping (epochs=10)",
        vicuna["X_tr_ben"], vicuna["X_tr_atk"],
        vicuna["X_te_ben"], vicuna["X_te_atk"],
        proj_dim=64, epochs=10, weight_decay=1e-5,
        device=device,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Summary table
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("SUMMARY TABLE")
    print("=" * 78)
    print(f"\n{'Config':<55s}  {'TrAUC':>7s}  {'TeAUC':>7s}  "
          f"{'TeTPR5':>7s}  {'gap':>7s}  {'C4_TPR5':>8s}")
    print("-" * 102)
    for key, r in results.items():
        print(
            f"{r['name']:<55s}  "
            f"{r['hps_train_auroc']:>7.4f}  "
            f"{r['hps_test_auroc']:>7.4f}  "
            f"{r['hps_test_tpr5']:>7.4f}  "
            f"{r['hps_train_test_gap']:>+7.4f}  "
            f"{r['c4_test_tpr5']:>8.4f}"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Diagnosis logic
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("DIAGNOSIS")
    print("=" * 78)

    diagnoses = []

    # Test: data scarcity hypothesis
    if not args.skip_llama_full:
        a = results["A_llama_full_default"]["hps_test_tpr5"]
        b = results["B_llama_matched_default"]["hps_test_tpr5"]
        if (a - b) > 0.05:
            diagnoses.append(
                f"DATA_SCARCITY_CONFIRMED: Llama-3 HPS drops from "
                f"{a:.3f} → {b:.3f} when subsampled to Vicuna size. "
                f"HPS is data-hungry."
            )
        else:
            diagnoses.append(
                f"DATA_SCARCITY_REFUTED: Llama-3 HPS is robust to "
                f"subsampling ({a:.3f} → {b:.3f}). "
                f"Something Vicuna-specific is at play."
            )

    # Test: which fix works best on Vicuna?
    d = results["D_vicuna_baseline"]["hps_test_tpr5"]
    e = results["E_vicuna_drop_shallow"]["hps_test_tpr5"]
    f_ = results["F_vicuna_regularized"]["hps_test_tpr5"]
    g = results["G_vicuna_combined"]["hps_test_tpr5"]
    h = results["H_vicuna_early_stop"]["hps_test_tpr5"]
    c4_d = results["D_vicuna_baseline"]["c4_test_tpr5"]

    print(f"\n  Vicuna baseline HPS TPR5:               {d:.4f}")
    print(f"  Vicuna drop-shallow HPS TPR5:           {e:.4f}  "
          f"(Δ vs baseline: {e - d:+.4f})")
    print(f"  Vicuna regularize HPS TPR5:             {f_:.4f}  "
          f"(Δ vs baseline: {f_ - d:+.4f})")
    print(f"  Vicuna combined HPS TPR5:               {g:.4f}  "
          f"(Δ vs baseline: {g - d:+.4f})")
    print(f"  Vicuna early-stop HPS TPR5:             {h:.4f}  "
          f"(Δ vs baseline: {h - d:+.4f})")
    print(f"  Vicuna C4 baseline TPR5:                {c4_d:.4f}")

    best_fix_tpr = max(e, f_, g, h)
    best_fix_name = {e: "drop_shallow", f_: "regularize",
                      g: "combined", h: "early_stop"}[best_fix_tpr]

    if best_fix_tpr - d >= 0.05:
        diagnoses.append(
            f"FIX_HELPS: '{best_fix_name}' improves Vicuna HPS TPR5 from "
            f"{d:.3f} → {best_fix_tpr:.3f} (Δ = +{best_fix_tpr - d:.3f})"
        )
    else:
        diagnoses.append(
            f"NO_FIX_HELPS: All tried fixes leave Vicuna HPS within "
            f"{best_fix_tpr - d:+.3f} of baseline. Issue is more fundamental "
            f"than overfitting + shallow noise."
        )

    if best_fix_tpr >= c4_d - 0.02:
        diagnoses.append(
            f"FIX_CLOSES_GAP: With '{best_fix_name}', Vicuna HPS reaches "
            f"{best_fix_tpr:.3f}, comparable to C4 ({c4_d:.3f}). "
            f"HPS Vicuna failure was a methodology artifact (overfitting + "
            f"shallow-layer noise), not an intrinsic limitation."
        )
    else:
        diagnoses.append(
            f"GAP_REMAINS: Best fix gives {best_fix_tpr:.3f}, but C4 still "
            f"reaches {c4_d:.3f} (gap of {c4_d - best_fix_tpr:.3f}). "
            f"Vicuna may have an intrinsic property that disadvantages HPS "
            f"beyond simple overfitting."
        )

    print()
    for d_str in diagnoses:
        print(f"  • {d_str}")
    print()

    # ─────────────────────────────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────────────────────────────
    output = {
        "config": {
            "device": device,
            "epochs": args.epochs,
            "seed": args.seed,
            "llama_layers_full": LLAMA_LAYERS_FULL,
            "vicuna_layers_full": VICUNA_LAYERS_FULL,
            "vicuna_layers_deep": VICUNA_LAYERS_DEEP,
        },
        "results": results,
        "diagnoses": diagnoses,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=_np_default)
    print(f"Saved full results to {output_path}")

    # ─────────────────────────────────────────────────────────────────────
    # Plot: TPR5 bar chart for each config
    # ─────────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    figdir = Path("results/figs")
    figdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    labels = [r["name"][:42] for r in results.values()]
    hps_tpr = [r["hps_test_tpr5"] for r in results.values()]
    c4_tpr = [r["c4_test_tpr5"] for r in results.values()]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, hps_tpr, width=w, label="HPS test TPR@5%FPR",
           color="tab:blue")
    ax.bar(x + w/2, c4_tpr, width=w, label="C4 test TPR@5%FPR",
           color="tab:orange", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Test TPR @ 5% FPR")
    ax.set_title("HPS vs C4 across configurations — diagnosing Vicuna failure")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim([0, 1.05])
    for xi, v in enumerate(hps_tpr):
        ax.text(xi - w/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
    for xi, v in enumerate(c4_tpr):
        ax.text(xi + w/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=7,
                alpha=0.7)
    plt.tight_layout()
    fig.savefig(figdir / "vicuna_overfitting_test.png", dpi=120)
    plt.close(fig)
    print(f"  saved {figdir}/vicuna_overfitting_test.png")

    # Train vs test AUROC plot (overfitting visualization)
    fig, ax = plt.subplots(figsize=(11, 5))
    train_auc = [r["hps_train_auroc"] for r in results.values()]
    test_auc = [r["hps_test_auroc"] for r in results.values()]
    ax.plot(x, train_auc, "o-", label="HPS train AUROC", color="tab:blue", lw=2)
    ax.plot(x, test_auc, "s-", label="HPS test AUROC", color="tab:red", lw=2)
    for xi, (t, te) in enumerate(zip(train_auc, test_auc)):
        ax.plot([xi, xi], [t, te], "k-", alpha=0.3, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("AUROC")
    ax.set_title("Train vs Test AUROC — overfitting gap visualization")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim([0.5, 1.05])
    plt.tight_layout()
    fig.savefig(figdir / "vicuna_overfitting_gap.png", dpi=120)
    plt.close(fig)
    print(f"  saved {figdir}/vicuna_overfitting_gap.png")


if __name__ == "__main__":
    main()
