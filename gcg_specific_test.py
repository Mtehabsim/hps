"""
gcg_specific_test.py — Confirm the alignment-strength → GCG-detectability hypothesis.

vicuna_imbalance_test.py revealed that HPS specifically fails on GCG attacks
on Vicuna (37.5% detection vs C4's 100%) while succeeding on PAIR (100%),
prompt_with_random_search (100%), and JBC (90.5%).

This script tests whether the failure is GCG-on-Vicuna-specific, by computing
per-attack detection rates for HPS and C4 on **Llama-3-8B**.

Hypothesis:
  - On Llama-3-8B (SFT + RLHF, stronger alignment), GCG attacks produce a
    distinctive activation signature that survives HPS's compression to 12
    trajectory features. HPS catches Llama-3 GCG at ≥95%.
  - On Vicuna-13B (SFT only, weaker alignment), GCG attacks produce more
    diffuse activation patterns that get filtered out by HPS's compression.
    HPS catches Vicuna GCG at <50%.

If this hypothesis holds, the paper has a clean mechanistic explanation:
"HPS's geometric compression preserves attack signal only when the
underlying signal is sufficiently concentrated; weaker alignment produces
more diffuse signals that get filtered out."

Usage:
  python gcg_specific_test.py
  python gcg_specific_test.py --attacks-json llama3_attacks.json
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
from vicuna_imbalance_test import (
    load_llama_cache,
    load_vicuna_cache,
    train_hps,
    score_via_hps,
    score_via_c4,
    auroc,
    tpr_at_fpr,
    per_attack_breakdown,
)

LLAMA_LAYERS = [0, 2, 17, 24, 28, 31]
VICUNA_LAYERS = [0, 2, 22, 31, 35, 39]


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


# ---------------------------------------------------------------------------
# Replicate hps_llama3.py's attack indexing to derive test method labels
# ---------------------------------------------------------------------------

def derive_llama_attack_methods(attacks_json_path, seed=42):
    """
    Replicate the EXACT attack indexing logic from hps_llama3.py to recover
    which test attacks belong to which category.

    From hps_llama3.py:
        with open(args.test_attacks) as f:
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
        train_atk = [attack_prompts[i] for i in atk_idx[:n_atk_tr]]
        test_atk = [attack_prompts[i] for i in atk_idx[n_atk_tr:]]
        test_methods = [attack_methods[i] for i in atk_idx[n_atk_tr:]]
    """
    if not Path(attacks_json_path).exists():
        raise FileNotFoundError(
            f"Attack JSON not found: {attacks_json_path}. "
            f"This is the file passed via --test-attacks to hps_llama3.py."
        )
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
    train_methods = [attack_methods[i] for i in atk_idx[:n_atk_tr]]
    test_methods = [attack_methods[i] for i in atk_idx[n_atk_tr:]]
    return train_methods, test_methods


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama3_cache",
                        default="results/llama3_activations_cache.npz")
    parser.add_argument("--vicuna_cache",
                        default="results/vicuna_activations_cache.npz")
    parser.add_argument("--attacks_json",
                        default="llama3_attacks.json",
                        help="Attack JSON used by hps_llama3.py "
                             "(needed to recover per-attack labels)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output",
                        default="results/gcg_specific_test.json")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print("GCG-SPECIFIC CROSS-MODEL TEST")
    print("=" * 78)
    print(f"  Llama-3 cache: {args.llama3_cache}")
    print(f"  Vicuna cache:  {args.vicuna_cache}")
    print(f"  Attacks JSON:  {args.attacks_json}")
    print(f"  Device:        {device}")
    print()

    # ── Load Llama-3 attack labels ──
    print("Recovering Llama-3 attack labels...")
    try:
        llama_train_methods, llama_test_methods = derive_llama_attack_methods(
            args.attacks_json, seed=args.seed,
        )
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        print(f"\n  Cannot recover Llama-3 per-attack labels without the JSON.")
        print(f"  Make sure {args.attacks_json} is the same file you used in")
        print(f"  hps_llama3.py and that it's in the current directory.")
        sys.exit(1)

    train_counts = Counter(llama_train_methods)
    test_counts = Counter(llama_test_methods)
    print(f"  Llama-3 attack methods (train): {dict(train_counts)}")
    print(f"  Llama-3 attack methods (test):  {dict(test_counts)}")
    print(f"  Number of unique attack methods on Llama-3: "
          f"{len(set(llama_train_methods + llama_test_methods))}")
    print()

    # ── Load activation caches ──
    print("Loading activation caches...")
    llama = load_llama_cache(args.llama3_cache, LLAMA_LAYERS)
    vicuna = load_vicuna_cache(args.vicuna_cache, VICUNA_LAYERS)
    print(f"  Llama-3 train: {len(llama['X_tr_ben'])}+{len(llama['X_tr_atk'])}, "
          f"test {len(llama['X_te_ben'])}+{len(llama['X_te_atk'])}")
    print(f"  Vicuna  train: {len(vicuna['X_tr_ben'])}+{len(vicuna['X_tr_atk'])}, "
          f"test {len(vicuna['X_te_ben'])}+{len(vicuna['X_te_atk'])}")

    # Sanity check: derived test_methods length should match cache test atk size
    if len(llama_test_methods) != len(llama["X_te_atk"]):
        print(f"\n  WARNING: derived test_methods length ({len(llama_test_methods)}) "
              f"!= cache test_atk length ({len(llama['X_te_atk'])})")
        print(f"  This means the attack JSON may not match the cache's "
              f"split. Per-attack breakdown may be misaligned.")
        # Try to truncate/use what we can
        n = min(len(llama_test_methods), len(llama["X_te_atk"]))
        llama_test_methods = llama_test_methods[:n]
        print(f"  Truncating to first {n} examples.")

    # ── Phase 1: Train HPS + C4 on Llama-3, get per-attack test scores ──
    print("\n" + "=" * 78)
    print("PHASE 1 — Llama-3 per-attack breakdown")
    print("=" * 78)

    X_train_l = np.concatenate([llama["X_tr_ben"], llama["X_tr_atk"]], axis=0)
    y_train_l = np.concatenate([
        np.zeros(len(llama["X_tr_ben"])),
        np.ones(len(llama["X_tr_atk"])),
    ])
    n_layers_l = X_train_l.shape[1]
    d_l = X_train_l.shape[2]

    print(f"\n  Training HPS on Llama-3 (default config, {args.epochs} epochs)...")
    proj_l, losses_l = train_hps(
        X_train_l, y_train_l, n_layers_l, d_l,
        epochs=args.epochs, seed=args.seed, device=device,
    )

    test_eval_l = np.concatenate([llama["X_te_ben"], llama["X_te_atk"]], axis=0)
    test_y_l = np.concatenate([
        np.zeros(len(llama["X_te_ben"])),
        np.ones(len(llama["X_te_atk"])),
    ])
    test_scores_hps_l = score_via_hps(
        proj_l, X_train_l, y_train_l, test_eval_l, args.seed)
    test_scores_c4_l = score_via_c4(
        llama["X_tr_ben"], llama["X_tr_atk"], test_eval_l, args.seed)

    # Calibrate thresholds at 5% FPR on test benign
    n_ben_te_l = len(llama["X_te_ben"])
    thr_hps_l = float(np.percentile(test_scores_hps_l[:n_ben_te_l], 95))
    thr_c4_l  = float(np.percentile(test_scores_c4_l[:n_ben_te_l], 95))

    print(f"  HPS overall test AUROC: "
          f"{auroc(test_y_l, test_scores_hps_l):.4f}, "
          f"TPR@5%FPR: {tpr_at_fpr(test_y_l, test_scores_hps_l):.4f}")
    print(f"  C4  overall test AUROC: "
          f"{auroc(test_y_l, test_scores_c4_l):.4f}, "
          f"TPR@5%FPR: {tpr_at_fpr(test_y_l, test_scores_c4_l):.4f}")

    # Per-attack breakdown
    breakdown_l = per_attack_breakdown(
        test_scores_hps_l.tolist(), test_scores_c4_l.tolist(),
        test_y_l.tolist(), llama_test_methods,
        threshold_hps=thr_hps_l, threshold_c4=thr_c4_l,
    )
    print(f"\n  Llama-3 thresholds at 5% FPR: "
          f"HPS={thr_hps_l:.4f}, C4={thr_c4_l:.4f}")
    print(f"\n  {'Attack':<30s}  {'N':>4s}  {'HPS rate':>9s}  "
          f"{'C4 rate':>9s}  {'gap (C4-HPS)':>13s}")
    print("  " + "-" * 75)
    for m in sorted(breakdown_l.keys()):
        b = breakdown_l[m]
        gap = b["c4_detection_rate"] - b["hps_detection_rate"]
        print(f"  {str(m):<30s}  {b['n_total']:>4d}  "
              f"{b['hps_detection_rate']:>9.4f}  "
              f"{b['c4_detection_rate']:>9.4f}  "
              f"{gap:>+13.4f}")

    # ── Phase 2: Vicuna per-attack (rerun for fresh comparison) ──
    print("\n" + "=" * 78)
    print("PHASE 2 — Vicuna per-attack breakdown")
    print("=" * 78)

    X_train_v = np.concatenate([vicuna["X_tr_ben"], vicuna["X_tr_atk"]], axis=0)
    y_train_v = np.concatenate([
        np.zeros(len(vicuna["X_tr_ben"])),
        np.ones(len(vicuna["X_tr_atk"])),
    ])
    n_layers_v = X_train_v.shape[1]
    d_v = X_train_v.shape[2]

    print(f"\n  Training HPS on Vicuna (default config, {args.epochs} epochs)...")
    proj_v, losses_v = train_hps(
        X_train_v, y_train_v, n_layers_v, d_v,
        epochs=args.epochs, seed=args.seed, device=device,
    )

    test_eval_v = np.concatenate([vicuna["X_te_ben"], vicuna["X_te_atk"]], axis=0)
    test_y_v = np.concatenate([
        np.zeros(len(vicuna["X_te_ben"])),
        np.ones(len(vicuna["X_te_atk"])),
    ])
    test_scores_hps_v = score_via_hps(
        proj_v, X_train_v, y_train_v, test_eval_v, args.seed)
    test_scores_c4_v = score_via_c4(
        vicuna["X_tr_ben"], vicuna["X_tr_atk"], test_eval_v, args.seed)

    n_ben_te_v = len(vicuna["X_te_ben"])
    thr_hps_v = float(np.percentile(test_scores_hps_v[:n_ben_te_v], 95))
    thr_c4_v  = float(np.percentile(test_scores_c4_v[:n_ben_te_v], 95))

    breakdown_v = per_attack_breakdown(
        test_scores_hps_v.tolist(), test_scores_c4_v.tolist(),
        test_y_v.tolist(), vicuna["atk_methods_test"],
        threshold_hps=thr_hps_v, threshold_c4=thr_c4_v,
    )
    print(f"\n  Vicuna thresholds at 5% FPR: "
          f"HPS={thr_hps_v:.4f}, C4={thr_c4_v:.4f}")
    print(f"\n  {'Attack':<30s}  {'N':>4s}  {'HPS rate':>9s}  "
          f"{'C4 rate':>9s}  {'gap (C4-HPS)':>13s}")
    print("  " + "-" * 75)
    for m in sorted(breakdown_v.keys()):
        b = breakdown_v[m]
        gap = b["c4_detection_rate"] - b["hps_detection_rate"]
        print(f"  {str(m):<30s}  {b['n_total']:>4d}  "
              f"{b['hps_detection_rate']:>9.4f}  "
              f"{b['c4_detection_rate']:>9.4f}  "
              f"{gap:>+13.4f}")

    # ── Phase 3: Side-by-side comparison ──
    print("\n" + "=" * 78)
    print("PHASE 3 — Side-by-side: HPS detection rate per attack, both LLMs")
    print("=" * 78)

    # Build a case-insensitive method lookup so "gcg" and "GCG" align
    def _norm(m):
        return str(m).lower() if m is not None else None
    breakdown_l_norm = {_norm(k): v for k, v in breakdown_l.items()}
    breakdown_v_norm = {_norm(k): v for k, v in breakdown_v.items()}
    all_methods_norm = sorted(set(list(breakdown_l_norm.keys())
                                  + list(breakdown_v_norm.keys())))

    print(f"\n  {'Attack (case-insensitive)':<30s}  "
          f"{'Llama-HPS':>9s}  {'Llama-C4':>9s}  "
          f"{'Vicuna-HPS':>10s}  {'Vicuna-C4':>10s}  "
          f"{'HPS Δ':>7s}")
    print("  " + "-" * 90)
    cross = {}
    for m in all_methods_norm:
        bl = breakdown_l_norm.get(m, {"hps_detection_rate": float("nan"),
                                       "c4_detection_rate": float("nan"),
                                       "n_total": 0})
        bv = breakdown_v_norm.get(m, {"hps_detection_rate": float("nan"),
                                       "c4_detection_rate": float("nan"),
                                       "n_total": 0})
        if (np.isnan(bl["hps_detection_rate"])
                or np.isnan(bv["hps_detection_rate"])):
            hps_delta = float("nan")
        else:
            hps_delta = bl["hps_detection_rate"] - bv["hps_detection_rate"]
        cross[m] = {
            "llama_n": bl["n_total"],
            "llama_hps": bl["hps_detection_rate"],
            "llama_c4": bl["c4_detection_rate"],
            "vicuna_n": bv["n_total"],
            "vicuna_hps": bv["hps_detection_rate"],
            "vicuna_c4": bv["c4_detection_rate"],
            "hps_llama_minus_vicuna": hps_delta,
        }
        print(f"  {str(m):<30s}  "
              f"{bl['hps_detection_rate']:>9.4f}  "
              f"{bl['c4_detection_rate']:>9.4f}  "
              f"{bv['hps_detection_rate']:>10.4f}  "
              f"{bv['c4_detection_rate']:>10.4f}  "
              f"{hps_delta:>+7.4f}")

    # ── Diagnosis ──
    print("\n" + "=" * 78)
    print("DIAGNOSIS")
    print("=" * 78)

    diagnoses = []

    # Check GCG specifically (case-insensitive)
    gcg_key = "gcg"  # normalized
    if gcg_key in cross:
        gcg = cross[gcg_key]
        if gcg["llama_hps"] >= 0.95 and gcg["vicuna_hps"] < 0.6:
            diagnoses.append(
                f"ALIGNMENT_HYPOTHESIS_CONFIRMED: HPS catches Llama-3 GCG at "
                f"{gcg['llama_hps']:.3f} ({gcg['llama_n']} samples) but "
                f"Vicuna GCG at only {gcg['vicuna_hps']:.3f} "
                f"({gcg['vicuna_n']} samples). The alignment-strength → "
                f"signal-concentration → HPS-detectability hypothesis holds. "
                f"Strong RLHF alignment produces concentrated GCG signatures "
                f"that survive HPS's compression; weak SFT-only alignment "
                f"produces diffuse signatures that get filtered out."
            )
        elif gcg["llama_hps"] < 0.6 and gcg["vicuna_hps"] < 0.6:
            diagnoses.append(
                f"GCG_INTRINSICALLY_HARD: HPS catches GCG at "
                f"{gcg['llama_hps']:.3f} on Llama-3 and "
                f"{gcg['vicuna_hps']:.3f} on Vicuna. HPS has a fundamental "
                f"weakness against gradient-optimized adversarial suffixes, "
                f"independent of model alignment."
            )
        elif gcg["llama_hps"] >= 0.95 and gcg["vicuna_hps"] >= 0.95:
            diagnoses.append(
                f"GCG_NOT_THE_ISSUE: HPS catches GCG at {gcg['llama_hps']:.3f} "
                f"on Llama-3 and {gcg['vicuna_hps']:.3f} on Vicuna. The "
                f"earlier GCG-failure pattern was sample-noise; need to look "
                f"elsewhere for the Vicuna gap."
            )
        else:
            diagnoses.append(
                f"GCG_PARTIAL: HPS catches Llama-3 GCG at "
                f"{gcg['llama_hps']:.3f} ({gcg['llama_n']} samples) and "
                f"Vicuna GCG at {gcg['vicuna_hps']:.3f} ({gcg['vicuna_n']} "
                f"samples). Alignment hypothesis partially supported; "
                f"the gap is real but smaller than predicted."
            )

    # Check overall HPS-Vicuna gap composition
    llama_avg = np.mean([
        b["hps_detection_rate"] for b in breakdown_l.values()
        if not np.isnan(b["hps_detection_rate"])
    ])
    vicuna_avg = np.mean([
        b["hps_detection_rate"] for b in breakdown_v.values()
        if not np.isnan(b["hps_detection_rate"])
    ])
    diagnoses.append(
        f"OVERALL: Average HPS detection rate Llama-3={llama_avg:.4f}, "
        f"Vicuna={vicuna_avg:.4f}. Gap = {llama_avg - vicuna_avg:+.4f}."
    )

    print()
    for d in diagnoses:
        print(f"  • {d}")

    # ── Save ──
    output = {
        "config": {
            "device": device,
            "epochs": args.epochs,
            "seed": args.seed,
            "llama_layers": LLAMA_LAYERS,
            "vicuna_layers": VICUNA_LAYERS,
        },
        "llama_attack_method_counts_train": dict(train_counts),
        "llama_attack_method_counts_test": dict(test_counts),
        "llama_thresholds": {"hps": thr_hps_l, "c4": thr_c4_l},
        "vicuna_thresholds": {"hps": thr_hps_v, "c4": thr_c4_v},
        "llama_per_attack": breakdown_l,
        "vicuna_per_attack": breakdown_v,
        "cross_model": cross,
        "diagnoses": diagnoses,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=_np_default)
    print(f"\nSaved to {output_path}")

    # ── Plot ──
    print("\nGenerating plots...")
    figdir = Path("results/figs")
    figdir.mkdir(parents=True, exist_ok=True)

    # Per-attack HPS detection rate, side-by-side
    valid_methods = [m for m in all_methods_norm
                      if not np.isnan(cross[m]["llama_hps"])
                      or not np.isnan(cross[m]["vicuna_hps"])]
    x = np.arange(len(valid_methods))
    fig, ax = plt.subplots(figsize=(11, 5))
    w = 0.2
    llama_hps = [cross[m]["llama_hps"] if not np.isnan(cross[m]["llama_hps"])
                  else 0.0 for m in valid_methods]
    llama_c4 = [cross[m]["llama_c4"] if not np.isnan(cross[m]["llama_c4"])
                  else 0.0 for m in valid_methods]
    vic_hps = [cross[m]["vicuna_hps"] if not np.isnan(cross[m]["vicuna_hps"])
                else 0.0 for m in valid_methods]
    vic_c4 = [cross[m]["vicuna_c4"] if not np.isnan(cross[m]["vicuna_c4"])
                else 0.0 for m in valid_methods]
    ax.bar(x - 1.5*w, llama_hps, width=w, label="Llama-3 HPS", color="tab:blue")
    ax.bar(x - 0.5*w, llama_c4, width=w, label="Llama-3 C4",
           color="tab:cyan", alpha=0.7)
    ax.bar(x + 0.5*w, vic_hps, width=w, label="Vicuna HPS",
           color="tab:red")
    ax.bar(x + 1.5*w, vic_c4, width=w, label="Vicuna C4",
           color="tab:orange", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_methods, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Detection rate (5% FPR threshold)")
    ax.set_title("Per-attack detection: HPS vs C4 across both LLMs")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim([0, 1.1])
    plt.tight_layout()
    fig.savefig(figdir / "gcg_per_attack_cross_model.png", dpi=120)
    plt.close(fig)
    print(f"  saved {figdir}/gcg_per_attack_cross_model.png")

    # GCG-specific bar
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if "gcg" in cross:
        gcg = cross["gcg"]
        configs = ["Llama-3 HPS", "Llama-3 C4", "Vicuna HPS", "Vicuna C4"]
        values = [gcg["llama_hps"], gcg["llama_c4"],
                  gcg["vicuna_hps"], gcg["vicuna_c4"]]
        # Replace NaN with 0 for plotting
        values = [0.0 if (v != v) else v for v in values]
        colors = ["tab:blue", "tab:cyan", "tab:red", "tab:orange"]
        ax.bar(configs, values, color=colors)
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
        ax.set_ylim([0, 1.1])
        ax.set_ylabel("GCG detection rate")
        ax.set_title("GCG attack detection: HPS vs C4 on both LLMs")
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        fig.savefig(figdir / "gcg_specific.png", dpi=120)
        plt.close(fig)
        print(f"  saved {figdir}/gcg_specific.png")


if __name__ == "__main__":
    main()
