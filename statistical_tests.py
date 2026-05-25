"""
statistical_tests.py — Bootstrap CIs + paired statistical tests for HPS vs C4.

Addresses evaluator critical issue C4 (no confidence intervals on key claims).

Self-contained: trains HPS and C4 inline from the activation cache produced by
hps_llama3.py. Multi-seed support for robust statistics.

Outputs to results/statistical_tests.json:
  - Per-method bootstrap 95% CIs on AUROC and TPR @ 5% FPR
  - Paired bootstrap on (HPS - C4) difference + p-value
  - McNemar's test on per-example correctness
  - Cohen's d effect size
  - Multi-seed mean ± std

Usage:
    # Run with defaults (5 seeds, 10000 bootstrap):
    python statistical_tests.py

    # Quick mode (1 seed, 1000 bootstrap):
    python statistical_tests.py --n_seeds 1 --n_bootstrap 1000

    # More rigorous:
    python statistical_tests.py --n_seeds 10 --n_bootstrap 20000
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Reuse the HPS framework from existing pipeline (self-contained, no transformers)
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hps_core import (
    LorentzProjection,
    contrastive_loss,
    extract_trajectory_features,
)


# ---------------------------------------------------------------------------
# Constants matching hps_llama3.py
# ---------------------------------------------------------------------------

HPS_LAYERS = [0, 2, 17, 24, 28, 31]
KAPPA_INIT = 0.1
EPOCHS = 50
FPR_TARGET = 0.05
PROJ_DIM = 64


# ---------------------------------------------------------------------------
# JSON serializer for numpy types
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


# ---------------------------------------------------------------------------
# Statistical primitives
# ---------------------------------------------------------------------------

def auroc(y_true, y_score):
    return float(roc_auc_score(y_true, y_score))


def tpr_at_fpr(y_true, y_score, target_fpr=0.05):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    valid = fpr <= target_fpr
    if not valid.any():
        return 0.0
    return float(tpr[valid].max())


def bootstrap_metric(y_true, y_score, metric_fn, n_bootstrap=10000, seed=42):
    """Bootstrap a scalar metric. Returns mean, 95% CI, std, n_samples."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    samples = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        try:
            val = metric_fn(y_true[idx], y_score[idx])
            if np.isfinite(val):
                samples.append(val)
        except Exception:
            continue
    samples = np.array(samples)
    return {
        "mean": float(np.mean(samples)) if len(samples) else float("nan"),
        "ci_low": float(np.percentile(samples, 2.5)) if len(samples) else float("nan"),
        "ci_high": float(np.percentile(samples, 97.5)) if len(samples) else float("nan"),
        "std": float(np.std(samples)) if len(samples) else float("nan"),
        "n_samples": len(samples),
    }


def paired_bootstrap_diff(y_true, score_a, score_b, metric_fn,
                          n_bootstrap=10000, seed=42):
    """Paired bootstrap of metric(a) - metric(b)."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        try:
            a = metric_fn(y_true[idx], score_a[idx])
            b = metric_fn(y_true[idx], score_b[idx])
            if np.isfinite(a) and np.isfinite(b):
                diffs.append(a - b)
        except Exception:
            continue
    diffs = np.array(diffs)
    if len(diffs) == 0:
        return {
            "mean_diff": float("nan"), "ci_low": float("nan"),
            "ci_high": float("nan"), "p_value_two_sided": float("nan"),
            "significant_at_05": False, "n_samples": 0,
        }
    p_two_sided = 2 * min(
        (diffs <= 0).mean(),
        (diffs >= 0).mean(),
    )
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_value_two_sided": float(p_two_sided),
        "significant_at_05": bool(p_two_sided < 0.05),
        "n_samples": len(diffs),
    }


def mcnemar_test(y_true, pred_a, pred_b):
    """McNemar's test on paired binary predictions."""
    correct_a = (pred_a == y_true).astype(int)
    correct_b = (pred_b == y_true).astype(int)
    b = int(((correct_a == 1) & (correct_b == 0)).sum())
    c = int(((correct_a == 0) & (correct_b == 1)).sum())

    if b + c == 0:
        return {
            "b": 0, "c": 0, "statistic": 0.0,
            "p_value": 1.0, "significant_at_05": False,
            "test_type": "no_disagreement",
        }

    if b + c < 25:
        from scipy.stats import binomtest
        result = binomtest(min(b, c), n=b + c, p=0.5, alternative="two-sided")
        return {
            "b": b, "c": c,
            "statistic": float(min(b, c)),
            "p_value": float(result.pvalue),
            "test_type": "exact_binomial",
            "significant_at_05": bool(result.pvalue < 0.05),
        }
    else:
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        from scipy.stats import chi2
        p = 1 - chi2.cdf(statistic, df=1)
        return {
            "b": b, "c": c,
            "statistic": float(statistic),
            "p_value": float(p),
            "test_type": "chi_squared_continuity_correction",
            "significant_at_05": bool(p < 0.05),
        }


def cohens_d(scores_a, scores_b):
    diff = scores_a - scores_b
    if diff.std() == 0:
        return 0.0
    return float(diff.mean() / diff.std())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_activation_cache(cache_path="results/llama3_activations_cache.npz"):
    """Load the activation cache produced by hps_llama3.py."""
    if not Path(cache_path).exists():
        raise FileNotFoundError(
            f"\n  Activation cache not found at: {cache_path}\n\n"
            f"  This script depends on the cache produced by hps_llama3.py.\n"
            f"  Either run hps_llama3.py first, or copy the cache from your\n"
            f"  remote DGX server to this path.\n"
        )
    cache = np.load(cache_path, allow_pickle=True)
    return cache


def to_hps_array(hs_list, layers=HPS_LAYERS):
    """Convert list-of-dicts to (N, n_layers, d) array."""
    return np.array([[hs[l][-1] for l in layers] for hs in hs_list])


# ---------------------------------------------------------------------------
# Method scorers
# ---------------------------------------------------------------------------

def train_hps(X_train, y_train, n_layers, d_hidden, seed=42, epochs=EPOCHS,
              kappa_init=KAPPA_INIT, device="cpu", verbose=False):
    """Train HPS Lorentz projection + LR classifier."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    proj = LorentzProjection(d_hidden, PROJ_DIM, kappa_init,
                             n_layers=n_layers).to(device)
    proj.log_k.requires_grad = False  # frozen κ (TEST 9 optimal)
    opt = optim.Adam(
        [p for p in proj.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=1e-5,
    )
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)

    for epoch in range(epochs):
        loss = torch.tensor(0.0, device=device)
        for l in range(n_layers):
            h = proj(X_t[:, l, :])
            loss = loss + contrastive_loss(h, y_t, k=proj.k, tau=proj.tau(l))
        loss = loss / n_layers
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and (epoch + 1) % 10 == 0:
            print(f"    [seed={seed}] epoch {epoch+1}/{epochs} "
                  f"loss={loss.item():.4f}")
    proj.eval()
    return proj


def score_hps(proj, X_layers, y_train_for_clf=None, X_train_for_clf=None,
              seed=42):
    """
    Compute HPS detection scores. Trains LR classifier on train trajectory
    features then scores X_layers.
    """
    feats_train = extract_trajectory_features(proj, X_train_for_clf)
    feats_eval = extract_trajectory_features(proj, X_layers)
    sc = StandardScaler().fit(feats_train)
    clf = LogisticRegression(max_iter=2000, random_state=seed,
                             class_weight="balanced")
    clf.fit(sc.transform(feats_train), y_train_for_clf)
    return clf.predict_proba(sc.transform(feats_eval))[:, 1]


def train_and_score_c4(X_tr_ben, X_tr_atk, X_te_ben, X_te_atk, seed=42):
    """C4: mean-pool layers, then LR. Returns (test_y, test_scores)."""
    np.random.seed(seed)
    X_pos = X_tr_atk.mean(axis=1)
    X_neg = X_tr_ben.mean(axis=1)
    X_train = np.concatenate([X_neg, X_pos], axis=0)
    y_train = np.concatenate(
        [np.zeros(len(X_neg)), np.ones(len(X_pos))], axis=0)

    sc = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=2000, random_state=seed,
                             class_weight="balanced")
    clf.fit(sc.transform(X_train), y_train)

    eval_ben = clf.predict_proba(sc.transform(X_te_ben.mean(axis=1)))[:, 1]
    eval_atk = clf.predict_proba(sc.transform(X_te_atk.mean(axis=1)))[:, 1]
    return eval_ben, eval_atk


def train_and_score_hps(X_tr_ben, X_tr_atk, X_te_ben, X_te_atk, seed=42,
                        device="cpu", verbose=False):
    """Full HPS pipeline. Returns (test_ben_scores, test_atk_scores)."""
    X_train = np.concatenate([X_tr_ben, X_tr_atk], axis=0)
    y_train = np.concatenate(
        [np.zeros(len(X_tr_ben)), np.ones(len(X_tr_atk))], axis=0)
    n_layers = X_train.shape[1]
    d_hidden = X_train.shape[2]

    proj = train_hps(X_train, y_train, n_layers, d_hidden,
                     seed=seed, device=device, verbose=verbose)

    feats_train = extract_trajectory_features(proj, X_train)
    feats_te_ben = extract_trajectory_features(proj, X_te_ben)
    feats_te_atk = extract_trajectory_features(proj, X_te_atk)

    sc = StandardScaler().fit(feats_train)
    clf = LogisticRegression(max_iter=2000, random_state=seed,
                             class_weight="balanced")
    clf.fit(sc.transform(feats_train), y_train)

    score_ben = clf.predict_proba(sc.transform(feats_te_ben))[:, 1]
    score_atk = clf.predict_proba(sc.transform(feats_te_atk))[:, 1]
    return score_ben, score_atk, proj


# ---------------------------------------------------------------------------
# Calibrated threshold (held-out split for FPR target)
# ---------------------------------------------------------------------------

def calibrated_predictions(scores_ben, scores_atk, target_fpr=FPR_TARGET,
                           seed=46):
    """
    Split benign scores into calibration + held-out, calibrate threshold
    on calibration, return (eval_y, eval_scores, eval_predictions, threshold).
    """
    rng = np.random.RandomState(seed)
    n_calib = max(50, len(scores_ben) // 4)
    calib_idx = rng.choice(len(scores_ben), size=n_calib, replace=False)
    eval_mask = np.ones(len(scores_ben), dtype=bool)
    eval_mask[calib_idx] = False

    calib_scores = scores_ben[calib_idx]
    threshold = float(np.percentile(calib_scores, 100 * (1 - target_fpr)))

    eval_ben = scores_ben[eval_mask]
    eval_y = np.concatenate([
        np.zeros(eval_mask.sum()),
        np.ones(len(scores_atk)),
    ])
    eval_scores = np.concatenate([eval_ben, scores_atk])
    eval_pred = (eval_scores > threshold).astype(int)
    return eval_y, eval_scores, eval_pred, threshold


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_one_seed(X_tr_ben, X_tr_atk, X_te_ben, X_te_atk, seed, device,
                 verbose=False):
    """Run HPS + C4 with one seed; return scores."""
    if verbose:
        print(f"\n  Seed {seed}: training HPS...")
    hps_ben, hps_atk, _ = train_and_score_hps(
        X_tr_ben, X_tr_atk, X_te_ben, X_te_atk,
        seed=seed, device=device, verbose=verbose,
    )
    if verbose:
        print(f"  Seed {seed}: training C4...")
    c4_ben, c4_atk = train_and_score_c4(
        X_tr_ben, X_tr_atk, X_te_ben, X_te_atk, seed=seed,
    )

    test_y = np.concatenate([
        np.zeros(len(hps_ben)),
        np.ones(len(hps_atk)),
    ])
    test_scores_hps = np.concatenate([hps_ben, hps_atk])
    test_scores_c4 = np.concatenate([c4_ben, c4_atk])

    return {
        "y": test_y,
        "hps_scores": test_scores_hps,
        "c4_scores": test_scores_c4,
        "hps_ben": hps_ben, "hps_atk": hps_atk,
        "c4_ben": c4_ben, "c4_atk": c4_atk,
        "auroc_hps": auroc(test_y, test_scores_hps),
        "auroc_c4": auroc(test_y, test_scores_c4),
        "tpr5_hps": tpr_at_fpr(test_y, test_scores_hps, FPR_TARGET),
        "tpr5_c4": tpr_at_fpr(test_y, test_scores_c4, FPR_TARGET),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache",
                        default="results/llama3_activations_cache.npz",
                        help="Activation cache path from hps_llama3.py")
    parser.add_argument("--n_seeds", type=int, default=5,
                        help="Number of HPS training seeds")
    parser.add_argument("--n_bootstrap", type=int, default=10000,
                        help="Bootstrap iterations")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level")
    parser.add_argument("--device", default=None,
                        help="cuda or cpu (auto-detected)")
    parser.add_argument("--output",
                        default="results/statistical_tests.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print("STATISTICAL SIGNIFICANCE TESTS — HPS vs C4 (Llama-3-8B)")
    print("=" * 78)
    print(f"  Cache:                {args.cache}")
    print(f"  Number of seeds:      {args.n_seeds}")
    print(f"  Bootstrap iterations: {args.n_bootstrap}")
    print(f"  Significance level:   α = {args.alpha}")
    print(f"  Device:               {device}")
    print()

    # ── Load cached activations ──
    print("Loading activation cache...")
    cache = load_activation_cache(args.cache)
    hs_train_ben = cache["hs_train_ben"].tolist()
    hs_train_atk = cache["hs_train_atk"].tolist()
    hs_test_ben = cache["hs_test_ben"].tolist()
    hs_test_atk = cache["hs_test_atk"].tolist()

    X_tr_ben = to_hps_array(hs_train_ben)
    X_tr_atk = to_hps_array(hs_train_atk)
    X_te_ben = to_hps_array(hs_test_ben)
    X_te_atk = to_hps_array(hs_test_atk)
    print(f"  Train benign: {X_tr_ben.shape}, attacks: {X_tr_atk.shape}")
    print(f"  Test  benign: {X_te_ben.shape}, attacks: {X_te_atk.shape}")
    print()

    # ── Multi-seed runs ──
    print(f"Running {args.n_seeds} seeds for HPS + C4...")
    seeds = list(range(42, 42 + args.n_seeds))
    seed_results = []
    for s in seeds:
        r = run_one_seed(X_tr_ben, X_tr_atk, X_te_ben, X_te_atk, s, device,
                         verbose=args.verbose)
        seed_results.append(r)
        print(f"  seed={s}  HPS AUROC={r['auroc_hps']:.4f} TPR5={r['tpr5_hps']:.4f}  "
              f"C4 AUROC={r['auroc_c4']:.4f} TPR5={r['tpr5_c4']:.4f}")
    print()

    # ── Multi-seed summary (mean ± std across seeds) ──
    auroc_hps_seeds = np.array([r["auroc_hps"] for r in seed_results])
    auroc_c4_seeds = np.array([r["auroc_c4"] for r in seed_results])
    tpr5_hps_seeds = np.array([r["tpr5_hps"] for r in seed_results])
    tpr5_c4_seeds = np.array([r["tpr5_c4"] for r in seed_results])

    multi_seed = {
        "n_seeds": args.n_seeds,
        "seeds": seeds,
        "auroc_hps_mean": float(auroc_hps_seeds.mean()),
        "auroc_hps_std": float(auroc_hps_seeds.std()),
        "auroc_c4_mean": float(auroc_c4_seeds.mean()),
        "auroc_c4_std": float(auroc_c4_seeds.std()),
        "tpr5_hps_mean": float(tpr5_hps_seeds.mean()),
        "tpr5_hps_std": float(tpr5_hps_seeds.std()),
        "tpr5_c4_mean": float(tpr5_c4_seeds.mean()),
        "tpr5_c4_std": float(tpr5_c4_seeds.std()),
    }

    # Use the FIRST seed's scores for bootstrap (representative single run)
    primary = seed_results[0]
    test_y = primary["y"]
    test_scores_hps = primary["hps_scores"]
    test_scores_c4 = primary["c4_scores"]

    # ── Bootstrap CIs (per method, each metric) ──
    print(f"Computing bootstrap 95% CIs (n_bootstrap={args.n_bootstrap})...")

    boot_auroc_hps = bootstrap_metric(
        test_y, test_scores_hps, auroc, args.n_bootstrap, seed=100)
    boot_auroc_c4 = bootstrap_metric(
        test_y, test_scores_c4, auroc, args.n_bootstrap, seed=101)

    tpr_fn = lambda y, s: tpr_at_fpr(y, s, FPR_TARGET)
    boot_tpr5_hps = bootstrap_metric(
        test_y, test_scores_hps, tpr_fn, args.n_bootstrap, seed=102)
    boot_tpr5_c4 = bootstrap_metric(
        test_y, test_scores_c4, tpr_fn, args.n_bootstrap, seed=103)

    print()
    print("AUROC bootstrap 95% CIs:")
    print(f"  HPS:  {boot_auroc_hps['mean']:.4f}  "
          f"[CI: {boot_auroc_hps['ci_low']:.4f}, "
          f"{boot_auroc_hps['ci_high']:.4f}]")
    print(f"  C4:   {boot_auroc_c4['mean']:.4f}  "
          f"[CI: {boot_auroc_c4['ci_low']:.4f}, "
          f"{boot_auroc_c4['ci_high']:.4f}]")

    print("TPR @ 5% FPR bootstrap 95% CIs:")
    print(f"  HPS:  {boot_tpr5_hps['mean']:.4f}  "
          f"[CI: {boot_tpr5_hps['ci_low']:.4f}, "
          f"{boot_tpr5_hps['ci_high']:.4f}]")
    print(f"  C4:   {boot_tpr5_c4['mean']:.4f}  "
          f"[CI: {boot_tpr5_c4['ci_low']:.4f}, "
          f"{boot_tpr5_c4['ci_high']:.4f}]")
    print()

    # ── Paired bootstrap (HPS - C4 difference) ──
    print("Paired bootstrap: HPS - C4 difference")
    diff_auroc = paired_bootstrap_diff(
        test_y, test_scores_hps, test_scores_c4, auroc,
        args.n_bootstrap, seed=200)
    diff_tpr5 = paired_bootstrap_diff(
        test_y, test_scores_hps, test_scores_c4, tpr_fn,
        args.n_bootstrap, seed=201)

    sig_auroc = "SIGNIFICANT" if diff_auroc["significant_at_05"] else "NOT SIG"
    sig_tpr5 = "SIGNIFICANT" if diff_tpr5["significant_at_05"] else "NOT SIG"
    print(f"  ΔAUROC = {diff_auroc['mean_diff']:+.4f}  "
          f"[CI: {diff_auroc['ci_low']:+.4f}, {diff_auroc['ci_high']:+.4f}], "
          f"p={diff_auroc['p_value_two_sided']:.4f} ({sig_auroc})")
    print(f"  ΔTPR5  = {diff_tpr5['mean_diff']:+.4f}  "
          f"[CI: {diff_tpr5['ci_low']:+.4f}, {diff_tpr5['ci_high']:+.4f}], "
          f"p={diff_tpr5['p_value_two_sided']:.4f} ({sig_tpr5})")
    print()

    # ── McNemar's test (per-example correctness) ──
    print("McNemar's test on per-example binary correctness:")
    eval_y_h, _, pred_hps, thr_hps = calibrated_predictions(
        primary["hps_ben"], primary["hps_atk"], FPR_TARGET, seed=46)
    eval_y_c, _, pred_c4, thr_c4 = calibrated_predictions(
        primary["c4_ben"], primary["c4_atk"], FPR_TARGET, seed=46)
    # Should be aligned because we used same seed for calib split
    assert (eval_y_h == eval_y_c).all(), "Calibration alignment broken"
    eval_y = eval_y_h

    mcnemar = mcnemar_test(eval_y, pred_hps, pred_c4)
    sig_mc = "SIGNIFICANT" if mcnemar["significant_at_05"] else "NOT SIG"
    print(f"  HPS correct only: {mcnemar['b']}, "
          f"C4 correct only: {mcnemar['c']}")
    print(f"  test type: {mcnemar.get('test_type', '?')}")
    print(f"  p-value: {mcnemar['p_value']:.4f} ({sig_mc})")
    print()

    # ── Cohen's d ──
    cd = cohens_d(test_scores_hps, test_scores_c4)
    if abs(cd) < 0.2:
        effect = "negligible"
    elif abs(cd) < 0.5:
        effect = "small"
    elif abs(cd) < 0.8:
        effect = "medium"
    else:
        effect = "large"
    print(f"Cohen's d (HPS - C4 score distributions): {cd:+.4f} ({effect})")
    print()

    # ── Save results ──
    results = {
        "config": {
            "n_seeds": args.n_seeds,
            "seeds": seeds,
            "n_bootstrap": args.n_bootstrap,
            "alpha": args.alpha,
            "device": device,
            "n_train_benign": int(X_tr_ben.shape[0]),
            "n_train_attacks": int(X_tr_atk.shape[0]),
            "n_test_benign": int(X_te_ben.shape[0]),
            "n_test_attacks": int(X_te_atk.shape[0]),
            "n_layers": int(X_tr_ben.shape[1]),
            "d_hidden": int(X_tr_ben.shape[2]),
            "hps_layers": HPS_LAYERS,
            "kappa_init": KAPPA_INIT,
            "epochs": EPOCHS,
            "fpr_target": FPR_TARGET,
        },
        "multi_seed": multi_seed,
        "per_seed": [
            {
                "seed": s,
                "auroc_hps": float(r["auroc_hps"]),
                "auroc_c4": float(r["auroc_c4"]),
                "tpr5_hps": float(r["tpr5_hps"]),
                "tpr5_c4": float(r["tpr5_c4"]),
            }
            for s, r in zip(seeds, seed_results)
        ],
        "bootstrap": {
            "auroc_hps": boot_auroc_hps,
            "auroc_c4": boot_auroc_c4,
            "tpr5_hps": boot_tpr5_hps,
            "tpr5_c4": boot_tpr5_c4,
        },
        "paired_bootstrap": {
            "auroc": diff_auroc,
            "tpr5": diff_tpr5,
        },
        "mcnemar": mcnemar,
        "thresholds": {
            "hps": float(thr_hps),
            "c4": float(thr_c4),
        },
        "cohens_d": {
            "value": cd,
            "interpretation": effect,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_np_default)

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print()
    print(f"HPS vs C4 on Llama-3-8B (n_seeds={args.n_seeds}):")
    print(f"  AUROC: HPS = {multi_seed['auroc_hps_mean']:.4f} ± "
          f"{multi_seed['auroc_hps_std']:.4f}   "
          f"C4 = {multi_seed['auroc_c4_mean']:.4f} ± "
          f"{multi_seed['auroc_c4_std']:.4f}")
    print(f"  TPR5 : HPS = {multi_seed['tpr5_hps_mean']:.4f} ± "
          f"{multi_seed['tpr5_hps_std']:.4f}   "
          f"C4 = {multi_seed['tpr5_c4_mean']:.4f} ± "
          f"{multi_seed['tpr5_c4_std']:.4f}")
    print()
    print(f"  ΔAUROC = {diff_auroc['mean_diff']:+.4f} "
          f"[95% CI: {diff_auroc['ci_low']:+.4f}, {diff_auroc['ci_high']:+.4f}]"
          f"  p = {diff_auroc['p_value_two_sided']:.4f}")
    print(f"  ΔTPR5  = {diff_tpr5['mean_diff']:+.4f} "
          f"[95% CI: {diff_tpr5['ci_low']:+.4f}, {diff_tpr5['ci_high']:+.4f}]"
          f"  p = {diff_tpr5['p_value_two_sided']:.4f}")
    print(f"  McNemar's p = {mcnemar['p_value']:.4f}")
    print()
    if (not diff_auroc["significant_at_05"]
            and not diff_tpr5["significant_at_05"]
            and not mcnemar["significant_at_05"]):
        print("  CONCLUSION: HPS - C4 difference is NOT statistically")
        print("  significant at α=0.05 across multiple tests.")
        print("  The observed difference is plausibly within noise.")
    elif diff_auroc["significant_at_05"] or diff_tpr5["significant_at_05"]:
        print("  CONCLUSION: Some metrics show statistical significance.")
        print("  Inspect per-metric details above.")
    print()
    print(f"Saved full results to {output_path}")


if __name__ == "__main__":
    main()
