"""
Control experiments — does the geometric prior actually do anything?

Tests whether HPS's mean_r feature can be replicated by simpler baselines:
  C1: Raw L2 norm (mean across layers) — single scalar, no projection
  C2: Per-layer L2 norm — 6-dim vector, no projection
  C3: Random untrained Lorentz projection — tests if training matters
  C4: Linear probe on mean-pooled activations — 5120-dim, full linearity
  C5: Linear probe on flattened activations — 6×5120 = 30720-dim
  C6: Prompt length analysis — are attacks just longer prompts?

If any control matches HPS (mean_r alone), the hyperbolic story weakens.
If all fail, the geometric prior is genuine.

Each control evaluated on:
  - Same-distribution (full data)
  - Cold-start N=5 per method (cross-attack)
  - Cold-start N=25 per method (cross-attack)
  - Vicuna-like (4 methods × 25)

Usage:
  python control_experiments.py \
    --test-attacks llama3_attacks.json \
    --harmless data_harmless_6500.csv

Runtime: ~15-20 minutes on cached activations.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from experiment7 import LorentzProjection, contrastive_loss, extract_trajectory_features
from rtv_standalone import FPR_TARGET

HPS_LAYERS = [0, 2, 17, 24, 28, 31]
KAPPA_INIT = 0.1
device = "cuda" if torch.cuda.is_available() else "cpu"


def to_arr(hs_list, layers):
    return np.array([[hs[l][-1] for l in layers] for hs in hs_list])


def evaluate(feats_train, y_train, feats_te_ben, feats_te_atk, seed=42):
    """Train classifier with calibration split. Returns (auroc, tpr, fpr)."""
    if feats_train.ndim == 1:
        feats_train = feats_train.reshape(-1, 1)
        feats_te_ben = feats_te_ben.reshape(-1, 1)
        feats_te_atk = feats_te_atk.reshape(-1, 1)
    sc = StandardScaler()
    f_tr_s = sc.fit_transform(feats_train)
    clf = LogisticRegression(max_iter=2000, random_state=seed, C=1.0)
    clf.fit(f_tr_s, y_train)
    n_calib = len(feats_te_ben) // 2
    if n_calib < 5:
        s_calib = clf.predict_proba(sc.transform(feats_te_ben))[:, 1]
        s_ben = s_calib
    else:
        s_calib = clf.predict_proba(sc.transform(feats_te_ben[:n_calib]))[:, 1]
        s_ben = clf.predict_proba(sc.transform(feats_te_ben[n_calib:]))[:, 1]
    s_atk = clf.predict_proba(sc.transform(feats_te_atk))[:, 1]
    thr = float(np.quantile(s_calib, 1.0 - FPR_TARGET))
    tpr = float((s_atk > thr).mean())
    fpr = float((s_ben > thr).mean())
    auroc = roc_auc_score(np.array([0]*len(s_ben) + [1]*len(s_atk)),
                          np.concatenate([s_ben, s_atk]))
    return auroc, tpr, fpr


# ═══════════════════════════════════════════════════════════════════
#  Control feature extractors
# ═══════════════════════════════════════════════════════════════════
def feat_raw_norm_scalar(X):
    """C1: Mean L2 norm across all selected layers (1 scalar per sample)."""
    # X: (N, n_layers, d_hidden)
    return np.linalg.norm(X, axis=2).mean(axis=1)  # → (N,)


def feat_raw_norm_perlayer(X):
    """C2: Per-layer L2 norm (n_layers-dim vector per sample)."""
    return np.linalg.norm(X, axis=2)  # → (N, n_layers)


def feat_random_lorentz_radius(X, seed=0):
    """C3: Random untrained Lorentz projection, mean radius (1 scalar per sample)."""
    n_layers = X.shape[1]
    d_hidden = X.shape[2]
    torch.manual_seed(seed)
    proj = LorentzProjection(d_hidden, 64, KAPPA_INIT, n_layers=n_layers).to(device)
    proj.eval()
    radii = []
    with torch.no_grad():
        for i in range(len(X)):
            x = torch.tensor(X[i], dtype=torch.float32, device=device)
            h = proj(x)  # (n_layers, 65)
            r = h[:, 0].cpu().numpy()
            radii.append(r.mean())
    return np.array(radii)


def feat_mean_pooled_activations(X):
    """C4: Mean-pooled activations across layers (d_hidden-dim per sample)."""
    return X.mean(axis=1)  # → (N, d_hidden)


def feat_flattened_activations(X):
    """C5: Flattened activations (n_layers × d_hidden per sample)."""
    return X.reshape(X.shape[0], -1)  # → (N, n_layers*d_hidden)


def feat_hps_mean_r(X, proj):
    """Reference: HPS mean_r — single scalar from trained Lorentz projection."""
    radii = []
    with torch.no_grad():
        for i in range(len(X)):
            x = torch.tensor(X[i], dtype=torch.float32, device=device)
            h = proj(x)
            r = h[:, 0].cpu().numpy()
            radii.append(r.mean())
    return np.array(radii)


def train_hps(X_train, y_train, seed=42, epochs=50, freeze_kappa=True):
    """Train HPS Lorentz projection."""
    n_layers = X_train.shape[1]
    d_hidden = X_train.shape[2]
    torch.manual_seed(seed)
    np.random.seed(seed)
    proj = LorentzProjection(d_hidden, 64, KAPPA_INIT, n_layers=n_layers).to(device)
    if freeze_kappa:
        proj.log_k.requires_grad = False
    opt = optim.Adam([p for p in proj.parameters() if p.requires_grad],
                     lr=1e-3, weight_decay=1e-5)
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)
    for _ in range(epochs):
        loss = torch.tensor(0.0, device=device)
        for l in range(n_layers):
            h = proj(X_t[:, l, :])
            loss = loss + contrastive_loss(h, y_t, k=proj.k, tau=proj.tau(l))
        loss = loss / n_layers
        opt.zero_grad(); loss.backward(); opt.step()
    proj.eval()
    return proj


# ═══════════════════════════════════════════════════════════════════
#  Per-control evaluation helper
# ═══════════════════════════════════════════════════════════════════
def eval_control_same_dist(name, feat_fn, X_train, y_train, X_te_ben, X_te_atk,
                            extra_kwargs=None):
    """Evaluate a non-HPS control on same-distribution data."""
    f_tr = feat_fn(X_train) if extra_kwargs is None else feat_fn(X_train, **extra_kwargs)
    f_be = feat_fn(X_te_ben) if extra_kwargs is None else feat_fn(X_te_ben, **extra_kwargs)
    f_at = feat_fn(X_te_atk) if extra_kwargs is None else feat_fn(X_te_atk, **extra_kwargs)
    auroc, tpr, fpr = evaluate(f_tr, y_train, f_be, f_at)
    n_dim = 1 if f_tr.ndim == 1 else f_tr.shape[1]
    return {"name": name, "n_dim": int(n_dim), "auroc": auroc, "tpr": tpr, "fpr": fpr}


def eval_control_cold_start(name, feat_fn, X_all_atk, all_methods, methods_unique,
                             X_all_ben, n_per, extra_kwargs_factory=None):
    """Evaluate control in cold-start cross-attack setup.
    extra_kwargs_factory: callable(seed) → kwargs (for reproducibility per fold)"""
    hs_by_method = {m: [] for m in methods_unique}
    for act, method in zip(X_all_atk, all_methods):
        hs_by_method[method].append(act)
    for m in methods_unique:
        hs_by_method[m] = np.array(hs_by_method[m])

    ben_split = int(0.8 * len(X_all_ben))
    cv_ben_tr = X_all_ben[:ben_split]
    cv_ben_te = X_all_ben[ben_split:]

    tprs, fprs = [], []
    for held_out in methods_unique:
        sub_atk = []
        for m in methods_unique:
            if m != held_out:
                avail = hs_by_method[m]
                take = min(n_per, len(avail))
                sub_atk.append(avail[:take])
        train_atk = np.concatenate(sub_atk)
        test_atk = hs_by_method[held_out]
        if len(test_atk) < 5:
            continue
        X_tr = np.concatenate([cv_ben_tr, train_atk])
        y_tr = np.array([0]*len(cv_ben_tr) + [1]*len(train_atk))
        kw = extra_kwargs_factory(0) if extra_kwargs_factory else None
        f_tr = feat_fn(X_tr) if kw is None else feat_fn(X_tr, **kw)
        f_be = feat_fn(cv_ben_te) if kw is None else feat_fn(cv_ben_te, **kw)
        f_at = feat_fn(test_atk) if kw is None else feat_fn(test_atk, **kw)
        _, t, fpr = evaluate(f_tr, y_tr, f_be, f_at)
        tprs.append(t)
        fprs.append(fpr)
    return {"name": name, "tpr_mean": float(np.mean(tprs)) if tprs else 0.0,
            "fpr_mean": float(np.mean(fprs)) if fprs else 0.0,
            "fpr_std": float(np.std(fprs)) if fprs else 0.0}


def eval_hps_cold_start(name, X_all_atk, all_methods, methods_unique, X_all_ben, n_per):
    """Special HPS handler: trains projection per fold, then extracts mean_r."""
    hs_by_method = {m: [] for m in methods_unique}
    for act, method in zip(X_all_atk, all_methods):
        hs_by_method[method].append(act)
    for m in methods_unique:
        hs_by_method[m] = np.array(hs_by_method[m])

    ben_split = int(0.8 * len(X_all_ben))
    cv_ben_tr = X_all_ben[:ben_split]
    cv_ben_te = X_all_ben[ben_split:]

    tprs, fprs = [], []
    for held_out in methods_unique:
        sub_atk = []
        for m in methods_unique:
            if m != held_out:
                avail = hs_by_method[m]
                take = min(n_per, len(avail))
                sub_atk.append(avail[:take])
        train_atk = np.concatenate(sub_atk)
        test_atk = hs_by_method[held_out]
        if len(test_atk) < 5:
            continue
        X_tr = np.concatenate([cv_ben_tr, train_atk])
        y_tr = np.array([0]*len(cv_ben_tr) + [1]*len(train_atk))
        proj_fold = train_hps(X_tr, y_tr, seed=42)
        f_tr = feat_hps_mean_r(X_tr, proj_fold)
        f_be = feat_hps_mean_r(cv_ben_te, proj_fold)
        f_at = feat_hps_mean_r(test_atk, proj_fold)
        _, t, fpr = evaluate(f_tr, y_tr, f_be, f_at)
        tprs.append(t)
        fprs.append(fpr)
    return {"name": name, "tpr_mean": float(np.mean(tprs)) if tprs else 0.0,
            "fpr_mean": float(np.mean(fprs)) if fprs else 0.0,
            "fpr_std": float(np.std(fprs)) if fprs else 0.0}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-attacks", required=True)
    parser.add_argument("--harmless", required=True)
    parser.add_argument("--cache", default="results/llama3_activations_cache.npz")
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"  CONTROL EXPERIMENTS — Does the geometric prior matter?")
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

    X_train = np.concatenate([X_tr_ben, X_tr_atk])
    y_train = np.array([0]*len(X_tr_ben) + [1]*len(X_tr_atk))

    # ══════════════════════════════════════════════════════════════
    #  Reference: HPS trained on full data
    # ══════════════════════════════════════════════════════════════
    print(f"  Training HPS reference (full data)...")
    proj_full = train_hps(X_train, y_train, seed=42)

    # ══════════════════════════════════════════════════════════════
    #  PART A: Same-distribution (full data)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  PART A: Same-distribution (full data)")
    print(f"  Compare each control to HPS (mean_r alone)")
    print(f"{'─'*60}\n")
    print(f"  {'Control':<35} | {'#dim':>5} | {'AUROC':>6} | {'TPR':>6} | {'FPR':>5}")
    print(f"  {'─'*35}─┼─{'─'*5}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*5}")

    sd_results = []

    # HPS reference
    f_tr_h = feat_hps_mean_r(X_train, proj_full)
    f_be_h = feat_hps_mean_r(X_te_ben, proj_full)
    f_at_h = feat_hps_mean_r(X_te_atk, proj_full)
    a, t, f = evaluate(f_tr_h, y_train, f_be_h, f_at_h)
    sd_results.append({"name": "HPS (mean_r, trained)", "n_dim": 1, "auroc": a, "tpr": t, "fpr": f})
    print(f"  {'HPS (mean_r, trained) ★':<35} | {1:>5} | {a:>6.3f} | {t:>6.3f} | {f:>5.3f}")

    # Controls
    controls = [
        ("C1: Raw L2 norm (mean over layers)", feat_raw_norm_scalar, None),
        ("C2: Per-layer L2 norm (6-dim)", feat_raw_norm_perlayer, None),
        ("C3: Random Lorentz mean_r (untrained)", feat_random_lorentz_radius, {"seed": 0}),
        ("C4: LR on mean-pooled activations", feat_mean_pooled_activations, None),
        ("C5: LR on flattened activations", feat_flattened_activations, None),
    ]
    for name, fn, kw in controls:
        try:
            r = eval_control_same_dist(name, fn, X_train, y_train, X_te_ben, X_te_atk, kw)
            sd_results.append(r)
            print(f"  {name:<35} | {r['n_dim']:>5} | {r['auroc']:>6.3f} | {r['tpr']:>6.3f} | {r['fpr']:>5.3f}")
        except Exception as e:
            print(f"  {name:<35} | FAILED: {type(e).__name__}: {e}")

    # ══════════════════════════════════════════════════════════════
    #  PART B: Cold-start regimes
    # ══════════════════════════════════════════════════════════════
    cs_results = {}
    for n_per in [5, 25, 100]:
        print(f"\n{'─'*60}")
        print(f"  PART B: Cold-start cross-attack (N={n_per} per method)")
        print(f"{'─'*60}\n")
        print(f"  {'Control':<35} | {'TPR':>6} | {'FPR':>5}")
        print(f"  {'─'*35}─┼─{'─'*6}─┼─{'─'*5}")

        cs_results[n_per] = []
        # HPS reference (retrained per fold)
        r_hps = eval_hps_cold_start("HPS (mean_r, trained)", X_all_atk, all_methods,
                                     methods_unique, X_all_ben, n_per)
        cs_results[n_per].append(r_hps)
        print(f"  {'HPS (mean_r, trained) ★':<35} | {r_hps['tpr_mean']:>6.3f} | {r_hps['fpr_mean']:>5.3f}±{r_hps['fpr_std']:.3f}")

        # Controls
        for name, fn, kw in controls:
            kw_factory = (lambda s, k=kw: k) if kw else None
            try:
                r = eval_control_cold_start(name, fn, X_all_atk, all_methods, methods_unique,
                                             X_all_ben, n_per, kw_factory)
                cs_results[n_per].append(r)
                print(f"  {name:<35} | {r['tpr_mean']:>6.3f} | {r['fpr_mean']:>5.3f}±{r['fpr_std']:.3f}")
            except Exception as e:
                print(f"  {name:<35} | FAILED: {type(e).__name__}: {e}")

    # ══════════════════════════════════════════════════════════════
    #  PART C: Vicuna-like regime (4 methods × 25)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  PART C: Vicuna-like regime (4 methods × 25 attacks)")
    print(f"{'─'*60}\n")
    print(f"  {'Control':<35} | {'TPR':>6} | {'FPR':>5}")
    print(f"  {'─'*35}─┼─{'─'*6}─┼─{'─'*5}")

    vicuna_methods = methods_unique[:4]
    print(f"  Methods: {vicuna_methods}\n")

    hs_by_method = {m: [] for m in methods_unique}
    for act, method in zip(X_all_atk, all_methods):
        hs_by_method[method].append(act)
    for m in methods_unique:
        hs_by_method[m] = np.array(hs_by_method[m])

    ben_split = int(0.8 * len(X_all_ben))
    cv_ben_tr = X_all_ben[:ben_split]
    cv_ben_te = X_all_ben[ben_split:]

    sub_atk_by_m = {}
    for m in vicuna_methods:
        avail = hs_by_method[m]
        take = min(25, len(avail))
        sub_atk_by_m[m] = avail[:take]

    vic_results = []

    def vic_eval(feat_fn, kw=None, hps_proj=None):
        tprs, fprs = [], []
        for held_out in vicuna_methods:
            others = [m for m in vicuna_methods if m != held_out]
            train_atk = np.concatenate([sub_atk_by_m[m] for m in others])
            test_atk = sub_atk_by_m[held_out]
            X_tr = np.concatenate([cv_ben_tr, train_atk])
            y_tr = np.array([0]*len(cv_ben_tr) + [1]*len(train_atk))
            if hps_proj == "retrain":
                proj = train_hps(X_tr, y_tr, seed=42)
                f_tr = feat_hps_mean_r(X_tr, proj)
                f_be = feat_hps_mean_r(cv_ben_te, proj)
                f_at = feat_hps_mean_r(test_atk, proj)
            else:
                f_tr = feat_fn(X_tr) if kw is None else feat_fn(X_tr, **kw)
                f_be = feat_fn(cv_ben_te) if kw is None else feat_fn(cv_ben_te, **kw)
                f_at = feat_fn(test_atk) if kw is None else feat_fn(test_atk, **kw)
            _, t, fpr = evaluate(f_tr, y_tr, f_be, f_at)
            tprs.append(t)
            fprs.append(fpr)
        return float(np.mean(tprs)), float(np.mean(fprs))

    t, f = vic_eval(None, None, hps_proj="retrain")
    vic_results.append({"name": "HPS (mean_r, trained)", "tpr": t, "fpr": f})
    print(f"  {'HPS (mean_r, trained) ★':<35} | {t:>6.3f} | {f:>5.3f}")
    for name, fn, kw in controls:
        try:
            t, f = vic_eval(fn, kw)
            vic_results.append({"name": name, "tpr": t, "fpr": f})
            print(f"  {name:<35} | {t:>6.3f} | {f:>5.3f}")
        except Exception as e:
            print(f"  {name:<35} | FAILED: {type(e).__name__}: {e}")

    # ══════════════════════════════════════════════════════════════
    #  PART D: Prompt length analysis (do attacks have longer prompts?)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  PART D: Activation magnitude vs prompt length")
    print(f"{'─'*60}\n")

    # Use raw L2 norm at each layer to check if attacks have systematically larger norms
    # (this is C1 broken down for analysis)
    benign_norms = np.linalg.norm(X_te_ben, axis=2)  # (N_ben, n_layers)
    attack_norms = np.linalg.norm(X_te_atk, axis=2)  # (N_atk, n_layers)

    print(f"  Activation L2 norm by layer (test set):")
    print(f"  {'Layer':<8} | {'Benign mean±std':<20} | {'Attack mean±std':<20} | {'Δ':>7}")
    print(f"  {'─'*8}─┼─{'─'*20}─┼─{'─'*20}─┼─{'─'*7}")
    layer_analysis = []
    for j, l in enumerate(HPS_LAYERS):
        b_mean = benign_norms[:, j].mean()
        b_std = benign_norms[:, j].std()
        a_mean = attack_norms[:, j].mean()
        a_std = attack_norms[:, j].std()
        delta = a_mean - b_mean
        print(f"  {l:<8} | {b_mean:>9.2f} ± {b_std:>5.2f}    | {a_mean:>9.2f} ± {a_std:>5.2f}    | {delta:>+7.2f}")
        layer_analysis.append({"layer": l, "benign_mean": float(b_mean), "benign_std": float(b_std),
                                "attack_mean": float(a_mean), "attack_std": float(a_std)})

    # ══════════════════════════════════════════════════════════════
    #  Decision Summary
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  DECISION SUMMARY")
    print(f"{'═'*60}\n")

    hps_sd_tpr = sd_results[0]["tpr"]
    hps_cs5 = cs_results[5][0]["tpr_mean"]
    hps_cs25 = cs_results[25][0]["tpr_mean"]
    hps_cs100 = cs_results[100][0]["tpr_mean"]
    hps_vic = vic_results[0]["tpr"]

    print(f"  HPS reference:")
    print(f"    Same-dist:        TPR = {hps_sd_tpr:.3f}")
    print(f"    Cold-start N=5:   TPR = {hps_cs5:.3f}")
    print(f"    Cold-start N=25:  TPR = {hps_cs25:.3f}")
    print(f"    Cold-start N=100: TPR = {hps_cs100:.3f}")
    print(f"    Vicuna-like:      TPR = {hps_vic:.3f}")

    print(f"\n  Controls vs HPS (Δ TPR; positive = control matches/exceeds HPS):")
    print(f"  {'Control':<35} | {'Same-dist':>10} | {'CS N=5':>7} | {'CS N=25':>7} | {'CS N=100':>8} | {'Vicuna':>7}")
    print(f"  {'─'*35}─┼─{'─'*10}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*8}─┼─{'─'*7}")
    for i, (name, _, _) in enumerate(controls):
        sd_t = sd_results[i + 1]["tpr"]
        cs5_t = cs_results[5][i + 1]["tpr_mean"]
        cs25_t = cs_results[25][i + 1]["tpr_mean"]
        cs100_t = cs_results[100][i + 1]["tpr_mean"]
        vic_t = vic_results[i + 1]["tpr"]
        d_sd = sd_t - hps_sd_tpr
        d_5 = cs5_t - hps_cs5
        d_25 = cs25_t - hps_cs25
        d_100 = cs100_t - hps_cs100
        d_vic = vic_t - hps_vic
        print(f"  {name:<35} | {d_sd:>+10.3f} | {d_5:>+7.3f} | {d_25:>+7.3f} | {d_100:>+8.3f} | {d_vic:>+7.3f}")

    print(f"\n  ── Verdict ──")
    # Check if any control matches HPS in cold-start regime (within 0.02)
    danger_thresh = -0.02   # control TPR within 0.02 of HPS = "matches"
    cs5_max_control = max(cs_results[5][i+1]["tpr_mean"] for i in range(len(controls)))
    vic_max_control = max(vic_results[i+1]["tpr"] for i in range(len(controls)))
    print(f"  Best control TPR at CS N=5:  {cs5_max_control:.3f}  (HPS: {hps_cs5:.3f})")
    print(f"  Best control TPR at Vicuna:  {vic_max_control:.3f}  (HPS: {hps_vic:.3f})")

    if cs5_max_control >= hps_cs5 - 0.02 and vic_max_control >= hps_vic - 0.02:
        print(f"\n  ⚠ PROBLEM — A simpler control matches HPS at low data.")
        print(f"    The hyperbolic projection may not be doing essential work.")
        print(f"    Reframe needed: 'mean_r' might just be detecting activation magnitude,")
        print(f"    which any L2-based method would catch.")
    elif cs5_max_control >= hps_cs5 - 0.05:
        print(f"\n  ⚠ MARGINAL — Controls come close but don't match HPS.")
        print(f"    Defensible but reviewers may push back.")
        print(f"    Consider testing on Vicuna data too.")
    else:
        print(f"\n  ✓ HPS GENUINELY BETTER")
        print(f"    All simpler controls fail to match HPS in cold-start regime.")
        print(f"    The geometric prior is doing real work, not detecting magnitude.")

    # Save results
    out = {
        "config": {"layers": HPS_LAYERS, "kappa_init": KAPPA_INIT},
        "same_dist": sd_results,
        "cold_start": {str(k): v for k, v in cs_results.items()},
        "vicuna_like": vic_results,
        "layer_norm_analysis": layer_analysis,
    }
    out_path = "results/control_experiments.json"
    os.makedirs("results", exist_ok=True)

    def _np_default(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"Type {type(o)} not serializable")

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=_np_default)
    print(f"\n  Saved → {out_path}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
