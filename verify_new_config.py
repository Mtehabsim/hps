"""
Verification script for the updated HPS configuration.

Confirms that the new hyperparameters from the diagnostic findings:
  - HPS_LAYERS = [0, 2, 17, 24, 28, 31]   (spread layers, TEST 7)
  - κ_init = 0.1, frozen                   (TEST 9 best)
  - 50 epochs, no early stopping           (TEST 5 finding)

give:
  Part A — HPS ≈ AUROC=1.0 on Llama-3-8B same-distribution (matches Euclidean)
  Part B — Cold-start curve still shows large HPS advantage at low N per method

Uses cached activations only — no LLM forward passes needed.
Runtime: ~10-15 minutes.

Usage:
  python verify_new_config.py \
    --test-attacks llama3_attacks.json \
    --harmless data_harmless_6500.csv
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

# ── New config from diagnostic findings ──
HPS_LAYERS = [0, 2, 17, 24, 28, 31]   # spread (TEST 7 best)
KAPPA_INIT = 0.1                       # TEST 9 best
FREEZE_KAPPA = True                    # TEST 9 — learnable κ=0.1 was unstable
EPOCHS = 50                            # TEST 5 — past 50 epochs HPS overfits
device = "cuda" if torch.cuda.is_available() else "cpu"


def train_hps(X_train, y_train, seed=42):
    """Train HPS with the new optimal config. Returns trained projection."""
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


def train_euclidean(X_train, y_train, seed=42, max_epochs=200, patience=20):
    """Train parameter-matched Euclidean baseline."""
    n_layers = X_train.shape[1]
    d_hidden = X_train.shape[2]
    torch.manual_seed(seed)
    np.random.seed(seed)
    proj_e = nn.Linear(d_hidden, 64, bias=False).to(device)
    nn.init.xavier_uniform_(proj_e.weight)
    scale_per_layer = nn.Parameter(torch.ones(n_layers, device=device) / 8.0)
    log_margin = nn.Parameter(torch.tensor(np.log(2.0), device=device))
    opt = optim.Adam(list(proj_e.parameters()) + [scale_per_layer, log_margin],
                     lr=1e-3, weight_decay=1e-5)
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)
    best = float('inf'); pat = 0
    for _ in range(max_epochs):
        loss = torch.tensor(0.0, device=device)
        margin = torch.exp(log_margin).clamp(0.5, 5.0)
        for l in range(n_layers):
            h = proj_e(X_t[:, l, :]) * scale_per_layer[l]
            dists = torch.cdist(h, h)
            sm = (y_t.unsqueeze(0) == y_t.unsqueeze(1)).float()
            dm = 1.0 - sm
            tr = torch.triu(torch.ones(h.shape[0], h.shape[0], device=device), diagonal=1)
            ns = (sm * tr).sum().clamp(min=1)
            nd = (dm * tr).sum().clamp(min=1)
            loss = loss + ((dists**2 * sm * tr).sum()/ns +
                           (torch.clamp(margin - dists, min=0)**2 * dm * tr).sum()/nd) / 2
        loss = loss / n_layers
        opt.zero_grad(); loss.backward(); opt.step()
        if loss.item() < best - 1e-4: best = loss.item(); pat = 0
        else: pat += 1
        if pat >= patience: break
    proj_e.eval()
    return proj_e, scale_per_layer


def euc_feats(proj_e, scale_per_layer, X):
    """Euclidean trajectory features."""
    n_layers = X.shape[1]
    feats = []
    with torch.no_grad():
        for i in range(len(X)):
            x = torch.tensor(X[i], dtype=torch.float32, device=device)
            pts = []
            for l in range(n_layers):
                pts.append((proj_e(x[l:l+1]) * scale_per_layer[l]).squeeze(0).cpu().numpy())
            h = np.array(pts)
            norms = np.linalg.norm(h, axis=1)
            d_total = float(np.linalg.norm(h[-1] - h[0]))
            path_len = sum(float(np.linalg.norm(h[j+1] - h[j])) for j in range(n_layers - 1))
            curvs = []
            for j in range(1, n_layers - 1):
                dp = float(np.linalg.norm(h[j] - h[j-1]))
                dn = float(np.linalg.norm(h[j+1] - h[j]))
                ds = float(np.linalg.norm(h[j+1] - h[j-1]))
                curvs.append(abs(dp + dn - ds) / (dp + dn + 1e-8))
            curvs = np.array(curvs) if curvs else np.array([0.0])
            feats.append([
                norms.mean(), norms.max(), norms.min(), norms.std(), norms.max() - norms.min(),
                curvs.max(), curvs.mean(), curvs.std() if len(curvs) > 1 else 0,
                np.argmax(curvs) / max(len(curvs), 1),
                d_total, path_len, d_total / (path_len + 1e-8),
            ])
    return np.array(feats)


def evaluate(feats_train, y_train, feats_te_ben, feats_te_atk, seed=42):
    """Train classifier, return (auroc, tpr@5%FPR, actual_fpr) with calibration split."""
    sc = StandardScaler()
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(sc.fit_transform(feats_train), y_train)
    n_calib = len(feats_te_ben) // 2
    s_calib = clf.predict_proba(sc.transform(feats_te_ben[:n_calib]))[:, 1]
    s_ben = clf.predict_proba(sc.transform(feats_te_ben[n_calib:]))[:, 1]
    s_atk = clf.predict_proba(sc.transform(feats_te_atk))[:, 1]
    thr = float(np.quantile(s_calib, 1.0 - FPR_TARGET))
    tpr = float((s_atk > thr).mean())
    fpr = float((s_ben > thr).mean())
    auroc = roc_auc_score(np.array([0]*len(s_ben) + [1]*len(s_atk)),
                          np.concatenate([s_ben, s_atk]))
    return auroc, tpr, fpr, s_atk, thr


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-attacks", required=True)
    parser.add_argument("--harmless", required=True)
    parser.add_argument("--cache", default="results/llama3_activations_cache.npz")
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"  VERIFICATION — New HPS Config")
    print(f"  Layers:  {HPS_LAYERS}")
    print(f"  κ:       {KAPPA_INIT} ({'frozen' if FREEZE_KAPPA else 'learnable'})")
    print(f"  Epochs:  {EPOCHS}")
    print(f"{'═'*60}\n")

    if not os.path.exists(args.cache):
        print(f"ERROR: Cache not found at {args.cache}")
        return

    print("  Loading cached activations...")
    cache = np.load(args.cache, allow_pickle=True)
    hs_train_ben = cache["hs_train_ben"].tolist()
    hs_train_atk = cache["hs_train_atk"].tolist()
    hs_test_ben = cache["hs_test_ben"].tolist()
    hs_test_atk = cache["hs_test_atk"].tolist()

    # Verify cache has all spread layers
    sample_keys = sorted(hs_train_ben[0].keys())
    missing = [l for l in HPS_LAYERS if l not in sample_keys]
    if missing:
        print(f"ERROR: Cache missing layers {missing}. Cache has {sample_keys}")
        return

    def to_hps(hs_list):
        return np.array([[hs[l][-1] for l in HPS_LAYERS] for hs in hs_list])

    X_tr_ben = to_hps(hs_train_ben)
    X_tr_atk = to_hps(hs_train_atk)
    X_te_ben = to_hps(hs_test_ben)
    X_te_atk = to_hps(hs_test_atk)

    # Load attack methods
    with open(args.test_attacks) as f:
        categorized = json.load(f)
    attack_methods = []
    for method, prompts in categorized.items():
        for p in prompts:
            if p:
                attack_methods.append(method)
    rng = np.random.RandomState(42)
    atk_idx = rng.permutation(len(attack_methods))
    n_atk_tr = int(0.8 * len(atk_idx))
    test_methods = [attack_methods[i] for i in atk_idx[n_atk_tr:]]
    methods_unique = sorted(set(attack_methods))

    print(f"  Train: {len(X_tr_ben)} benign + {len(X_tr_atk)} attacks")
    print(f"  Test:  {len(X_te_ben)} benign + {len(X_te_atk)} attacks")
    print(f"  Methods: {len(methods_unique)}")

    # ══════════════════════════════════════════════════════════════
    #  PART A: Full-data same-distribution comparison
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  PART A: Same-distribution full-data sanity check")
    print(f"  Expectation: HPS AUROC ~1.0, matching Euclidean")
    print(f"{'─'*60}\n")

    X_train = np.concatenate([X_tr_ben, X_tr_atk])
    y_train = np.array([0]*len(X_tr_ben) + [1]*len(X_tr_atk))

    print("  Training HPS (κ=0.1 frozen, spread layers, 50 epochs)...")
    proj_h = train_hps(X_train, y_train, seed=42)
    feats_h_train = extract_trajectory_features(proj_h, X_train)
    feats_h_ben = extract_trajectory_features(proj_h, X_te_ben)
    feats_h_atk = extract_trajectory_features(proj_h, X_te_atk)
    auroc_h, tpr_h, fpr_h, scores_h_atk, thr_h = evaluate(feats_h_train, y_train, feats_h_ben, feats_h_atk)

    print("  Training Euclidean baseline (parameter-matched)...")
    proj_e, scale_e = train_euclidean(X_train, y_train, seed=42)
    feats_e_train = euc_feats(proj_e, scale_e, X_train)
    feats_e_ben = euc_feats(proj_e, scale_e, X_te_ben)
    feats_e_atk = euc_feats(proj_e, scale_e, X_te_atk)
    auroc_e, tpr_e, fpr_e, scores_e_atk, thr_e = evaluate(feats_e_train, y_train, feats_e_ben, feats_e_atk)

    print(f"\n  {'Method':<12} | {'AUROC':>6} | {'TPR@5%':>7} | {'FPR':>6}")
    print(f"  {'─'*12}─┼─{'─'*6}─┼─{'─'*7}─┼─{'─'*6}")
    print(f"  {'HPS (new)':<12} | {auroc_h:>6.3f} | {tpr_h:>7.3f} | {fpr_h:>6.3f}")
    print(f"  {'Euclidean':<12} | {auroc_e:>6.3f} | {tpr_e:>7.3f} | {fpr_e:>6.3f}")
    print(f"  {'Δ':<12} | {auroc_h-auroc_e:>+6.3f} | {tpr_h-tpr_e:>+7.3f} | {'-':>6}")

    # Per-attack breakdown for HPS
    print(f"\n  Per-attack-type (HPS):")
    print(f"  {'Method':<12} | {'N':>4} | {'TPR':>6}")
    print(f"  {'─'*12}─┼─{'─'*4}─┼─{'─'*6}")
    for m in sorted(set(test_methods)):
        idx = [i for i, x in enumerate(test_methods) if x == m]
        if not idx:
            continue
        m_tpr = float((scores_h_atk[idx] > thr_h).mean())
        print(f"  {m:<12} | {len(idx):>4} | {m_tpr:>6.3f}")

    # ══════════════════════════════════════════════════════════════
    #  PART B: Cold-start curve (TEST 2 with new config)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  PART B: Cold-start curve (cross-attack, vary N per method)")
    print(f"  Expectation: HPS dominates at low N, gap shrinks at high N")
    print(f"{'─'*60}\n")

    # Group attacks by method
    X_all_atk = np.concatenate([X_tr_atk, X_te_atk])
    all_methods = [attack_methods[i] for i in atk_idx]
    hs_by_method = {m: [] for m in methods_unique}
    for act, method in zip(X_all_atk, all_methods):
        hs_by_method[method].append(act)
    for m in methods_unique:
        hs_by_method[m] = np.array(hs_by_method[m])

    X_all_ben = np.concatenate([X_tr_ben, X_te_ben])
    ben_split = int(0.8 * len(X_all_ben))
    cv_ben_tr = X_all_ben[:ben_split]
    cv_ben_te = X_all_ben[ben_split:]

    def cross_attack_eval(per_method_n, euclidean=False):
        """Leave-one-method-out cross-attack at fixed N per method."""
        # Subsample each method
        sub_atk_by_m = {}
        for m in methods_unique:
            avail = hs_by_method[m]
            take = min(per_method_n, len(avail))
            sub_atk_by_m[m] = avail[:take]

        tprs = []
        for held_out in methods_unique:
            train_atk = np.concatenate([sub_atk_by_m[m] for m in methods_unique if m != held_out])
            test_atk = sub_atk_by_m[held_out]
            if len(test_atk) < 5:
                continue
            X_tr = np.concatenate([cv_ben_tr, train_atk])
            y_tr = np.array([0]*len(cv_ben_tr) + [1]*len(train_atk))
            if euclidean:
                proj_x, scale_x = train_euclidean(X_tr, y_tr, seed=42)
                f_tr = euc_feats(proj_x, scale_x, X_tr)
                f_be = euc_feats(proj_x, scale_x, cv_ben_te)
                f_at = euc_feats(proj_x, scale_x, test_atk)
            else:
                proj_x = train_hps(X_tr, y_tr, seed=42)
                f_tr = extract_trajectory_features(proj_x, X_tr)
                f_be = extract_trajectory_features(proj_x, cv_ben_te)
                f_at = extract_trajectory_features(proj_x, test_atk)
            _, t, _, _, _ = evaluate(f_tr, y_tr, f_be, f_at)
            tprs.append(t)
        return float(np.mean(tprs)) if tprs else 0.0

    sizes = [25, 50, 100, 250, 500]
    print(f"  {'N/method':<9} | {'HPS TPR':>7} | {'Euc TPR':>7} | {'Δ':>7}")
    print(f"  {'─'*9}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}")
    cold_start_results = []
    for n in sizes:
        h_t = cross_attack_eval(n, euclidean=False)
        e_t = cross_attack_eval(n, euclidean=True)
        print(f"  {n:<9} | {h_t:>7.3f} | {e_t:>7.3f} | {h_t-e_t:>+7.3f}")
        cold_start_results.append({"n": n, "hps": h_t, "euc": e_t, "delta": h_t - e_t})

    # ══════════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  SUMMARY")
    print(f"{'═'*60}")
    print(f"  Same-distribution (full data, 9 methods, 5216 attacks):")
    print(f"    HPS:       AUROC={auroc_h:.3f}  TPR={tpr_h:.3f}")
    print(f"    Euclidean: AUROC={auroc_e:.3f}  TPR={tpr_e:.3f}")
    print(f"    Δ AUROC:   {auroc_h-auroc_e:+.3f}")
    print(f"")
    print(f"  Cold-start (cross-attack, 25 examples/method):")
    cs25 = cold_start_results[0]
    print(f"    HPS:       TPR={cs25['hps']:.3f}")
    print(f"    Euclidean: TPR={cs25['euc']:.3f}")
    print(f"    Δ:         {cs25['delta']:+.3f}")
    print(f"")
    print(f"  Cold-start (cross-attack, 500 examples/method):")
    cs500 = cold_start_results[-1]
    print(f"    HPS:       TPR={cs500['hps']:.3f}")
    print(f"    Euclidean: TPR={cs500['euc']:.3f}")
    print(f"    Δ:         {cs500['delta']:+.3f}")
    print(f"")
    if auroc_h >= 0.99 and cs25['delta'] >= 0.3:
        print(f"  ✓ VERIFICATION PASSED")
        print(f"    - HPS matches Euclidean on full data (AUROC≈1.0)")
        print(f"    - HPS dominates Euclidean at low data (Δ ≥ 0.3 at 25/method)")
        print(f"    - Ready to commit to new defaults and re-run main pipeline")
    else:
        print(f"  ⚠ VERIFICATION INCOMPLETE")
        if auroc_h < 0.99:
            print(f"    - HPS AUROC {auroc_h:.3f} < expected ~0.99")
        if cs25['delta'] < 0.3:
            print(f"    - Cold-start Δ {cs25['delta']:.3f} < expected ≥ 0.3")
        print(f"    Investigate before committing.")
    print(f"{'═'*60}\n")

    # Save results
    out = {
        "config": {
            "layers": HPS_LAYERS,
            "kappa_init": KAPPA_INIT,
            "freeze_kappa": FREEZE_KAPPA,
            "epochs": EPOCHS,
        },
        "same_dist": {
            "hps": {"auroc": float(auroc_h), "tpr": float(tpr_h), "fpr": float(fpr_h)},
            "euc": {"auroc": float(auroc_e), "tpr": float(tpr_e), "fpr": float(fpr_e)},
        },
        "cold_start": cold_start_results,
    }
    out_path = "results/verify_new_config.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
