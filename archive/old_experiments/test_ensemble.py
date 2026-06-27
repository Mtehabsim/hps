"""
Ensemble: HPS (supervised) + RTV (zero-shot) combined detection.
Tests two fusion strategies:
  Option 1: OR gate — flag if either detector fires
  Option 2: Feature concatenation — 27-dim logistic regression

Usage:
  python test_ensemble.py \
    --harmless JBShield/data/harmless_calibration.csv \
    --harmful JBShield/data/harmful_calibration.csv
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.optim as optim
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.covariance import LedoitWolf

import config
from rtv_standalone import (
    load_model, extract_hidden_states, compute_refusal_directions,
    compute_fingerprint, MODEL_LAYERS, TOKEN_POSITIONS, FPR_TARGET
)
from experiment7 import LorentzProjection, contrastive_loss, extract_trajectory_features
from dataset import BENIGN, ADVERSARIAL

HPS_LAYERS = [0, 1, 2, 35, 36, 37, 38, 39]
RTV_LAYERS = MODEL_LAYERS.get("lmsys/vicuna-13b-v1.5", [12, 16, 26])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lmsys/vicuna-13b-v1.5")
    parser.add_argument("--harmless", required=True, help="CSV for RTV calibration")
    parser.add_argument("--harmful", required=True, help="CSV for RTV calibration")
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"  ENSEMBLE: HPS (supervised) + RTV (zero-shot)")
    print(f"{'═'*60}\n")

    # ── Load RTV calibration data ──
    df_h = pd.read_csv(args.harmless)
    df_harm = pd.read_csv(args.harmful)
    for col in ["prompt", "goal", "text", "instruction", "query"]:
        if col in df_h.columns: harmless_cal = df_h[col].dropna().tolist(); break
    else: harmless_cal = df_h.iloc[:, 0].dropna().tolist()
    for col in ["prompt", "goal", "text", "instruction", "query"]:
        if col in df_harm.columns: harmful_cal = df_harm[col].dropna().tolist(); break
    else: harmful_cal = df_harm.iloc[:, 0].dropna().tolist()

    n_rtv_cal = min(len(harmless_cal), len(harmful_cal))

    # ── Load HPS data (attacks + benign from dataset.py) ──
    cat_path = os.path.join(config.RESULTS_DIR, "validated_attacks_categorized.json")
    with open(cat_path) as f:
        categorized = json.load(f)
    attack_prompts, attack_methods = [], []
    for method, prompts in categorized.items():
        for p in prompts:
            attack_prompts.append(p); attack_methods.append(method)

    benign_prompts = list(BENIGN)

    # 80/20 split
    rng = np.random.RandomState(42)
    ben_idx = rng.permutation(len(benign_prompts))
    atk_idx = rng.permutation(len(attack_prompts))
    n_ben_tr = int(0.8 * len(ben_idx))
    n_atk_tr = int(0.8 * len(atk_idx))

    train_benign = [benign_prompts[i] for i in ben_idx[:n_ben_tr]]
    test_benign = [benign_prompts[i] for i in ben_idx[n_ben_tr:]]
    train_attacks = [attack_prompts[i] for i in atk_idx[:n_atk_tr]]
    test_attacks = [attack_prompts[i] for i in atk_idx[n_atk_tr:]]
    test_methods = [attack_methods[i] for i in atk_idx[n_atk_tr:]]

    print(f"  RTV calibration: {n_rtv_cal} harmless + {n_rtv_cal} harmful")
    print(f"  HPS train: {len(train_benign)} benign + {len(train_attacks)} attacks")
    print(f"  Test: {len(test_benign)} benign + {len(test_attacks)} attacks")

    # ── Load model ──
    model, tokenizer = load_model(args.model)
    device = config.DEVICE

    # ══════════════════════════════════════════════════════════════════════
    #  RTV: Compute refusal directions + fingerprints (zero-shot)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[RTV] Computing refusal directions...")
    refusal_dirs = compute_refusal_directions(
        model, tokenizer, harmful_cal[:n_rtv_cal], harmless_cal[:n_rtv_cal],
        RTV_LAYERS, n_rtv_cal
    )

    print(f"[RTV] Extracting fingerprints for all prompts...")
    def get_rtv_fp(prompt):
        hs = extract_hidden_states(model, tokenizer, prompt, RTV_LAYERS)
        return compute_fingerprint(hs, refusal_dirs, RTV_LAYERS, TOKEN_POSITIONS)

    # RTV fingerprints for train and test
    print("  Train benign RTV...")
    rtv_train_ben = []
    for i, p in enumerate(train_benign):
        rtv_train_ben.append(get_rtv_fp(p))
        if (i+1) % 50 == 0: print(f"    train benign: {i+1}/{len(train_benign)}")
    rtv_train_ben = np.array(rtv_train_ben)

    print("  Train attacks RTV...")
    rtv_train_atk = []
    for i, p in enumerate(train_attacks):
        rtv_train_atk.append(get_rtv_fp(p))
        if (i+1) % 50 == 0: print(f"    train attacks: {i+1}/{len(train_attacks)}")
    rtv_train_atk = np.array(rtv_train_atk)

    print("  Test benign RTV...")
    rtv_test_ben = []
    for i, p in enumerate(test_benign):
        rtv_test_ben.append(get_rtv_fp(p))
        if (i+1) % 50 == 0: print(f"    test benign: {i+1}/{len(test_benign)}")
    rtv_test_ben = np.array(rtv_test_ben)

    print("  Test attacks RTV...")
    rtv_test_atk = []
    for i, p in enumerate(test_attacks):
        rtv_test_atk.append(get_rtv_fp(p))
        if (i+1) % 20 == 0: print(f"    test attacks: {i+1}/{len(test_attacks)}")
    rtv_test_atk = np.array(rtv_test_atk)

    # ══════════════════════════════════════════════════════════════════════
    #  HPS: Train projection + extract trajectory features (supervised)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[HPS] Extracting activations at layers {HPS_LAYERS}...")

    def get_hps_acts(prompts, label):
        acts = []
        for i, p in enumerate(prompts):
            hs = extract_hidden_states(model, tokenizer, p, HPS_LAYERS)
            acts.append(np.array([hs[l][-1] for l in HPS_LAYERS]))
            if (i+1) % 50 == 0: print(f"    {label}: {i+1}/{len(prompts)}")
        return np.array(acts)

    X_tr_ben = get_hps_acts(train_benign, "train benign")
    X_tr_atk = get_hps_acts(train_attacks, "train attacks")
    X_te_ben = get_hps_acts(test_benign, "test benign")
    X_te_atk = get_hps_acts(test_attacks, "test attacks")

    X_train = np.concatenate([X_tr_ben, X_tr_atk])
    y_train = np.array([0]*len(X_tr_ben) + [1]*len(X_tr_atk))

    print(f"\n[HPS] Training Lorentz projection...")
    n_layers = len(HPS_LAYERS)
    d_hidden = X_train.shape[2]
    torch.manual_seed(42)
    proj = LorentzProjection(d_hidden, config.PROJECTION_DIM, 1.0, n_layers=n_layers).to("cpu")
    opt = optim.Adam(proj.parameters(), lr=1e-3, weight_decay=1e-5)
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)

    for epoch in range(120):
        loss = torch.tensor(0.0)
        for l in range(n_layers):
            h = proj(X_t[:, l, :])
            loss = loss + contrastive_loss(h, y_t, k=proj.k, tau=proj.tau(l))
        loss = loss / n_layers
        opt.zero_grad(); loss.backward(); opt.step()
        if (epoch+1) % 40 == 0:
            print(f"    Epoch {epoch+1}/120 loss={loss.item():.4f}")
    proj.eval()

    # Extract trajectory features
    hps_train = extract_trajectory_features(proj, X_train)
    hps_test_ben = extract_trajectory_features(proj, X_te_ben)
    hps_test_atk = extract_trajectory_features(proj, X_te_atk)

    # ══════════════════════════════════════════════════════════════════════
    #  OPTION 1: OR Gate
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  OPTION 1: OR Gate (flag if either fires)")
    print(f"{'─'*60}")

    # HPS standalone classifier
    sc_hps = StandardScaler()
    hps_train_s = sc_hps.fit_transform(hps_train)
    clf_hps = LogisticRegression(max_iter=2000, random_state=42)
    clf_hps.fit(hps_train_s, y_train)

    hps_scores_ben = clf_hps.predict_proba(sc_hps.transform(hps_test_ben))[:, 1]
    hps_scores_atk = clf_hps.predict_proba(sc_hps.transform(hps_test_atk))[:, 1]
    hps_thr = float(np.quantile(hps_scores_ben, 1.0 - FPR_TARGET))

    # RTV standalone (Mahalanobis on calibration)
    # Fit on RTV calibration data (harmless + harmful from CSV)
    rtv_cal_harmless = []
    for p in harmless_cal[:n_rtv_cal]:
        rtv_cal_harmless.append(get_rtv_fp(p))
    rtv_cal_harmless = np.array(rtv_cal_harmless)
    rtv_cal_harmful = []
    for p in harmful_cal[:n_rtv_cal]:
        rtv_cal_harmful.append(get_rtv_fp(p))
    rtv_cal_harmful = np.array(rtv_cal_harmful)

    lw_pos = LedoitWolf().fit(rtv_cal_harmless)
    lw_neg = LedoitWolf().fit(rtv_cal_harmful)

    def rtv_score(fp):
        d_p = np.sqrt(max(0, (fp - lw_pos.location_) @ lw_pos.precision_ @ (fp - lw_pos.location_)))
        d_n = np.sqrt(max(0, (fp - lw_neg.location_) @ lw_neg.precision_ @ (fp - lw_neg.location_)))
        return min(d_p, d_n)

    rtv_scores_ben = np.array([rtv_score(fp) for fp in rtv_test_ben])
    rtv_scores_atk = np.array([rtv_score(fp) for fp in rtv_test_atk])
    rtv_thr = float(np.quantile(rtv_scores_ben, 1.0 - FPR_TARGET))

    # OR gate: flag if HPS OR RTV fires
    or_ben = ((hps_scores_ben > hps_thr) | (rtv_scores_ben > rtv_thr)).astype(float)
    or_atk = ((hps_scores_atk > hps_thr) | (rtv_scores_atk > rtv_thr)).astype(float)
    or_fpr = or_ben.mean()
    or_tpr = or_atk.mean()

    # Individual results
    hps_tpr = float((hps_scores_atk > hps_thr).mean())
    hps_fpr = float((hps_scores_ben > hps_thr).mean())
    rtv_tpr = float((rtv_scores_atk > rtv_thr).mean())
    rtv_fpr = float((rtv_scores_ben > rtv_thr).mean())

    hps_auroc = roc_auc_score(np.array([0]*len(hps_scores_ben)+[1]*len(hps_scores_atk)),
                              np.concatenate([hps_scores_ben, hps_scores_atk]))
    rtv_auroc = roc_auc_score(np.array([0]*len(rtv_scores_ben)+[1]*len(rtv_scores_atk)),
                              np.concatenate([rtv_scores_ben, rtv_scores_atk]))

    print(f"\n  {'Method':<25} | {'AUROC':>6} | {'TPR':>6} | {'FPR':>6}")
    print(f"  {'─'*25}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*6}")
    print(f"  {'HPS alone':<25} | {hps_auroc:>6.3f} | {hps_tpr:>6.3f} | {hps_fpr:>6.3f}")
    print(f"  {'RTV alone':<25} | {rtv_auroc:>6.3f} | {rtv_tpr:>6.3f} | {rtv_fpr:>6.3f}")
    print(f"  {'OR gate (either fires)':<25} | {'—':>6} | {or_tpr:>6.3f} | {or_fpr:>6.3f}")

    # Per-method OR gate
    print(f"\n  Per-attack OR gate:")
    for m in sorted(set(test_methods)):
        idx = [i for i, x in enumerate(test_methods) if x == m]
        h_det = (hps_scores_atk[idx] > hps_thr)
        r_det = (rtv_scores_atk[idx] > rtv_thr)
        or_det = (h_det | r_det).mean()
        print(f"    {m:<30}: OR={or_det:.3f}  (HPS={h_det.mean():.3f}, RTV={r_det.mean():.3f})")

    # ══════════════════════════════════════════════════════════════════════
    #  OPTION 2: Feature Concatenation (27-dim logistic regression)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  OPTION 2: Feature Concatenation (12 HPS + 15 RTV = 27 features)")
    print(f"{'─'*60}")

    # Concatenate features
    rtv_train = np.concatenate([rtv_train_ben, rtv_train_atk])
    ens_train = np.concatenate([hps_train, rtv_train], axis=1)
    ens_test_ben = np.concatenate([hps_test_ben, rtv_test_ben], axis=1)
    ens_test_atk = np.concatenate([hps_test_atk, rtv_test_atk], axis=1)

    sc_ens = StandardScaler()
    ens_train_s = sc_ens.fit_transform(ens_train)
    ens_test_ben_s = sc_ens.transform(ens_test_ben)
    ens_test_atk_s = sc_ens.transform(ens_test_atk)

    clf_ens = LogisticRegression(max_iter=2000, random_state=42)
    clf_ens.fit(ens_train_s, y_train)

    ens_scores_ben = clf_ens.predict_proba(ens_test_ben_s)[:, 1]
    ens_scores_atk = clf_ens.predict_proba(ens_test_atk_s)[:, 1]

    ens_thr = float(np.quantile(ens_scores_ben, 1.0 - FPR_TARGET))
    ens_tpr = float((ens_scores_atk > ens_thr).mean())
    ens_fpr = float((ens_scores_ben > ens_thr).mean())
    ens_auroc = roc_auc_score(np.array([0]*len(ens_scores_ben)+[1]*len(ens_scores_atk)),
                              np.concatenate([ens_scores_ben, ens_scores_atk]))

    print(f"\n  {'Method':<25} | {'AUROC':>6} | {'TPR@5%':>7} | {'FPR':>6}")
    print(f"  {'─'*25}─┼─{'─'*6}─┼─{'─'*7}─┼─{'─'*6}")
    print(f"  {'HPS alone':<25} | {hps_auroc:>6.3f} | {hps_tpr:>7.3f} | {hps_fpr:>6.3f}")
    print(f"  {'RTV alone':<25} | {rtv_auroc:>6.3f} | {rtv_tpr:>7.3f} | {rtv_fpr:>6.3f}")
    print(f"  {'Ensemble (concat)':<25} | {ens_auroc:>6.3f} | {ens_tpr:>7.3f} | {ens_fpr:>6.3f}")

    # Per-method ensemble
    print(f"\n  Per-attack-type:")
    print(f"  {'Method':<30} | {'N':>3} | {'HPS':>5} | {'RTV':>5} | {'OR':>5} | {'Ensemble':>8}")
    print(f"  {'─'*30}─┼─{'─'*3}─┼─{'─'*5}─┼─{'─'*5}─┼─{'─'*5}─┼─{'─'*8}")
    for m in sorted(set(test_methods)):
        idx = [i for i, x in enumerate(test_methods) if x == m]
        h_t = float((hps_scores_atk[idx] > hps_thr).mean())
        r_t = float((rtv_scores_atk[idx] > rtv_thr).mean())
        o_t = float(((hps_scores_atk[idx] > hps_thr) | (rtv_scores_atk[idx] > rtv_thr)).mean())
        e_t = float((ens_scores_atk[idx] > ens_thr).mean())
        print(f"  {m:<30} | {len(idx):>3} | {h_t:>5.3f} | {r_t:>5.3f} | {o_t:>5.3f} | {e_t:>8.3f}")

    # ── Save ──
    results = {
        "hps": {"auroc": float(hps_auroc), "tpr": float(hps_tpr), "fpr": float(hps_fpr)},
        "rtv": {"auroc": float(rtv_auroc), "tpr": float(rtv_tpr), "fpr": float(rtv_fpr)},
        "or_gate": {"tpr": float(or_tpr), "fpr": float(or_fpr)},
        "ensemble": {"auroc": float(ens_auroc), "tpr": float(ens_tpr), "fpr": float(ens_fpr)},
    }
    out_path = os.path.join(config.RESULTS_DIR, "ensemble_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {out_path}")
    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    main()
