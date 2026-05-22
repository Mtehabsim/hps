"""
Cross-Attack Ensemble: Train HPS on 3 attack methods + RTV features, test on held-out 4th.
Shows whether RTV's refusal-direction signal helps where HPS fails (especially PAIR).

Usage:
  python test_cross_attack_ensemble.py \
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

import config
from rtv_standalone import (
    load_model, extract_hidden_states, compute_refusal_directions,
    compute_fingerprint, MODEL_LAYERS, TOKEN_POSITIONS, FPR_TARGET
)
from experiment7 import LorentzProjection, contrastive_loss, extract_trajectory_features
from dataset import BENIGN

HPS_LAYERS = [0, 1, 2, 35, 36, 37, 38, 39]
RTV_LAYERS = MODEL_LAYERS.get("lmsys/vicuna-13b-v1.5", [12, 16, 26])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lmsys/vicuna-13b-v1.5")
    parser.add_argument("--harmless", required=True)
    parser.add_argument("--harmful", required=True)
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"  CROSS-ATTACK ENSEMBLE")
    print(f"  Train on 3 methods, test on held-out 4th")
    print(f"{'═'*60}\n")

    # Load data
    cat_path = os.path.join(config.RESULTS_DIR, "validated_attacks_categorized.json")
    with open(cat_path) as f:
        categorized = json.load(f)

    df_h = pd.read_csv(args.harmless)
    df_harm = pd.read_csv(args.harmful)
    for col in ["prompt", "goal", "text", "instruction", "query"]:
        if col in df_h.columns: harmless_cal = df_h[col].dropna().tolist(); break
    else: harmless_cal = df_h.iloc[:, 0].dropna().tolist()
    for col in ["prompt", "goal", "text", "instruction", "query"]:
        if col in df_harm.columns: harmful_cal = df_harm[col].dropna().tolist(); break
    else: harmful_cal = df_harm.iloc[:, 0].dropna().tolist()

    n_rtv_cal = min(len(harmless_cal), len(harmful_cal))
    benign_prompts = list(BENIGN)
    methods = sorted(categorized.keys())

    print(f"  Methods: {methods}")
    print(f"  Benign: {len(benign_prompts)}")
    print(f"  RTV calibration: {n_rtv_cal}+{n_rtv_cal}")

    # Load model
    model, tokenizer = load_model(args.model)

    # ── Compute RTV refusal directions (once, shared across folds) ──
    print(f"\n[RTV] Computing refusal directions...")
    refusal_dirs = compute_refusal_directions(
        model, tokenizer, harmful_cal[:n_rtv_cal], harmless_cal[:n_rtv_cal],
        RTV_LAYERS, n_rtv_cal
    )

    # ── Extract ALL activations upfront ──
    print(f"\n  Extracting all activations (HPS + RTV)...")

    # Benign: 80/20 split
    rng = np.random.RandomState(42)
    ben_idx = rng.permutation(len(benign_prompts))
    n_ben_tr = int(0.8 * len(ben_idx))
    train_benign = [benign_prompts[i] for i in ben_idx[:n_ben_tr]]
    test_benign = [benign_prompts[i] for i in ben_idx[n_ben_tr:]]

    def extract_both(prompts, label):
        hps_acts, rtv_fps = [], []
        for i, p in enumerate(prompts):
            hs = extract_hidden_states(model, tokenizer, p, list(set(HPS_LAYERS + RTV_LAYERS)))
            hps_acts.append(np.array([hs[l][-1] for l in HPS_LAYERS]))
            rtv_fps.append(compute_fingerprint(hs, refusal_dirs, RTV_LAYERS, TOKEN_POSITIONS))
            if (i+1) % 50 == 0: print(f"    {label}: {i+1}/{len(prompts)}")
        return np.array(hps_acts), np.array(rtv_fps)

    print("  Train benign...")
    hps_ben_tr, rtv_ben_tr = extract_both(train_benign, "train benign")
    print("  Test benign...")
    hps_ben_te, rtv_ben_te = extract_both(test_benign, "test benign")

    # Attacks per method
    hps_by_method, rtv_by_method = {}, {}
    for m in methods:
        print(f"  {m}...")
        hps_by_method[m], rtv_by_method[m] = extract_both(categorized[m], m)

    # ── Cross-attack evaluation ──
    print(f"\n{'─'*60}")
    print(f"  CROSS-ATTACK RESULTS")
    print(f"{'─'*60}")
    print(f"\n  {'Held-out':<28} | {'HPS':>6} | {'RTV':>6} | {'Ensemble':>8} | {'N':>3}")
    print(f"  {'─'*28}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*8}─┼─{'─'*3}")

    n_layers = len(HPS_LAYERS)
    d_hidden = hps_ben_tr.shape[2]
    results = {}

    for held_out in methods:
        # Train attacks = all methods except held_out
        train_methods = [m for m in methods if m != held_out]
        hps_atk_tr = np.concatenate([hps_by_method[m] for m in train_methods])
        rtv_atk_tr = np.concatenate([rtv_by_method[m] for m in train_methods])

        # Test attacks = held_out method
        hps_atk_te = hps_by_method[held_out]
        rtv_atk_te = rtv_by_method[held_out]

        # Train HPS projection
        X_train = np.concatenate([hps_ben_tr, hps_atk_tr])
        y_train = np.array([0]*len(hps_ben_tr) + [1]*len(hps_atk_tr))

        torch.manual_seed(42)
        proj = LorentzProjection(d_hidden, config.PROJECTION_DIM, 1.0, n_layers=n_layers).to("cpu")
        opt = optim.Adam(proj.parameters(), lr=1e-3, weight_decay=1e-5)
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.long)
        for _ in range(120):
            loss = torch.tensor(0.0)
            for l in range(n_layers):
                h = proj(X_t[:, l, :])
                loss = loss + contrastive_loss(h, y_t, k=proj.k, tau=proj.tau(l))
            loss = loss / n_layers
            opt.zero_grad(); loss.backward(); opt.step()
        proj.eval()

        # Extract features
        feats_train = extract_trajectory_features(proj, X_train)
        feats_te_ben = extract_trajectory_features(proj, hps_ben_te)
        feats_te_atk = extract_trajectory_features(proj, hps_atk_te)

        # ── HPS alone ──
        sc_hps = StandardScaler()
        clf_hps = LogisticRegression(max_iter=2000, random_state=42)
        clf_hps.fit(sc_hps.fit_transform(feats_train), y_train)
        hps_ben_scores = clf_hps.predict_proba(sc_hps.transform(feats_te_ben))[:, 1]
        hps_atk_scores = clf_hps.predict_proba(sc_hps.transform(feats_te_atk))[:, 1]
        hps_thr = float(np.quantile(hps_ben_scores, 1.0 - FPR_TARGET))
        hps_tpr = float((hps_atk_scores > hps_thr).mean())

        # ── Ensemble (HPS + RTV features) ──
        rtv_train = np.concatenate([rtv_ben_tr, rtv_atk_tr])
        ens_train = np.concatenate([feats_train, rtv_train], axis=1)
        ens_te_ben = np.concatenate([feats_te_ben, rtv_ben_te], axis=1)
        ens_te_atk = np.concatenate([feats_te_atk, rtv_atk_te], axis=1)

        sc_ens = StandardScaler()
        clf_ens = LogisticRegression(max_iter=2000, random_state=42)
        clf_ens.fit(sc_ens.fit_transform(ens_train), y_train)
        ens_ben_scores = clf_ens.predict_proba(sc_ens.transform(ens_te_ben))[:, 1]
        ens_atk_scores = clf_ens.predict_proba(sc_ens.transform(ens_te_atk))[:, 1]
        ens_thr = float(np.quantile(ens_ben_scores, 1.0 - FPR_TARGET))
        ens_tpr = float((ens_atk_scores > ens_thr).mean())

        # ── RTV alone (threshold on test benign) ──
        from sklearn.covariance import LedoitWolf
        rtv_cal_h = np.array([compute_fingerprint(
            extract_hidden_states(model, tokenizer, p, RTV_LAYERS),
            refusal_dirs, RTV_LAYERS, TOKEN_POSITIONS) for p in harmless_cal[:n_rtv_cal]])
        rtv_cal_harm = np.array([compute_fingerprint(
            extract_hidden_states(model, tokenizer, p, RTV_LAYERS),
            refusal_dirs, RTV_LAYERS, TOKEN_POSITIONS) for p in harmful_cal[:n_rtv_cal]])
        lw_p = LedoitWolf().fit(rtv_cal_h)
        lw_n = LedoitWolf().fit(rtv_cal_harm)

        def rtv_sc(fp):
            dp = np.sqrt(max(0, (fp-lw_p.location_)@lw_p.precision_@(fp-lw_p.location_)))
            dn = np.sqrt(max(0, (fp-lw_n.location_)@lw_n.precision_@(fp-lw_n.location_)))
            return min(dp, dn)

        rtv_ben_s = np.array([rtv_sc(fp) for fp in rtv_ben_te])
        rtv_atk_s = np.array([rtv_sc(fp) for fp in rtv_atk_te])
        rtv_thr = float(np.quantile(rtv_ben_s, 1.0 - FPR_TARGET))
        rtv_tpr = float((rtv_atk_s > rtv_thr).mean())

        n_test = len(hps_atk_te)
        print(f"  {held_out:<28} | {hps_tpr:>6.3f} | {rtv_tpr:>6.3f} | {ens_tpr:>8.3f} | {n_test:>3}")

        results[held_out] = {
            "hps_tpr": float(hps_tpr),
            "rtv_tpr": float(rtv_tpr),
            "ensemble_tpr": float(ens_tpr),
            "n_test": n_test,
        }

    # Summary
    mean_hps = np.mean([r["hps_tpr"] for r in results.values()])
    mean_rtv = np.mean([r["rtv_tpr"] for r in results.values()])
    mean_ens = np.mean([r["ensemble_tpr"] for r in results.values()])
    print(f"  {'─'*28}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*8}─┼─{'─'*3}")
    print(f"  {'MEAN':<28} | {mean_hps:>6.3f} | {mean_rtv:>6.3f} | {mean_ens:>8.3f} |")

    print(f"\n  Δ Ensemble vs HPS: {mean_ens - mean_hps:+.3f}")
    print(f"  Δ Ensemble vs RTV: {mean_ens - mean_rtv:+.3f}")

    # Save
    results["mean"] = {"hps": float(mean_hps), "rtv": float(mean_rtv), "ensemble": float(mean_ens)}
    out_path = os.path.join(config.RESULTS_DIR, "cross_attack_ensemble.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {out_path}")
    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    main()
