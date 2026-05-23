"""
HPS vs RTV on Llama-3-8B — apples-to-apples comparison.
Same test set for both methods. Single output table.

Usage:
  python hps_llama3.py --test-attacks llama3_attacks.json \
    --harmless data_harmless_100.csv --harmful data_harmful_100.csv
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

from rtv_standalone import (
    load_model, extract_hidden_states, compute_refusal_directions,
    compute_fingerprint, TOKEN_POSITIONS, FPR_TARGET
)
from experiment7 import LorentzProjection, contrastive_loss, extract_trajectory_features

HPS_LAYERS = [0, 1, 2, 28, 29, 30, 31]
RTV_LAYERS = [17, 24, 31]
ALL_LAYERS = sorted(set(HPS_LAYERS + RTV_LAYERS))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--test-attacks", required=True)
    parser.add_argument("--harmless", required=True)
    parser.add_argument("--harmful", required=True)
    parser.add_argument("--n-cal", type=int, default=100)
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"  HPS vs RTV — Llama-3-8B (apples-to-apples)")
    print(f"  HPS layers: {HPS_LAYERS}")
    print(f"  RTV layers: {RTV_LAYERS}")
    print(f"{'═'*60}\n")

    # Load data
    with open(args.test_attacks) as f:
        categorized = json.load(f)
    attack_prompts, attack_methods = [], []
    for method, prompts in categorized.items():
        for p in prompts:
            if p:
                attack_prompts.append(p)
                attack_methods.append(method)

    df_h = pd.read_csv(args.harmless)
    df_harm = pd.read_csv(args.harmful)
    for col in ["prompt", "goal", "text", "instruction", "query"]:
        if col in df_h.columns: harmless = df_h[col].dropna().tolist(); break
    else: harmless = df_h.iloc[:, 0].dropna().tolist()
    for col in ["prompt", "goal", "text", "instruction", "query"]:
        if col in df_harm.columns: harmful = df_harm[col].dropna().tolist(); break
    else: harmful = df_harm.iloc[:, 0].dropna().tolist()

    n_cal = min(args.n_cal, len(harmless), len(harmful))
    print(f"  Attacks: {len(attack_prompts)} across {len(set(attack_methods))} methods")
    print(f"  RTV calibration: {n_cal}+{n_cal}")

    # 80/20 split on attacks; use harmless as benign for HPS
    rng = np.random.RandomState(42)
    atk_idx = rng.permutation(len(attack_prompts))
    n_atk_tr = int(0.8 * len(atk_idx))
    train_atk = [attack_prompts[i] for i in atk_idx[:n_atk_tr]]
    test_atk = [attack_prompts[i] for i in atk_idx[n_atk_tr:]]
    test_methods = [attack_methods[i] for i in atk_idx[n_atk_tr:]]

    # Use harmless as benign (split 80/20)
    ben_idx = rng.permutation(len(harmless))
    n_ben_tr = int(0.8 * len(ben_idx))
    train_ben = [harmless[i] for i in ben_idx[:n_ben_tr]]
    test_ben = [harmless[i] for i in ben_idx[n_ben_tr:]]

    print(f"  HPS train: {len(train_ben)} benign + {len(train_atk)} attacks")
    print(f"  Test (shared): {len(test_ben)} benign + {len(test_atk)} attacks")

    # Load model
    model, tokenizer = load_model(args.model)

    # ── Extract ALL activations (HPS + RTV layers) in one pass ──
    # Cache to disk to avoid re-extraction
    cache_path = "results/llama3_activations_cache.npz"

    # Cache validation: invalidate if config changes
    import hashlib
    _cfg_str = f"{ALL_LAYERS}|{args.test_attacks}|{len(train_ben)}|{len(train_atk)}"
    _cfg_hash = hashlib.md5(_cfg_str.encode()).hexdigest()[:8]

    def extract_all(prompts, label):
        results = []
        for i, p in enumerate(prompts):
            hs = extract_hidden_states(model, tokenizer, p, ALL_LAYERS)
            results.append(hs)
            if (i+1) % 100 == 0:
                print(f"    {label}: {i+1}/{len(prompts)}")
        return results

    def to_hps_array(hs_list):
        return np.array([[hs[l][-1] for l in HPS_LAYERS] for hs in hs_list])

    # RTV: compute refusal directions
    print("  RTV calibration (refusal directions)...")
    refusal_dirs = compute_refusal_directions(
        model, tokenizer, harmful[:n_cal], harmless[:n_cal], RTV_LAYERS, n_cal
    )

    # RTV: fit Mahalanobis on calibration
    print("  RTV: extracting calibration fingerprints...")
    fps_cal_harmless = []
    for i, p in enumerate(harmless[:n_cal]):
        hs = extract_hidden_states(model, tokenizer, p, RTV_LAYERS)
        fps_cal_harmless.append(compute_fingerprint(hs, refusal_dirs, RTV_LAYERS, TOKEN_POSITIONS))
    fps_cal_harmful = []
    for i, p in enumerate(harmful[:n_cal]):
        hs = extract_hidden_states(model, tokenizer, p, RTV_LAYERS)
        fps_cal_harmful.append(compute_fingerprint(hs, refusal_dirs, RTV_LAYERS, TOKEN_POSITIONS))
    fps_cal_harmless = np.array(fps_cal_harmless)
    fps_cal_harmful = np.array(fps_cal_harmful)

    lw_pos = LedoitWolf().fit(fps_cal_harmless)
    lw_neg = LedoitWolf().fit(fps_cal_harmful)

    def rtv_score(fp):
        dp = np.sqrt(max(0, (fp - lw_pos.location_) @ lw_pos.precision_ @ (fp - lw_pos.location_)))
        dn = np.sqrt(max(0, (fp - lw_neg.location_) @ lw_neg.precision_ @ (fp - lw_neg.location_)))
        return min(dp, dn)

    if os.path.exists(cache_path):
        cache = np.load(cache_path, allow_pickle=True)
        cached_hash = str(cache.get("cfg_hash", "")) if "cfg_hash" in cache else ""
        if cached_hash == _cfg_hash:
            print(f"  Loading cached activations from {cache_path} (hash={_cfg_hash})")
            hs_train_ben = cache["hs_train_ben"].tolist()
            hs_train_atk = cache["hs_train_atk"].tolist()
            hs_test_ben = cache["hs_test_ben"].tolist()
            hs_test_atk = cache["hs_test_atk"].tolist()
        else:
            print(f"  Cache stale (expected {_cfg_hash}, got {cached_hash}). Re-extracting...")
            os.remove(cache_path)
    if not os.path.exists(cache_path):
        print(f"\n  Extracting activations (all layers: {ALL_LAYERS})...")
        print("  Train benign...")
        hs_train_ben = extract_all(train_ben, "train benign")
        print("  Train attacks...")
        hs_train_atk = extract_all(train_atk, "train attacks")
        print("  Test benign...")
        hs_test_ben = extract_all(test_ben, "test benign")
        print("  Test attacks...")
        hs_test_atk = extract_all(test_atk, "test attacks")

        # Save cache
        np.savez(cache_path,
                 hs_train_ben=np.array(hs_train_ben, dtype=object),
                 hs_train_atk=np.array(hs_train_atk, dtype=object),
                 hs_test_ben=np.array(hs_test_ben, dtype=object),
                 hs_test_atk=np.array(hs_test_atk, dtype=object),
                 cfg_hash=np.array(_cfg_hash))
        print(f"  Cached activations → {cache_path} (hash={_cfg_hash})")

    # ── HPS ──
    print(f"\n  Training HPS Lorentz projection...")
    X_tr_ben = to_hps_array(hs_train_ben)
    X_tr_atk = to_hps_array(hs_train_atk)
    X_te_ben = to_hps_array(hs_test_ben)
    X_te_atk = to_hps_array(hs_test_atk)

    X_train = np.concatenate([X_tr_ben, X_tr_atk])
    y_train = np.array([0]*len(X_tr_ben) + [1]*len(X_tr_atk))
    n_layers = len(HPS_LAYERS)
    d_hidden = X_train.shape[2]

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    proj_path = "results/hps_llama3_projection.pt"
    if os.path.exists(proj_path):
        print(f"  Loading saved projection from {proj_path}")
        ckpt = torch.load(proj_path, map_location=device, weights_only=False)
        proj = LorentzProjection(d_hidden, 64, 1.0, n_layers=n_layers).to(device)
        proj.load_state_dict(ckpt["state_dict"])
    else:
        proj = LorentzProjection(d_hidden, 64, 1.0, n_layers=n_layers).to(device)
        opt = optim.Adam(proj.parameters(), lr=1e-3, weight_decay=1e-5)
        X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_train, dtype=torch.long, device=device)

        best_loss = float('inf'); patience_ctr = 0
        for epoch in range(200):
            loss = torch.tensor(0.0, device=device)
            for l in range(n_layers):
                h = proj(X_t[:, l, :])
                loss = loss + contrastive_loss(h, y_t, k=proj.k, tau=proj.tau(l))
            loss = loss / n_layers
            opt.zero_grad(); loss.backward(); opt.step()
            if loss.item() < best_loss - 1e-4:
                best_loss = loss.item(); patience_ctr = 0
            else:
                patience_ctr += 1
            if patience_ctr >= 20:
                print(f"    Early stop at epoch {epoch+1}")
                break
            if (epoch+1) % 50 == 0:
                print(f"    Epoch {epoch+1}/200 loss={loss.item():.4f}")

        torch.save({
            "state_dict": proj.state_dict(),
            "d_in": d_hidden, "d_proj": 64, "n_layers": n_layers, "layers": HPS_LAYERS,
        }, proj_path)
        print(f"  Saved → {proj_path}")
    proj.eval()

    feats_train = extract_trajectory_features(proj, X_train)
    feats_te_ben = extract_trajectory_features(proj, X_te_ben)
    feats_te_atk = extract_trajectory_features(proj, X_te_atk)

    sc = StandardScaler()
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(sc.fit_transform(feats_train), y_train)
    hps_scores_ben = clf.predict_proba(sc.transform(feats_te_ben))[:, 1]
    hps_scores_atk = clf.predict_proba(sc.transform(feats_te_atk))[:, 1]

    # Split test benign into calibration (threshold) + held-out (evaluation)
    n_calib = len(hps_scores_ben) // 2
    hps_scores_calib = hps_scores_ben[:n_calib]
    hps_scores_ben = hps_scores_ben[n_calib:]  # held-out only
    hps_thr = float(np.quantile(hps_scores_calib, 1.0 - FPR_TARGET))

    # ── RTV on same test set ──
    print(f"  Computing RTV scores on test set...")
    rtv_scores_ben = np.array([rtv_score(compute_fingerprint(hs, refusal_dirs, RTV_LAYERS, TOKEN_POSITIONS))
                               for hs in hs_test_ben])
    rtv_scores_atk = np.array([rtv_score(compute_fingerprint(hs, refusal_dirs, RTV_LAYERS, TOKEN_POSITIONS))
                               for hs in hs_test_atk])
    rtv_scores_calib = rtv_scores_ben[:n_calib]
    rtv_scores_ben = rtv_scores_ben[n_calib:]
    rtv_thr = float(np.quantile(rtv_scores_calib, 1.0 - FPR_TARGET))

    # ── Ensemble (HPS features + RTV fingerprints) ──
    print(f"  Computing Ensemble (12 HPS + 15 RTV = 27 features)...")
    rtv_feats_train = np.array([compute_fingerprint(hs, refusal_dirs, RTV_LAYERS, TOKEN_POSITIONS)
                                for hs in hs_train_ben + hs_train_atk])
    rtv_feats_te_ben = np.array([compute_fingerprint(hs, refusal_dirs, RTV_LAYERS, TOKEN_POSITIONS)
                                 for hs in hs_test_ben])
    rtv_feats_te_atk = np.array([compute_fingerprint(hs, refusal_dirs, RTV_LAYERS, TOKEN_POSITIONS)
                                 for hs in hs_test_atk])

    ens_train = np.concatenate([feats_train, rtv_feats_train], axis=1)
    ens_te_ben = np.concatenate([feats_te_ben, rtv_feats_te_ben], axis=1)
    ens_te_atk = np.concatenate([feats_te_atk, rtv_feats_te_atk], axis=1)

    sc_ens = StandardScaler()
    clf_ens = LogisticRegression(max_iter=2000, random_state=42)
    clf_ens.fit(sc_ens.fit_transform(ens_train), y_train)
    ens_scores_ben = clf_ens.predict_proba(sc_ens.transform(ens_te_ben))[:, 1]
    ens_scores_atk = clf_ens.predict_proba(sc_ens.transform(ens_te_atk))[:, 1]
    ens_scores_calib = ens_scores_ben[:n_calib]
    ens_scores_ben = ens_scores_ben[n_calib:]
    ens_thr = float(np.quantile(ens_scores_calib, 1.0 - FPR_TARGET))
    ens_tpr = float((ens_scores_atk > ens_thr).mean())
    ens_auroc = roc_auc_score(np.array([0]*len(ens_scores_ben)+[1]*len(ens_scores_atk)),
                              np.concatenate([ens_scores_ben, ens_scores_atk]))

    # ── Results ──
    print(f"\n{'═'*60}")
    print(f"  RESULTS — Llama-3-8B (same test set)")
    print(f"{'═'*60}")
    hps_fpr = float((hps_scores_ben > hps_thr).mean())
    rtv_fpr = float((rtv_scores_ben > rtv_thr).mean())
    ens_fpr = float((ens_scores_ben > ens_thr).mean())

    # Full metrics
    def full_metrics(scores_ben, scores_atk, thr, name):
        tp = int((scores_atk > thr).sum())
        fn = int((scores_atk <= thr).sum())
        fp = int((scores_ben > thr).sum())
        tn = int((scores_ben <= thr).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        print(f"\n  {name}:")
        print(f"    TP={tp}  FN={fn}  FP={fp}  TN={tn}")
        print(f"    Precision={precision:.3f}  Recall(TPR)={recall:.3f}  F1={f1:.3f}")
        print(f"    FPR={fpr:.3f}  Accuracy={accuracy:.3f}")
        return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
                "precision": precision, "recall": recall, "f1": f1,
                "fpr": fpr, "accuracy": accuracy}

    print(f"\n  Test set: {len(hps_scores_ben)} benign + {len(hps_scores_atk)} attacks")
    hps_metrics = full_metrics(hps_scores_ben, hps_scores_atk, hps_thr, "HPS")
    rtv_metrics = full_metrics(rtv_scores_ben, rtv_scores_atk, rtv_thr, "RTV")
    ens_metrics = full_metrics(ens_scores_ben, ens_scores_atk, ens_thr, "Ensemble")

    print(f"\n  {'Method':<20} | {'AUROC':>6} | {'TPR':>6} | {'FPR':>6} | {'F1':>6} | {'Acc':>6}")
    print(f"  {'─'*20}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*6}")

    hps_auroc = roc_auc_score(np.array([0]*len(hps_scores_ben)+[1]*len(hps_scores_atk)),
                              np.concatenate([hps_scores_ben, hps_scores_atk]))
    rtv_auroc = roc_auc_score(np.array([0]*len(rtv_scores_ben)+[1]*len(rtv_scores_atk)),
                              np.concatenate([rtv_scores_ben, rtv_scores_atk]))
    ens_auroc = roc_auc_score(np.array([0]*len(ens_scores_ben)+[1]*len(ens_scores_atk)),
                              np.concatenate([ens_scores_ben, ens_scores_atk]))
    hps_tpr = hps_metrics['recall']
    rtv_tpr = rtv_metrics['recall']
    ens_tpr = ens_metrics['recall']
    hps_fpr = hps_metrics['fpr']
    rtv_fpr = rtv_metrics['fpr']
    ens_fpr = ens_metrics['fpr']

    print(f"  {'RTV':<20} | {rtv_auroc:>6.3f} | {rtv_tpr:>6.3f} | {rtv_fpr:>6.3f} | {rtv_metrics['f1']:>6.3f} | {rtv_metrics['accuracy']:>6.3f}")
    print(f"  {'HPS':<20} | {hps_auroc:>6.3f} | {hps_tpr:>6.3f} | {hps_fpr:>6.3f} | {hps_metrics['f1']:>6.3f} | {hps_metrics['accuracy']:>6.3f}")
    print(f"  {'Ensemble':<20} | {ens_auroc:>6.3f} | {ens_tpr:>6.3f} | {ens_fpr:>6.3f} | {ens_metrics['f1']:>6.3f} | {ens_metrics['accuracy']:>6.3f}")

    # Per-method
    print(f"\n  Per-attack-type:")
    print(f"  {'Method':<15} | {'N':>5} | {'HPS':>6} | {'RTV':>6} | {'Ens':>6}")
    print(f"  {'─'*15}─┼─{'─'*5}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*6}")
    for m in sorted(set(test_methods)):
        idx = [i for i, x in enumerate(test_methods) if x == m]
        ht = float((hps_scores_atk[idx] > hps_thr).mean())
        rt = float((rtv_scores_atk[idx] > rtv_thr).mean())
        et = float((ens_scores_atk[idx] > ens_thr).mean())
        print(f"  {m:<15} | {len(idx):>5} | {ht:>6.3f} | {rt:>6.3f} | {et:>6.3f}")

    # Save
    results = {
        "model": args.model,
        "same_dist": {
            "hps": {"auroc": float(hps_auroc), "tpr": float(hps_tpr), "layers": HPS_LAYERS},
            "rtv": {"auroc": float(rtv_auroc), "tpr": float(rtv_tpr), "layers": RTV_LAYERS},
            "ensemble": {"auroc": float(ens_auroc), "tpr": float(ens_tpr)},
        },
        "n_test_ben": len(test_ben), "n_test_atk": len(test_atk),
    }

    # ── Plot: 3-panel comparison (HPS / RTV / Ensemble) ──
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    n_b = len(feats_te_ben)
    methods_sorted = sorted(set(test_methods))
    colors = plt.cm.Set1(np.linspace(0, 1, len(methods_sorted)))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    panels = [
        ("HPS Features (12D)", np.vstack([feats_te_ben, feats_te_atk]), hps_auroc),
        ("RTV Features (15D)", np.vstack([rtv_feats_te_ben, rtv_feats_te_atk]), rtv_auroc),
        ("Ensemble (27D)", np.vstack([ens_te_ben, ens_te_atk]), ens_auroc),
    ]

    for ax, (title, X_feat, auroc) in zip(axes, panels):
        pca_p = PCA(n_components=2, random_state=42)
        X_2d = pca_p.fit_transform(X_feat)
        ax.scatter(X_2d[:n_b, 0], X_2d[:n_b, 1], c='#2ecc71', label='Benign', alpha=0.7, s=40)
        for i, m in enumerate(methods_sorted):
            idx = [j for j, x in enumerate(test_methods) if x == m]
            ax.scatter(X_2d[n_b + np.array(idx), 0], X_2d[n_b + np.array(idx), 1],
                       c=[colors[i]], label=m, alpha=0.4, s=15)
        ax.set_xlabel(f"PC1 ({pca_p.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca_p.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title(f"{title}\nAUROC={auroc:.3f}", fontsize=11)
        ax.legend(fontsize=6, loc='upper right')

    plt.suptitle("Feature Space Comparison — Llama-3-8B", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/hps_llama3_clusters.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved → results/hps_llama3_clusters.png")

    # ══════════════════════════════════════════════════════════════════════
    #  CROSS-ATTACK: Train HPS on 8 methods, test on held-out 9th
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  CROSS-ATTACK (train on 8, test on held-out 1)")
    print(f"{'═'*60}")

    methods_unique = sorted(set(attack_methods))
    # Group hidden states by method
    hs_by_method = {m: [] for m in methods_unique}
    # We need to re-extract per method from the full attack list
    # Use the already-extracted train+test hidden states mapped back
    all_atk_hs = hs_train_atk + hs_test_atk
    all_atk_methods = [attack_methods[i] for i in atk_idx]
    for hs, m in zip(all_atk_hs, all_atk_methods):
        hs_by_method[m].append(hs)

    # All benign (train+test combined for cross-attack, then split)
    all_ben_hs = hs_train_ben + hs_test_ben
    ben_split = int(0.8 * len(all_ben_hs))
    cv_ben_train_hs = all_ben_hs[:ben_split]
    cv_ben_test_hs = all_ben_hs[ben_split:]

    print(f"\n  {'Held-out':<15} | {'N':>5} | {'HPS':>6} | {'RTV':>6} | {'Ens':>6}")
    print(f"  {'─'*15}─┼─{'─'*5}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*6}")

    cross_results = {}
    for held_out in methods_unique:
        # Train on all methods except held_out
        train_hs = []
        for m in methods_unique:
            if m != held_out:
                train_hs.extend(hs_by_method[m])
        test_hs = hs_by_method[held_out]

        if len(test_hs) < 5:
            continue

        # HPS: train projection
        X_cv_ben = to_hps_array(cv_ben_train_hs)
        X_cv_atk = to_hps_array(train_hs)
        X_cv_train = np.concatenate([X_cv_ben, X_cv_atk])
        y_cv_train = np.array([0]*len(X_cv_ben) + [1]*len(X_cv_atk))

        X_cv_te_ben = to_hps_array(cv_ben_test_hs)
        X_cv_te_atk = to_hps_array(test_hs)

        # Multi-seed HPS cross-attack (3 seeds)
        X_cv_t = torch.tensor(X_cv_train, dtype=torch.float32, device=device)
        y_cv_t = torch.tensor(y_cv_train, dtype=torch.long, device=device)
        hps_seed_tprs = []
        for _seed in range(3):
            torch.manual_seed(_seed)
            proj_cv = LorentzProjection(d_hidden, 64, 1.0, n_layers=n_layers).to(device)
            opt_cv = optim.Adam(proj_cv.parameters(), lr=1e-3, weight_decay=1e-5)
            _bl = float('inf'); _pc = 0
            for _ep in range(200):
                l = torch.tensor(0.0, device=device)
                for li in range(n_layers):
                    h = proj_cv(X_cv_t[:, li, :])
                    l = l + contrastive_loss(h, y_cv_t, k=proj_cv.k, tau=proj_cv.tau(li))
                l = l / n_layers
                opt_cv.zero_grad(); l.backward(); opt_cv.step()
                if l.item() < _bl - 1e-4: _bl = l.item(); _pc = 0
                else: _pc += 1
                if _pc >= 20: break
            proj_cv.eval()
            f_tr = extract_trajectory_features(proj_cv, X_cv_train)
            f_te_ben = extract_trajectory_features(proj_cv, X_cv_te_ben)
            f_te_atk = extract_trajectory_features(proj_cv, X_cv_te_atk)
            sc_cv = StandardScaler()
            clf_cv = LogisticRegression(max_iter=2000, random_state=_seed)
            clf_cv.fit(sc_cv.fit_transform(f_tr), y_cv_train)
            s_ben = clf_cv.predict_proba(sc_cv.transform(f_te_ben))[:, 1]
            s_atk = clf_cv.predict_proba(sc_cv.transform(f_te_atk))[:, 1]
            n_cal = len(s_ben) // 2
            thr_cv = float(np.quantile(s_ben[:n_cal], 1.0 - FPR_TARGET))
            hps_seed_tprs.append(float((s_atk > thr_cv).mean()))
        hps_cv_tpr = float(np.mean(hps_seed_tprs))

        # RTV on same held-out test
        rtv_s_ben = np.array([rtv_score(compute_fingerprint(hs, refusal_dirs, RTV_LAYERS, TOKEN_POSITIONS))
                              for hs in cv_ben_test_hs])
        rtv_s_atk = np.array([rtv_score(compute_fingerprint(hs, refusal_dirs, RTV_LAYERS, TOKEN_POSITIONS))
                              for hs in test_hs])
        rtv_thr_cv = float(np.quantile(rtv_s_ben[:n_cal], 1.0 - FPR_TARGET))
        rtv_cv_tpr = float((rtv_s_atk > rtv_thr_cv).mean())

        # Ensemble on same held-out test
        rtv_f_tr = np.array([compute_fingerprint(hs, refusal_dirs, RTV_LAYERS, TOKEN_POSITIONS)
                             for hs in cv_ben_train_hs + train_hs])
        rtv_f_te_ben = np.array([compute_fingerprint(hs, refusal_dirs, RTV_LAYERS, TOKEN_POSITIONS)
                                 for hs in cv_ben_test_hs])
        rtv_f_te_atk = np.array([compute_fingerprint(hs, refusal_dirs, RTV_LAYERS, TOKEN_POSITIONS)
                                 for hs in test_hs])
        ens_f_tr = np.concatenate([f_tr, rtv_f_tr], axis=1)
        ens_f_te_ben = np.concatenate([f_te_ben, rtv_f_te_ben], axis=1)
        ens_f_te_atk = np.concatenate([f_te_atk, rtv_f_te_atk], axis=1)
        sc_ens_cv = StandardScaler()
        clf_ens_cv = LogisticRegression(max_iter=2000, random_state=42)
        clf_ens_cv.fit(sc_ens_cv.fit_transform(ens_f_tr), y_cv_train)
        es_ben = clf_ens_cv.predict_proba(sc_ens_cv.transform(ens_f_te_ben))[:, 1]
        es_atk = clf_ens_cv.predict_proba(sc_ens_cv.transform(ens_f_te_atk))[:, 1]
        ens_thr_cv = float(np.quantile(es_ben[:n_cal], 1.0 - FPR_TARGET))
        ens_cv_tpr = float((es_atk > ens_thr_cv).mean())

        print(f"  {held_out:<15} | {len(test_hs):>5} | {hps_cv_tpr:>6.3f} | {rtv_cv_tpr:>6.3f} | {ens_cv_tpr:>6.3f}")
        cross_results[held_out] = {"hps": hps_cv_tpr, "rtv": rtv_cv_tpr, "ens": ens_cv_tpr, "n": len(test_hs)}

    mean_hps_cv = np.mean([v["hps"] for v in cross_results.values()])
    mean_rtv_cv = np.mean([v["rtv"] for v in cross_results.values()])
    mean_ens_cv = np.mean([v["ens"] for v in cross_results.values()])
    print(f"  {'─'*15}─┼─{'─'*5}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*6}")
    print(f"  {'MEAN':<15} | {'':>5} | {mean_hps_cv:>6.3f} | {mean_rtv_cv:>6.3f} | {mean_ens_cv:>6.3f}")

    results["cross_attack"] = cross_results
    results["cross_attack_mean"] = {"hps": float(mean_hps_cv), "rtv": float(mean_rtv_cv)}

    out = "results/hps_vs_rtv_llama3.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {out}")
    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    main()
