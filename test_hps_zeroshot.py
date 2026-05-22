"""
HPS trained on harmful vs harmless (zero-shot, same data as RTV).
Apples-to-apples comparison: does multi-layer hyperbolic trajectory
beat RTV's fingerprint when both see identical calibration data?

Pipeline:
  1. Extract activations at 8 HPS layers for harmful + harmless calibration
  2. Train Lorentz projection with contrastive loss (harmful=1, harmless=0)
  3. Extract 12 trajectory features
  4. Fit logistic regression
  5. Score attacks as outliers (high attack probability = flagged)
  6. Compare to RTV (0.843 AUROC)

Usage:
  python test_hps_zeroshot.py \
    --harmless JBShield/data/harmless_calibration.csv \
    --harmful JBShield/data/harmful_calibration.csv \
    --test-attacks results/validated_attacks_categorized.json
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
from rtv_standalone import load_model, extract_hidden_states
from experiment7 import LorentzProjection, contrastive_loss, extract_trajectory_features

HPS_LAYERS = [0, 1, 2, 35, 36, 37, 38, 39]
FPR_TARGET = 0.05


def extract_hps_acts(model, tokenizer, prompts, label=""):
    """Extract activations at HPS layers, last token."""
    n = len(prompts)
    d_hidden = None
    acts = []
    for i, p in enumerate(prompts):
        hs = extract_hidden_states(model, tokenizer, p, HPS_LAYERS)
        if d_hidden is None:
            d_hidden = hs[HPS_LAYERS[0]].shape[-1]
        vec = np.array([hs[l][-1] for l in HPS_LAYERS])  # (8, d_hidden)
        acts.append(vec)
        if (i + 1) % 20 == 0:
            print(f"    {label}: {i+1}/{n}")
    return np.array(acts)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lmsys/vicuna-13b-v1.5")
    parser.add_argument("--harmless", required=True)
    parser.add_argument("--harmful", required=True)
    parser.add_argument("--test-attacks", required=True)
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"  HPS Zero-Shot: Train on harmful vs harmless")
    print(f"  (Same calibration data as RTV — no attacks)")
    print(f"{'═'*60}\n")

    # Load data
    df_h = pd.read_csv(args.harmless)
    df_harm = pd.read_csv(args.harmful)
    for col in ["prompt", "goal", "text", "instruction", "query"]:
        if col in df_h.columns: harmless = df_h[col].dropna().tolist(); break
    else: harmless = df_h.iloc[:, 0].dropna().tolist()
    for col in ["prompt", "goal", "text", "instruction", "query"]:
        if col in df_harm.columns: harmful = df_harm[col].dropna().tolist(); break
    else: harmful = df_harm.iloc[:, 0].dropna().tolist()

    with open(args.test_attacks) as f:
        data = json.load(f)
    attacks, methods = [], []
    for m, ps in data.items():
        for p in ps:
            attacks.append(p); methods.append(m)

    n_cal = min(len(harmless), len(harmful))
    print(f"  Calibration: {n_cal} harmless + {n_cal} harmful")
    print(f"  Test attacks: {len(attacks)}")

    # Load model
    model, tokenizer = load_model(args.model)

    # Extract activations
    print(f"\n  Extracting HPS-layer activations...")
    X_harmless = extract_hps_acts(model, tokenizer, harmless[:n_cal], "harmless")
    X_harmful = extract_hps_acts(model, tokenizer, harmful[:n_cal], "harmful")
    X_attacks = extract_hps_acts(model, tokenizer, attacks, "attacks")

    n_layers = len(HPS_LAYERS)
    d_hidden = X_harmless.shape[2]

    # ── Full evaluation: train on all, test on attacks ──
    print(f"\n  Training HPS (Lorentz) on harmful vs harmless...")
    X_train = np.concatenate([X_harmless, X_harmful])
    y_train = np.array([0] * len(X_harmless) + [1] * len(X_harmful))

    device = "cpu"
    torch.manual_seed(42)
    proj = LorentzProjection(d_hidden, config.PROJECTION_DIM, 1.0, n_layers=n_layers).to(device)
    opt = optim.Adam(proj.parameters(), lr=1e-3, weight_decay=1e-5)
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)

    for epoch in range(150):
        total_loss = torch.tensor(0.0, device=device)
        for l in range(n_layers):
            h = proj(X_t[:, l, :])
            tau_l = proj.tau(l)
            total_loss = total_loss + contrastive_loss(h, y_t, k=proj.k, tau=tau_l)
        total_loss = total_loss / n_layers
        opt.zero_grad(); total_loss.backward(); opt.step()
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/150 loss={total_loss.item():.4f} k={proj.k.item():.3f}")

    proj.eval()

    # Extract trajectory features
    feats_train = extract_trajectory_features(proj, X_train)
    feats_attacks = extract_trajectory_features(proj, X_attacks)

    # Fit classifier
    sc = StandardScaler()
    feats_train_s = sc.fit_transform(feats_train)
    feats_attacks_s = sc.transform(feats_attacks)
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(feats_train_s, y_train)

    # Score: probability of being "harmful" class
    train_scores = clf.predict_proba(feats_train_s)[:, 1]
    attack_scores = clf.predict_proba(feats_attacks_s)[:, 1]

    # Threshold at 95th percentile of calibration scores
    threshold = float(np.quantile(train_scores, 1.0 - FPR_TARGET))

    # Metrics
    harmless_scores = attack_scores  # we test on attacks
    # For AUROC: harmless calibration vs attacks
    cal_harmless_scores = train_scores[y_train == 0]
    y_true = np.array([0] * len(cal_harmless_scores) + [1] * len(attack_scores))
    all_scores = np.concatenate([cal_harmless_scores, attack_scores])
    auroc = roc_auc_score(y_true, all_scores)
    tpr = float((attack_scores > threshold).mean())

    print(f"\n{'─'*60}")
    print(f"  RESULTS (full calibration)")
    print(f"{'─'*60}")
    print(f"  AUROC: {auroc:.3f}")
    print(f"  TPR@5%FPR: {tpr:.3f}")
    print(f"  Threshold: {threshold:.4f}")

    # Per-method
    print(f"\n  Per-attack-type:")
    print(f"  {'Method':<30} | {'N':>4} | {'TPR':>6}")
    print(f"  {'─'*30}─┼─{'─'*4}─┼─{'─'*6}")
    for m in sorted(set(methods)):
        idx = [i for i, x in enumerate(methods) if x == m]
        mt = float((attack_scores[idx] > threshold).mean())
        print(f"  {m:<30} | {len(idx):>4} | {mt:>6.3f}")

    # ── Cross-validation (train 20+20, threshold on 10+10) ──
    print(f"\n{'─'*60}")
    print(f"  CROSS-VALIDATION (5 folds, train 20+20, threshold 10+10)")
    print(f"{'─'*60}")

    cv_aurocs, cv_tprs = [], []
    for fold in range(5):
        rng = np.random.RandomState(fold)
        h_idx = rng.permutation(n_cal)
        harm_idx = rng.permutation(n_cal)
        tr_h, te_h = h_idx[:20], h_idx[20:]
        tr_harm, te_harm = harm_idx[:20], harm_idx[20:]

        X_tr = np.concatenate([X_harmless[tr_h], X_harmful[tr_harm]])
        y_tr = np.array([0]*len(tr_h) + [1]*len(tr_harm))

        torch.manual_seed(fold)
        proj_cv = LorentzProjection(d_hidden, config.PROJECTION_DIM, 1.0, n_layers=n_layers).to(device)
        opt_cv = optim.Adam(proj_cv.parameters(), lr=1e-3, weight_decay=1e-5)
        X_t_cv = torch.tensor(X_tr, dtype=torch.float32, device=device)
        y_t_cv = torch.tensor(y_tr, dtype=torch.long, device=device)

        for epoch in range(150):
            total_loss = torch.tensor(0.0, device=device)
            for l in range(n_layers):
                h = proj_cv(X_t_cv[:, l, :])
                tau_l = proj_cv.tau(l)
                total_loss = total_loss + contrastive_loss(h, y_t_cv, k=proj_cv.k, tau=tau_l)
            total_loss = total_loss / n_layers
            opt_cv.zero_grad(); total_loss.backward(); opt_cv.step()
        proj_cv.eval()

        # Features
        X_te = np.concatenate([X_harmless[te_h], X_harmful[te_harm]])
        feats_tr = extract_trajectory_features(proj_cv, X_tr)
        feats_te = extract_trajectory_features(proj_cv, X_te)
        feats_atk = extract_trajectory_features(proj_cv, X_attacks)

        sc_cv = StandardScaler()
        feats_tr_s = sc_cv.fit_transform(feats_tr)
        feats_te_s = sc_cv.transform(feats_te)
        feats_atk_s = sc_cv.transform(feats_atk)

        clf_cv = LogisticRegression(max_iter=2000, random_state=42)
        clf_cv.fit(feats_tr_s, y_tr)

        # Threshold from held-out calibration
        te_scores = clf_cv.predict_proba(feats_te_s)[:, 1]
        thr_cv = float(np.quantile(te_scores, 1.0 - FPR_TARGET))

        atk_scores_cv = clf_cv.predict_proba(feats_atk_s)[:, 1]
        tpr_cv = float((atk_scores_cv > thr_cv).mean())

        # AUROC: held-out harmless vs attacks
        te_harmless_scores = te_scores[:len(te_h)]
        y_cv = np.array([0]*len(te_harmless_scores) + [1]*len(atk_scores_cv))
        all_cv = np.concatenate([te_harmless_scores, atk_scores_cv])
        auroc_cv = roc_auc_score(y_cv, all_cv)

        cv_aurocs.append(auroc_cv)
        cv_tprs.append(tpr_cv)
        print(f"  Fold {fold+1}: AUROC={auroc_cv:.3f}  TPR@5%={tpr_cv:.3f}")

    print(f"\n  CV Mean AUROC: {np.mean(cv_aurocs):.3f} ± {np.std(cv_aurocs):.3f}")
    print(f"  CV Mean TPR:   {np.mean(cv_tprs):.3f} ± {np.std(cv_tprs):.3f}")

    print(f"\n{'─'*60}")
    print(f"  COMPARISON")
    print(f"{'─'*60}")
    print(f"  {'Method':<35} | {'AUROC':>6} | {'TPR@5%':>7}")
    print(f"  {'─'*35}─┼─{'─'*6}─┼─{'─'*7}")
    print(f"  {'RTV (fingerprint+Mahalanobis)':<35} | {'0.843':>6} | {'0.566':>7}")
    print(f"  {'HPS zero-shot (full cal)':<35} | {auroc:>6.3f} | {tpr:>7.3f}")
    print(f"  {'HPS zero-shot (CV mean)':<35} | {np.mean(cv_aurocs):>6.3f} | {np.mean(cv_tprs):>7.3f}")
    print(f"  {'Lorentz single-layer PCA (CV)':<35} | {'0.845':>6} | {'0.480':>7}")

    # ── Plot ──
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    feats_harmless = feats_train[y_train == 0]
    feats_harmful_cal = feats_train[y_train == 1]

    X_plot = np.vstack([feats_harmless, feats_harmful_cal, feats_attacks])
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_plot)

    n_h = len(feats_harmless)
    n_harm = len(feats_harmful_cal)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(X_2d[:n_h, 0], X_2d[:n_h, 1], c='#2ecc71', label='Harmless', alpha=0.7, s=50)
    ax.scatter(X_2d[n_h:n_h+n_harm, 0], X_2d[n_h:n_h+n_harm, 1], c='#e74c3c', label='Harmful', alpha=0.7, s=50)
    ax.scatter(X_2d[n_h+n_harm:, 0], X_2d[n_h+n_harm:, 1], c='#9b59b6', label='Attacks', alpha=0.4, s=20)
    ax.set_title(f"HPS Zero-Shot: Trajectory Features (PCA)\n"
                 f"AUROC={auroc:.3f} | CV AUROC={np.mean(cv_aurocs):.3f}±{np.std(cv_aurocs):.3f}",
                 fontsize=12)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plot_path = os.path.join(config.RESULTS_DIR, "hps_zeroshot_clusters.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n  Plot saved → {plot_path}")

    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    main()
