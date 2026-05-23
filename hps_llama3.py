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
    print(f"\n  Extracting activations (all layers: {ALL_LAYERS})...")

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

    # Extract train + test
    print("  Train benign...")
    hs_train_ben = extract_all(train_ben, "train benign")
    print("  Train attacks...")
    hs_train_atk = extract_all(train_atk, "train attacks")
    print("  Test benign...")
    hs_test_ben = extract_all(test_ben, "test benign")
    print("  Test attacks...")
    hs_test_atk = extract_all(test_atk, "test attacks")

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
    proj = LorentzProjection(d_hidden, 64, 1.0, n_layers=n_layers).to(device)
    opt = optim.Adam(proj.parameters(), lr=1e-3, weight_decay=1e-5)
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)

    for epoch in range(120):
        loss = torch.tensor(0.0, device=device)
        for l in range(n_layers):
            h = proj(X_t[:, l, :])
            loss = loss + contrastive_loss(h, y_t, k=proj.k, tau=proj.tau(l))
        loss = loss / n_layers
        opt.zero_grad(); loss.backward(); opt.step()
        if (epoch+1) % 40 == 0:
            print(f"    Epoch {epoch+1}/120 loss={loss.item():.4f}")
    proj.eval()

    feats_train = extract_trajectory_features(proj, X_train)
    feats_te_ben = extract_trajectory_features(proj, X_te_ben)
    feats_te_atk = extract_trajectory_features(proj, X_te_atk)

    sc = StandardScaler()
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(sc.fit_transform(feats_train), y_train)
    hps_scores_ben = clf.predict_proba(sc.transform(feats_te_ben))[:, 1]
    hps_scores_atk = clf.predict_proba(sc.transform(feats_te_atk))[:, 1]
    hps_thr = float(np.quantile(hps_scores_ben, 1.0 - FPR_TARGET))

    # ── RTV on same test set ──
    print(f"  Computing RTV scores on test set...")
    rtv_scores_ben = np.array([rtv_score(compute_fingerprint(hs, refusal_dirs, RTV_LAYERS, TOKEN_POSITIONS))
                               for hs in hs_test_ben])
    rtv_scores_atk = np.array([rtv_score(compute_fingerprint(hs, refusal_dirs, RTV_LAYERS, TOKEN_POSITIONS))
                               for hs in hs_test_atk])
    rtv_thr = float(np.quantile(rtv_scores_ben, 1.0 - FPR_TARGET))

    # ── Results ──
    hps_tpr = float((hps_scores_atk > hps_thr).mean())
    rtv_tpr = float((rtv_scores_atk > rtv_thr).mean())
    hps_auroc = roc_auc_score(np.array([0]*len(hps_scores_ben)+[1]*len(hps_scores_atk)),
                              np.concatenate([hps_scores_ben, hps_scores_atk]))
    rtv_auroc = roc_auc_score(np.array([0]*len(rtv_scores_ben)+[1]*len(rtv_scores_atk)),
                              np.concatenate([rtv_scores_ben, rtv_scores_atk]))

    print(f"\n{'═'*60}")
    print(f"  RESULTS — Llama-3-8B (same test set)")
    print(f"{'═'*60}")
    print(f"  {'Method':<20} | {'AUROC':>6} | {'TPR@5%':>7}")
    print(f"  {'─'*20}─┼─{'─'*6}─┼─{'─'*7}")
    print(f"  {'RTV':<20} | {rtv_auroc:>6.3f} | {rtv_tpr:>7.3f}")
    print(f"  {'HPS':<20} | {hps_auroc:>6.3f} | {hps_tpr:>7.3f}")

    # Per-method
    print(f"\n  Per-attack-type:")
    print(f"  {'Method':<15} | {'N':>5} | {'HPS':>6} | {'RTV':>6}")
    print(f"  {'─'*15}─┼─{'─'*5}─┼─{'─'*6}─┼─{'─'*6}")
    for m in sorted(set(test_methods)):
        idx = [i for i, x in enumerate(test_methods) if x == m]
        ht = float((hps_scores_atk[idx] > hps_thr).mean())
        rt = float((rtv_scores_atk[idx] > rtv_thr).mean())
        print(f"  {m:<15} | {len(idx):>5} | {ht:>6.3f} | {rt:>6.3f}")

    # Save
    results = {
        "model": args.model,
        "hps": {"auroc": float(hps_auroc), "tpr": float(hps_tpr), "layers": HPS_LAYERS},
        "rtv": {"auroc": float(rtv_auroc), "tpr": float(rtv_tpr), "layers": RTV_LAYERS},
        "n_test_ben": len(test_ben), "n_test_atk": len(test_atk),
    }
    out = "results/hps_vs_rtv_llama3.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {out}")
    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    main()
