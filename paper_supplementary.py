"""
Supplementary experiments for paper revision:
  1. Euclidean cross-attack at 5% FPR (addresses reviewer concern about 0% TPR)
  2. Multi-seed stability (5 seeds, reports mean±std)
  3. Learning curve (AUROC vs number of training attacks)
  4. Computational cost (ms per prompt)

Uses cached Llama-3 activations. Fast — no model loading needed for 2/3/4.

Usage:
  python paper_supplementary.py --test-attacks llama3_attacks.json --harmless data_harmless_6500.csv
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from experiment7 import LorentzProjection, contrastive_loss, extract_trajectory_features
from rtv_standalone import FPR_TARGET

HPS_LAYERS = [0, 1, 2, 28, 29, 30, 31]


def train_and_eval(X_train, y_train, X_te_ben, X_te_atk, seed=42, euclidean=False):
    """Train projection + classifier, return AUROC and TPR@5%."""
    n_layers = X_train.shape[1]
    d_hidden = X_train.shape[2]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    proj = LorentzProjection(d_hidden, 64, 1.0, n_layers=n_layers).to(device)
    opt = optim.Adam(proj.parameters(), lr=1e-3, weight_decay=1e-5)
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)

    if euclidean:
        # Euclidean: L2 contrastive with per-layer scale (matches HPS parameter count)
        proj_e = nn.Linear(d_hidden, 64, bias=False).to(device)
        nn.init.xavier_uniform_(proj_e.weight)
        scale_per_layer = nn.Parameter(torch.ones(n_layers, device=device) / 8.0)
        log_margin = nn.Parameter(torch.tensor(np.log(2.0), device=device))  # learnable margin (analog to κ)
        opt = optim.Adam(list(proj_e.parameters()) + [scale_per_layer, log_margin], lr=1e-3, weight_decay=1e-5)
        best_loss = float('inf'); patience = 0
        for _ in range(200):
            loss = torch.tensor(0.0, device=device)
            for l in range(n_layers):
                h = proj_e(X_t[:, l, :]) * scale_per_layer[l]
                dists = torch.cdist(h, h)
                margin = torch.exp(log_margin).clamp(0.5, 5.0)
                sm = (y_t.unsqueeze(0) == y_t.unsqueeze(1)).float()
                dm = 1.0 - sm
                tr = torch.triu(torch.ones(h.shape[0], h.shape[0], device=device), diagonal=1)
                ns = (sm * tr).sum().clamp(min=1)
                nd = (dm * tr).sum().clamp(min=1)
                loss = loss + ((dists**2 * sm * tr).sum()/ns + (torch.clamp(margin-dists,min=0)**2 * dm * tr).sum()/nd) / 2
            loss = loss / n_layers
            opt.zero_grad(); loss.backward(); opt.step()
            if loss.item() < best_loss - 1e-4:
                best_loss = loss.item(); patience = 0
            else:
                patience += 1
            if patience >= 20:
                break
        # Extract Euclidean features (norms + distances, analogous to HPS trajectory)
        proj_e.eval()
        def euc_feats(X):
            feats = []
            with torch.no_grad():
                for i in range(len(X)):
                    x = torch.tensor(X[i], dtype=torch.float32, device=device)
                    layer_pts = []
                    for l in range(n_layers):
                        layer_pts.append((proj_e(x[l:l+1]) * scale_per_layer[l]).squeeze(0).cpu().numpy())
                    h_np = np.array(layer_pts)
                    norms = np.linalg.norm(h_np, axis=1)
                    d_total = np.linalg.norm(h_np[-1] - h_np[0])
                    path_len = sum(np.linalg.norm(h_np[j+1]-h_np[j]) for j in range(n_layers-1))
                    curvs = []
                    for j in range(1, n_layers-1):
                        dp = np.linalg.norm(h_np[j]-h_np[j-1])
                        dn = np.linalg.norm(h_np[j+1]-h_np[j])
                        ds = np.linalg.norm(h_np[j+1]-h_np[j-1])
                        curvs.append(abs(dp+dn-ds)/(dp+dn+1e-8))
                    curvs = np.array(curvs) if curvs else np.array([0.0])
                    feats.append([norms.mean(), norms.max(), norms.min(), norms.std(), norms.max()-norms.min(),
                                  curvs.max(), curvs.mean(), curvs.std() if len(curvs)>1 else 0,
                                  np.argmax(curvs)/max(len(curvs),1), d_total, path_len, d_total/(path_len+1e-8)])
            return np.array(feats)
        feats_train = euc_feats(X_train)
        feats_te_ben = euc_feats(X_te_ben)
        feats_te_atk = euc_feats(X_te_atk)
    else:
        best_loss = float('inf'); patience = 0
        for _ in range(200):
            loss = torch.tensor(0.0, device=device)
            for l in range(n_layers):
                h = proj(X_t[:, l, :])
                loss = loss + contrastive_loss(h, y_t, k=proj.k, tau=proj.tau(l))
            loss = loss / n_layers
            opt.zero_grad(); loss.backward(); opt.step()
            if loss.item() < best_loss - 1e-4:
                best_loss = loss.item(); patience = 0
            else:
                patience += 1
            if patience >= 20:
                break
        proj.eval()
        feats_train = extract_trajectory_features(proj, X_train)
        feats_te_ben = extract_trajectory_features(proj, X_te_ben)
        feats_te_atk = extract_trajectory_features(proj, X_te_atk)

    sc = StandardScaler()
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(sc.fit_transform(feats_train), y_train)

    # Split test benign into calibration (threshold) + held-out (FPR verification)
    n_calib = len(feats_te_ben) // 2
    feats_calib = feats_te_ben[:n_calib]
    feats_test_ben = feats_te_ben[n_calib:]

    scores_calib = clf.predict_proba(sc.transform(feats_calib))[:, 1]
    scores_test_ben = clf.predict_proba(sc.transform(feats_test_ben))[:, 1]
    scores_atk = clf.predict_proba(sc.transform(feats_te_atk))[:, 1]

    thr = float(np.quantile(scores_calib, 1.0 - FPR_TARGET))
    tpr = float((scores_atk > thr).mean())
    fpr_actual = float((scores_test_ben > thr).mean())
    auroc = roc_auc_score(
        np.array([0]*len(scores_test_ben) + [1]*len(scores_atk)),
        np.concatenate([scores_test_ben, scores_atk])
    )
    return auroc, tpr, fpr_actual


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-attacks", required=True)
    parser.add_argument("--harmless", required=True)
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"  Paper Supplementary Experiments")
    print(f"{'═'*60}\n")

    # Load cached activations
    cache_path = "results/llama3_activations_cache.npz"
    if not os.path.exists(cache_path):
        print("ERROR: Run hps_llama3.py first to generate cache.")
        return

    print("  Loading cached activations...")
    cache = np.load(cache_path, allow_pickle=True)
    hs_train_ben = cache["hs_train_ben"].tolist()
    hs_train_atk = cache["hs_train_atk"].tolist()
    hs_test_ben = cache["hs_test_ben"].tolist()
    hs_test_atk = cache["hs_test_atk"].tolist()

    def to_hps(hs_list):
        return np.array([[hs[l][-1] for l in HPS_LAYERS] for hs in hs_list])

    X_tr_ben = to_hps(hs_train_ben)
    X_tr_atk = to_hps(hs_train_atk)
    X_te_ben = to_hps(hs_test_ben)
    X_te_atk = to_hps(hs_test_atk)

    # Load attack methods for cross-attack
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

    print(f"  Train: {len(X_tr_ben)} benign + {len(X_tr_atk)} attacks")
    print(f"  Test: {len(X_te_ben)} benign + {len(X_te_atk)} attacks")

    # ══════════════════════════════════════════════════════════════════════
    #  EXPERIMENT 1: Multi-seed stability (5 seeds)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  EXP 1: Multi-seed stability (5 seeds)")
    print(f"{'─'*60}")

    X_train = np.concatenate([X_tr_ben, X_tr_atk])
    y_train = np.array([0]*len(X_tr_ben) + [1]*len(X_tr_atk))

    aurocs, tprs = [], []
    euc_aurocs, euc_tprs = [], []
    for seed in range(5):
        a, t, fpr = train_and_eval(X_train, y_train, X_te_ben, X_te_atk, seed=seed)
        aurocs.append(a); tprs.append(t)
        print(f"    Seed {seed} (HPS): AUROC={a:.3f}  TPR@5%={t:.3f}  (FPR={fpr:.3f})")

    print(f"\n  HPS Mean AUROC: {np.mean(aurocs):.3f} ± {np.std(aurocs):.3f}")
    print(f"  HPS Mean TPR:   {np.mean(tprs):.3f} ± {np.std(tprs):.3f}")

    print(f"\n  Euclidean baseline (5 seeds):")
    for seed in range(5):
        a, t, fpr = train_and_eval(X_train, y_train, X_te_ben, X_te_atk, seed=seed, euclidean=True)
        euc_aurocs.append(a); euc_tprs.append(t)
        print(f"    Seed {seed} (Euc): AUROC={a:.3f}  TPR@5%={t:.3f}  (FPR={fpr:.3f})")

    print(f"\n  Euc Mean AUROC: {np.mean(euc_aurocs):.3f} ± {np.std(euc_aurocs):.3f}")
    print(f"  Euc Mean TPR:   {np.mean(euc_tprs):.3f} ± {np.std(euc_tprs):.3f}")
    print(f"\n  Δ AUROC (HPS - Euc): {np.mean(aurocs) - np.mean(euc_aurocs):.3f}")

    # ══════════════════════════════════════════════════════════════════════
    #  EXPERIMENT 2: Learning curve (AUROC vs N training attacks)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  EXP 2: Learning curve (vary training attacks)")
    print(f"{'─'*60}")

    n_values = [50, 100, 200, 500, 1000, 2000, len(X_tr_atk)]
    print(f"\n  {'N_attacks':<10} | {'AUROC':>6} | {'TPR@5%':>7}")
    print(f"  {'─'*10}─┼─{'─'*6}─┼─{'─'*7}")

    for n in n_values:
        if n > len(X_tr_atk):
            continue
        X_sub = np.concatenate([X_tr_ben[:n], X_tr_atk[:n]])
        y_sub = np.array([0]*min(n, len(X_tr_ben)) + [1]*n)
        # Balance benign to match
        n_ben = min(n, len(X_tr_ben))
        X_sub = np.concatenate([X_tr_ben[:n_ben], X_tr_atk[:n]])
        y_sub = np.array([0]*n_ben + [1]*n)
        a, t, _ = train_and_eval(X_sub, y_sub, X_te_ben, X_te_atk, seed=42)
        print(f"  {n:<10} | {a:>6.3f} | {t:>7.3f}")

    # ══════════════════════════════════════════════════════════════════════
    #  EXPERIMENT 3: Cross-attack at FPR=5% (Euclidean check)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  EXP 3: Cross-attack per-method (HPS at FPR=5%)")
    print(f"{'─'*60}")

    # Group by method
    all_atk_hps = np.concatenate([X_tr_atk, X_te_atk])
    all_atk_methods = [attack_methods[i] for i in atk_idx]
    methods_unique = sorted(set(attack_methods))

    hs_by_method = {m: [] for m in methods_unique}
    for act, method in zip(all_atk_hps, all_atk_methods):
        hs_by_method[method].append(act)
    for m in methods_unique:
        hs_by_method[m] = np.array(hs_by_method[m])

    all_ben = np.concatenate([X_tr_ben, X_te_ben])
    ben_split = int(0.8 * len(all_ben))
    cv_ben_tr = all_ben[:ben_split]
    cv_ben_te = all_ben[ben_split:]

    print(f"\n  {'Held-out':<15} | {'TPR@5% (mean±std)':>20}")
    print(f"  {'─'*15}─┼─{'─'*20}")

    cross_tprs = []
    for held_out in methods_unique:
        train_atk = np.concatenate([hs_by_method[m] for m in methods_unique if m != held_out])
        test_atk = hs_by_method[held_out]
        if len(test_atk) < 5:
            continue

        X_cv = np.concatenate([cv_ben_tr, train_atk])
        y_cv = np.array([0]*len(cv_ben_tr) + [1]*len(train_atk))
        seed_tprs = []
        for s in range(3):
            _, t, _ = train_and_eval(X_cv, y_cv, cv_ben_te, test_atk, seed=s)
            seed_tprs.append(t)
        mean_t = np.mean(seed_tprs)
        print(f"  {held_out:<15} | {mean_t:>7.3f} ± {np.std(seed_tprs):.3f}")
        cross_tprs.append(mean_t)

    print(f"  {'─'*15}─┼─{'─'*20}")
    print(f"  {'MEAN':<15} | {np.mean(cross_tprs):>7.3f}")

    # Euclidean cross-attack (3 seeds)
    print(f"\n  Euclidean cross-attack:")
    print(f"  {'Held-out':<15} | {'TPR@5% (mean±std)':>20}")
    print(f"  {'─'*15}─┼─{'─'*20}")
    euc_cross_tprs = []
    for held_out in methods_unique:
        train_atk = np.concatenate([hs_by_method[m] for m in methods_unique if m != held_out])
        test_atk_cv = hs_by_method[held_out]
        if len(test_atk_cv) < 5:
            continue
        X_cv = np.concatenate([cv_ben_tr, train_atk])
        y_cv = np.array([0]*len(cv_ben_tr) + [1]*len(train_atk))
        seed_tprs = []
        for s in range(3):
            _, t, _ = train_and_eval(X_cv, y_cv, cv_ben_te, test_atk_cv, seed=s, euclidean=True)
            seed_tprs.append(t)
        mean_t = np.mean(seed_tprs)
        print(f"  {held_out:<15} | {mean_t:>7.3f} ± {np.std(seed_tprs):.3f}")
        euc_cross_tprs.append(mean_t)
    print(f"  {'─'*15}─┼─{'─'*20}")
    print(f"  {'MEAN':<15} | {np.mean(euc_cross_tprs):>7.3f}")
    print(f"\n  Δ cross-attack mean (HPS - Euc): {np.mean(cross_tprs) - np.mean(euc_cross_tprs):+.3f}")

    # ══════════════════════════════════════════════════════════════════════
    #  EXPERIMENT 4: Computational cost
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  EXP 4: Computational cost (projection + features + classification)")
    print(f"{'─'*60}")

    # Load projection
    proj_path = "results/hps_llama3_projection.pt"
    ckpt = torch.load(proj_path, map_location="cpu", weights_only=False)
    proj = LorentzProjection(ckpt["d_in"], ckpt["d_proj"], 1.0, n_layers=ckpt["n_layers"])
    proj.load_state_dict(ckpt["state_dict"])
    proj.eval()

    # Time 100 samples
    n_time = 100
    X_sample = X_te_atk[:n_time]

    start = time.time()
    for _ in range(3):  # 3 runs for stability
        feats = extract_trajectory_features(proj, X_sample)
        sc = StandardScaler()
        sc.fit(feats)
        sc.transform(feats)
    elapsed = (time.time() - start) / 3

    ms_per_prompt = (elapsed / n_time) * 1000
    print(f"  Projection + features + scaling: {ms_per_prompt:.2f} ms/prompt")
    print(f"  (Excludes LLM forward pass — activation extraction is the bottleneck)")
    print(f"  Total for 100 prompts: {elapsed*1000:.1f} ms")

    # Save all results
    results = {
        "multi_seed": {"aurocs": aurocs, "tprs": tprs,
                       "mean_auroc": float(np.mean(aurocs)), "std_auroc": float(np.std(aurocs)),
                       "mean_tpr": float(np.mean(tprs)), "std_tpr": float(np.std(tprs))},
        "computational_cost_ms_per_prompt": ms_per_prompt,
    }
    with open("results/paper_supplementary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → results/paper_supplementary.json")
    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    main()
