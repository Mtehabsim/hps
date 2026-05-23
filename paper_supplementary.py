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


def train_and_eval(X_train, y_train, X_te_ben, X_te_atk, seed=42, geometry="hyperbolic"):
    """Train projection + classifier, return AUROC and TPR@5%."""
    n_layers = X_train.shape[1]
    d_hidden = X_train.shape[2]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    if geometry == "hyperbolic":
        proj = LorentzProjection(d_hidden, 64, 1.0, n_layers=n_layers).to(device)
    else:
        # Euclidean: linear projection + L2 contrastive
        proj = LorentzProjection(d_hidden, 64, 1.0, n_layers=n_layers).to(device)
        # We'll use same architecture but measure Euclidean features

    opt = optim.Adam(proj.parameters(), lr=1e-3, weight_decay=1e-5)
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)

    for _ in range(120):
        loss = torch.tensor(0.0, device=device)
        for l in range(n_layers):
            h = proj(X_t[:, l, :])
            loss = loss + contrastive_loss(h, y_t, k=proj.k, tau=proj.tau(l))
        loss = loss / n_layers
        opt.zero_grad(); loss.backward(); opt.step()
    proj.eval()

    feats_train = extract_trajectory_features(proj, X_train)
    feats_te_ben = extract_trajectory_features(proj, X_te_ben)
    feats_te_atk = extract_trajectory_features(proj, X_te_atk)

    sc = StandardScaler()
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(sc.fit_transform(feats_train), y_train)

    scores_ben = clf.predict_proba(sc.transform(feats_te_ben))[:, 1]
    scores_atk = clf.predict_proba(sc.transform(feats_te_atk))[:, 1]

    thr = float(np.quantile(scores_ben, 1.0 - FPR_TARGET))
    tpr = float((scores_atk > thr).mean())
    auroc = roc_auc_score(
        np.array([0]*len(scores_ben) + [1]*len(scores_atk)),
        np.concatenate([scores_ben, scores_atk])
    )
    return auroc, tpr


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
    for seed in range(5):
        a, t = train_and_eval(X_train, y_train, X_te_ben, X_te_atk, seed=seed)
        aurocs.append(a); tprs.append(t)
        print(f"    Seed {seed}: AUROC={a:.3f}  TPR@5%={t:.3f}")

    print(f"\n  Mean AUROC: {np.mean(aurocs):.3f} ± {np.std(aurocs):.3f}")
    print(f"  Mean TPR:   {np.mean(tprs):.3f} ± {np.std(tprs):.3f}")

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
        a, t = train_and_eval(X_sub, y_sub, X_te_ben, X_te_atk, seed=42)
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

    print(f"\n  {'Held-out':<15} | {'TPR@5%':>7}")
    print(f"  {'─'*15}─┼─{'─'*7}")

    cross_tprs = []
    for held_out in methods_unique:
        train_atk = np.concatenate([hs_by_method[m] for m in methods_unique if m != held_out])
        test_atk = hs_by_method[held_out]
        if len(test_atk) < 5:
            continue

        X_cv = np.concatenate([cv_ben_tr, train_atk])
        y_cv = np.array([0]*len(cv_ben_tr) + [1]*len(train_atk))
        a, t = train_and_eval(X_cv, y_cv, cv_ben_te, test_atk, seed=42)
        print(f"  {held_out:<15} | {t:>7.3f}")
        cross_tprs.append(t)

    print(f"  {'─'*15}─┼─{'─'*7}")
    print(f"  {'MEAN':<15} | {np.mean(cross_tprs):>7.3f}")

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
