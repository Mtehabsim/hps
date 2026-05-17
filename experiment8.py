"""
Experiment 8 — Critical Diagnostics
════════════════════════════════════
Answers the questions that decide the paper's framing:

1. Per-method activation norms (signal general or attack-specific?)
2. Layer ablation: early vs mid vs late (lexical or semantic?)
3. Cross-attack generalization: Raw vs Euclidean vs Hyperbolic on held-out methods
   ← THE CRITICAL TEST — does geometry help when generalization is hard?
4. Refused attacks scoring (trajectory theory test)

Usage:
  python experiment8.py
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score

import config
from utils import load_model, save_json
from experiment7 import (
    extract_all_layers, LorentzProjection, contrastive_loss,
    extract_trajectory_features
)
from dataset import BENIGN, REFUSED


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: train Hyperbolic projection and extract 12 features
# ─────────────────────────────────────────────────────────────────────────────

def train_and_extract_hyperbolic(X_train, y_train, X_eval, n_layers, epochs=120):
    """Train HPS-Full projection on X_train,y_train. Return features for X_eval."""
    device = config.DEVICE
    d_hidden = X_train.shape[2]
    proj = LorentzProjection(d_hidden, config.PROJECTION_DIM, 1.0, n_layers=n_layers).to(device)
    opt = optim.Adam(proj.parameters(), lr=1e-3, weight_decay=1e-5)
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)
    for _ in range(epochs):
        total_loss = torch.tensor(0.0, device=device)
        for l in range(n_layers):
            h = proj(X_t[:, l, :])
            tau_l = proj.tau(l)
            total_loss = total_loss + contrastive_loss(h, y_t, k=proj.k, tau=tau_l)
        total_loss = total_loss / n_layers
        opt.zero_grad()
        total_loss.backward()
        opt.step()
    feats_train = extract_trajectory_features(proj, X_train)
    feats_eval = extract_trajectory_features(proj, X_eval)
    return feats_train, feats_eval


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: train Euclidean projection and extract 12 features
# ─────────────────────────────────────────────────────────────────────────────

def train_and_extract_euclidean(X_train, y_train, X_eval, n_layers, epochs=120):
    """Train Euclidean projection (linear+L2 contrastive). Return 12 features."""
    device = config.DEVICE
    d_hidden = X_train.shape[2]
    d_proj = config.PROJECTION_DIM
    proj_e = nn.Linear(d_hidden, d_proj, bias=False).to(device)
    nn.init.xavier_uniform_(proj_e.weight)
    scale_e = nn.Parameter(torch.tensor(1.0 / np.sqrt(d_proj)).to(device))
    opt = optim.Adam(list(proj_e.parameters()) + [scale_e], lr=1e-3, weight_decay=1e-5)

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)

    for _ in range(epochs):
        total_loss = torch.tensor(0.0, device=device)
        for l in range(n_layers):
            h = (proj_e(X_t[:, l, :]) * scale_e).float()
            dists = torch.cdist(h, h)
            sm = (y_t.unsqueeze(0) == y_t.unsqueeze(1)).float()
            dm = 1.0 - sm
            tr = torch.triu(torch.ones(h.shape[0], h.shape[0], device=device), diagonal=1)
            ns = (sm * tr).sum().clamp(min=1)
            nd = (dm * tr).sum().clamp(min=1)
            same_loss = (dists ** 2 * sm * tr).sum() / ns
            diff_loss = (torch.clamp(2.0 - dists, min=0) ** 2 * dm * tr).sum() / nd
            total_loss = total_loss + (same_loss + diff_loss) / 2.0
        total_loss = total_loss / n_layers
        opt.zero_grad()
        total_loss.backward()
        opt.step()

    proj_e.eval()
    def feats_for(X):
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        out = []
        with torch.no_grad():
            for i in range(len(X)):
                h = (proj_e(Xt[i]) * scale_e).cpu().numpy()
                norms = np.linalg.norm(h, axis=1)
                curv = []
                for j in range(1, n_layers - 1):
                    d_p = np.linalg.norm(h[j] - h[j-1])
                    d_n = np.linalg.norm(h[j+1] - h[j])
                    d_s = np.linalg.norm(h[j+1] - h[j-1])
                    de = d_p + d_n
                    curv.append(0.0 if de < 1e-8 else abs(de - d_s) / de)
                curv = np.array(curv) if curv else np.array([0.0])
                d_total = np.linalg.norm(h[-1] - h[0])
                path = sum(np.linalg.norm(h[j+1] - h[j]) for j in range(n_layers - 1))
                out.append(np.array([
                    norms.mean(), norms.max(), norms.min(), norms.std(),
                    norms.max() - norms.min(),
                    curv.max(), curv.mean(), curv.std(),
                    np.argmax(curv) / max(len(curv), 1),
                    d_total, path, d_total / (path + 1e-8),
                ]))
        return np.array(out)
    return feats_for(X_train), feats_for(X_eval)


def evaluate_classifier(X_train, y_train, X_test, y_test, C=1.0):
    """Fit logistic regression, return AUROC and F1 on test."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    clf = LogisticRegression(C=C, max_iter=2000, random_state=42)
    clf.fit(X_tr_s, y_train)
    scores = clf.predict_proba(X_te_s)[:, 1]
    auroc = roc_auc_score(y_test, scores)
    f1 = f1_score(y_test, (scores > 0.5).astype(int))
    return auroc, f1, scores


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'═'*60}")
    print(f"  Experiment 8 — Critical Diagnostics")
    print(f"  Model: {config.MODEL_NAME}")
    print(f"{'═'*60}\n")

    # ── Load attack method labels ──
    cat_path = os.path.join(config.RESULTS_DIR, "validated_attacks_categorized.json")
    with open(cat_path) as f:
        categorized = json.load(f)

    attack_prompts = []
    attack_methods = []
    for method, prompts in categorized.items():
        for p in prompts:
            attack_prompts.append(p)
            attack_methods.append(method)

    benign_prompts = BENIGN[:350]
    print(f"[exp8] {len(attack_prompts)} attacks, {len(benign_prompts)} benign")
    print(f"  Methods: {dict((m, attack_methods.count(m)) for m in set(attack_methods))}")

    # ── Load model ──
    model, tokenizer = load_model(config.MODEL_NAME, config.DEVICE, config.DTYPE)

    # ── Extract activations at all relevant layers ──
    all_layers = [0, 1, 2, 20, 22, 24, 26, 28, 30, 35, 36, 37, 38, 39]
    all_prompts = benign_prompts + attack_prompts
    labels = np.array([0] * len(benign_prompts) + [1] * len(attack_prompts))
    methods_arr = ["benign"] * len(benign_prompts) + attack_methods

    print(f"\n[exp8] Extracting activations ({len(all_prompts)} prompts, {len(all_layers)} layers)...")
    acts_cache = []
    for i, p in enumerate(all_prompts):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(all_prompts)}")
        acts_cache.append(extract_all_layers(model, tokenizer, p, config.DEVICE, "last"))

    d_hidden = acts_cache[0][all_layers[0]].shape[0]

    def build_X(layer_subset):
        X = np.zeros((len(all_prompts), len(layer_subset), d_hidden))
        for i, act_dict in enumerate(acts_cache):
            for j, l in enumerate(layer_subset):
                if l in act_dict:
                    X[i, j] = act_dict[l]
        return X

    results = {}

    # ══════════════════════════════════════════════════════════════════════════
    #  DIAGNOSTIC 1: Per-method activation norms
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  DIAGNOSTIC 1: Per-method activation norms (early vs late layers)")
    print(f"{'─'*60}")

    norm_results = {}
    for layer_group, gname in [([0, 1, 2], "early"), ([35, 36, 37, 38, 39], "late")]:
        print(f"\n  {gname.capitalize()} layers {layer_group}:")
        X_g = build_X(layer_group)
        norms = np.linalg.norm(X_g.reshape(len(all_prompts), -1), axis=1)
        norm_results[gname] = {}
        benign_n = norms[labels == 0]
        norm_results[gname]["benign"] = {"mean": float(benign_n.mean()), "std": float(benign_n.std())}
        print(f"    {'benign':<32}: {benign_n.mean():.2f} ± {benign_n.std():.2f}")
        for m in sorted(set(attack_methods)):
            mask = np.array([x == m for x in methods_arr])
            mn = norms[mask]
            norm_results[gname][m] = {"mean": float(mn.mean()), "std": float(mn.std())}
            print(f"    {m:<32}: {mn.mean():.2f} ± {mn.std():.2f}")
    results["per_method_norms"] = norm_results

    # ══════════════════════════════════════════════════════════════════════════
    #  DIAGNOSTIC 2: Layer ablation
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  DIAGNOSTIC 2: Layer ablation (Fisher Raw, 80/20 split)")
    print(f"{'─'*60}")

    layer_configs = {
        "Early only (0-2)": [0, 1, 2],
        "Late only (35-39)": [35, 36, 37, 38, 39],
        "Mid only (20-30)": [20, 22, 24, 26, 28, 30],
        "Late+Mid (20-39)": [20, 22, 24, 26, 28, 30, 35, 36, 37, 38, 39],
        "Combined (0-2 + 35-39)": [0, 1, 2, 35, 36, 37, 38, 39],
    }

    n_train = int(0.8 * len(all_prompts))
    perm = np.random.RandomState(42).permutation(len(all_prompts))
    tr_idx, te_idx = perm[:n_train], perm[n_train:]

    print(f"\n  {'Config':<28} | {'AUROC':>7} | {'F1':>6}")
    print(f"  {'─'*28}─┼─{'─'*7}─┼─{'─'*6}")
    layer_results = {}
    for name, lset in layer_configs.items():
        X_l = build_X(lset).reshape(len(all_prompts), -1)
        auroc, f1, _ = evaluate_classifier(X_l[tr_idx], labels[tr_idx], X_l[te_idx], labels[te_idx], C=0.01)
        layer_results[name] = {"auroc": auroc, "f1": f1}
        print(f"  {name:<28} | {auroc:>7.3f} | {f1:>6.3f}")
    results["layer_ablation"] = layer_results

    # ══════════════════════════════════════════════════════════════════════════
    #  DIAGNOSTIC 3: Cross-attack — Raw vs Euclidean vs Hyperbolic
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  DIAGNOSTIC 3: Cross-attack generalization (THE CRITICAL TEST)")
    print(f"  Train on N-1 methods, test on held-out method")
    print(f"  Three projections compared: Raw / Euclidean-Trained / Hyperbolic-Trained")
    print(f"{'─'*60}")

    selected_layers = [0, 1, 2, 35, 36, 37, 38, 39]
    X_full = build_X(selected_layers)
    n_layers_sel = len(selected_layers)
    methods_unique = sorted(set(attack_methods))

    print(f"\n  {'Held-out':<28} | {'n_test':>6} | {'Raw':>6} | {'Euclidean':>9} | {'Hyperbolic':>10}")
    print(f"  {'─'*28}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*9}─┼─{'─'*10}")

    cross_results = {}
    for held_out in methods_unique:
        train_mask = np.array([
            (labels[i] == 0) or (methods_arr[i] != held_out)
            for i in range(len(all_prompts))
        ])
        test_mask = np.array([
            (labels[i] == 0) or (methods_arr[i] == held_out)
            for i in range(len(all_prompts))
        ])
        n_test_atk = int((labels[test_mask] == 1).sum())

        # Raw baseline (high-dim, needs heavy regularization)
        X_flat = X_full.reshape(len(all_prompts), -1)
        auroc_raw, _, scores_raw = evaluate_classifier(
            X_flat[train_mask], labels[train_mask],
            X_flat[test_mask], labels[test_mask],
            C=0.01,
        )

        # Set seed for fair Hyperbolic vs Euclidean comparison
        torch.manual_seed(42)
        np.random.seed(42)

        # Euclidean
        feat_e_tr, feat_e_te = train_and_extract_euclidean(
            X_full[train_mask], labels[train_mask],
            X_full[test_mask], n_layers_sel
        )
        auroc_euc, _, scores_euc = evaluate_classifier(
            feat_e_tr, labels[train_mask], feat_e_te, labels[test_mask]
        )

        # Reset seed before Hyperbolic so both start from same RNG state
        torch.manual_seed(42)
        np.random.seed(42)

        # Hyperbolic
        feat_h_tr, feat_h_te = train_and_extract_hyperbolic(
            X_full[train_mask], labels[train_mask],
            X_full[test_mask], n_layers_sel
        )
        auroc_hyp, _, scores_hyp = evaluate_classifier(
            feat_h_tr, labels[train_mask], feat_h_te, labels[test_mask]
        )

        # Compute FPR/TPR at threshold=0.5
        y_test = labels[test_mask]
        def fpr_tpr(scores, y, t=0.5):
            preds = (scores > t).astype(int)
            n_ben = int((y == 0).sum())
            n_atk = int((y == 1).sum())
            fp = int(((preds == 1) & (y == 0)).sum())
            tp = int(((preds == 1) & (y == 1)).sum())
            return (fp / max(n_ben, 1), tp / max(n_atk, 1))

        fpr_raw, tpr_raw = fpr_tpr(scores_raw, y_test)
        fpr_euc, tpr_euc = fpr_tpr(scores_euc, y_test)
        fpr_hyp, tpr_hyp = fpr_tpr(scores_hyp, y_test)

        print(f"  {held_out:<28} | {n_test_atk:>6} | {auroc_raw:>6.3f} | {auroc_euc:>9.3f} | {auroc_hyp:>10.3f}")
        print(f"    {'@ threshold=0.5: Raw FPR/TPR={:.3f}/{:.3f}  Euc={:.3f}/{:.3f}  Hyp={:.3f}/{:.3f}'.format(fpr_raw, tpr_raw, fpr_euc, tpr_euc, fpr_hyp, tpr_hyp)}")

        cross_results[held_out] = {
            "raw": {"auroc": auroc_raw, "fpr": fpr_raw, "tpr": tpr_raw},
            "euclidean": {"auroc": auroc_euc, "fpr": fpr_euc, "tpr": tpr_euc},
            "hyperbolic": {"auroc": auroc_hyp, "fpr": fpr_hyp, "tpr": tpr_hyp},
            "n_test": n_test_atk,
        }
    results["cross_attack"] = cross_results

    # Aggregate
    mean_raw = np.mean([r["raw"]["auroc"] for r in cross_results.values()])
    mean_euc = np.mean([r["euclidean"]["auroc"] for r in cross_results.values()])
    mean_hyp = np.mean([r["hyperbolic"]["auroc"] for r in cross_results.values()])
    mean_fpr_raw = np.mean([r["raw"]["fpr"] for r in cross_results.values()])
    mean_fpr_euc = np.mean([r["euclidean"]["fpr"] for r in cross_results.values()])
    mean_fpr_hyp = np.mean([r["hyperbolic"]["fpr"] for r in cross_results.values()])
    print(f"  {'─'*28}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*9}─┼─{'─'*10}")
    print(f"  {'MEAN':<28} | {'':>6} | {mean_raw:>6.3f} | {mean_euc:>9.3f} | {mean_hyp:>10.3f}")
    results["cross_attack_mean"] = {"raw": mean_raw, "euclidean": mean_euc, "hyperbolic": mean_hyp}

    # ══════════════════════════════════════════════════════════════════════════
    #  DIAGNOSTIC 4: Refused attacks
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  DIAGNOSTIC 4: Refused attacks — where do they score?")
    print(f"{'─'*60}")

    refused_results = {}
    if REFUSED:
        print(f"  Extracting {len(REFUSED)} refused attack activations...")
        refused_acts = []
        for p in REFUSED:
            refused_acts.append(extract_all_layers(model, tokenizer, p, config.DEVICE, "last"))

        X_ref = np.zeros((len(REFUSED), n_layers_sel, d_hidden))
        for i, act_dict in enumerate(refused_acts):
            for j, l in enumerate(selected_layers):
                if l in act_dict:
                    X_ref[i, j] = act_dict[l]

        # Score with raw classifier trained on all
        X_flat = X_full.reshape(len(all_prompts), -1)
        X_ref_flat = X_ref.reshape(len(REFUSED), -1)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_flat)
        X_ref_s = scaler.transform(X_ref_flat)
        clf_full = LogisticRegression(C=0.01, max_iter=2000, random_state=42)
        clf_full.fit(X_s, labels)
        ref_scores = clf_full.predict_proba(X_ref_s)[:, 1]
        ben_scores = clf_full.predict_proba(X_s[labels == 0])[:, 1]
        atk_scores = clf_full.predict_proba(X_s[labels == 1])[:, 1]

        print(f"\n  {'Category':<22} | {'Mean':>8} | {'Std':>6} | {'Range'}")
        print(f"  {'─'*22}─┼─{'─'*8}─┼─{'─'*6}─┼─{'─'*15}")
        print(f"  {'Benign':<22} | {ben_scores.mean():>8.4f} | {ben_scores.std():>6.4f} | [{ben_scores.min():.3f}, {ben_scores.max():.3f}]")
        print(f"  {'Refused attacks':<22} | {ref_scores.mean():>8.4f} | {ref_scores.std():>6.4f} | [{ref_scores.min():.3f}, {ref_scores.max():.3f}]")
        print(f"  {'Successful attacks':<22} | {atk_scores.mean():>8.4f} | {atk_scores.std():>6.4f} | [{atk_scores.min():.3f}, {atk_scores.max():.3f}]")

        refused_results = {
            "benign_mean": float(ben_scores.mean()),
            "refused_mean": float(ref_scores.mean()),
            "attack_mean": float(atk_scores.mean()),
        }

        if ref_scores.mean() > ben_scores.mean() + 0.1 and ref_scores.mean() < atk_scores.mean() - 0.1:
            print(f"\n  ✓ Refused attacks BETWEEN benign and adversarial — trajectory theory supported")
            refused_results["interpretation"] = "between"
        elif ref_scores.mean() < ben_scores.mean() + 0.1:
            print(f"\n  → Refused attacks score like benign — model resistance is geometrically invisible")
            refused_results["interpretation"] = "like_benign"
        else:
            print(f"\n  → Refused attacks score like successful attacks — detector cannot distinguish")
            refused_results["interpretation"] = "like_attack"
    results["refused_analysis"] = refused_results

    # ── Save ──
    save_json(results, "experiment8_diagnostics.json", config.RESULTS_DIR)

    # ── Summary ──
    print(f"\n{'═'*60}")
    print(f"  DIAGNOSTICS COMPLETE — KEY FINDINGS")
    print(f"{'═'*60}")
    print(f"  Cross-attack mean AUROC:")
    print(f"    Raw:        {mean_raw:.3f}    FPR on benign @ 0.5: {mean_fpr_raw:.3f}")
    print(f"    Euclidean:  {mean_euc:.3f}    FPR on benign @ 0.5: {mean_fpr_euc:.3f}")
    print(f"    Hyperbolic: {mean_hyp:.3f}    FPR on benign @ 0.5: {mean_fpr_hyp:.3f}")
    if mean_hyp > mean_euc + 0.02:
        print(f"  ✓ Hyperbolic generalizes BETTER than Euclidean (Δ={mean_hyp-mean_euc:.3f})")
    elif mean_hyp < mean_euc - 0.02:
        print(f"  ✗ Hyperbolic generalizes WORSE than Euclidean (Δ={mean_hyp-mean_euc:.3f})")
    else:
        print(f"  ≈ Hyperbolic ≈ Euclidean on cross-attack — geometry adds no value")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
