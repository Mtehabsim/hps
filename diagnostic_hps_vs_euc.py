"""
Diagnostic: Isolate why Euclidean beats HPS on Llama-3-8B.

Confounds between Vicuna (HPS originally won) and Llama-3 (Euclidean now wins):
  1. Training set size (252 vs 5216)
  2. Number of attack methods (4 vs 9)
  3. Model architecture (13B vs 8B)
  4. Data source (dataset.py vs JBShield)
  5. Lorentz distance computation (this script previously had a bug)
  6. Training duration (early stopping vs fixed epochs)
  7. Layer selection
  8. Curvature κ initialization

This script tests each variable independently using the Llama-3 cached activations.

Bug fixes vs original version:
  - extract_trajectory_features now uses PROPER Lorentz geodesic distance
    (1/sqrt(k)) * arccosh(-k * inner_lorentz), matching experiment7.py exactly.
    Previous version used sqrt(-inner) which is NOT the geodesic distance.
  - contrastive_loss now uses k-dependent clamp (-1/k instead of hardcoded -1.0)
  - contrastive_loss now uses vectorized matmul (faster than for-loop)
  - extract_trajectory_features no longer relies on global n_layers
  - TEST 7 only uses layer subsets present in the cache
"""

import numpy as np
import os, json, torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# ── Config ──
HPS_LAYERS = [0, 1, 2, 28, 29, 30, 31]
CACHED_LAYERS = [0, 1, 2, 17, 24, 28, 29, 30, 31]   # what the cache actually has
FPR_TARGET = 0.05
device = "cuda" if torch.cuda.is_available() else "cpu"


class LorentzProjection(nn.Module):
    def __init__(self, d_in, d_proj, k_init=1.0, n_layers=7):
        super().__init__()
        self.proj = nn.Linear(d_in, d_proj, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(1.0 / np.sqrt(d_proj)))
        self.log_k = nn.Parameter(torch.tensor(float(np.log(k_init))))
        # match experiment7.py clamp ranges for consistency
        self._tau = nn.Parameter(torch.zeros(n_layers))

    @property
    def k(self):
        return torch.exp(self.log_k).clamp(0.1, 10.0)

    def tau(self, l):
        return torch.exp(self._tau[l]).clamp(0.01, 10.0)

    def forward(self, x):
        h = self.proj(x) * self.scale
        x0 = torch.sqrt(1.0 / self.k + (h ** 2).sum(-1, keepdim=True))
        return torch.cat([x0, h], dim=-1)


def contrastive_loss(embeddings, labels, k, tau, margin=2.0):
    """Vectorized Lorentz contrastive loss with k-dependent clamp.

    Matches experiment7.py:contrastive_loss exactly.
    """
    n = embeddings.shape[0]
    embeddings = embeddings.float()
    # Vectorized Lorentz inner product: -t0^2 + sum(s_i^2)
    inner = -embeddings[:, 0:1] @ embeddings[:, 0:1].T + embeddings[:, 1:] @ embeddings[:, 1:].T

    # k-dependent clamp (was hardcoded -1.0 in old version - WRONG for k != 1)
    if isinstance(k, torch.Tensor):
        clamp_val = (-1.0 / k - 1e-6).detach().item()
        inner = torch.clamp(inner, max=clamp_val)
        dists = (1.0 / torch.sqrt(k)) * torch.acosh(-k * inner)
    else:
        inner = torch.clamp(inner, max=-1.0 / k - 1e-6)
        dists = (1.0 / np.sqrt(k)) * torch.acosh(-k * inner)

    dists = dists / tau

    same = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    diff = 1.0 - same
    triu = torch.triu(torch.ones(n, n, device=embeddings.device), diagonal=1)
    ns = (same * triu).sum().clamp(min=1)
    nd = (diff * triu).sum().clamp(min=1)
    return ((dists**2 * same * triu).sum() / ns +
            (torch.clamp(margin - dists, min=0)**2 * diff * triu).sum() / nd) / 2


def lorentz_dist_np(x, y, k):
    """Proper Lorentz geodesic distance: (1/sqrt(k)) * arccosh(-k * inner_L).

    Matches utils.lorentz_distance used by experiment7.py.
    Old version was: sqrt(-inner_L), which is NOT the geodesic distance.
    """
    inner = -x[0] * y[0] + np.dot(x[1:], y[1:])
    arg = max(-k * inner, 1.0 + 1e-7)   # ensure arccosh argument >= 1
    return float(np.arccosh(arg) / np.sqrt(k))


def extract_trajectory_features(proj, X):
    """Extract 12 trajectory features matching experiment7.py:extract_trajectory_features.

    n_layers is derived from X.shape[1] (was a global before — fragile).
    """
    n_layers_local = X.shape[1]
    k_val = float(proj.k.item())
    feats = []
    proj.eval()
    with torch.no_grad():
        for i in range(len(X)):
            x = torch.tensor(X[i], dtype=torch.float32, device=device)
            pts = []
            for l in range(n_layers_local):
                pts.append(proj(x[l:l+1]).squeeze(0).cpu().numpy())
            h = np.array(pts)
            radii = h[:, 0]
            d_total = lorentz_dist_np(h[0], h[-1], k_val)
            path_len = sum(lorentz_dist_np(h[j], h[j+1], k_val) for j in range(n_layers_local - 1))
            curvs = []
            for j in range(1, n_layers_local - 1):
                dp = lorentz_dist_np(h[j-1], h[j], k_val)
                dn = lorentz_dist_np(h[j], h[j+1], k_val)
                ds = lorentz_dist_np(h[j-1], h[j+1], k_val)
                curvs.append(abs(dp + dn - ds) / (dp + dn + 1e-8))
            curvs = np.array(curvs) if curvs else np.array([0.0])
            feats.append([
                radii.mean(), radii.max(), radii.min(), radii.std(), radii.max() - radii.min(),
                curvs.max(), curvs.mean(), curvs.std() if len(curvs) > 1 else 0,
                np.argmax(curvs) / max(len(curvs), 1),
                d_total, path_len, d_total / (path_len + 1e-8),
            ])
    return np.array(feats)


def extract_euclidean_features(proj_e, scale_per_layer, X):
    """Euclidean trajectory features (12-dim, analogous to hyperbolic version)."""
    n_layers_local = X.shape[1]
    feats = []
    with torch.no_grad():
        for i in range(len(X)):
            x = torch.tensor(X[i], dtype=torch.float32, device=device)
            pts = []
            for l in range(n_layers_local):
                pts.append((proj_e(x[l:l+1]) * scale_per_layer[l]).squeeze(0).cpu().numpy())
            h = np.array(pts)
            norms = np.linalg.norm(h, axis=1)
            d_total = float(np.linalg.norm(h[-1] - h[0]))
            path_len = sum(float(np.linalg.norm(h[j+1] - h[j])) for j in range(n_layers_local - 1))
            curvs = []
            for j in range(1, n_layers_local - 1):
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


def train_and_eval(X_train, y_train, X_te_ben, X_te_atk, seed=42, euclidean=False,
                   max_epochs=200, patience=20, k_init=1.0, freeze_kappa=False):
    """Train projection + classifier. Returns (auroc, tpr@FPR, actual_fpr).

    max_epochs / patience: control training duration (set patience=max_epochs+1
    to disable early stopping).
    k_init / freeze_kappa: control curvature for ablations.
    """
    n_layers = X_train.shape[1]
    d_hidden = X_train.shape[2]
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)

    if euclidean:
        proj_e = nn.Linear(d_hidden, 64, bias=False).to(device)
        nn.init.xavier_uniform_(proj_e.weight)
        scale_per_layer = nn.Parameter(torch.ones(n_layers, device=device) / 8.0)
        log_margin = nn.Parameter(torch.tensor(np.log(2.0), device=device))
        opt = optim.Adam(list(proj_e.parameters()) + [scale_per_layer, log_margin],
                         lr=1e-3, weight_decay=1e-5)
        best_loss = float('inf'); pat = 0
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
                loss = loss + ((dists**2 * sm * tr).sum() / ns +
                               (torch.clamp(margin - dists, min=0)**2 * dm * tr).sum() / nd) / 2
            loss = loss / n_layers
            opt.zero_grad(); loss.backward(); opt.step()
            if loss.item() < best_loss - 1e-4: best_loss = loss.item(); pat = 0
            else: pat += 1
            if pat >= patience: break
        proj_e.eval()
        feats_train = extract_euclidean_features(proj_e, scale_per_layer, X_train)
        feats_te_ben = extract_euclidean_features(proj_e, scale_per_layer, X_te_ben)
        feats_te_atk = extract_euclidean_features(proj_e, scale_per_layer, X_te_atk)
    else:
        proj = LorentzProjection(d_hidden, 64, k_init=k_init, n_layers=n_layers).to(device)
        if freeze_kappa:
            proj.log_k.requires_grad = False
        opt = optim.Adam([p for p in proj.parameters() if p.requires_grad],
                         lr=1e-3, weight_decay=1e-5)
        best_loss = float('inf'); pat = 0
        for _ in range(max_epochs):
            loss = torch.tensor(0.0, device=device)
            for l in range(n_layers):
                h = proj(X_t[:, l, :])
                loss = loss + contrastive_loss(h, y_t, k=proj.k, tau=proj.tau(l))
            loss = loss / n_layers
            opt.zero_grad(); loss.backward(); opt.step()
            if loss.item() < best_loss - 1e-4: best_loss = loss.item(); pat = 0
            else: pat += 1
            if pat >= patience: break
        proj.eval()
        feats_train = extract_trajectory_features(proj, X_train)
        feats_te_ben = extract_trajectory_features(proj, X_te_ben)
        feats_te_atk = extract_trajectory_features(proj, X_te_atk)

    sc = StandardScaler()
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(sc.fit_transform(feats_train), y_train)
    n_calib = len(feats_te_ben) // 2
    if n_calib < 5:
        # too few benign for calib split — fall back to using all as calib (still no leak vs attacks)
        scores_calib = clf.predict_proba(sc.transform(feats_te_ben))[:, 1]
        scores_ben = scores_calib
    else:
        scores_calib = clf.predict_proba(sc.transform(feats_te_ben[:n_calib]))[:, 1]
        scores_ben = clf.predict_proba(sc.transform(feats_te_ben[n_calib:]))[:, 1]
    scores_atk = clf.predict_proba(sc.transform(feats_te_atk))[:, 1]
    thr = float(np.quantile(scores_calib, 1.0 - FPR_TARGET))
    tpr = float((scores_atk > thr).mean())
    fpr_actual = float((scores_ben > thr).mean())
    auroc = roc_auc_score(np.array([0]*len(scores_ben) + [1]*len(scores_atk)),
                          np.concatenate([scores_ben, scores_atk]))
    return auroc, tpr, fpr_actual


def cross_attack_eval(X_all_atk, methods_all, X_ben, methods_unique,
                      euclidean=False, seed=42, max_epochs=200, patience=20):
    """Leave-one-method-out cross-attack evaluation."""
    ben_split = int(0.8 * len(X_ben))
    ben_tr, ben_te = X_ben[:ben_split], X_ben[ben_split:]
    hs_by_method = {m: [] for m in methods_unique}
    for act, method in zip(X_all_atk, methods_all):
        hs_by_method[method].append(act)
    for m in methods_unique:
        hs_by_method[m] = np.array(hs_by_method[m])

    tprs = []
    for held_out in methods_unique:
        train_atk = np.concatenate([hs_by_method[m] for m in methods_unique if m != held_out])
        test_atk = hs_by_method[held_out]
        if len(test_atk) < 5:
            continue
        X_tr = np.concatenate([ben_tr, train_atk])
        y_tr = np.array([0]*len(ben_tr) + [1]*len(train_atk))
        _, t, _ = train_and_eval(X_tr, y_tr, ben_te, test_atk, seed=seed,
                                  euclidean=euclidean, max_epochs=max_epochs,
                                  patience=patience)
        tprs.append(t)
    return np.mean(tprs) if tprs else 0.0


def to_hps_array(hs_list, layers):
    """Convert list of activation dicts to numpy array using specified layers.
    Raises KeyError if any layer is not in the cache."""
    return np.array([[hs[l][-1] for l in layers] for hs in hs_list])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-attacks", required=True)
    parser.add_argument("--harmless", required=True)
    parser.add_argument("--cache", default="results/llama3_activations_cache.npz",
                        help="Path to activation cache")
    parser.add_argument("--vicuna-cache", default=None,
                        help="Optional Vicuna activation cache (run TEST 10 if provided)")
    parser.add_argument("--tests", default="all",
                        help="Comma-separated test numbers to run (e.g. '9' or '4,7,9'). Default 'all'.")
    args = parser.parse_args()

    # Parse which tests to run
    if args.tests.lower() == "all":
        tests_to_run = set(range(1, 11))
    else:
        tests_to_run = set(int(t.strip()) for t in args.tests.split(",") if t.strip())

    print(f"\n{'═'*60}")
    print(f"  DIAGNOSTIC: Why does Euclidean beat HPS on Llama-3?")
    print(f"  Cache: {args.cache}")
    print(f"  Cached layers: {CACHED_LAYERS}")
    print(f"{'═'*60}\n")

    if not os.path.exists(args.cache):
        print(f"ERROR: Cache not found at {args.cache}. Run hps_llama3.py first.")
        return

    cache = np.load(args.cache, allow_pickle=True)
    hs_train_ben = cache["hs_train_ben"].tolist()
    hs_train_atk = cache["hs_train_atk"].tolist()
    hs_test_ben = cache["hs_test_ben"].tolist()
    hs_test_atk = cache["hs_test_atk"].tolist()

    # Verify cache layer compatibility
    sample_keys = sorted(hs_train_ben[0].keys())
    print(f"  Cache contains layers: {sample_keys}")
    if not all(l in sample_keys for l in HPS_LAYERS):
        print(f"  ⚠ Default HPS_LAYERS {HPS_LAYERS} not all present in cache!")

    X_tr_ben = to_hps_array(hs_train_ben, HPS_LAYERS)
    X_tr_atk = to_hps_array(hs_train_atk, HPS_LAYERS)
    X_te_ben = to_hps_array(hs_test_ben, HPS_LAYERS)
    X_te_atk = to_hps_array(hs_test_atk, HPS_LAYERS)

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
    all_methods = [attack_methods[i] for i in atk_idx]
    methods_unique = sorted(set(attack_methods))

    X_all_atk = np.concatenate([X_tr_atk, X_te_atk])
    X_all_ben = np.concatenate([X_tr_ben, X_te_ben])

    # Group attacks by method (used by multiple tests)
    hs_by_method_arr = {m: [] for m in methods_unique}
    for act, method in zip(X_all_atk, all_methods):
        hs_by_method_arr[method].append(act)
    for m in methods_unique:
        hs_by_method_arr[m] = np.array(hs_by_method_arr[m])

    # ══════════════════════════════════════════════════════════════
    #  TEST 1: Data size effect on SAME-DISTRIBUTION
    # ══════════════════════════════════════════════════════════════
    if 1 in tests_to_run:
        print(f"{'─'*60}")
        print(f"  TEST 1: Effect of training set size (same-dist)")
        print(f"  (All 9 methods, vary N attacks)")
        print(f"{'─'*60}\n")

        sizes = [100, 250, 500, 1000, 2000, 5216]
        print(f"  {'N':<6} | {'HPS AUROC':>9} | {'Euc AUROC':>9} | {'Δ':>7}")
        print(f"  {'─'*6}─┼─{'─'*9}─┼─{'─'*9}─┼─{'─'*7}")

        for n in sizes:
            if n > len(X_tr_atk):
                continue
            n_ben = min(n, len(X_tr_ben))
            X_sub = np.concatenate([X_tr_ben[:n_ben], X_tr_atk[:n]])
            y_sub = np.array([0]*n_ben + [1]*n)
            a_hps, _, _ = train_and_eval(X_sub, y_sub, X_te_ben, X_te_atk, seed=42, euclidean=False)
            a_euc, _, _ = train_and_eval(X_sub, y_sub, X_te_ben, X_te_atk, seed=42, euclidean=True)
            print(f"  {n:<6} | {a_hps:>9.3f} | {a_euc:>9.3f} | {a_hps-a_euc:>+7.3f}")

    # ══════════════════════════════════════════════════════════════
    #  TEST 2: Data size effect on CROSS-ATTACK
    # ══════════════════════════════════════════════════════════════
    if 2 in tests_to_run:
        print(f"\n{'─'*60}")
        print(f"  TEST 2: Effect of training set size (cross-attack)")
        print(f"  (All 9 methods, vary N per method, leave-one-out)")
        print(f"{'─'*60}\n")

        per_method_sizes = [25, 50, 100, 250, 500]
        print(f"  {'N/method':<9} | {'HPS TPR':>7} | {'Euc TPR':>7} | {'Δ':>7}")
        print(f"  {'─'*9}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}")

        for n_per in per_method_sizes:
            sub_atk = []
            sub_methods = []
            for m in methods_unique:
                available = hs_by_method_arr[m]
                take = min(n_per, len(available))
                sub_atk.append(available[:take])
                sub_methods.extend([m] * take)
            sub_atk = np.concatenate(sub_atk)

            hps_tpr = cross_attack_eval(sub_atk, sub_methods, X_all_ben, methods_unique, euclidean=False)
            euc_tpr = cross_attack_eval(sub_atk, sub_methods, X_all_ben, methods_unique, euclidean=True)
            print(f"  {n_per:<9} | {hps_tpr:>7.3f} | {euc_tpr:>7.3f} | {hps_tpr-euc_tpr:>+7.3f}")

    # ══════════════════════════════════════════════════════════════
    #  TEST 3: Number of methods (cross-attack)
    # ══════════════════════════════════════════════════════════════
    if 3 in tests_to_run:
        print(f"\n{'─'*60}")
        print(f"  TEST 3: Effect of method diversity (cross-attack)")
        print(f"  (Fix ~250 attacks/method, vary number of methods)")
        print(f"{'─'*60}\n")

        method_counts = [3, 4, 5, 7, 9]
        print(f"  {'#methods':<9} | {'HPS TPR':>7} | {'Euc TPR':>7} | {'Δ':>7}")
        print(f"  {'─'*9}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}")

        for n_methods in method_counts:
            if n_methods > len(methods_unique):
                continue
            subset_methods = methods_unique[:n_methods]
            sub_atk = []
            sub_meth = []
            for m in subset_methods:
                available = hs_by_method_arr[m]
                take = min(250, len(available))
                sub_atk.append(available[:take])
                sub_meth.extend([m] * take)
            sub_atk = np.concatenate(sub_atk)

            hps_tpr = cross_attack_eval(sub_atk, sub_meth, X_all_ben, subset_methods, euclidean=False)
            euc_tpr = cross_attack_eval(sub_atk, sub_meth, X_all_ben, subset_methods, euclidean=True)
            print(f"  {n_methods:<9} | {hps_tpr:>7.3f} | {euc_tpr:>7.3f} | {hps_tpr-euc_tpr:>+7.3f}")

    # ══════════════════════════════════════════════════════════════
    #  TEST 4: Vicuna-like conditions on Llama-3 data
    # ══════════════════════════════════════════════════════════════
    if 4 in tests_to_run:
        print(f"\n{'─'*60}")
        print(f"  TEST 4: Vicuna-like conditions on Llama-3 data")
        print(f"  (4 methods, ~60 attacks each = ~240 total)")
        print(f"{'─'*60}\n")

        # Pick 4 methods that are most analogous to Vicuna's original 4
        # Vicuna had: GCG (suffix), PAIR (semantic), JBC (role-play), prompt_with_random_search (suffix)
        # Closest Llama-3 equivalents: gcg, pair, autodan (template), saa (suffix-search-style)
        vicuna_like = ["gcg", "pair", "autodan", "saa"]
        available_methods = [m for m in vicuna_like if m in hs_by_method_arr]
        print(f"  Using methods: {available_methods}")

        sub_atk = []
        sub_meth = []
        for m in available_methods:
            available = hs_by_method_arr[m]
            take = min(60, len(available))
            sub_atk.append(available[:take])
            sub_meth.extend([m] * take)
        sub_atk = np.concatenate(sub_atk)
        print(f"  Total attacks: {len(sub_atk)} across {len(available_methods)} methods")

        hps_tpr = cross_attack_eval(sub_atk, sub_meth, X_all_ben, available_methods, euclidean=False)
        euc_tpr = cross_attack_eval(sub_atk, sub_meth, X_all_ben, available_methods, euclidean=True)
        print(f"\n  HPS cross-attack TPR:  {hps_tpr:.3f}")
        print(f"  Euc cross-attack TPR:  {euc_tpr:.3f}")
        print(f"  Δ (HPS - Euc):         {hps_tpr - euc_tpr:+.3f}")

    # ══════════════════════════════════════════════════════════════
    #  TEST 5: Early stopping effect on HPS
    # ══════════════════════════════════════════════════════════════
    # X_train_full / y_train_full are needed by TEST 5, 6, 9 — build once if any of these run
    need_full_train = bool({5, 6, 9} & tests_to_run)
    if need_full_train:
        X_train_full = np.concatenate([X_tr_ben, X_tr_atk])
        y_train_full = np.array([0]*len(X_tr_ben) + [1]*len(X_tr_atk))

    if 5 in tests_to_run:
        print(f"\n{'─'*60}")
        print(f"  TEST 5: Early stopping effect on HPS")
        print(f"  (Full data, vary max epochs, NO early stopping)")
        print(f"{'─'*60}\n")

        epoch_counts = [50, 100, 200, 400, 800]
        print(f"  {'Epochs':<7} | {'AUROC':>6} | {'TPR@5%':>7} | {'FPR':>5}")
        print(f"  {'─'*7}─┼─{'─'*6}─┼─{'─'*7}─┼─{'─'*5}")

        for max_ep in epoch_counts:
            # patience > max_epochs disables early stopping
            a, t, fpr = train_and_eval(X_train_full, y_train_full, X_te_ben, X_te_atk,
                                        seed=42, euclidean=False,
                                        max_epochs=max_ep, patience=max_ep + 1)
            print(f"  {max_ep:<7} | {a:>6.3f} | {t:>7.3f} | {fpr:>5.3f}")

    # ══════════════════════════════════════════════════════════════
    #  TEST 6: SAA feature investigation
    # ══════════════════════════════════════════════════════════════
    if 6 in tests_to_run:
        print(f"\n{'─'*60}")
        print(f"  TEST 6: SAA attack investigation")
        print(f"  (Why does HPS score SAA as benign?)")
        print(f"{'─'*60}\n")

        torch.manual_seed(42)
        n_layers_full = X_train_full.shape[1]
        proj_diag = LorentzProjection(X_train_full.shape[2], 64, k_init=1.0,
                                       n_layers=n_layers_full).to(device)
        opt = optim.Adam(proj_diag.parameters(), lr=1e-3, weight_decay=1e-5)
        X_t = torch.tensor(X_train_full, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_train_full, dtype=torch.long, device=device)
        best_loss = float('inf'); pat = 0
        for _ in range(200):
            loss = torch.tensor(0.0, device=device)
            for l in range(n_layers_full):
                h = proj_diag(X_t[:, l, :])
                loss = loss + contrastive_loss(h, y_t, k=proj_diag.k, tau=proj_diag.tau(l))
            loss = loss / n_layers_full
            opt.zero_grad(); loss.backward(); opt.step()
            if loss.item() < best_loss - 1e-4: best_loss = loss.item(); pat = 0
            else: pat += 1
            if pat >= 20: break
        proj_diag.eval()

        saa_acts = hs_by_method_arr.get("saa", np.array([]))
        gcg_acts = hs_by_method_arr.get("gcg", np.array([]))
        pair_acts = hs_by_method_arr.get("pair", np.array([]))

        if len(saa_acts) > 0 and len(gcg_acts) > 0:
            feats_saa = extract_trajectory_features(proj_diag, saa_acts[:100])
            feats_gcg = extract_trajectory_features(proj_diag, gcg_acts[:100])
            feats_pair = extract_trajectory_features(proj_diag, pair_acts[:100]) if len(pair_acts) >= 100 else None
            feats_ben_sample = extract_trajectory_features(proj_diag, X_all_ben[:100])

            feat_names = ["mean_r", "max_r", "min_r", "std_r", "range_r",
                          "max_κ", "mean_κ", "std_κ", "spike_loc",
                          "displacement", "path_len", "progress"]

            print(f"  {'Feature':<12} | {'Benign':>8} | {'SAA':>8} | {'GCG':>8} | {'PAIR':>8} | SAA close to ben?")
            print(f"  {'─'*12}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*15}")
            for i, name in enumerate(feat_names):
                b = feats_ben_sample[:, i].mean()
                s = feats_saa[:, i].mean()
                g = feats_gcg[:, i].mean()
                p = feats_pair[:, i].mean() if feats_pair is not None else float('nan')
                close = "YES" if abs(s - b) < abs(g - b) * 0.5 else "no"
                print(f"  {name:<12} | {b:>8.3f} | {s:>8.3f} | {g:>8.3f} | {p:>8.3f} | {close}")

    # ══════════════════════════════════════════════════════════════
    #  TEST 7: Layer selection effect (only cached layers)
    # ══════════════════════════════════════════════════════════════
    if 7 in tests_to_run:
        print(f"\n{'─'*60}")
        print(f"  TEST 7: Layer selection effect — HPS vs Euclidean")
        print(f"  (Only configurations within cached layers: {CACHED_LAYERS})")
        print(f"{'─'*60}\n")

        layer_configs = {
            "current [0,1,2,28-31]":    [0, 1, 2, 28, 29, 30, 31],
            "shallow only [0,1,2]":     [0, 1, 2],
            "late only [28,29,30,31]":  [28, 29, 30, 31],
            "RTV layers [17,24,31]":    [17, 24, 31],
            "spread [0,2,17,24,28,31]": [0, 2, 17, 24, 28, 31],
            "all cached":               list(CACHED_LAYERS),
        }

        print(f"  {'Config':<28} | {'HPS AUROC':>9} | {'HPS TPR':>7} | {'Euc AUROC':>9} | {'Euc TPR':>7} | {'Δ AUROC':>8}")
        print(f"  {'─'*28}─┼─{'─'*9}─┼─{'─'*7}─┼─{'─'*9}─┼─{'─'*7}─┼─{'─'*8}")

        for name, layers in layer_configs.items():
            if not all(l in CACHED_LAYERS for l in layers):
                print(f"  {name:<28} | (layer not cached)")
                continue
            try:
                X_tr_b2 = to_hps_array(hs_train_ben, layers)
                X_tr_a2 = to_hps_array(hs_train_atk, layers)
                X_te_b2 = to_hps_array(hs_test_ben, layers)
                X_te_a2 = to_hps_array(hs_test_atk, layers)
            except KeyError as e:
                print(f"  {name:<28} | (cache miss: layer {e})")
                continue
            X_sub = np.concatenate([X_tr_b2, X_tr_a2])
            y_sub = np.array([0]*len(X_tr_b2) + [1]*len(X_tr_a2))
            a_h, t_h, _ = train_and_eval(X_sub, y_sub, X_te_b2, X_te_a2, seed=42, euclidean=False)
            a_e, t_e, _ = train_and_eval(X_sub, y_sub, X_te_b2, X_te_a2, seed=42, euclidean=True)
            print(f"  {name:<28} | {a_h:>9.3f} | {t_h:>7.3f} | {a_e:>9.3f} | {t_e:>7.3f} | {a_h-a_e:>+8.3f}")

    # ══════════════════════════════════════════════════════════════
    #  TEST 8: Generalization stress test (4-method train, 5 unseen)
    # ══════════════════════════════════════════════════════════════
    if 8 in tests_to_run:
        print(f"\n{'─'*60}")
        print(f"  TEST 8: Euclidean generalization stress test")
        print(f"  (Train on 4 methods only, test on 5 unseen methods)")
        print(f"{'─'*60}\n")

        train_methods = ["gcg", "autodan", "pair", "drattack"]
        train_methods = [m for m in train_methods if m in hs_by_method_arr]
        test_methods_ood = [m for m in methods_unique if m not in train_methods]

        train_atk_stress = np.concatenate([hs_by_method_arr[m] for m in train_methods]) if train_methods else np.array([])
        test_atk_stress = np.concatenate([hs_by_method_arr[m] for m in test_methods_ood]) if test_methods_ood else np.array([])

        if len(train_atk_stress) == 0 or len(test_atk_stress) == 0:
            print("  Skip: not enough data")
        else:
            n_train_ben = min(4000, len(X_all_ben) - 200)
            X_tr_stress = np.concatenate([X_all_ben[:n_train_ben], train_atk_stress])
            y_tr_stress = np.array([0]*n_train_ben + [1]*len(train_atk_stress))
            X_te_ben_stress = X_all_ben[n_train_ben:]

            print(f"  Train: {n_train_ben} benign + {len(train_atk_stress)} attacks ({train_methods})")
            print(f"  Test:  {len(X_te_ben_stress)} benign + {len(test_atk_stress)} attacks ({test_methods_ood})")

            a_hps, t_hps, _ = train_and_eval(X_tr_stress, y_tr_stress, X_te_ben_stress, test_atk_stress,
                                              seed=42, euclidean=False)
            a_euc, t_euc, _ = train_and_eval(X_tr_stress, y_tr_stress, X_te_ben_stress, test_atk_stress,
                                              seed=42, euclidean=True)

            print(f"\n  {'Method':<10} | {'AUROC':>6} | {'TPR@5%':>7}")
            print(f"  {'─'*10}─┼─{'─'*6}─┼─{'─'*7}")
            print(f"  {'HPS':<10} | {a_hps:>6.3f} | {t_hps:>7.3f}")
            print(f"  {'Euclidean':<10} | {a_euc:>6.3f} | {t_euc:>7.3f}")
            print(f"  {'Δ':<10} | {a_hps-a_euc:>+6.3f} | {t_hps-t_euc:>+7.3f}")

    # ══════════════════════════════════════════════════════════════
    #  TEST 9: Curvature κ ablation
    # ══════════════════════════════════════════════════════════════
    if 9 in tests_to_run:
        print(f"\n{'─'*60}")
        print(f"  TEST 9: Curvature κ effect (HPS only)")
        print(f"  (Vary k_init and freeze status)")
        print(f"{'─'*60}\n")

        print(f"  {'κ_init':<8} | {'frozen?':<8} | {'AUROC':>6} | {'TPR@5%':>7} | {'final κ':>8}")
        print(f"  {'─'*8}─┼─{'─'*8}─┼─{'─'*6}─┼─{'─'*7}─┼─{'─'*8}")

        def sanitize(X):
            """Replace NaN/Inf and add tiny noise to zero-variance columns."""
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            col_std = X.std(axis=0)
            zero_cols = col_std < 1e-12
            if zero_cols.any():
                X = X.copy()
                X[:, zero_cols] += np.random.RandomState(0).normal(
                    scale=1e-6, size=(X.shape[0], int(zero_cols.sum())))
            return X

        for k_init in [0.1, 0.5, 1.0, 2.0, 10.0]:
            for frozen in [False, True]:
                try:
                    torch.manual_seed(42)
                    np.random.seed(42)
                    n_layers_kab = X_train_full.shape[1]
                    proj_kab = LorentzProjection(X_train_full.shape[2], 64, k_init=k_init,
                                                  n_layers=n_layers_kab).to(device)
                    if frozen:
                        proj_kab.log_k.requires_grad = False
                    opt_kab = optim.Adam([p for p in proj_kab.parameters() if p.requires_grad],
                                          lr=1e-3, weight_decay=1e-5)
                    X_t = torch.tensor(X_train_full, dtype=torch.float32, device=device)
                    y_t = torch.tensor(y_train_full, dtype=torch.long, device=device)
                    bl = float('inf'); pc = 0
                    for _ in range(200):
                        loss = torch.tensor(0.0, device=device)
                        for l in range(n_layers_kab):
                            h = proj_kab(X_t[:, l, :])
                            loss = loss + contrastive_loss(h, y_t, k=proj_kab.k, tau=proj_kab.tau(l))
                        loss = loss / n_layers_kab
                        opt_kab.zero_grad(); loss.backward(); opt_kab.step()
                        if not torch.isfinite(loss):
                            raise ValueError(f"Non-finite loss at k_init={k_init}, frozen={frozen}")
                        if loss.item() < bl - 1e-4: bl = loss.item(); pc = 0
                        else: pc += 1
                        if pc >= 20: break
                    proj_kab.eval()
                    f_tr = sanitize(extract_trajectory_features(proj_kab, X_train_full))
                    f_ben = sanitize(extract_trajectory_features(proj_kab, X_te_ben))
                    f_atk = sanitize(extract_trajectory_features(proj_kab, X_te_atk))
                    sc_kab = StandardScaler()
                    clf_kab = LogisticRegression(max_iter=2000, random_state=42)
                    clf_kab.fit(sc_kab.fit_transform(f_tr), y_train_full)
                    n_cal = len(f_ben) // 2
                    s_cal = clf_kab.predict_proba(sc_kab.transform(f_ben[:n_cal]))[:, 1]
                    s_ben_eval = clf_kab.predict_proba(sc_kab.transform(f_ben[n_cal:]))[:, 1]
                    s_atk = clf_kab.predict_proba(sc_kab.transform(f_atk))[:, 1]
                    thr = float(np.quantile(s_cal, 1.0 - FPR_TARGET))
                    tpr = float((s_atk > thr).mean())
                    auroc = roc_auc_score(np.array([0]*len(s_ben_eval) + [1]*len(s_atk)),
                                          np.concatenate([s_ben_eval, s_atk]))
                    final_k = float(proj_kab.k.item())
                    print(f"  {k_init:<8.2f} | {str(frozen):<8} | {auroc:>6.3f} | {tpr:>7.3f} | {final_k:>8.3f}")
                except Exception as e:
                    print(f"  {k_init:<8.2f} | {str(frozen):<8} | (failed: {type(e).__name__})")

    # ══════════════════════════════════════════════════════════════
    #  TEST 10: Vicuna activations (optional, if cache provided)
    # ══════════════════════════════════════════════════════════════
    if 10 in tests_to_run and args.vicuna_cache and os.path.exists(args.vicuna_cache):
        print(f"\n{'─'*60}")
        print(f"  TEST 10: Diagnostic on Vicuna activations")
        print(f"  (Cache: {args.vicuna_cache})")
        print(f"{'─'*60}\n")
        try:
            vc = np.load(args.vicuna_cache, allow_pickle=True)
            X_v_train = vc["X_train"]   # (N, n_layers, d_hidden)
            y_v = vc["labels"]
            print(f"  Vicuna cache loaded: {X_v_train.shape}, {len(y_v)} labels")
            # Held-out 20% split
            rng_v = np.random.RandomState(42)
            idx_v = rng_v.permutation(len(X_v_train))
            split_v = int(0.8 * len(idx_v))
            tr_idx, te_idx = idx_v[:split_v], idx_v[split_v:]
            X_v_tr, y_v_tr = X_v_train[tr_idx], y_v[tr_idx]
            X_v_te, y_v_te = X_v_train[te_idx], y_v[te_idx]
            X_v_te_ben = X_v_te[y_v_te == 0]
            X_v_te_atk = X_v_te[y_v_te == 1]

            a_h, t_h, _ = train_and_eval(X_v_tr, y_v_tr, X_v_te_ben, X_v_te_atk,
                                          seed=42, euclidean=False)
            a_e, t_e, _ = train_and_eval(X_v_tr, y_v_tr, X_v_te_ben, X_v_te_atk,
                                          seed=42, euclidean=True)
            print(f"\n  {'Method':<10} | {'AUROC':>6} | {'TPR@5%':>7}")
            print(f"  {'─'*10}─┼─{'─'*6}─┼─{'─'*7}")
            print(f"  {'HPS':<10} | {a_h:>6.3f} | {t_h:>7.3f}")
            print(f"  {'Euclidean':<10} | {a_e:>6.3f} | {t_e:>7.3f}")
            print(f"  {'Δ':<10} | {a_h-a_e:>+6.3f} | {t_h-t_e:>+7.3f}")
        except Exception as e:
            print(f"  Failed to load/run Vicuna cache: {e}")
    elif 10 in tests_to_run:
        print(f"\n  (TEST 10 skipped — pass --vicuna-cache to run on Vicuna data)")

    # ══════════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  HOW TO INTERPRET RESULTS")
    print(f"{'═'*60}")
    print(f"  TEST 1+2: Δ should DECREASE as N grows if hyperbolic is a low-data regularizer.")
    print(f"  TEST 3:   Δ should DECREASE as #methods grows if hyperbolic helps low-diversity.")
    print(f"  TEST 4:   Δ > 0 here would replicate the original Vicuna 'hyperbolic wins'.")
    print(f"  TEST 5:   If AUROC keeps rising past 200 epochs, early stopping is the culprit.")
    print(f"  TEST 6:   SAA feature distribution — explains the per-attack failure mode.")
    print(f"  TEST 7:   Best layer config may differ from the Fisher-discovered set.")
    print(f"  TEST 8:   Train-on-4 / test-on-5 — extreme cross-attack stress test.")
    print(f"  TEST 9:   Does learnable κ matter? Compare frozen vs learned at different inits.")
    print(f"  TEST 10:  Direct test on Vicuna data (if cache supplied).")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
