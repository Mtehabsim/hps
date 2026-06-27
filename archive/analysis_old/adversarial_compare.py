"""
Adversarial PGD comparison: HPS (mean_r) vs C4 (LR on mean-pooled activations).

Question: Does the geometric constraint in HPS provide robustness that an
unconstrained linear probe lacks?

Setup:
  - Both detectors trained on the same Llama-3 cached activations
  - Both use the same threshold calibration (held-out benign, FPR=5%)
  - PGD attacks the activation tensor (shape: n_layers × d_hidden) directly
  - Attack minimizes the detector's logit (avoids gradient masking)
  - Compare evasion rates at multiple ε budgets

Detectors:
  HPS: Lorentz projection → mean radial position → LR(1 feature)
  C4:  Mean-pool activations across layers → LR(d_hidden features)

Hypothesis: If HPS is robust due to the geometric constraint, evasion rate
should grow more slowly with ε than C4. If both behave similarly, the
geometric prior provides no robustness benefit.

Usage:
  python adversarial_compare.py \
    --test-attacks llama3_attacks.json \
    --harmless data_harmless_6500.csv

Runtime: ~10-20 minutes on cached activations.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from experiment7 import LorentzProjection, contrastive_loss
from rtv_standalone import FPR_TARGET

HPS_LAYERS = [0, 2, 17, 24, 28, 31]
KAPPA_INIT = 0.1
device = "cuda" if torch.cuda.is_available() else "cpu"

EVAL_EPSILONS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
PGD_STEPS = 50


def to_arr(hs_list, layers):
    return np.array([[hs[l][-1] for l in layers] for hs in hs_list])


# ═══════════════════════════════════════════════════════════════════
#  HPS scorer (differentiable, returns logit)
# ═══════════════════════════════════════════════════════════════════
class HPSScorer(nn.Module):
    """Differentiable scorer: projection → mean_r → LR (single weight)."""

    def __init__(self, proj, scaler_mean, scaler_std, clf_coef, clf_intercept):
        super().__init__()
        self.proj = proj
        self.register_buffer("scaler_mean", torch.tensor(scaler_mean, dtype=torch.float32))
        self.register_buffer("scaler_std", torch.tensor(scaler_std, dtype=torch.float32))
        self.register_buffer("clf_coef", torch.tensor(clf_coef, dtype=torch.float32))
        self.register_buffer("clf_intercept", torch.tensor(clf_intercept, dtype=torch.float32))

    def forward(self, h, return_logit=False):
        """h: (n_layers, d_hidden) → logit or sigmoid score."""
        n_layers = h.shape[0]
        # Project each layer
        radii = []
        for l in range(n_layers):
            x = self.proj(h[l].unsqueeze(0)).squeeze(0)  # (d_proj+1,)
            radii.append(x[0])  # time coordinate
        mean_r = torch.stack(radii).mean()  # single scalar
        # Standardize
        feat = (mean_r - self.scaler_mean[0]) / (self.scaler_std[0] + 1e-8)
        logit = feat * self.clf_coef[0] + self.clf_intercept
        if return_logit:
            return logit
        return torch.sigmoid(logit)


# ═══════════════════════════════════════════════════════════════════
#  C4 scorer (differentiable, returns logit)
# ═══════════════════════════════════════════════════════════════════
class C4Scorer(nn.Module):
    """Differentiable scorer: mean-pool activations → LR (d_hidden weights)."""

    def __init__(self, scaler_mean, scaler_std, clf_coef, clf_intercept):
        super().__init__()
        self.register_buffer("scaler_mean", torch.tensor(scaler_mean, dtype=torch.float32))
        self.register_buffer("scaler_std", torch.tensor(scaler_std, dtype=torch.float32))
        self.register_buffer("clf_coef", torch.tensor(clf_coef, dtype=torch.float32))
        self.register_buffer("clf_intercept", torch.tensor(clf_intercept, dtype=torch.float32))

    def forward(self, h, return_logit=False):
        """h: (n_layers, d_hidden) → logit or sigmoid score."""
        # Mean-pool across layers
        pooled = h.mean(dim=0)  # (d_hidden,)
        # Standardize
        feat = (pooled - self.scaler_mean) / (self.scaler_std + 1e-8)
        logit = (feat * self.clf_coef).sum() + self.clf_intercept
        if return_logit:
            return logit
        return torch.sigmoid(logit)


# ═══════════════════════════════════════════════════════════════════
#  Training helpers
# ═══════════════════════════════════════════════════════════════════
def train_hps(X_train, y_train, seed=42, epochs=50):
    n_layers = X_train.shape[1]
    d_hidden = X_train.shape[2]
    torch.manual_seed(seed)
    np.random.seed(seed)
    proj = LorentzProjection(d_hidden, 64, KAPPA_INIT, n_layers=n_layers).to(device)
    proj.log_k.requires_grad = False
    opt = optim.Adam([p for p in proj.parameters() if p.requires_grad],
                     lr=1e-3, weight_decay=1e-5)
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)
    for _ in range(epochs):
        loss = torch.tensor(0.0, device=device)
        for l in range(n_layers):
            h = proj(X_t[:, l, :])
            loss = loss + contrastive_loss(h, y_t, k=proj.k, tau=proj.tau(l))
        loss = loss / n_layers
        opt.zero_grad(); loss.backward(); opt.step()
    proj.eval()
    return proj


def compute_hps_mean_r(X, proj):
    """Compute mean_r feature per sample."""
    radii = []
    with torch.no_grad():
        for i in range(len(X)):
            x = torch.tensor(X[i], dtype=torch.float32, device=device)
            h = proj(x)
            r = h[:, 0].cpu().numpy()
            radii.append(r.mean())
    return np.array(radii).reshape(-1, 1)


def compute_c4_features(X):
    """Mean-pool across layers."""
    return X.mean(axis=1)  # (N, d_hidden)


def calibrate_threshold(scores_calib):
    """5% FPR threshold."""
    return float(np.quantile(scores_calib, 1.0 - FPR_TARGET))


# ═══════════════════════════════════════════════════════════════════
#  PGD attack on logit (avoids gradient masking)
# ═══════════════════════════════════════════════════════════════════
def pgd_minimize(scorer, h, eps, n_steps=50):
    """Minimize the logit (push toward benign)."""
    h0 = h.detach().clone()
    delta = torch.zeros_like(h0, requires_grad=True)
    lr = (eps / max(n_steps, 1)) * 2.5
    for _ in range(n_steps):
        logit = scorer(h0 + delta, return_logit=True)
        grad = torch.autograd.grad(logit, delta, create_graph=False)[0]
        with torch.no_grad():
            delta_new = delta - lr * torch.sign(grad)
            delta_new = torch.clamp(delta_new, -eps, +eps)
        delta = delta_new.detach().requires_grad_(True)
    with torch.no_grad():
        return float(scorer(h0 + delta.detach()))


def pgd_maximize(scorer, h, eps, n_steps=50):
    """Maximize the logit (push toward attack)."""
    h0 = h.detach().clone()
    delta = torch.zeros_like(h0, requires_grad=True)
    lr = (eps / max(n_steps, 1)) * 2.5
    for _ in range(n_steps):
        logit = scorer(h0 + delta, return_logit=True)
        grad = torch.autograd.grad(logit, delta, create_graph=False)[0]
        with torch.no_grad():
            delta_new = delta + lr * torch.sign(grad)
            delta_new = torch.clamp(delta_new, -eps, +eps)
        delta = delta_new.detach().requires_grad_(True)
    with torch.no_grad():
        return float(scorer(h0 + delta.detach()))


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-attacks", required=True)
    parser.add_argument("--harmless", required=True)
    parser.add_argument("--cache", default="results/llama3_activations_cache.npz")
    parser.add_argument("--n-test", type=int, default=100,
                        help="Number of test attacks/benign for PGD evaluation")
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"  ADVERSARIAL COMPARISON — HPS vs C4 under PGD")
    print(f"{'═'*60}\n")

    if not os.path.exists(args.cache):
        print(f"ERROR: Cache not found at {args.cache}")
        return

    cache = np.load(args.cache, allow_pickle=True)
    hs_train_ben = cache["hs_train_ben"].tolist()
    hs_train_atk = cache["hs_train_atk"].tolist()
    hs_test_ben = cache["hs_test_ben"].tolist()
    hs_test_atk = cache["hs_test_atk"].tolist()

    X_tr_ben = to_arr(hs_train_ben, HPS_LAYERS)
    X_tr_atk = to_arr(hs_train_atk, HPS_LAYERS)
    X_te_ben = to_arr(hs_test_ben, HPS_LAYERS)
    X_te_atk = to_arr(hs_test_atk, HPS_LAYERS)

    X_train = np.concatenate([X_tr_ben, X_tr_atk])
    y_train = np.array([0]*len(X_tr_ben) + [1]*len(X_tr_atk))

    print(f"  Train: {len(X_tr_ben)} benign + {len(X_tr_atk)} attacks")
    print(f"  Test:  {len(X_te_ben)} benign + {len(X_te_atk)} attacks")
    print(f"  PGD:   {PGD_STEPS} steps, attacking activation tensor")

    # ══════════════════════════════════════════════════════════════
    #  Train HPS
    # ══════════════════════════════════════════════════════════════
    print(f"\n  [1/4] Training HPS projection...")
    proj_hps = train_hps(X_train, y_train, seed=42)
    feats_hps_train = compute_hps_mean_r(X_train, proj_hps)
    feats_hps_te_ben = compute_hps_mean_r(X_te_ben, proj_hps)
    sc_hps = StandardScaler()
    feats_hps_train_s = sc_hps.fit_transform(feats_hps_train)
    clf_hps = LogisticRegression(max_iter=2000, random_state=42)
    clf_hps.fit(feats_hps_train_s, y_train)
    hps_scores_calib = clf_hps.predict_proba(sc_hps.transform(feats_hps_te_ben))[:, 1]
    hps_thr = calibrate_threshold(hps_scores_calib)
    print(f"    HPS threshold (5% FPR): {hps_thr:.4f}")

    hps_scorer = HPSScorer(
        proj_hps,
        sc_hps.mean_, sc_hps.scale_,
        clf_hps.coef_[0], float(clf_hps.intercept_[0])
    ).to(device).eval()

    # ══════════════════════════════════════════════════════════════
    #  Train C4
    # ══════════════════════════════════════════════════════════════
    print(f"\n  [2/4] Training C4 (linear probe on mean-pooled activations)...")
    feats_c4_train = compute_c4_features(X_train)
    feats_c4_te_ben = compute_c4_features(X_te_ben)
    sc_c4 = StandardScaler()
    feats_c4_train_s = sc_c4.fit_transform(feats_c4_train)
    clf_c4 = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
    clf_c4.fit(feats_c4_train_s, y_train)
    c4_scores_calib = clf_c4.predict_proba(sc_c4.transform(feats_c4_te_ben))[:, 1]
    c4_thr = calibrate_threshold(c4_scores_calib)
    print(f"    C4 threshold (5% FPR):  {c4_thr:.4f}")

    c4_scorer = C4Scorer(
        sc_c4.mean_, sc_c4.scale_,
        clf_c4.coef_[0], float(clf_c4.intercept_[0])
    ).to(device).eval()

    # ══════════════════════════════════════════════════════════════
    #  Sanity check: clean attack detection rates
    # ══════════════════════════════════════════════════════════════
    print(f"\n  [3/4] Clean (no-attack) baseline:")
    n_test = min(args.n_test, len(X_te_atk), len(X_te_ben))
    test_attacks_idx = list(range(n_test))
    test_benign_idx = list(range(n_test))

    hps_clean_scores_atk = []
    c4_clean_scores_atk = []
    for i in test_attacks_idx:
        h = torch.tensor(X_te_atk[i], dtype=torch.float32, device=device)
        with torch.no_grad():
            hps_clean_scores_atk.append(float(hps_scorer(h)))
            c4_clean_scores_atk.append(float(c4_scorer(h)))
    hps_clean_tpr = float((np.array(hps_clean_scores_atk) > hps_thr).mean())
    c4_clean_tpr = float((np.array(c4_clean_scores_atk) > c4_thr).mean())
    print(f"    HPS clean TPR: {hps_clean_tpr:.3f}")
    print(f"    C4  clean TPR: {c4_clean_tpr:.3f}")

    # ══════════════════════════════════════════════════════════════
    #  PGD attack: minimize score on attacks (evasion)
    # ══════════════════════════════════════════════════════════════
    print(f"\n  [4/4] Running PGD evasion attack...\n")
    print(f"  {'ε':<8} | {'HPS evasion':>12} | {'C4 evasion':>11} | {'Δ (HPS more robust if +)':>26}")
    print(f"  {'─'*8}─┼─{'─'*12}─┼─{'─'*11}─┼─{'─'*26}")

    pgd_results = {}
    for eps in EVAL_EPSILONS:
        hps_evaded = 0
        c4_evaded = 0
        for i in test_attacks_idx:
            h = torch.tensor(X_te_atk[i], dtype=torch.float32, device=device)
            # Attack HPS
            hps_score_after = pgd_minimize(hps_scorer, h, eps, n_steps=PGD_STEPS)
            if hps_score_after < hps_thr:
                hps_evaded += 1
            # Attack C4 (separate PGD run, separate delta — fair comparison)
            c4_score_after = pgd_minimize(c4_scorer, h, eps, n_steps=PGD_STEPS)
            if c4_score_after < c4_thr:
                c4_evaded += 1
        hps_evasion = hps_evaded / n_test
        c4_evasion = c4_evaded / n_test
        delta = c4_evasion - hps_evasion   # positive = HPS is more robust
        delta_marker = "  ←  HPS robust" if delta > 0.05 else ("  ←  HPS weak" if delta < -0.05 else "")
        print(f"  {eps:<8.4f} | {hps_evasion:>12.3f} | {c4_evasion:>11.3f} | {delta:>+10.3f}{delta_marker}")
        pgd_results[eps] = {"hps_evasion": hps_evasion, "c4_evasion": c4_evasion}

    # ══════════════════════════════════════════════════════════════
    #  Decision
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  DECISION SUMMARY")
    print(f"{'═'*60}\n")

    print(f"  Robustness gap (C4 evasion - HPS evasion) at small ε:")
    print(f"    ε=0.001: Δ = {pgd_results[0.001]['c4_evasion'] - pgd_results[0.001]['hps_evasion']:+.3f}")
    print(f"    ε=0.005: Δ = {pgd_results[0.005]['c4_evasion'] - pgd_results[0.005]['hps_evasion']:+.3f}")
    print(f"    ε=0.01:  Δ = {pgd_results[0.01]['c4_evasion'] - pgd_results[0.01]['hps_evasion']:+.3f}")
    print(f"    ε=0.05:  Δ = {pgd_results[0.05]['c4_evasion'] - pgd_results[0.05]['hps_evasion']:+.3f}")

    avg_small_eps_delta = np.mean([
        pgd_results[0.001]['c4_evasion'] - pgd_results[0.001]['hps_evasion'],
        pgd_results[0.005]['c4_evasion'] - pgd_results[0.005]['hps_evasion'],
        pgd_results[0.01]['c4_evasion'] - pgd_results[0.01]['hps_evasion'],
    ])

    print(f"\n  ── Verdict ──")
    if avg_small_eps_delta >= 0.20:
        print(f"  ✓ HPS PROVIDES SUBSTANTIAL ROBUSTNESS")
        print(f"    Average Δ at ε∈[0.001, 0.01]: {avg_small_eps_delta:+.3f}")
        print(f"    Geometric prior makes the detector harder to evade")
        print(f"    → Strong contribution for security venue")
    elif avg_small_eps_delta >= 0.05:
        print(f"  ⚠ HPS MODESTLY MORE ROBUST")
        print(f"    Average Δ at ε∈[0.001, 0.01]: {avg_small_eps_delta:+.3f}")
        print(f"    Small but real robustness benefit")
        print(f"    → Defensible but not headline-worthy")
    elif avg_small_eps_delta >= -0.05:
        print(f"  ≈ HPS AND C4 EQUALLY ROBUST (no benefit)")
        print(f"    Average Δ at ε∈[0.001, 0.01]: {avg_small_eps_delta:+.3f}")
        print(f"    The geometric prior provides no robustness advantage")
        print(f"    → Robustness is NOT a paper differentiator")
    else:
        print(f"  ✗ HPS LESS ROBUST THAN C4")
        print(f"    Average Δ at ε∈[0.001, 0.01]: {avg_small_eps_delta:+.3f}")
        print(f"    The 1-feature bottleneck makes HPS easier to attack")
        print(f"    → Robustness story disappears; pivot needed")

    # Save
    out = {
        "config": {"layers": HPS_LAYERS, "kappa_init": KAPPA_INIT, "pgd_steps": PGD_STEPS,
                   "n_test": n_test},
        "thresholds": {"hps": hps_thr, "c4": c4_thr},
        "clean_tpr": {"hps": hps_clean_tpr, "c4": c4_clean_tpr},
        "pgd_evasion": {str(k): v for k, v in pgd_results.items()},
    }
    out_path = "results/adversarial_compare.json"
    os.makedirs("results", exist_ok=True)

    def _np_default(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"Type {type(o)} not serializable")

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=_np_default)
    print(f"\n  Saved → {out_path}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
