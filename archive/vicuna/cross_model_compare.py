"""
Cross-model replication: Does HPS vs C4 finding hold on Vicuna-13B?

Direct weight transfer is impossible (Vicuna 5120-dim vs Llama-3 4096-dim).
Instead, we test whether the same conclusions reproduce when the pipeline
is re-trained on Vicuna activations.

Tests:
  Same-distribution: Does HPS=C4 hold on Vicuna?
  Cold-start: Does HPS dominate at low data on Vicuna?
  Adversarial: Is C4 more robust on Vicuna too?

If HPS shows a unique advantage on Vicuna that C4 lacks (or vice versa),
that's a model-dependent finding worth reporting.

Usage:
  # First-time: extract activations (slow, ~30 min)
  python cross_model_compare.py --extract --max-prompts 320

  # Subsequent runs (uses cache):
  python cross_model_compare.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import config
from utils import load_model
from experiment7 import LorentzProjection, contrastive_loss, extract_all_layers
from rtv_standalone import FPR_TARGET

# Vicuna spread layers (40-layer model, proportional to Llama-3's [0,2,17,24,28,31] in 32-layer model)
VICUNA_SPREAD_LAYERS = [0, 2, 22, 31, 35, 39]
KAPPA_INIT = 0.1
EPOCHS = 50
PGD_STEPS = 50
FPR_TARGET_LOCAL = 0.05
device = "cuda" if torch.cuda.is_available() else "cpu"

VICUNA_CACHE = "results/vicuna_activations_cache.npz"


# ═══════════════════════════════════════════════════════════════════
#  Activation extraction (slow — uses LLM)
# ═══════════════════════════════════════════════════════════════════
def extract_vicuna_activations(max_prompts=None, layers=VICUNA_SPREAD_LAYERS):
    """Extract Vicuna activations and save to cache."""
    from dataset import BENIGN, ADVERSARIAL
    import json as _json

    cat_path = os.path.join(config.RESULTS_DIR, "validated_attacks_categorized.json")
    if os.path.exists(cat_path):
        with open(cat_path) as f:
            categorized = _json.load(f)
        attack_methods_list = []
        attacks = []
        for method, prompts in categorized.items():
            for p in prompts:
                if p:
                    attacks.append(p)
                    attack_methods_list.append(method)
    else:
        attacks = ADVERSARIAL
        attack_methods_list = ["unknown"] * len(attacks)

    benign = list(BENIGN)

    if max_prompts:
        # Sample proportionally
        n_atk = min(max_prompts // 2, len(attacks))
        n_ben = min(max_prompts - n_atk, len(benign))
        # Stratify attacks by method
        rng = np.random.RandomState(42)
        atk_idx = rng.permutation(len(attacks))[:n_atk]
        ben_idx = rng.permutation(len(benign))[:n_ben]
        attacks = [attacks[i] for i in atk_idx]
        attack_methods_list = [attack_methods_list[i] for i in atk_idx]
        benign = [benign[i] for i in ben_idx]

    print(f"[extract] Vicuna: {len(benign)} benign + {len(attacks)} attacks")
    print(f"[extract] Attack methods: {sorted(set(attack_methods_list))}")
    print(f"[extract] Layers: {layers}")

    print(f"[extract] Loading Vicuna-13B...")
    model, tokenizer = load_model(config.MODEL_NAME, config.DEVICE, config.DTYPE)

    print(f"[extract] Extracting benign activations...")
    ben_acts = []
    for i, p in enumerate(benign):
        d = extract_all_layers(model, tokenizer, p, config.DEVICE, "last")
        ben_acts.append(np.array([d[l] for l in layers if l in d]))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(benign)}")

    print(f"[extract] Extracting attack activations...")
    atk_acts = []
    for i, p in enumerate(attacks):
        d = extract_all_layers(model, tokenizer, p, config.DEVICE, "last")
        atk_acts.append(np.array([d[l] for l in layers if l in d]))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(attacks)}")

    X_ben = np.array(ben_acts)
    X_atk = np.array(atk_acts)

    np.savez(VICUNA_CACHE,
             X_benign=X_ben, X_attack=X_atk,
             attack_methods=np.array(attack_methods_list, dtype=object),
             layers=np.array(layers))
    print(f"[extract] Saved → {VICUNA_CACHE}")
    print(f"[extract] X_benign: {X_ben.shape}, X_attack: {X_atk.shape}")


# ═══════════════════════════════════════════════════════════════════
#  Training & evaluation helpers (mirror adversarial_compare.py)
# ═══════════════════════════════════════════════════════════════════
def train_hps(X_train, y_train, seed=42, epochs=50):
    n_layers = X_train.shape[1]
    d_hidden = X_train.shape[2]
    torch.manual_seed(seed); np.random.seed(seed)
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


def feat_hps_mean_r(X, proj):
    radii = []
    with torch.no_grad():
        for i in range(len(X)):
            x = torch.tensor(X[i], dtype=torch.float32, device=device)
            h = proj(x)
            r = h[:, 0].cpu().numpy()
            radii.append(r.mean())
    return np.array(radii).reshape(-1, 1)


def feat_c4(X):
    return X.mean(axis=1)  # (N, d_hidden)


def evaluate_clf(feats_train, y_train, feats_te_ben, feats_te_atk, seed=42):
    if feats_train.ndim == 1:
        feats_train = feats_train.reshape(-1, 1)
        feats_te_ben = feats_te_ben.reshape(-1, 1)
        feats_te_atk = feats_te_atk.reshape(-1, 1)
    sc = StandardScaler()
    f_tr_s = sc.fit_transform(feats_train)
    clf = LogisticRegression(max_iter=2000, random_state=seed, C=1.0)
    clf.fit(f_tr_s, y_train)
    n_calib = max(len(feats_te_ben) // 2, 1)
    s_calib = clf.predict_proba(sc.transform(feats_te_ben[:n_calib]))[:, 1]
    s_ben = clf.predict_proba(sc.transform(feats_te_ben[n_calib:]))[:, 1] if n_calib < len(feats_te_ben) else s_calib
    s_atk = clf.predict_proba(sc.transform(feats_te_atk))[:, 1]
    thr = float(np.quantile(s_calib, 1.0 - FPR_TARGET_LOCAL))
    tpr = float((s_atk > thr).mean())
    fpr = float((s_ben > thr).mean()) if len(s_ben) > 0 else 0.0
    auroc = roc_auc_score(np.array([0]*len(s_ben) + [1]*len(s_atk)),
                          np.concatenate([s_ben, s_atk])) if len(s_ben) > 0 else float('nan')
    return auroc, tpr, fpr, sc, clf


# ═══════════════════════════════════════════════════════════════════
#  PGD (mirroring adversarial_compare.py)
# ═══════════════════════════════════════════════════════════════════
class HPSScorer(nn.Module):
    def __init__(self, proj, sc, clf):
        super().__init__()
        self.proj = proj
        self.register_buffer("scaler_mean", torch.tensor(sc.mean_, dtype=torch.float32))
        self.register_buffer("scaler_std", torch.tensor(sc.scale_, dtype=torch.float32))
        self.register_buffer("clf_coef", torch.tensor(clf.coef_[0], dtype=torch.float32))
        self.register_buffer("clf_intercept", torch.tensor(float(clf.intercept_[0]), dtype=torch.float32))

    def forward(self, h, return_logit=False):
        n_layers = h.shape[0]
        radii = []
        for l in range(n_layers):
            x = self.proj(h[l].unsqueeze(0)).squeeze(0)
            radii.append(x[0])
        mean_r = torch.stack(radii).mean()
        feat = (mean_r - self.scaler_mean[0]) / (self.scaler_std[0] + 1e-8)
        logit = feat * self.clf_coef[0] + self.clf_intercept
        return logit if return_logit else torch.sigmoid(logit)


class C4Scorer(nn.Module):
    def __init__(self, sc, clf):
        super().__init__()
        self.register_buffer("scaler_mean", torch.tensor(sc.mean_, dtype=torch.float32))
        self.register_buffer("scaler_std", torch.tensor(sc.scale_, dtype=torch.float32))
        self.register_buffer("clf_coef", torch.tensor(clf.coef_[0], dtype=torch.float32))
        self.register_buffer("clf_intercept", torch.tensor(float(clf.intercept_[0]), dtype=torch.float32))

    def forward(self, h, return_logit=False):
        pooled = h.mean(dim=0)
        feat = (pooled - self.scaler_mean) / (self.scaler_std + 1e-8)
        logit = (feat * self.clf_coef).sum() + self.clf_intercept
        return logit if return_logit else torch.sigmoid(logit)


def pgd_minimize(scorer, h, eps, n_steps=50):
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


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true",
                        help="Extract Vicuna activations (slow, requires Vicuna model)")
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Cap on total prompts when extracting (for testing)")
    parser.add_argument("--n-pgd-test", type=int, default=50)
    args = parser.parse_args()

    if args.extract:
        extract_vicuna_activations(max_prompts=args.max_prompts)
        return

    print(f"\n{'═'*60}")
    print(f"  CROSS-MODEL REPLICATION — Vicuna-13B")
    print(f"  Question: Do HPS vs C4 findings reproduce on Vicuna?")
    print(f"{'═'*60}\n")

    if not os.path.exists(VICUNA_CACHE):
        print(f"ERROR: Vicuna cache not found at {VICUNA_CACHE}")
        print(f"Run with --extract first (takes ~30 min).")
        return

    print(f"  Loading Vicuna activation cache...")
    cache = np.load(VICUNA_CACHE, allow_pickle=True)
    X_ben = cache["X_benign"]    # (N_ben, n_layers, d_hidden)
    X_atk = cache["X_attack"]    # (N_atk, n_layers, d_hidden)
    attack_methods = cache["attack_methods"].tolist()
    layers_used = cache["layers"].tolist()
    print(f"  Vicuna: {len(X_ben)} benign, {len(X_atk)} attacks, layers={layers_used}")
    print(f"  Attack methods: {sorted(set(attack_methods))}")

    rng = np.random.RandomState(42)
    n_ben_te = min(int(0.2 * len(X_ben)), len(X_ben))
    n_atk_te = min(int(0.2 * len(X_atk)), len(X_atk))
    ben_idx = rng.permutation(len(X_ben))
    atk_idx = rng.permutation(len(X_atk))
    test_ben = X_ben[ben_idx[:n_ben_te]]
    train_ben = X_ben[ben_idx[n_ben_te:]]
    test_atk = X_atk[atk_idx[:n_atk_te]]
    train_atk = X_atk[atk_idx[n_atk_te:]]
    test_atk_methods = [attack_methods[atk_idx[i]] for i in range(n_atk_te)]
    train_atk_methods = [attack_methods[atk_idx[i]] for i in range(n_atk_te, len(atk_idx))]

    methods_unique = sorted(set(attack_methods))
    print(f"  Train: {len(train_ben)} benign + {len(train_atk)} attacks")
    print(f"  Test:  {len(test_ben)} benign + {len(test_atk)} attacks")

    X_train = np.concatenate([train_ben, train_atk])
    y_train = np.array([0]*len(train_ben) + [1]*len(train_atk))

    # ══════════════════════════════════════════════════════════════
    #  PART A: Same-distribution (Vicuna)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  PART A: Same-distribution full data (Vicuna)")
    print(f"{'─'*60}\n")

    print(f"  Training HPS...")
    proj_hps = train_hps(X_train, y_train, seed=42)
    f_tr_h = feat_hps_mean_r(X_train, proj_hps)
    f_be_h = feat_hps_mean_r(test_ben, proj_hps)
    f_at_h = feat_hps_mean_r(test_atk, proj_hps)
    a_h, t_h, fpr_h, sc_h, clf_h = evaluate_clf(f_tr_h, y_train, f_be_h, f_at_h)

    print(f"  Training C4 (LR on mean-pooled activations)...")
    f_tr_c = feat_c4(X_train)
    f_be_c = feat_c4(test_ben)
    f_at_c = feat_c4(test_atk)
    a_c, t_c, fpr_c, sc_c, clf_c = evaluate_clf(f_tr_c, y_train, f_be_c, f_at_c)

    print(f"\n  {'Method':<10} | {'AUROC':>6} | {'TPR@5%':>7} | {'FPR':>5}")
    print(f"  {'─'*10}─┼─{'─'*6}─┼─{'─'*7}─┼─{'─'*5}")
    print(f"  {'HPS':<10} | {a_h:>6.3f} | {t_h:>7.3f} | {fpr_h:>5.3f}")
    print(f"  {'C4':<10} | {a_c:>6.3f} | {t_c:>7.3f} | {fpr_c:>5.3f}")
    print(f"  {'Δ AUROC':<10} | {a_h-a_c:>+6.3f}")

    # ══════════════════════════════════════════════════════════════
    #  PART B: Cold-start cross-attack (4 methods, varying N)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  PART B: Cold-start cross-attack (Vicuna, leave-one-out)")
    print(f"{'─'*60}\n")

    # Group all attacks by method
    all_atk_array = np.concatenate([test_atk, train_atk])
    all_atk_methods = test_atk_methods + train_atk_methods
    hs_by_method = {m: [] for m in methods_unique}
    for act, m in zip(all_atk_array, all_atk_methods):
        hs_by_method[m].append(act)
    for m in methods_unique:
        hs_by_method[m] = np.array(hs_by_method[m])

    all_ben_array = np.concatenate([test_ben, train_ben])
    ben_split = int(0.8 * len(all_ben_array))
    cv_ben_tr = all_ben_array[:ben_split]
    cv_ben_te = all_ben_array[ben_split:]

    print(f"  {'N/method':<9} | {'HPS TPR':>7} | {'C4 TPR':>7} | {'Δ':>7}")
    print(f"  {'─'*9}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}")
    cold_results = []
    sizes = [5, 10, 25, 50]
    for n_per in sizes:
        # Skip if any method has too few examples for the held-out test
        valid = all(len(hs_by_method[m]) >= 5 for m in methods_unique)
        if not valid:
            print(f"  {n_per:<9} | (insufficient data per method)")
            continue
        hps_tprs, c4_tprs = [], []
        for held_out in methods_unique:
            sub_atk = []
            for m in methods_unique:
                if m != held_out:
                    avail = hs_by_method[m]
                    take = min(n_per, len(avail))
                    sub_atk.append(avail[:take])
            if not sub_atk:
                continue
            train_atk_sub = np.concatenate(sub_atk)
            test_atk_held = hs_by_method[held_out]
            X_tr = np.concatenate([cv_ben_tr, train_atk_sub])
            y_tr = np.array([0]*len(cv_ben_tr) + [1]*len(train_atk_sub))
            # HPS
            proj_x = train_hps(X_tr, y_tr, seed=42)
            f_h_tr = feat_hps_mean_r(X_tr, proj_x)
            f_h_be = feat_hps_mean_r(cv_ben_te, proj_x)
            f_h_at = feat_hps_mean_r(test_atk_held, proj_x)
            _, t_hps, _, _, _ = evaluate_clf(f_h_tr, y_tr, f_h_be, f_h_at)
            # C4
            f_c_tr = feat_c4(X_tr)
            f_c_be = feat_c4(cv_ben_te)
            f_c_at = feat_c4(test_atk_held)
            _, t_c4, _, _, _ = evaluate_clf(f_c_tr, y_tr, f_c_be, f_c_at)
            hps_tprs.append(t_hps)
            c4_tprs.append(t_c4)
        if hps_tprs:
            mean_hps = float(np.mean(hps_tprs))
            mean_c4 = float(np.mean(c4_tprs))
            delta = mean_hps - mean_c4
            print(f"  {n_per:<9} | {mean_hps:>7.3f} | {mean_c4:>7.3f} | {delta:>+7.3f}")
            cold_results.append({"n_per": n_per, "hps": mean_hps, "c4": mean_c4, "delta": delta})

    # ══════════════════════════════════════════════════════════════
    #  PART C: Adversarial PGD (Vicuna)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  PART C: Adversarial PGD comparison (Vicuna)")
    print(f"{'─'*60}\n")

    hps_scorer = HPSScorer(proj_hps, sc_h, clf_h).to(device).eval()
    c4_scorer = C4Scorer(sc_c, clf_c).to(device).eval()

    n_test = min(args.n_pgd_test, len(test_atk))
    hps_thr = float(np.quantile(
        clf_h.predict_proba(sc_h.transform(f_be_h[:max(len(f_be_h)//2, 1)]))[:, 1],
        1.0 - FPR_TARGET_LOCAL))
    c4_thr = float(np.quantile(
        clf_c.predict_proba(sc_c.transform(f_be_c[:max(len(f_be_c)//2, 1)]))[:, 1],
        1.0 - FPR_TARGET_LOCAL))
    print(f"  HPS threshold: {hps_thr:.4f}")
    print(f"  C4 threshold:  {c4_thr:.4f}")
    print(f"  PGD test attacks: {n_test}\n")

    print(f"  {'ε':<8} | {'HPS evasion':>12} | {'C4 evasion':>11} | {'Δ':>7}")
    print(f"  {'─'*8}─┼─{'─'*12}─┼─{'─'*11}─┼─{'─'*7}")
    pgd_results = []
    for eps in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
        hps_evaded, c4_evaded = 0, 0
        for i in range(n_test):
            h = torch.tensor(test_atk[i], dtype=torch.float32, device=device)
            hps_score = pgd_minimize(hps_scorer, h, eps, n_steps=PGD_STEPS)
            c4_score = pgd_minimize(c4_scorer, h, eps, n_steps=PGD_STEPS)
            if hps_score < hps_thr: hps_evaded += 1
            if c4_score < c4_thr: c4_evaded += 1
        hps_e = hps_evaded / n_test
        c4_e = c4_evaded / n_test
        delta = c4_e - hps_e
        print(f"  {eps:<8.4f} | {hps_e:>12.3f} | {c4_e:>11.3f} | {delta:>+7.3f}")
        pgd_results.append({"eps": eps, "hps_evasion": hps_e, "c4_evasion": c4_e})

    # ══════════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  SUMMARY — Vicuna vs Llama-3 finding consistency")
    print(f"{'═'*60}")
    print(f"\n  Same-distribution (Vicuna):  HPS={a_h:.3f}, C4={a_c:.3f}, Δ={a_h-a_c:+.3f}")
    print(f"  (Llama-3 reference:          HPS=1.000, C4=0.999, Δ=+0.001)")
    if cold_results:
        cs5 = next((r for r in cold_results if r["n_per"] == 5), None)
        cs25 = next((r for r in cold_results if r["n_per"] == 25), None)
        if cs5:
            print(f"\n  Cold-start N=5:   HPS={cs5['hps']:.3f}, C4={cs5['c4']:.3f}, Δ={cs5['delta']:+.3f}")
            print(f"  (Llama-3 reference: HPS=0.996, C4=0.992, Δ=+0.004)")
        if cs25:
            print(f"  Cold-start N=25:  HPS={cs25['hps']:.3f}, C4={cs25['c4']:.3f}, Δ={cs25['delta']:+.3f}")
            print(f"  (Llama-3 reference: HPS=0.998, C4=0.996, Δ=+0.002)")
    if pgd_results:
        eps05 = next((r for r in pgd_results if r["eps"] == 0.05), None)
        if eps05:
            print(f"\n  PGD at ε=0.05:    HPS evasion={eps05['hps_evasion']:.3f}, C4 evasion={eps05['c4_evasion']:.3f}")
            print(f"  (Llama-3 reference: HPS=0.660, C4=0.020 — C4 much more robust)")

    # Decision
    print(f"\n  ── Cross-Model Consistency Verdict ──")
    if cold_results:
        max_delta = max(r["delta"] for r in cold_results)
        if max_delta > 0.30:
            print(f"  ✓ HPS UNIQUELY ADVANTAGEOUS ON VICUNA")
            print(f"    Max Δ = +{max_delta:.3f} at low data — much larger than Llama-3")
            print(f"    The original Vicuna +0.302 finding REPRODUCES with proper Euclidean")
            print(f"    → USENIX-quality contribution: hyperbolic helps in some model regimes")
        elif max_delta > 0.10:
            print(f"  ⚠ MODEST VICUNA ADVANTAGE")
            print(f"    Max Δ = +{max_delta:.3f}")
            print(f"    Hyperbolic shows benefit on Vicuna but smaller than original claim")
            print(f"    → Defensible but not headline-worthy")
        else:
            print(f"  ✗ NO UNIQUE HPS ADVANTAGE ON VICUNA EITHER")
            print(f"    Max Δ = +{max_delta:.3f} (similar to Llama-3)")
            print(f"    The original Vicuna +0.302 was an artifact, not a model property")
            print(f"    → No cross-model differentiator; pivot to compression/study framing")

    # Save
    out = {
        "config": {"layers": layers_used, "kappa_init": KAPPA_INIT, "epochs": EPOCHS},
        "same_dist": {"hps": {"auroc": a_h, "tpr": t_h, "fpr": fpr_h},
                      "c4": {"auroc": a_c, "tpr": t_c, "fpr": fpr_c}},
        "cold_start": cold_results,
        "pgd": pgd_results,
    }
    out_path = "results/cross_model_compare.json"

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
