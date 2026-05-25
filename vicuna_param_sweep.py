"""
Quick Vicuna hyperparameter diagnostic — can HPS be rescued?

Tests whether HPS's catastrophic Vicuna performance is due to:
  - Wrong layer selection
  - Wrong κ initialization
  - Wrong epoch count

Sweeps:
  - Layers: spread / Fisher-discovered / late-only / RTV-equivalent
  - κ:      0.1, 0.5, 1.0, 2.0
  - Epochs: 25, 50, 100

Compares to C4 (linear probe) baseline on the same data.

Decision rule:
  - If ANY config gives HPS ≥ C4 on Vicuna cold-start (N=25): hyperparameter issue
  - If NO config rescues HPS: result is robust, paper needs Framing B

Usage:
  python vicuna_param_sweep.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from experiment7 import LorentzProjection, contrastive_loss

FPR_TARGET = 0.05
device = "cuda" if torch.cuda.is_available() else "cpu"
VICUNA_CACHE = "results/vicuna_activations_cache.npz"


def train_hps(X_train, y_train, k_init=0.1, epochs=50, freeze_kappa=True, seed=42):
    n_layers = X_train.shape[1]
    d_hidden = X_train.shape[2]
    torch.manual_seed(seed); np.random.seed(seed)
    proj = LorentzProjection(d_hidden, 64, k_init, n_layers=n_layers).to(device)
    if freeze_kappa:
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
    return X.mean(axis=1)


def evaluate(feats_train, y_train, feats_te_ben, feats_te_atk, seed=42):
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
    thr = float(np.quantile(s_calib, 1.0 - FPR_TARGET))
    tpr = float((s_atk > thr).mean())
    auroc = roc_auc_score(np.array([0]*len(s_ben) + [1]*len(s_atk)),
                          np.concatenate([s_ben, s_atk])) if len(s_ben) > 0 else float('nan')
    return auroc, tpr


def cross_attack_eval(X_all_atk, all_methods, methods_unique, X_all_ben,
                      n_per, layers_config, k_init, epochs, freeze_kappa, c4=False):
    """Leave-one-out cross-attack with given config."""
    # Re-extract from full cache using only specified layers
    hs_by_method = {m: [] for m in methods_unique}
    for act, m in zip(X_all_atk, all_methods):
        hs_by_method[m].append(act)
    for m in methods_unique:
        hs_by_method[m] = np.array(hs_by_method[m])

    ben_split = int(0.8 * len(X_all_ben))
    cv_ben_tr = X_all_ben[:ben_split]
    cv_ben_te = X_all_ben[ben_split:]

    tprs = []
    for held_out in methods_unique:
        sub_atk = []
        for m in methods_unique:
            if m != held_out:
                avail = hs_by_method[m]
                take = min(n_per, len(avail))
                sub_atk.append(avail[:take])
        if not sub_atk:
            continue
        train_atk = np.concatenate(sub_atk)
        test_atk = hs_by_method[held_out]
        if len(test_atk) < 5:
            continue

        # Apply layer config: select subset of cached layers
        # X has all 6 layers from cache; layers_config is indices into [0..5]
        X_tr_sub = np.concatenate([cv_ben_tr[:, layers_config], train_atk[:, layers_config]])
        y_tr = np.array([0]*len(cv_ben_tr) + [1]*len(train_atk))
        X_te_ben_sub = cv_ben_te[:, layers_config]
        X_te_atk_sub = test_atk[:, layers_config]

        if c4:
            f_tr = feat_c4(X_tr_sub)
            f_be = feat_c4(X_te_ben_sub)
            f_at = feat_c4(X_te_atk_sub)
        else:
            proj = train_hps(X_tr_sub, y_tr, k_init=k_init, epochs=epochs,
                             freeze_kappa=freeze_kappa)
            f_tr = feat_hps_mean_r(X_tr_sub, proj)
            f_be = feat_hps_mean_r(X_te_ben_sub, proj)
            f_at = feat_hps_mean_r(X_te_atk_sub, proj)

        _, t = evaluate(f_tr, y_tr, f_be, f_at)
        tprs.append(t)
    return float(np.mean(tprs)) if tprs else 0.0


def main():
    print(f"\n{'═'*60}")
    print(f"  VICUNA HYPERPARAMETER SWEEP")
    print(f"  Can HPS be rescued on Vicuna?")
    print(f"{'═'*60}\n")

    if not os.path.exists(VICUNA_CACHE):
        print(f"ERROR: {VICUNA_CACHE} not found. Run cross_model_compare.py --extract first.")
        return

    cache = np.load(VICUNA_CACHE, allow_pickle=True)
    X_ben = cache["X_benign"]    # (520, 6, 5120)
    X_atk = cache["X_attack"]    # (316, 6, 5120)
    attack_methods = cache["attack_methods"].tolist()
    cached_layers = cache["layers"].tolist()
    print(f"  Vicuna cache: {len(X_ben)} benign, {len(X_atk)} attacks")
    print(f"  Cached layers: {cached_layers}")
    print(f"  Attack methods: {sorted(set(attack_methods))}")

    methods_unique = sorted(set(attack_methods))

    # Layer configs (indices into cached_layers)
    # cached_layers = [0, 2, 22, 31, 35, 39] → indices [0, 1, 2, 3, 4, 5]
    LAYER_CONFIGS = {
        "spread (all 6)":     [0, 1, 2, 3, 4, 5],   # all cached
        "shallow (0,2,22)":   [0, 1, 2],
        "late (31,35,39)":    [3, 4, 5],
        "deepest (35,39)":    [4, 5],
        "minimal (0,39)":     [0, 5],
    }

    # ══════════════════════════════════════════════════════════════
    #  Step 1: C4 baseline (no hyperparameters, just LR)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  REFERENCE: C4 (linear probe) on Vicuna at N=25 cold-start")
    print(f"{'─'*60}\n")

    # Use all 6 layers for C4 baseline
    X_all_ben = X_ben
    X_all_atk = X_atk
    c4_tpr = cross_attack_eval(X_all_atk, attack_methods, methods_unique,
                                X_all_ben, 25, [0,1,2,3,4,5], 0.1, 50, True, c4=True)
    print(f"  C4 cold-start TPR (N=25): {c4_tpr:.3f}")
    print(f"  → This is the target for HPS to match\n")

    # ══════════════════════════════════════════════════════════════
    #  Step 2: HPS hyperparameter sweep
    # ══════════════════════════════════════════════════════════════
    print(f"{'─'*60}")
    print(f"  HPS HYPERPARAMETER SWEEP (cold-start cross-attack, N=25)")
    print(f"{'─'*60}\n")

    print(f"  {'Layers':<22} | {'κ':>5} | {'Frz?':>4} | {'Epoch':>5} | {'HPS TPR':>7} | {'Δ vs C4':>8}")
    print(f"  {'─'*22}─┼─{'─'*5}─┼─{'─'*4}─┼─{'─'*5}─┼─{'─'*7}─┼─{'─'*8}")

    sweep_results = []
    best_config = None
    best_tpr = 0.0

    # Reasonable subset of configs (full grid would be 5×4×3×2 = 120 configs)
    configs = []
    for layer_name, layer_idx in LAYER_CONFIGS.items():
        for k_init in [0.1, 1.0]:                       # Try low and high curvature
            for freeze in [True, False]:                 # Frozen vs learnable
                for epochs in [50]:                      # Skip epoch sweep first
                    configs.append((layer_name, layer_idx, k_init, freeze, epochs))

    # Add a few full-epoch sweeps for the most promising layer config
    for k_init in [0.1, 0.5, 1.0, 2.0]:
        for epochs in [25, 100]:
            configs.append(("spread (all 6)", [0,1,2,3,4,5], k_init, True, epochs))

    for layer_name, layer_idx, k_init, freeze, epochs in configs:
        try:
            tpr = cross_attack_eval(X_all_atk, attack_methods, methods_unique,
                                     X_all_ben, 25, layer_idx, k_init, epochs, freeze)
            delta = tpr - c4_tpr
            sweep_results.append({
                "layers": layer_name, "k_init": k_init, "frozen": freeze,
                "epochs": epochs, "hps_tpr": tpr, "delta_vs_c4": delta,
            })
            print(f"  {layer_name:<22} | {k_init:>5.1f} | {str(freeze):>4} | {epochs:>5} | {tpr:>7.3f} | {delta:>+8.3f}")
            if tpr > best_tpr:
                best_tpr = tpr
                best_config = (layer_name, layer_idx, k_init, freeze, epochs)
        except Exception as e:
            print(f"  {layer_name:<22} | {k_init:>5.1f} | {str(freeze):>4} | {epochs:>5} | (failed: {type(e).__name__})")

    # ══════════════════════════════════════════════════════════════
    #  Step 3: Best config diagnostic at multiple N
    # ══════════════════════════════════════════════════════════════
    if best_config:
        print(f"\n{'─'*60}")
        print(f"  Best HPS config: layers={best_config[0]}, κ={best_config[2]}, frozen={best_config[3]}, epochs={best_config[4]}")
        print(f"  Verifying across N values:")
        print(f"{'─'*60}\n")

        layer_name, layer_idx, k_init, freeze, epochs = best_config
        print(f"  {'N/method':<9} | {'HPS TPR':>7} | {'C4 TPR':>7} | {'Δ':>7}")
        print(f"  {'─'*9}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}")

        n_results = []
        for n_per in [5, 10, 25, 50]:
            hps_t = cross_attack_eval(X_all_atk, attack_methods, methods_unique,
                                       X_all_ben, n_per, layer_idx, k_init, epochs, freeze)
            c4_t = cross_attack_eval(X_all_atk, attack_methods, methods_unique,
                                      X_all_ben, n_per, layer_idx, k_init, epochs, freeze, c4=True)
            delta = hps_t - c4_t
            n_results.append({"n_per": n_per, "hps": hps_t, "c4": c4_t, "delta": delta})
            print(f"  {n_per:<9} | {hps_t:>7.3f} | {c4_t:>7.3f} | {delta:>+7.3f}")

    # ══════════════════════════════════════════════════════════════
    #  Verdict
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  VERDICT")
    print(f"{'═'*60}\n")

    print(f"  C4 cold-start TPR (N=25): {c4_tpr:.3f}")
    print(f"  Best HPS cold-start TPR:  {best_tpr:.3f}")
    print(f"  Gap (HPS - C4):           {best_tpr - c4_tpr:+.3f}")

    if best_tpr >= c4_tpr - 0.02:
        print(f"\n  ✓ HPS RESCUED — the previous failure was a hyperparameter issue")
        print(f"    Best config: {best_config}")
        print(f"    Hyperbolic story salvageable, but requires careful tuning per model")
    elif best_tpr >= c4_tpr - 0.10:
        print(f"\n  ⚠ HPS PARTIALLY RECOVERED — still below C4 by {c4_tpr - best_tpr:.2f}")
        print(f"    Hyperparameter tuning helps but doesn't close the gap")
    else:
        print(f"\n  ✗ HPS CANNOT BE RESCUED — best config still {c4_tpr - best_tpr:.2f} below C4")
        print(f"    The negative result is robust to hyperparameter choice")
        print(f"    → Pivot to Framing B (linear probes are sufficient)")

    out = {
        "c4_baseline_tpr": c4_tpr,
        "best_hps_tpr": best_tpr,
        "best_config": list(best_config) if best_config else None,
        "all_sweep_results": sweep_results,
        "best_config_n_sweep": n_results if best_config else None,
    }
    out_path = "results/vicuna_param_sweep.json"

    def _np_default(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, list): return o
        raise TypeError(f"Type {type(o)} not serializable")

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=_np_default)
    print(f"\n  Saved → {out_path}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
