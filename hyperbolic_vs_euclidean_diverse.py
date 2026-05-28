"""
hyperbolic_vs_euclidean_diverse.py — Re-test HPS vs HPS-Euclidean vs C4
on the diverse-benign cleaned cache.

The original geometric-vs-flat ablations (in experiment1_hyperbolic_methods.py
and the cold-start sweeps) were run on the OLD cache with narrow Alpaca
benign. Now that we have:
  - Diverse benign matched to attack lengths (length-only AUROC=0.318)
  - Train/test contamination fixed (15 -> 0 overlaps)
  - Reversed radial inversion finding (geometry IS meaningful)

we should re-run the comparison to see if conclusions change.

Tests:
  Part A: Same-distribution full-data — HPS vs HPS-Euclidean vs C4
  Part B: Cold-start sweep (N attacks/method = 5, 10, 25, 50, 100, 250)
  Part C: 5-seed stability for the most informative regime

Usage:
  python hyperbolic_vs_euclidean_diverse.py
  python hyperbolic_vs_euclidean_diverse.py \
      --llama_cache results/llama3_activations_cache_diverse_clean.npz
"""

import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

LLAMA_LAYERS = [0, 2, 17, 24, 28, 31]
KAPPA = 0.1
EPOCHS = 50
PROJ_DIM = 64
LR = 1e-3
WEIGHT_DECAY = 1e-5
device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------- Lorentz primitives ----------------------
def lorentz_inner(x, y):
    return -x[..., 0:1] * y[..., 0:1] + (x[..., 1:] * y[..., 1:]).sum(-1, keepdim=True)


def lorentz_distance(x, y, k=1.0):
    inner = lorentz_inner(x, y).squeeze(-1)
    arg = (-k * inner).clamp(min=1.0 + 1e-7)
    return torch.acosh(arg) / np.sqrt(k)


def to_hyperboloid(x_spatial, k=1.0):
    x_0 = torch.sqrt(1.0 / k + (x_spatial ** 2).sum(-1, keepdim=True))
    return torch.cat([x_0, x_spatial], dim=-1)


class LorentzProjection(nn.Module):
    def __init__(self, d_in, d_proj, k_init=0.1, freeze_k=True):
        super().__init__()
        self.proj = nn.Linear(d_in, d_proj, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(1.0 / 8.0))
        self.log_k = nn.Parameter(torch.tensor(float(np.log(k_init))))
        if freeze_k:
            self.log_k.requires_grad = False

    @property
    def k(self):
        return torch.exp(self.log_k).clamp(0.01, 100.0)

    def forward(self, x):
        h_spatial = self.proj(x) * self.scale
        return to_hyperboloid(h_spatial, k=self.k.item())


def contrastive_minibatch(h, y, k, margin=2.0, batch=512, lorentz=True, seed=0):
    n = h.shape[0]
    if n <= batch:
        idx = torch.arange(n, device=h.device)
    else:
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(n, generator=g)[:batch].to(h.device)
    hb = h[idx]
    yb = y[idx]
    if lorentz:
        d = lorentz_distance(hb.unsqueeze(0), hb.unsqueeze(1), k=k)
    else:
        d = torch.cdist(hb, hb)
    same = (yb.unsqueeze(0) == yb.unsqueeze(1)).float()
    diff = 1.0 - same
    triu = torch.triu(torch.ones_like(d), diagonal=1)
    ns = (same * triu).sum().clamp(min=1)
    nd = (diff * triu).sum().clamp(min=1)
    return ((d ** 2 * same * triu).sum() / ns +
            (torch.clamp(margin - d, min=0) ** 2 * diff * triu).sum() / nd) / 2


# ---------------------- Methods ----------------------
def eval_features(f_tr, y_tr, f_be, f_at, seed=42):
    if f_tr.ndim == 1:
        f_tr = f_tr.reshape(-1, 1)
        f_be = f_be.reshape(-1, 1)
        f_at = f_at.reshape(-1, 1)
    sc = StandardScaler()
    Xtr = sc.fit_transform(f_tr)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(Xtr, y_tr)
    n_calib = max(1, len(f_be) // 2)
    s_calib = clf.predict_proba(sc.transform(f_be[:n_calib]))[:, 1]
    s_ben = clf.predict_proba(sc.transform(f_be[n_calib:]))[:, 1]
    s_atk = clf.predict_proba(sc.transform(f_at))[:, 1]
    thr = float(np.quantile(s_calib, 0.95))
    tpr = float((s_atk > thr).mean())
    fpr = float((s_ben > thr).mean())
    auroc = roc_auc_score(np.array([0]*len(s_ben)+[1]*len(s_atk)),
                          np.concatenate([s_ben, s_atk]))
    return {"auroc": float(auroc), "tpr": tpr, "fpr": fpr}


def method_hps(data, seed=42, epochs=EPOCHS):
    n_layers = data["X_tr_ben"].shape[1]
    d_h = data["X_tr_ben"].shape[2]
    X_tr = np.concatenate([data["X_tr_ben"], data["X_tr_atk"]])
    y_tr = np.array([0]*len(data["X_tr_ben"]) + [1]*len(data["X_tr_atk"]))
    torch.manual_seed(seed)
    proj = LorentzProjection(d_h, PROJ_DIM, k_init=KAPPA).to(device)
    opt = optim.Adam([p for p in proj.parameters() if p.requires_grad],
                     lr=LR, weight_decay=WEIGHT_DECAY)
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.long, device=device)
    for ep in range(epochs):
        loss = torch.tensor(0.0, device=device)
        for li in range(n_layers):
            h = proj(Xt[:, li, :])
            loss = loss + contrastive_minibatch(h, yt, k=proj.k.item(),
                                                 batch=512, lorentz=True, seed=ep)
        loss = loss / n_layers
        opt.zero_grad(); loss.backward(); opt.step()
    proj.eval()
    def feats(X):
        out = []
        with torch.no_grad():
            for i in range(len(X)):
                xi = torch.tensor(X[i], dtype=torch.float32, device=device)
                h = proj(xi)
                out.append(float(h[:, 0].mean().cpu()))
        return np.array(out).reshape(-1, 1)
    return eval_features(feats(X_tr), y_tr, feats(data["X_te_ben"]),
                         feats(data["X_te_atk"]), seed=seed)


def method_euclidean_matched(data, seed=42, epochs=EPOCHS):
    n_layers = data["X_tr_ben"].shape[1]
    d_h = data["X_tr_ben"].shape[2]
    X_tr = np.concatenate([data["X_tr_ben"], data["X_tr_atk"]])
    y_tr = np.array([0]*len(data["X_tr_ben"]) + [1]*len(data["X_tr_atk"]))
    torch.manual_seed(seed)
    proj = nn.Linear(d_h, PROJ_DIM, bias=False).to(device)
    nn.init.xavier_uniform_(proj.weight)
    scale_pl = nn.Parameter(torch.ones(n_layers, device=device) / 8.0)
    log_margin = nn.Parameter(torch.tensor(np.log(2.0), device=device))
    opt = optim.Adam(list(proj.parameters()) + [scale_pl, log_margin],
                     lr=LR, weight_decay=WEIGHT_DECAY)
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.long, device=device)
    for ep in range(epochs):
        loss = torch.tensor(0.0, device=device)
        margin = torch.exp(log_margin).clamp(0.5, 5.0)
        for li in range(n_layers):
            h = proj(Xt[:, li, :]) * scale_pl[li]
            loss = loss + contrastive_minibatch(h, yt, k=1.0, margin=margin.item(),
                                                 batch=512, lorentz=False, seed=ep)
        loss = loss / n_layers
        opt.zero_grad(); loss.backward(); opt.step()
    proj.eval()
    def feats(X):
        out = []
        with torch.no_grad():
            for i in range(len(X)):
                xi = torch.tensor(X[i], dtype=torch.float32, device=device)
                pts = []
                for li in range(n_layers):
                    pts.append(proj(xi[li:li+1]) * scale_pl[li])
                pts = torch.cat(pts)
                out.append(float(torch.norm(pts, dim=1).mean().cpu()))
        return np.array(out).reshape(-1, 1)
    return eval_features(feats(X_tr), y_tr, feats(data["X_te_ben"]),
                         feats(data["X_te_atk"]), seed=seed)


def method_c4(data, seed=42):
    f_tr = np.concatenate([data["X_tr_ben"].mean(axis=1),
                           data["X_tr_atk"].mean(axis=1)])
    y_tr = np.array([0]*len(data["X_tr_ben"]) + [1]*len(data["X_tr_atk"]))
    f_be = data["X_te_ben"].mean(axis=1)
    f_at = data["X_te_atk"].mean(axis=1)
    return eval_features(f_tr, y_tr, f_be, f_at, seed=seed)


# ---------------------- Data ----------------------
def load_llama_cache(path, layers):
    cache = np.load(path, allow_pickle=True)
    def to_arr(hs_list):
        return np.array([[hs[l][-1] for l in layers] for hs in hs_list])
    return {
        "X_tr_ben": to_arr(cache["hs_train_ben"].tolist()),
        "X_tr_atk": to_arr(cache["hs_train_atk"].tolist()),
        "X_te_ben": to_arr(cache["hs_test_ben"].tolist()),
        "X_te_atk": to_arr(cache["hs_test_atk"].tolist()),
    }


# ---------------------- Cold-start sweep ----------------------
def cold_start_sweep(data, n_per_method_list=[5, 10, 25, 50, 100, 250], seed=42):
    """Subsample N attacks per method, evaluate HPS / Euc / C4."""
    print("\n  Cold-start sweep (N attacks/method, single seed):")
    print(f"  {'N':<5} | {'HPS':>10} | {'Euclidean':>10} | {'C4':>10} | {'HPS-Euc':>10}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    # We don't have per-method labels here for Llama-3 cache (cache doesn't store them).
    # Approximate by random subsampling on attacks (uniform across method distribution).
    rng = np.random.RandomState(seed)
    n_total_atk = len(data["X_tr_atk"])
    results = []
    for n_per in n_per_method_list:
        n_total = min(n_per * 9, n_total_atk)  # ~9 methods on Llama-3
        atk_idx = rng.permutation(n_total_atk)[:n_total]
        ben_idx = rng.permutation(len(data["X_tr_ben"]))[:n_total]
        sub_data = {
            "X_tr_ben": data["X_tr_ben"][ben_idx],
            "X_tr_atk": data["X_tr_atk"][atk_idx],
            "X_te_ben": data["X_te_ben"],
            "X_te_atk": data["X_te_atk"],
        }
        r_hps = method_hps(sub_data, seed=seed)
        r_euc = method_euclidean_matched(sub_data, seed=seed)
        r_c4 = method_c4(sub_data, seed=seed)
        delta = r_hps["tpr"] - r_euc["tpr"]
        print(f"  {n_total:<5} | {r_hps['tpr']:>10.4f} | {r_euc['tpr']:>10.4f} | "
              f"{r_c4['tpr']:>10.4f} | {delta:>+10.4f}")
        results.append({"n_total": n_total, "n_per": n_per,
                        "hps": r_hps, "euc": r_euc, "c4": r_c4,
                        "hps_minus_euc": delta})
    return results


# ---------------------- Main ----------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llama_cache",
                   default="results/llama3_activations_cache_diverse.npz")
    p.add_argument("--output", default="results/hyperbolic_vs_euclidean_diverse.json")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--cold_start", action="store_true",
                   help="Run cold-start sweep (longer)")
    args = p.parse_args()

    print("\n" + "=" * 70)
    print(" HYPERBOLIC vs EUCLIDEAN — Diverse Cache Re-test")
    print("=" * 70)
    print(f"  Cache: {args.llama_cache}")

    data = load_llama_cache(args.llama_cache, LLAMA_LAYERS)
    print(f"  Train: {len(data['X_tr_ben'])} ben + {len(data['X_tr_atk'])} atk")
    print(f"  Test:  {len(data['X_te_ben'])} ben + {len(data['X_te_atk'])} atk")

    out = {"cache": args.llama_cache, "seeds": args.seeds}

    # Part A: full-data multi-seed
    print("\n" + "-" * 70)
    print(" PART A: Same-distribution, full data, 5 seeds")
    print("-" * 70)
    print(f"\n  {'Method':<22} | {'AUROC mean':>11} | {'TPR mean':>9} | {'AUROC std':>10}")
    print(f"  {'-'*22}-+-{'-'*11}-+-{'-'*9}-+-{'-'*10}")

    out["part_a"] = {}
    for name, fn in [("HPS (Lorentz)", method_hps),
                     ("HPS-Euclidean (matched)", method_euclidean_matched),
                     ("C4 (linear probe)", method_c4)]:
        aurocs = []
        tprs = []
        for s in args.seeds:
            r = fn(data, seed=s)
            aurocs.append(r["auroc"])
            tprs.append(r["tpr"])
        out["part_a"][name] = {
            "aurocs": aurocs, "tprs": tprs,
            "auroc_mean": float(np.mean(aurocs)),
            "auroc_std": float(np.std(aurocs)),
            "tpr_mean": float(np.mean(tprs)),
            "tpr_std": float(np.std(tprs)),
        }
        print(f"  {name:<22} | {np.mean(aurocs):>11.4f} | "
              f"{np.mean(tprs):>9.4f} | {np.std(aurocs):>10.4f}")

    # Part B: cold-start
    if args.cold_start:
        print("\n" + "-" * 70)
        print(" PART B: Cold-start sweep (single seed)")
        print("-" * 70)
        out["part_b"] = cold_start_sweep(data, seed=42)

    # Part C: comparison summary
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    a = out["part_a"]
    print(f"\n  HPS vs HPS-Euclidean (matched parameters):")
    delta_auroc = a["HPS (Lorentz)"]["auroc_mean"] - a["HPS-Euclidean (matched)"]["auroc_mean"]
    delta_tpr = a["HPS (Lorentz)"]["tpr_mean"] - a["HPS-Euclidean (matched)"]["tpr_mean"]
    print(f"    Δ AUROC: {delta_auroc:+.4f}")
    print(f"    Δ TPR:   {delta_tpr:+.4f}")

    print(f"\n  HPS vs C4 (linear probe baseline):")
    delta_auroc_c4 = a["HPS (Lorentz)"]["auroc_mean"] - a["C4 (linear probe)"]["auroc_mean"]
    delta_tpr_c4 = a["HPS (Lorentz)"]["tpr_mean"] - a["C4 (linear probe)"]["tpr_mean"]
    print(f"    Δ AUROC: {delta_auroc_c4:+.4f}")
    print(f"    Δ TPR:   {delta_tpr_c4:+.4f}")

    if abs(delta_tpr) < 0.01 and abs(delta_tpr_c4) < 0.01:
        print(f"\n  -> Geometry doesn't help: HPS = HPS-Euc = C4 within 1% TPR")
        print(f"     Saturated benchmark; method comparison uninformative at this scale.")
        print(f"     Cold-start regime (--cold_start) is more informative.")
    elif delta_tpr > 0.02:
        print(f"\n  -> HPS beats HPS-Euclidean by {delta_tpr:+.3f} TPR.")
        print(f"     Geometric prior provides measurable advantage on diverse benign.")
    else:
        print(f"\n  -> HPS slightly better than Euclidean ({delta_tpr:+.3f}) but small.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved -> {args.output}")


if __name__ == "__main__":
    main()
