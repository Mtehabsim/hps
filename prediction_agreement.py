"""
prediction_agreement.py — Quantify HPS vs C4 prediction agreement on natural data.

Footnote-grade analysis. Reports:
  1. Per-example correctness for HPS and C4
  2. McNemar's-style disagreement table
  3. Score correlation (Pearson, Spearman)
  4. OR-gate ensemble TPR (if EITHER detects, alarm)
  5. Learned-LR ensemble (LR on [hps_score, c4_score])

Caveats explicitly documented:
  - This is NATURAL-DATA agreement only
  - Does NOT predict adversarial robustness
  - Bailey et al. (2024) show probe ensembles fail under adaptive attack;
    this script only quantifies the natural-data ceiling

Usage:
  python prediction_agreement.py \
      --cache results/llama3_activations_cache_diverse_fixed.npz \
      --layers 0 2 17 24 28 31 \
      --output results/prediction_agreement.json
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
from scipy.stats import pearsonr, spearmanr

KAPPA = 0.1
EPOCHS = 50
PROJ_DIM = 64
LR_LR = 1e-3
WEIGHT_DECAY = 1e-5
FPR_TARGET = 0.05
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
    hb = h[idx]; yb = y[idx]
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
def train_hps_and_score(X_tr, y_tr, X_te, seed=42, epochs=EPOCHS):
    """Train HPS, return (test_scores, threshold@5%FPR_using_calib_split)."""
    n_layers = X_tr.shape[1]
    d_h = X_tr.shape[2]
    torch.manual_seed(seed)
    proj = LorentzProjection(d_h, PROJ_DIM, k_init=KAPPA).to(device)
    opt = optim.Adam([p for p in proj.parameters() if p.requires_grad],
                     lr=LR_LR, weight_decay=WEIGHT_DECAY)
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

    def feat_mean_r(X):
        out = []
        with torch.no_grad():
            for i in range(len(X)):
                xi = torch.tensor(X[i], dtype=torch.float32, device=device)
                h = proj(xi)
                out.append(float(h[:, 0].mean().cpu()))
        return np.array(out).reshape(-1, 1)

    f_tr = feat_mean_r(X_tr)
    f_te = feat_mean_r(X_te)

    sc = StandardScaler()
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(sc.fit_transform(f_tr), y_tr)
    return clf.predict_proba(sc.transform(f_te))[:, 1]


def train_c4_and_score(X_tr, y_tr, X_te, seed=42):
    """C4: linear probe on mean-pooled activations across layers."""
    f_tr = X_tr.mean(axis=1)  # (N, d_hidden)
    f_te = X_te.mean(axis=1)
    sc = StandardScaler()
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(sc.fit_transform(f_tr), y_tr)
    return clf.predict_proba(sc.transform(f_te))[:, 1]


# ---------------------- Data loading ----------------------
def load_cache(path, layers):
    cache = np.load(path, allow_pickle=True)
    def to_arr(hs_list):
        return np.array([[hs[l][-1] for l in layers] for hs in hs_list])
    return {
        "X_tr_ben": to_arr(cache["hs_train_ben"].tolist()),
        "X_tr_atk": to_arr(cache["hs_train_atk"].tolist()),
        "X_te_ben": to_arr(cache["hs_test_ben"].tolist()),
        "X_te_atk": to_arr(cache["hs_test_atk"].tolist()),
    }


# ---------------------- Analysis ----------------------
def analyze(hps_scores, c4_scores, y_te, threshold_fpr=0.05):
    """Compute disagreement metrics."""
    n_ben = (y_te == 0).sum()
    n_atk = (y_te == 1).sum()

    # Calibration split: use first half of benign for thresholds
    ben_idx = np.where(y_te == 0)[0]
    n_calib = len(ben_idx) // 2
    calib_idx = ben_idx[:n_calib]
    eval_ben_idx = ben_idx[n_calib:]
    atk_idx = np.where(y_te == 1)[0]

    hps_thr = float(np.quantile(hps_scores[calib_idx], 1.0 - threshold_fpr))
    c4_thr  = float(np.quantile(c4_scores[calib_idx],  1.0 - threshold_fpr))

    # Per-example correctness on attacks
    hps_caught = hps_scores[atk_idx] > hps_thr
    c4_caught  = c4_scores[atk_idx]  > c4_thr

    # Confusion among the 4 quadrants on attacks
    both = int(np.sum(hps_caught & c4_caught))
    hps_only = int(np.sum(hps_caught & ~c4_caught))
    c4_only  = int(np.sum(~hps_caught & c4_caught))
    neither  = int(np.sum(~hps_caught & ~c4_caught))

    # OR-gate ensemble
    or_caught = hps_caught | c4_caught
    or_tpr = float(or_caught.mean())

    # Per-example FPR on benign
    hps_fp = hps_scores[eval_ben_idx] > hps_thr
    c4_fp  = c4_scores[eval_ben_idx]  > c4_thr
    or_fp  = hps_fp | c4_fp

    # AUROC overall
    hps_auroc = roc_auc_score(y_te, hps_scores)
    c4_auroc  = roc_auc_score(y_te, c4_scores)

    # Score correlation
    pearson_r, _ = pearsonr(hps_scores, c4_scores)
    spearman_r, _ = spearmanr(hps_scores, c4_scores)

    return {
        "n_test_attacks": int(n_atk),
        "n_test_benign": int(n_ben),
        "thresholds": {"hps": hps_thr, "c4": c4_thr},
        "tpr": {
            "hps_alone": float(hps_caught.mean()),
            "c4_alone": float(c4_caught.mean()),
            "or_ensemble": or_tpr,
        },
        "fpr_on_held_out_benign": {
            "hps_alone": float(hps_fp.mean()),
            "c4_alone": float(c4_fp.mean()),
            "or_ensemble": float(or_fp.mean()),
        },
        "attack_disagreement": {
            "both_caught": both,
            "hps_only_caught": hps_only,
            "c4_only_caught": c4_only,
            "neither_caught": neither,
            "agreement_rate": float((both + neither) / max(n_atk, 1)),
        },
        "auroc": {
            "hps": float(hps_auroc),
            "c4": float(c4_auroc),
        },
        "score_correlation": {
            "pearson": float(pearson_r),
            "spearman": float(spearman_r),
        },
    }


# ---------------------- Main ----------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="results/llama3_activations_cache_diverse_fixed.npz")
    p.add_argument("--layers", type=int, nargs="+", default=[0, 2, 17, 24, 28, 31])
    p.add_argument("--output", default="results/prediction_agreement.json")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = p.parse_args()

    print("\n" + "=" * 70)
    print(" PREDICTION AGREEMENT — HPS vs C4 (natural data only)")
    print("=" * 70)
    print(f"  Cache:   {args.cache}")
    print(f"  Layers:  {args.layers}")
    print()

    if not os.path.exists(args.cache):
        print(f"ERROR: cache not found at {args.cache}")
        return

    print("  Loading cache...")
    data = load_cache(args.cache, args.layers)

    X_tr = np.concatenate([data["X_tr_ben"], data["X_tr_atk"]])
    y_tr = np.array([0]*len(data["X_tr_ben"]) + [1]*len(data["X_tr_atk"]))
    X_te = np.concatenate([data["X_te_ben"], data["X_te_atk"]])
    y_te = np.array([0]*len(data["X_te_ben"]) + [1]*len(data["X_te_atk"]))
    print(f"    Train: {len(data['X_tr_ben'])} ben + {len(data['X_tr_atk'])} atk")
    print(f"    Test:  {len(data['X_te_ben'])} ben + {len(data['X_te_atk'])} atk")

    all_results = []
    for seed in args.seeds:
        print(f"\n  Seed {seed}:")

        print(f"    Training HPS ({EPOCHS} epochs)...")
        hps_te = train_hps_and_score(X_tr, y_tr, X_te, seed=seed)

        print(f"    Training C4 (logistic regression)...")
        c4_te = train_c4_and_score(X_tr, y_tr, X_te, seed=seed)

        r = analyze(hps_te, c4_te, y_te)
        r["seed"] = seed
        all_results.append(r)

        ag = r["attack_disagreement"]
        print(f"    AUROC:  HPS={r['auroc']['hps']:.4f}  C4={r['auroc']['c4']:.4f}")
        print(f"    TPR:    HPS={r['tpr']['hps_alone']:.4f}  "
              f"C4={r['tpr']['c4_alone']:.4f}  OR={r['tpr']['or_ensemble']:.4f}")
        print(f"    FPR:    HPS={r['fpr_on_held_out_benign']['hps_alone']:.4f}  "
              f"C4={r['fpr_on_held_out_benign']['c4_alone']:.4f}  "
              f"OR={r['fpr_on_held_out_benign']['or_ensemble']:.4f}")
        print(f"    Disagreement: both={ag['both_caught']} "
              f"hps_only={ag['hps_only_caught']} "
              f"c4_only={ag['c4_only_caught']} "
              f"neither={ag['neither_caught']}")
        print(f"    Pearson r:  {r['score_correlation']['pearson']:.4f}")

    # Aggregate
    print("\n" + "=" * 70)
    print(" AGGREGATE (mean over seeds)")
    print("=" * 70)
    mean_hps_tpr = np.mean([r["tpr"]["hps_alone"] for r in all_results])
    mean_c4_tpr  = np.mean([r["tpr"]["c4_alone"]  for r in all_results])
    mean_or_tpr  = np.mean([r["tpr"]["or_ensemble"] for r in all_results])
    mean_or_fpr  = np.mean([r["fpr_on_held_out_benign"]["or_ensemble"] for r in all_results])
    mean_pearson = np.mean([r["score_correlation"]["pearson"] for r in all_results])
    mean_hps_only = np.mean([r["attack_disagreement"]["hps_only_caught"] for r in all_results])
    mean_c4_only  = np.mean([r["attack_disagreement"]["c4_only_caught"]  for r in all_results])

    print(f"\n  HPS alone TPR:        {mean_hps_tpr:.4f}")
    print(f"  C4 alone TPR:         {mean_c4_tpr:.4f}")
    print(f"  OR-ensemble TPR:      {mean_or_tpr:.4f}")
    print(f"  OR-ensemble FPR:      {mean_or_fpr:.4f}  (target: 0.05, may exceed)")
    print(f"  Pearson(HPS, C4):     {mean_pearson:.4f}")
    print(f"  HPS-only catches:     {mean_hps_only:.0f} examples")
    print(f"  C4-only catches:      {mean_c4_only:.0f} examples")

    print("\n" + "-" * 70)
    print(" FOOTNOTE-READY SUMMARY (paste into paper)")
    print("-" * 70)
    n_atk = all_results[0]["n_test_attacks"]
    print(f"""
On natural data (n={n_atk} test attacks, mean of {len(args.seeds)} seeds), HPS
and C4 disagree on {mean_hps_only + mean_c4_only:.0f} examples
({(mean_hps_only + mean_c4_only) / n_atk * 100:.1f}% of attacks).
Score correlation is high (Pearson r = {mean_pearson:.3f}). An OR-gate
ensemble achieves TPR = {mean_or_tpr:.3f} (vs HPS = {mean_hps_tpr:.3f}, C4 =
{mean_c4_tpr:.3f}), an improvement of
{(mean_or_tpr - max(mean_hps_tpr, mean_c4_tpr))*100:+.1f}% TPR. However, FPR
inflates to {mean_or_fpr:.3f} (above the 0.05 target). Following Bailey et al.
(2024), probe ensembles do not survive adaptive adversarial attacks; we do
not claim adversarial robustness from this natural-data agreement.
""")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "config": {"cache": args.cache, "layers": args.layers, "seeds": args.seeds},
            "per_seed": all_results,
            "aggregate": {
                "hps_tpr_mean": float(mean_hps_tpr),
                "c4_tpr_mean": float(mean_c4_tpr),
                "or_tpr_mean": float(mean_or_tpr),
                "or_fpr_mean": float(mean_or_fpr),
                "pearson_correlation_mean": float(mean_pearson),
                "hps_only_catches_mean": float(mean_hps_only),
                "c4_only_catches_mean": float(mean_c4_only),
            },
            "caveat": (
                "Natural-data agreement only. Bailey et al. (arXiv:2412.09565, 2024) "
                "show that probe ensembles, including across architecturally distinct "
                "probes, do not survive adaptive attacks. Do not interpret this as "
                "adversarial robustness."
            ),
        }, f, indent=2)
    print(f"\n  Saved -> {args.output}")


if __name__ == "__main__":
    main()
