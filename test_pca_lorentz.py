"""
Quick test: Raw activations → PCA → Lorentz projection (zero-shot)
Can hyperbolic geometry on richer representations beat RTV's 0.843?
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.covariance import LedoitWolf

from rtv_standalone import load_model, extract_hidden_states, MODEL_LAYERS

LAYER = 16  # peak refusal layer for Vicuna-13B
PCA_DIM = 50
D_PROJ = 8
EPOCHS = 200
FPR_TARGET = 0.05


def lorentz_distance(x, y, k):
    inner = -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)
    arg = torch.clamp(-k * inner, min=1.0 + 1e-7)
    return torch.log(arg + torch.sqrt(arg * arg - 1.0)) / torch.sqrt(k)


def contrastive_loss(pts, labels, k, margin=2.0):
    n = pts.shape[0]
    inner = -pts[:, 0:1] @ pts[:, 0:1].T + pts[:, 1:] @ pts[:, 1:].T
    inner = torch.clamp(inner, max=(-1.0 / k - 1e-6).detach().item())
    dists = torch.log(-k * inner + torch.sqrt((-k * inner)**2 - 1.0 + 1e-7)) / torch.sqrt(k)
    same = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    diff = 1.0 - same
    triu = torch.triu(torch.ones(n, n), diagonal=1)
    same_loss = (dists**2 * same * triu).sum() / (same * triu).sum().clamp(min=1)
    diff_loss = (torch.clamp(margin - dists, min=0)**2 * diff * triu).sum() / (diff * triu).sum().clamp(min=1)
    return (same_loss + diff_loss) / 2.0


class LorentzProj(nn.Module):
    def __init__(self, d_in, d_proj=8):
        super().__init__()
        self.proj = nn.Linear(d_in, d_proj, bias=False)
        self.scale = nn.Parameter(torch.tensor(1.0 / np.sqrt(d_proj)))
        self.log_k = nn.Parameter(torch.tensor(0.0))
        nn.init.xavier_uniform_(self.proj.weight)

    @property
    def k(self):
        return torch.exp(self.log_k).clamp(0.1, 10.0)

    def forward(self, x):
        xp = self.proj(x.float()) * self.scale
        x0 = torch.sqrt(1.0 / self.k + (xp**2).sum(-1, keepdim=True))
        return torch.cat([x0, xp], dim=-1)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lmsys/vicuna-13b-v1.5")
    parser.add_argument("--harmless", required=True)
    parser.add_argument("--harmful", required=True)
    parser.add_argument("--test-attacks", required=True)
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"  Quick Test: Raw Activations → PCA({PCA_DIM}) → Lorentz({D_PROJ})")
    print(f"  Layer: {LAYER}")
    print(f"{'═'*60}\n")

    # Load data
    df_h = pd.read_csv(args.harmless)
    df_harm = pd.read_csv(args.harmful)
    for col in ["prompt", "goal", "text", "instruction", "query"]:
        if col in df_h.columns: harmless = df_h[col].dropna().tolist(); break
    else: harmless = df_h.iloc[:, 0].dropna().tolist()
    for col in ["prompt", "goal", "text", "instruction", "query"]:
        if col in df_harm.columns: harmful = df_harm[col].dropna().tolist(); break
    else: harmful = df_harm.iloc[:, 0].dropna().tolist()

    with open(args.test_attacks) as f:
        data = json.load(f)
    attacks, methods = [], []
    for m, ps in data.items():
        for p in ps:
            attacks.append(p); methods.append(m)

    n_cal = min(len(harmless), len(harmful))
    print(f"  Calibration: {n_cal}+{n_cal}, Attacks: {len(attacks)}")

    # Load model & extract
    model, tokenizer = load_model(args.model)

    print(f"\n  Extracting layer {LAYER} activations...")
    def get_act(prompt):
        hs = extract_hidden_states(model, tokenizer, prompt, [LAYER])
        return hs[LAYER][-1]  # last token, shape (5120,)

    acts_harmless = np.array([get_act(p) for i, p in enumerate(harmless[:n_cal])
                              if not (i % 10 == 9 and print(f"    harmless {i+1}/{n_cal}"))])
    # Redo without the print hack
    print("    Extracting harmless...")
    acts_harmless = []
    for i in range(n_cal):
        acts_harmless.append(get_act(harmless[i]))
        if (i+1) % 10 == 0: print(f"      {i+1}/{n_cal}")
    acts_harmless = np.array(acts_harmless)

    print("    Extracting harmful...")
    acts_harmful = []
    for i in range(n_cal):
        acts_harmful.append(get_act(harmful[i]))
        if (i+1) % 10 == 0: print(f"      {i+1}/{n_cal}")
    acts_harmful = np.array(acts_harmful)

    print("    Extracting attacks...")
    acts_attacks = []
    for i in range(len(attacks)):
        acts_attacks.append(get_act(attacks[i]))
        if (i+1) % 50 == 0: print(f"      {i+1}/{len(attacks)}")
    acts_attacks = np.array(acts_attacks)

    # PCA
    print(f"\n  PCA: {acts_harmless.shape[1]} → {PCA_DIM}")
    X_cal = np.vstack([acts_harmless, acts_harmful])
    pca = PCA(n_components=PCA_DIM, random_state=42)
    pca.fit(X_cal)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    X_harmless = pca.transform(acts_harmless)
    X_harmful = pca.transform(acts_harmful)
    X_attacks = pca.transform(acts_attacks)

    # Train Lorentz
    print(f"\n  Training Lorentz projection ({PCA_DIM} → {D_PROJ})...")
    X_t = torch.tensor(np.vstack([X_harmless, X_harmful]), dtype=torch.float32)
    y_t = torch.tensor([0]*len(X_harmless) + [1]*len(X_harmful), dtype=torch.long)

    torch.manual_seed(42)
    proj = LorentzProj(PCA_DIM, D_PROJ)
    opt = optim.Adam(proj.parameters(), lr=5e-3, weight_decay=1e-5)
    for ep in range(EPOCHS):
        pts = proj(X_t)
        loss = contrastive_loss(pts, y_t, proj.k)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1) % 50 == 0:
            print(f"    Epoch {ep+1}/{EPOCHS} loss={loss.item():.4f} k={proj.k.item():.3f}")
    proj.eval()

    # Score
    print(f"\n  Scoring...")
    with torch.no_grad():
        p_harmless = proj(torch.tensor(X_harmless, dtype=torch.float32))
        p_harmful = proj(torch.tensor(X_harmful, dtype=torch.float32))
        p_attacks = proj(torch.tensor(X_attacks, dtype=torch.float32))
        c_pos = p_harmless.mean(0, keepdim=True)
        c_neg = p_harmful.mean(0, keepdim=True)
        k = proj.k

    def score(pts):
        s = []
        for i in range(pts.shape[0]):
            d1 = lorentz_distance(pts[i:i+1], c_pos, k).item()
            d2 = lorentz_distance(pts[i:i+1], c_neg, k).item()
            s.append(min(d1, d2))
        return np.array(s)

    s_harmless = score(p_harmless)
    s_harmful = score(p_harmful)
    s_attacks = score(p_attacks)

    cal_s = np.concatenate([s_harmless, s_harmful])
    threshold = float(np.quantile(cal_s, 1.0 - FPR_TARGET))

    tpr = float((s_attacks > threshold).mean())
    auroc = roc_auc_score(
        np.array([0]*len(s_harmless) + [1]*len(s_attacks)),
        np.concatenate([s_harmless, s_attacks])
    )

    # Also compute raw Mahalanobis baseline on same PCA features
    lw_pos = LedoitWolf().fit(X_harmless)
    lw_neg = LedoitWolf().fit(X_harmful)
    def mahal(x):
        d1 = np.sqrt(max(0, (x - lw_pos.location_) @ lw_pos.precision_ @ (x - lw_pos.location_)))
        d2 = np.sqrt(max(0, (x - lw_neg.location_) @ lw_neg.precision_ @ (x - lw_neg.location_)))
        return min(d1, d2)

    ms_harmless = np.array([mahal(x) for x in X_harmless])
    ms_attacks = np.array([mahal(x) for x in X_attacks])
    ms_harmful = np.array([mahal(x) for x in X_harmful])
    ms_cal = np.concatenate([ms_harmless, ms_harmful])
    ms_thr = float(np.quantile(ms_cal, 1.0 - FPR_TARGET))
    ms_tpr = float((ms_attacks > ms_thr).mean())
    ms_auroc = roc_auc_score(
        np.array([0]*len(ms_harmless) + [1]*len(ms_attacks)),
        np.concatenate([ms_harmless, ms_attacks])
    )

    # Results
    print(f"\n{'─'*60}")
    print(f"  RESULTS (Layer {LAYER}, PCA→{PCA_DIM})")
    print(f"{'─'*60}")
    print(f"  {'Method':<30} | {'AUROC':>6} | {'TPR@5%':>7}")
    print(f"  {'─'*30}─┼─{'─'*6}─┼─{'─'*7}")
    print(f"  {'Mahalanobis (PCA features)':<30} | {ms_auroc:>6.3f} | {ms_tpr:>7.3f}")
    print(f"  {'Lorentz (geodesic)':<30} | {auroc:>6.3f} | {tpr:>7.3f}")
    print(f"  {'RTV baseline (fingerprint)':<30} | {'0.843':>6} | {'0.566':>7}")

    # Per method
    print(f"\n  Per-attack (Lorentz):")
    for m in sorted(set(methods)):
        idx = [i for i, x in enumerate(methods) if x == m]
        mt = float((s_attacks[idx] > threshold).mean())
        print(f"    {m:<30}: TPR={mt:.3f} (n={len(idx)})")

    print(f"\n  Per-attack (Mahalanobis on PCA):")
    for m in sorted(set(methods)):
        idx = [i for i, x in enumerate(methods) if x == m]
        mt = float((ms_attacks[idx] > ms_thr).mean())
        print(f"    {m:<30}: TPR={mt:.3f} (n={len(idx)})")

    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    main()
