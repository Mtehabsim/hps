"""
HPS-RTV: Hyperbolic RTV — Zero-shot jailbreak detection using refusal-direction
fingerprints projected into hyperbolic space.

Same setup as RTV (no attack examples needed), but replaces Euclidean Mahalanobis
with Lorentz projection + geodesic distance.

Hypothesis: Hyperbolic geometry's exponential volume growth may better separate
jailbreak fingerprints from legitimate (harmful/harmless) fingerprints, especially
for attacks that land near the harmful cluster boundary.

Usage:
  python hps_rtv_inspired.py \
    --model lmsys/vicuna-13b-v1.5 \
    --harmless JBShield/data/harmless_calibration.csv \
    --harmful JBShield/data/harmful_calibration.csv \
    --test-attacks results/validated_attacks_categorized.json
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from rtv_standalone import (
    load_model, extract_hidden_states, compute_refusal_directions,
    compute_fingerprint, MODEL_LAYERS, TOKEN_POSITIONS, FPR_TARGET
)


# ═══════════════════════════════════════════════════════════════════════════
#  Hyperbolic Projection (Lorentz model)
# ═══════════════════════════════════════════════════════════════════════════

class LorentzProjection(nn.Module):
    """Projects fingerprints onto the Lorentz hyperboloid."""
    def __init__(self, d_in, d_proj=8, k=1.0):
        super().__init__()
        self.proj = nn.Linear(d_in, d_proj, bias=False)
        self.scale = nn.Parameter(torch.tensor(1.0 / np.sqrt(d_proj)))
        self.log_k = nn.Parameter(torch.tensor(np.log(k)))
        nn.init.xavier_uniform_(self.proj.weight)

    @property
    def k(self):
        return torch.exp(self.log_k).clamp(min=0.1, max=10.0)

    def forward(self, x):
        x = x.float()
        x_proj = self.proj(x) * self.scale
        norm_sq = (x_proj ** 2).sum(dim=-1, keepdim=True)
        x0 = torch.sqrt(1.0 / self.k + norm_sq)
        return torch.cat([x0, x_proj], dim=-1)


def lorentz_distance(x, y, k):
    """Geodesic distance on the Lorentz hyperboloid."""
    inner = -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)
    arg = -k * inner
    arg = torch.clamp(arg, min=1.0 + 1e-7)
    return torch.log(arg + torch.sqrt(arg * arg - 1.0)) / torch.sqrt(k)


def contrastive_loss(pts, labels, k, margin=2.0):
    """Contrastive loss in hyperbolic space."""
    n = pts.shape[0]
    inner = -pts[:, 0:1] @ pts[:, 0:1].T + pts[:, 1:] @ pts[:, 1:].T
    clamp_val = (-1.0 / k - 1e-6).detach().item()
    inner = torch.clamp(inner, max=clamp_val)
    dists = torch.log(-k * inner + torch.sqrt((-k * inner)**2 - 1.0 + 1e-7)) / torch.sqrt(k)

    same = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    diff = 1.0 - same
    triu = torch.triu(torch.ones(n, n, device=pts.device), diagonal=1)

    same_loss = (dists ** 2 * same * triu).sum() / (same * triu).sum().clamp(min=1)
    diff_loss = (torch.clamp(margin - dists, min=0) ** 2 * diff * triu).sum() / (diff * triu).sum().clamp(min=1)
    return (same_loss + diff_loss) / 2.0


def lorentz_log_map(mu, x, k):
    """Log map: project point x on hyperboloid to tangent space at mu.
    Returns tangent vector in R^(d_proj) (drops the time component constraint).
    mu: (1, d+1), x: (N, d+1)
    """
    # Lorentz inner product <mu, x>_L
    inner = -mu[..., 0:1] * x[..., 0:1] + (mu[..., 1:] * x[..., 1:]).sum(dim=-1, keepdim=True)
    # Distance
    arg = torch.clamp(-k * inner, min=1.0 + 1e-7)
    dist = torch.log(arg + torch.sqrt(arg * arg - 1.0)) / torch.sqrt(k)
    # Direction in ambient space: normalize (x + k*<mu,x>_L * mu)
    direction = x + k * inner * mu
    # Remove time component for tangent vector (project to spatial)
    direction_spatial = direction[..., 1:]
    dir_norm = torch.norm(direction_spatial, dim=-1, keepdim=True).clamp(min=1e-8)
    # Tangent vector = dist * normalized_direction
    tangent = dist * direction_spatial / dir_norm
    return tangent


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="HPS-RTV: Hyperbolic zero-shot jailbreak detection")
    parser.add_argument("--model", default="lmsys/vicuna-13b-v1.5")
    parser.add_argument("--harmless", required=True)
    parser.add_argument("--harmful", required=True)
    parser.add_argument("--test-attacks", required=True)
    parser.add_argument("--n-cal", type=int, default=100)
    parser.add_argument("--d-proj", type=int, default=8, help="Hyperbolic projection dim")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--output", default="results/hps_rtv_results.json")
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"  HPS-RTV: Hyperbolic Zero-Shot Jailbreak Detection")
    print(f"  Model: {args.model}")
    print(f"{'═'*60}\n")

    # ── Layers ──
    layers = MODEL_LAYERS.get(args.model, [12, 16, 26])
    print(f"[HPS-RTV] Layers: {layers}")

    # ── Load data ──
    df_harmless = pd.read_csv(args.harmless)
    df_harmful = pd.read_csv(args.harmful)
    for col in ["prompt", "goal", "text", "instruction", "query"]:
        if col in df_harmless.columns:
            harmless_prompts = df_harmless[col].dropna().tolist(); break
    else:
        harmless_prompts = df_harmless.iloc[:, 0].dropna().tolist()
    for col in ["prompt", "goal", "text", "instruction", "query"]:
        if col in df_harmful.columns:
            harmful_prompts = df_harmful[col].dropna().tolist(); break
    else:
        harmful_prompts = df_harmful.iloc[:, 0].dropna().tolist()

    n_cal = min(args.n_cal, len(harmless_prompts), len(harmful_prompts))
    print(f"[HPS-RTV] Calibration: {n_cal} harmless + {n_cal} harmful")

    # ── Load attacks ──
    with open(args.test_attacks) as f:
        data = json.load(f)
    attack_prompts, attack_methods = [], []
    for method, prompts in data.items():
        for p in prompts:
            attack_prompts.append(p)
            attack_methods.append(method)
    print(f"[HPS-RTV] Test attacks: {len(attack_prompts)}")

    # ── Load model ──
    model, tokenizer = load_model(args.model)

    # ── Compute refusal directions ──
    refusal_dirs = compute_refusal_directions(
        model, tokenizer, harmful_prompts[:n_cal], harmless_prompts[:n_cal], layers, n_cal
    )

    # ── Extract fingerprints ──
    print(f"\n[HPS-RTV] Extracting fingerprints...")
    fps_harmless = []
    for i in range(n_cal):
        hs = extract_hidden_states(model, tokenizer, harmless_prompts[i], layers)
        fps_harmless.append(compute_fingerprint(hs, refusal_dirs, layers, TOKEN_POSITIONS))
        if (i + 1) % 20 == 0: print(f"  Harmless: {i+1}/{n_cal}")
    fps_harmless = np.array(fps_harmless)

    fps_harmful = []
    for i in range(n_cal):
        hs = extract_hidden_states(model, tokenizer, harmful_prompts[i], layers)
        fps_harmful.append(compute_fingerprint(hs, refusal_dirs, layers, TOKEN_POSITIONS))
        if (i + 1) % 20 == 0: print(f"  Harmful: {i+1}/{n_cal}")
    fps_harmful = np.array(fps_harmful)

    fps_attacks = []
    for i, p in enumerate(attack_prompts):
        hs = extract_hidden_states(model, tokenizer, p, layers)
        fps_attacks.append(compute_fingerprint(hs, refusal_dirs, layers, TOKEN_POSITIONS))
        if (i + 1) % 50 == 0: print(f"  Attacks: {i+1}/{len(attack_prompts)}")
    fps_attacks = np.array(fps_attacks)

    # ══════════════════════════════════════════════════════════════════════
    #  Train Lorentz projection on harmless vs harmful (NO attack data)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[HPS-RTV] Training Lorentz projection (d_in={fps_harmless.shape[1]} → d_proj={args.d_proj})...")
    d_in = fps_harmless.shape[1]  # 15
    device = "cpu"  # fingerprints are small, CPU is fine

    X_cal = np.concatenate([fps_harmless, fps_harmful])
    y_cal = np.array([0] * len(fps_harmless) + [1] * len(fps_harmful))

    X_t = torch.tensor(X_cal, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_cal, dtype=torch.long, device=device)

    torch.manual_seed(42)
    proj = LorentzProjection(d_in, args.d_proj, k=1.0).to(device)
    opt = optim.Adam(proj.parameters(), lr=5e-3, weight_decay=1e-5)

    for epoch in range(args.epochs):
        pts = proj(X_t)
        loss = contrastive_loss(pts, y_t, proj.k)
        opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{args.epochs} loss={loss.item():.4f} k={proj.k.item():.3f}")

    proj.eval()

    # ══════════════════════════════════════════════════════════════════════
    #  Compute hyperbolic distances for scoring
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[HPS-RTV] Computing hyperbolic scores...")
    with torch.no_grad():
        pts_harmless = proj(torch.tensor(fps_harmless, dtype=torch.float32))
        pts_harmful = proj(torch.tensor(fps_harmful, dtype=torch.float32))
        pts_attacks = proj(torch.tensor(fps_attacks, dtype=torch.float32))

        # Centroids (Fréchet mean approximation: Euclidean mean then re-lift)
        centroid_harmless = pts_harmless.mean(dim=0, keepdim=True)
        centroid_harmful = pts_harmful.mean(dim=0, keepdim=True)

    def score_points(pts, centroid_h, centroid_harm, k):
        """M(x) = min(d_harmless, d_harmful) in hyperbolic space."""
        scores = []
        for i in range(pts.shape[0]):
            d_h = lorentz_distance(pts[i:i+1], centroid_h, k).item()
            d_harm = lorentz_distance(pts[i:i+1], centroid_harm, k).item()
            scores.append(min(d_h, d_harm))
        return np.array(scores)

    k = proj.k.detach()
    scores_harmless = score_points(pts_harmless, centroid_harmless, centroid_harmful, k)
    scores_harmful = score_points(pts_harmful, centroid_harmless, centroid_harmful, k)
    scores_attacks = score_points(pts_attacks, centroid_harmless, centroid_harmful, k)

    # Threshold at 95th percentile of calibration
    cal_scores = np.concatenate([scores_harmless, scores_harmful])
    threshold = float(np.quantile(cal_scores, 1.0 - FPR_TARGET))

    print(f"\n  Harmless median: {np.median(scores_harmless):.3f}")
    print(f"  Harmful median:  {np.median(scores_harmful):.3f}")
    print(f"  Attack median:   {np.median(scores_attacks):.3f}")
    print(f"  Threshold (95th pct): {threshold:.3f}")

    # ── Method 3: Tangent-Space Mahalanobis (best of both worlds) ──
    print(f"\n[HPS-RTV] Computing tangent-space Mahalanobis scores...")
    with torch.no_grad():
        # Log-map all points to tangent space at harmless centroid
        tan_harmless = lorentz_log_map(centroid_harmless, pts_harmless, k).numpy()
        tan_harmful = lorentz_log_map(centroid_harmless, pts_harmful, k).numpy()
        tan_attacks = lorentz_log_map(centroid_harmless, pts_attacks, k).numpy()

    # Fit LedoitWolf on tangent vectors of calibration data
    tan_cal = np.vstack([tan_harmless, tan_harmful])
    # Fit separate clusters in tangent space
    lw_tan_pos = LedoitWolf().fit(tan_harmless)
    lw_tan_neg = LedoitWolf().fit(tan_harmful)
    mu_tan_pos, prec_tan_pos = lw_tan_pos.location_, lw_tan_pos.precision_
    mu_tan_neg, prec_tan_neg = lw_tan_neg.location_, lw_tan_neg.precision_

    def tan_mahal_score(v):
        d_p = np.sqrt(max(0, (v - mu_tan_pos) @ prec_tan_pos @ (v - mu_tan_pos)))
        d_n = np.sqrt(max(0, (v - mu_tan_neg) @ prec_tan_neg @ (v - mu_tan_neg)))
        return min(d_p, d_n)

    ts_scores_harmless = np.array([tan_mahal_score(v) for v in tan_harmless])
    ts_scores_harmful = np.array([tan_mahal_score(v) for v in tan_harmful])
    ts_scores_attacks = np.array([tan_mahal_score(v) for v in tan_attacks])

    ts_cal = np.concatenate([ts_scores_harmless, ts_scores_harmful])
    ts_threshold = float(np.quantile(ts_cal, 1.0 - FPR_TARGET))
    ts_tpr = float((ts_scores_attacks > ts_threshold).mean())
    ts_auroc = roc_auc_score(
        np.array([0]*len(ts_scores_harmless) + [1]*len(ts_scores_attacks)),
        np.concatenate([ts_scores_harmless, ts_scores_attacks])
    )
    print(f"  Tangent-Mahalanobis AUROC: {ts_auroc:.3f}, TPR@5%: {ts_tpr:.3f}")

    # ══════════════════════════════════════════════════════════════════════
    #  Results
    # ══════════════════════════════════════════════════════════════════════
    tpr = float((scores_attacks > threshold).mean())
    y_true = np.array([0] * len(scores_harmless) + [1] * len(scores_attacks))
    all_scores = np.concatenate([scores_harmless, scores_attacks])
    auroc = roc_auc_score(y_true, all_scores)

    print(f"\n{'─'*60}")
    print(f"  HPS-RTV RESULTS")
    print(f"{'─'*60}")
    print(f"  AUROC (attacks vs harmless): {auroc:.3f}")
    print(f"  TPR @ {FPR_TARGET*100:.0f}% FPR: {tpr:.3f}")

    # Per-method breakdown
    methods_unique = sorted(set(attack_methods))
    print(f"\n  Per-attack-type breakdown:")
    print(f"  {'Method':<30} | {'N':>4} | {'TPR@5%':>7} | {'Median':>7} | {'Mean':>7}")
    print(f"  {'─'*30}─┼─{'─'*4}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}")
    per_method = {}
    for method in methods_unique:
        idx = [i for i, m in enumerate(attack_methods) if m == method]
        m_scores = scores_attacks[idx]
        m_tpr = float((m_scores > threshold).mean())
        m_median = float(np.median(m_scores))
        print(f"  {method:<30} | {len(idx):>4} | {m_tpr:>7.3f} | {m_median:>7.3f} | {np.mean(m_scores):>7.3f}")
        per_method[method] = {"n": len(idx), "tpr": m_tpr, "median": m_median}

    # ══════════════════════════════════════════════════════════════════════
    #  Comparison with standard RTV
    # ══════════════════════════════════════════════════════════════════════
    # Also compute standard Mahalanobis for direct comparison
    lw_pos = LedoitWolf().fit(fps_harmless)
    lw_neg = LedoitWolf().fit(fps_harmful)
    mu_pos, prec_pos = lw_pos.location_, lw_pos.precision_
    mu_neg, prec_neg = lw_neg.location_, lw_neg.precision_

    def mahal_score(fp):
        d_p = np.sqrt(max(0, (fp - mu_pos) @ prec_pos @ (fp - mu_pos)))
        d_n = np.sqrt(max(0, (fp - mu_neg) @ prec_neg @ (fp - mu_neg)))
        return min(d_p, d_n)

    rtv_scores_harmless = np.array([mahal_score(fp) for fp in fps_harmless])
    rtv_scores_attacks = np.array([mahal_score(fp) for fp in fps_attacks])
    rtv_cal = np.concatenate([rtv_scores_harmless,
                              np.array([mahal_score(fp) for fp in fps_harmful])])
    rtv_threshold = float(np.quantile(rtv_cal, 1.0 - FPR_TARGET))
    rtv_tpr = float((rtv_scores_attacks > rtv_threshold).mean())
    rtv_auroc = roc_auc_score(
        np.array([0]*len(rtv_scores_harmless) + [1]*len(rtv_scores_attacks)),
        np.concatenate([rtv_scores_harmless, rtv_scores_attacks])
    )

    print(f"\n{'─'*60}")
    print(f"  COMPARISON: All Three Methods")
    print(f"{'─'*60}")
    print(f"  {'Method':<25} | {'AUROC':>6} | {'TPR@5%':>7}")
    print(f"  {'─'*25}─┼─{'─'*6}─┼─{'─'*7}")
    print(f"  {'RTV (Mahalanobis)':<25} | {rtv_auroc:>6.3f} | {rtv_tpr:>7.3f}")
    print(f"  {'HPS-RTV (geodesic)':<25} | {auroc:>6.3f} | {tpr:>7.3f}")
    print(f"  {'HPS-RTV (tangent+Mahal)':<25} | {ts_auroc:>6.3f} | {ts_tpr:>7.3f}")
    best_auroc = max(auroc, ts_auroc)
    best_method = "geodesic" if auroc > ts_auroc else "tangent+Mahal"
    diff_auroc = best_auroc - rtv_auroc
    print(f"  {'─'*25}─┼─{'─'*6}─┼─{'─'*7}")
    print(f"  Best HPS-RTV vs RTV:     Δ AUROC = {diff_auroc:+.3f} ({best_method})")

    # ══════════════════════════════════════════════════════════════════════
    #  Visualization
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: PCA of raw fingerprints (same as RTV)
    X_all = np.vstack([fps_harmless, fps_harmful, fps_attacks])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_all)
    n_h, n_harm = len(fps_harmless), len(fps_harmful)

    ax = axes[0]
    ax.scatter(X_pca[:n_h, 0], X_pca[:n_h, 1], c='#2ecc71', label='Harmless', alpha=0.6, s=30)
    ax.scatter(X_pca[n_h:n_h+n_harm, 0], X_pca[n_h:n_h+n_harm, 1], c='#e74c3c', label='Harmful', alpha=0.6, s=30)
    ax.scatter(X_pca[n_h+n_harm:, 0], X_pca[n_h+n_harm:, 1], c='#9b59b6', label='Attacks', alpha=0.4, s=20)
    ax.set_title(f"Euclidean (RTV)\nAUROC={rtv_auroc:.3f}, TPR={rtv_tpr:.3f}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend()

    # Right: PCA of Lorentz-projected spatial coordinates
    with torch.no_grad():
        all_pts = proj(torch.tensor(X_all, dtype=torch.float32))
    all_spatial = all_pts[:, 1:].numpy()  # drop time coordinate
    pca2 = PCA(n_components=2)
    X_pca2 = pca2.fit_transform(all_spatial)

    ax = axes[1]
    ax.scatter(X_pca2[:n_h, 0], X_pca2[:n_h, 1], c='#2ecc71', label='Harmless', alpha=0.6, s=30)
    ax.scatter(X_pca2[n_h:n_h+n_harm, 0], X_pca2[n_h:n_h+n_harm, 1], c='#e74c3c', label='Harmful', alpha=0.6, s=30)
    ax.scatter(X_pca2[n_h+n_harm:, 0], X_pca2[n_h+n_harm:, 1], c='#9b59b6', label='Attacks', alpha=0.4, s=20)
    ax.set_title(f"Hyperbolic (HPS-RTV)\nAUROC={auroc:.3f}, TPR={tpr:.3f}")
    ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend()

    plt.suptitle("RTV vs HPS-RTV: Fingerprint Space Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    plot_path = args.output.replace('.json', '_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[HPS-RTV] Plot saved → {plot_path}")

    # ── Save ──
    results = {
        "model": args.model, "layers": layers, "d_proj": args.d_proj,
        "hps_rtv_geodesic": {"auroc": float(auroc), "tpr": float(tpr), "threshold": float(threshold)},
        "hps_rtv_tangent_mahal": {"auroc": float(ts_auroc), "tpr": float(ts_tpr), "threshold": float(ts_threshold)},
        "rtv_baseline": {"auroc": float(rtv_auroc), "tpr": float(rtv_tpr), "threshold": float(rtv_threshold)},
        "per_method": per_method,
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[HPS-RTV] Results saved → {args.output}")

    print(f"\n{'═'*60}")
    print(f"  HPS-RTV COMPLETE")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
