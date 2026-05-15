"""
Experiment 6 — Full HPS Pipeline on Validated Data
═══════════════════════════════════════════════════
Uses validated_attack_prompts.json (confirmed jailbreaks) and
validated_benign_prompts.json (confirmed normal responses).

Pipeline:
  1. Load validated prompts
  2. Extract activations
  3. Train Lorentz projection head (contrastive loss)
  4. Extract trajectory features
  5. Train classifier probe
  6. Report AUROC, FPR, dual-use FPR
  7. Save projection head + threshold for chat.py

Usage:
  python experiment6.py

After this, run:
  python chat.py   (loads the saved model automatically)
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve, f1_score

import config
from utils import (
    load_model, extract_activations,
    hyperbolic_curvature, lorentz_distance, save_json,
)
from dataset import DUAL_USE_BUILTIN


# ═══════════════════════════════════════════════════════════════════════════════
#  Projection Head
# ═══════════════════════════════════════════════════════════════════════════════

class LorentzProjection(nn.Module):
    def __init__(self, d_in, d_proj=256, k=1.0):
        super().__init__()
        self.proj = nn.Linear(d_in, d_proj, bias=False)
        self.scale = nn.Parameter(torch.tensor(1.0 / np.sqrt(d_proj)))
        self.k = k
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x):
        x_proj = self.proj(x) * self.scale
        norm_sq = (x_proj ** 2).sum(dim=-1, keepdim=True)
        x0 = torch.sqrt(1.0 / self.k + norm_sq)
        return torch.cat([x0, x_proj], dim=-1)


def lorentz_distance_torch(x, y, k=1.0):
    inner = -x[:, 0] * y[:, 0] + (x[:, 1:] * y[:, 1:]).sum(dim=-1)
    inner = torch.clamp(inner, max=-1.0 / k - 1e-6)
    return (1.0 / np.sqrt(k)) * torch.acosh(-k * inner)


def contrastive_loss(anchors, labels, k=1.0, margin=2.0):
    n = anchors.shape[0]
    loss = torch.tensor(0.0, device=anchors.device)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = lorentz_distance_torch(anchors[i:i+1], anchors[j:j+1], k=k).squeeze()
            if labels[i] == labels[j]:
                loss = loss + d ** 2
            else:
                loss = loss + torch.clamp(margin - d, min=0) ** 2
            count += 1
    return loss / max(count, 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature Extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_trajectory_features(proj, X_all, layers, k=1.0):
    """Extract features from learned projections."""
    device = next(proj.parameters()).device
    n_samples, n_layers, d_hidden = X_all.shape
    features = []

    proj.eval()
    with torch.no_grad():
        for i in range(n_samples):
            x = torch.tensor(X_all[i], dtype=torch.float32, device=device)
            h = proj(x)
            h_np = h.cpu().numpy()

            radii = h_np[:, 0]
            lorentz_pts = [h_np[j] for j in range(n_layers)]
            curv = hyperbolic_curvature(lorentz_pts, k=k)

            d_total = lorentz_distance(h_np[0], h_np[-1], k=k)
            path_len = sum(lorentz_distance(h_np[j], h_np[j+1], k=k) for j in range(n_layers - 1))

            feat = np.array([
                np.mean(radii),
                np.max(radii),
                np.min(radii),
                np.std(radii),
                float(np.max(radii) - np.min(radii)),
                float(curv.max()) if len(curv) > 0 else 0,
                float(curv.mean()) if len(curv) > 0 else 0,
                float(curv.std()) if len(curv) > 0 else 0,
                float(np.argmax(curv) / max(len(curv), 1)) if len(curv) > 0 else 0,
                d_total,
                path_len,
                d_total / (path_len + 1e-8),
            ])
            features.append(feat)

    return np.array(features)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'═'*60}")
    print(f"  Experiment 6 — Full HPS Pipeline (Validated Data)")
    print(f"  Model: {config.MODEL_NAME}")
    print(f"{'═'*60}\n")

    # ── Load validated data ──
    attacks_path = os.path.join(config.RESULTS_DIR, "validated_attack_prompts.json")
    benign_path = os.path.join(config.RESULTS_DIR, "validated_benign_prompts.json")

    assert os.path.exists(attacks_path), f"Run validate_attacks.py first! Missing: {attacks_path}"
    assert os.path.exists(benign_path), f"Run validate_benign.py first! Missing: {benign_path}"

    with open(attacks_path) as f:
        attack_prompts = json.load(f)
    with open(benign_path) as f:
        benign_prompts = json.load(f)

    # Balance classes
    n = min(len(attack_prompts), len(benign_prompts), 100)
    attack_prompts = attack_prompts[:n]
    benign_prompts = benign_prompts[:n]

    print(f"[exp6] Using {n} attacks + {n} benign = {2*n} total")
    print(f"[exp6] Dual-use evaluation set: {len(DUAL_USE_BUILTIN)} prompts")

    all_prompts = benign_prompts + attack_prompts
    labels = np.array([0] * n + [1] * n)

    # ── Load model ──
    model, tokenizer = load_model(config.MODEL_NAME, config.DEVICE, config.DTYPE)
    layers = config.TARGET_LAYERS

    # ── Extract activations ──
    print(f"\n[exp6] Extracting activations for {len(all_prompts)} prompts...")
    all_acts = []
    for i, p in enumerate(all_prompts):
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(all_prompts)}")
        all_acts.append(extract_activations(model, tokenizer, p, layers, config.DEVICE))

    d_hidden = all_acts[0][layers[0]].shape[0]
    n_layers = len(layers)
    X_all = np.zeros((len(all_prompts), n_layers, d_hidden))
    for i, act_dict in enumerate(all_acts):
        for j, l in enumerate(layers):
            if l in act_dict:
                X_all[i, j] = act_dict[l]

    # ── Train projection head ──
    print(f"\n[exp6] Training Lorentz projection head ({d_hidden}→256, 100 epochs)...")
    device = config.DEVICE
    proj = LorentzProjection(d_hidden, 256, config.HYPERBOLIC_K).to(device)
    optimizer = optim.Adam(proj.parameters(), lr=1e-3, weight_decay=1e-5)

    X_t = torch.tensor(X_all, dtype=torch.float32, device=device)
    y_t = torch.tensor(labels, dtype=torch.long, device=device)

    proj.train()
    for epoch in range(100):
        # Per-layer contrastive loss (Option C): train projection to separate at every layer
        total_loss = torch.tensor(0.0, device=device)
        for l in range(n_layers):
            h = proj(X_t[:, l, :])
            total_loss = total_loss + contrastive_loss(h, y_t, k=config.HYPERBOLIC_K)
        total_loss = total_loss / n_layers

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1}/100 — loss: {total_loss.item():.4f}")

    # ── Extract features ──
    print(f"\n[exp6] Extracting trajectory features...")
    X_feat = extract_trajectory_features(proj, X_all, layers, k=config.HYPERBOLIC_K)

    # ── Train probe (5-fold CV) ──
    print(f"\n[exp6] Training probe (5-fold CV)...")
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_feat)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    y_scores = cross_val_predict(clf, X_s, labels, cv=cv, method="predict_proba")[:, 1]
    y_pred = (y_scores > 0.5).astype(int)

    auroc = roc_auc_score(labels, y_scores)
    f1 = f1_score(labels, y_pred)
    fpr, tpr, _ = roc_curve(labels, y_scores)
    idx_95 = np.searchsorted(tpr, 0.95)
    fpr_at_95 = float(fpr[min(idx_95, len(fpr) - 1)])

    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │  AUROC:      {auroc:.3f}                 │")
    print(f"  │  FPR@95TPR:  {fpr_at_95:.3f}                 │")
    print(f"  │  F1:         {f1:.3f}                 │")
    print(f"  └─────────────────────────────────────┘")

    # ── Dual-use FPR ──
    print(f"\n[exp6] Evaluating dual-use FPR ({len(DUAL_USE_BUILTIN)} prompts)...")
    du_acts = []
    for i, p in enumerate(DUAL_USE_BUILTIN):
        du_acts.append(extract_activations(model, tokenizer, p, layers, config.DEVICE))

    X_du = np.zeros((len(DUAL_USE_BUILTIN), n_layers, d_hidden))
    for i, act_dict in enumerate(du_acts):
        for j, l in enumerate(layers):
            if l in act_dict:
                X_du[i, j] = act_dict[l]

    X_du_feat = extract_trajectory_features(proj, X_du, layers, k=config.HYPERBOLIC_K)

    # Train final classifier on full data for threshold
    clf_final = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf_final.fit(X_s, labels)

    X_du_s = scaler.transform(X_du_feat)
    du_scores = clf_final.predict_proba(X_du_s)[:, 1]
    du_pred = (du_scores > 0.5).astype(int)
    du_fpr = float(du_pred.sum()) / len(du_pred)

    print(f"  Dual-use FPR: {du_fpr:.3f} ({du_pred.sum()}/{len(du_pred)} flagged)")

    # ── Radii analysis ──
    proj.eval()
    with torch.no_grad():
        h_all = proj(X_t[:, mid, :])
        radii_all = h_all[:, 0].cpu().numpy()

    benign_radii = radii_all[labels == 0]
    adv_radii = radii_all[labels == 1]
    separation = abs(adv_radii.mean() - benign_radii.mean())

    print(f"\n[exp6] Radii (learned, mid-layer):")
    print(f"  Benign:      {benign_radii.mean():.4f} ± {benign_radii.std():.4f}")
    print(f"  Adversarial: {adv_radii.mean():.4f} ± {adv_radii.std():.4f}")
    print(f"  Separation:  {separation:.4f}")

    # ── Compute threshold for chat.py ──
    # Use the score that gives 95% TPR
    train_scores = clf_final.predict_proba(X_s)[:, 1]
    benign_scores = train_scores[labels == 0]
    adv_scores = train_scores[labels == 1]
    # Threshold = value where 95% of attacks are above it
    threshold = float(np.percentile(adv_scores, 5))  # 5th percentile of attacks = 95% TPR
    # But also ensure low FPR: take max of that and 95th percentile of benign
    threshold = max(threshold, float(np.percentile(benign_scores, 95)))

    print(f"\n[exp6] Decision threshold: {threshold:.4f}")
    print(f"  (benign max score: {benign_scores.max():.4f}, attack min score: {adv_scores.min():.4f})")

    # ── Save everything for chat.py ──
    proj_path = os.path.join(config.RESULTS_DIR, "hps_projection_head.pt")
    torch.save({
        "state_dict": proj.state_dict(),
        "d_in": d_hidden,
        "d_proj": 256,
        "k": config.HYPERBOLIC_K,
    }, proj_path)

    clf_path = os.path.join(config.RESULTS_DIR, "hps_classifier.json")
    with open(clf_path, "w") as f:
        json.dump({
            "threshold": threshold,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_std": scaler.scale_.tolist(),
            "clf_coef": clf_final.coef_[0].tolist(),
            "clf_intercept": float(clf_final.intercept_[0]),
        }, f)

    np.save(os.path.join(config.RESULTS_DIR, "hps_threshold.npy"), threshold)

    print(f"\n[exp6] Saved:")
    print(f"  Projection head → {proj_path}")
    print(f"  Classifier      → {clf_path}")

    # ── Plots ──
    # ROC
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#4CAF50", linewidth=2, label=f"HPS-Full (AUROC={auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.axhline(0.95, color="orange", linestyle=":", alpha=0.5, label="95% TPR")
    ax.axvline(fpr_at_95, color="orange", linestyle=":", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Experiment 6 — ROC (Validated Data)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, "exp6_roc.png"), dpi=150)
    plt.close()

    # Radii histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(benign_radii, bins=25, alpha=0.6, color="#4CAF50", label=f"Benign (n={n})")
    ax.hist(adv_radii, bins=25, alpha=0.6, color="#F44336", label=f"Adversarial (n={n})")
    ax.axvline(benign_radii.mean(), color="#4CAF50", linestyle="--", linewidth=2)
    ax.axvline(adv_radii.mean(), color="#F44336", linestyle="--", linewidth=2)
    ax.set_xlabel("Lorentz Radius (learned)")
    ax.set_ylabel("Count")
    ax.set_title("Experiment 6 — Radial Separation (Validated Data)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, "exp6_radii.png"), dpi=150)
    plt.close()

    # Score distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(train_scores[labels == 0], bins=25, alpha=0.6, color="#4CAF50", label="Benign")
    ax.hist(train_scores[labels == 1], bins=25, alpha=0.6, color="#F44336", label="Adversarial")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold={threshold:.3f}")
    ax.set_xlabel("Classifier Score")
    ax.set_ylabel("Count")
    ax.set_title("Experiment 6 — Score Distribution", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, "exp6_scores.png"), dpi=150)
    plt.close()

    print(f"\n[exp6] Plots saved to {config.PLOTS_DIR}/exp6_*.png")

    # ── Final summary ──
    results = {
        "model": config.MODEL_NAME,
        "n_benign": n,
        "n_adversarial": n,
        "n_dual_use": len(DUAL_USE_BUILTIN),
        "auroc": auroc,
        "fpr_at_95": fpr_at_95,
        "f1": f1,
        "dual_use_fpr": du_fpr,
        "threshold": threshold,
        "radii": {
            "benign_mean": float(benign_radii.mean()),
            "benign_std": float(benign_radii.std()),
            "adversarial_mean": float(adv_radii.mean()),
            "adversarial_std": float(adv_radii.std()),
            "separation": float(separation),
        },
    }
    save_json(results, "experiment6_results.json", config.RESULTS_DIR)

    print(f"\n{'═'*60}")
    print(f"  EXPERIMENT 6 COMPLETE")
    print(f"  AUROC: {auroc:.3f} | FPR@95: {fpr_at_95:.3f} | Dual-use FPR: {du_fpr:.3f}")
    print(f"  Threshold saved. Run 'python chat.py' to test interactively.")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
