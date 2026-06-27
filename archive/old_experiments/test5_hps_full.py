"""
TEST 5 — HPS-Full: Trained Hyperbolic Projection Head
══════════════════════════════════════════════════════════
QUESTION: Does a learned projection into hyperbolic space outperform
          the naive projection (HPS-Lite) from Tests 3/4?

HOW IT WORKS
  1. Extract activations (reuses Test 4 pipeline)
  2. Train a small linear projection head (d → d_proj) with a contrastive
     loss in Lorentz space: push benign/adversarial trajectories apart
  3. Extract trajectory features from the LEARNED projections
  4. Train the same logistic regression probe and compare AUROC

  This is the "Plan B" validation: does training the geometry in
  produce better separation than naive projection?

OUTPUTS
  results/test5_hps_full.json
  plots/test5_roc_comparison.png
  plots/test5_learned_radii.png

RUN
  python test5_hps_full.py
"""

import sys, os
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
    load_model, extract_activations_batch,
    lorentz_distance, lorentz_inner,
    hyperbolic_curvature, save_json,
)
from dataset import BENIGN, ADVERSARIAL, DUAL_USE


# ═══════════════════════════════════════════════════════════════════════════════
#  Learned Lorentz Projection Head (PyTorch)
# ═══════════════════════════════════════════════════════════════════════════════

class LorentzProjection(nn.Module):
    """Projects Euclidean activations onto the Lorentz hyperboloid."""

    def __init__(self, d_in: int, d_proj: int = 256, k: float = 1.0):
        super().__init__()
        self.proj = nn.Linear(d_in, d_proj, bias=False)
        self.scale = nn.Parameter(torch.tensor(1.0 / np.sqrt(d_proj)))
        self.k = k
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, d_in) Euclidean activation
        Returns: (batch, d_proj+1) point on Lorentz hyperboloid
        """
        x_proj = self.proj(x) * self.scale  # (batch, d_proj)
        norm_sq = (x_proj ** 2).sum(dim=-1, keepdim=True)  # (batch, 1)
        x0 = torch.sqrt(1.0 / self.k + norm_sq)  # time coordinate
        return torch.cat([x0, x_proj], dim=-1)  # (batch, d_proj+1)


def lorentz_distance_torch(x: torch.Tensor, y: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """Batched Lorentz geodesic distance."""
    inner = -x[:, 0] * y[:, 0] + (x[:, 1:] * y[:, 1:]).sum(dim=-1)
    inner = torch.clamp(inner, max=-1.0 / k - 1e-6)
    return (1.0 / np.sqrt(k)) * torch.acosh(-k * inner)


# ═══════════════════════════════════════════════════════════════════════════════
#  Contrastive Loss in Lorentz Space
# ═══════════════════════════════════════════════════════════════════════════════

def contrastive_lorentz_loss(anchors: torch.Tensor, labels: torch.Tensor, k: float = 1.0, margin: float = 2.0):
    """
    Pushes same-class points together and different-class points apart
    in Lorentz geodesic distance.
    """
    n = anchors.shape[0]
    loss = torch.tensor(0.0, device=anchors.device)
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            d = lorentz_distance_torch(
                anchors[i:i+1], anchors[j:j+1], k=k
            ).squeeze()
            if labels[i] == labels[j]:
                loss = loss + d ** 2
            else:
                loss = loss + torch.clamp(margin - d, min=0) ** 2
            count += 1

    return loss / max(count, 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

def train_projection_head(
    activations: list[dict],
    labels: np.ndarray,
    layer_indices: list[int],
    d_proj: int = 256,
    epochs: int = 80,
    lr: float = 1e-3,
    k: float = 1.0,
):
    """Train the Lorentz projection head with contrastive loss."""

    # Stack all layer activations into (n_samples, n_layers, d_hidden)
    d_hidden = activations[0][layer_indices[0]].shape[0]
    n_layers = len(layer_indices)
    n_samples = len(activations)

    X_all = np.zeros((n_samples, n_layers, d_hidden))
    for i, act_dict in enumerate(activations):
        for j, l in enumerate(layer_indices):
            if l in act_dict:
                X_all[i, j] = act_dict[l]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # One projection head shared across layers
    model = LorentzProjection(d_hidden, d_proj, k=k).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    X_tensor = torch.tensor(X_all, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(labels, dtype=torch.long, device=device)

    print(f"  [test5] Training projection head: {d_hidden}→{d_proj}, {epochs} epochs")

    model.train()
    for epoch in range(epochs):
        # Use middle layer as representative for contrastive training
        mid_layer_idx = n_layers // 2
        x_batch = X_tensor[:, mid_layer_idx, :]  # (n_samples, d_hidden)

        h = model(x_batch)  # (n_samples, d_proj+1) on hyperboloid
        loss = contrastive_lorentz_loss(h, y_tensor, k=k)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs} — loss: {loss.item():.4f}")

    model.eval()
    return model, X_all


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature Extraction from Learned Projections
# ═══════════════════════════════════════════════════════════════════════════════

def extract_learned_features(
    model: LorentzProjection,
    X_all: np.ndarray,
    layer_indices: list[int],
    k: float = 1.0,
) -> np.ndarray:
    """Extract trajectory features using the learned projection."""
    device = next(model.parameters()).device
    n_samples, n_layers, d_hidden = X_all.shape
    features = []

    model.eval()
    with torch.no_grad():
        for i in range(n_samples):
            # Project each layer's activation through the learned head
            x = torch.tensor(X_all[i], dtype=torch.float32, device=device)  # (n_layers, d_hidden)
            h = model(x)  # (n_layers, d_proj+1) on hyperboloid
            h_np = h.cpu().numpy()

            # Radial features (time coordinate = depth proxy)
            radii = h_np[:, 0]  # time coordinates

            # Curvature features
            lorentz_pts = [h_np[j] for j in range(n_layers)]
            curv = hyperbolic_curvature(lorentz_pts, k=k)

            # Displacement in Lorentz space
            d_total = lorentz_distance(h_np[0], h_np[-1], k=k)
            path_len = sum(
                lorentz_distance(h_np[j], h_np[j+1], k=k)
                for j in range(n_layers - 1)
            )

            feat = np.array([
                np.mean(radii),
                np.max(radii),
                np.min(radii),
                np.std(radii),
                float(np.max(radii) - np.min(radii)),  # radius range
                float(curv.max()) if len(curv) > 0 else 0,
                float(curv.mean()) if len(curv) > 0 else 0,
                float(curv.std()) if len(curv) > 0 else 0,
                float(np.argmax(curv) / max(len(curv), 1)) if len(curv) > 0 else 0,
                d_total,
                path_len,
                d_total / (path_len + 1e-8),  # progress ratio
            ])
            features.append(feat)

    return np.array(features)


# ═══════════════════════════════════════════════════════════════════════════════
#  Probe Training (same as Test 4)
# ═══════════════════════════════════════════════════════════════════════════════

def train_probe(X: np.ndarray, y: np.ndarray, name: str) -> dict:
    """5-fold CV logistic regression probe."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    y_scores = cross_val_predict(clf, X_s, y, cv=cv, method="predict_proba")[:, 1]
    y_pred = (y_scores > 0.5).astype(int)

    auroc = roc_auc_score(y, y_scores)
    f1 = f1_score(y, y_pred)

    fpr, tpr, thresholds = roc_curve(y, y_scores)
    idx_95 = np.searchsorted(tpr, 0.95)
    fpr_at_95 = float(fpr[min(idx_95, len(fpr) - 1)])

    print(f"  [{name}] AUROC={auroc:.3f}  FPR@95TPR={fpr_at_95:.3f}  F1={f1:.3f}")
    return {
        "auroc": auroc,
        "fpr_at_95": fpr_at_95,
        "f1": f1,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'═'*60}")
    print(f"  TEST 5 — HPS-Full: Trained Hyperbolic Projection")
    print(f"{'═'*60}")

    model, tokenizer = load_model(config.MODEL_NAME, config.DEVICE, config.DTYPE)
    layers = config.TARGET_LAYERS

    # ── Prepare data ──
    # Binary: benign=0, adversarial=1 (exclude dual-use from training)
    train_prompts = BENIGN + ADVERSARIAL
    train_labels = np.array([0] * len(BENIGN) + [1] * len(ADVERSARIAL))

    print(f"\n[test5] Extracting activations for {len(train_prompts)} prompts…")
    all_acts = extract_activations_batch(model, tokenizer, train_prompts, layers, config.DEVICE)

    # ── Train projection head ──
    print(f"\n[test5] Training Lorentz projection head…")
    proj_model, X_all = train_projection_head(
        all_acts, train_labels, layers,
        d_proj=256, epochs=80, lr=1e-3, k=config.HYPERBOLIC_K,
    )

    # ── Extract features from learned projections ──
    print(f"\n[test5] Extracting learned hyperbolic features…")
    X_learned = extract_learned_features(proj_model, X_all, layers, k=config.HYPERBOLIC_K)

    # ── Also extract naive features for comparison (same as Test 4) ──
    from utils import to_lorentz, hyperbolic_curvature as hyp_curv
    X_naive = []
    for act_dict in all_acts:
        points = [act_dict[l] for l in layers if l in act_dict]
        if len(points) < 3:
            X_naive.append(np.zeros(10))
            continue
        lorentz_pts = [to_lorentz(p, k=config.HYPERBOLIC_K) for p in points]
        radii = [float(lp[0]) for lp in lorentz_pts]
        curv = hyp_curv(lorentz_pts, k=config.HYPERBOLIC_K)
        X_naive.append(np.array([
            np.mean(radii), np.max(radii), float(np.max(radii) - np.min(radii)), np.std(radii),
            float(curv.max()), float(curv.mean()), float(curv.std()),
            float(np.argmax(curv) / max(len(curv), 1)),
            np.mean(radii), np.std(radii),
        ]))
    X_naive = np.array(X_naive)

    # ── Train probes and compare ──
    print(f"\n[test5] Training probes (5-fold CV)…")
    results = {}
    results["hps_lite"] = train_probe(X_naive, train_labels, "HPS-Lite (naive)")
    results["hps_full"] = train_probe(X_learned, train_labels, "HPS-Full (learned)")
    results["combined"] = train_probe(
        np.hstack([X_naive, X_learned]), train_labels, "Combined"
    )

    # ── Evaluate on dual-use ──
    print(f"\n[test5] Evaluating on dual-use prompts ({len(DUAL_USE)})…")
    du_acts = extract_activations_batch(model, tokenizer, DUAL_USE, layers, config.DEVICE)

    # Dual-use with learned projection
    d_hidden = all_acts[0][layers[0]].shape[0]
    n_layers_actual = len(layers)
    X_du_all = np.zeros((len(du_acts), n_layers_actual, d_hidden))
    for i, act_dict in enumerate(du_acts):
        for j, l in enumerate(layers):
            if l in act_dict:
                X_du_all[i, j] = act_dict[l]

    X_du_learned = extract_learned_features(proj_model, X_du_all, layers, k=config.HYPERBOLIC_K)

    # Train final classifiers on full training set for FPR evaluation
    scaler_l = StandardScaler()
    X_l_s = scaler_l.fit_transform(X_learned)
    clf_l = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(X_l_s, train_labels)

    X_du_l_s = scaler_l.transform(X_du_learned)
    du_pred_learned = clf_l.predict(X_du_l_s)
    du_fpr_learned = float(du_pred_learned.sum()) / len(du_pred_learned)

    scaler_n = StandardScaler()
    X_n_s = scaler_n.fit_transform(X_naive)
    clf_n = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(X_n_s, train_labels)

    X_du_naive = []
    for act_dict in du_acts:
        points = [act_dict[l] for l in layers if l in act_dict]
        if len(points) < 3:
            X_du_naive.append(np.zeros(10))
            continue
        lorentz_pts = [to_lorentz(p, k=config.HYPERBOLIC_K) for p in points]
        radii = [float(lp[0]) for lp in lorentz_pts]
        curv = hyp_curv(lorentz_pts, k=config.HYPERBOLIC_K)
        X_du_naive.append(np.array([
            np.mean(radii), np.max(radii), float(np.max(radii) - np.min(radii)), np.std(radii),
            float(curv.max()), float(curv.mean()), float(curv.std()),
            float(np.argmax(curv) / max(len(curv), 1)),
            np.mean(radii), np.std(radii),
        ]))
    X_du_naive = np.array(X_du_naive)
    X_du_n_s = scaler_n.transform(X_du_naive)
    du_pred_naive = clf_n.predict(X_du_n_s)
    du_fpr_naive = float(du_pred_naive.sum()) / len(du_pred_naive)

    print(f"  HPS-Lite dual-use FPR: {du_fpr_naive:.3f}")
    print(f"  HPS-Full dual-use FPR: {du_fpr_learned:.3f}")

    results["dual_use_fpr"] = {
        "hps_lite": du_fpr_naive,
        "hps_full": du_fpr_learned,
    }

    # ── Radii analysis ──
    # Check if learned projection produces meaningful radial variation
    with torch.no_grad():
        device = next(proj_model.parameters()).device
        benign_radii = []
        adv_radii = []
        mid = n_layers_actual // 2
        for i in range(len(train_labels)):
            x = torch.tensor(X_all[i, mid:mid+1], dtype=torch.float32, device=device)
            h = proj_model(x)
            r = h[0, 0].item()
            if train_labels[i] == 0:
                benign_radii.append(r)
            else:
                adv_radii.append(r)

    results["radii_analysis"] = {
        "benign_mean": float(np.mean(benign_radii)),
        "benign_std": float(np.std(benign_radii)),
        "adversarial_mean": float(np.mean(adv_radii)),
        "adversarial_std": float(np.std(adv_radii)),
        "separation": abs(float(np.mean(adv_radii)) - float(np.mean(benign_radii))),
    }
    print(f"\n[test5] Radii analysis (learned projection, mid-layer):")
    print(f"  Benign mean radius:      {np.mean(benign_radii):.4f} ± {np.std(benign_radii):.4f}")
    print(f"  Adversarial mean radius:  {np.mean(adv_radii):.4f} ± {np.std(adv_radii):.4f}")
    print(f"  Separation:               {results['radii_analysis']['separation']:.4f}")

    # ── Summary ──
    delta = results["hps_full"]["auroc"] - results["hps_lite"]["auroc"]
    results["delta_auroc_full_vs_lite"] = delta
    verdict = (
        "HPS-Full IMPROVES over HPS-Lite"
        if delta > 0.01 else
        "HPS-Full ≈ HPS-Lite (trained projection not clearly better)"
    )
    results["verdict"] = verdict
    print(f"\n[test5] SUMMARY")
    print(f"  HPS-Lite AUROC:  {results['hps_lite']['auroc']:.3f}")
    print(f"  HPS-Full AUROC:  {results['hps_full']['auroc']:.3f}")
    print(f"  Δ AUROC:         {delta:+.3f}")
    print(f"  Verdict:         {verdict}")

    # ── Plots ──
    # ROC comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, color, label in [
        ("hps_lite", "#FF9800", "HPS-Lite (naive projection)"),
        ("hps_full", "#4CAF50", "HPS-Full (learned projection)"),
        ("combined", "#2196F3", "Combined"),
    ]:
        r = results[name]
        ax.plot(r["fpr"], r["tpr"], color=color, linewidth=2,
                label=f"{label} (AUROC={r['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Test 5 — ROC: HPS-Lite vs HPS-Full", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(config.PLOTS_DIR, "test5_roc_comparison.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"[test5] ROC plot saved → {roc_path}")

    # Radii distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(benign_radii, bins=20, alpha=0.6, color="#4CAF50", label="Benign")
    ax.hist(adv_radii, bins=20, alpha=0.6, color="#F44336", label="Adversarial")
    ax.set_xlabel("Lorentz Time Coordinate (learned)")
    ax.set_ylabel("Count")
    ax.set_title("Test 5 — Learned Radial Distribution", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    radii_path = os.path.join(config.PLOTS_DIR, "test5_learned_radii.png")
    plt.savefig(radii_path, dpi=150)
    plt.close()
    print(f"[test5] Radii plot saved → {radii_path}")

    # ── Save ──
    # Remove non-serializable items
    save_results = {k: v for k, v in results.items() if k not in ("fpr", "tpr")}
    save_json(save_results, "test5_hps_full.json", config.RESULTS_DIR)
    print(f"\n[test5] Done. ✓")


if __name__ == "__main__":
    main()
