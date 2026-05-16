"""
Experiment 7 — Layer Discovery + Token Pooling Ablation
═══════════════════════════════════════════════════════════
1. Discovers optimal layers (highest benign/adversarial separation)
2. Compares last-token vs mean-pooling vs last-5 mean
3. Trains HPS-Full on the best configuration
4. Saves model for chat.py

Usage:
  python experiment7.py

Requires: results/validated_attack_prompts.json, results/validated_benign_prompts.json
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
    load_model, hyperbolic_curvature, lorentz_distance, save_json,
)
from dataset import DUAL_USE_BUILTIN


# ═══════════════════════════════════════════════════════════════════════════════
#  Activation Extraction (supports multiple pooling modes)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Cannot find transformer layers")


def extract_all_layers(model, tokenizer, prompt, device, pool_mode="last"):
    """
    Extract hidden states from ALL layers for a single prompt.
    pool_mode: "last" | "mean" | "last5"
    Returns: dict[layer_idx] -> numpy array (hidden_dim,)
    """
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512, padding=False
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    seq_len = inputs["input_ids"].shape[1]

    captured = {}
    hooks = []
    layers = _get_transformer_layers(model)

    def make_hook(idx):
        def hook(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            # hidden shape: (1, seq_len, hidden_dim)
            h = hidden[0].detach().cpu().float()  # (seq_len, hidden_dim)
            if pool_mode == "last":
                captured[idx] = h[-1].numpy()
            elif pool_mode == "mean":
                captured[idx] = h.mean(dim=0).numpy()
            elif pool_mode == "last5":
                n = min(5, h.shape[0])
                captured[idx] = h[-n:].mean(dim=0).numpy()
        return hook

    for idx in range(len(layers)):
        hooks.append(layers[idx].register_forward_hook(make_hook(idx)))

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    return captured


# ═══════════════════════════════════════════════════════════════════════════════
#  Layer Discovery
# ═══════════════════════════════════════════════════════════════════════════════

def discover_layers(model, tokenizer, benign_prompts, attack_prompts, device,
                    pool_mode="last", n_samples=30):
    """
    Find layers with maximum separation between benign and adversarial.
    Returns sorted list of (layer_idx, separation_score).
    """
    n = min(n_samples, len(benign_prompts), len(attack_prompts))
    print(f"\n[exp7] Layer discovery: {n} benign + {n} adversarial, pool={pool_mode}")

    # Collect activations from all layers
    benign_acts = []  # list of dicts
    for i in range(n):
        if (i + 1) % 10 == 0:
            print(f"  Benign {i+1}/{n}")
        benign_acts.append(extract_all_layers(model, tokenizer, benign_prompts[i], device, pool_mode))

    attack_acts = []
    for i in range(n):
        if (i + 1) % 10 == 0:
            print(f"  Attack {i+1}/{n}")
        attack_acts.append(extract_all_layers(model, tokenizer, attack_prompts[i], device, pool_mode))

    # Compute separation at each layer
    n_layers = len(benign_acts[0])
    separations = []

    for l in range(n_layers):
        b_vecs = np.array([a[l] for a in benign_acts])
        a_vecs = np.array([a[l] for a in attack_acts])

        # Separation = distance between class means / average within-class std
        b_mean = b_vecs.mean(axis=0)
        a_mean = a_vecs.mean(axis=0)

        between = np.linalg.norm(a_mean - b_mean)
        within_b = np.mean([np.linalg.norm(v - b_mean) for v in b_vecs])
        within_a = np.mean([np.linalg.norm(v - a_mean) for v in a_vecs])
        within = (within_b + within_a) / 2

        score = between / (within + 1e-8)
        separations.append((l, score))

    separations.sort(key=lambda x: -x[1])
    return separations


# ═══════════════════════════════════════════════════════════════════════════════
#  Projection Head + Training (same as experiment6)
# ═══════════════════════════════════════════════════════════════════════════════

class LorentzProjection(nn.Module):
    def __init__(self, d_in, d_proj=None, k=1.0, n_layers=8):
        super().__init__()
        d_proj = d_proj or config.PROJECTION_DIM
        self.proj = nn.Linear(d_in, d_proj, bias=False)
        self.scale = nn.Parameter(torch.tensor(1.0 / np.sqrt(d_proj)))
        self.log_k = nn.Parameter(torch.tensor(np.log(k)))  # learnable curvature
        # Learnable temperature per layer (calibrates distance scale)
        self.log_tau = nn.Parameter(torch.zeros(n_layers))  # τ_l, initialized to 1.0 (log(1)=0)
        nn.init.xavier_uniform_(self.proj.weight)

    @property
    def k(self):
        return torch.exp(self.log_k).clamp(min=0.1, max=10.0)

    def tau(self, layer_idx):
        """Per-layer temperature, clamped for stability."""
        return torch.exp(self.log_tau[layer_idx]).clamp(min=0.01, max=10.0)

    def forward(self, x):
        # FP32 for geometric operations (numerical stability)
        x_fp32 = x.float()
        x_proj = self.proj(x_fp32) * self.scale
        norm_sq = (x_proj ** 2).sum(dim=-1, keepdim=True)
        x0 = torch.sqrt(1.0 / self.k + norm_sq)
        return torch.cat([x0, x_proj], dim=-1)


def lorentz_distance_torch(x, y, k=1.0):
    """k can be a scalar or a tensor (for learnable curvature)."""
    inner = -x[:, 0] * y[:, 0] + (x[:, 1:] * y[:, 1:]).sum(dim=-1)
    if isinstance(k, torch.Tensor):
        inner = torch.clamp(inner, max=(-1.0 / k - 1e-6).item())
        return (1.0 / torch.sqrt(k)) * torch.acosh(-k * inner)
    else:
        inner = torch.clamp(inner, max=-1.0 / k - 1e-6)
        return (1.0 / np.sqrt(k)) * torch.acosh(-k * inner)


def contrastive_loss(anchors, labels, k=1.0, margin=2.0, tau=1.0):
    """Vectorized contrastive loss with temperature scaling. FP32 for stability."""
    n = anchors.shape[0]
    # All pairwise Lorentz distances (vectorized, FP32)
    anchors = anchors.float()
    inner = -anchors[:, 0:1] @ anchors[:, 0:1].T + anchors[:, 1:] @ anchors[:, 1:].T
    inner = torch.clamp(inner, max=-1.0 / k - 1e-6)
    dists = (1.0 / np.sqrt(k)) * torch.acosh(-k * inner)

    # Temperature scaling
    dists = dists / tau

    # Masks
    same_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    diff_mask = 1.0 - same_mask
    triu = torch.triu(torch.ones(n, n, device=anchors.device), diagonal=1)

    same_loss = (dists ** 2 * same_mask * triu).sum()
    diff_loss = (torch.clamp(margin - dists, min=0) ** 2 * diff_mask * triu).sum()
    count = triu.sum().clamp(min=1)
    return (same_loss + diff_loss) / count


def extract_trajectory_features(proj, X_all, k=None):
    if k is None:
        k = proj.k.item()
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
            curv = hyperbolic_curvature([h_np[j] for j in range(n_layers)], k=k)
            d_total = lorentz_distance(h_np[0], h_np[-1], k=k)
            path_len = sum(lorentz_distance(h_np[j], h_np[j+1], k=k) for j in range(n_layers - 1))
            feat = np.array([
                np.mean(radii), np.max(radii), np.min(radii), np.std(radii),
                float(np.max(radii) - np.min(radii)),
                float(curv.max()) if len(curv) > 0 else 0,
                float(curv.mean()) if len(curv) > 0 else 0,
                float(curv.std()) if len(curv) > 0 else 0,
                float(np.argmax(curv) / max(len(curv), 1)) if len(curv) > 0 else 0,
                d_total, path_len, d_total / (path_len + 1e-8),
            ])
            features.append(feat)
    return np.array(features)


def train_and_evaluate(X_all, labels, k, tag=""):
    """Train projection + probe, return metrics."""
    device = config.DEVICE
    n_samples, n_layers, d_hidden = X_all.shape

    proj = LorentzProjection(d_hidden, config.PROJECTION_DIM, k, n_layers=n_layers).to(device)
    optimizer = optim.Adam(proj.parameters(), lr=1e-3, weight_decay=1e-5)
    X_t = torch.tensor(X_all, dtype=torch.float32, device=device)
    y_t = torch.tensor(labels, dtype=torch.long, device=device)

    proj.train()
    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(200):
        # Option C: per-layer contrastive loss, summed across all layers
        total_loss = torch.tensor(0.0, device=device)
        for l in range(n_layers):
            h = proj(X_t[:, l, :])  # project layer l for all samples
            tau_l = proj.tau(l)     # per-layer temperature
            total_loss = total_loss + contrastive_loss(h, y_t, k=proj.k.item(), tau=tau_l)
        total_loss = total_loss / n_layers

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Early stopping
        if total_loss.item() < best_loss - 1e-4:
            best_loss = total_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= 20:
            print(f"    [{tag}] Early stopping at epoch {epoch+1}, loss: {best_loss:.4f}, κ: {proj.k.item():.3f}")
            break

        if (epoch + 1) % 25 == 0:
            print(f"    [{tag}] Epoch {epoch+1}/200 — loss: {total_loss.item():.4f}, κ: {proj.k.item():.3f}")

    X_feat = extract_trajectory_features(proj, X_all, k=k)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_feat)

    n_splits = min(5, min(np.sum(labels == 0), np.sum(labels == 1)))
    assert n_splits >= 2, f"Not enough samples for CV: {np.sum(labels==0)} benign, {np.sum(labels==1)} adversarial. Need ≥2 per class."
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    y_scores = cross_val_predict(clf, X_s, labels, cv=cv, method="predict_proba")[:, 1]

    auroc = roc_auc_score(labels, y_scores)
    f1 = f1_score(labels, (y_scores > 0.5).astype(int))
    fpr, tpr, _ = roc_curve(labels, y_scores)
    idx_95 = np.searchsorted(tpr, 0.95)
    fpr_at_95 = float(fpr[min(idx_95, len(fpr) - 1)])

    print(f"  [{tag}] AUROC={auroc:.3f}  FPR@95={fpr_at_95:.3f}  F1={f1:.3f}")
    return {"auroc": auroc, "fpr_at_95": fpr_at_95, "f1": f1, "proj": proj, "scaler": scaler}


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'═'*60}")
    print(f"  Experiment 7 — Layer Discovery + Pooling Ablation")
    print(f"  Model: {config.MODEL_NAME}")
    print(f"{'═'*60}\n")

    # ── Load data from dataset.py (benign from HuggingFace, attacks from JSON) ──
    from dataset import BENIGN, ADVERSARIAL

    attack_prompts = ADVERSARIAL
    benign_prompts = BENIGN

    n_atk = len(attack_prompts)
    n_ben = min(len(benign_prompts), n_atk * 4)  # cap benign at 4x attacks
    benign_prompts = benign_prompts[:n_ben]
    print(f"[exp7] Data: {n_atk} attacks + {n_ben} benign")

    # ── Split: calibration (for layer discovery) vs training (for projection) ──
    n_calib_atk = min(30, n_atk // 3)
    n_calib_ben = min(30, len(benign_prompts) // 3)
    calib_attack = attack_prompts[:n_calib_atk]
    calib_benign = benign_prompts[:n_calib_ben]
    train_attack = attack_prompts[n_calib_atk:]
    train_benign = benign_prompts[n_calib_ben:]
    print(f"[exp7] Calibration: {n_calib_atk} attacks + {n_calib_ben} benign")
    print(f"[exp7] Training:    {len(train_attack)} attacks + {len(train_benign)} benign")

    # ── Load model ──
    model, tokenizer = load_model(config.MODEL_NAME, config.DEVICE, config.DTYPE)

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 1: Layer Discovery (on calibration split ONLY)
    # ══════════════════════════════════════════════════════════════════════════

    separations = discover_layers(
        model, tokenizer, calib_benign, calib_attack,
        config.DEVICE, pool_mode="last", n_samples=min(n_calib_atk, n_calib_ben)
    )

    # Print all layers ranked
    print(f"\n[exp7] Layer separation ranking (all {len(separations)} layers):")
    print(f"  {'Layer':>5} | {'Score':>8}")
    print(f"  {'─'*5}─┼─{'─'*8}")
    for l, s in separations[:15]:
        print(f"  {l:>5} | {s:>8.4f}")

    # Pick top 8 layers
    top_layers = sorted([l for l, _ in separations[:8]])
    print(f"\n[exp7] Selected top-8 layers: {top_layers}")

    # Plot layer separations
    all_layers_sorted = sorted(separations, key=lambda x: x[0])
    fig, ax = plt.subplots(figsize=(10, 4))
    xs = [l for l, _ in all_layers_sorted]
    ys = [s for _, s in all_layers_sorted]
    ax.bar(xs, ys, color=["#4CAF50" if l in top_layers else "#BDBDBD" for l in xs])
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Separation Score")
    ax.set_title("Experiment 7 — Layer Separation (Benign vs Adversarial)", fontweight="bold")
    ax.set_xticks(xs)
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, "exp7_layer_separation.png"), dpi=150)
    plt.close()
    print(f"[exp7] Layer plot saved → {config.PLOTS_DIR}/exp7_layer_separation.png")

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 2: Pooling Ablation (last vs mean vs last5) on top layers
    # ══════════════════════════════════════════════════════════════════════════

    all_prompts = train_benign + train_attack
    labels = np.array([0] * len(train_benign) + [1] * len(train_attack))

    ablation_results = {}

    for pool_mode in ["last", "mean", "last5"]:
        print(f"\n[exp7] Extracting activations (pool={pool_mode}, layers={top_layers})...")
        acts_list = []
        for i, p in enumerate(all_prompts):
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(all_prompts)}")
            act_dict = extract_all_layers(model, tokenizer, p, config.DEVICE, pool_mode)
            acts_list.append(act_dict)

        d_hidden = acts_list[0][top_layers[0]].shape[0]
        n_layers = len(top_layers)
        X_all = np.zeros((len(all_prompts), n_layers, d_hidden))
        for i, act_dict in enumerate(acts_list):
            for j, l in enumerate(top_layers):
                if l in act_dict:
                    X_all[i, j] = act_dict[l]

        result = train_and_evaluate(X_all, labels, config.HYPERBOLIC_K, tag=pool_mode)
        ablation_results[pool_mode] = {
            "auroc": result["auroc"],
            "fpr_at_95": result["fpr_at_95"],
            "f1": result["f1"],
        }

        # Keep best for saving
        if pool_mode == "last":
            best_result = result
            best_X = X_all
            best_pool = pool_mode

    # Find best pooling mode
    for mode, res in ablation_results.items():
        if res["auroc"] > ablation_results[best_pool]["auroc"]:
            best_pool = mode

    print(f"\n[exp7] ═══ Pooling Ablation Results ═══")
    print(f"  {'Mode':<8} | {'AUROC':>7} | {'FPR@95':>7} | {'F1':>6}")
    print(f"  {'─'*8}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*6}")
    for mode, res in ablation_results.items():
        marker = " ← best" if mode == best_pool else ""
        print(f"  {mode:<8} | {res['auroc']:>7.3f} | {res['fpr_at_95']:>7.3f} | {res['f1']:>6.3f}{marker}")

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 2b: Critical Baselines (using best pooling mode)
    # ══════════════════════════════════════════════════════════════════════════

    # Re-extract with best pooling for all baselines
    print(f"\n[exp7] Re-extracting with best pool={best_pool} for baseline comparison...")
    acts_list = []
    for i, p in enumerate(all_prompts):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(all_prompts)}")
        acts_list.append(extract_all_layers(model, tokenizer, p, config.DEVICE, best_pool))

    d_hidden = acts_list[0][top_layers[0]].shape[0]
    n_layers_sel = len(top_layers)
    best_X = np.zeros((len(all_prompts), n_layers_sel, d_hidden))
    for i, act_dict in enumerate(acts_list):
        for j, l in enumerate(top_layers):
            if l in act_dict:
                best_X[i, j] = act_dict[l]

    baseline_results = {}

    # ── Baseline 1: Fisher-8 Raw (no projection, just concatenate and classify) ──
    print(f"\n[exp7] Baseline: Fisher-8 Raw (concatenated activations → logistic regression)...")
    X_concat = best_X.reshape(len(all_prompts), -1)  # (n_samples, n_layers * d_hidden)
    scaler_raw = StandardScaler()
    X_concat_s = scaler_raw.fit_transform(X_concat)
    cv = StratifiedKFold(n_splits=min(5, min(np.sum(labels == 0), np.sum(labels == 1))), shuffle=True, random_state=42)
    clf_raw = LogisticRegression(C=0.01, max_iter=2000, random_state=42)  # low C to prevent overfit on high-dim
    y_scores_raw = cross_val_predict(clf_raw, X_concat_s, labels, cv=cv, method="predict_proba")[:, 1]
    auroc_raw = roc_auc_score(labels, y_scores_raw)
    fpr_raw, tpr_raw, _ = roc_curve(labels, y_scores_raw)
    idx_95 = np.searchsorted(tpr_raw, 0.95)
    fpr95_raw = float(fpr_raw[min(idx_95, len(fpr_raw) - 1)])
    f1_raw = f1_score(labels, (y_scores_raw > 0.5).astype(int))
    baseline_results["fisher8_raw"] = {"auroc": auroc_raw, "fpr_at_95": fpr95_raw, "f1": f1_raw}
    print(f"  [Fisher-8 Raw] AUROC={auroc_raw:.3f}  FPR@95={fpr95_raw:.3f}  F1={f1_raw:.3f}")

    # ── Baseline 2: Euclidean-Trained (linear head + contrastive in L2) ──
    print(f"\n[exp7] Baseline: Euclidean-Trained (linear head + L2 contrastive)...")
    device = config.DEVICE
    d_proj = config.PROJECTION_DIM
    proj_euc = nn.Linear(d_hidden, d_proj, bias=False).to(device)
    nn.init.xavier_uniform_(proj_euc.weight)
    scale_euc = nn.Parameter(torch.tensor(1.0 / np.sqrt(d_proj)).to(device))
    opt_euc = optim.Adam(list(proj_euc.parameters()) + [scale_euc], lr=1e-3, weight_decay=1e-5)
    X_t = torch.tensor(best_X, dtype=torch.float32, device=device)
    y_t = torch.tensor(labels, dtype=torch.long, device=device)

    for epoch in range(100):
        total_loss = torch.tensor(0.0, device=device)
        for l in range(n_layers_sel):
            h = proj_euc(X_t[:, l, :]) * scale_euc  # (n, 256) in flat space
            # L2 contrastive loss
            n_s = h.shape[0]
            loss_l = torch.tensor(0.0, device=device)
            count = 0
            for i in range(n_s):
                for j in range(i + 1, n_s):
                    d = torch.norm(h[i] - h[j])
                    if y_t[i] == y_t[j]:
                        loss_l = loss_l + d ** 2
                    else:
                        loss_l = loss_l + torch.clamp(2.0 - d, min=0) ** 2
                    count += 1
            total_loss = total_loss + loss_l / max(count, 1)
        total_loss = total_loss / n_layers_sel
        opt_euc.zero_grad()
        total_loss.backward()
        opt_euc.step()

    # Extract Euclidean features (same 12 features but using L2 distances)
    proj_euc.eval()
    X_euc_feat = []
    with torch.no_grad():
        for i in range(len(all_prompts)):
            x = X_t[i]  # (n_layers, d_hidden)
            h = (proj_euc(x) * scale_euc).cpu().numpy()  # (n_layers, d_proj)
            norms = np.linalg.norm(h, axis=1)
            diffs = [np.linalg.norm(h[j+1] - h[j]) for j in range(n_layers_sel - 1)]
            # Curvature analog in Euclidean
            curv_e = []
            for j in range(1, n_layers_sel - 1):
                v = h[j] - h[j-1]
                a = h[j+1] - 2*h[j] + h[j-1]
                vn = np.linalg.norm(v)
                if vn < 1e-8:
                    curv_e.append(0.0)
                else:
                    cross_sq = np.dot(v,v)*np.dot(a,a) - np.dot(v,a)**2
                    curv_e.append(np.sqrt(max(cross_sq, 0)) / vn**3)
            curv_e = np.array(curv_e) if curv_e else np.array([0.0])
            d_total = np.linalg.norm(h[-1] - h[0])
            path_len = sum(diffs)
            feat = np.array([
                np.mean(norms), np.max(norms), np.min(norms), np.std(norms),
                float(np.max(norms) - np.min(norms)),
                float(curv_e.max()), float(curv_e.mean()), float(curv_e.std()),
                float(np.argmax(curv_e) / max(len(curv_e), 1)),
                d_total, path_len, d_total / (path_len + 1e-8),
            ])
            X_euc_feat.append(feat)
    X_euc_feat = np.array(X_euc_feat)

    scaler_euc = StandardScaler()
    X_euc_s = scaler_euc.fit_transform(X_euc_feat)
    clf_euc = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    y_scores_euc = cross_val_predict(clf_euc, X_euc_s, labels, cv=cv, method="predict_proba")[:, 1]
    auroc_euc = roc_auc_score(labels, y_scores_euc)
    fpr_euc, tpr_euc, _ = roc_curve(labels, y_scores_euc)
    idx_95 = np.searchsorted(tpr_euc, 0.95)
    fpr95_euc = float(fpr_euc[min(idx_95, len(fpr_euc) - 1)])
    f1_euc = f1_score(labels, (y_scores_euc > 0.5).astype(int))
    baseline_results["euclidean_trained"] = {"auroc": auroc_euc, "fpr_at_95": fpr95_euc, "f1": f1_euc}
    print(f"  [Euclidean-Trained] AUROC={auroc_euc:.3f}  FPR@95={fpr95_euc:.3f}  F1={f1_euc:.3f}")

    # ── Baseline 3: HPS-Full (Hyperbolic-Trained) ──
    print(f"\n[exp7] HPS-Full (Hyperbolic-Trained)...")
    hps_result = train_and_evaluate(best_X, labels, config.HYPERBOLIC_K, tag="HPS-Full")
    baseline_results["hyperbolic_trained"] = {
        "auroc": hps_result["auroc"], "fpr_at_95": hps_result["fpr_at_95"], "f1": hps_result["f1"]
    }

    # ── Summary table ──
    print(f"\n[exp7] ═══ Critical Baseline Comparison ═══")
    print(f"  {'Method':<22} | {'AUROC':>7} | {'FPR@95':>7} | {'F1':>6}")
    print(f"  {'─'*22}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*6}")
    for name, res in baseline_results.items():
        print(f"  {name:<22} | {res['auroc']:>7.3f} | {res['fpr_at_95']:>7.3f} | {res['f1']:>6.3f}")

    ablation_results["baselines"] = baseline_results

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 3: Retrain best config and save for chat.py
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n[exp7] Retraining best config (pool={best_pool}, layers={top_layers})...")

    # best_X already computed in baseline comparison phase
    final = train_and_evaluate(best_X, labels, config.HYPERBOLIC_K, tag="FINAL")

    # ── Dual-use evaluation ──
    print(f"\n[exp7] Dual-use FPR evaluation ({len(DUAL_USE_BUILTIN)} prompts)...")
    du_acts = []
    for p in DUAL_USE_BUILTIN:
        du_acts.append(extract_all_layers(model, tokenizer, p, config.DEVICE, best_pool))

    X_du = np.zeros((len(DUAL_USE_BUILTIN), len(top_layers), d_hidden))
    for i, act_dict in enumerate(du_acts):
        for j, l in enumerate(top_layers):
            if l in act_dict:
                X_du[i, j] = act_dict[l]

    X_du_feat = extract_trajectory_features(final["proj"], X_du, k=config.HYPERBOLIC_K)
    X_du_s = final["scaler"].transform(X_du_feat)

    clf_final = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    X_train_feat = extract_trajectory_features(final["proj"], best_X, k=config.HYPERBOLIC_K)
    X_train_s = final["scaler"].transform(X_train_feat)
    clf_final.fit(X_train_s, labels)

    # ── Compute threshold from cross-val scores (not in-sample) ──
    # Use the CV predictions from train_and_evaluate which are out-of-fold
    cv_scores = cross_val_predict(
        LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        X_train_s, labels,
        cv=StratifiedKFold(n_splits=min(5, min(np.sum(labels==0), np.sum(labels==1))), shuffle=True, random_state=42),
        method="predict_proba"
    )[:, 1]
    benign_scores = cv_scores[labels == 0]
    adv_scores = cv_scores[labels == 1]
    threshold = max(float(np.percentile(adv_scores, 5)), float(np.percentile(benign_scores, 95)))
    print(f"\n[exp7] Threshold (from CV scores): {threshold:.4f}")

    # ── Dual-use FPR at the actual threshold ──
    du_scores = clf_final.predict_proba(X_du_s)[:, 1]
    du_fpr = float((du_scores > threshold).sum()) / len(du_scores)
    print(f"  Dual-use FPR (at threshold={threshold:.3f}): {du_fpr:.3f}")

    # ── Save for chat.py ──
    proj_path = os.path.join(config.RESULTS_DIR, "hps_projection_head.pt")
    torch.save({
        "state_dict": final["proj"].state_dict(),
        "d_in": d_hidden,
        "d_proj": config.PROJECTION_DIM,
        "k": final["proj"].k.item(),  # learned curvature
        "layers": top_layers,
        "pool_mode": best_pool,
    }, proj_path)

    np.save(os.path.join(config.RESULTS_DIR, "hps_threshold.npy"), threshold)

    clf_path = os.path.join(config.RESULTS_DIR, "hps_classifier.json")
    with open(clf_path, "w") as f:
        json.dump({
            "threshold": threshold,
            "scaler_mean": final["scaler"].mean_.tolist(),
            "scaler_std": final["scaler"].scale_.tolist(),
            "clf_coef": clf_final.coef_[0].tolist(),
            "clf_intercept": float(clf_final.intercept_[0]),
            "layers": top_layers,
            "pool_mode": best_pool,
        }, f)

    # ── Save results ──
    results = {
        "model": config.MODEL_NAME,
        "discovered_layers": top_layers,
        "layer_separations": {str(l): s for l, s in separations},
        "pooling_ablation": ablation_results,
        "best_pool_mode": best_pool,
        "final_auroc": final["auroc"],
        "final_fpr_at_95": final["fpr_at_95"],
        "final_f1": final["f1"],
        "dual_use_fpr": du_fpr,
        "threshold": threshold,
        "n_benign_total": len(benign_prompts),
        "n_adversarial_total": len(attack_prompts),
        "n_benign_train": len(train_benign),
        "n_adversarial_train": len(train_attack),
        "n_calibration": n_calib_atk + n_calib_ben,
    }
    save_json(results, "experiment7_results.json", config.RESULTS_DIR)

    # ── Summary ──
    print(f"\n{'═'*60}")
    print(f"  EXPERIMENT 7 COMPLETE")
    print(f"  Best layers: {top_layers}")
    print(f"  Best pooling: {best_pool}")
    print(f"  AUROC: {final['auroc']:.3f} | FPR@95: {final['fpr_at_95']:.3f} | Dual-use FPR: {du_fpr:.3f}")
    print(f"  Saved. Run 'python chat.py' to test.")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
