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

    # Balanced weighting: normalize same-class and diff-class contributions equally
    n_same = (same_mask * triu).sum().clamp(min=1)
    n_diff = (diff_mask * triu).sum().clamp(min=1)
    return (same_loss / n_same + diff_loss / n_diff) / 2.0


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
            # Pass k as tensor for gradient flow through curvature
            total_loss = total_loss + contrastive_loss(h, y_t, k=proj.k.detach().item(), tau=tau_l)
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

    X_feat = extract_trajectory_features(proj, X_all)  # uses proj.k (learned)
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
    #  PHASE 2: Pooling Ablation — single extraction pass, cache all modes
    # ══════════════════════════════════════════════════════════════════════════

    all_prompts = train_benign + train_attack
    labels = np.array([0] * len(train_benign) + [1] * len(train_attack))

    # Guard: need enough samples
    n_minority = min(np.sum(labels == 0), np.sum(labels == 1))
    assert n_minority >= 10, f"Only {n_minority} in minority class. Need more data."

    # Single extraction pass — cache all pool modes
    print(f"\n[exp7] Extracting activations (all pool modes, layers={top_layers})...")
    all_acts_cache = []
    for i, p in enumerate(all_prompts):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(all_prompts)}")
        act_dict = extract_all_layers(model, tokenizer, p, config.DEVICE, "last")
        # extract_all_layers returns per-layer vectors; we need all modes
        # Re-extract with a combined hook
        all_acts_cache.append(act_dict)

    # For mean/last5 we need the full sequence — but extract_all_layers only returns one mode.
    # Extract once per mode but reuse model (unavoidable without rewriting hooks)
    X_by_mode = {}
    d_hidden = all_acts_cache[0][top_layers[0]].shape[0]
    n_layers = len(top_layers)

    # "last" mode already cached
    X_last = np.zeros((len(all_prompts), n_layers, d_hidden))
    for i, act_dict in enumerate(all_acts_cache):
        for j, l in enumerate(top_layers):
            if l in act_dict:
                X_last[i, j] = act_dict[l]
    X_by_mode["last"] = X_last

    # Extract mean and last5
    for pool_mode in ["mean", "last5"]:
        print(f"  Extracting pool={pool_mode}...")
        X_mode = np.zeros((len(all_prompts), n_layers, d_hidden))
        for i, p in enumerate(all_prompts):
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(all_prompts)}")
            act_dict = extract_all_layers(model, tokenizer, p, config.DEVICE, pool_mode)
            for j, l in enumerate(top_layers):
                if l in act_dict:
                    X_mode[i, j] = act_dict[l]
        X_by_mode[pool_mode] = X_mode

    # Run ablation
    ablation_results = {}
    for pool_mode in ["last", "mean", "last5"]:
        result = train_and_evaluate(X_by_mode[pool_mode], labels, config.HYPERBOLIC_K, tag=pool_mode)
        ablation_results[pool_mode] = {
            "auroc": result["auroc"],
            "fpr_at_95": result["fpr_at_95"],
            "f1": result["f1"],
        }

    # Find best pooling mode
    best_pool = max(ablation_results, key=lambda m: ablation_results[m]["auroc"])
    best_X = X_by_mode[best_pool]  # guaranteed correct

    print(f"\n[exp7] ═══ Pooling Ablation Results ═══")
    print(f"  {'Mode':<8} | {'AUROC':>7} | {'FPR@95':>7} | {'F1':>6}")
    print(f"  {'─'*8}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*6}")
    for mode, res in ablation_results.items():
        marker = " ← best" if mode == best_pool else ""
        print(f"  {mode:<8} | {res['auroc']:>7.3f} | {res['fpr_at_95']:>7.3f} | {res['f1']:>6.3f}{marker}")

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 2b: Critical Baselines (on best_X)
    # ══════════════════════════════════════════════════════════════════════════

    n_layers_sel = len(top_layers)
    baseline_results = {}

    # ── Baseline 1: Fisher-8 Raw ──
    print(f"\n[exp7] Baseline: Fisher-8 Raw...")
    X_concat = best_X.reshape(len(all_prompts), -1)
    scaler_raw = StandardScaler()
    X_concat_s = scaler_raw.fit_transform(X_concat)
    n_splits = min(5, int(n_minority))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf_raw = LogisticRegression(C=0.01, max_iter=2000, random_state=42)
    y_scores_raw = cross_val_predict(clf_raw, X_concat_s, labels, cv=cv, method="predict_proba")[:, 1]
    auroc_raw = roc_auc_score(labels, y_scores_raw)
    fpr_raw, tpr_raw, _ = roc_curve(labels, y_scores_raw)
    idx_95 = np.searchsorted(tpr_raw, 0.95)
    fpr95_raw = float(fpr_raw[min(idx_95, len(fpr_raw) - 1)])
    f1_raw = f1_score(labels, (y_scores_raw > 0.5).astype(int))
    baseline_results["fisher8_raw"] = {"auroc": auroc_raw, "fpr_at_95": fpr95_raw, "f1": f1_raw}
    print(f"  [Fisher-8 Raw] AUROC={auroc_raw:.3f}  FPR@95={fpr95_raw:.3f}  F1={f1_raw:.3f}")

    # ── Baseline 2: Euclidean-Trained (vectorized) ──
    print(f"\n[exp7] Baseline: Euclidean-Trained...")
    device = config.DEVICE
    d_proj = config.PROJECTION_DIM
    proj_euc = nn.Linear(d_hidden, d_proj, bias=False).to(device)
    nn.init.xavier_uniform_(proj_euc.weight)
    scale_euc = nn.Parameter(torch.tensor(1.0 / np.sqrt(d_proj)).to(device))
    opt_euc = optim.Adam(list(proj_euc.parameters()) + [scale_euc], lr=1e-3, weight_decay=1e-5)
    X_t = torch.tensor(best_X, dtype=torch.float32, device=device)
    y_t = torch.tensor(labels, dtype=torch.long, device=device)

    # Vectorized L2 contrastive with early stopping (same budget as HPS-Full)
    best_loss_euc = float('inf')
    patience_euc = 0
    for epoch in range(200):
        total_loss = torch.tensor(0.0, device=device)
        for l in range(n_layers_sel):
            h = (proj_euc(X_t[:, l, :]) * scale_euc).float()
            dists = torch.cdist(h, h)
            same_mask = (y_t.unsqueeze(0) == y_t.unsqueeze(1)).float()
            diff_mask = 1.0 - same_mask
            triu = torch.triu(torch.ones(h.shape[0], h.shape[0], device=device), diagonal=1)
            same_loss = (dists ** 2 * same_mask * triu).sum()
            diff_loss = (torch.clamp(2.0 - dists, min=0) ** 2 * diff_mask * triu).sum()
            count = triu.sum().clamp(min=1)
            total_loss = total_loss + (same_loss + diff_loss) / count
        total_loss = total_loss / n_layers_sel
        opt_euc.zero_grad()
        total_loss.backward()
        opt_euc.step()
        if total_loss.item() < best_loss_euc - 1e-4:
            best_loss_euc = total_loss.item()
            patience_euc = 0
        else:
            patience_euc += 1
        if patience_euc >= 20:
            break

    # Extract Euclidean features using SAME curvature formula as HPS (triangle inequality deviation)
    proj_euc.eval()
    X_euc_feat = []
    with torch.no_grad():
        for i in range(len(all_prompts)):
            h = (proj_euc(X_t[i]) * scale_euc).cpu().numpy()
            norms = np.linalg.norm(h, axis=1)
            # Triangle inequality curvature (same as hyperbolic version, but with L2)
            curv_e = []
            for j in range(1, n_layers_sel - 1):
                d_prev = np.linalg.norm(h[j] - h[j-1])
                d_next = np.linalg.norm(h[j+1] - h[j])
                d_span = np.linalg.norm(h[j+1] - h[j-1])
                denom = d_prev + d_next
                curv_e.append(0.0 if denom < 1e-8 else abs(d_prev + d_next - d_span) / denom)
            curv_e = np.array(curv_e) if curv_e else np.array([0.0])
            d_total = np.linalg.norm(h[-1] - h[0])
            path_len = sum(np.linalg.norm(h[j+1] - h[j]) for j in range(n_layers_sel - 1))
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
    y_scores_euc = cross_val_predict(
        LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        X_euc_s, labels, cv=cv, method="predict_proba"
    )[:, 1]
    auroc_euc = roc_auc_score(labels, y_scores_euc)
    fpr_euc, tpr_euc, _ = roc_curve(labels, y_scores_euc)
    idx_95 = np.searchsorted(tpr_euc, 0.95)
    fpr95_euc = float(fpr_euc[min(idx_95, len(fpr_euc) - 1)])
    f1_euc = f1_score(labels, (y_scores_euc > 0.5).astype(int))
    baseline_results["euclidean_trained"] = {"auroc": auroc_euc, "fpr_at_95": fpr95_euc, "f1": f1_euc}
    print(f"  [Euclidean-Trained] AUROC={auroc_euc:.3f}  FPR@95={fpr95_euc:.3f}  F1={f1_euc:.3f}")

    # ── Baseline 3: Nonlinear-Euclidean (LayerNorm + tanh + L2 contrastive) ──
    print(f"\n[exp7] Baseline: Nonlinear-Euclidean (LayerNorm + tanh)...")
    proj_nl = nn.Sequential(
        nn.Linear(d_hidden, d_proj, bias=False),
        nn.LayerNorm(d_proj),
        nn.Tanh(),
    ).to(device)
    nn.init.xavier_uniform_(proj_nl[0].weight)
    opt_nl = optim.Adam(proj_nl.parameters(), lr=1e-3, weight_decay=1e-5)

    best_loss_nl = float('inf')
    patience_nl = 0
    for epoch in range(200):
        total_loss = torch.tensor(0.0, device=device)
        for l in range(n_layers_sel):
            h = proj_nl(X_t[:, l, :].float())
            dists = torch.cdist(h, h)
            same_mask = (y_t.unsqueeze(0) == y_t.unsqueeze(1)).float()
            diff_mask = 1.0 - same_mask
            triu = torch.triu(torch.ones(h.shape[0], h.shape[0], device=device), diagonal=1)
            same_loss = (dists ** 2 * same_mask * triu).sum()
            diff_loss = (torch.clamp(2.0 - dists, min=0) ** 2 * diff_mask * triu).sum()
            count = triu.sum().clamp(min=1)
            total_loss = total_loss + (same_loss + diff_loss) / count
        total_loss = total_loss / n_layers_sel
        opt_nl.zero_grad()
        total_loss.backward()
        opt_nl.step()
        if total_loss.item() < best_loss_nl - 1e-4:
            best_loss_nl = total_loss.item()
            patience_nl = 0
        else:
            patience_nl += 1
        if patience_nl >= 20:
            break

    proj_nl.eval()
    X_nl_feat = []
    with torch.no_grad():
        for i in range(len(all_prompts)):
            h = proj_nl(X_t[i].float()).cpu().numpy()
            norms = np.linalg.norm(h, axis=1)
            curv_e = []
            for j in range(1, n_layers_sel - 1):
                d_prev = np.linalg.norm(h[j] - h[j-1])
                d_next = np.linalg.norm(h[j+1] - h[j])
                d_span = np.linalg.norm(h[j+1] - h[j-1])
                denom = d_prev + d_next
                curv_e.append(0.0 if denom < 1e-8 else abs(d_prev + d_next - d_span) / denom)
            curv_e = np.array(curv_e) if curv_e else np.array([0.0])
            d_total = np.linalg.norm(h[-1] - h[0])
            path_len = sum(np.linalg.norm(h[j+1] - h[j]) for j in range(n_layers_sel - 1))
            feat = np.array([
                np.mean(norms), np.max(norms), np.min(norms), np.std(norms),
                float(np.max(norms) - np.min(norms)),
                float(curv_e.max()), float(curv_e.mean()), float(curv_e.std()),
                float(np.argmax(curv_e) / max(len(curv_e), 1)),
                d_total, path_len, d_total / (path_len + 1e-8),
            ])
            X_nl_feat.append(feat)
    X_nl_feat = np.array(X_nl_feat)

    scaler_nl = StandardScaler()
    X_nl_s = scaler_nl.fit_transform(X_nl_feat)
    y_scores_nl = cross_val_predict(
        LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        X_nl_s, labels, cv=cv, method="predict_proba"
    )[:, 1]
    auroc_nl = roc_auc_score(labels, y_scores_nl)
    fpr_nl, tpr_nl, _ = roc_curve(labels, y_scores_nl)
    idx_95 = np.searchsorted(tpr_nl, 0.95)
    fpr95_nl = float(fpr_nl[min(idx_95, len(fpr_nl) - 1)])
    f1_nl = f1_score(labels, (y_scores_nl > 0.5).astype(int))
    baseline_results["nonlinear_euclidean"] = {"auroc": auroc_nl, "fpr_at_95": fpr95_nl, "f1": f1_nl}
    print(f"  [Nonlinear-Euclidean] AUROC={auroc_nl:.3f}  FPR@95={fpr95_nl:.3f}  F1={f1_nl:.3f}")

    # ── Baseline 4: HPS-Full ──
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
    #  PHASE 3: Final model + threshold + dual-use + plots
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n[exp7] Retraining final model (pool={best_pool}, layers={top_layers})...")
    final = train_and_evaluate(best_X, labels, config.HYPERBOLIC_K, tag="FINAL")

    # Extract features for threshold and dual-use
    X_train_feat = extract_trajectory_features(final["proj"], best_X)
    X_train_s = final["scaler"].transform(X_train_feat)
    clf_final = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf_final.fit(X_train_s, labels)

    # Threshold from CV scores (out-of-fold, not in-sample)
    cv_scores = cross_val_predict(
        LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        X_train_s, labels,
        cv=StratifiedKFold(n_splits=min(5, int(n_minority)), shuffle=True, random_state=42),
        method="predict_proba"
    )[:, 1]
    benign_cv = cv_scores[labels == 0]
    adv_cv = cv_scores[labels == 1]
    threshold = max(float(np.percentile(adv_cv, 10)), float(np.percentile(benign_cv, 90)))
    print(f"[exp7] Threshold (from CV): {threshold:.4f}")

    # Dual-use FPR at correct threshold
    print(f"\n[exp7] Dual-use FPR ({len(DUAL_USE_BUILTIN)} prompts, threshold={threshold:.3f})...")
    du_acts = []
    for p in DUAL_USE_BUILTIN:
        du_acts.append(extract_all_layers(model, tokenizer, p, config.DEVICE, best_pool))
    X_du = np.zeros((len(DUAL_USE_BUILTIN), n_layers, d_hidden))
    for i, act_dict in enumerate(du_acts):
        for j, l in enumerate(top_layers):
            if l in act_dict:
                X_du[i, j] = act_dict[l]
    X_du_feat = extract_trajectory_features(final["proj"], X_du)
    X_du_s = final["scaler"].transform(X_du_feat)
    du_scores = clf_final.predict_proba(X_du_s)[:, 1]
    du_fpr = float((du_scores > threshold).sum()) / len(du_scores)
    print(f"  Dual-use FPR: {du_fpr:.3f}")

    # ── Plots ──
    # ROC comparison
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_raw, tpr_raw, color="#9E9E9E", linestyle=":", linewidth=2,
            label=f"Fisher-8 Raw (AUROC={auroc_raw:.3f})")
    ax.plot(fpr_euc, tpr_euc, color="#2196F3", linestyle="--", linewidth=2,
            label=f"Euclidean-Trained (AUROC={auroc_euc:.3f})")
    ax.plot(fpr_nl, tpr_nl, color="#FF9800", linestyle="-.", linewidth=2,
            label=f"Nonlinear-Euclidean (AUROC={auroc_nl:.3f})")
    # HPS-Full ROC from CV scores
    hps_cv = cross_val_predict(
        LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        X_train_s, labels,
        cv=StratifiedKFold(n_splits=min(5, int(n_minority)), shuffle=True, random_state=42),
        method="predict_proba"
    )[:, 1]
    fpr_hyp, tpr_hyp, _ = roc_curve(labels, hps_cv)
    ax.plot(fpr_hyp, tpr_hyp, color="#F44336", linestyle="-", linewidth=2,
            label=f"HPS-Full (AUROC={hps_result['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], "k:", linewidth=1, alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Experiment 7 — Critical Baseline Comparison", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, "exp7_roc.png"), dpi=150)
    plt.close()
    print(f"[exp7] ROC plot → {config.PLOTS_DIR}/exp7_roc.png")

    # Radii histogram
    radii_benign = X_train_feat[labels == 0, 0]
    radii_adv = X_train_feat[labels == 1, 0]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(radii_benign, bins=20, alpha=0.6, color="#4CAF50", label="Benign", density=True)
    ax.hist(radii_adv, bins=20, alpha=0.6, color="#F44336", label="Adversarial", density=True)
    ax.set_xlabel("Mean Lorentz Radius")
    ax.set_ylabel("Density")
    ax.set_title("Experiment 7 — Radial Separation", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, "exp7_radii.png"), dpi=150)
    plt.close()
    print(f"[exp7] Radii plot → {config.PLOTS_DIR}/exp7_radii.png")

    # Score distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cv_scores[labels == 0], bins=20, alpha=0.6, color="#4CAF50", label="Benign")
    ax.hist(cv_scores[labels == 1], bins=20, alpha=0.6, color="#F44336", label="Adversarial")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold={threshold:.3f}")
    ax.set_xlabel("Classifier Score")
    ax.set_title("Experiment 7 — Score Distribution (CV)", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, "exp7_scores.png"), dpi=150)
    plt.close()
    print(f"[exp7] Scores plot → {config.PLOTS_DIR}/exp7_scores.png")

    # ── Save for chat.py ──
    proj_path = os.path.join(config.RESULTS_DIR, "hps_projection_head.pt")
    torch.save({
        "state_dict": final["proj"].state_dict(),
        "d_in": d_hidden,
        "d_proj": config.PROJECTION_DIM,
        "k": final["proj"].k.item(),
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
        "pooling_ablation": {k: v for k, v in ablation_results.items() if k != "baselines"},
        "baselines": baseline_results,
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
        "learned_kappa": final["proj"].k.item(),
    }
    save_json(results, "experiment7_results.json", config.RESULTS_DIR)

    # ── Summary ──
    print(f"\n{'═'*60}")
    print(f"  EXPERIMENT 7 COMPLETE")
    print(f"  Best layers: {top_layers}")
    print(f"  Best pooling: {best_pool}")
    print(f"  Learned κ: {final['proj'].k.item():.3f}")
    print(f"  AUROC: {final['auroc']:.3f} | FPR@95: {final['fpr_at_95']:.3f} | Dual-use FPR: {du_fpr:.3f}")
    print(f"  Saved. Run 'python chat.py' to test.")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
