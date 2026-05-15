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


def extract_trajectory_features(proj, X_all, k=1.0):
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

    proj = LorentzProjection(d_hidden, 256, k).to(device)
    optimizer = optim.Adam(proj.parameters(), lr=1e-3, weight_decay=1e-5)
    X_t = torch.tensor(X_all, dtype=torch.float32, device=device)
    y_t = torch.tensor(labels, dtype=torch.long, device=device)

    proj.train()
    mid = n_layers // 2
    for epoch in range(100):
        h = proj(X_t[:, mid, :])
        loss = contrastive_loss(h, y_t, k=k)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    X_feat = extract_trajectory_features(proj, X_all, k=k)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_feat)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
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

    # ── Load validated data ──
    attacks_path = os.path.join(config.RESULTS_DIR, "validated_attack_prompts.json")
    benign_path = os.path.join(config.RESULTS_DIR, "validated_benign_prompts.json")
    assert os.path.exists(attacks_path), f"Run validate_attacks.py first! Missing: {attacks_path}"
    assert os.path.exists(benign_path), f"Run validate_benign.py first! Missing: {benign_path}"

    with open(attacks_path) as f:
        attack_prompts = json.load(f)
    with open(benign_path) as f:
        benign_prompts = json.load(f)

    n = min(len(attack_prompts), len(benign_prompts), 100)
    attack_prompts = attack_prompts[:n]
    benign_prompts = benign_prompts[:n]
    print(f"[exp7] Data: {n} attacks + {n} benign")

    # ── Load model ──
    model, tokenizer = load_model(config.MODEL_NAME, config.DEVICE, config.DTYPE)

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 1: Layer Discovery (using last-token, 30 samples per class)
    # ══════════════════════════════════════════════════════════════════════════

    separations = discover_layers(
        model, tokenizer, benign_prompts, attack_prompts,
        config.DEVICE, pool_mode="last", n_samples=30
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

    all_prompts = benign_prompts + attack_prompts
    labels = np.array([0] * n + [1] * n)

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
    #  PHASE 3: Retrain best config and save for chat.py
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n[exp7] Retraining best config (pool={best_pool}, layers={top_layers})...")

    # Re-extract with best pooling if needed
    if best_pool != "last":
        print(f"  Re-extracting with pool={best_pool}...")
        acts_list = []
        for i, p in enumerate(all_prompts):
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(all_prompts)}")
            acts_list.append(extract_all_layers(model, tokenizer, p, config.DEVICE, best_pool))
        d_hidden = acts_list[0][top_layers[0]].shape[0]
        best_X = np.zeros((len(all_prompts), len(top_layers), d_hidden))
        for i, act_dict in enumerate(acts_list):
            for j, l in enumerate(top_layers):
                if l in act_dict:
                    best_X[i, j] = act_dict[l]

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

    du_scores = clf_final.predict_proba(X_du_s)[:, 1]
    du_fpr = float((du_scores > 0.5).sum()) / len(du_scores)
    print(f"  Dual-use FPR: {du_fpr:.3f}")

    # ── Compute threshold ──
    train_scores = clf_final.predict_proba(X_train_s)[:, 1]
    benign_scores = train_scores[labels == 0]
    adv_scores = train_scores[labels == 1]
    threshold = max(float(np.percentile(adv_scores, 5)), float(np.percentile(benign_scores, 95)))

    # ── Save for chat.py ──
    proj_path = os.path.join(config.RESULTS_DIR, "hps_projection_head.pt")
    torch.save({
        "state_dict": final["proj"].state_dict(),
        "d_in": d_hidden,
        "d_proj": 256,
        "k": config.HYPERBOLIC_K,
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
        "n_benign": n,
        "n_adversarial": n,
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
