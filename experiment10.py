"""
Experiment 10 — RTV Baseline (Representation Trajectory Verification)
══════════════════════════════════════════════════════════════════════
Implements RTV from Derya & Sunar (arXiv:2605.03095, 2026):
  "Revisiting JBShield: Breaking and Rebuilding Representation-Level
   Jailbreak Defenses"

RTV is a strong published baseline for our cross-attack experiment.
Key properties:
  - NO training (uses Arditi et al. refusal direction = μ_harmful - μ_harmless)
  - 15-dim fingerprint: 3 layers × 5 token positions
  - Mahalanobis distance with Ledoit-Wolf shrinkage
  - Reported AUROC = 0.99 against JB-GCG adaptive attack on Llama-3-8B
  - 7% ASR under full white-box adaptive attack (vs 46.2% for JBShield)

This experiment:
  1. Implements RTV
  2. Runs cross-attack evaluation (same protocol as experiment8 Diagnostic 3)
  3. Empirically tests whether HPS's learned radial direction approximates
     RTV's analytical refusal direction (cosine similarity test)

Layer mapping for Vicuna-13B (40 layers):
  Llama-3-8B paper used {18, 25, 32} of 32 layers (56%, 78%, 100%).
  For Vicuna-13B: {22, 31, 39}.

Token positions: last 5 tokens of the prompt (indices -1 to -5).

Usage:
  python experiment10.py
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
np.random.seed(42)
import torch
torch.manual_seed(42)
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

import config
from utils import load_model, save_json
from utils import _get_transformer_layers
from dataset import BENIGN, ADVERSARIAL


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

RTV_LAYERS = [22, 31, 39]            # Vicuna-13B equivalent of Llama-3 {18,25,32}
RTV_TOKEN_POSITIONS = [-1, -2, -3, -4, -5]  # last 5 tokens (negative = from end)
N_FEATURES = len(RTV_LAYERS) * len(RTV_TOKEN_POSITIONS)  # 15


# ═══════════════════════════════════════════════════════════════════════════
#  Multi-position layer extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_multi_position(model, tokenizer, prompt, layers, positions, device):
    """Extract hidden states at specified layers AND token positions.
    Returns dict: {layer: {position: vector}}.
    Negative position p means index seq_len + p (so -1 = last token).
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    seq_len = inputs["input_ids"].shape[1]

    captured = {}
    hooks = []
    transformer_layers = _get_transformer_layers(model)

    def make_hook(layer_idx):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            # h: (batch, seq, hidden)
            captured[layer_idx] = h[0].detach().cpu().float().numpy()
        return hook

    for l in layers:
        hooks.append(transformer_layers[l].register_forward_hook(make_hook(l)))

    with torch.no_grad():
        model(**inputs)
    for h in hooks:
        h.remove()

    out = {}
    for l in layers:
        h_seq = captured[l]  # (seq_len, hidden)
        out[l] = {}
        for p in positions:
            pos = seq_len + p if p <= 0 else p
            pos = max(0, min(pos, seq_len - 1))
            out[l][p] = h_seq[pos]
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  RTV Components
# ═══════════════════════════════════════════════════════════════════════════

def compute_refusal_directions(activations_harmful, activations_harmless, layers):
    """Per Arditi et al. (2024): r_l = (μ_harmful - μ_harmless) / ||...||.
    activations_*: list of dicts {layer: {position: vec}}; we use position=-1 (last token)
    by convention for the refusal direction itself (RTV uses last token).
    """
    refusal_dirs = {}
    for l in layers:
        h_harmful = np.array([a[l][-1] for a in activations_harmful])
        h_harmless = np.array([a[l][-1] for a in activations_harmless])
        mu_h = h_harmful.mean(axis=0)
        mu_b = h_harmless.mean(axis=0)
        r = mu_h - mu_b
        norm = np.linalg.norm(r) + 1e-8
        refusal_dirs[l] = r / norm
    return refusal_dirs


def compute_fingerprint(activation_dict, refusal_dirs, layers, positions):
    """F[l, p] = cos(r_l, h_l^(p)). Returns flat 15-dim vector.
    activation_dict: {layer: {position: vec}}
    """
    fp = []
    for l in layers:
        r = refusal_dirs[l]
        for p in positions:
            h = activation_dict[l][p]
            cos_sim = float(np.dot(h, r) / (np.linalg.norm(h) * np.linalg.norm(r) + 1e-8))
            fp.append(cos_sim)
    return np.array(fp)


def fit_mahalanobis(fingerprints):
    """Fit Ledoit-Wolf precision matrix on fingerprints."""
    lw = LedoitWolf().fit(fingerprints)
    return lw.location_, lw.precision_


def mahalanobis_distance(x, mu, precision):
    diff = x - mu
    return float(np.sqrt(diff @ precision @ diff))


def rtv_score(fingerprints, mu, precision):
    """Per-fingerprint Mahalanobis distance from legitimate cluster mean.
    Higher = more anomalous = more attack-like."""
    return np.array([mahalanobis_distance(fp, mu, precision) for fp in fingerprints])


# ═══════════════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════════════

def tpr_at_fixed_fpr(scores, y, target_fpr=0.01):
    ben = scores[y == 0]
    atk = scores[y == 1]
    if len(ben) == 0 or len(atk) == 0:
        return 0.0, 0.0
    threshold = float(np.quantile(ben, 1.0 - target_fpr))
    tpr = float((atk > threshold).mean())
    actual_fpr = float((ben > threshold).mean())
    return tpr, actual_fpr


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'═'*60}")
    print(f"  Experiment 10 — RTV Baseline + HPS Direction Comparison")
    print(f"  Model: {config.MODEL_NAME}")
    print(f"  RTV layers: {RTV_LAYERS}")
    print(f"  RTV token positions: {RTV_TOKEN_POSITIONS}")
    print(f"  Fingerprint dim: {N_FEATURES}")
    print(f"{'═'*60}\n")

    print(f"[exp10] {len(ADVERSARIAL)} attacks, {len(BENIGN)} benign")

    # ── Load attack method labels (same pattern as exp8) ──
    cat_path = os.path.join(config.RESULTS_DIR, "validated_attacks_categorized.json")
    with open(cat_path) as f:
        categorized = json.load(f)

    attack_prompts = []
    attack_methods = []
    for method, prompts in categorized.items():
        for p in prompts:
            attack_prompts.append(p)
            attack_methods.append(method)

    benign_prompts = list(BENIGN)
    methods_arr = np.array(["benign"] * len(benign_prompts) + attack_methods)
    print(f"  Methods: {dict((m, attack_methods.count(m)) for m in set(attack_methods))}")

    # ── Load model ──
    model, tokenizer = load_model(config.MODEL_NAME, config.DEVICE, config.DTYPE)

    # ── Setup data ──
    all_prompts = list(benign_prompts) + list(attack_prompts)
    labels = np.array([0] * len(benign_prompts) + [1] * len(attack_prompts))

    # ── Extract activations at RTV layers + positions ──
    print(f"\n[exp10] Extracting activations (3 layers × 5 positions)...")
    all_acts = []
    for i, p in enumerate(all_prompts):
        ad = extract_multi_position(model, tokenizer, p, RTV_LAYERS, RTV_TOKEN_POSITIONS, config.DEVICE)
        all_acts.append(ad)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(all_prompts)}")

    benign_acts = [all_acts[i] for i in range(len(all_prompts)) if labels[i] == 0]
    adv_acts = [all_acts[i] for i in range(len(all_prompts)) if labels[i] == 1]
    print(f"[exp10] Benign acts: {len(benign_acts)}, Adversarial acts: {len(adv_acts)}")

    results = {
        "config": {
            "model": config.MODEL_NAME,
            "rtv_layers": RTV_LAYERS,
            "rtv_token_positions": RTV_TOKEN_POSITIONS,
            "fingerprint_dim": N_FEATURES,
        },
        "n_benign": len(benign_acts),
        "n_adversarial": len(adv_acts),
    }

    # ══════════════════════════════════════════════════════════════════════
    #  CROSS-ATTACK EVALUATION (same protocol as experiment8 Diagnostic 3)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  RTV CROSS-ATTACK GENERALIZATION")
    print(f"{'─'*60}")

    # Benign 80/20 split (consistent with exp8)
    benign_idx = np.where(labels == 0)[0]
    rng = np.random.RandomState(42)
    benign_perm = rng.permutation(benign_idx)
    n_benign_train = int(0.8 * len(benign_perm))
    benign_train_idx = set(benign_perm[:n_benign_train].tolist())
    benign_test_idx = set(benign_perm[n_benign_train:].tolist())

    methods_unique = sorted(set(attack_methods))

    print(f"\n  Benign split: {len(benign_train_idx)} train / {len(benign_test_idx)} test")
    print(f"\n  {'Held-out':<28} | {'n_test':>6} | {'AUROC':>6} | {'TPR@FPR=1%':>11}")
    print(f"  {'─'*28}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*11}")

    cross_results = {}
    for held_out in methods_unique:
        train_mask = np.array([
            (i in benign_train_idx) or (labels[i] == 1 and methods_arr[i] != held_out)
            for i in range(len(all_prompts))
        ])
        test_mask = np.array([
            (i in benign_test_idx) or (labels[i] == 1 and methods_arr[i] == held_out)
            for i in range(len(all_prompts))
        ])

        # Training acts
        train_benign_acts = [all_acts[i] for i in range(len(all_prompts))
                             if train_mask[i] and labels[i] == 0]
        train_adv_acts = [all_acts[i] for i in range(len(all_prompts))
                          if train_mask[i] and labels[i] == 1]

        # Compute refusal directions on training data
        refusal_dirs = compute_refusal_directions(
            train_adv_acts, train_benign_acts, RTV_LAYERS
        )

        # Compute fingerprints for ALL prompts (train + test)
        all_fps = np.array([
            compute_fingerprint(ad, refusal_dirs, RTV_LAYERS, RTV_TOKEN_POSITIONS)
            for ad in all_acts
        ])

        # Fit Mahalanobis on TRAIN BENIGN only (the "legitimate" cluster)
        # Per RTV: legitimate = benign. Anomaly = attack.
        train_benign_fps = all_fps[train_mask & (labels == 0)]
        mu, prec = fit_mahalanobis(train_benign_fps)

        # Score test set
        test_fps = all_fps[test_mask]
        test_labels = labels[test_mask]
        test_scores = rtv_score(test_fps, mu, prec)

        # Metrics
        auroc = roc_auc_score(test_labels, test_scores)
        tpr, actual_fpr = tpr_at_fixed_fpr(test_scores, test_labels, 0.01)
        n_test_atk = int((test_labels == 1).sum())
        n_test_ben = int((test_labels == 0).sum())

        print(f"  {held_out:<28} | {n_test_atk:>6} | {auroc:>6.3f} | {tpr:>11.3f}")

        cross_results[held_out] = {
            "auroc": auroc,
            "tpr_at_fpr01": tpr,
            "actual_fpr": actual_fpr,
            "n_test_attacks": n_test_atk,
            "n_test_benign": n_test_ben,
        }

    mean_auroc = np.mean([r["auroc"] for r in cross_results.values()])
    mean_tpr = np.mean([r["tpr_at_fpr01"] for r in cross_results.values()])
    print(f"  {'─'*28}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*11}")
    print(f"  {'MEAN':<28} | {'':>6} | {mean_auroc:>6.3f} | {mean_tpr:>11.3f}")

    results["cross_attack"] = cross_results
    results["cross_attack_mean"] = {"auroc": mean_auroc, "tpr_at_fpr01": mean_tpr}

    # ══════════════════════════════════════════════════════════════════════
    #  COMPARISON WITH HPS RESULTS (from experiment8)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  COMPARISON WITH HPS / EUCLIDEAN / RAW (from experiment8)")
    print(f"{'─'*60}")
    print(f"  Cross-attack mean AUROC and TPR@FPR=1%:")
    print(f"\n  {'Method':<14} | {'AUROC':>6} | {'TPR@FPR=1%':>11}")
    print(f"  {'─'*14}─┼─{'─'*6}─┼─{'─'*11}")
    print(f"  {'Raw':<14} | {0.983:>6.3f} | {0.735:>11.3f}  (from exp8)")
    print(f"  {'Euclidean':<14} | {0.513:>6.3f} | {0.000:>11.3f}  (from exp8)")
    print(f"  {'Hyperbolic':<14} | {0.815:>6.3f} | {0.236:>11.3f}  (from exp8)")
    print(f"  {'RTV (this)':<14} | {mean_auroc:>6.3f} | {mean_tpr:>11.3f}")

    # ══════════════════════════════════════════════════════════════════════
    #  EMPIRICAL TEST: HPS's learned direction ≈ RTV's refusal direction?
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  EMPIRICAL TEST: HPS direction vs RTV refusal direction")
    print(f"{'─'*60}")
    print(f"  Theory (Fisher's LDA + contrastive loss):")
    print(f"    HPS's W matrix should learn a direction approximately aligned")
    print(f"    with the refusal direction r = mu_adversarial - mu_benign.")
    print(f"    Cosine similarity between W's principal direction and r")
    print(f"    measures empirical agreement with this theoretical claim.")

    # Compute global refusal direction (using ALL attacks vs ALL benign)
    # at each RTV layer
    global_refusal = compute_refusal_directions(
        adv_acts, benign_acts, RTV_LAYERS
    )

    # Try to load HPS-trained W from exp8 results / saved model
    hps_W_path = os.path.join(
        os.path.dirname(__file__),
        "results", "hps_full_model.pt"
    )

    direction_results = {}
    if os.path.exists(hps_W_path):
        try:
            ckpt = torch.load(hps_W_path, map_location="cpu")
            W = ckpt.get("W")
            if W is not None:
                # W: (d_proj, d_hidden) — principal singular direction
                if isinstance(W, torch.Tensor):
                    W_np = W.detach().cpu().numpy()
                else:
                    W_np = np.asarray(W)
                U, S, Vh = np.linalg.svd(W_np, full_matrices=False)
                # Principal right-singular vector = direction in INPUT space (d_hidden)
                # that gets stretched most by W.
                hps_principal = Vh[0]  # (d_hidden,)

                print(f"\n  Cosine similarity at each RTV layer:")
                print(f"    {'Layer':<8} | {'cos(W_principal, r_l)':>20}")
                print(f"    {'─'*8}─┼─{'─'*20}")
                for l in RTV_LAYERS:
                    r_l = global_refusal[l]
                    cos = float(
                        np.dot(hps_principal, r_l) /
                        (np.linalg.norm(hps_principal) * np.linalg.norm(r_l) + 1e-8)
                    )
                    direction_results[f"layer_{l}"] = cos
                    print(f"    {l:<8} | {abs(cos):>20.4f}")

                avg_cos = float(np.mean([abs(v) for v in direction_results.values()]))
                print(f"\n    Mean |cos|: {avg_cos:.4f}")
                if avg_cos > 0.7:
                    print(f"    → STRONG agreement: HPS learns approximately the refusal direction.")
                elif avg_cos > 0.4:
                    print(f"    → MODERATE agreement: HPS learns a related but distinct direction.")
                else:
                    print(f"    → WEAK agreement: HPS learns something fundamentally different.")
                direction_results["mean_abs_cos"] = avg_cos
            else:
                print(f"\n  No 'W' tensor found in {hps_W_path}")
        except Exception as e:
            print(f"\n  Could not load HPS model: {e}")
    else:
        print(f"\n  HPS model not found at {hps_W_path}")
        print(f"  (Run experiment6.py first to save it)")

    results["direction_comparison"] = direction_results

    # ── Save ──
    save_json(results, "experiment10_rtv.json", config.RESULTS_DIR)

    # ── Summary ──
    print(f"\n{'═'*60}")
    print(f"  RTV EVALUATION COMPLETE")
    print(f"{'═'*60}")
    print(f"  RTV cross-attack mean AUROC:        {mean_auroc:.3f}")
    print(f"  RTV cross-attack mean TPR@FPR=1%:   {mean_tpr:.3f}")
    print(f"\n  Comparison vs HPS (from exp8):")
    if mean_auroc > 0.815 + 0.05:
        print(f"  → RTV outperforms Hyperbolic (Δ AUROC = {mean_auroc - 0.815:+.3f})")
    elif mean_auroc < 0.815 - 0.05:
        print(f"  → Hyperbolic outperforms RTV (Δ AUROC = {0.815 - mean_auroc:+.3f})")
    else:
        print(f"  → RTV ≈ Hyperbolic (within 5% AUROC)")

    if mean_tpr > 0.236 + 0.05:
        print(f"  → RTV deployment beats Hyperbolic (Δ TPR = {mean_tpr - 0.236:+.3f})")
    elif mean_tpr < 0.236 - 0.05:
        print(f"  → Hyperbolic deployment beats RTV (Δ TPR = {0.236 - mean_tpr:+.3f})")
    else:
        print(f"  → RTV ≈ Hyperbolic at deployment (within 5% TPR)")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
