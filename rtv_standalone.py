"""
RTV — Representation Trajectory Verification (Derya & Sunar, arXiv:2605.03095)
═══════════════════════════════════════════════════════════════════════════════
Faithful reimplementation based on Section 7 of the paper.

Key design (from paper):
  - Refusal direction: r_l = mean(harmful_l) - mean(harmless_l)  [Eq. 6]
    NOTE: "harmful" = raw harmful QUERIES (e.g. HarmBench), NOT refused outputs.
  - Fingerprint: F[l,p] = cos(r_l, h_l^(p))  [Eq. 11]
    K=3 layers, P=5 token positions → 15-dim fingerprint
  - Calibration: Fit Mahalanobis (LedoitWolf) on BOTH harmless AND harmful
  - Detection: M(x) = min(d_harmless(x), d_harmful(x))  [Eq. 13]
    Flag when M(x) > T (95th percentile of calibration scores)

Layer selection (model-specific):
  - Llama-3-8B (32 layers): {18, 25, 32}
  - Vicuna-13B (40 layers): {22, 31, 39} (proportional mapping)

Data sources (from JBShield repo):
  - Harmless: data/harmless_calibration.csv (Alpaca)
  - Harmful: data/harmful_calibration.csv (AdvBench/HexPHI)

Usage:
  python rtv_standalone.py --model lmsys/vicuna-13b-v1.5 \
                           --harmless data/harmless_calibration.csv \
                           --harmful data/harmful_calibration.csv \
                           --test-attacks results/validated_attacks_categorized.json
"""

import argparse
import os
import json
import numpy as np
import torch
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

MODEL_LAYERS = {
    "lmsys/vicuna-13b-v1.5": [12, 16, 26],      # empirically discovered (uniform high separation)
    "meta-llama/Meta-Llama-3-8B-Instruct": [18, 25, 32],  # paper's exact layers
    "lmsys/vicuna-7b-v1.5": [14, 19, 25],       # 32 layers, same proportions
}

TOKEN_POSITIONS = [-1, -2, -3, -4, -5]  # last 5 tokens
FPR_TARGET = 0.05  # paper uses 95th percentile → 5% FPR


# ═══════════════════════════════════════════════════════════════════════════
#  Model Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_model(model_name, device="cuda"):
    """Load model with 4-bit quantization if on GPU."""
    print(f"[RTV] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda" and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map="auto"
        )
    model.eval()
    print(f"[RTV] Model loaded.")
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════
#  Hidden State Extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_hidden_states(model, tokenizer, prompt, layers, device="cuda"):
    """Extract hidden states at specified layers for all token positions.
    Returns: dict[layer] → numpy array (seq_len, d_hidden)
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # tuple of (1, seq_len, d_hidden)
    result = {}
    for l in layers:
        # hidden_states[0] is embedding, hidden_states[1] is layer 0, etc.
        # So hidden_states[l+1] is layer l's output
        if l + 1 < len(hidden_states):
            result[l] = hidden_states[l + 1][0].detach().cpu().float().numpy()
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  RTV Core (per paper Section 7.2)
# ═══════════════════════════════════════════════════════════════════════════

def compute_refusal_directions(model, tokenizer, harmful_prompts, harmless_prompts,
                               layers, n_samples=100):
    """Eq. 6: r_l = mean(h_l(harmful)) - mean(h_l(harmless)), normalized.
    Uses LAST TOKEN position for direction extraction (per paper Section 3.3).
    """
    n = min(n_samples, len(harmful_prompts), len(harmless_prompts))
    print(f"[RTV] Computing refusal directions from {n} harmful + {n} harmless...")

    harmful_acts = {l: [] for l in layers}
    for i in range(n):
        hs = extract_hidden_states(model, tokenizer, harmful_prompts[i], layers)
        for l in layers:
            harmful_acts[l].append(hs[l][-1])  # last token
        if (i + 1) % 20 == 0:
            print(f"  Harmful: {i+1}/{n}")

    harmless_acts = {l: [] for l in layers}
    for i in range(n):
        hs = extract_hidden_states(model, tokenizer, harmless_prompts[i], layers)
        for l in layers:
            harmless_acts[l].append(hs[l][-1])  # last token
        if (i + 1) % 20 == 0:
            print(f"  Harmless: {i+1}/{n}")

    refusal_dirs = {}
    for l in layers:
        mu_h = np.mean(harmful_acts[l], axis=0)
        mu_b = np.mean(harmless_acts[l], axis=0)
        r = mu_h - mu_b
        refusal_dirs[l] = r / (np.linalg.norm(r) + 1e-8)

    return refusal_dirs


def compute_fingerprint(hidden_states, refusal_dirs, layers, positions):
    """Eq. 11: F[l,p] = cos(r_l, h_l^(p)). Returns flat 15-dim vector."""
    fp = []
    for l in layers:
        r = refusal_dirs[l]
        h_seq = hidden_states[l]  # (seq_len, d_hidden)
        seq_len = h_seq.shape[0]
        for p in positions:
            pos = seq_len + p if p < 0 else p
            pos = max(0, min(pos, seq_len - 1))
            h = h_seq[pos]
            cos = float(np.dot(h, r) / (np.linalg.norm(h) * np.linalg.norm(r) + 1e-8))
            fp.append(cos)
    return np.array(fp)


def calibrate_rtv(fingerprints_harmless, fingerprints_harmful):
    """Fit Mahalanobis parameters for BOTH clusters (per paper Eq. 12-13).
    Returns: (mu_pos, prec_pos, mu_neg, prec_neg, threshold)
    """
    # Harmless cluster
    lw_pos = LedoitWolf().fit(fingerprints_harmless)
    mu_pos = lw_pos.location_
    prec_pos = lw_pos.precision_

    # Harmful cluster
    lw_neg = LedoitWolf().fit(fingerprints_harmful)
    mu_neg = lw_neg.location_
    prec_neg = lw_neg.precision_

    # Threshold: 95th percentile of M(x) on calibration data (combined)
    # M(x) = min(d_pos, d_neg) — Eq. 13
    all_fps = np.vstack([fingerprints_harmless, fingerprints_harmful])
    scores = []
    for fp in all_fps:
        d_pos = np.sqrt((fp - mu_pos) @ prec_pos @ (fp - mu_pos))
        d_neg = np.sqrt((fp - mu_neg) @ prec_neg @ (fp - mu_neg))
        scores.append(min(d_pos, d_neg))
    scores = np.array(scores)
    threshold = float(np.quantile(scores, 1.0 - FPR_TARGET))

    print(f"[RTV] Calibration complete:")
    print(f"  Harmless median M(x): {np.median(scores[:len(fingerprints_harmless)]):.3f}")
    print(f"  Harmful median M(x):  {np.median(scores[len(fingerprints_harmless):]):.3f}")
    print(f"  Threshold (95th pct): {threshold:.3f}")

    return mu_pos, prec_pos, mu_neg, prec_neg, threshold


def rtv_score(fp, mu_pos, prec_pos, mu_neg, prec_neg):
    """Eq. 13: M(x) = min(d_harmless, d_harmful)."""
    d_pos = np.sqrt(max(0, (fp - mu_pos) @ prec_pos @ (fp - mu_pos)))
    d_neg = np.sqrt(max(0, (fp - mu_neg) @ prec_neg @ (fp - mu_neg)))
    return min(d_pos, d_neg)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RTV Detector (Derya & Sunar 2026)")
    parser.add_argument("--model", default="lmsys/vicuna-13b-v1.5")
    parser.add_argument("--harmless", required=True, help="CSV with harmless prompts")
    parser.add_argument("--harmful", required=True, help="CSV with harmful prompts")
    parser.add_argument("--test-attacks", default=None, help="JSON with attack prompts")
    parser.add_argument("--test-benign", default=None, help="CSV with test benign prompts")
    parser.add_argument("--n-cal", type=int, default=100, help="Calibration samples per class")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="rtv_results.json")
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"  RTV — Representation Trajectory Verification")
    print(f"  Model: {args.model}")
    print(f"{'═'*60}\n")

    # ── Determine layers ──
    layers = MODEL_LAYERS.get(args.model)
    if layers is None:
        # Default: proportional mapping assuming refusal at ~56%, 78%, 100% depth
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model)
        n_layers = cfg.num_hidden_layers
        layers = [int(n_layers * 0.56), int(n_layers * 0.78), n_layers - 1]
        print(f"[RTV] Auto-selected layers: {layers} (for {n_layers}-layer model)")
    else:
        print(f"[RTV] Using paper layers: {layers}")

    # ── Load data ──
    print(f"[RTV] Loading calibration data...")
    df_harmless = pd.read_csv(args.harmless)
    df_harmful = pd.read_csv(args.harmful)

    # Try common column names
    for col in ["prompt", "goal", "text", "instruction", "query"]:
        if col in df_harmless.columns:
            harmless_prompts = df_harmless[col].dropna().tolist()
            break
    else:
        harmless_prompts = df_harmless.iloc[:, 0].dropna().tolist()

    for col in ["prompt", "goal", "text", "instruction", "query"]:
        if col in df_harmful.columns:
            harmful_prompts = df_harmful[col].dropna().tolist()
            break
    else:
        harmful_prompts = df_harmful.iloc[:, 0].dropna().tolist()

    print(f"  Harmless: {len(harmless_prompts)} prompts")
    print(f"  Harmful:  {len(harmful_prompts)} prompts")

    n_cal = min(args.n_cal, len(harmless_prompts), len(harmful_prompts))

    # ── Load model ──
    model, tokenizer = load_model(args.model, args.device)

    # ── Compute refusal directions ──
    refusal_dirs = compute_refusal_directions(
        model, tokenizer, harmful_prompts[:n_cal], harmless_prompts[:n_cal], layers, n_cal
    )

    # ── Extract calibration fingerprints ──
    print(f"\n[RTV] Extracting calibration fingerprints...")
    fps_harmless = []
    for i in range(n_cal):
        hs = extract_hidden_states(model, tokenizer, harmless_prompts[i], layers)
        fps_harmless.append(compute_fingerprint(hs, refusal_dirs, layers, TOKEN_POSITIONS))
        if (i + 1) % 20 == 0:
            print(f"  Harmless fingerprints: {i+1}/{n_cal}")
    fps_harmless = np.array(fps_harmless)

    fps_harmful = []
    for i in range(n_cal):
        hs = extract_hidden_states(model, tokenizer, harmful_prompts[i], layers)
        fps_harmful.append(compute_fingerprint(hs, refusal_dirs, layers, TOKEN_POSITIONS))
        if (i + 1) % 20 == 0:
            print(f"  Harmful fingerprints: {i+1}/{n_cal}")
    fps_harmful = np.array(fps_harmful)

    # ── Calibrate ──
    mu_pos, prec_pos, mu_neg, prec_neg, threshold = calibrate_rtv(fps_harmless, fps_harmful)

    results = {
        "model": args.model,
        "layers": layers,
        "n_calibration": n_cal,
        "threshold": float(threshold),
        "fingerprint_dim": len(layers) * len(TOKEN_POSITIONS),
    }

    # ── Test on attacks ──
    fps_attacks = None
    if args.test_attacks:
        print(f"\n[RTV] Evaluating on attack prompts...")
        if args.test_attacks.endswith(".json"):
            with open(args.test_attacks) as f:
                data = json.load(f)
            if isinstance(data, dict):
                attack_prompts = []
                attack_methods = []
                for method, prompts in data.items():
                    for p in prompts:
                        attack_prompts.append(p)
                        attack_methods.append(method)
            else:
                attack_prompts = data
                attack_methods = ["unknown"] * len(data)
        else:
            df_atk = pd.read_csv(args.test_attacks)
            attack_prompts = df_atk.iloc[:, 0].dropna().tolist()
            attack_methods = ["unknown"] * len(attack_prompts)

        print(f"  {len(attack_prompts)} attack prompts")
        attack_scores = []
        fps_attack_list = []
        for i, p in enumerate(attack_prompts):
            hs = extract_hidden_states(model, tokenizer, p, layers)
            fp = compute_fingerprint(hs, refusal_dirs, layers, TOKEN_POSITIONS)
            s = rtv_score(fp, mu_pos, prec_pos, mu_neg, prec_neg)
            attack_scores.append(s)
            fps_attack_list.append(fp)
            if (i + 1) % 50 == 0:
                print(f"  Attacks scored: {i+1}/{len(attack_prompts)}")

        attack_scores = np.array(attack_scores)
        fps_attacks = np.array(fps_attack_list)
        tpr = float((attack_scores > threshold).mean())
        print(f"\n  Attack detection (TPR @ {FPR_TARGET*100:.0f}% FPR): {tpr:.3f}")
        print(f"  Attack median M(x): {np.median(attack_scores):.3f}")
        print(f"  Attack mean M(x):   {np.mean(attack_scores):.3f}")

        results["attack_detection"] = {
            "n_attacks": len(attack_prompts),
            "tpr": float(tpr),
            "median_score": float(np.median(attack_scores)),
            "mean_score": float(np.mean(attack_scores)),
        }

        # AUROC if we have both benign and attack scores
        # Use calibration harmless scores as "benign test" for AUROC
        cal_scores = np.array([rtv_score(fp, mu_pos, prec_pos, mu_neg, prec_neg)
                               for fp in fps_harmless])
        y_true = np.array([0] * len(cal_scores) + [1] * len(attack_scores))
        all_scores = np.concatenate([cal_scores, attack_scores])
        auroc = roc_auc_score(y_true, all_scores)
        print(f"  AUROC (attacks vs harmless): {auroc:.3f}")
        results["auroc"] = float(auroc)

        # Per-method breakdown
        methods_unique = sorted(set(attack_methods))
        if len(methods_unique) > 1:
            print(f"\n  Per-attack-type breakdown:")
            print(f"  {'Method':<30} | {'N':>4} | {'TPR@5%':>7} | {'Median M(x)':>11} | {'Mean M(x)':>10}")
            print(f"  {'─'*30}─┼─{'─'*4}─┼─{'─'*7}─┼─{'─'*11}─┼─{'─'*10}")
            per_method = {}
            for method in methods_unique:
                idx = [i for i, m in enumerate(attack_methods) if m == method]
                m_scores = attack_scores[idx]
                m_tpr = float((m_scores > threshold).mean())
                m_median = float(np.median(m_scores))
                m_mean = float(np.mean(m_scores))
                print(f"  {method:<30} | {len(idx):>4} | {m_tpr:>7.3f} | {m_median:>11.3f} | {m_mean:>10.3f}")
                per_method[method] = {"n": len(idx), "tpr": m_tpr, "median": m_median, "mean": m_mean}
            results["per_method"] = per_method

    # ── Test on benign (optional) ──
    if args.test_benign:
        print(f"\n[RTV] Evaluating on test benign...")
        df_ben = pd.read_csv(args.test_benign)
        for col in ["prompt", "goal", "text", "instruction", "query"]:
            if col in df_ben.columns:
                test_benign = df_ben[col].dropna().tolist()
                break
        else:
            test_benign = df_ben.iloc[:, 0].dropna().tolist()

        benign_scores = []
        for i, p in enumerate(test_benign):
            hs = extract_hidden_states(model, tokenizer, p, layers)
            fp = compute_fingerprint(hs, refusal_dirs, layers, TOKEN_POSITIONS)
            s = rtv_score(fp, mu_pos, prec_pos, mu_neg, prec_neg)
            benign_scores.append(s)
        benign_scores = np.array(benign_scores)
        fpr = float((benign_scores > threshold).mean())
        print(f"  Benign FPR: {fpr:.3f}")
        results["benign_fpr"] = float(fpr)

    # ── Save ──
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[RTV] Results saved → {args.output}")

    # ── Visualize clusters ──
    print(f"[RTV] Generating cluster visualization...")
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    all_fps = [fps_harmless, fps_harmful]
    labels_viz = ['Harmless', 'Harmful']
    colors = ['#2ecc71', '#e74c3c']

    if args.test_attacks and fps_attacks is not None:
        all_fps.append(fps_attacks)
        labels_viz.append('Attacks')
        colors.append('#9b59b6')

    X_all = np.vstack(all_fps)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_all)

    fig, ax = plt.subplots(figsize=(10, 8))
    offset = 0
    for i, (label, color) in enumerate(zip(labels_viz, colors)):
        n = len(all_fps[i])
        ax.scatter(X_pca[offset:offset+n, 0], X_pca[offset:offset+n, 1],
                   c=color, label=label, alpha=0.6, s=40, edgecolors='w', linewidth=0.3)
        offset += n

    ax.set_title(f"RTV Fingerprint Space (PCA) — Layers {layers}\n"
                 f"AUROC={results.get('auroc', 'N/A'):.3f}", fontsize=12)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plot_path = args.output.replace('.json', '_clusters.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[RTV] Cluster plot saved → {plot_path}")

    print(f"\n{'═'*60}")
    print(f"  RTV EVALUATION COMPLETE")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
