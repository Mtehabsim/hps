"""
diagnose_norm_confound.py — Investigate the source of the activation-norm
confound on the diverse-benign cache.

After norm_check_diverse.py revealed norm-only AUROC=1.000 on the diverse
cache (vs 0.917 on the original Alpaca cache), we need to find the cause.
The benign norms at deep layers dropped from 155.6 to 35.5, while attacks
stayed at 153.0. Possible causes:

  1. Chat template inconsistency — diverse benign may not be wrapped with
     apply_chat_template, while attacks WERE wrapped originally
  2. Tokenization domain shift — different prompt sources have different
     token distributions that propagate to activation magnitudes
  3. Attack-prompt structural patterns (DRAttack templates, IJP suffixes,
     GCG noise tokens) that drive norms regardless of semantic content

Hypothesis: chat template is the dominant cause. To test, re-extract a small
sample of benign + attack prompts WITH and WITHOUT chat template applied,
and compare the per-layer norms.

Outputs:
  results/norm_confound_diagnosis.json  — full per-prompt norm data
  results/figs/norm_confound_*.png      — diagnostic plots

Usage:
  # Quick diagnostic on 50 benign + 50 attack prompts (~10 min):
  python diagnose_norm_confound.py \\
      --model meta-llama/Meta-Llama-3-8B-Instruct \\
      --diverse_benign results/data_harmless_diverse.csv \\
      --attacks_json llama3_attacks_clean.json \\
      --n_samples 50

  # Larger diagnostic (~30 min):
  python diagnose_norm_confound.py --n_samples 200
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


LLAMA_LAYERS = [0, 2, 17, 24, 28, 31]
VICUNA_LAYERS = [0, 2, 22, 31, 35, 39]


# ---------------------------------------------------------------------------
# Sources to test
# ---------------------------------------------------------------------------

def load_diverse_benign_by_source(csv_path, n_per_source=10):
    """
    Try to load benign prompts STRATIFIED by source, so we can compare
    norms across source domains.

    Returns dict[source -> list[str]].
    """
    df = pd.read_csv(csv_path)

    if "source" in df.columns:
        sources = df["source"].dropna().unique()
        out = {}
        for src in sources:
            sub = df[df["source"] == src]
            col = "prompt" if "prompt" in df.columns else df.columns[0]
            prompts = sub[col].dropna().astype(str).tolist()[:n_per_source]
            out[str(src)] = prompts
        return out
    else:
        # No source column — return all as one bucket
        col = "prompt" if "prompt" in df.columns else df.columns[0]
        prompts = df[col].dropna().astype(str).tolist()[:n_per_source]
        return {"diverse_all": prompts}


def load_attacks_by_method(attacks_json, n_per_method=10):
    """Load attack prompts stratified by method."""
    with open(attacks_json) as f:
        data = json.load(f)

    if isinstance(data, dict):
        out = {}
        for method, prompts in data.items():
            out[method] = list(prompts[:n_per_method])
        return out
    elif isinstance(data, list):
        # List of dicts {prompt, method}
        out = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            m = item.get("method", "unknown")
            p = item.get("prompt", item.get("jailbreak_prompt"))
            if isinstance(p, str):
                out.setdefault(m, []).append(p)
        for m in out:
            out[m] = out[m][:n_per_method]
        return out
    else:
        raise ValueError(f"Unsupported attack JSON format")


# ---------------------------------------------------------------------------
# Activation extraction (with/without chat template)
# ---------------------------------------------------------------------------

def get_activations(model, tokenizer, prompts, layers, device,
                     use_chat_template, max_length=2048):
    """Forward each prompt; return per-layer last-token L2 norms."""
    norms_per_prompt = []  # list of (n_layers,) arrays

    model.eval()
    for prompt in prompts:
        with torch.no_grad():
            if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
                try:
                    text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    text = prompt
            else:
                text = prompt

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)

            outputs = model(**inputs, output_hidden_states=True)
            norms = []
            for l in layers:
                last_token_hs = outputs.hidden_states[l][0, -1]
                norms.append(float(last_token_hs.norm().item()))
            norms_per_prompt.append(np.array(norms, dtype=np.float32))

    return np.stack(norms_per_prompt, axis=0)  # (N, n_layers)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_norms_by_condition(results, output_path):
    """Per-layer norms, comparing 4 conditions:
       - benign with chat template
       - benign WITHOUT chat template
       - attack with chat template
       - attack WITHOUT chat template
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    layers = results["layers"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, group, title in [
        (axes[0], "with_template", "WITH chat template"),
        (axes[1], "without_template", "WITHOUT chat template"),
    ]:
        ben_means = np.mean([
            results[group]["benign_by_source"][src]
            for src in results[group]["benign_by_source"]
        ], axis=0).mean(axis=0)

        atk_means = np.mean([
            results[group]["attack_by_method"][m]
            for m in results[group]["attack_by_method"]
        ], axis=0).mean(axis=0)

        x = np.arange(len(layers))
        width = 0.35
        ax.bar(x - width / 2, ben_means, width, label="Benign",
                color="#3498db", alpha=0.85)
        ax.bar(x + width / 2, atk_means, width, label="Attack",
                color="#e74c3c", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in layers])
        ax.set_title(title)
        ax.set_ylabel("Mean L2 norm of last-token activation")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Activation norms with vs without chat template")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {output_path}")


def plot_per_source_breakdown(results, output_path):
    """Per-source benign norms (with template) vs per-method attack norms."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    layers = results["layers"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, group, title in [
        (axes[0], "with_template", "WITH chat template"),
        (axes[1], "without_template", "WITHOUT chat template"),
    ]:
        # Benign per source
        for src, norms in results[group]["benign_by_source"].items():
            arr = np.array(norms)
            ax.plot(layers, arr.mean(axis=0), marker="o", alpha=0.7,
                    label=f"benign:{src}")

        # Attack per method
        for m, norms in results[group]["attack_by_method"].items():
            arr = np.array(norms)
            ax.plot(layers, arr.mean(axis=0), marker="s", alpha=0.7,
                    linestyle="--", label=f"attack:{m}")

        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean L2 norm")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="upper left", ncol=2)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {output_path}")


def plot_diagnostic_summary(results, output_path):
    """The key diagnostic: per-layer benign-vs-attack norm gap, with-vs-
       without template."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    layers = results["layers"]

    diffs = {}
    for group in ["with_template", "without_template"]:
        ben_all = np.concatenate([
            np.array(results[group]["benign_by_source"][src])
            for src in results[group]["benign_by_source"]
        ])
        atk_all = np.concatenate([
            np.array(results[group]["attack_by_method"][m])
            for m in results[group]["attack_by_method"]
        ])
        diffs[group] = atk_all.mean(axis=0) - ben_all.mean(axis=0)

    x = np.arange(len(layers))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, diffs["with_template"], width,
           label="WITH chat template", color="#9b59b6", alpha=0.85)
    ax.bar(x + width / 2, diffs["without_template"], width,
           label="WITHOUT chat template", color="#f39c12", alpha=0.85)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Attack norm − Benign norm (L2)")
    ax.set_title("Diagnostic: norm gap between attack and benign\n"
                 "(positive = attacks have larger norm = norm confound)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--diverse_benign",
                        default="results/data_harmless_diverse.csv")
    parser.add_argument("--attacks_json", default="llama3_attacks_clean.json")
    parser.add_argument("--n_samples", type=int, default=50,
                        help="Number of prompts per source/method")
    parser.add_argument("--layers", type=int, nargs="+", default=LLAMA_LAYERS)
    parser.add_argument("--output",
                        default="results/norm_confound_diagnosis.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    figs_dir = "results/figs"
    os.makedirs(figs_dir, exist_ok=True)

    print("=" * 70)
    print("NORM CONFOUND DIAGNOSIS")
    print("=" * 70)
    print(f"  Model:          {args.model}")
    print(f"  Diverse benign: {args.diverse_benign}")
    print(f"  Attacks JSON:   {args.attacks_json}")
    print(f"  N per source:   {args.n_samples}")
    print(f"  Layers:         {args.layers}")
    print(f"  Device:         {device}")
    print()

    # Load data
    print(f"Loading diverse benign by source...")
    benign_by_source = load_diverse_benign_by_source(
        args.diverse_benign, n_per_source=args.n_samples,
    )
    for src, prompts in benign_by_source.items():
        print(f"  {src}: {len(prompts)} prompts")

    print(f"\nLoading attacks by method...")
    attacks_by_method = load_attacks_by_method(
        args.attacks_json, n_per_method=args.n_samples,
    )
    for m, prompts in attacks_by_method.items():
        print(f"  {m}: {len(prompts)} prompts")

    # Load model
    print(f"\nLoading model {args.model}...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Loaded. Hidden: {model.config.hidden_size}")

    # Sanity check chat template
    has_template = (hasattr(tokenizer, "apply_chat_template")
                    and tokenizer.chat_template is not None)
    print(f"\n  Has chat template: {has_template}")
    if has_template:
        sample_in = tokenizer.apply_chat_template(
            [{"role": "user", "content": "test"}],
            tokenize=False, add_generation_prompt=True,
        )
        print(f"  Sample wrapped: {repr(sample_in[:200])}")

    # Run all 4 conditions
    print("\n" + "=" * 70)
    print("Extracting norms (4 conditions)")
    print("=" * 70)

    results = {
        "model": args.model,
        "layers": args.layers,
        "with_template": {"benign_by_source": {}, "attack_by_method": {}},
        "without_template": {"benign_by_source": {}, "attack_by_method": {}},
    }

    for use_template, group in [(True, "with_template"),
                                  (False, "without_template")]:
        print(f"\n  Group: {group}")

        # Benign per source
        for src, prompts in benign_by_source.items():
            t0 = time.time()
            norms = get_activations(
                model, tokenizer, prompts, args.layers, device,
                use_chat_template=use_template, max_length=args.max_length,
            )
            results[group]["benign_by_source"][src] = norms.tolist()
            avg = norms.mean(axis=0)
            print(f"    benign/{src} ({len(prompts)} prompts, "
                  f"{time.time()-t0:.1f}s): "
                  f"L{args.layers[-1]}={avg[-1]:.1f}")

        # Attack per method
        for m, prompts in attacks_by_method.items():
            t0 = time.time()
            norms = get_activations(
                model, tokenizer, prompts, args.layers, device,
                use_chat_template=use_template, max_length=args.max_length,
            )
            results[group]["attack_by_method"][m] = norms.tolist()
            avg = norms.mean(axis=0)
            print(f"    attack/{m} ({len(prompts)} prompts, "
                  f"{time.time()-t0:.1f}s): "
                  f"L{args.layers[-1]}={avg[-1]:.1f}")

    # ---------------------------------------------------------------------
    # Diagnostic computation
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    diagnosis = {}
    for group in ["with_template", "without_template"]:
        ben_all = np.concatenate([
            np.array(results[group]["benign_by_source"][src])
            for src in results[group]["benign_by_source"]
        ])
        atk_all = np.concatenate([
            np.array(results[group]["attack_by_method"][m])
            for m in results[group]["attack_by_method"]
        ])

        ben_mean_per_layer = ben_all.mean(axis=0)
        atk_mean_per_layer = atk_all.mean(axis=0)
        diff = atk_mean_per_layer - ben_mean_per_layer

        # Norm-only AUROC at final layer
        from sklearn.metrics import roc_auc_score
        scores_final = np.concatenate([ben_all[:, -1], atk_all[:, -1]])
        labels = np.array([0] * len(ben_all) + [1] * len(atk_all))
        auroc_final = roc_auc_score(labels, scores_final)

        # Multi-layer norm-only AUROC (use all 6 norms)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        all_norms = np.concatenate([ben_all, atk_all], axis=0)
        sc = StandardScaler()
        clf = LogisticRegression(max_iter=2000, random_state=42)
        clf.fit(sc.fit_transform(all_norms), labels)
        scores_multi = clf.predict_proba(sc.transform(all_norms))[:, 1]
        auroc_multi = roc_auc_score(labels, scores_multi)

        diagnosis[group] = {
            "ben_mean_per_layer": ben_mean_per_layer.tolist(),
            "atk_mean_per_layer": atk_mean_per_layer.tolist(),
            "diff_per_layer": diff.tolist(),
            "norm_only_auroc_final_layer": float(auroc_final),
            "norm_only_auroc_multi_layer": float(auroc_multi),
        }

        print(f"\n  {group.upper()}:")
        print(f"    Layer | Benign | Attack | Diff")
        for i, l in enumerate(args.layers):
            print(f"    {l:5d} | {ben_mean_per_layer[i]:6.2f} | "
                  f"{atk_mean_per_layer[i]:6.2f} | "
                  f"{diff[i]:+7.2f}")
        print(f"    Norm-only AUROC (final layer):    {auroc_final:.4f}")
        print(f"    Norm-only AUROC (all 6 layers):   {auroc_multi:.4f}")

    results["diagnosis"] = diagnosis

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    auroc_with = diagnosis["with_template"]["norm_only_auroc_multi_layer"]
    auroc_without = diagnosis["without_template"]["norm_only_auroc_multi_layer"]

    if auroc_with < 0.7 and auroc_without > 0.9:
        verdict = "CHAT_TEMPLATE_IS_CAUSE"
        explanation = (
            "Norm confound disappears with chat template applied uniformly. "
            "Re-extract activations with --use_chat_template enforced for "
            "ALL prompt types, including diverse benign sources."
        )
    elif auroc_with > 0.9 and auroc_without > 0.9:
        verdict = "FUNDAMENTAL_NORM_CONFOUND"
        explanation = (
            "Norm confound persists regardless of chat template. The attacks "
            "have systematically different activation magnitudes than benign "
            "prompts, independent of formatting. This is a methodology issue "
            "in the entire field — paper must reframe as a 2-confound critique."
        )
    elif auroc_with < 0.7 and auroc_without < 0.7:
        verdict = "FALSE_ALARM"
        explanation = (
            "Norm confound is small in both conditions. The earlier "
            "AUROC=1.000 may have been an artifact of larger sample size. "
            "Re-run norm_check_diverse.py with these prompts."
        )
    else:
        verdict = "PARTIAL"
        explanation = (
            f"Mixed: with-template AUROC {auroc_with:.4f}, "
            f"without-template AUROC {auroc_without:.4f}. "
            "Chat template is a partial cause but not the whole story."
        )

    diagnosis["verdict"] = verdict
    diagnosis["verdict_explanation"] = explanation

    print(f"\n  Verdict: {verdict}")
    print(f"  {explanation}")

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {args.output}")

    # Plots
    plot_norms_by_condition(results,
                              os.path.join(figs_dir, "norm_confound_summary.png"))
    plot_per_source_breakdown(results,
                              os.path.join(figs_dir,
                                          "norm_confound_per_source.png"))
    plot_diagnostic_summary(results,
                              os.path.join(figs_dir, "norm_confound_gap.png"))


if __name__ == "__main__":
    main()
