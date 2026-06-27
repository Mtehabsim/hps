"""
extract_llama3_base_activations.py — Direct alignment ablation.

Extract activations using the BASE Llama-3-8B (no SFT, no RLHF) instead of
Llama-3-8B-Instruct. Same architecture, same tokenizer, same diverse benign,
same attacks. Only difference: alignment training.

This is the cleanest possible test of the alignment-mediated hypothesis:
  - If HPS detection drops sharply on base: alignment concentrates harm
    features into the geometry. The geometric prior depends on RLHF.
  - If HPS stays at TPR=1.000 on base: harm features are present in
    pretrained representations; alignment doesn't introduce them.
  - If HPS drops similarly to C4: the gap is alignment-strength independent.

Critical implementation details:
  - Base model has NO chat template — pass raw prompts directly
  - Same layer indices [0, 2, 17, 24, 28, 31] for direct comparison
  - Same diverse benign (data_harmless_diverse.csv) and same attacks JSON
  - Output cache format matches Llama-3 dict-style for compatibility with
    downstream scripts (verify_saturation, statistical_tests, etc.)

Usage on DGX (~3-4 hours GPU):
  python extract_llama3_base_activations.py \\
      --model meta-llama/Meta-Llama-3-8B \\
      --diverse_benign results/data_harmless_diverse.csv \\
      --attacks_json llama3_attacks_clean.json \\
      --output results/llama3_BASE_activations_cache_diverse.npz \\
      --layers 0 2 17 24 28 31

  # Compare to Instruct model (already extracted as llama3_activations_cache_diverse.npz):
  python statistical_tests.py --cache results/llama3_BASE_activations_cache_diverse.npz
  python verify_saturation.py --llama3_cache results/llama3_BASE_activations_cache_diverse.npz
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


# ---------------------------------------------------------------------------
# Activation extraction (no chat template — raw prompts only)
# ---------------------------------------------------------------------------

def extract_activations(model, tokenizer, prompts, layers, device,
                         max_length=2048, store_full_sequence=True,
                         use_chat_template=False, verbose=True):
    """
    Forward each prompt through model; capture hidden states at specified
    layers.

    Args:
        use_chat_template: If True, wrap prompt in chat template (for Instruct
            models). If False, pass raw prompt (for base models).
    """
    results = []
    n = len(prompts)
    t_start = time.time()

    model.eval()
    for i, prompt in enumerate(prompts):
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

            try:
                outputs = model(**inputs, output_hidden_states=True)
            except Exception as e:
                print(f"  ERROR on prompt {i}: {e}")
                results.append({l: np.zeros(model.config.hidden_size,
                                              dtype=np.float32)
                                for l in layers})
                continue

            hs = {}
            for l in layers:
                tensor = outputs.hidden_states[l][0]
                if store_full_sequence:
                    hs[l] = tensor.cpu().numpy().astype(np.float32)
                else:
                    hs[l] = tensor[-1].cpu().numpy().astype(np.float32)
            results.append(hs)

        if verbose and (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(f"    [{i+1}/{n}] {rate:.1f} prompts/sec, "
                  f"ETA {eta/60:.1f} min")

    if verbose:
        elapsed = time.time() - t_start
        print(f"    Done: {n} prompts in {elapsed/60:.1f} min "
              f"({n/elapsed:.1f} prompts/sec)")
    return results


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_attacks(attacks_json):
    """Load attack prompts. Supports two formats:
       (a) dict {attack_name: [prompt, ...]} -> return ([all_prompts],
           [all_methods])
       (b) list[{"prompt": ..., "method": ...}] -> ditto
    """
    with open(attacks_json) as f:
        data = json.load(f)

    flat = []
    methods = []
    if isinstance(data, dict):
        for attack_name, prompts in data.items():
            for p in prompts:
                if isinstance(p, str) and len(p.strip()) >= 5:
                    flat.append(p)
                    methods.append(attack_name)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                p = item.get("prompt", item.get("jailbreak_prompt"))
                m = item.get("method", item.get("attack", "unknown"))
                if isinstance(p, str) and len(p.strip()) >= 5:
                    flat.append(p)
                    methods.append(m)
            elif isinstance(item, str):
                flat.append(item)
                methods.append("unknown")
    else:
        raise ValueError(f"Unsupported attack JSON format in {attacks_json}")

    return flat, methods


def load_diverse_benign(csv_path, n=None, seed=42):
    """Load diverse benign prompts; subsample to n if specified."""
    df = pd.read_csv(csv_path)
    if "prompt" not in df.columns:
        prompts = df.iloc[:, 0].dropna().astype(str).tolist()
    else:
        prompts = df["prompt"].dropna().astype(str).tolist()

    if n is None or n > len(prompts):
        n = len(prompts)

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(prompts))[:n]
    return [prompts[i] for i in idx]


# ---------------------------------------------------------------------------
# Cache savers
# ---------------------------------------------------------------------------

def save_dict_cache(output_path, hs_train_ben, hs_train_atk,
                     hs_test_ben, hs_test_atk, cfg_hash="base_diverse"):
    np.savez(
        output_path,
        hs_train_ben=np.array(hs_train_ben, dtype=object),
        hs_train_atk=np.array(hs_train_atk, dtype=object),
        hs_test_ben=np.array(hs_test_ben, dtype=object),
        hs_test_atk=np.array(hs_test_atk, dtype=object),
        cfg_hash=np.array(cfg_hash),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B",
                        help="HF model name (default: base Llama-3-8B)")
    parser.add_argument("--diverse_benign",
                        default="results/data_harmless_diverse.csv")
    parser.add_argument("--attacks_json", default="llama3_attacks_clean.json",
                        help="Attack JSON (use the cleaned/dedup version)")
    parser.add_argument("--output", required=True,
                        help="Output cache path (.npz)")
    parser.add_argument("--layers", type=int, nargs="+",
                        default=[0, 2, 17, 24, 28, 31],
                        help="Layer indices (same as Instruct for fairness)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_chat_template", action="store_true",
                        help="Apply chat template to prompts. Default OFF for "
                             "base models. Set ON only when extracting "
                             "Instruct activations as control.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("LLAMA-3 BASE ACTIVATION EXTRACTION (Alignment Ablation)")
    print("=" * 70)
    print(f"  Model:              {args.model}")
    print(f"  Diverse benign:     {args.diverse_benign}")
    print(f"  Attacks JSON:       {args.attacks_json}")
    print(f"  Output:             {args.output}")
    print(f"  Layers:             {args.layers}")
    print(f"  Use chat template:  {args.use_chat_template}")
    print(f"  Device:             {device}")
    print()

    if not args.use_chat_template:
        print("  NOTE: chat template OFF — passing raw prompts.")
        print("  This is correct for BASE models (no SFT/RLHF).")
        print("  For Instruct models as control, add --use_chat_template.")
        print()

    # Sanity checks
    for path in [args.diverse_benign, args.attacks_json]:
        if not os.path.exists(path):
            print(f"ERROR: {path} not found")
            sys.exit(1)

    # Load attacks
    print(f"Loading attacks from {args.attacks_json}...")
    atk_prompts, atk_methods = load_attacks(args.attacks_json)
    print(f"  {len(atk_prompts)} attacks across "
          f"{len(set(atk_methods))} methods")
    method_counts = {m: atk_methods.count(m) for m in sorted(set(atk_methods))}
    for m, n in method_counts.items():
        print(f"    {m}: {n}")

    # Train/test split (same seed/proportion as our existing cache)
    rng = np.random.RandomState(args.seed)
    a_idx = rng.permutation(len(atk_prompts))
    n_atk_train = int(args.train_frac * len(a_idx))
    train_atk = [atk_prompts[i] for i in a_idx[:n_atk_train]]
    test_atk = [atk_prompts[i] for i in a_idx[n_atk_train:]]
    test_atk_methods = [atk_methods[i] for i in a_idx[n_atk_train:]]
    print(f"  Train: {len(train_atk)}, Test: {len(test_atk)}")

    # Load benign
    print(f"\nLoading benign from {args.diverse_benign}...")
    n_ben = max(len(atk_prompts), 5000)
    ben_prompts = load_diverse_benign(args.diverse_benign, n=n_ben,
                                       seed=args.seed)
    n_ben_train = int(args.train_frac * len(ben_prompts))
    train_ben = ben_prompts[:n_ben_train]
    test_ben = ben_prompts[n_ben_train:]
    print(f"  {len(ben_prompts)} benign  "
          f"(train {len(train_ben)}, test {len(test_ben)})")

    # Load model
    print(f"\nLoading model: {args.model}")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed")
        sys.exit(1)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except Exception as e:
        print(f"ERROR loading tokenizer: {e}")
        print("If gated repo: huggingface-cli login")
        sys.exit(1)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("  Loading weights...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Loaded. Hidden: {model.config.hidden_size}, "
          f"Layers: {model.config.num_hidden_layers}")

    n_layers_max = model.config.num_hidden_layers + 1
    if max(args.layers) >= n_layers_max:
        print(f"ERROR: requested layer {max(args.layers)} >= {n_layers_max}")
        sys.exit(1)

    # Extract
    use_template = args.use_chat_template
    print(f"\nExtracting train benign ({len(train_ben)})...")
    hs_train_ben = extract_activations(
        model, tokenizer, train_ben, args.layers, device,
        max_length=args.max_length, store_full_sequence=True,
        use_chat_template=use_template,
    )

    print(f"\nExtracting test benign ({len(test_ben)})...")
    hs_test_ben = extract_activations(
        model, tokenizer, test_ben, args.layers, device,
        max_length=args.max_length, store_full_sequence=True,
        use_chat_template=use_template,
    )

    print(f"\nExtracting train attacks ({len(train_atk)})...")
    hs_train_atk = extract_activations(
        model, tokenizer, train_atk, args.layers, device,
        max_length=args.max_length, store_full_sequence=True,
        use_chat_template=use_template,
    )

    print(f"\nExtracting test attacks ({len(test_atk)})...")
    hs_test_atk = extract_activations(
        model, tokenizer, test_atk, args.layers, device,
        max_length=args.max_length, store_full_sequence=True,
        use_chat_template=use_template,
    )

    # Save
    print(f"\nSaving to {args.output}...")
    cfg_hash = "base_template" if use_template else "base_no_template"
    save_dict_cache(args.output, hs_train_ben, hs_train_atk,
                     hs_test_ben, hs_test_atk, cfg_hash=cfg_hash)
    print(f"  train: {len(hs_train_ben)} ben + {len(hs_train_atk)} atk")
    print(f"  test:  {len(hs_test_ben)} ben + {len(hs_test_atk)} atk")

    # Save test methods for per-attack analysis
    methods_path = args.output.replace(".npz", "_test_methods.json")
    with open(methods_path, "w") as f:
        json.dump({"test_atk_methods": test_atk_methods}, f)
    print(f"  test attack methods: {methods_path}")

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE — alignment ablation cache ready")
    print("=" * 70)
    print()
    print("  Compare base model to Instruct model:")
    print(f"    python statistical_tests.py --cache {args.output}")
    print(f"    python verify_saturation.py --llama3_cache {args.output}")
    print(f"    python radial_distribution_check.py --cache {args.output}")
    print()
    print("  Then compare deltas to your Instruct cache "
          "(llama3_activations_cache_diverse.npz):")
    print()
    print("    Predictions:")
    print("      A) HPS drops significantly on base    "
          "→ alignment-mediated geometry")
    print("      B) Both HPS and C4 drop equally       "
          "→ alignment amplifies, not concentrates")
    print("      C) Both stay near TPR=1.000           "
          "→ harm features in pretrained activations")


if __name__ == "__main__":
    main()
