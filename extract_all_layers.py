"""
extract_all_layers.py — Re-extract activations including all 32 layers of Llama-3-8B.

The original cache (llama3_activations_cache_diverse_fixed.npz) was built with
6 specific layers [0, 2, 17, 24, 28, 31]. For the layer ablation experiment,
we need access to all layers (or at minimum, the safety zone 6-12 from Li et al. 2024).

This script extracts ALL 32 layers' last-token activations for the same data:
  - 5905 benign prompts (diverse benign dataset)
  - 6474 attack prompts (clean attacks, deduplicated)

Output cache size estimate (fp16):
  - 12,379 prompts × 32 layers × 4096 dim × 2 bytes = 3.2 GB

Storage format matches existing cache: dict-style hs_train_ben, hs_train_atk, etc.
Each prompt's activations stored as a dict mapping layer_idx -> array.

Runtime estimate: ~30-45 min on A100 80GB.

Usage:
    python extract_all_layers.py \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --benign_csv results/data_harmless_diverse.csv \\
        --attacks_json llama3_attacks_clean.json \\
        --output results/llama3_activations_cache_alllayers.npz \\
        --max_length 2048
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_activations(model, tokenizer, prompts, max_length=2048, batch_size=1):
    """
    For each prompt, get last-token activations across ALL layers.
    Returns list of dicts {layer_idx: array}.
    """
    all_results = []
    n_layers = model.config.num_hidden_layers
    t_start = time.time()

    for i, prompt in enumerate(prompts):
        # Apply chat template
        messages = [{"role": "user", "content": prompt}]
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(templated, return_tensors="pt", truncation=True,
                         max_length=max_length, add_special_tokens=False
                         ).input_ids.to(DEVICE)

        with torch.no_grad():
            outputs = model(input_ids=ids, output_hidden_states=True,
                             return_dict=True, use_cache=False)

        # outputs.hidden_states is tuple of (n_layers + 1) tensors
        # Index 0 is embedding output; indices 1..n_layers are layer outputs
        # We want indices 1..n_layers, last token only
        layer_acts = {}
        for layer_idx in range(n_layers):
            acts = outputs.hidden_states[layer_idx + 1][0, -1, :].cpu().numpy().astype(np.float16)
            layer_acts[layer_idx] = acts

        all_results.append(layer_acts)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(prompts) - i - 1) / rate / 60
            print(f"    [{i+1}/{len(prompts)}] {rate:.1f} prompts/sec, ETA {eta:.1f} min")

    return all_results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--benign_csv", default="results/data_harmless_diverse.csv")
    p.add_argument("--attacks_json", default="llama3_attacks_clean.json")
    p.add_argument("--output", default="results/llama3_activations_cache_alllayers.npz")
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print("=" * 70)
    print("  ALL-LAYER ACTIVATION EXTRACTION")
    print("=" * 70)
    print(f"  Model:        {args.model}")
    print(f"  Benign CSV:   {args.benign_csv}")
    print(f"  Attacks JSON: {args.attacks_json}")
    print(f"  Output:       {args.output}")
    print(f"  Max length:   {args.max_length}")
    print()

    # ---- Load model ----
    print("  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    print(f"    Loaded. Layers: {model.config.num_hidden_layers}, hidden: {model.config.hidden_size}")
    print()

    # ---- Load data ----
    print("  Loading benign data...")
    benign_df = pd.read_csv(args.benign_csv)
    benign_prompts = benign_df.iloc[:, 0].astype(str).tolist()
    print(f"    {len(benign_prompts)} benign prompts")

    print("  Loading attack data...")
    with open(args.attacks_json) as f:
        attacks_data = json.load(f)
    if isinstance(attacks_data, list):
        attack_prompts = [a if isinstance(a, str) else a.get("prompt", "") for a in attacks_data]
    elif isinstance(attacks_data, dict):
        # Common format: {"method_name": [list of prompts]}
        attack_prompts = []
        for method, prompts in attacks_data.items():
            for p in prompts:
                attack_prompts.append(p if isinstance(p, str) else p.get("prompt", ""))
    print(f"    {len(attack_prompts)} attack prompts")
    print()

    # ---- Train/test split ----
    np.random.seed(args.seed)
    n_ben = len(benign_prompts)
    n_atk = len(attack_prompts)
    perm_ben = np.random.permutation(n_ben)
    perm_atk = np.random.permutation(n_atk)
    n_tr_ben = int(args.train_frac * n_ben)
    n_tr_atk = int(args.train_frac * n_atk)
    train_ben = [benign_prompts[i] for i in perm_ben[:n_tr_ben]]
    test_ben = [benign_prompts[i] for i in perm_ben[n_tr_ben:]]
    train_atk = [attack_prompts[i] for i in perm_atk[:n_tr_atk]]
    test_atk = [attack_prompts[i] for i in perm_atk[n_tr_atk:]]
    print(f"  Train: {len(train_ben)} ben + {len(train_atk)} atk")
    print(f"  Test:  {len(test_ben)} ben + {len(test_atk)} atk")
    print()

    # ---- Extract ----
    t0 = time.time()
    print("  Extracting train benign...")
    hs_train_ben = extract_activations(model, tokenizer, train_ben, args.max_length)

    print("  Extracting test benign...")
    hs_test_ben = extract_activations(model, tokenizer, test_ben, args.max_length)

    print("  Extracting train attacks...")
    hs_train_atk = extract_activations(model, tokenizer, train_atk, args.max_length)

    print("  Extracting test attacks...")
    hs_test_atk = extract_activations(model, tokenizer, test_atk, args.max_length)

    elapsed = (time.time() - t0) / 60
    print(f"\n  Total extraction time: {elapsed:.1f} min")
    print()

    # ---- Save ----
    print(f"  Saving to {args.output}...")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(
        args.output,
        hs_train_ben=np.array(hs_train_ben, dtype=object),
        hs_train_atk=np.array(hs_train_atk, dtype=object),
        hs_test_ben=np.array(hs_test_ben, dtype=object),
        hs_test_atk=np.array(hs_test_atk, dtype=object),
    )

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"  Saved. File size: {size_mb:.1f} MB")
    print()
    print("  Now you can run the layer ablation:")
    print(f"    CACHE={args.output} bash run_layer_ablation.sh")


if __name__ == "__main__":
    main()
