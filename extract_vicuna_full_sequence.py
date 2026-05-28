"""
extract_vicuna_full_sequence.py — Re-extract Vicuna activations with FULL
sequence storage (not just last-token), to enable Anthropic mean-token probe.

The existing fix_cache_max_length.py for Vicuna saves only last-token in
array format. Anthropic's MTP requires the full token sequence to compute
mean-over-tokens. This script:

  1. Loads existing diverse benign + clean attacks list
  2. Re-extracts BOTH benign + attacks at max_length=2048 with full sequences
  3. Saves as dict-format cache (compatible with anthropic_mean_token_probe.py)

Output:
  results/vicuna_activations_cache_diverse_fullseq.npz

Usage:
  python extract_vicuna_full_sequence.py 2>&1 | tee results/log_vicuna_fullseq.txt

This takes ~3 hours on the DGX. Skip if you don't need MTP on Vicuna.
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


def extract_activations(model, tokenizer, prompts, layers, device,
                         max_length=2048, verbose=True):
    """Extract FULL SEQUENCE activations as dict[layer -> (T, d)] per prompt."""
    results = []
    n = len(prompts)
    t0 = time.time()
    model.eval()
    for i, prompt in enumerate(prompts):
        with torch.no_grad():
            inputs = tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=max_length,
            ).to(device)
            try:
                out = model(**inputs, output_hidden_states=True)
                hs = {}
                for l in layers:
                    hs[l] = out.hidden_states[l][0].cpu().numpy().astype(np.float32)
                results.append(hs)
            except Exception as e:
                print(f"  ERROR on prompt {i}: {e}")
                results.append({l: np.zeros(model.config.hidden_size,
                                              dtype=np.float32)
                                for l in layers})

        if verbose and (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(f"    [{i+1}/{n}] {rate:.1f} prompts/sec, "
                  f"ETA {eta/60:.1f} min")
    elapsed = time.time() - t0
    print(f"    Done: {n} prompts in {elapsed/60:.1f} min")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lmsys/vicuna-13b-v1.5")
    parser.add_argument("--diverse_benign",
                        default="results/data_harmless_diverse.csv")
    parser.add_argument("--attacks_json", default="llama3_attacks_clean.json")
    parser.add_argument("--output",
                        default="results/vicuna_activations_cache_diverse_fullseq.npz")
    parser.add_argument("--layers", type=int, nargs="+",
                        default=[0, 2, 22, 31, 35, 39])
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("VICUNA FULL-SEQUENCE EXTRACTION — for MTP support")
    print("=" * 70)
    print(f"  Model:          {args.model}")
    print(f"  Diverse benign: {args.diverse_benign}")
    print(f"  Attacks JSON:   {args.attacks_json}")
    print(f"  Output:         {args.output}")
    print(f"  Layers:         {args.layers}")
    print(f"  Max length:     {args.max_length}")
    print(f"  Device:         {device}")
    print()

    # Load attacks
    with open(args.attacks_json) as f:
        data = json.load(f)
    flat, methods = [], []
    if isinstance(data, dict):
        for m, prompts in data.items():
            for p in prompts:
                if isinstance(p, str) and len(p.strip()) >= 5:
                    flat.append(p)
                    methods.append(m)
    print(f"  Attacks loaded: {len(flat)}")

    rng = np.random.RandomState(args.seed)
    a_idx = rng.permutation(len(flat))
    n_atk_train = int(args.train_frac * len(a_idx))
    train_atk = [flat[i] for i in a_idx[:n_atk_train]]
    test_atk = [flat[i] for i in a_idx[n_atk_train:]]
    test_atk_methods = [methods[i] for i in a_idx[n_atk_train:]]
    print(f"  Train: {len(train_atk)} attacks, Test: {len(test_atk)}")

    # Load benign
    df = pd.read_csv(args.diverse_benign)
    col = "prompt" if "prompt" in df.columns else df.columns[0]
    benign_prompts = df[col].dropna().astype(str).tolist()
    rng2 = np.random.RandomState(args.seed)
    b_idx = rng2.permutation(len(benign_prompts))
    n_target = max(len(flat), 5000)
    n_target = min(n_target, len(b_idx))
    benign_prompts = [benign_prompts[i] for i in b_idx[:n_target]]
    n_ben_train = int(args.train_frac * len(benign_prompts))
    train_ben = benign_prompts[:n_ben_train]
    test_ben = benign_prompts[n_ben_train:]
    print(f"  Benign train: {len(train_ben)}, test: {len(test_ben)}")

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\nLoading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device, low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Loaded. Hidden: {model.config.hidden_size}")

    # Extract all four buckets
    print(f"\nExtracting train_ben ({len(train_ben)})...")
    hs_train_ben = extract_activations(model, tokenizer, train_ben,
                                          args.layers, device, args.max_length)
    print(f"\nExtracting test_ben ({len(test_ben)})...")
    hs_test_ben = extract_activations(model, tokenizer, test_ben,
                                         args.layers, device, args.max_length)
    print(f"\nExtracting train_atk ({len(train_atk)})...")
    hs_train_atk = extract_activations(model, tokenizer, train_atk,
                                          args.layers, device, args.max_length)
    print(f"\nExtracting test_atk ({len(test_atk)})...")
    hs_test_atk = extract_activations(model, tokenizer, test_atk,
                                         args.layers, device, args.max_length)

    # Save in DICT format (compatible with anthropic_mean_token_probe.py)
    print(f"\nSaving to {args.output}...")
    np.savez(
        args.output,
        hs_train_ben=np.array(hs_train_ben, dtype=object),
        hs_train_atk=np.array(hs_train_atk, dtype=object),
        hs_test_ben=np.array(hs_test_ben, dtype=object),
        hs_test_atk=np.array(hs_test_atk, dtype=object),
        cfg_hash=np.array("vicuna_fullseq"),
    )

    # Save sidecar with test methods
    sidecar = args.output.replace(".npz", "_test_methods.json")
    with open(sidecar, "w") as f:
        json.dump({"test_atk_methods": test_atk_methods}, f)

    print(f"  ✓ {args.output}")
    print(f"  ✓ {sidecar}")
    print(f"  train: {len(hs_train_ben)} ben + {len(hs_train_atk)} atk")
    print(f"  test:  {len(hs_test_ben)} ben + {len(hs_test_atk)} atk")
    print("\nNow run:")
    print(f"  python anthropic_mean_token_probe.py \\")
    print(f"      --cache {args.output} \\")
    print(f"      --layers {' '.join(map(str, args.layers))} \\")
    print(f"      --hidden_dim 5120 \\")
    print(f"      --output results/anthropic_mtp_vicuna.json")


if __name__ == "__main__":
    main()
