"""
fix_cache_max_length.py — Re-extract attack activations at max_length=2048
to match the diverse benign extraction, eliminating the max_length mismatch
that caused the norm-only AUROC to jump from 0.917 to 1.000.

ROOT CAUSE (identified in audit):
  Original cache extraction (hps_llama3.py via utils.py):
    benign and attacks both extracted with max_length=512

  Diverse benign re-extraction (extract_diverse_benign_activations.py):
    benign extracted with max_length=2048
    attacks REUSED from old cache (still max_length=512)
    → MISMATCH causes deep-layer norm gap (155 vs 35 at layer 31)

This script re-extracts all attacks at max_length=2048 from the cleaned
attacks JSON, then combines them with the diverse benign activations.

Usage on DGX (~3 hours GPU):
  python fix_cache_max_length.py \\
      --model meta-llama/Meta-Llama-3-8B-Instruct \\
      --diverse_cache results/llama3_activations_cache_diverse.npz \\
      --attacks_json llama3_attacks_clean.json \\
      --output results/llama3_activations_cache_diverse_fixed.npz \\
      --layers 0 2 17 24 28 31 \\
      --max_length 2048

  python fix_cache_max_length.py \\
      --model lmsys/vicuna-13b-v1.5 \\
      --diverse_cache results/vicuna_activations_cache_diverse.npz \\
      --attacks_json llama3_attacks_clean.json \\
      --output results/vicuna_activations_cache_diverse_fixed.npz \\
      --layers 0 2 22 31 35 39 \\
      --max_length 2048

After running, verify the fix with:
  python norm_check_diverse.py  (with --cache_path pointing to *_fixed.npz)

  Expected: norm-only AUROC drops from 1.000 to ~0.7-0.9 (still some
  confound from content differences, but no longer perfect).
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
# Activation extraction at user-specified max_length
# ---------------------------------------------------------------------------

def extract_activations(model, tokenizer, prompts, layers, device,
                         max_length, store_full_sequence=True, verbose=True):
    """Forward each prompt; capture hidden states at specified layers.
       Uses the same protocol as utils.py extract_activations and
       extract_diverse_benign_activations.py — no chat template, raw prompt.
    """
    results = []
    n = len(prompts)
    t_start = time.time()

    model.eval()
    for i, prompt in enumerate(prompts):
        with torch.no_grad():
            inputs = tokenizer(
                prompt,
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
                tensor = outputs.hidden_states[l][0]  # (T, d)
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
# Cache I/O
# ---------------------------------------------------------------------------

def detect_cache_format(cache_path):
    cache = np.load(cache_path, allow_pickle=True)
    keys = list(cache.keys())
    if "hs_train_ben" in keys:
        return "dict"
    if "X_benign" in keys:
        return "array"
    raise ValueError(f"Unknown cache format: {keys}")


def hs_dicts_to_lasttoken_array(hs_list, layers):
    """For Vicuna-style cache: convert list-of-dicts to last-token array."""
    n = len(hs_list)
    if n == 0:
        return np.empty((0, len(layers), 0), dtype=np.float32)
    sample = hs_list[0][layers[0]]
    d = sample.shape[1] if sample.ndim == 2 else sample.shape[0]
    out = np.empty((n, len(layers), d), dtype=np.float32)
    for i, hs in enumerate(hs_list):
        for li, l in enumerate(layers):
            t = hs[l]
            out[i, li, :] = t[-1] if t.ndim == 2 else t
    return out


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_attacks(attacks_json, train_frac=0.8, seed=42):
    """Return (train_prompts, test_prompts, test_methods).
       Uses identical split logic to ensure compatibility with cleaned cache.
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
    else:
        raise ValueError(f"Unknown attack JSON format")

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(flat))
    n_tr = int(train_frac * len(idx))
    train = [flat[i] for i in idx[:n_tr]]
    test = [flat[i] for i in idx[n_tr:]]
    test_methods = [methods[i] for i in idx[n_tr:]]
    return train, test, test_methods


# ---------------------------------------------------------------------------
# Cache savers
# ---------------------------------------------------------------------------

def save_dict_cache(output_path, hs_train_ben, hs_train_atk,
                     hs_test_ben, hs_test_atk, cfg_hash="diverse_fixed"):
    np.savez(
        output_path,
        hs_train_ben=np.array(hs_train_ben, dtype=object),
        hs_train_atk=np.array(hs_train_atk, dtype=object),
        hs_test_ben=np.array(hs_test_ben, dtype=object),
        hs_test_atk=np.array(hs_test_atk, dtype=object),
        cfg_hash=np.array(cfg_hash),
    )


def save_array_cache(output_path, X_benign, X_attack, attack_methods, layers):
    np.savez(
        output_path,
        X_benign=X_benign,
        X_attack=X_attack,
        attack_methods=np.array(attack_methods, dtype=object),
        layers=np.array(layers),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--diverse_cache", required=True,
                        help="Existing diverse-benign cache (we keep its benign,"
                             " re-extract attacks at consistent max_length)")
    parser.add_argument("--attacks_json", required=True,
                        help="Attack prompts JSON (use cleaned/dedup version)")
    parser.add_argument("--output", required=True,
                        help="Output cache path (.npz)")
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Truncation cap. MUST match the cap used for "
                             "diverse benign extraction (default 2048).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--store_full_sequence", action="store_true",
                        help="Store FULL SEQUENCE activations (needed for "
                             "Anthropic mean-token probe). Default off for "
                             "backwards compat with array-format Vicuna cache.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("CACHE FIX — Re-extract attacks at consistent max_length")
    print("=" * 70)
    print(f"  Model:         {args.model}")
    print(f"  Diverse cache: {args.diverse_cache}")
    print(f"  Attacks JSON:  {args.attacks_json}")
    print(f"  Output:        {args.output}")
    print(f"  Layers:        {args.layers}")
    print(f"  Max length:    {args.max_length}  ← matches diverse benign")
    print(f"  Device:        {device}")
    print()

    # Sanity checks
    if not os.path.exists(args.diverse_cache):
        print(f"ERROR: {args.diverse_cache} not found")
        sys.exit(1)
    if not os.path.exists(args.attacks_json):
        print(f"ERROR: {args.attacks_json} not found")
        sys.exit(1)

    # Detect existing cache format
    fmt = detect_cache_format(args.diverse_cache)
    print(f"  Cache format: {fmt}")

    # Load existing diverse cache (we'll keep its benign activations)
    print(f"\n  Loading diverse cache (keeping benign activations)...")
    existing = np.load(args.diverse_cache, allow_pickle=True)

    if fmt == "dict":
        hs_train_ben = list(existing["hs_train_ben"])
        hs_test_ben = list(existing["hs_test_ben"])
        print(f"    Train benign: {len(hs_train_ben)}")
        print(f"    Test benign:  {len(hs_test_ben)}")
    else:
        X_benign = existing["X_benign"]
        print(f"    Benign:       {X_benign.shape}")

    # Load attacks (matches the same split as before)
    print(f"\n  Loading attacks from {args.attacks_json}...")
    train_atk, test_atk, test_methods = load_attacks(
        args.attacks_json, train_frac=args.train_frac, seed=args.seed,
    )
    print(f"    Train attacks: {len(train_atk)}")
    print(f"    Test attacks:  {len(test_atk)}")

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

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Loaded. Hidden: {model.config.hidden_size}, "
          f"Layers: {model.config.num_hidden_layers}")

    # Re-extract attacks at consistent max_length
    # If --store_full_sequence is set, we also save full token sequences,
    # which enables Anthropic mean-token probe later.
    store_seq = args.store_full_sequence or (fmt == "dict")
    print(f"\n  Re-extracting train attacks at max_length={args.max_length}, "
          f"store_full_sequence={store_seq}...")
    hs_train_atk = extract_activations(
        model, tokenizer, train_atk, args.layers, device,
        max_length=args.max_length,
        store_full_sequence=store_seq,
    )

    print(f"\n  Re-extracting test attacks at max_length={args.max_length}, "
          f"store_full_sequence={store_seq}...")
    hs_test_atk = extract_activations(
        model, tokenizer, test_atk, args.layers, device,
        max_length=args.max_length,
        store_full_sequence=store_seq,
    )

    # Save in matching format
    print(f"\n  Saving fixed cache to: {args.output}")
    if fmt == "dict":
        save_dict_cache(args.output, hs_train_ben, hs_train_atk,
                         hs_test_ben, hs_test_atk,
                         cfg_hash=f"diverse_fixed_ml{args.max_length}")
        print(f"    train: {len(hs_train_ben)} ben + {len(hs_train_atk)} atk")
        print(f"    test:  {len(hs_test_ben)} ben + {len(hs_test_atk)} atk")
    else:
        # Vicuna-style: combine train+test attacks since cache uses single arrays
        all_atk = train_atk + test_atk
        all_atk_methods = (
            ["unknown"] * len(train_atk) +  # placeholder for train methods
            test_methods
        )
        # Actually, we need to track methods for ALL prompts, so reload
        all_atk_prompts, all_atk_methods_full = [], []
        with open(args.attacks_json) as f:
            data = json.load(f)
        if isinstance(data, dict):
            for atk_name, prompts in data.items():
                for p in prompts:
                    if isinstance(p, str) and len(p.strip()) >= 5:
                        all_atk_prompts.append(p)
                        all_atk_methods_full.append(atk_name)

        # Re-extract all attacks together for array-format cache
        # (we already extracted train+test, but they're in different order)
        # Actually we should re-extract ALL attacks in original order
        print(f"\n  (Vicuna-format cache requires re-extraction of ALL attacks "
              f"in order)")
        print(f"  Re-extracting {len(all_atk_prompts)} total attacks "
              f"(last_token)...")
        hs_all_atk = extract_activations(
            model, tokenizer, all_atk_prompts, args.layers, device,
            max_length=args.max_length, store_full_sequence=False,
        )
        X_attack = np.stack(
            [np.stack([h[l] for l in args.layers], axis=0)
             for h in hs_all_atk], axis=0,
        )
        save_array_cache(args.output, X_benign, X_attack, all_atk_methods_full,
                          args.layers)
        print(f"    X_benign: {X_benign.shape}, X_attack: {X_attack.shape}")

    # Save test methods sidecar
    methods_path = args.output.replace(".npz", "_test_methods.json")
    with open(methods_path, "w") as f:
        json.dump({"test_atk_methods": test_methods}, f)
    print(f"    Test methods sidecar: {methods_path}")

    print("\n" + "=" * 70)
    print("CACHE FIXED")
    print("=" * 70)
    print(f"  Output cache: {args.output}")
    print()
    print("  Verify the fix:")
    print(f"    python norm_check_diverse.py  "
          f"# (point at {args.output})")
    print()
    print("  Then re-run the experiments:")
    print(f"    python verify_saturation.py --llama3_cache {args.output}")
    print(f"    python statistical_tests.py --cache {args.output}")
    print(f"    python radial_distribution_check.py --cache {args.output}")


if __name__ == "__main__":
    main()
