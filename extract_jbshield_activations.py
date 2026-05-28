"""
extract_jbshield_activations.py — Extract activations on JBShield attacks +
diverse benign, producing a fresh activation cache.

Unlike extract_diverse_benign_activations.py (which reuses existing attack
activations from our cache), this script re-extracts BOTH benign and attacks
from scratch using JBShield's prompts. This is needed because:

  1. JBShield's prompts may differ slightly from our llama3_attacks.json
     (different splits, deduplication, formatting)
  2. We want a clean independent measurement to cross-validate saturation

Output cache format matches our existing caches:
  - For Llama-3-style models: dict-of-objects (hs_train_ben, hs_train_atk, ...)
  - For Vicuna-style models:  pre-extracted (X_benign, X_attack, ...)

Usage:
  # Vicuna-13B-v1.5 (~3 hrs GPU):
  python extract_jbshield_activations.py \\
      --model lmsys/vicuna-13b-v1.5 \\
      --attacks_json results/jbshield_vicuna13b_attacks.json \\
      --diverse_benign results/data_harmless_diverse.csv \\
      --output results/vicuna_activations_cache_jbshield.npz \\
      --layers 0 2 22 31 35 39 \\
      --cache_format array

  # Llama-3-8B-Instruct (~3 hrs GPU):
  python extract_jbshield_activations.py \\
      --model meta-llama/Meta-Llama-3-8B-Instruct \\
      --attacks_json results/jbshield_llama3_attacks.json \\
      --diverse_benign results/data_harmless_diverse.csv \\
      --output results/llama3_activations_cache_jbshield.npz \\
      --layers 0 2 17 24 28 31 \\
      --cache_format dict
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
# Activation extraction — copied from extract_diverse_benign_activations.py
# to keep this script self-contained. Same per-prompt forward pass.
# ---------------------------------------------------------------------------

def extract_activations_for_prompts(model, tokenizer, prompts, layers,
                                      device, max_length=2048,
                                      store_full_sequence=True,
                                      verbose=True):
    """Forward each prompt; capture hidden states at specified layers."""
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
# Cache savers
# ---------------------------------------------------------------------------

def save_dict_cache(output_path, hs_train_ben, hs_train_atk,
                     hs_test_ben, hs_test_atk, cfg_hash="jbshield"):
    """Save in Llama-3 dict-format."""
    np.savez(
        output_path,
        hs_train_ben=np.array(hs_train_ben, dtype=object),
        hs_train_atk=np.array(hs_train_atk, dtype=object),
        hs_test_ben=np.array(hs_test_ben, dtype=object),
        hs_test_atk=np.array(hs_test_atk, dtype=object),
        cfg_hash=np.array(cfg_hash),
    )


def save_array_cache(output_path, X_benign, X_attack,
                      attack_methods, layers,
                      train_test_idx_benign=None, train_test_idx_attack=None):
    """Save in Vicuna array-format with optional split indices."""
    save_dict = dict(
        X_benign=X_benign,
        X_attack=X_attack,
        attack_methods=np.array(attack_methods, dtype=object),
        layers=np.array(layers),
    )
    if train_test_idx_benign is not None:
        save_dict["train_idx_benign"] = train_test_idx_benign["train"]
        save_dict["test_idx_benign"] = train_test_idx_benign["test"]
    if train_test_idx_attack is not None:
        save_dict["train_idx_attack"] = train_test_idx_attack["train"]
        save_dict["test_idx_attack"] = train_test_idx_attack["test"]
    np.savez(output_path, **save_dict)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_attacks_with_methods(attacks_json):
    """
    Load JBShield attack JSON in our cache-compatible dict format.
    Returns:
      flat_prompts: list[str]
      methods: list[str]   (parallel to flat_prompts)
    """
    with open(attacks_json) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected dict[attack_name -> list[str]] in {attacks_json}, "
            f"got {type(data)}"
        )

    flat = []
    methods = []
    for attack_name, prompts in data.items():
        for p in prompts:
            if isinstance(p, str) and len(p.strip()) >= 5:
                flat.append(p)
                methods.append(attack_name)
    return flat, methods


def load_diverse_benign(csv_path, n=None, seed=42):
    """Load diverse benign prompts (single column "prompt" or first column)."""
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


def split_train_test(arr, n_train, seed=42):
    """Random 80/20 split (or exact n_train if specified)."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(arr))
    if n_train > len(idx):
        n_train = int(0.8 * len(idx))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Cache builders
# ---------------------------------------------------------------------------

def hs_dicts_to_lasttoken_array(hs_list, layers):
    """Convert list-of-dicts (with full-sequence tensors) to (N, n_layers, d)
       last-token array for Vicuna-style cache."""
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


def build_dict_cache(model, tokenizer, attacks_json, diverse_benign_csv,
                     layers, output_path, device, max_length=2048,
                     train_frac=0.8, seed=42):
    """Llama-3-style cache: full-sequence hidden states stored as dicts."""

    # Load attacks
    print(f"\n  Loading attacks from {attacks_json}...")
    atk_prompts, atk_methods = load_attacks_with_methods(attacks_json)
    print(f"    {len(atk_prompts)} attacks across {len(set(atk_methods))} "
          f"methods")
    method_counts = {m: atk_methods.count(m) for m in sorted(set(atk_methods))}
    for m, n in method_counts.items():
        print(f"      {m}: {n}")

    # Train/test split for attacks (matches our existing 80/20 convention)
    rng = np.random.RandomState(seed)
    a_idx = rng.permutation(len(atk_prompts))
    n_atk_train = int(train_frac * len(a_idx))
    train_atk_prompts = [atk_prompts[i] for i in a_idx[:n_atk_train]]
    test_atk_prompts = [atk_prompts[i] for i in a_idx[n_atk_train:]]
    test_atk_methods = [atk_methods[i] for i in a_idx[n_atk_train:]]
    print(f"    Train: {len(train_atk_prompts)}  "
          f"Test: {len(test_atk_prompts)}")

    # Load benign
    print(f"\n  Loading diverse benign from {diverse_benign_csv}...")
    n_ben_target = max(len(atk_prompts), 5000)
    ben_prompts = load_diverse_benign(diverse_benign_csv, n=n_ben_target,
                                       seed=seed)
    n_ben_train = int(train_frac * len(ben_prompts))
    train_ben_prompts = ben_prompts[:n_ben_train]
    test_ben_prompts = ben_prompts[n_ben_train:]
    print(f"    {len(ben_prompts)} benign prompts "
          f"(train {len(train_ben_prompts)}, test {len(test_ben_prompts)})")

    # Extract activations
    print(f"\n  Extracting train benign ({len(train_ben_prompts)})...")
    hs_train_ben = extract_activations_for_prompts(
        model, tokenizer, train_ben_prompts, layers, device,
        max_length=max_length, store_full_sequence=True,
    )

    print(f"\n  Extracting test benign ({len(test_ben_prompts)})...")
    hs_test_ben = extract_activations_for_prompts(
        model, tokenizer, test_ben_prompts, layers, device,
        max_length=max_length, store_full_sequence=True,
    )

    print(f"\n  Extracting train attacks ({len(train_atk_prompts)})...")
    hs_train_atk = extract_activations_for_prompts(
        model, tokenizer, train_atk_prompts, layers, device,
        max_length=max_length, store_full_sequence=True,
    )

    print(f"\n  Extracting test attacks ({len(test_atk_prompts)})...")
    hs_test_atk = extract_activations_for_prompts(
        model, tokenizer, test_atk_prompts, layers, device,
        max_length=max_length, store_full_sequence=True,
    )

    # Save
    print(f"\n  Saving combined cache to: {output_path}")
    save_dict_cache(output_path, hs_train_ben, hs_train_atk,
                     hs_test_ben, hs_test_atk, cfg_hash="jbshield")
    print(f"    train: {len(hs_train_ben)} ben + {len(hs_train_atk)} atk")
    print(f"    test:  {len(hs_test_ben)} ben + {len(hs_test_atk)} atk")

    # Save test-attack methods for downstream per-attack analysis
    methods_path = output_path.replace(".npz", "_test_methods.json")
    with open(methods_path, "w") as f:
        json.dump({"test_atk_methods": test_atk_methods}, f)
    print(f"    test attack methods: {methods_path}")


def build_array_cache(model, tokenizer, attacks_json, diverse_benign_csv,
                       layers, output_path, device, max_length=2048,
                       train_frac=0.8, seed=42):
    """Vicuna-style cache: pre-extracted last-token activations as arrays."""

    # Load attacks
    print(f"\n  Loading attacks from {attacks_json}...")
    atk_prompts, atk_methods = load_attacks_with_methods(attacks_json)
    print(f"    {len(atk_prompts)} attacks across {len(set(atk_methods))} "
          f"methods")
    method_counts = {m: atk_methods.count(m) for m in sorted(set(atk_methods))}
    for m, n in method_counts.items():
        print(f"      {m}: {n}")

    # Load benign
    print(f"\n  Loading diverse benign from {diverse_benign_csv}...")
    n_ben_target = max(int(len(atk_prompts) * 1.65), 500)
    ben_prompts = load_diverse_benign(diverse_benign_csv, n=n_ben_target,
                                       seed=seed)
    print(f"    {len(ben_prompts)} benign prompts")

    # Extract activations
    print(f"\n  Extracting benign ({len(ben_prompts)})...")
    hs_ben = extract_activations_for_prompts(
        model, tokenizer, ben_prompts, layers, device,
        max_length=max_length, store_full_sequence=False,
    )
    X_benign = np.stack(
        [np.stack([h[l] for l in layers], axis=0) for h in hs_ben], axis=0,
    )

    print(f"\n  Extracting attacks ({len(atk_prompts)})...")
    hs_atk = extract_activations_for_prompts(
        model, tokenizer, atk_prompts, layers, device,
        max_length=max_length, store_full_sequence=False,
    )
    X_attack = np.stack(
        [np.stack([h[l] for l in layers], axis=0) for h in hs_atk], axis=0,
    )

    # 80/20 splits for downstream
    rng = np.random.RandomState(seed)
    ben_idx = rng.permutation(len(ben_prompts))
    n_ben_train = int(train_frac * len(ben_idx))
    train_idx_ben = ben_idx[:n_ben_train]
    test_idx_ben = ben_idx[n_ben_train:]

    atk_idx = rng.permutation(len(atk_prompts))
    n_atk_train = int(train_frac * len(atk_idx))
    train_idx_atk = atk_idx[:n_atk_train]
    test_idx_atk = atk_idx[n_atk_train:]

    print(f"\n  Saving combined cache to: {output_path}")
    save_array_cache(
        output_path, X_benign, X_attack, atk_methods, layers,
        train_test_idx_benign={"train": train_idx_ben, "test": test_idx_ben},
        train_test_idx_attack={"train": train_idx_atk, "test": test_idx_atk},
    )
    print(f"    X_benign: {X_benign.shape}  X_attack: {X_attack.shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="HuggingFace model name (e.g., lmsys/vicuna-13b-v1.5)")
    parser.add_argument("--attacks_json", required=True,
                        help="Attack JSON from build_jbshield_attacks.py")
    parser.add_argument("--diverse_benign", required=True,
                        help="Diverse benign CSV from build_diverse_benign.py")
    parser.add_argument("--output", required=True,
                        help="Output cache path (.npz)")
    parser.add_argument("--layers", type=int, nargs="+", required=True,
                        help="Layer indices")
    parser.add_argument("--cache_format", choices=["dict", "array"],
                        required=True,
                        help="dict (Llama-3 style) or array (Vicuna style)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("JBShield Activation Extraction")
    print("=" * 70)
    print(f"  Model:          {args.model}")
    print(f"  Attacks JSON:   {args.attacks_json}")
    print(f"  Diverse benign: {args.diverse_benign}")
    print(f"  Output:         {args.output}")
    print(f"  Layers:         {args.layers}")
    print(f"  Cache format:   {args.cache_format}")
    print(f"  Device:         {device}")
    print()

    # Sanity checks
    if not os.path.exists(args.attacks_json):
        print(f"ERROR: {args.attacks_json} not found")
        print("  Run build_jbshield_attacks.py first.")
        sys.exit(1)
    if not os.path.exists(args.diverse_benign):
        print(f"ERROR: {args.diverse_benign} not found")
        print("  Run build_diverse_benign.py first.")
        sys.exit(1)

    # Load model
    print(f"Loading model: {args.model}")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  ERROR: transformers not installed")
        sys.exit(1)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except Exception as e:
        print(f"  ERROR loading tokenizer: {e}")
        print(f"  If gated repo: huggingface-cli login")
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
    print(f"  Loaded. Hidden size: {model.config.hidden_size}, "
          f"Layers: {model.config.num_hidden_layers}")

    n_layers_max = model.config.num_hidden_layers + 1
    if max(args.layers) >= n_layers_max:
        print(f"ERROR: requested layer {max(args.layers)} >= {n_layers_max}")
        sys.exit(1)

    # Dispatch
    if args.cache_format == "dict":
        build_dict_cache(
            model, tokenizer, args.attacks_json, args.diverse_benign,
            args.layers, args.output, device,
            max_length=args.max_length,
            train_frac=args.train_frac, seed=args.seed,
        )
    else:
        build_array_cache(
            model, tokenizer, args.attacks_json, args.diverse_benign,
            args.layers, args.output, device,
            max_length=args.max_length,
            train_frac=args.train_frac, seed=args.seed,
        )

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"  Output cache: {args.output}")
    print()
    print("  Run downstream experiments:")
    print(f"    python verify_saturation.py --llama3_cache {args.output}")
    print(f"    python statistical_tests.py --cache {args.output}")
    print(f"    python vicuna_imbalance_test.py --vicuna_cache {args.output}")
    print(f"    python gcg_specific_test.py "
          f"--vicuna_cache {args.output}")


if __name__ == "__main__":
    main()
