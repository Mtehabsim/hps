"""
extract_diverse_benign_activations.py — Extract activations on diverse benign
prompts and produce updated activation caches.

After verify_saturation.py revealed that prompt-length is a major confound
(length-only AUROC=0.973), we built a diverse benign set
(results/data_harmless_diverse.csv) that includes long-form text matching
attack lengths. This script forward-passes those prompts through both LLMs
and produces new activation caches.

Output (drop-in replacements for existing caches):
  - results/llama3_activations_cache_diverse.npz  — Llama-3 with diverse benign
  - results/vicuna_activations_cache_diverse.npz  — Vicuna with diverse benign

The attack activations are reused from the existing caches (no need to re-extract).
Only the benign portion is replaced.

Usage on DGX:
  # Llama-3-8B (needs ~3-5h compute):
  python extract_diverse_benign_activations.py \\
      --model meta-llama/Meta-Llama-3-8B-Instruct \\
      --diverse_benign results/data_harmless_diverse.csv \\
      --existing_cache results/llama3_activations_cache.npz \\
      --output results/llama3_activations_cache_diverse.npz \\
      --layers 0 2 17 24 28 31

  # Vicuna-13B-v1.5 (needs ~3-5h compute):
  python extract_diverse_benign_activations.py \\
      --model lmsys/vicuna-13b-v1.5 \\
      --diverse_benign results/data_harmless_diverse.csv \\
      --existing_cache results/vicuna_activations_cache.npz \\
      --output results/vicuna_activations_cache_diverse.npz \\
      --layers 0 2 22 31 35 39

Notes:
  - Llama-3 requires HuggingFace token (huggingface-cli login)
  - The script auto-detects existing cache format (Llama-3 dict-style vs
    Vicuna pre-extracted-array-style)
  - Long prompts are truncated to max_length=2048 tokens to control memory
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
# Activation extraction
# ---------------------------------------------------------------------------

def extract_activations_for_prompts(model, tokenizer, prompts, layers,
                                      device, max_length=2048,
                                      batch_size=1, store_full_sequence=True,
                                      verbose=True):
    """
    Forward-pass each prompt and capture hidden states at specified layers.

    Args:
        prompts: list of strings
        layers: list of int layer indices
        store_full_sequence: if True, store (T, d) per layer (matches Llama-3
            cache format). If False, store (d,) per layer (last-token only,
            matches Vicuna format).

    Returns:
        list of dicts {layer_idx: np.ndarray} — one per prompt
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
                # Skip with empty dict to keep alignment
                results.append({l: np.zeros(model.config.hidden_size,
                                              dtype=np.float32)
                                for l in layers})
                continue

            # outputs.hidden_states is tuple of (n_layers+1) tensors
            # each shape (batch=1, seq_len, hidden_dim)
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
# Cache I/O — handle both formats
# ---------------------------------------------------------------------------

def detect_cache_format(cache_path):
    """Return 'dict' (Llama-3 style) or 'array' (Vicuna style) based on keys."""
    cache = np.load(cache_path, allow_pickle=True)
    keys = list(cache.keys())
    if "hs_train_ben" in keys:
        return "dict"
    if "X_benign" in keys:
        return "array"
    raise ValueError(
        f"Unknown cache format. Keys: {keys}"
    )


def save_dict_cache(output_path, hs_train_ben, hs_train_atk,
                     hs_test_ben, hs_test_atk, cfg_hash="diverse"):
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
                      attack_methods, layers):
    """Save in Vicuna array-format."""
    np.savez(
        output_path,
        X_benign=X_benign,
        X_attack=X_attack,
        attack_methods=np.array(attack_methods, dtype=object),
        layers=np.array(layers),
    )


def hs_dicts_to_array(hs_list, layers):
    """Convert list-of-dicts to (N, n_layers, d) array (last-token)."""
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
# Main extraction logic
# ---------------------------------------------------------------------------

def load_diverse_benign(csv_path, n_train=None, n_test=None,
                         seed=42, source_filter=None):
    """Load diverse benign prompts; split 80/20 by default."""
    df = pd.read_csv(csv_path)
    if "prompt" not in df.columns:
        # Use first column
        prompts = df.iloc[:, 0].dropna().astype(str).tolist()
    else:
        if source_filter:
            df = df[df["source"].isin(source_filter)] \
                if "source" in df.columns else df
        prompts = df["prompt"].dropna().astype(str).tolist()

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(prompts))
    if n_train is None or n_test is None:
        n_train = int(0.8 * len(prompts))
        n_test = len(prompts) - n_train
    train = [prompts[i] for i in idx[:n_train]]
    test = [prompts[i] for i in idx[n_train:n_train + n_test]]
    return train, test


def extract_for_dict_cache(model, tokenizer, args, device):
    """Process Llama-3 style cache: replace benign in dict format."""
    print(f"\n  Loading existing cache: {args.existing_cache}")
    existing = np.load(args.existing_cache, allow_pickle=True)

    # Reuse attacks unchanged
    hs_train_atk = list(existing["hs_train_atk"])
    hs_test_atk = list(existing["hs_test_atk"])
    print(f"    Reusing {len(hs_train_atk)} train + {len(hs_test_atk)} test attacks")

    # Determine train/test sizes for benign (match attack counts)
    n_train_ben = len(hs_train_atk)
    n_test_ben = len(hs_test_atk)

    print(f"\n  Loading diverse benign: {args.diverse_benign}")
    train_ben_prompts, test_ben_prompts = load_diverse_benign(
        args.diverse_benign,
        n_train=n_train_ben,
        n_test=n_test_ben,
        seed=args.seed,
    )
    print(f"    Got {len(train_ben_prompts)} train + {len(test_ben_prompts)} test "
          f"benign from {args.diverse_benign}")

    if len(train_ben_prompts) < n_train_ben:
        print(f"    WARNING: only {len(train_ben_prompts)} train benign available, "
              f"requested {n_train_ben}")

    print(f"\n  Extracting train benign activations ({len(train_ben_prompts)} prompts)...")
    hs_train_ben = extract_activations_for_prompts(
        model, tokenizer, train_ben_prompts, args.layers, device,
        max_length=args.max_length, store_full_sequence=True,
    )
    print(f"\n  Extracting test benign activations ({len(test_ben_prompts)} prompts)...")
    hs_test_ben = extract_activations_for_prompts(
        model, tokenizer, test_ben_prompts, args.layers, device,
        max_length=args.max_length, store_full_sequence=True,
    )

    print(f"\n  Saving combined cache to: {args.output}")
    save_dict_cache(
        args.output, hs_train_ben, hs_train_atk, hs_test_ben, hs_test_atk,
        cfg_hash=f"diverse_{args.seed}",
    )
    print(f"    train: {len(hs_train_ben)} ben + {len(hs_train_atk)} atk")
    print(f"    test:  {len(hs_test_ben)} ben + {len(hs_test_atk)} atk")


def extract_for_array_cache(model, tokenizer, args, device):
    """Process Vicuna style cache: replace X_benign with diverse activations."""
    print(f"\n  Loading existing cache: {args.existing_cache}")
    existing = np.load(args.existing_cache, allow_pickle=True)
    X_attack = existing["X_attack"]
    attack_methods = existing["attack_methods"].tolist()
    cached_layers = existing["layers"].tolist()
    print(f"    Reusing {len(X_attack)} attacks across "
          f"{len(set(attack_methods))} methods")

    # Verify layer match
    if cached_layers != args.layers:
        print(f"    Cached layers: {cached_layers}")
        print(f"    Requested:     {args.layers}")
        print(f"    Note: produced cache will use REQUESTED layers; "
              f"existing X_attack must match.")

    # Load diverse benign
    n_target = max(int(len(X_attack) * 1.65), 500)  # similar ratio as before
    print(f"\n  Loading diverse benign: {args.diverse_benign} "
          f"(target ~{n_target} prompts)")
    train_ben, test_ben = load_diverse_benign(
        args.diverse_benign, n_train=n_target, n_test=0, seed=args.seed,
    )
    benign_prompts = train_ben  # combine — will be split downstream
    print(f"    Got {len(benign_prompts)} benign prompts")

    print(f"\n  Extracting benign activations ({len(benign_prompts)} prompts)...")
    benign_hs = extract_activations_for_prompts(
        model, tokenizer, benign_prompts, args.layers, device,
        max_length=args.max_length, store_full_sequence=False,
    )

    # Convert to (N, n_layers, d) array
    X_benign_new = np.stack(
        [np.stack([b[l] for l in args.layers], axis=0) for b in benign_hs],
        axis=0,
    )
    print(f"    X_benign shape: {X_benign_new.shape}")

    # Use existing attack array as-is (assume layers match)
    # If layers differ, the user needs to re-extract attacks too
    X_attack_arr = np.array(X_attack)

    print(f"\n  Saving combined cache to: {args.output}")
    save_array_cache(
        args.output, X_benign_new, X_attack_arr, attack_methods, args.layers,
    )
    print(f"    X_benign: {X_benign_new.shape}, X_attack: {X_attack_arr.shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="HuggingFace model name or path")
    parser.add_argument("--diverse_benign", required=True,
                        help="CSV file with diverse benign prompts (output of "
                             "build_diverse_benign.py)")
    parser.add_argument("--existing_cache", required=True,
                        help="Existing activation cache (we keep attacks, "
                             "replace benign)")
    parser.add_argument("--output", required=True,
                        help="Output cache path")
    parser.add_argument("--layers", type=int, nargs="+", required=True,
                        help="Layer indices to extract")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Max input tokens (truncate longer prompts)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("DIVERSE BENIGN ACTIVATION EXTRACTION")
    print("=" * 70)
    print(f"  Model:          {args.model}")
    print(f"  Diverse benign: {args.diverse_benign}")
    print(f"  Existing cache: {args.existing_cache}")
    print(f"  Output:         {args.output}")
    print(f"  Layers:         {args.layers}")
    print(f"  Device:         {device}")
    print(f"  Max length:     {args.max_length}")
    print()

    # Detect existing cache format
    print("Detecting existing cache format...")
    fmt = detect_cache_format(args.existing_cache)
    print(f"  Format: {fmt} ({'Llama-3 dict-style' if fmt == 'dict' else 'Vicuna array-style'})")

    # Load model
    print(f"\nLoading model: {args.model}")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  ERROR: transformers not installed")
        sys.exit(1)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except Exception as e:
        print(f"  ERROR loading tokenizer: {e}")
        print(f"  If gated repo, run: huggingface-cli login")
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

    # Validate layers
    n_layers_max = model.config.num_hidden_layers + 1  # +1 for embedding
    if max(args.layers) >= n_layers_max:
        print(f"  ERROR: requested layer {max(args.layers)} >= {n_layers_max}")
        sys.exit(1)

    # Dispatch by format
    if fmt == "dict":
        extract_for_dict_cache(model, tokenizer, args, device)
    else:
        extract_for_array_cache(model, tokenizer, args, device)

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"  Output cache: {args.output}")
    print(f"\n  Next step: re-run experiments with the new cache:")
    print(f"    python verify_saturation.py --llama3_cache {args.output}")
    print(f"    python statistical_tests.py --cache {args.output}")
    print(f"    python vicuna_imbalance_test.py --vicuna_cache {args.output}")


if __name__ == "__main__":
    main()
