"""
extract_generation_activations.py — Extract activations across teacher-forced
generation tokens for training HPS-Gen probe.

For each (prompt, harmful_target) pair we forward-pass the model on
chat-templated [prompt + target] and capture hidden states at the layer
positions corresponding to the TARGET tokens. This produces per-sample
(n_layers, n_target_tokens, hidden_dim) arrays that train_hps_gen_probe()
in adaptive_attack.py consumes.

Both benign and harmful prompts use the SAME teacher-forced harmful target
(HARMFUL_TARGET from adaptive_attack.py). This makes the activation
distributions comparable: the probe learns whether a prompt's internal state
DURING the forced harmful target looks like a harmful prompt's natural state
or a benign prompt's natural state.

Output cache schema (NumPy npz):
  hs_train_ben_gen — list of dicts {layer_idx: ndarray (n_target_tokens, hidden_dim)}
  hs_train_atk_gen — same for harmful train prompts
  hs_test_ben_gen  — same for benign test prompts
  hs_test_atk_gen  — same for harmful test prompts
  target_text      — string used as teacher-forced target
  config           — dict of {model_name, layers, max_length, n_target_tokens, ...}

Usage on DGX:
  python extract_generation_activations.py \\
      --model meta-llama/Meta-Llama-3-8B-Instruct \\
      --benign_csv results/data_harmless_diverse.csv \\
      --attack_cache results/llama3_activations_cache_diverse_fixed.npz \\
      --output results/llama3_gen_activations_cache.npz \\
      --layers 0 2 17 24 28 31

  # Quick test (first 100 samples each):
  python extract_generation_activations.py \\
      --max_samples 100 \\
      --output results/llama3_gen_activations_cache_small.npz
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse the canonical harmful target string from the rest of the pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptive_attack import HARMFUL_TARGET


def load_attack_prompts(attacks_json_path, seed=42, train_frac=0.8,
                         fallback_paths=None):
    """Load attack prompts and split train/test.

    Tries the primary attacks_json_path first. If it has 0 prompts (empty
    stub), iterates through fallback_paths and uses the first one with
    >= 50 prompts.

    Files are expected to be dict {attack_name: [prompts...]} (the JBShield
    convention used by extract_jbshield_activations.py).
    """
    if fallback_paths is None:
        fallback_paths = [
            "results/validated_attacks_categorized.json",
            "results/novel_attacks_2025.json",
            "data/extra_attacks.json",
        ]

    paths_to_try = [attacks_json_path] + fallback_paths

    flat = []
    methods = []
    used_path = None

    for path in paths_to_try:
        if not os.path.exists(path):
            print(f"  Skipping {path}: not found")
            continue

        print(f"  Loading attacks from {path}...")
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"    Error: {e}")
            continue

        if not isinstance(data, dict):
            print(f"    Not a dict (got {type(data).__name__}); skipping")
            continue

        # Try direct {name: [prompts]} format first
        candidates_flat = []
        candidates_methods = []
        for attack_name, prompts in data.items():
            if not isinstance(prompts, list):
                continue
            for p in prompts:
                if isinstance(p, str) and len(p.strip()) >= 5:
                    candidates_flat.append(p)
                    candidates_methods.append(attack_name)

        # Handle nested format like validated_attacks.json:
        # {validated_attacks: [{prompt: ..., method: ...}, ...]}
        if not candidates_flat and "validated_attacks" in data:
            va = data["validated_attacks"]
            if isinstance(va, list):
                for item in va:
                    if isinstance(item, dict):
                        prompt = item.get("prompt") or item.get("attack_prompt")
                        method = item.get("method") or item.get("attack") or "unknown"
                        if isinstance(prompt, str) and len(prompt.strip()) >= 5:
                            candidates_flat.append(prompt)
                            candidates_methods.append(method)

        n_loaded = len(candidates_flat)
        print(f"    Found {n_loaded} prompts across "
              f"{len(set(candidates_methods))} methods")

        if n_loaded >= 50:
            flat = candidates_flat
            methods = candidates_methods
            used_path = path
            print(f"  Using {path} as the attack source.")
            break
        elif n_loaded > 0:
            print(f"    Too few prompts ({n_loaded} < 50); trying next source...")

    if not flat:
        raise FileNotFoundError(
            "No usable attack prompt source found. Tried:\n  - "
            + "\n  - ".join(paths_to_try)
            + "\nEach was either missing, empty, or had <50 prompts.\n"
            "Either rebuild jbshield_llama3_attacks.json (build_jbshield_attacks.py)\n"
            "or pass --attacks_json with a valid JBShield-style dict file."
        )

    counts = {m: methods.count(m) for m in sorted(set(methods))}
    print(f"  Total: {len(flat)} prompts from {used_path}")
    for m, n in counts.items():
        print(f"    {m}: {n}")

    # Same split convention as extract_jbshield_activations.py: seed=42, 80/20
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(flat))
    n_train = int(train_frac * len(flat))
    train = [flat[i] for i in perm[:n_train]]
    test = [flat[i] for i in perm[n_train:]]
    print(f"  Split: {len(train)} train, {len(test)} test (seed={seed})")
    return train, test


def load_benign_prompts(benign_csv, train_count, test_count, seed=42):
    """Load and split benign prompts."""
    if not os.path.exists(benign_csv):
        raise FileNotFoundError(f"Benign CSV not found: {benign_csv}")
    prompts = []
    with open(benign_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row and row[0].strip():
                prompts.append(row[0])
    rng = np.random.RandomState(seed)
    rng.shuffle(prompts)
    train = prompts[:train_count]
    test = prompts[train_count:train_count + test_count]
    print(f"  Loaded {len(prompts)} benign prompts; using "
          f"{len(train)} train + {len(test)} test")
    return train, test


def extract_gen_activations(
    model, tokenizer, prompts, harmful_target, layers,
    device, max_length=2048, log_every=200,
):
    """Forward each (prompt + harmful_target) and extract activations at
    target token positions.

    Returns: list of dicts {layer_idx: ndarray (n_target_tokens, hidden_dim)}
    """
    target_ids = tokenizer(
        harmful_target, return_tensors="pt", add_special_tokens=False,
    ).input_ids.to(device)
    n_target = target_ids.shape[1]
    print(f"  Target tokens: {n_target}  ({tokenizer.decode(target_ids[0])!r})")

    results = []
    t_start = time.time()

    for i, prompt in enumerate(prompts):
        with torch.no_grad():
            messages = [{"role": "user", "content": prompt}]
            templated = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            input_ids = tokenizer(
                templated, return_tensors="pt",
                truncation=True, max_length=max_length,
                add_special_tokens=False,  # chat template already has BOS
            ).input_ids.to(device)
            T_input = input_ids.shape[1]

            # Concatenate input + target_ids
            full_ids = torch.cat([input_ids, target_ids], dim=1)
            outputs = model(
                input_ids=full_ids, output_hidden_states=True,
                return_dict=True, use_cache=False,
            )
            # Extract activations at target positions for each requested layer
            sample = {}
            for layer_idx in layers:
                # hidden_states[layer_idx + 1] has shape (1, T_input + n_target, H)
                acts = outputs.hidden_states[layer_idx + 1][
                    0, T_input:T_input + n_target, :
                ].cpu().numpy().astype(np.float32)  # (n_target, H)
                sample[layer_idx] = acts
            results.append(sample)

            del outputs, full_ids, input_ids
            torch.cuda.empty_cache()

        if (i + 1) % log_every == 0 or i == len(prompts) - 1:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (len(prompts) - i - 1)
            print(f"    [{i + 1}/{len(prompts)}] elapsed={elapsed/60:.1f}min "
                  f"eta={eta/60:.1f}min")

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--benign_csv", default="results/data_harmless_diverse.csv")
    p.add_argument("--attacks_json",
                   default="results/jbshield_llama3_attacks.json",
                   help="JBShield attack prompts JSON (dict attack_name -> [prompts])")
    p.add_argument("--attack_cache",
                   default="results/llama3_activations_cache_diverse_fixed.npz",
                   help="(unused, kept for back-compat with earlier signature)")
    p.add_argument("--output", required=True)
    p.add_argument("--layers", type=int, nargs="+", default=[0, 2, 17, 24, 28, 31])
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap number of samples per split (for quick testing)")
    p.add_argument("--torch_dtype", default="float16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_frac", type=float, default=0.8,
                   help="Fraction of data for train split (matches existing cache convention)")
    p.add_argument("--harmful_target", default=HARMFUL_TARGET)
    p.add_argument("--benign_train_count", type=int, default=5000)
    p.add_argument("--benign_test_count", type=int, default=500)
    args = p.parse_args()

    print("=" * 70)
    print("  EXTRACT GENERATION ACTIVATIONS (HPS-Gen training data)")
    print("=" * 70)
    print(f"  Model:           {args.model}")
    print(f"  Benign CSV:      {args.benign_csv}")
    print(f"  Attack cache:    {args.attack_cache}")
    print(f"  Output:          {args.output}")
    print(f"  Layers:          {args.layers}")
    print(f"  Max length:      {args.max_length}")
    print(f"  Max samples:     {args.max_samples or 'all'}")
    print(f"  Harmful target:  {args.harmful_target!r}")
    print()

    # Load model
    print("  Loading model...")
    dtype = torch.float16 if args.torch_dtype == "float16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    device = next(model.parameters()).device
    print(f"  Loaded. Hidden size: {model.config.hidden_size}, "
          f"layers: {model.config.num_hidden_layers}")

    # Load prompts
    print("\n  Loading attack prompts...")
    train_atk, test_atk = load_attack_prompts(
        args.attacks_json, seed=args.seed, train_frac=args.train_frac,
    )
    print("\n  Loading benign prompts...")
    train_ben, test_ben = load_benign_prompts(
        args.benign_csv, args.benign_train_count, args.benign_test_count,
        seed=args.seed,
    )

    if args.max_samples:
        train_ben = train_ben[:args.max_samples]
        test_ben = test_ben[:args.max_samples]
        train_atk = train_atk[:args.max_samples]
        test_atk = test_atk[:args.max_samples]
        print(f"  Capped to {args.max_samples} per split")

    print(f"\n  Final counts:")
    print(f"    train_ben: {len(train_ben)}")
    print(f"    train_atk: {len(train_atk)}")
    print(f"    test_ben:  {len(test_ben)}")
    print(f"    test_atk:  {len(test_atk)}")

    # Extract for each split
    print("\n" + "─" * 70)
    print("  Extracting train_ben...")
    hs_train_ben = extract_gen_activations(
        model, tokenizer, train_ben, args.harmful_target, args.layers,
        device, max_length=args.max_length,
    )
    print("\n  Extracting train_atk...")
    hs_train_atk = extract_gen_activations(
        model, tokenizer, train_atk, args.harmful_target, args.layers,
        device, max_length=args.max_length,
    )
    print("\n  Extracting test_ben...")
    hs_test_ben = extract_gen_activations(
        model, tokenizer, test_ben, args.harmful_target, args.layers,
        device, max_length=args.max_length,
    )
    print("\n  Extracting test_atk...")
    hs_test_atk = extract_gen_activations(
        model, tokenizer, test_atk, args.harmful_target, args.layers,
        device, max_length=args.max_length,
    )

    # Save
    print(f"\n  Saving to {args.output} ...")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    config = {
        "model": args.model,
        "layers": args.layers,
        "max_length": args.max_length,
        "harmful_target": args.harmful_target,
        "seed": args.seed,
    }
    np.savez_compressed(
        args.output,
        hs_train_ben_gen=np.array(hs_train_ben, dtype=object),
        hs_train_atk_gen=np.array(hs_train_atk, dtype=object),
        hs_test_ben_gen=np.array(hs_test_ben, dtype=object),
        hs_test_atk_gen=np.array(hs_test_atk, dtype=object),
        target_text=args.harmful_target,
        config=json.dumps(config),
    )
    print(f"  Done. File size: {os.path.getsize(args.output) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
