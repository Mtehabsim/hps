"""
build_jbshield_attacks.py — Build attack JSONs from JBShield's public dataset.

JBShield (USENIX Security 2025) released test+calibration data for 9 attack
methods × 5 LLMs at https://github.com/NISPLab/JBShield. Crucially, they
include Vicuna-13B-v1.5 and Llama-3-8B-Instruct — the EXACT same models we
use. This gives us:

  Vicuna-13B: 9 attacks (currently we have only 4)
  Llama-3-8B: 9 attacks for cross-validation against our existing cache
  Plus 3 more LLMs (Mistral-7B, Llama-2-7B, Vicuna-7B) for free.

This script:
  1. Locates the cloned JBShield repo (or asks user to clone it)
  2. Loads test + calibration JSONs for each (model, attack) pair
  3. Produces single attack-JSON per LLM in our cache-compatible format
     (matches the format of llama3_attacks.json: dict[attack_name -> list[str]])

Output (one per LLM):
  results/jbshield_vicuna13b_attacks.json
  results/jbshield_llama3_attacks.json
  results/jbshield_mistral7b_attacks.json (if requested)
  results/jbshield_llama27b_attacks.json (if requested)
  results/jbshield_vicuna7b_attacks.json (if requested)

Usage:
  # Clone JBShield first (one-time):
  git clone https://github.com/NISPLab/JBShield.git /tmp/JBShield

  # Then run this:
  python build_jbshield_attacks.py --jbshield_dir /tmp/JBShield
  python build_jbshield_attacks.py --jbshield_dir /tmp/JBShield \\
      --models vicuna-13b-v1.5 Meta-Llama-3-8B-Instruct
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict


# JBShield's 9 attack names in their data directory
JBSHIELD_ATTACKS = [
    "autodan", "base64", "drattack", "gcg",
    "ijp", "pair", "puzzler", "saa", "zulu",
]

# Known LLM directory names in JBShield (must match what's saved in JSONs)
JBSHIELD_MODELS = {
    "vicuna-13b-v1.5":            "vicuna13b",
    "vicuna-7b-v1.5":             "vicuna7b",
    "Meta-Llama-3-8B-Instruct":   "llama3",
    "Llama-2-7b-chat-hf":         "llama27b",
    "Mistral-7B-Instruct-v0.2":   "mistral7b",
}

# Aliases JBShield might use in the JSON filenames
MODEL_ALIASES = {
    "vicuna-13b-v1.5": [
        "vicuna-13b-v1.5",
        "vicuna_13b_v1.5",
        "vicuna13b",
        "vicuna-13b",
    ],
    "vicuna-7b-v1.5": [
        "vicuna-7b-v1.5",
        "vicuna_7b_v1.5",
        "vicuna7b",
        "vicuna-7b",
    ],
    "Meta-Llama-3-8B-Instruct": [
        "Meta-Llama-3-8B-Instruct",
        "llama-3-8b-instruct",
        "llama3",
        "Llama-3-8B-Instruct",
    ],
    "Llama-2-7b-chat-hf": [
        "Llama-2-7b-chat-hf",
        "llama-2-7b-chat",
        "llama27b",
        "Llama-2-7b-chat",
    ],
    "Mistral-7B-Instruct-v0.2": [
        "Mistral-7B-Instruct-v0.2",
        "mistral-7b-instruct",
        "mistral7b",
        "Mistral-7B-Instruct",
    ],
}


def find_attack_jsons(jbshield_dir, attack, model_aliases):
    """
    For a given (attack, model) pair, find both test and calibration JSONs.
    Returns dict {split_name: filepath} or {} if not found.

    Tries multiple alias patterns since different JBShield versions used
    different filename conventions.
    """
    attack_dir = Path(jbshield_dir) / "data" / "jailbreak" / attack
    if not attack_dir.exists():
        return {}

    found = {}
    for split in ["test", "calibration"]:
        for alias in model_aliases:
            for pat in [
                f"{alias}_{split}.json",
                f"{alias}_{split}.csv",
                f"{alias}_{split}.txt",
            ]:
                cand = attack_dir / pat
                if cand.exists():
                    found[split] = cand
                    break
            if split in found:
                break
    return found


def load_prompts_from_file(filepath):
    """
    Load prompts from a JBShield data file. Supports JSON (list of str OR list
    of dicts) and CSV (column-based).

    Returns list[str] of prompt strings.
    """
    suffix = filepath.suffix.lower()
    prompts = []

    if suffix == ".json":
        try:
            with open(filepath) as f:
                data = json.load(f)
        except Exception as e:
            print(f"    ERROR loading {filepath}: {e}")
            return []

        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    prompts.append(item)
                elif isinstance(item, dict):
                    # Try common keys in priority order
                    for key in ["prompt", "jailbreak_prompt", "goal",
                                "input", "text", "attack", "harmful_prompt"]:
                        if key in item and isinstance(item[key], str):
                            prompts.append(item[key])
                            break
        elif isinstance(data, dict):
            # Sometimes a single dict containing a list under a key
            for key in ["prompts", "data", "test", "calibration"]:
                if key in data and isinstance(data[key], list):
                    for it in data[key]:
                        if isinstance(it, str):
                            prompts.append(it)
                        elif isinstance(it, dict):
                            for k in ["prompt", "jailbreak_prompt"]:
                                if k in it and isinstance(it[k], str):
                                    prompts.append(it[k])
                                    break
                    break

    elif suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(filepath)
        # Try common column names
        for col in ["prompt", "jailbreak_prompt", "goal", "input", "text"]:
            if col in df.columns:
                prompts = df[col].dropna().astype(str).tolist()
                break
        if not prompts and len(df.columns) >= 1:
            # First non-numeric column
            for col in df.columns:
                if df[col].dtype == object:
                    prompts = df[col].dropna().astype(str).tolist()
                    break

    elif suffix == ".txt":
        with open(filepath) as f:
            prompts = [line.rstrip("\n") for line in f if line.strip()]

    # Filter empty / very short strings
    prompts = [p for p in prompts if isinstance(p, str) and len(p.strip()) >= 5]
    return prompts


def build_attacks_for_model(jbshield_dir, model_name, attack_list,
                              include_calibration=True, verbose=True):
    """
    Build a single dict[attack_name -> list[str]] for one LLM.

    Args:
        jbshield_dir: path to cloned JBShield repo
        model_name: canonical model name (key in MODEL_ALIASES)
        attack_list: list of attack names to include
        include_calibration: combine test+calibration if True
    """
    aliases = MODEL_ALIASES.get(model_name, [model_name])
    out = {}
    stats = {}

    for attack in attack_list:
        files = find_attack_jsons(jbshield_dir, attack, aliases)
        prompts = []

        if "test" in files:
            test_prompts = load_prompts_from_file(files["test"])
            prompts.extend(test_prompts)
            if verbose:
                print(f"    {attack}/test:  {len(test_prompts):4d} prompts "
                      f"from {files['test'].name}")

        if include_calibration and "calibration" in files:
            cal_prompts = load_prompts_from_file(files["calibration"])
            prompts.extend(cal_prompts)
            if verbose:
                print(f"    {attack}/cal:   {len(cal_prompts):4d} prompts "
                      f"from {files['calibration'].name}")

        # Deduplicate within this attack
        seen = set()
        unique = []
        for p in prompts:
            if p not in seen:
                seen.add(p)
                unique.append(p)

        out[attack] = unique
        stats[attack] = {
            "n_test": len(load_prompts_from_file(files["test"])) if "test" in files else 0,
            "n_calibration": (len(load_prompts_from_file(files["calibration"]))
                              if include_calibration and "calibration" in files else 0),
            "n_unique": len(unique),
            "files_found": list(files.keys()),
        }

        if not unique and verbose:
            print(f"    WARNING: no prompts found for {attack} on {model_name}")

    return out, stats


def length_stats(prompts):
    """Compute length statistics for a list of prompts."""
    if not prompts:
        return {}
    lengths = [len(p) for p in prompts]
    import numpy as np
    return {
        "n": len(lengths),
        "mean_chars": float(np.mean(lengths)),
        "median_chars": float(np.median(lengths)),
        "min_chars": int(np.min(lengths)),
        "max_chars": int(np.max(lengths)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jbshield_dir", required=True,
                        help="Path to cloned JBShield repo "
                             "(should contain data/jailbreak/...)")
    parser.add_argument("--models", nargs="+",
                        default=["vicuna-13b-v1.5", "Meta-Llama-3-8B-Instruct"],
                        help="Canonical model names (matching MODEL_ALIASES keys)")
    parser.add_argument("--attacks", nargs="+", default=JBSHIELD_ATTACKS,
                        help="Attack names to extract")
    parser.add_argument("--output_dir", default="results",
                        help="Where to save attack JSONs")
    parser.add_argument("--no_calibration", action="store_true",
                        help="Use test split only (don't combine with calibration)")
    args = parser.parse_args()

    jbshield_dir = Path(args.jbshield_dir).expanduser().resolve()
    if not jbshield_dir.exists():
        print(f"ERROR: JBShield directory not found at {jbshield_dir}")
        print("Clone with: git clone https://github.com/NISPLab/JBShield.git")
        sys.exit(1)

    data_dir = jbshield_dir / "data" / "jailbreak"
    if not data_dir.exists():
        print(f"ERROR: data/jailbreak not found in {jbshield_dir}")
        print("Expected layout: <jbshield_dir>/data/jailbreak/<attack>/...")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("JBShield Attack Dataset Builder")
    print("=" * 70)
    print(f"  JBShield repo:    {jbshield_dir}")
    print(f"  Models:           {args.models}")
    print(f"  Attacks:          {args.attacks}")
    print(f"  Include cal:      {not args.no_calibration}")
    print(f"  Output dir:       {args.output_dir}")
    print()

    overall_stats = {}

    for model_name in args.models:
        print(f"\n  Building attacks for: {model_name}")
        print("  " + "-" * 50)

        attacks, stats = build_attacks_for_model(
            jbshield_dir, model_name, args.attacks,
            include_calibration=not args.no_calibration,
        )

        # Per-attack length stats
        per_attack_lengths = {a: length_stats(p) for a, p in attacks.items()}

        # Save model JSON
        slug = model_name.lower().replace("-", "").replace("_", "")
        slug = slug.replace(".", "")
        # Make slug similar to "vicuna13bv15"
        for old, new in [("vicuna13bv15", "vicuna13b"),
                         ("vicuna7bv15", "vicuna7b"),
                         ("metallama38binstruct", "llama3"),
                         ("llama27bchathf", "llama27b"),
                         ("mistral7binstructv02", "mistral7b")]:
            if slug.startswith(old):
                slug = new
                break

        out_path = Path(args.output_dir) / f"jbshield_{slug}_attacks.json"
        with open(out_path, "w") as f:
            json.dump(attacks, f, indent=2)
        print(f"\n  Saved: {out_path}")

        total = sum(len(v) for v in attacks.values())
        print(f"  Total prompts: {total}")
        print(f"  Per attack:")
        for a in args.attacks:
            n = len(attacks.get(a, []))
            ls = per_attack_lengths.get(a, {})
            mean_len = ls.get("mean_chars", 0.0)
            print(f"    {a:12s}: n={n:4d}  mean_len={mean_len:7.1f} chars")

        overall_stats[model_name] = {
            "output_file": str(out_path),
            "total_prompts": total,
            "per_attack": stats,
            "per_attack_lengths": per_attack_lengths,
        }

    # Save overall stats
    stats_path = Path(args.output_dir) / "jbshield_build_stats.json"
    with open(stats_path, "w") as f:
        json.dump(overall_stats, f, indent=2)
    print(f"\n  Stats saved: {stats_path}")

    print("\n" + "=" * 70)
    print("DONE — next steps")
    print("=" * 70)
    print()
    print("  Extract Vicuna-13B activations on JBShield attacks:")
    print("  python extract_jbshield_activations.py \\")
    print("    --model lmsys/vicuna-13b-v1.5 \\")
    print("    --attacks_json results/jbshield_vicuna13b_attacks.json \\")
    print("    --diverse_benign results/data_harmless_diverse.csv \\")
    print("    --output results/vicuna_activations_cache_jbshield.npz \\")
    print("    --layers 0 2 22 31 35 39")
    print()
    print("  Same for Llama-3-8B-Instruct (cross-validation):")
    print("  python extract_jbshield_activations.py \\")
    print("    --model meta-llama/Meta-Llama-3-8B-Instruct \\")
    print("    --attacks_json results/jbshield_llama3_attacks.json \\")
    print("    --diverse_benign results/data_harmless_diverse.csv \\")
    print("    --output results/llama3_activations_cache_jbshield.npz \\")
    print("    --layers 0 2 17 24 28 31")
    print()


if __name__ == "__main__":
    main()
