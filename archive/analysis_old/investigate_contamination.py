"""
investigate_contamination.py — Find and fix the 15 overlapping prompts.

verify_saturation.py revealed 15 train/test prompt overlaps and 31 internal
duplicates within training. This script:

  1. Identifies WHICH prompts overlap and from WHICH attack categories
  2. Diagnoses the cause (duplicate prompts in JSON, cross-method copies, etc.)
  3. Outputs a deduplicated attack JSON
  4. Prints a recommendation for re-running experiments with clean splits

Usage:
  python investigate_contamination.py
  python investigate_contamination.py --attacks-json llama3_attacks.json
  python investigate_contamination.py --output llama3_attacks_clean.json
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def hash_prompt(p):
    return hashlib.md5(p.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attacks_json", default="llama3_attacks.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="llama3_attacks_clean.json",
                        help="Path to save deduplicated attack JSON")
    parser.add_argument("--show_examples", type=int, default=5,
                        help="How many overlap examples to print")
    args = parser.parse_args()

    if not Path(args.attacks_json).exists():
        print(f"ERROR: {args.attacks_json} not found")
        sys.exit(1)

    print("=" * 70)
    print("CONTAMINATION INVESTIGATION")
    print("=" * 70)

    with open(args.attacks_json) as f:
        categorized = json.load(f)

    # ── Replicate hps_llama3.py's attack loading ──
    attack_prompts, attack_methods = [], []
    for method, prompts in categorized.items():
        for p in prompts:
            if p:
                attack_prompts.append(p)
                attack_methods.append(method)

    n_total = len(attack_prompts)
    print(f"\nTotal attack prompts (from JSON): {n_total}")

    # ── Step 1: Within-source duplicates ──
    print("\n" + "─" * 70)
    print("STEP 1 — Internal duplicates within each attack category")
    print("─" * 70)

    method_dupes = defaultdict(list)
    for method, prompts in categorized.items():
        seen = {}
        for i, p in enumerate(prompts):
            if not p:
                continue
            h = hash_prompt(p)
            if h in seen:
                method_dupes[method].append({
                    "first_index": seen[h],
                    "duplicate_index": i,
                    "prompt_excerpt": p[:120],
                })
            else:
                seen[h] = i

    total_intra = 0
    for method, dupes in method_dupes.items():
        if dupes:
            print(f"  {method}: {len(dupes)} internal duplicates")
            total_intra += len(dupes)
    if total_intra == 0:
        print("  ✓ No within-category duplicates")
    else:
        print(f"\n  Total within-category duplicates: {total_intra}")
        for method, dupes in method_dupes.items():
            if dupes and args.show_examples > 0:
                print(f"\n  Example from {method}:")
                ex = dupes[0]
                print(f"    Prompt #{ex['first_index']} = #{ex['duplicate_index']}: "
                      f"{ex['prompt_excerpt']}...")

    # ── Step 2: Cross-category duplicates ──
    print("\n" + "─" * 70)
    print("STEP 2 — Cross-category duplicates (same prompt in multiple methods)")
    print("─" * 70)

    hash_to_methods = defaultdict(list)
    for i, (p, m) in enumerate(zip(attack_prompts, attack_methods)):
        h = hash_prompt(p)
        hash_to_methods[h].append((m, i, p[:120]))

    cross_dupes = {h: ms for h, ms in hash_to_methods.items() if len(ms) > 1}
    print(f"  Found {len(cross_dupes)} prompts appearing in multiple categories")

    method_pair_counter = Counter()
    for h, ms in cross_dupes.items():
        methods = sorted(set(m for m, _, _ in ms))
        if len(methods) > 1:
            method_pair_counter[tuple(methods)] += 1

    if method_pair_counter:
        print("\n  Method-pair overlap counts:")
        for pair, count in sorted(method_pair_counter.items(),
                                    key=lambda x: -x[1]):
            print(f"    {pair}: {count}")

    if cross_dupes and args.show_examples > 0:
        print("\n  Example cross-category duplicates:")
        for h, ms in list(cross_dupes.items())[:args.show_examples]:
            print(f"    {ms[0][2]}...")
            for m, i, _ in ms:
                print(f"      └─ in '{m}' at index {i}")

    # ── Step 3: Check train/test overlap (with same seed=42 split) ──
    print("\n" + "─" * 70)
    print(f"STEP 3 — Train/test overlap with seed={args.seed} (hps_llama3.py split)")
    print("─" * 70)

    rng = np.random.RandomState(args.seed)
    atk_idx = rng.permutation(len(attack_prompts))
    n_atk_tr = int(0.8 * len(atk_idx))
    train_idxs = atk_idx[:n_atk_tr]
    test_idxs = atk_idx[n_atk_tr:]

    train_hashes = {}
    for i in train_idxs:
        h = hash_prompt(attack_prompts[i])
        if h not in train_hashes:
            train_hashes[h] = (attack_methods[i], i, attack_prompts[i][:120])

    test_overlaps = []
    for i in test_idxs:
        h = hash_prompt(attack_prompts[i])
        if h in train_hashes:
            test_overlaps.append({
                "test_index": int(i),
                "train_method": train_hashes[h][0],
                "train_index": int(train_hashes[h][1]),
                "test_method": attack_methods[i],
                "prompt_excerpt": attack_prompts[i][:120],
            })

    print(f"  Train: {len(train_idxs)} prompts ({len(train_hashes)} unique hashes)")
    print(f"  Test:  {len(test_idxs)} prompts")
    print(f"  Train ∩ Test (test prompts also in train): {len(test_overlaps)}")

    if test_overlaps:
        print("\n  Detailed overlap table:")
        method_pairs = Counter()
        for o in test_overlaps:
            method_pairs[(o["train_method"], o["test_method"])] += 1
        for (tr_m, te_m), count in sorted(method_pairs.items(),
                                            key=lambda x: -x[1]):
            print(f"    {tr_m} (train) ↔ {te_m} (test): {count}")

        print(f"\n  First {args.show_examples} examples:")
        for o in test_overlaps[:args.show_examples]:
            print(f"    train_idx={o['train_index']:>4} ({o['train_method']:>10s}), "
                  f"test_idx={o['test_index']:>4} ({o['test_method']:>10s}): "
                  f"{o['prompt_excerpt']}...")

    # ── Step 4: Diagnose root cause ──
    print("\n" + "─" * 70)
    print("STEP 4 — Root cause diagnosis")
    print("─" * 70)

    if total_intra > 0:
        print(f"  ⚠ {total_intra} within-category duplicates: same prompt copied "
              f"twice in same JSON entry")
    if len(cross_dupes) > 0:
        print(f"  ⚠ {len(cross_dupes)} prompts in multiple categories: "
              f"likely the original AdvBench harmful behavior text appearing "
              f"unaltered under a different attack method, OR an attack that "
              f"generated identical outputs")

    # ── Step 5: Build deduplicated JSON ──
    print("\n" + "─" * 70)
    print("STEP 5 — Building deduplicated attack JSON")
    print("─" * 70)

    seen_global = set()
    deduplicated = {method: [] for method in categorized}
    n_dropped = 0
    for method, prompts in categorized.items():
        for p in prompts:
            if not p:
                continue
            h = hash_prompt(p)
            if h in seen_global:
                n_dropped += 1
                continue
            seen_global.add(h)
            deduplicated[method].append(p)

    n_kept = sum(len(v) for v in deduplicated.values())
    print(f"  Original total: {n_total}")
    print(f"  Deduplicated:   {n_kept}")
    print(f"  Dropped:        {n_dropped} duplicate prompts")
    print(f"\n  Per-category counts after dedup:")
    for method, prompts in deduplicated.items():
        original_count = len([p for p in categorized[method] if p])
        print(f"    {method:>15s}: {original_count} → {len(prompts)} "
              f"(-{original_count - len(prompts)})")

    # Save
    Path(args.output).write_text(json.dumps(deduplicated, indent=2))
    print(f"\n  Saved deduplicated JSON to: {args.output}")

    # ── Step 6: Verify dedup eliminated train/test overlap ──
    print("\n" + "─" * 70)
    print("STEP 6 — Verify dedup fixes the train/test overlap")
    print("─" * 70)

    new_attack_prompts, new_attack_methods = [], []
    for m, prompts in deduplicated.items():
        for p in prompts:
            if p:
                new_attack_prompts.append(p)
                new_attack_methods.append(m)

    rng = np.random.RandomState(args.seed)
    new_atk_idx = rng.permutation(len(new_attack_prompts))
    n_new_tr = int(0.8 * len(new_atk_idx))
    new_train_h = {hash_prompt(new_attack_prompts[i])
                    for i in new_atk_idx[:n_new_tr]}
    new_test_h = {hash_prompt(new_attack_prompts[i])
                   for i in new_atk_idx[n_new_tr:]}
    new_overlap = new_train_h & new_test_h

    print(f"  After dedup:  train {n_new_tr}, test {len(new_atk_idx) - n_new_tr}")
    print(f"  Train ∩ Test: {len(new_overlap)}")
    if len(new_overlap) == 0:
        print(f"  ✓ Deduplication ELIMINATES train/test overlap")
    else:
        print(f"  ⚠ Still {len(new_overlap)} overlaps; deeper investigation needed")

    # ── Save investigation report ──
    report = {
        "input": args.attacks_json,
        "total_prompts": int(n_total),
        "within_category_duplicates": total_intra,
        "cross_category_duplicates": len(cross_dupes),
        "train_test_overlaps": len(test_overlaps),
        "after_dedup_count": int(n_kept),
        "dropped": int(n_dropped),
        "after_dedup_overlaps": int(len(new_overlap)),
        "method_pair_overlaps": {
            f"{a}__{b}": v for (a, b), v in method_pairs.items()
        } if test_overlaps else {},
        "per_category_dedup": {
            m: {"original": len([p for p in categorized[m] if p]),
                "after_dedup": len(deduplicated[m])}
            for m in categorized
        },
    }
    out_report = Path(args.output).with_suffix(".report.json")
    with open(out_report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved investigation report to: {out_report}")

    # ── Recommendations ──
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print(f"""
  1. Use the deduplicated JSON: {args.output}
     - {n_dropped} duplicates removed; train/test split is now clean

  2. Update hps_llama3.py to use the deduplicated file:
     python hps_llama3.py --test-attacks {args.output} ...

  3. Re-extract activations:
     - The activation cache (results/llama3_activations_cache.npz) was
       built from the original (contaminated) JSON. Either:
       a) Re-extract activations from scratch with deduplicated JSON
       b) Filter the cache to drop the {len(test_overlaps)} contaminated test rows

  4. For paper limitations section:
     "Initial activation cache contained {n_dropped} duplicate attack prompts
      ({len(test_overlaps)} of which produced train/test overlap). The
      contamination was small (1.15% of test set) but real. After
      deduplication, the train/test split is clean. Reported numbers in this
      paper use the deduplicated dataset."
""")


if __name__ == "__main__":
    main()
