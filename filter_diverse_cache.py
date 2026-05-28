"""
filter_diverse_cache.py — Remove the 15 contaminated test rows from the
diverse Llama-3 cache.

The diverse cache (created in Phase C) reused the original attack list, so
it still contains the 9 test prompts that overlap with training (after dedup
and split, the unique overlap drops from 15 to 9). This script:

1. Re-derives the train/test split using the deduplicated attacks JSON
2. Identifies which rows in the test set correspond to duplicate prompts
3. Drops those rows from the cache
4. Saves a fully-clean cache

Usage:
  python filter_diverse_cache.py
"""

import os, sys, json, hashlib, argparse
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_cache",
                   default="results/llama3_activations_cache_diverse.npz")
    p.add_argument("--clean_attacks_json", default="llama3_attacks_clean.json")
    p.add_argument("--orig_attacks_json", default="llama3_attacks.json")
    p.add_argument("--output_cache",
                   default="results/llama3_activations_cache_diverse_clean.npz")
    args = p.parse_args()

    print("\n" + "=" * 70)
    print(" FILTER DIVERSE CACHE — drop contaminated test rows")
    print("=" * 70)

    if not os.path.exists(args.input_cache):
        print(f"  ERROR: input cache not found at {args.input_cache}")
        return

    if not os.path.exists(args.orig_attacks_json):
        print(f"  ERROR: original attacks JSON not found at {args.orig_attacks_json}")
        return

    # Step 1: reproduce the original train/test split (the one used to build the cache)
    print(f"\n  Reproducing original split from {args.orig_attacks_json}...")
    with open(args.orig_attacks_json) as f:
        cat = json.load(f)
    orig_attacks = []
    orig_methods = []
    for m, prompts in cat.items():
        for p in prompts:
            if p:
                orig_attacks.append(p)
                orig_methods.append(m)

    rng = np.random.RandomState(42)
    a_idx = rng.permutation(len(orig_attacks))
    n_tr = int(0.8 * len(a_idx))
    train_atk = [orig_attacks[i] for i in a_idx[:n_tr]]
    test_atk = [orig_attacks[i] for i in a_idx[n_tr:]]
    test_methods = [orig_methods[i] for i in a_idx[n_tr:]]
    print(f"    Original split: {len(train_atk)} train / {len(test_atk)} test")

    # Step 2: identify which test rows are contaminated
    train_hashes = set(hashlib.md5(p.encode()).hexdigest() for p in train_atk)
    contaminated_test_idx = []
    for i, p in enumerate(test_atk):
        h = hashlib.md5(p.encode()).hexdigest()
        if h in train_hashes:
            contaminated_test_idx.append(i)
    print(f"    Contaminated test rows: {len(contaminated_test_idx)}")
    print(f"    Their indices: {contaminated_test_idx[:10]}{'...' if len(contaminated_test_idx) > 10 else ''}")

    # Also identify within-test duplicates (same prompt appears twice in test)
    test_hash_count = {}
    for i, p in enumerate(test_atk):
        h = hashlib.md5(p.encode()).hexdigest()
        test_hash_count.setdefault(h, []).append(i)
    test_internal_dupes = []
    for h, idxs in test_hash_count.items():
        if len(idxs) > 1:
            # Keep first occurrence, drop the rest
            test_internal_dupes.extend(idxs[1:])
    print(f"    Within-test duplicates to drop: {len(test_internal_dupes)}")

    # Combined drop list (sort + dedup)
    drop_idx = sorted(set(contaminated_test_idx + test_internal_dupes))
    keep_idx = [i for i in range(len(test_atk)) if i not in set(drop_idx)]
    print(f"    Total drops: {len(drop_idx)}; keeping {len(keep_idx)} test rows")

    # Per-method breakdown of drops
    method_dropped = {}
    for i in drop_idx:
        m = test_methods[i]
        method_dropped[m] = method_dropped.get(m, 0) + 1
    print(f"    Per-method drops: {method_dropped}")

    # Step 3: load cache and apply filter
    print(f"\n  Loading {args.input_cache}...")
    cache = np.load(args.input_cache, allow_pickle=True)
    hs_train_ben = cache["hs_train_ben"]
    hs_train_atk = cache["hs_train_atk"]
    hs_test_ben = cache["hs_test_ben"]
    hs_test_atk = cache["hs_test_atk"]

    print(f"    Original cache: train {len(hs_train_ben)} ben + {len(hs_train_atk)} atk; "
          f"test {len(hs_test_ben)} ben + {len(hs_test_atk)} atk")

    # Sanity check: cache test_atk should match our reproduced split
    if len(hs_test_atk) != len(test_atk):
        print(f"  WARNING: cache test_atk length ({len(hs_test_atk)}) does not match "
              f"reproduced split length ({len(test_atk)}). Filter may be incorrect.")

    # Apply filter
    keep = np.array(keep_idx, dtype=int)
    keep = keep[keep < len(hs_test_atk)]  # safety bound
    hs_test_atk_clean = hs_test_atk[keep]

    # Also drop duplicates from train (won't affect test eval but matters for re-runs)
    train_hash_count = {}
    for i, p in enumerate(train_atk):
        h = hashlib.md5(p.encode()).hexdigest()
        train_hash_count.setdefault(h, []).append(i)
    train_drop_idx = []
    for h, idxs in train_hash_count.items():
        if len(idxs) > 1:
            train_drop_idx.extend(idxs[1:])
    train_drop_idx = sorted(set(train_drop_idx))
    train_keep_idx = [i for i in range(len(train_atk)) if i not in set(train_drop_idx)]
    print(f"    Train duplicates to drop: {len(train_drop_idx)}; "
          f"keeping {len(train_keep_idx)} train rows")

    keep_tr = np.array(train_keep_idx, dtype=int)
    keep_tr = keep_tr[keep_tr < len(hs_train_atk)]
    hs_train_atk_clean = hs_train_atk[keep_tr]

    # Save cleaned cache (preserving format)
    print(f"\n  Saving cleaned cache to {args.output_cache}...")
    np.savez(
        args.output_cache,
        hs_train_ben=hs_train_ben,
        hs_train_atk=hs_train_atk_clean,
        hs_test_ben=hs_test_ben,
        hs_test_atk=hs_test_atk_clean,
    )
    print(f"    Cleaned cache: train {len(hs_train_ben)} ben + {len(hs_train_atk_clean)} atk; "
          f"test {len(hs_test_ben)} ben + {len(hs_test_atk_clean)} atk")

    # Verify by re-checking contamination on cleaned data
    print("\n  Verification: re-checking contamination on cleaned indices...")
    train_atk_clean = [train_atk[i] for i in train_keep_idx]
    test_atk_clean = [test_atk[i] for i in keep_idx]
    train_hashes_clean = set(hashlib.md5(p.encode()).hexdigest() for p in train_atk_clean)
    test_hashes_clean = set(hashlib.md5(p.encode()).hexdigest() for p in test_atk_clean)
    overlap = train_hashes_clean & test_hashes_clean
    print(f"    Train hashes: {len(train_hashes_clean)}")
    print(f"    Test hashes:  {len(test_hashes_clean)}")
    print(f"    Overlap:      {len(overlap)}")
    if len(overlap) == 0:
        print(f"    SUCCESS: clean cache has zero train/test contamination.")
    else:
        print(f"    WARNING: still {len(overlap)} overlap; check filter logic.")

    # Save metadata
    meta_path = args.output_cache.replace(".npz", ".meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "input_cache": args.input_cache,
            "output_cache": args.output_cache,
            "test_atk_dropped": drop_idx,
            "train_atk_dropped": train_drop_idx,
            "test_methods_after_filter": [test_methods[i] for i in keep_idx],
            "method_drops": method_dropped,
        }, f, indent=2)
    print(f"  Metadata -> {meta_path}")


if __name__ == "__main__":
    main()
