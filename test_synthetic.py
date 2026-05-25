"""
test_synthetic.py — Generate a tiny synthetic activation cache to verify
that statistical_tests.py and radial_distribution_check.py work end-to-end.

This is for local testing only — NOT a real evaluation. The user runs the
real scripts on the DGX with the actual activation cache.
"""

import os
import shutil
from pathlib import Path

import numpy as np


def make_synthetic_cache(out_path, d=4096, n_layers_total=9,
                          n_train_ben=200, n_train_atk=200,
                          n_test_ben=100, n_test_atk=100,
                          seed=42):
    """
    Make a fake cache that matches the structure of llama3_activations_cache.npz:

      - hs_train_ben, hs_train_atk, hs_test_ben, hs_test_atk:
            list of dicts { layer_idx -> (T, d) array }
      - cfg_hash: string
    """
    rng = np.random.RandomState(seed)
    layers = sorted({0, 1, 2, 17, 24, 28, 29, 30, 31})
    assert len(layers) == n_layers_total

    def gen(n, label):
        out = []
        # Make benign and attack distributions linearly separable for sanity
        offset_per_layer = {l: rng.randn(d) * 0.1 for l in layers}
        if label == 1:  # attack
            shift = rng.randn(d) * 0.5
        else:
            shift = np.zeros(d)
        for _ in range(n):
            # Each example has T=10 token positions
            T = 10
            d_per_layer = {}
            for l in layers:
                base = rng.randn(T, d) * 0.5 + offset_per_layer[l]
                if label == 1:
                    # Attacks have slight shift in last token at deeper layers
                    if l >= 17:
                        base[-1] += shift
                d_per_layer[l] = base.astype(np.float32)
            out.append(d_per_layer)
        return out

    hs_train_ben = gen(n_train_ben, label=0)
    hs_train_atk = gen(n_train_atk, label=1)
    hs_test_ben = gen(n_test_ben, label=0)
    hs_test_atk = gen(n_test_atk, label=1)

    np.savez(
        out_path,
        hs_train_ben=np.array(hs_train_ben, dtype=object),
        hs_train_atk=np.array(hs_train_atk, dtype=object),
        hs_test_ben=np.array(hs_test_ben, dtype=object),
        hs_test_atk=np.array(hs_test_atk, dtype=object),
        cfg_hash=np.array("synthetic"),
    )
    print(f"  saved synthetic cache → {out_path}")
    print(f"  benign train/test:  {n_train_ben}/{n_test_ben}")
    print(f"  attacks train/test: {n_train_atk}/{n_test_atk}")
    print(f"  layers: {layers}")
    print(f"  d_hidden: {d}")


def main():
    cache_path = Path("results/llama3_activations_cache.npz")
    cache_existed = cache_path.exists()
    backup = None
    if cache_existed:
        backup = cache_path.with_suffix(".npz.real_backup")
        shutil.copy(cache_path, backup)
        print(f"BACKED UP existing cache → {backup}")

    print("Generating synthetic cache for testing...")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Smaller dimensions to make local testing fast
    make_synthetic_cache(cache_path, d=256, n_train_ben=100, n_train_atk=100,
                          n_test_ben=50, n_test_atk=50)

    print()
    print("=" * 70)
    print("Now run the analysis scripts:")
    print()
    print("  python statistical_tests.py --n_seeds 2 --n_bootstrap 500")
    print("  python radial_distribution_check.py --n_seeds 2 \\")
    print("      --total_epochs 20 --epochs_to_check 5 10 20 \\")
    print("      --kappas 0.1 1.0 --skip_plots")
    print()
    print("If they complete without error, the scripts work correctly.")
    print()
    if cache_existed:
        print(f"To restore your real cache:")
        print(f"  mv {backup} {cache_path}")


if __name__ == "__main__":
    main()
