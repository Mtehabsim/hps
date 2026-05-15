"""
TEST 1 — Gromov Delta Hyperbolicity
═══════════════════════════════════
QUESTION: Is the residual stream intrinsically hyperbolic?

HOW IT WORKS
  For each layer, we treat prompt activations as points in a metric space.
  We compute pairwise Euclidean distances, then calculate the Gromov δ-hyperbolicity.

  A metric space is δ-hyperbolic if every geodesic triangle is δ-slim:
      d(x,y) + d(z,w) ≤ max(d(x,z)+d(y,w), d(x,w)+d(y,z)) + 2δ

  δ ≈ 0  →  tree-like (hyperbolic)  →  Poincaré projection is valid
  δ large →  flat (Euclidean-like)  →  projection adds noise, not signal

EXPECTED RESULT
  Token embeddings are known to be mildly hyperbolic (δ < 1).
  If residual stream layers show small δ, the framework is geometrically justified.

OUTPUTS
  results/test1_gromov_delta.json
  plots/test1_gromov_delta_by_layer.png
  plots/test1_gromov_delta_by_category.png

RUN
  python test1_gromov_delta.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import combinations
from sklearn.decomposition import PCA

import config
from utils import load_model, extract_activations_batch, save_json
from dataset import BENIGN, ADVERSARIAL, DUAL_USE, LABEL_COLORS


# ─────────────────────────────────────────────────────────────────────────────
#  Core math
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_euclidean(vectors: list[np.ndarray]) -> np.ndarray:
    """Return an n×n symmetric distance matrix."""
    n = len(vectors)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(vectors[i] - vectors[j]))
            D[i, j] = D[j, i] = d
    return D


def gromov_delta_exact(D: np.ndarray, max_samples: int = 2000) -> dict:
    """
    Compute the Gromov δ-hyperbolicity from a distance matrix D.

    For each 4-tuple (i,j,k,l), the four-point condition gives:
        δ = (s1 - s2) / 2   where s1 ≥ s2 ≥ s3 are the three pairwise sums.

    Returns the maximum δ across all sampled 4-tuples.
    """
    n = len(D)
    all_quads = list(combinations(range(n), 4))

    # Sample if too many
    rng = np.random.default_rng(42)
    if len(all_quads) > max_samples:
        indices = rng.choice(len(all_quads), max_samples, replace=False)
        quads = [all_quads[i] for i in indices]
    else:
        quads = all_quads

    max_delta = 0.0
    delta_values = []

    for (i, j, k, l) in quads:
        s1 = D[i, j] + D[k, l]
        s2 = D[i, k] + D[j, l]
        s3 = D[i, l] + D[j, k]
        sums = sorted([s1, s2, s3], reverse=True)
        delta = (sums[0] - sums[1]) / 2.0
        delta_values.append(delta)
        max_delta = max(max_delta, delta)

    return {
        "max_delta":  float(max_delta),
        "mean_delta": float(np.mean(delta_values)),
        "std_delta":  float(np.std(delta_values)),
        "n_quads":    len(quads),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  PCA compression (to keep distances meaningful across high-dim layers)
# ─────────────────────────────────────────────────────────────────────────────

def compress(vectors: list[np.ndarray], n_components: int = 64) -> list[np.ndarray]:
    """Reduce dimensionality before computing distances (speeds up and stabilises)."""
    mat = np.stack(vectors)
    n_comp = min(n_components, mat.shape[0] - 1, mat.shape[1])
    if n_comp < 2:
        return vectors
    pca = PCA(n_components=n_comp, random_state=42)
    reduced = pca.fit_transform(mat)
    return [reduced[i] for i in range(len(vectors))]


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_delta_by_layer(layer_results: dict, save_path: str):
    """Line plot of max Gromov δ across layer indices for each category."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Test 1 — Gromov δ-Hyperbolicity Across Layers", fontsize=14, fontweight="bold")

    categories = list(layer_results.keys())
    layers = sorted(next(iter(layer_results.values())).keys())

    colors = {"benign": LABEL_COLORS[0], "adversarial": LABEL_COLORS[1], "dual_use": LABEL_COLORS[2]}
    markers = {"benign": "o", "adversarial": "s", "dual_use": "^"}

    # ── Left: max delta ──
    ax = axes[0]
    for cat in categories:
        ys = [layer_results[cat][l]["max_delta"] for l in layers]
        ax.plot(layers, ys, marker=markers.get(cat, "o"),
                color=colors.get(cat, "gray"), label=cat.capitalize(), linewidth=2)
    ax.axhline(0.0, color="black", linestyle="--", alpha=0.3, label="δ=0 (perfect tree)")
    ax.axhline(1.0, color="red",   linestyle=":",  alpha=0.4, label="δ=1 (rough guide for hyperbolic)")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Max Gromov δ")
    ax.set_title("Max δ (lower = more tree-like)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Right: mean delta with std bands ──
    ax = axes[1]
    for cat in categories:
        ys    = np.array([layer_results[cat][l]["mean_delta"] for l in layers])
        stds  = np.array([layer_results[cat][l]["std_delta"]  for l in layers])
        ax.plot(layers, ys, marker=markers.get(cat, "o"),
                color=colors.get(cat, "gray"), label=cat.capitalize(), linewidth=2)
        ax.fill_between(layers, ys - stds, ys + stds,
                        alpha=0.15, color=colors.get(cat, "gray"))
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Mean Gromov δ (± 1 std)")
    ax.set_title("Mean δ with variance bands")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[test1] Plot saved → {save_path}")


def plot_heatmap(layer_results: dict, save_path: str):
    """Heatmap: rows = categories, columns = layers."""
    categories = list(layer_results.keys())
    layers = sorted(next(iter(layer_results.values())).keys())

    data = np.array([[layer_results[cat][l]["max_delta"] for l in layers]
                     for cat in categories])

    fig, ax = plt.subplots(figsize=(max(8, len(layers)), 3))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(layers)));   ax.set_xticklabels(layers)
    ax.set_yticks(range(len(categories))); ax.set_yticklabels([c.capitalize() for c in categories])
    ax.set_xlabel("Layer Index")
    ax.set_title("Gromov δ Heatmap (Max) — darker = more Euclidean")
    plt.colorbar(im, ax=ax, label="Max δ")
    for i in range(len(categories)):
        for j in range(len(layers)):
            ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[test1] Heatmap saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 60)
    print("  TEST 1 — Gromov Delta Hyperbolicity")
    print("═" * 60)

    model, tokenizer = load_model(config.MODEL_NAME, config.DEVICE, config.DTYPE)

    prompt_groups = {
        "benign":      BENIGN[:15],
        "adversarial": ADVERSARIAL[:15],
        "dual_use":    DUAL_USE[:12],
    }

    all_results = {}
    layer_results = {cat: {} for cat in prompt_groups}

    for cat, prompts in prompt_groups.items():
        print(f"\n[test1] Extracting activations for '{cat}' ({len(prompts)} prompts)…")
        act_list = extract_activations_batch(
            model, tokenizer, prompts, config.TARGET_LAYERS, config.DEVICE
        )

        for layer_idx in config.TARGET_LAYERS:
            vecs = [a[layer_idx] for a in act_list if layer_idx in a]
            if len(vecs) < 4:
                print(f"  [skip] layer {layer_idx}: not enough vectors ({len(vecs)})")
                continue

            # Compress to 64-D before computing distances
            vecs_compressed = compress(vecs, n_components=64)
            D = pairwise_euclidean(vecs_compressed)
            stats = gromov_delta_exact(D)
            layer_results[cat][layer_idx] = stats

            print(f"  Layer {layer_idx:2d} | max_δ={stats['max_delta']:.4f} "
                  f"| mean_δ={stats['mean_delta']:.4f} | n_quads={stats['n_quads']}")

    # ── Summary interpretation ──
    print("\n[test1] INTERPRETATION")
    for cat in prompt_groups:
        deltas = [layer_results[cat][l]["max_delta"] for l in layer_results[cat]]
        if deltas:
            avg = np.mean(deltas)
            verdict = "HYPERBOLIC (projection likely valid)" if avg < 1.0 else \
                      "MILDLY HYPERBOLIC"                   if avg < 2.0 else \
                      "FLAT (Euclidean-like, projection may add noise)"
            print(f"  {cat:15s}: avg max_δ = {avg:.3f}  →  {verdict}")

    # ── Save ──
    all_results = {
        "config": {
            "model":  config.MODEL_NAME,
            "layers": config.TARGET_LAYERS,
            "groups": {k: len(v) for k, v in prompt_groups.items()},
        },
        "layer_results": {
            cat: {str(l): stats for l, stats in lyr.items()}
            for cat, lyr in layer_results.items()
        },
    }
    save_json(all_results, "test1_gromov_delta.json", config.RESULTS_DIR)

    # ── Plot ──
    plot_delta_by_layer(
        {cat: {l: layer_results[cat][l] for l in layer_results[cat]} for cat in layer_results},
        os.path.join(config.PLOTS_DIR, "test1_gromov_delta_by_layer.png"),
    )
    plot_heatmap(
        {cat: {l: layer_results[cat][l] for l in layer_results[cat]} for cat in layer_results},
        os.path.join(config.PLOTS_DIR, "test1_gromov_delta_heatmap.png"),
    )

    print("\n[test1] Done. ✓")
    return layer_results


if __name__ == "__main__":
    main()
