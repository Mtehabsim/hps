"""
HPS Experimental Design Visualization
══════════════════════════════════════
Shows the data composition and how the 80/20 splits are constructed
across the different experiments. Useful for explaining the methodology
to mentors / reviewers.

Output:
  figures/hps_experimental_design.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Load real counts from data files
# ═══════════════════════════════════════════════════════════════════════════

def load_real_counts():
    base = os.path.dirname(os.path.abspath(__file__))
    counts = {}

    # Attack methods + counts from validated_attacks_categorized.json
    cat_path = os.path.join(base, "results", "validated_attacks_categorized.json")
    if os.path.exists(cat_path):
        with open(cat_path) as f:
            cat = json.load(f)
        counts["attacks"] = {method: len(prompts) for method, prompts in cat.items()}
    else:
        print(f"WARNING: {cat_path} not found — using last-known counts")
        counts["attacks"] = {"GCG": 68, "JBC": 84, "PAIR": 66, "prompt_with_random_search": 98}

    # Refused attacks
    refused_path = os.path.join(base, "results", "validated_attacks.json")
    if os.path.exists(refused_path):
        with open(refused_path) as f:
            data = json.load(f)
        counts["refused"] = len(data.get("refused_attacks", []))
    else:
        counts["refused"] = 66

    # Hard benign count from hard_benign module
    try:
        from hard_benign import HARD_BENIGN
        counts["benign_hard"] = len(HARD_BENIGN)
    except ImportError:
        counts["benign_hard"] = 130

    # Clean benign sources (these are configured in dataset.py)
    counts["benign_sources"] = {
        "Alpaca instructions":  100,
        "HumanEval code":        50,
        "GSM8K math":           100,
        "WritingPrompts":        50,
        "Winogrande":            50,
    }
    counts["benign_clean_total"] = sum(counts["benign_sources"].values())
    counts["benign_total"] = counts["benign_clean_total"] + counts["benign_hard"]
    counts["attack_total"] = sum(counts["attacks"].values())
    counts["grand_total"] = counts["benign_total"] + counts["attack_total"]

    # Train/test split (80/20 stratified, seed=42)
    counts["train_benign"] = int(0.8 * counts["benign_total"])
    counts["test_benign"] = counts["benign_total"] - counts["train_benign"]
    counts["train_attacks"] = int(0.8 * counts["attack_total"])
    counts["test_attacks"] = counts["attack_total"] - counts["train_attacks"]

    return counts


REAL = load_real_counts()
print("Loaded real data counts:")
for k, v in REAL.items():
    print(f"  {k}: {v}")


plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "figure.titlesize": 14,
})

# Colors
C_BENIGN_CLEAN = "#5BA659"
C_BENIGN_HARD = "#3B7A38"
C_GCG = "#D62828"
C_JBC = "#F77F00"
C_PAIR = "#7E1F86"
C_RANDOM = "#FCBF49"
C_REFUSED = "#888888"
C_TRAIN = "#2E86AB"
C_TEST = "#E63946"


def make_design_figure():
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.4, 1.4, 1.6], hspace=0.55)

    # ═══════════════════════════════════════════════════════════════════
    #  Panel 1: Dataset composition
    # ═══════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 12)
    ax1.axis("off")
    ax1.set_title("Dataset Composition (Vicuna-13B Validated)",
                  fontweight="bold", loc="left", pad=10)

    # Benign breakdown (left side)
    ax1.text(0, 11, f"BENIGN ({REAL['benign_total']} prompts)",
             fontweight="bold", fontsize=11, color=C_BENIGN_CLEAN)
    benign_segments = []
    for source, count in REAL["benign_sources"].items():
        benign_segments.append((source, count, C_BENIGN_CLEAN))
    benign_segments.append((f"Hard benign\n(security/edgy/sensitive)",
                             REAL["benign_hard"], C_BENIGN_HARD))
    x = 0
    width_scale = 50.0 / REAL["benign_total"]
    for label, count, color in benign_segments:
        w = count * width_scale
        rect = mpatches.Rectangle((x, 8), w, 1.5, facecolor=color, edgecolor="black", linewidth=0.5)
        ax1.add_patch(rect)
        ax1.text(x + w/2, 8.75, str(count), ha="center", va="center", color="white",
                 fontweight="bold", fontsize=9)
        ax1.text(x + w/2, 7.3, label, ha="center", va="top", fontsize=7.5, rotation=0)
        x += w

    # Attack breakdown (right side)
    ax1.text(55, 11, f"ADVERSARIAL ({REAL['attack_total']} confirmed jailbreaks)",
             fontweight="bold", fontsize=11, color=C_GCG)
    attack_label_map = {
        "GCG": ("GCG\n(gradient suffix)", C_GCG),
        "JBC": ("JBC\n(roleplay)", C_JBC),
        "PAIR": ("PAIR\n(paraphrase)", C_PAIR),
        "prompt_with_random_search": ("Random Search\n(adaptive)", C_RANDOM),
    }
    # Order: GCG, JBC, PAIR, Random Search (consistent with paper)
    attack_order = ["GCG", "JBC", "PAIR", "prompt_with_random_search"]
    attack_segments = [(attack_label_map[m][0], REAL["attacks"].get(m, 0),
                         attack_label_map[m][1]) for m in attack_order if m in REAL["attacks"]]
    x = 55
    width_scale_atk = 40.0 / REAL["attack_total"]
    for label, count, color in attack_segments:
        w = count * width_scale_atk
        rect = mpatches.Rectangle((x, 8), w, 1.5, facecolor=color, edgecolor="black", linewidth=0.5)
        ax1.add_patch(rect)
        ax1.text(x + w/2, 8.75, str(count), ha="center", va="center", color="white",
                 fontweight="bold", fontsize=9)
        ax1.text(x + w/2, 7.3, label, ha="center", va="top", fontsize=7.5)
        x += w

    # Refused attacks
    ax1.text(55, 5.0, f"REFUSED ({REAL['refused']}) — analysis only, NOT used in training",
             fontsize=9, color=C_REFUSED, fontweight="bold")
    ax1.add_patch(mpatches.Rectangle((55, 4.0), 8, 0.8, facecolor=C_REFUSED,
                                      edgecolor="black", linewidth=0.5, alpha=0.5))
    ax1.text(59, 4.4, str(REAL['refused']), ha="center", va="center", color="white",
             fontweight="bold", fontsize=9)

    # Total
    ax1.text(50, 1.5,
             f"TOTAL: {REAL['grand_total']} prompts ({REAL['benign_total']} benign + "
             f"{REAL['attack_total']} attacks) used for train/test",
             ha="center", fontsize=10.5, fontweight="bold", style="italic")

    # ═══════════════════════════════════════════════════════════════════
    #  Panel 2: 80/20 split mechanics
    # ═══════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.set_title("How the 80/20 Train/Test Split is Constructed",
                  fontweight="bold", loc="left", pad=10)

    # Visual: stacked bars showing benign and attack split
    # Benign 80/20
    ax2.text(2, 8.6, f"BENIGN ({REAL['benign_total']})", fontweight="bold", fontsize=10, color=C_BENIGN_CLEAN)
    ben_train_w = 80.0 * 0.8
    ben_test_w = 80.0 * 0.2
    ax2.add_patch(mpatches.Rectangle((2, 7), ben_train_w, 1.0, facecolor=C_TRAIN,
                                      edgecolor="black", linewidth=0.7))
    ax2.text(2 + ben_train_w/2, 7.5, f"TRAIN: {REAL['train_benign']} benign (80%)",
             ha="center", va="center", color="white", fontweight="bold", fontsize=10)
    ax2.add_patch(mpatches.Rectangle((2 + ben_train_w, 7), ben_test_w, 1.0,
                                      facecolor=C_TEST, edgecolor="black", linewidth=0.7))
    ax2.text(2 + ben_train_w + ben_test_w/2, 7.5, f"TEST: {REAL['test_benign']}", ha="center",
             va="center", color="white", fontweight="bold", fontsize=10)

    # Attack 80/20
    ax2.text(2, 5.4, f"ADVERSARIAL ({REAL['attack_total']})", fontweight="bold", fontsize=10, color=C_GCG)
    atk_train_w = 80.0 * 0.8
    atk_test_w = 80.0 * 0.2
    ax2.add_patch(mpatches.Rectangle((2, 3.8), atk_train_w, 1.0, facecolor=C_TRAIN,
                                      edgecolor="black", linewidth=0.7))
    ax2.text(2 + atk_train_w/2, 4.3, f"TRAIN: {REAL['train_attacks']} attacks (80%)",
             ha="center", va="center", color="white", fontweight="bold", fontsize=10)
    ax2.add_patch(mpatches.Rectangle((2 + atk_train_w, 3.8), atk_test_w, 1.0,
                                      facecolor=C_TEST, edgecolor="black", linewidth=0.7))
    ax2.text(2 + atk_train_w + atk_test_w/2, 4.3, f"TEST: {REAL['test_attacks']}", ha="center",
             va="center", color="white", fontweight="bold", fontsize=10)

    # Combined train and test
    train_total = REAL['train_benign'] + REAL['train_attacks']
    test_total = REAL['test_benign'] + REAL['test_attacks']
    ax2.text(50, 2.0,
             f"→ Train set: {train_total} prompts ({REAL['train_benign']} benign + {REAL['train_attacks']} attacks)\n"
             f"→ Test set:  {test_total} prompts ({REAL['test_benign']} benign + {REAL['test_attacks']} attacks)",
             ha="center", fontsize=10, fontweight="bold", color="black")
    ax2.text(50, 0.5, "Random stratified split with seed=42 (reproducible across all experiments)",
             ha="center", fontsize=9, style="italic")

    # ═══════════════════════════════════════════════════════════════════
    #  Panel 3: Cross-attack split (different protocol)
    # ═══════════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[2])
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0, 12)
    ax3.axis("off")
    ax3.set_title("Cross-Attack Split (Experiment 8 / 10) — different protocol for generalization test",
                  fontweight="bold", loc="left", pad=10)

    # 4 rows showing 4 cross-attack folds
    fold_y_positions = [9, 6.5, 4, 1.5]
    fold_names = [
        ("GCG", C_GCG, ["JBC", "PAIR", "Random Search"], [C_JBC, C_PAIR, C_RANDOM]),
        ("JBC", C_JBC, ["GCG", "PAIR", "Random Search"], [C_GCG, C_PAIR, C_RANDOM]),
        ("PAIR", C_PAIR, ["GCG", "JBC", "Random Search"], [C_GCG, C_JBC, C_RANDOM]),
        ("Random Search", C_RANDOM, ["GCG", "JBC", "PAIR"], [C_GCG, C_JBC, C_PAIR]),
    ]

    ax3.text(2, 11, "Fold", fontweight="bold", fontsize=9.5)
    ax3.text(13, 11, "TRAIN attacks (3 methods, all data)",
             fontweight="bold", fontsize=9.5, color=C_TRAIN)
    ax3.text(45, 11, "TRAIN benign (80%)",
             fontweight="bold", fontsize=9.5, color=C_TRAIN)
    ax3.text(63, 11, "TEST attacks (held-out method)",
             fontweight="bold", fontsize=9.5, color=C_TEST)
    ax3.text(85, 11, "TEST benign (20%)",
             fontweight="bold", fontsize=9.5, color=C_TEST)

    for fold_idx, (held_out, ho_color, train_methods, train_colors) in enumerate(fold_names):
        y = fold_y_positions[fold_idx]
        # Fold label
        ax3.text(2, y, f"#{fold_idx+1}", fontsize=9.5, fontweight="bold", va="center")

        # Train attacks (3 small colored boxes)
        x = 13
        for tm, tc in zip(train_methods, train_colors):
            ax3.add_patch(mpatches.Rectangle((x, y - 0.4), 8, 0.8,
                                              facecolor=tc, edgecolor="black", linewidth=0.4, alpha=0.8))
            ax3.text(x + 4, y, tm, ha="center", va="center", color="white",
                     fontweight="bold", fontsize=7.5)
            x += 9

        # Train benign (one big box)
        ax3.add_patch(mpatches.Rectangle((45, y - 0.4), 14, 0.8,
                                          facecolor=C_TRAIN, edgecolor="black", linewidth=0.5))
        ax3.text(52, y, f"{REAL['train_benign']} benign", ha="center", va="center", color="white",
                 fontweight="bold", fontsize=8)

        # Test attacks (held-out, big box) — show count
        method_count_map = {"GCG": REAL["attacks"].get("GCG", 0),
                            "JBC": REAL["attacks"].get("JBC", 0),
                            "PAIR": REAL["attacks"].get("PAIR", 0),
                            "Random Search": REAL["attacks"].get("prompt_with_random_search", 0)}
        ho_count = method_count_map.get(held_out, 0)
        ax3.add_patch(mpatches.Rectangle((63, y - 0.4), 18, 0.8,
                                          facecolor=ho_color, edgecolor="black", linewidth=0.5))
        ax3.text(72, y, f"{held_out} ({ho_count})", ha="center", va="center", color="white",
                 fontweight="bold", fontsize=9)

        # Test benign
        ax3.add_patch(mpatches.Rectangle((85, y - 0.4), 12, 0.8,
                                          facecolor=C_TEST, edgecolor="black", linewidth=0.5))
        ax3.text(91, y, f"{REAL['test_benign']} benign", ha="center", va="center", color="white",
                 fontweight="bold", fontsize=8)

    ax3.text(50, -0.5,
             "Each fold trains on 3 attack methods + 80% benign, tests on the held-out method + 20% benign.\n"
             "Reports AUROC and TPR@FPR=1% per fold, then averages.",
             ha="center", fontsize=9.5, style="italic", color="#222")

    fig.suptitle("Experimental Design — Data Composition & Train/Test Splits",
                 fontweight="bold", fontsize=14, y=0.995)

    out_path = os.path.join(OUT_DIR, "hps_experimental_design.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    make_design_figure()
