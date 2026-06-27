"""
TEST 3 — TRACED Space & Trajectory Kinematics
══════════════════════════════════════════════
QUESTION: Do adversarial prompts produce geometrically distinct trajectories?

HOW IT WORKS
  For each prompt we extract one activation vector per sampled layer,
  forming a trajectory through activation space.

  We then compute two summary statistics per trajectory:
    • Displacement (Progress)  = total straight-line travel from layer 0 → last
    • Max Curvature (Stability) = peak bending along the path

  Plotting (Displacement, Curvature) — the "TRACED space" — should reveal
  four quadrants:
    ┌──────────────────────┬──────────────────────┐
    │ Low disp, High curv  │ High disp, High curv  │
    │   Hallucination      │   Jailbreak Pivot     │
    ├──────────────────────┼──────────────────────┤
    │ Low disp, Low curv   │ High disp, Low curv   │
    │   Stagnation         │   Correct Reasoning   │
    └──────────────────────┴──────────────────────┘

  We compute this in BOTH Euclidean and Hyperbolic (Lorentz) space and compare.
  Additionally we render individual trajectory plots for example prompts.

OUTPUTS
  results/test3_traced_space.json
  plots/test3_traced_euclidean.png   — Euclidean TRACED scatter
  plots/test3_traced_hyperbolic.png  — Hyperbolic TRACED scatter
  plots/test3_example_trajectories.png — per-layer curvature profiles

RUN
  python test3_traced_space.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

import config
from utils import (
    load_model, extract_activations_batch,
    euclidean_curvature, hyperbolic_curvature,
    to_lorentz, compute_displacement,
    save_json,
)
from dataset import BENIGN, ADVERSARIAL, DUAL_USE, LABEL_COLORS, LABEL_NAMES


# ─────────────────────────────────────────────────────────────────────────────
#  Feature extraction from a trajectory
# ─────────────────────────────────────────────────────────────────────────────

def trajectory_features_euclidean(act_dict: dict, layer_indices: list[int]) -> dict:
    points = [act_dict[l] for l in layer_indices if l in act_dict]
    if len(points) < 3:
        return {}
    curv  = euclidean_curvature(points)
    disp, path = compute_displacement(points)
    return {
        "max_curvature":  float(curv.max()),
        "mean_curvature": float(curv.mean()),
        "std_curvature":  float(curv.std()),
        "displacement":   disp,
        "path_length":    path,
        "progress_ratio": disp / (path + 1e-8),
        "spike_layer":    int(np.argmax(curv)),         # which layer the spike is at
        "curvature_profile": curv.tolist(),
    }


def trajectory_features_hyperbolic(act_dict: dict, layer_indices: list[int], k: float = 1.0) -> dict:
    points = [act_dict[l] for l in layer_indices if l in act_dict]
    if len(points) < 3:
        return {}
    # Lift to Lorentz manifold — to_lorentz handles scaling internally
    lorentz_pts = [to_lorentz(p, k=k) for p in points]
    curv  = hyperbolic_curvature(lorentz_pts, k=k)
    # Radial coordinates (time-coord on hyperboloid = abstraction depth proxy)
    radii = [float(lp[0]) for lp in lorentz_pts]
    return {
        "max_curvature":  float(curv.max()),
        "mean_curvature": float(curv.mean()),
        "std_curvature":  float(curv.std()),
        "mean_radius":    float(np.mean(radii)),
        "max_radius":     float(np.max(radii)),
        "radius_range":   float(np.max(radii) - np.min(radii)),
        "spike_layer":    int(np.argmax(curv)),
        "curvature_profile": curv.tolist(),
        "radius_profile":    radii,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  TRACED scatter plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_traced_space(records: list[dict], mode: str, save_path: str):
    """
    Scatter plot in (displacement, max_curvature) space.
    mode: "euclidean" or "hyperbolic"
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    cat_map = {0: "benign", 1: "adversarial", 2: "dual_use"}
    for label in sorted(set(r["label"] for r in records)):
        subset = [r for r in records if r["label"] == label]
        feat   = [r[mode] for r in subset if mode in r and r[mode]]
        if not feat:
            continue
        xs = [f.get("displacement",   f.get("mean_radius", 0)) for f in feat]
        ys = [f["max_curvature"] for f in feat]
        ax.scatter(xs, ys,
                   c=LABEL_COLORS.get(label, "gray"),
                   s=60, alpha=0.7, zorder=3,
                   label=LABEL_NAMES.get(label, str(label)))

    # Quadrant dividers
    all_feat = [r[mode] for r in records if mode in r and r[mode]]
    if all_feat:
        med_x = np.median([f.get("displacement", f.get("mean_radius", 0)) for f in all_feat])
        med_y = np.median([f["max_curvature"] for f in all_feat])
        ax.axvline(med_x, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(med_y, color="gray", linestyle="--", alpha=0.5)
        # Quadrant labels
        ax.text(0.02, 0.97, "Stagnation", transform=ax.transAxes,
                va="top", ha="left", fontsize=8, alpha=0.6)
        ax.text(0.97, 0.97, "Hallucination /\nJailbreak", transform=ax.transAxes,
                va="top", ha="right", fontsize=8, alpha=0.6)
        ax.text(0.02, 0.03, "Stagnant", transform=ax.transAxes,
                va="bottom", ha="left", fontsize=8, alpha=0.6)
        ax.text(0.97, 0.03, "Correct\nReasoning", transform=ax.transAxes,
                va="bottom", ha="right", fontsize=8, alpha=0.6)

    x_label = "Displacement (progress)" if mode == "euclidean" else "Mean Lorentz Radius (abstraction depth)"
    ax.set_xlabel(x_label)
    ax.set_ylabel("Max Curvature (instability)")
    ax.set_title(f"Test 3 — TRACED Space ({mode.capitalize()})\n"
                 f"Adversarial prompts should cluster: high curvature",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[test3] TRACED plot saved → {save_path}")


def plot_curvature_profiles(example_records: list[dict], save_path: str):
    """Per-layer curvature profiles for a handful of example prompts."""
    n_modes = 2  # euclidean + hyperbolic
    n_examples = min(len(example_records), 6)
    fig, axes = plt.subplots(n_examples, n_modes, figsize=(12, n_examples * 2.2))
    if n_examples == 1:
        axes = [axes]

    for row, rec in enumerate(example_records[:n_examples]):
        prompt_short = rec["prompt"][:55] + "…"
        label_name   = LABEL_NAMES.get(rec["label"], "?")
        color        = LABEL_COLORS.get(rec["label"], "gray")

        for col, mode in enumerate(["euclidean", "hyperbolic"]):
            ax  = axes[row][col]
            feat = rec.get(mode, {})
            profile = feat.get("curvature_profile", [])
            if profile:
                ax.plot(profile, color=color, linewidth=1.5, marker="o", markersize=4)
                ax.axhline(np.mean(profile), color="gray", linestyle="--", alpha=0.5, label="mean")
            ax.set_title(f"[{label_name}] {prompt_short}\n({mode})", fontsize=7)
            ax.set_xlabel("Layer interval", fontsize=7)
            ax.set_ylabel("κ", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)

    plt.suptitle("Test 3 — Per-Layer Curvature Profiles", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[test3] Curvature profiles saved → {save_path}")


def plot_curvature_distribution(records: list[dict], save_path: str):
    """Histogram of max curvature values per category (both spaces side by side)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Test 3 — Max Curvature Distribution by Category", fontsize=12, fontweight="bold")

    for ax, mode in zip(axes, ["euclidean", "hyperbolic"]):
        for label in [0, 1, 2]:
            vals = [r[mode]["max_curvature"] for r in records
                    if r.get("label") == label and mode in r and r[mode]]
            if vals:
                ax.hist(vals, bins=15, alpha=0.55,
                        color=LABEL_COLORS.get(label, "gray"),
                        label=LABEL_NAMES.get(label, str(label)),
                        edgecolor="white")
        ax.set_xlabel("Max Curvature")
        ax.set_ylabel("Count")
        ax.set_title(f"{mode.capitalize()} Space")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[test3] Distribution plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Quick AUROC using single threshold on max curvature
# ─────────────────────────────────────────────────────────────────────────────

def quick_auroc(records: list[dict], mode: str) -> float:
    """Binary AUROC: benign (0) vs adversarial (1), using max_curvature as score."""
    binary = [(r[mode]["max_curvature"], r["label"])
              for r in records
              if r.get("label") in (0, 1) and mode in r and r[mode]]
    if len(binary) < 4:
        return float("nan")
    scores = [b[0] for b in binary]
    labels = [b[1] for b in binary]
    return float(roc_auc_score(labels, scores))


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 60)
    print("  TEST 3 — TRACED Space & Trajectory Kinematics")
    print("═" * 60)

    model, tokenizer = load_model(config.MODEL_NAME, config.DEVICE, config.DTYPE)

    prompt_data = (
        [(p, 0) for p in BENIGN[:20]]
        + [(p, 1) for p in ADVERSARIAL[:20]]
        + [(p, 2) for p in DUAL_USE[:15]]
    )
    prompts = [pd[0] for pd in prompt_data]
    labels  = [pd[1] for pd in prompt_data]

    print(f"\n[test3] Extracting activations for {len(prompts)} prompts…")
    act_list = extract_activations_batch(
        model, tokenizer, prompts, config.TARGET_LAYERS, config.DEVICE
    )

    records = []
    print("\n[test3] Computing trajectory features…")
    for i, (act_dict, label, prompt) in enumerate(zip(act_list, labels, prompts)):
        eu_feat  = trajectory_features_euclidean(act_dict, config.TARGET_LAYERS)
        hyp_feat = trajectory_features_hyperbolic(act_dict, config.TARGET_LAYERS, config.HYPERBOLIC_K)
        records.append({
            "prompt":     prompt,
            "label":      label,
            "euclidean":  eu_feat,
            "hyperbolic": hyp_feat,
        })
        if eu_feat:
            print(f"  [{LABEL_NAMES.get(label,'?'):12s}] "
                  f"EU  curv={eu_feat['max_curvature']:.4f}  disp={eu_feat['displacement']:.4f} | "
                  f"HYP curv={hyp_feat.get('max_curvature', float('nan')):.4f}  "
                  f"r={hyp_feat.get('mean_radius', float('nan')):.4f}")

    # ── AUROC ──
    auroc_eu  = quick_auroc(records, "euclidean")
    auroc_hyp = quick_auroc(records, "hyperbolic")
    print(f"\n[test3] Quick AUROC (max_curvature as score):")
    print(f"  Euclidean  AUROC = {auroc_eu:.3f}")
    print(f"  Hyperbolic AUROC = {auroc_hyp:.3f}")
    if not np.isnan(auroc_hyp) and not np.isnan(auroc_eu):
        if auroc_hyp > auroc_eu:
            print("  → Hyperbolic space adds discriminative power ✓")
        else:
            print("  → Euclidean ≥ Hyperbolic — geometry may not be adding signal !")

    # ── Save ──
    save_records = [{k: v for k, v in r.items() if k != "prompt"} | {"prompt": r["prompt"][:80]}
                    for r in records]
    save_json({
        "config":  {"model": config.MODEL_NAME, "layers": config.TARGET_LAYERS},
        "auroc":   {"euclidean": auroc_eu, "hyperbolic": auroc_hyp},
        "records": save_records,
    }, "test3_traced_space.json", config.RESULTS_DIR)

    # ── Plots ──
    plot_traced_space(records, "euclidean",
                      os.path.join(config.PLOTS_DIR, "test3_traced_euclidean.png"))
    plot_traced_space(records, "hyperbolic",
                      os.path.join(config.PLOTS_DIR, "test3_traced_hyperbolic.png"))

    # Example trajectories: pick 2 benign, 2 adversarial, 2 dual-use
    examples = (
        [r for r in records if r["label"] == 0][:2]
        + [r for r in records if r["label"] == 1][:2]
        + [r for r in records if r["label"] == 2][:2]
    )
    plot_curvature_profiles(examples,
                            os.path.join(config.PLOTS_DIR, "test3_example_trajectories.png"))
    plot_curvature_distribution(records,
                                os.path.join(config.PLOTS_DIR, "test3_curvature_distribution.png"))

    print("\n[test3] Done. ✓")
    return records


if __name__ == "__main__":
    main()
