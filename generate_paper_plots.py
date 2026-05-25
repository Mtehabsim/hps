"""
Paper plot generator — produces all publication figures from saved results.

Inputs (read from results/ directory):
  - verify_new_config.json       (cold-start regime data)
  - hps_vs_rtv_llama3.json       (Llama-3 same-dist + cross-attack)
  - paper_supplementary.json     (multi-seed stability)
  - attack_ensemble_results.json (adaptive PGD)
  - hps_llama3_projection.pt     (for feature importance)
  - llama3_activations_cache.npz (for feature importance)

Outputs (results/figs/):
  - fig_cold_start_curve.{png,pdf}      headline: TPR vs N per method
  - fig_heatmap_diversity.{png,pdf}     2D: Δ vs (#methods, N)
  - fig_per_attack_bars.{png,pdf}       same-dist + cross-attack per method
  - fig_seed_stability.{png,pdf}        HPS vs Euc across seeds
  - fig_adaptive_pgd.{png,pdf}          evasion vs ε
  - fig_feature_importance.{png,pdf}    per-feature permutation importance
  - fig_per_method_cross_attack.{png,pdf}  cross-attack TPR per method
  - fig_learning_curve.{png,pdf}        AUROC vs N attacks (paper supplementary)

Usage:
  python generate_paper_plots.py                    # all plots
  python generate_paper_plots.py --plots cold_start,heatmap,features
  python generate_paper_plots.py --skip features    # skip the slow one
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Publication-quality matplotlib settings ──
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 13,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": ":",
})

# Color scheme
C_HPS = "#9b59b6"       # purple
C_EUC = "#3498db"       # blue
C_RTV = "#e67e22"       # orange
C_ENS = "#27ae60"       # green
C_BENIGN = "#2ecc71"
C_ATTACK = "#e74c3c"

RESULTS = "results"
FIG_DIR = os.path.join(RESULTS, "figs")


def save(fig, name):
    """Save as both PNG (preview) and PDF (paper-ready)."""
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ["png", "pdf"]:
        path = os.path.join(FIG_DIR, f"{name}.{ext}")
        fig.savefig(path)
        print(f"  → {path}")
    plt.close(fig)


def load_json(path):
    if not os.path.exists(path):
        print(f"  ⚠ Missing: {path}")
        return None
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════
#  PLOT 1: Cold-start curve (TPR vs N per method)
# ═══════════════════════════════════════════════════════════════════
def plot_cold_start():
    print("\n[fig 1] Cold-start curve (TPR vs N per method)...")
    data = load_json(os.path.join(RESULTS, "verify_new_config.json"))
    if data is None:
        print("  Skip: verify_new_config.json not found")
        return

    # Combine Part B (cold-start) and Part C (extreme low) to get full curve
    cold = data.get("cold_start", [])
    extreme = data.get("extreme_low_n", [])
    all_pts = sorted(extreme + cold, key=lambda r: r["n"])

    if not all_pts:
        print("  Skip: no cold-start data")
        return

    N = [r["n"] for r in all_pts]
    hps = [r["hps"] for r in all_pts]
    euc = [r["euc"] for r in all_pts]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(N, hps, "o-", color=C_HPS, linewidth=2.5, markersize=8,
            label="HPS (hyperbolic)", zorder=3)
    ax.plot(N, euc, "s-", color=C_EUC, linewidth=2.5, markersize=8,
            label="Euclidean (matched params)", zorder=2)
    ax.fill_between(N, hps, euc, where=[h > e for h, e in zip(hps, euc)],
                    alpha=0.15, color=C_HPS, zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("N attacks per method (training)")
    ax.set_ylabel("Cross-attack TPR @ 5% FPR")
    ax.set_title("Cold-Start Regime: HPS vs Euclidean")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(N)
    ax.set_xticklabels([str(n) for n in N], rotation=0)

    # Annotate max gap
    max_gap_idx = max(range(len(N)), key=lambda i: hps[i] - euc[i])
    max_n = N[max_gap_idx]
    max_gap = hps[max_gap_idx] - euc[max_gap_idx]
    ax.annotate(f"Δ = +{max_gap:.2f}",
                xy=(max_n, (hps[max_gap_idx] + euc[max_gap_idx]) / 2),
                xytext=(max_n * 1.5, 0.5),
                fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black", alpha=0.6))

    ax.legend(loc="lower right", framealpha=0.95)
    save(fig, "fig_cold_start_curve")


# ═══════════════════════════════════════════════════════════════════
#  PLOT 2: Heatmap of Δ vs (#methods, N per method)
# ═══════════════════════════════════════════════════════════════════
def plot_heatmap_diversity():
    print("\n[fig 2] Heatmap: Δ vs (#methods, N per method)...")
    data = load_json(os.path.join(RESULTS, "verify_new_config.json"))
    if data is None:
        return

    # Collect all (k, n, delta) data points from all parts
    points = []
    for r in data.get("cold_start", []):
        points.append((9, r["n"], r["delta"]))
    for r in data.get("extreme_low_n", []):
        points.append((9, r["n"], r["delta"]))
    for r in data.get("method_diversity", []):
        points.append((r["k"], 25, r["delta"]))
    for r in data.get("extreme_combined", []):
        points.append((r["k"], r["n"], r["delta"]))

    if not points:
        print("  Skip: no data")
        return

    # Build sparse matrix
    ks = sorted(set(p[0] for p in points))
    ns = sorted(set(p[1] for p in points))
    grid = np.full((len(ks), len(ns)), np.nan)
    for k, n, d in points:
        i = ks.index(k); j = ns.index(n)
        # If multiple values for same cell (across different parts), keep the latest
        grid[i, j] = d

    fig, ax = plt.subplots(figsize=(8, 4.5))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(grid, cmap=cmap, vmin=-0.1, vmax=0.8, aspect="auto", origin="lower")

    # Annotate cells
    for i in range(len(ks)):
        for j in range(len(ns)):
            v = grid[i, j]
            if not np.isnan(v):
                color = "white" if v < 0.4 else "black"
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([str(k) for k in ks])
    ax.set_xlabel("N attacks per method")
    ax.set_ylabel("Number of attack methods")
    ax.set_title("HPS – Euclidean Δ (cross-attack TPR)")
    ax.grid(False)

    cbar = plt.colorbar(im, ax=ax, fraction=0.04)
    cbar.set_label("Δ TPR (HPS − Euclidean)")
    save(fig, "fig_heatmap_diversity")


# ═══════════════════════════════════════════════════════════════════
#  PLOT 3: Per-attack-type bar chart (same-dist + cross-attack)
# ═══════════════════════════════════════════════════════════════════
def plot_per_attack_bars():
    print("\n[fig 3] Per-attack bar chart...")
    data = load_json(os.path.join(RESULTS, "hps_vs_rtv_llama3.json"))
    if data is None:
        return

    cross = data.get("cross_attack", {})
    if not cross:
        print("  Skip: no cross-attack data")
        return

    methods = sorted(cross.keys())
    hps_vals = [cross[m]["hps"] for m in methods]
    rtv_vals = [cross[m]["rtv"] for m in methods]
    ens_vals = [cross[m]["ens"] for m in methods]

    x = np.arange(len(methods))
    width = 0.27

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - width, rtv_vals, width, label="RTV", color=C_RTV, alpha=0.9)
    ax.bar(x, hps_vals, width, label="HPS", color=C_HPS, alpha=0.9)
    ax.bar(x + width, ens_vals, width, label="HPS+RTV Ensemble", color=C_ENS, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Cross-attack TPR @ 5% FPR")
    ax.set_title("Cross-Attack Detection by Attack Family (Llama-3-8B)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.axhline(0.95, color="black", linestyle=":", linewidth=1, alpha=0.4)

    save(fig, "fig_per_attack_bars")


# ═══════════════════════════════════════════════════════════════════
#  PLOT 4: Multi-seed stability
# ═══════════════════════════════════════════════════════════════════
def plot_seed_stability():
    print("\n[fig 4] Multi-seed stability...")
    data = load_json(os.path.join(RESULTS, "paper_supplementary.json"))
    if data is None:
        return

    ms = data.get("multi_seed", {})
    if not ms or "aurocs" not in ms:
        print("  Skip: no multi-seed data")
        return

    hps_aurocs = ms["aurocs"]
    hps_tprs = ms["tprs"]
    n_seeds = len(hps_aurocs)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    seeds = list(range(n_seeds))

    axes[0].bar(seeds, hps_aurocs, color=C_HPS, alpha=0.8)
    axes[0].axhline(np.mean(hps_aurocs), color="black", linestyle="--", linewidth=1,
                    label=f"Mean: {np.mean(hps_aurocs):.4f} ± {np.std(hps_aurocs):.4f}")
    axes[0].set_ylim(min(hps_aurocs) - 0.005, 1.001)
    axes[0].set_xticks(seeds)
    axes[0].set_xticklabels([f"S{s}" for s in seeds])
    axes[0].set_ylabel("AUROC")
    axes[0].set_title("HPS — Same-Distribution Stability")
    axes[0].legend()

    axes[1].bar(seeds, hps_tprs, color=C_HPS, alpha=0.8)
    axes[1].axhline(np.mean(hps_tprs), color="black", linestyle="--", linewidth=1,
                    label=f"Mean: {np.mean(hps_tprs):.4f} ± {np.std(hps_tprs):.4f}")
    axes[1].set_ylim(min(hps_tprs) - 0.01, 1.001)
    axes[1].set_xticks(seeds)
    axes[1].set_xticklabels([f"S{s}" for s in seeds])
    axes[1].set_ylabel("TPR @ 5% FPR")
    axes[1].set_title("HPS — TPR Stability")
    axes[1].legend()

    save(fig, "fig_seed_stability")


# ═══════════════════════════════════════════════════════════════════
#  PLOT 5: Adaptive PGD evasion vs ε
# ═══════════════════════════════════════════════════════════════════
def plot_adaptive_pgd():
    print("\n[fig 5] Adaptive PGD curve...")
    data = load_json(os.path.join(RESULTS, "attack_ensemble_results.json"))
    if data is None:
        return

    res = data.get("results", {})
    if not res:
        return

    eps = []
    evasion = []
    for k, v in res.items():
        try:
            e = float(k.replace("eps_", ""))
            eps.append(e)
            evasion.append(v["evasion"])
        except (ValueError, KeyError):
            continue

    if not eps:
        return

    sorted_idx = np.argsort(eps)
    eps = [eps[i] for i in sorted_idx]
    evasion = [evasion[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(eps, evasion, "o-", color=C_ENS, linewidth=2.5, markersize=8)
    ax.fill_between(eps, evasion, alpha=0.2, color=C_ENS)
    ax.set_xscale("log")
    ax.set_xlabel("Perturbation budget ε (L∞ on activations)")
    ax.set_ylabel("Evasion rate")
    ax.set_title("Adaptive PGD Attack on HPS+RTV Ensemble")
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1, alpha=0.4)
    ax.axvline(0.05, color="red", linestyle="--", linewidth=1, alpha=0.5,
               label="Defense breaks (ε=0.05)")
    ax.legend()
    save(fig, "fig_adaptive_pgd")


# ═══════════════════════════════════════════════════════════════════
#  PLOT 6: Feature importance (compute on the fly)
# ═══════════════════════════════════════════════════════════════════
def plot_feature_importance():
    print("\n[fig 6] Feature importance (computing permutation importance)...")
    proj_path = os.path.join(RESULTS, "hps_llama3_projection.pt")
    cache_path = os.path.join(RESULTS, "llama3_activations_cache.npz")
    if not (os.path.exists(proj_path) and os.path.exists(cache_path)):
        print("  Skip: missing projection or cache")
        return

    import torch
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.inspection import permutation_importance
    from experiment7 import LorentzProjection, extract_trajectory_features

    HPS_LAYERS = [0, 2, 17, 24, 28, 31]
    ckpt = torch.load(proj_path, map_location="cpu", weights_only=False)
    proj = LorentzProjection(ckpt["d_in"], ckpt["d_proj"], 1.0, n_layers=ckpt["n_layers"])
    proj.load_state_dict(ckpt["state_dict"])
    proj.eval()

    cache = np.load(cache_path, allow_pickle=True)
    hs_train_ben = cache["hs_train_ben"].tolist()
    hs_train_atk = cache["hs_train_atk"].tolist()
    hs_test_ben = cache["hs_test_ben"].tolist()
    hs_test_atk = cache["hs_test_atk"].tolist()

    def to_arr(hs_list):
        return np.array([[hs[l][-1] for l in HPS_LAYERS] for hs in hs_list])

    X_tr = np.concatenate([to_arr(hs_train_ben), to_arr(hs_train_atk)])
    y_tr = np.array([0]*len(hs_train_ben) + [1]*len(hs_train_atk))
    X_te = np.concatenate([to_arr(hs_test_ben), to_arr(hs_test_atk)])
    y_te = np.array([0]*len(hs_test_ben) + [1]*len(hs_test_atk))

    print("  Extracting features...")
    f_tr = extract_trajectory_features(proj, X_tr)
    f_te = extract_trajectory_features(proj, X_te)
    sc = StandardScaler()
    f_tr_s = sc.fit_transform(f_tr)
    f_te_s = sc.transform(f_te)
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(f_tr_s, y_tr)

    print("  Running permutation importance (n_repeats=10, scoring=roc_auc)...")
    perm = permutation_importance(clf, f_te_s, y_te, n_repeats=10,
                                   random_state=42, scoring="roc_auc")
    imp = perm.importances_mean
    std = perm.importances_std

    feat_names = ["mean_r", "max_r", "min_r", "std_r", "range_r",
                  "max_κ", "mean_κ", "std_κ", "spike_loc",
                  "displacement", "path_len", "progress"]
    categories = ["radial"]*5 + ["curvature"]*4 + ["displacement"]*3
    cat_colors = {"radial": "#9b59b6", "curvature": "#e67e22", "displacement": "#27ae60"}

    # Two-panel plot: per-feature + category aggregated
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Panel 1: per-feature
    order = np.argsort(imp)[::-1]
    sorted_names = [feat_names[i] for i in order]
    sorted_imp = imp[order]
    sorted_std = std[order]
    sorted_cats = [categories[i] for i in order]
    sorted_colors = [cat_colors[c] for c in sorted_cats]

    axes[0].barh(range(len(feat_names)), sorted_imp, xerr=sorted_std,
                  color=sorted_colors, alpha=0.85,
                  error_kw=dict(ecolor="black", lw=1, capsize=3))
    axes[0].set_yticks(range(len(feat_names)))
    axes[0].set_yticklabels(sorted_names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Permutation importance (Δ AUROC)")
    axes[0].set_title("Per-Feature Importance")
    axes[0].axvline(0, color="black", linewidth=0.5)

    # Panel 2: category aggregated (sum)
    cat_imp = {"radial": 0, "curvature": 0, "displacement": 0}
    cat_std = {"radial": 0, "curvature": 0, "displacement": 0}
    for i, c in enumerate(categories):
        cat_imp[c] += imp[i]
        cat_std[c] += std[i]**2  # sum of variances
    cat_std = {k: np.sqrt(v) for k, v in cat_std.items()}

    cat_order = sorted(cat_imp.keys(), key=lambda c: -cat_imp[c])
    bars2 = axes[1].bar([f"{c}\n(n={categories.count(c)})" for c in cat_order],
                         [cat_imp[c] for c in cat_order],
                         yerr=[cat_std[c] for c in cat_order],
                         color=[cat_colors[c] for c in cat_order], alpha=0.85,
                         error_kw=dict(ecolor="black", lw=1.5, capsize=5))
    axes[1].set_ylabel("Summed permutation importance (Δ AUROC)")
    axes[1].set_title("Category-Aggregated Importance")
    axes[1].axhline(0, color="black", linewidth=0.5)
    for bar, c in zip(bars2, cat_order):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                      f"{cat_imp[c]:.4f}", ha="center", va="bottom", fontsize=10,
                      fontweight="bold")

    # Shared legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=cat_colors[c], alpha=0.85)
               for c in ["radial", "curvature", "displacement"]]
    fig.legend(handles, ["Radial", "Curvature", "Displacement"],
                loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    save(fig, "fig_feature_importance")

    # Print summary
    print(f"\n  Per-feature ranking:")
    print(f"  {'Rank':<5} {'Feature':<12} {'Importance':>12} {'Category'}")
    for rank, (name, val, cat) in enumerate(zip(sorted_names, sorted_imp, sorted_cats), 1):
        flag = " ← top6" if rank <= 6 else ""
        print(f"  {rank:<5} {name:<12} {val:>12.5f}  {cat:<13}{flag}")
    print(f"\n  Category aggregated:")
    for c in cat_order:
        print(f"  {c:<13} (n={categories.count(c)})  importance={cat_imp[c]:.5f}")


# ═══════════════════════════════════════════════════════════════════
#  PLOT 6b: Feature ablation comparison (TPR + FPR per subset, per regime)
# ═══════════════════════════════════════════════════════════════════
def plot_feature_ablation():
    print("\n[fig 6b] Feature ablation comparison...")
    data = load_json(os.path.join(RESULTS, "feature_ablation.json"))
    if data is None:
        print("  Skip: feature_ablation.json not found (run feature_ablation.py first)")
        return

    same_dist = data.get("same_dist", {})
    cold_start = data.get("cold_start", {})
    vicuna = data.get("vicuna_like", {})

    # Subsets to display (in display order)
    subsets = ["all_12", "radial_disp_8", "top6_byimp", "top4_byimp",
               "radial_5", "displacement_3", "curvature_4", "top1_byimp"]
    # Filter to subsets actually present
    subsets = [s for s in subsets if s in same_dist]

    # TPR data
    regimes = ["Same-dist", "Cold-start\nN=5", "Cold-start\nN=25",
               "Cold-start\nN=100", "Vicuna-like"]
    tpr_data = []
    for s in subsets:
        row = [
            same_dist[s]["tpr"],
            cold_start.get("5", {}).get(s, {}).get("tpr", np.nan),
            cold_start.get("25", {}).get(s, {}).get("tpr", np.nan),
            cold_start.get("100", {}).get(s, {}).get("tpr", np.nan),
            vicuna.get(s, {}).get("tpr", np.nan),
        ]
        tpr_data.append(row)
    tpr_arr = np.array(tpr_data)

    fig, ax = plt.subplots(figsize=(11, 5))
    n_subsets = len(subsets)
    n_regimes = len(regimes)
    x = np.arange(n_regimes)
    width = 0.85 / n_subsets

    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, n_subsets))
    for i, s in enumerate(subsets):
        offset = (i - n_subsets / 2 + 0.5) * width
        n_feat = same_dist[s].get("n_features", "?")
        bars = ax.bar(x + offset, tpr_arr[i], width,
                       label=f"{s} (n={n_feat})", color=cmap[i], alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_ylabel("TPR @ 5% FPR")
    ax.set_title("Feature Ablation: TPR Across Regimes")
    ax.set_ylim(0.7, 1.02)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.5)
    ax.legend(ncol=2, fontsize=9, loc="lower right")
    save(fig, "fig_feature_ablation")


# ═══════════════════════════════════════════════════════════════════
#  PLOT 7: Per-method cross-attack (HPS vs RTV vs Ensemble)
# ═══════════════════════════════════════════════════════════════════
def plot_per_method_cross_attack():
    print("\n[fig 7] Per-method cross-attack comparison...")
    data = load_json(os.path.join(RESULTS, "hps_vs_rtv_llama3.json"))
    if data is None:
        return

    # This is essentially same as fig 3 — kept for backward compat
    cross = data.get("cross_attack", {})
    if not cross:
        return

    methods = sorted(cross.keys())
    n_per = [cross[m]["n"] for m in methods]
    hps_vals = [cross[m]["hps"] for m in methods]
    rtv_vals = [cross[m]["rtv"] for m in methods]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(methods))
    ax.scatter(x, hps_vals, color=C_HPS, s=120, marker="o", label="HPS", zorder=3, edgecolors="black")
    ax.scatter(x, rtv_vals, color=C_RTV, s=120, marker="s", label="RTV", zorder=3, edgecolors="black")

    for i, (h, r) in enumerate(zip(hps_vals, rtv_vals)):
        ax.plot([i, i], [r, h], "k-", linewidth=1, alpha=0.4, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Cross-attack TPR @ 5% FPR")
    ax.set_title("HPS vs RTV — Per-Attack Cross-Attack Performance")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    save(fig, "fig_per_method_cross_attack")


# ═══════════════════════════════════════════════════════════════════
#  PLOT 8: Learning curve (AUROC vs N attacks)
# ═══════════════════════════════════════════════════════════════════
def plot_learning_curve():
    print("\n[fig 8] Learning curve (from supplementary, same-dist)...")
    # paper_supplementary.json doesn't currently save the learning curve directly
    # but we can read it from the log if available, or skip.
    # For now, parse from EXP 2 of supplementary if data is saved.
    # If not available, this is a no-op.
    data = load_json(os.path.join(RESULTS, "paper_supplementary.json"))
    if data is None or "learning_curve" not in data:
        print("  Skip: learning curve data not in JSON (only in log)")
        return

    lc = data["learning_curve"]
    n = [r["n"] for r in lc]
    auroc = [r["auroc"] for r in lc]
    tpr = [r["tpr"] for r in lc]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(n, auroc, "o-", color=C_HPS, linewidth=2, markersize=8, label="AUROC")
    ax.plot(n, tpr, "s-", color=C_EUC, linewidth=2, markersize=8, label="TPR @ 5% FPR")
    ax.set_xscale("log")
    ax.set_xlabel("N training attacks")
    ax.set_ylabel("Performance")
    ax.set_title("HPS — Learning Curve (same-distribution)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    save(fig, "fig_learning_curve")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots", default="all",
                        help="Comma-separated plot names: cold_start, heatmap, "
                             "per_attack, seeds, adaptive, features, cross_attack, learning")
    parser.add_argument("--skip", default="",
                        help="Comma-separated plot names to skip")
    args = parser.parse_args()

    plot_funcs = {
        "cold_start": plot_cold_start,
        "heatmap": plot_heatmap_diversity,
        "per_attack": plot_per_attack_bars,
        "seeds": plot_seed_stability,
        "adaptive": plot_adaptive_pgd,
        "features": plot_feature_importance,
        "ablation": plot_feature_ablation,
        "cross_attack": plot_per_method_cross_attack,
        "learning": plot_learning_curve,
    }

    if args.plots == "all":
        to_run = list(plot_funcs.keys())
    else:
        to_run = [p.strip() for p in args.plots.split(",") if p.strip()]
    skip = set(p.strip() for p in args.skip.split(",") if p.strip())
    to_run = [p for p in to_run if p not in skip]

    print(f"Generating plots: {to_run}")
    print(f"Output dir: {FIG_DIR}")

    for name in to_run:
        if name not in plot_funcs:
            print(f"\n  ⚠ Unknown plot: {name}")
            continue
        try:
            plot_funcs[name]()
        except Exception as e:
            print(f"\n  ✗ {name} failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'═'*60}")
    print(f"  Done. Figures saved to {FIG_DIR}/")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
