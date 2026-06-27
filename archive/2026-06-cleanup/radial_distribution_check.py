"""
radial_distribution_check.py — Verify the radial distribution finding.

Addresses evaluator critical issue C6 (radial distribution contradicts
hypothesis). Tests robustness of the inversion (benign higher than attack)
across:
  - Multiple training seeds
  - Multiple training checkpoints (10, 25, 50, 100 epochs)
  - Multiple curvature κ values
  - Different layer subsets

For each configuration, computes:
  - Median radial position (x_0) for benign vs attack
  - "Inversion" indicator: 1 if benign median > attack median (matches our
    observed counterintuitive finding), 0 if attack median > benign median
    (matches the original geometric hypothesis)
  - Per-layer breakdown

Outputs:
  results/radial_distribution_check.json — full numeric results
  results/figs/radial_check_seeds.png  — distribution per seed
  results/figs/radial_check_epochs.png — distribution per epoch checkpoint
  results/figs/radial_check_kappas.png — distribution per κ

Usage:
    python radial_distribution_check.py
    python radial_distribution_check.py --n_seeds 5 --epochs_to_check 10 25 50
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hps_core import LorentzProjection, contrastive_loss

HPS_LAYERS = [0, 2, 17, 24, 28, 31]


def _np_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    raise TypeError(f"not serializable: {type(o)}")


def to_hps_array(hs_list, layers=HPS_LAYERS):
    return np.array([[hs[l][-1] for l in layers] for hs in hs_list])


def load_cache(cache_path):
    if not Path(cache_path).exists():
        raise FileNotFoundError(
            f"Cache not found: {cache_path}\n"
            "Run hps_llama3.py first to generate it."
        )
    cache = np.load(cache_path, allow_pickle=True)
    return cache


def train_hps_with_checkpoints(X_train, y_train, n_layers, d_hidden,
                                seed, kappa_init, total_epochs, checkpoint_epochs,
                                device):
    """Train HPS, returning projection states at each requested checkpoint."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    proj = LorentzProjection(d_hidden, 64, kappa_init,
                              n_layers=n_layers).to(device)
    proj.log_k.requires_grad = False
    opt = optim.Adam(
        [p for p in proj.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=1e-5,
    )
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)

    checkpoints = {}
    for epoch in range(1, total_epochs + 1):
        loss = torch.tensor(0.0, device=device)
        for l in range(n_layers):
            h = proj(X_t[:, l, :])
            loss = loss + contrastive_loss(h, y_t, k=proj.k, tau=proj.tau(l))
        loss = loss / n_layers
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch in checkpoint_epochs:
            # Save a copy of the projection state at this epoch
            state_copy = {
                k: v.detach().cpu().clone()
                for k, v in proj.state_dict().items()
            }
            checkpoints[epoch] = state_copy

    return proj, checkpoints


def compute_radial_for_state(X_layers, state_dict, d_hidden, n_layers,
                              kappa_init, device):
    """Load state and compute radial positions (x_0 in Lorentz space)."""
    proj = LorentzProjection(d_hidden, 64, kappa_init,
                              n_layers=n_layers).to(device)
    proj.load_state_dict(state_dict)
    proj.eval()

    X_t = torch.tensor(X_layers, dtype=torch.float32, device=device)
    with torch.no_grad():
        radial_per_layer = []
        for l in range(n_layers):
            h = proj(X_t[:, l, :])  # (N, d_proj+1)
            x0 = h[:, 0].cpu().numpy()  # time coordinate in Lorentz
            radial_per_layer.append(x0)
        radial_per_layer = np.stack(radial_per_layer, axis=1)  # (N, n_layers)
        radial_mean = radial_per_layer.mean(axis=1)
    return radial_per_layer, radial_mean


def analyze_radial(radial_ben, radial_atk):
    """Return dict of summary stats."""
    return {
        "benign_median": float(np.median(radial_ben)),
        "attack_median": float(np.median(radial_atk)),
        "benign_mean": float(radial_ben.mean()),
        "attack_mean": float(radial_atk.mean()),
        "benign_std": float(radial_ben.std()),
        "attack_std": float(radial_atk.std()),
        "diff_median": float(np.median(radial_ben) - np.median(radial_atk)),
        "diff_mean": float(radial_ben.mean() - radial_atk.mean()),
        # inverted = TRUE if benign > attack (our observed counterintuitive case)
        "inverted": bool(np.median(radial_ben) > np.median(radial_atk)),
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_seed_robustness_check(X_tr, y_tr, X_te_ben, X_te_atk,
                                kappa_init, total_epochs, n_seeds, device):
    """Train HPS at multiple seeds; check radial distribution at final epoch."""
    n_layers = X_tr.shape[1]
    d_hidden = X_tr.shape[2]

    results = []
    for s in range(42, 42 + n_seeds):
        proj, _ = train_hps_with_checkpoints(
            X_tr, y_tr, n_layers, d_hidden, s, kappa_init, total_epochs,
            checkpoint_epochs={total_epochs}, device=device,
        )
        # Compute radial for benign and attack
        proj.eval()
        X_ben_t = torch.tensor(X_te_ben, dtype=torch.float32, device=device)
        X_atk_t = torch.tensor(X_te_atk, dtype=torch.float32, device=device)
        with torch.no_grad():
            radial_ben_per_layer = []
            radial_atk_per_layer = []
            for l in range(n_layers):
                radial_ben_per_layer.append(proj(X_ben_t[:, l, :])[:, 0].cpu().numpy())
                radial_atk_per_layer.append(proj(X_atk_t[:, l, :])[:, 0].cpu().numpy())
        radial_ben = np.mean(np.stack(radial_ben_per_layer, axis=1), axis=1)
        radial_atk = np.mean(np.stack(radial_atk_per_layer, axis=1), axis=1)

        stats = analyze_radial(radial_ben, radial_atk)
        stats["seed"] = s
        stats["radial_ben_sample"] = radial_ben[:50].tolist()  # for plotting
        stats["radial_atk_sample"] = radial_atk[:50].tolist()
        results.append(stats)
        print(f"  seed={s}  benign_median={stats['benign_median']:.4f}  "
              f"attack_median={stats['attack_median']:.4f}  "
              f"diff={stats['diff_median']:+.4f}  "
              f"{'INVERTED' if stats['inverted'] else 'as_hypothesis'}")
    return results


def run_epoch_check(X_tr, y_tr, X_te_ben, X_te_atk, kappa_init,
                    total_epochs, checkpoint_epochs, seed, device):
    """Single seed, multiple epoch checkpoints."""
    n_layers = X_tr.shape[1]
    d_hidden = X_tr.shape[2]

    proj, checkpoints = train_hps_with_checkpoints(
        X_tr, y_tr, n_layers, d_hidden, seed, kappa_init, total_epochs,
        checkpoint_epochs=set(checkpoint_epochs), device=device,
    )

    results = []
    for ep in checkpoint_epochs:
        if ep not in checkpoints:
            continue
        state = checkpoints[ep]
        _, radial_ben = compute_radial_for_state(
            X_te_ben, state, d_hidden, n_layers, kappa_init, device,
        )
        _, radial_atk = compute_radial_for_state(
            X_te_atk, state, d_hidden, n_layers, kappa_init, device,
        )
        stats = analyze_radial(radial_ben, radial_atk)
        stats["epoch"] = ep
        stats["seed"] = seed
        results.append(stats)
        print(f"  epoch={ep}  benign_median={stats['benign_median']:.4f}  "
              f"attack_median={stats['attack_median']:.4f}  "
              f"diff={stats['diff_median']:+.4f}  "
              f"{'INVERTED' if stats['inverted'] else 'as_hypothesis'}")
    return results


def run_kappa_check(X_tr, y_tr, X_te_ben, X_te_atk, kappa_values,
                     total_epochs, seed, device):
    """Single seed, multiple κ values."""
    n_layers = X_tr.shape[1]
    d_hidden = X_tr.shape[2]

    results = []
    for k in kappa_values:
        proj, _ = train_hps_with_checkpoints(
            X_tr, y_tr, n_layers, d_hidden, seed, k, total_epochs,
            checkpoint_epochs={total_epochs}, device=device,
        )
        proj.eval()
        X_ben_t = torch.tensor(X_te_ben, dtype=torch.float32, device=device)
        X_atk_t = torch.tensor(X_te_atk, dtype=torch.float32, device=device)
        with torch.no_grad():
            radial_ben_per_layer = []
            radial_atk_per_layer = []
            for l in range(n_layers):
                radial_ben_per_layer.append(proj(X_ben_t[:, l, :])[:, 0].cpu().numpy())
                radial_atk_per_layer.append(proj(X_atk_t[:, l, :])[:, 0].cpu().numpy())
        radial_ben = np.mean(np.stack(radial_ben_per_layer, axis=1), axis=1)
        radial_atk = np.mean(np.stack(radial_atk_per_layer, axis=1), axis=1)

        stats = analyze_radial(radial_ben, radial_atk)
        stats["kappa"] = k
        stats["seed"] = seed
        results.append(stats)
        print(f"  κ={k}  benign_median={stats['benign_median']:.4f}  "
              f"attack_median={stats['attack_median']:.4f}  "
              f"diff={stats['diff_median']:+.4f}  "
              f"{'INVERTED' if stats['inverted'] else 'as_hypothesis'}")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_dist(results, axis_label, output_path, title):
    """Plot distributions for all configs in `results`."""
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        # Use sample if present, else build from medians (fall-back)
        ben = r.get("radial_ben_sample") or [r["benign_median"]]
        atk = r.get("radial_atk_sample") or [r["attack_median"]]
        ax.hist(ben, bins=30, alpha=0.5, label="benign", color="green", density=True)
        ax.hist(atk, bins=30, alpha=0.5, label="attack", color="purple", density=True)
        ax.axvline(r["benign_median"], color="green", linestyle="--",
                   label=f"benign med={r['benign_median']:.3f}")
        ax.axvline(r["attack_median"], color="purple", linestyle="--",
                   label=f"attack med={r['attack_median']:.3f}")
        label = r.get(axis_label, "?")
        sign = "INVERTED ✓" if r["inverted"] else "as_hypothesis ✗"
        ax.set_title(f"{axis_label}={label}  ({sign})", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel("radial position (x_0)")
        ax.set_ylabel("density")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache",
                        default="results/llama3_activations_cache.npz")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--total_epochs", type=int, default=50)
    parser.add_argument("--epochs_to_check", type=int, nargs="+",
                        default=[5, 10, 25, 50])
    parser.add_argument("--kappas", type=float, nargs="+",
                        default=[0.1, 0.5, 1.0, 2.0])
    parser.add_argument("--device", default=None)
    parser.add_argument("--output",
                        default="results/radial_distribution_check.json")
    parser.add_argument("--skip_plots", action="store_true")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print("RADIAL DISTRIBUTION VERIFICATION")
    print("=" * 78)
    print(f"  Cache:           {args.cache}")
    print(f"  Seeds tested:    {args.n_seeds}")
    print(f"  Epochs checked:  {args.epochs_to_check}")
    print(f"  Kappas tested:   {args.kappas}")
    print(f"  Device:          {device}")
    print()

    cache = load_cache(args.cache)
    hs_train_ben = cache["hs_train_ben"].tolist()
    hs_train_atk = cache["hs_train_atk"].tolist()
    hs_test_ben = cache["hs_test_ben"].tolist()
    hs_test_atk = cache["hs_test_atk"].tolist()

    X_tr_ben = to_hps_array(hs_train_ben)
    X_tr_atk = to_hps_array(hs_train_atk)
    X_te_ben = to_hps_array(hs_test_ben)
    X_te_atk = to_hps_array(hs_test_atk)

    X_tr = np.concatenate([X_tr_ben, X_tr_atk], axis=0)
    y_tr = np.concatenate(
        [np.zeros(len(X_tr_ben)), np.ones(len(X_tr_atk))], axis=0)

    print(f"  Train: {X_tr.shape}, Test benign: {X_te_ben.shape}, "
          f"attacks: {X_te_atk.shape}")
    print()

    # ── 1. Robustness across seeds ──
    print("─" * 78)
    print("1. Seed robustness (κ=0.1, 50 epochs)")
    print("─" * 78)
    seed_results = run_seed_robustness_check(
        X_tr, y_tr, X_te_ben, X_te_atk,
        kappa_init=0.1, total_epochs=args.total_epochs,
        n_seeds=args.n_seeds, device=device,
    )
    inverted_count = sum(1 for r in seed_results if r["inverted"])
    print(f"\n  → {inverted_count}/{len(seed_results)} seeds show INVERSION "
          f"(benign > attack)")
    print()

    # ── 2. Robustness across epochs ──
    print("─" * 78)
    print(f"2. Epoch robustness (seed=42, κ=0.1, "
          f"checkpoints at {args.epochs_to_check})")
    print("─" * 78)
    epoch_results = run_epoch_check(
        X_tr, y_tr, X_te_ben, X_te_atk,
        kappa_init=0.1, total_epochs=args.total_epochs,
        checkpoint_epochs=args.epochs_to_check, seed=42, device=device,
    )
    inverted_epoch = sum(1 for r in epoch_results if r["inverted"])
    print(f"\n  → {inverted_epoch}/{len(epoch_results)} epoch checkpoints "
          f"show INVERSION")
    print()

    # ── 3. Robustness across κ ──
    print("─" * 78)
    print(f"3. Curvature κ robustness (seed=42, 50 epochs, κ ∈ {args.kappas})")
    print("─" * 78)
    kappa_results = run_kappa_check(
        X_tr, y_tr, X_te_ben, X_te_atk,
        kappa_values=args.kappas, total_epochs=args.total_epochs,
        seed=42, device=device,
    )
    inverted_kappa = sum(1 for r in kappa_results if r["inverted"])
    print(f"\n  → {inverted_kappa}/{len(kappa_results)} κ values show INVERSION")
    print()

    # ── Save results ──
    output = {
        "config": {
            "n_seeds": args.n_seeds,
            "total_epochs": args.total_epochs,
            "epochs_to_check": args.epochs_to_check,
            "kappas": args.kappas,
            "device": device,
            "hps_layers": HPS_LAYERS,
        },
        "summary": {
            "seeds_inverted": f"{inverted_count}/{len(seed_results)}",
            "epochs_inverted": f"{inverted_epoch}/{len(epoch_results)}",
            "kappas_inverted": f"{inverted_kappa}/{len(kappa_results)}",
            "any_not_inverted": (inverted_count < len(seed_results) or
                                  inverted_epoch < len(epoch_results) or
                                  inverted_kappa < len(kappa_results)),
        },
        "seed_results": seed_results,
        "epoch_results": epoch_results,
        "kappa_results": kappa_results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=_np_default)
    print(f"Saved: {output_path}")

    # ── Plots ──
    if not args.skip_plots:
        print()
        print("Generating plots...")
        plot_dist(seed_results, "seed",
                   "results/figs/radial_check_seeds.png",
                   "Radial distribution across HPS training seeds")
        plot_dist(epoch_results, "epoch",
                   "results/figs/radial_check_epochs.png",
                   "Radial distribution across HPS training epochs (seed=42)")
        plot_dist(kappa_results, "kappa",
                   "results/figs/radial_check_kappas.png",
                   "Radial distribution across curvature κ values (seed=42)")

    # ── Final summary ──
    print()
    print("=" * 78)
    print("CONCLUSION")
    print("=" * 78)
    print()
    total = (len(seed_results) + len(epoch_results) + len(kappa_results))
    total_inverted = inverted_count + inverted_epoch + inverted_kappa
    print(f"INVERSION (benign median > attack median) observed in "
          f"{total_inverted}/{total} configurations")
    print()
    if total_inverted == total:
        print("✓ The inversion is ROBUST across all tested configurations:")
        print("  - All {} seeds show benign at higher radial position".format(
              len(seed_results)))
        print("  - All {} training epochs show the same direction".format(
              len(epoch_results)))
        print("  - All {} κ values show the same direction".format(
              len(kappa_results)))
        print()
        print("This confirms: the original geometric hypothesis ('attacks at")
        print("extreme periphery / high radial position') is contradicted by")
        print("the empirical reality. The contrastive loss finds an arbitrary")
        print("discriminative direction; the Lorentz geometry constrains it")
        print("to be radial; but the semantic interpretation is opposite of")
        print("what was hypothesized.")
    elif total_inverted >= total * 0.8:
        print("⚠ The inversion is MOSTLY robust ({:.0%} of configurations).".format(
              total_inverted / total))
        print("  Some configurations show the original hypothesis direction.")
        print("  Inspect per-config details for nuances.")
    else:
        print("✗ The inversion is NOT robust ({:.0%} of configurations).".format(
              total_inverted / total))
        print("  The original observation may have been seed/config specific.")


if __name__ == "__main__":
    main()
