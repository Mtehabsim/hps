"""
vicuna_diagnostic.py — Why does HPS fail on Vicuna but work on Llama-3?

This script systematically tests 6 hypotheses for the HPS-vs-C4 gap on Vicuna,
using the activation caches from both LLMs. Each test isolates one possible
mechanism so we can attribute the failure to a specific cause.

Hypotheses tested:
  H1. δ-hyperbolicity:    Vicuna activations are less tree-like than Llama-3's
  H2. Capacity bottleneck: 64-dim projection loses critical information that
                            5120-dim mean-pool retains
  H3. Layer-wise signal:  Safety information is concentrated on Llama-3 (good
                            for HPS's chosen layers) but spread across many
                            layers on Vicuna (better suited to C4's mean-pool)
  H4. Refusal direction:  The refusal direction is weaker / more diffuse on
                            Vicuna (consistent with no-RLHF in Vicuna v1.5)
  H5. Optimization:       Contrastive loss converges to a worse minimum on
                            Vicuna's higher-dim activations
  H6. Activation stats:   Distributional differences (norms, spectra) make
                            Vicuna activations harder to project usefully

Outputs:
  results/vicuna_diagnostic.json    — full numeric results
  results/figs/vicuna_diag_*.png    — diagnostic plots

Usage:
  python vicuna_diagnostic.py \
    --llama3_cache results/llama3_activations_cache.npz \
    --vicuna_cache results/vicuna_activations_cache.npz
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hps_core import (
    LorentzProjection,
    contrastive_loss,
    extract_trajectory_features,
)

LLAMA_LAYERS = [0, 2, 17, 24, 28, 31]
VICUNA_LAYERS = [0, 2, 22, 31, 35, 39]
KAPPA_INIT = 0.1
PROJ_DIM = 64
EPOCHS = 50


# ---------------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------------

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


def to_hps_array(hs_list, layers):
    """
    Convert a list of dicts {layer_idx -> (T, d) array} to (N, n_layers, d).

    Uses last-token activation. Streams to avoid duplicating memory.
    """
    n = len(hs_list)
    if n == 0:
        return np.empty((0, len(layers), 0), dtype=np.float32)
    # Probe first item to get d
    sample = hs_list[0][layers[0]]
    d = sample.shape[1] if sample.ndim == 2 else sample.shape[0]
    out = np.empty((n, len(layers), d), dtype=np.float32)
    for i, hs in enumerate(hs_list):
        for li, L in enumerate(layers):
            t = hs[L]
            out[i, li, :] = t[-1] if t.ndim == 2 else t
    return out


def _hs_array_to_hps(np_object_array, layers):
    """
    Same as to_hps_array but iterates over a numpy object array directly
    (avoiding .tolist() which materializes all dicts in memory at once).
    Use this for very large caches.
    """
    n = len(np_object_array)
    if n == 0:
        return np.empty((0, len(layers), 0), dtype=np.float32)
    # Probe first dict
    first = np_object_array[0]
    sample = first[layers[0]]
    d = sample.shape[1] if sample.ndim == 2 else sample.shape[0]
    out = np.empty((n, len(layers), d), dtype=np.float32)
    for i in range(n):
        hs = np_object_array[i]  # one dict at a time
        for li, L in enumerate(layers):
            t = hs[L]
            out[i, li, :] = t[-1] if t.ndim == 2 else t
    return out


def _split_pre_extracted(X, seed=42, test_frac=0.2):
    """Split a pre-extracted (N, n_layers, d) array into train/test."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    n_te = max(1, int(test_frac * len(X)))
    return X[idx[n_te:]], X[idx[:n_te]]


def load_cache(path, layers):
    """
    Load activation cache. Handles two formats:

    Format A (hps_llama3.py output):
      - hs_train_ben, hs_train_atk, hs_test_ben, hs_test_atk
        each is np.array(dtype=object) of dicts {layer_idx -> (T, d)}
      - cfg_hash

    Format B (cross_model_compare.py output):
      - X_benign:    (N_ben, n_layers, d_hidden) — already last-token-extracted
      - X_attack:    (N_atk, n_layers, d_hidden)
      - attack_methods: object array
      - layers: int array

    Returns dict with X_tr_ben, X_tr_atk, X_te_ben, X_te_atk arrays of
    shape (N, n_layers, d).
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Cache not found: {path}")
    cache = np.load(path, allow_pickle=True)
    keys = list(cache.keys())

    if "hs_train_ben" in keys:
        # Format A — Llama-3 style. Stream through numpy object array
        # without materializing via .tolist() (saves memory on huge caches)
        return {
            "X_tr_ben": _hs_array_to_hps(cache["hs_train_ben"], layers),
            "X_tr_atk": _hs_array_to_hps(cache["hs_train_atk"], layers),
            "X_te_ben": _hs_array_to_hps(cache["hs_test_ben"], layers),
            "X_te_atk": _hs_array_to_hps(cache["hs_test_atk"], layers),
        }

    if "X_benign" in keys and "X_attack" in keys:
        # Format B — Vicuna style (already last-token, layer-indexed)
        X_ben = np.array(cache["X_benign"])
        X_atk = np.array(cache["X_attack"])
        cached_layers = cache["layers"].tolist() \
            if "layers" in keys else None

        # Optional: subset to requested layers if cache contains a superset
        if cached_layers is not None and set(layers) <= set(cached_layers):
            layer_idx = [cached_layers.index(L) for L in layers]
            X_ben = X_ben[:, layer_idx, :]
            X_atk = X_atk[:, layer_idx, :]
        elif cached_layers is not None and set(layers) != set(cached_layers):
            print(f"  WARNING: requested layers {layers} differ from cache "
                  f"layers {cached_layers}; using cache as-is")

        # Split into train/test (cache doesn't have a pre-split for Vicuna)
        X_tr_ben, X_te_ben = _split_pre_extracted(X_ben, seed=42)
        X_tr_atk, X_te_atk = _split_pre_extracted(X_atk, seed=43)
        return {
            "X_tr_ben": X_tr_ben, "X_tr_atk": X_tr_atk,
            "X_te_ben": X_te_ben, "X_te_atk": X_te_atk,
        }

    raise ValueError(
        f"Unknown cache format. Keys: {keys}. "
        f"Expected either 'hs_train_ben' (Llama-3 format) or 'X_benign' "
        f"(Vicuna format)."
    )


def auroc(y, s):
    return float(roc_auc_score(y, s))


def tpr_at_fpr(y, s, target=0.05):
    fpr, tpr, _ = roc_curve(y, s)
    valid = fpr <= target
    return float(tpr[valid].max()) if valid.any() else 0.0


# ---------------------------------------------------------------------------
# H1. δ-hyperbolicity test
# ---------------------------------------------------------------------------

def gromov_delta_4point(d_ij, d_kl, d_ik, d_jl, d_il, d_jk):
    """4-point δ for one quadruple (used in Gromov δ-hyperbolicity)."""
    s1 = d_ij + d_kl
    s2 = d_ik + d_jl
    s3 = d_il + d_jk
    sums = sorted([s1, s2, s3], reverse=True)
    return (sums[0] - sums[1]) / 2.0


def compute_delta_hyperbolicity(activations, n_quads=2000, seed=42):
    """
    Estimate Gromov δ-hyperbolicity by sampling random 4-tuples.

    activations: (N, d)
    Returns: median δ (lower = more tree-like / hyperbolic)
    """
    rng = np.random.RandomState(seed)
    N = len(activations)
    if N < 4:
        return float("nan"), float("nan"), float("nan")

    # Compute pairwise distances on a sample
    sample_size = min(500, N)
    idx = rng.choice(N, size=sample_size, replace=False)
    A = activations[idx]
    A_norm = A - A.mean(axis=0, keepdims=True)
    # Use Euclidean distance (proxy; true Gromov δ is for general metrics)
    D = np.sqrt(((A_norm[:, None, :] - A_norm[None, :, :]) ** 2).sum(axis=-1))

    # Normalize so the diameter is 1 (makes δ scale-free)
    diameter = D.max()
    if diameter > 0:
        D = D / diameter

    # Sample 4-tuples
    deltas = []
    for _ in range(n_quads):
        i, j, k, l = rng.choice(sample_size, size=4, replace=False)
        d_ij = D[i, j]; d_kl = D[k, l]
        d_ik = D[i, k]; d_jl = D[j, l]
        d_il = D[i, l]; d_jk = D[j, k]
        deltas.append(gromov_delta_4point(d_ij, d_kl, d_ik, d_jl, d_il, d_jk))
    deltas = np.array(deltas)
    return float(np.median(deltas)), float(np.mean(deltas)), float(deltas.max())


def test_h1_hyperbolicity(llama_data, vicuna_data):
    """H1: Compare δ-hyperbolicity of activations."""
    print("\n" + "─" * 78)
    print("H1. δ-hyperbolicity (lower = more tree-like / more hyperbolic)")
    print("─" * 78)

    results = {}
    for name, data, layers in [
        ("Llama-3", llama_data, LLAMA_LAYERS),
        ("Vicuna", vicuna_data, VICUNA_LAYERS),
    ]:
        # Combine train+test, sample
        X = np.concatenate([data["X_tr_ben"], data["X_tr_atk"]], axis=0)
        # Per-layer analysis
        per_layer = {}
        for li, L in enumerate(layers):
            X_layer = X[:, li, :]  # (N, d)
            d_med, d_mean, d_max = compute_delta_hyperbolicity(X_layer)
            per_layer[f"layer_{L}"] = {
                "delta_median": d_med, "delta_mean": d_mean, "delta_max": d_max,
            }
        results[name] = per_layer

        print(f"  {name}:")
        for layer_key, vals in per_layer.items():
            print(f"    {layer_key:>10}  δ_median = {vals['delta_median']:.4f}  "
                  f"δ_mean = {vals['delta_mean']:.4f}")

    # Compare averages
    avg_llama = np.mean([v["delta_median"] for v in results["Llama-3"].values()])
    avg_vicuna = np.mean([v["delta_median"] for v in results["Vicuna"].values()])
    diff = avg_vicuna - avg_llama

    print(f"\n  Average δ_median across layers:")
    print(f"    Llama-3:  {avg_llama:.4f}")
    print(f"    Vicuna:   {avg_vicuna:.4f}")
    print(f"    Diff:     {diff:+.4f}")

    if diff > 0.005:
        print(f"  → Vicuna is LESS hyperbolic than Llama-3 (supports H1)")
        verdict = "SUPPORTS_H1"
    elif diff < -0.005:
        print(f"  → Vicuna is MORE hyperbolic than Llama-3 (contradicts H1)")
        verdict = "CONTRADICTS_H1"
    else:
        print(f"  → Comparable hyperbolicity (H1 inconclusive)")
        verdict = "INCONCLUSIVE_H1"

    return {
        "per_layer": results,
        "avg_delta_llama3": float(avg_llama),
        "avg_delta_vicuna": float(avg_vicuna),
        "diff": float(diff),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# H2. Capacity bottleneck test
# ---------------------------------------------------------------------------

def fit_lr(X_tr, y_tr, X_te, seed=42):
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(max_iter=2000, random_state=seed,
                             class_weight="balanced")
    clf.fit(sc.transform(X_tr), y_tr)
    return clf.predict_proba(sc.transform(X_te))[:, 1]


def test_h2_capacity(llama_data, vicuna_data):
    """
    H2: Force C4 into 64-dim via PCA. Does it still work on Vicuna?
    If yes: HPS's failure is NOT capacity — it's something HPS-specific
    If no:  Vicuna genuinely needs >64 dims of capacity
    """
    print("\n" + "─" * 78)
    print("H2. Capacity bottleneck — does C4-with-64dim-PCA still work?")
    print("─" * 78)

    results = {}
    for name, data in [("Llama-3", llama_data), ("Vicuna", vicuna_data)]:
        # Mean-pool layers (C4 features)
        f_tr = np.concatenate([
            data["X_tr_ben"].mean(axis=1),
            data["X_tr_atk"].mean(axis=1),
        ], axis=0)
        f_te_ben = data["X_te_ben"].mean(axis=1)
        f_te_atk = data["X_te_atk"].mean(axis=1)
        y_tr = np.concatenate([
            np.zeros(len(data["X_tr_ben"])),
            np.ones(len(data["X_tr_atk"])),
        ])
        y_te = np.concatenate([
            np.zeros(len(f_te_ben)),
            np.ones(len(f_te_atk)),
        ])
        f_te = np.concatenate([f_te_ben, f_te_atk], axis=0)

        # 1. Full-dim C4
        scores_full = fit_lr(f_tr, y_tr, f_te)
        full_auroc = auroc(y_te, scores_full)
        full_tpr = tpr_at_fpr(y_te, scores_full)

        # 2. C4 with PCA → 64 dims
        pca = PCA(n_components=64, random_state=42).fit(f_tr)
        f_tr_pca = pca.transform(f_tr)
        f_te_pca = pca.transform(f_te)
        scores_pca = fit_lr(f_tr_pca, y_tr, f_te_pca)
        pca_auroc = auroc(y_te, scores_pca)
        pca_tpr = tpr_at_fpr(y_te, scores_pca)

        # 3. C4 with random Gaussian projection → 64 dims
        rng = np.random.RandomState(42)
        d = f_tr.shape[1]
        W_rand = rng.randn(d, 64) / np.sqrt(d)
        f_tr_rand = f_tr @ W_rand
        f_te_rand = f_te @ W_rand
        scores_rand = fit_lr(f_tr_rand, y_tr, f_te_rand)
        rand_auroc = auroc(y_te, scores_rand)
        rand_tpr = tpr_at_fpr(y_te, scores_rand)

        results[name] = {
            "C4_full":  {"auroc": full_auroc, "tpr5": full_tpr,
                         "n_dims": int(f_tr.shape[1])},
            "C4_PCA64": {"auroc": pca_auroc,  "tpr5": pca_tpr,  "n_dims": 64},
            "C4_random64": {"auroc": rand_auroc, "tpr5": rand_tpr, "n_dims": 64},
        }
        print(f"  {name}:")
        print(f"    C4 full ({f_tr.shape[1]} dims): AUROC={full_auroc:.4f}  "
              f"TPR={full_tpr:.4f}")
        print(f"    C4 PCA→64 dims:                AUROC={pca_auroc:.4f}  "
              f"TPR={pca_tpr:.4f}  "
              f"(drop: {full_tpr - pca_tpr:+.4f})")
        print(f"    C4 random→64 dims:             AUROC={rand_auroc:.4f}  "
              f"TPR={rand_tpr:.4f}  "
              f"(drop: {full_tpr - rand_tpr:+.4f})")

    # Verdict logic
    llama_full_tpr = results["Llama-3"]["C4_full"]["tpr5"]
    llama_pca_tpr  = results["Llama-3"]["C4_PCA64"]["tpr5"]
    vic_full_tpr   = results["Vicuna"]["C4_full"]["tpr5"]
    vic_pca_tpr    = results["Vicuna"]["C4_PCA64"]["tpr5"]

    llama_drop = llama_full_tpr - llama_pca_tpr
    vic_drop   = vic_full_tpr - vic_pca_tpr

    print(f"\n  Performance drop when forced to 64 dims:")
    print(f"    Llama-3 C4 drop: {llama_drop:+.4f}")
    print(f"    Vicuna  C4 drop: {vic_drop:+.4f}")

    if vic_drop > llama_drop + 0.05:
        verdict = "SUPPORTS_H2"
        print(f"  → Vicuna SUFFERS more from 64-dim bottleneck (supports H2)")
    elif vic_pca_tpr > 0.85:
        verdict = "CONTRADICTS_H2"
        print(f"  → Vicuna C4-with-PCA64 still works ({vic_pca_tpr:.3f}) — "
              f"contradicts H2; HPS issue is NOT capacity")
    else:
        verdict = "MIXED_H2"
        print(f"  → Mixed evidence on H2")

    return {**results, "verdict": verdict,
            "llama_drop": float(llama_drop),
            "vicuna_drop": float(vic_drop)}


# ---------------------------------------------------------------------------
# H3. Per-layer signal concentration
# ---------------------------------------------------------------------------

def test_h3_per_layer_signal(llama_data, vicuna_data):
    """H3: How concentrated is the safety signal across layers?"""
    print("\n" + "─" * 78)
    print("H3. Per-layer signal — single-layer LR probe AUROC")
    print("─" * 78)

    results = {}
    for name, data, layers in [
        ("Llama-3", llama_data, LLAMA_LAYERS),
        ("Vicuna", vicuna_data, VICUNA_LAYERS),
    ]:
        per_layer = []
        for li, L in enumerate(layers):
            X_tr = np.concatenate([
                data["X_tr_ben"][:, li, :],
                data["X_tr_atk"][:, li, :],
            ], axis=0)
            y_tr = np.concatenate([
                np.zeros(len(data["X_tr_ben"])),
                np.ones(len(data["X_tr_atk"])),
            ])
            X_te = np.concatenate([
                data["X_te_ben"][:, li, :],
                data["X_te_atk"][:, li, :],
            ], axis=0)
            y_te = np.concatenate([
                np.zeros(len(data["X_te_ben"])),
                np.ones(len(data["X_te_atk"])),
            ])
            scores = fit_lr(X_tr, y_tr, X_te)
            per_layer.append({
                "layer": L,
                "auroc": auroc(y_te, scores),
                "tpr5": tpr_at_fpr(y_te, scores),
            })
        results[name] = per_layer

        aurocs = np.array([p["auroc"] for p in per_layer])
        max_a = float(aurocs.max())
        min_a = float(aurocs.min())
        spread = max_a - min_a
        std_a = float(aurocs.std())
        print(f"  {name}:")
        for p in per_layer:
            print(f"    layer {p['layer']:>2}: AUROC = {p['auroc']:.4f}  "
                  f"TPR@5%FPR = {p['tpr5']:.4f}")
        print(f"    summary: max={max_a:.4f}, min={min_a:.4f}, "
              f"spread={spread:.4f}, std={std_a:.4f}")

    # Compare cross-layer std (low std = uniformly informative;
    # high std = concentrated in specific layers)
    llama_std = np.std([p["auroc"] for p in results["Llama-3"]])
    vicuna_std = np.std([p["auroc"] for p in results["Vicuna"]])
    print(f"\n  Cross-layer AUROC std:")
    print(f"    Llama-3: {llama_std:.4f}")
    print(f"    Vicuna:  {vicuna_std:.4f}")
    if abs(vicuna_std - llama_std) < 0.01:
        verdict = "SIMILAR_DISTRIBUTION"
        print(f"  → Both LLMs have similar signal distribution across layers")
    elif vicuna_std > llama_std:
        verdict = "VICUNA_MORE_CONCENTRATED"
        print(f"  → Vicuna's signal is MORE concentrated (one layer carries it)")
    else:
        verdict = "VICUNA_MORE_SPREAD"
        print(f"  → Vicuna's signal is MORE spread across layers (supports H3)")

    return {
        "per_layer": results,
        "llama_cross_layer_std": float(llama_std),
        "vicuna_cross_layer_std": float(vicuna_std),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# H4. Refusal direction strength
# ---------------------------------------------------------------------------

def test_h4_refusal_direction(llama_data, vicuna_data):
    """H4: How concentrated is the refusal direction?"""
    print("\n" + "─" * 78)
    print("H4. Refusal direction strength")
    print("─" * 78)

    results = {}
    for name, data, layers in [
        ("Llama-3", llama_data, LLAMA_LAYERS),
        ("Vicuna", vicuna_data, VICUNA_LAYERS),
    ]:
        per_layer = []
        for li, L in enumerate(layers):
            ben = data["X_tr_ben"][:, li, :]  # (N_ben, d)
            atk = data["X_tr_atk"][:, li, :]  # (N_atk, d)

            # Refusal direction = mean(harmful) - mean(benign)
            r = atk.mean(axis=0) - ben.mean(axis=0)
            r_norm = float(np.linalg.norm(r))

            # Signal-to-noise ratio
            avg_act_norm = float(np.linalg.norm(
                np.concatenate([ben, atk], axis=0), axis=1).mean())
            snr = r_norm / (avg_act_norm + 1e-8)

            # Class separation along r (in standardized units)
            r_unit = r / (r_norm + 1e-8)
            ben_proj = ben @ r_unit
            atk_proj = atk @ r_unit
            d_prime = (atk_proj.mean() - ben_proj.mean()) / np.sqrt(
                (ben_proj.var() + atk_proj.var()) / 2 + 1e-8
            )
            per_layer.append({
                "layer": L,
                "refusal_norm": r_norm,
                "snr": float(snr),
                "d_prime": float(d_prime),
            })
        results[name] = per_layer

        avg_dprime = np.mean([p["d_prime"] for p in per_layer])
        avg_snr = np.mean([p["snr"] for p in per_layer])
        print(f"  {name}:")
        for p in per_layer:
            print(f"    layer {p['layer']:>2}: |r|={p['refusal_norm']:.2f}  "
                  f"SNR={p['snr']:.3f}  d′={p['d_prime']:.3f}")
        print(f"    avg d′: {avg_dprime:.4f}, avg SNR: {avg_snr:.4f}")

    llama_dprime = np.mean([p["d_prime"] for p in results["Llama-3"]])
    vicuna_dprime = np.mean([p["d_prime"] for p in results["Vicuna"]])
    diff = vicuna_dprime - llama_dprime
    print(f"\n  Average d′ (separation):")
    print(f"    Llama-3:  {llama_dprime:.4f}")
    print(f"    Vicuna:   {vicuna_dprime:.4f}")
    print(f"    Diff:     {diff:+.4f}")
    if diff < -0.3:
        verdict = "SUPPORTS_H4"
        print(f"  → Vicuna refusal direction is WEAKER (supports H4)")
    elif diff > 0.3:
        verdict = "CONTRADICTS_H4"
        print(f"  → Vicuna refusal direction is STRONGER (contradicts H4)")
    else:
        verdict = "INCONCLUSIVE_H4"
        print(f"  → Comparable refusal direction strength")

    return {
        "per_layer": results,
        "avg_dprime_llama3": float(llama_dprime),
        "avg_dprime_vicuna": float(vicuna_dprime),
        "diff": float(diff),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# H5. HPS optimization comparison
# ---------------------------------------------------------------------------

def train_hps_track_loss(X_tr, y_tr, n_layers, d, kappa_init, epochs, seed,
                          device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    proj = LorentzProjection(d, PROJ_DIM, kappa_init,
                              n_layers=n_layers).to(device)
    proj.log_k.requires_grad = False
    opt = optim.Adam(
        [p for p in proj.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=1e-5,
    )
    X_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_tr, dtype=torch.long, device=device)

    losses = []
    for epoch in range(epochs):
        loss = torch.tensor(0.0, device=device)
        for li in range(n_layers):
            h = proj(X_t[:, li, :])
            loss = loss + contrastive_loss(h, y_t, k=proj.k, tau=proj.tau(li))
        loss = loss / n_layers
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return proj, losses


def test_h5_optimization(llama_data, vicuna_data, device, epochs=EPOCHS):
    """H5: Does HPS converge to a worse minimum on Vicuna?"""
    print("\n" + "─" * 78)
    print("H5. HPS optimization comparison")
    print("─" * 78)

    results = {}
    for name, data in [("Llama-3", llama_data), ("Vicuna", vicuna_data)]:
        X_tr = np.concatenate([data["X_tr_ben"], data["X_tr_atk"]], axis=0)
        y_tr = np.concatenate([
            np.zeros(len(data["X_tr_ben"])),
            np.ones(len(data["X_tr_atk"])),
        ])
        proj, losses = train_hps_track_loss(
            X_tr, y_tr, X_tr.shape[1], X_tr.shape[2],
            KAPPA_INIT, epochs, seed=42, device=device,
        )

        # Evaluate post-training class separation
        feats_tr = extract_trajectory_features(proj, X_tr)
        feats_te_ben = extract_trajectory_features(proj, data["X_te_ben"])
        feats_te_atk = extract_trajectory_features(proj, data["X_te_atk"])
        sc = StandardScaler().fit(feats_tr)
        clf = LogisticRegression(max_iter=2000, random_state=42,
                                  class_weight="balanced")
        clf.fit(sc.transform(feats_tr), y_tr)
        scores_te = np.concatenate([
            clf.predict_proba(sc.transform(feats_te_ben))[:, 1],
            clf.predict_proba(sc.transform(feats_te_atk))[:, 1],
        ])
        y_te = np.concatenate([
            np.zeros(len(data["X_te_ben"])),
            np.ones(len(data["X_te_atk"])),
        ])
        results[name] = {
            "final_loss": losses[-1],
            "loss_history": losses,
            "test_auroc": auroc(y_te, scores_te),
            "test_tpr5": tpr_at_fpr(y_te, scores_te),
        }
        print(f"  {name}: final_loss={losses[-1]:.4f}  "
              f"test_AUROC={results[name]['test_auroc']:.4f}  "
              f"test_TPR={results[name]['test_tpr5']:.4f}")

    diff = results["Vicuna"]["final_loss"] - results["Llama-3"]["final_loss"]
    print(f"\n  Final loss diff (Vicuna − Llama-3): {diff:+.4f}")
    if diff > 0.05:
        verdict = "SUPPORTS_H5"
        print(f"  → Vicuna converges to worse loss (supports H5)")
    elif diff < -0.05:
        verdict = "CONTRADICTS_H5"
        print(f"  → Vicuna converges to better loss (contradicts H5)")
    else:
        verdict = "INCONCLUSIVE_H5"
        print(f"  → Comparable convergence loss")

    return {
        "Llama-3": results["Llama-3"],
        "Vicuna":  results["Vicuna"],
        "loss_diff": float(diff),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# H6. Activation distribution statistics
# ---------------------------------------------------------------------------

def test_h6_activation_stats(llama_data, vicuna_data):
    """H6: Do the activations have distributional differences?"""
    print("\n" + "─" * 78)
    print("H6. Activation distribution statistics")
    print("─" * 78)

    results = {}
    for name, data, layers in [
        ("Llama-3", llama_data, LLAMA_LAYERS),
        ("Vicuna", vicuna_data, VICUNA_LAYERS),
    ]:
        all_acts = np.concatenate(
            [data["X_tr_ben"], data["X_tr_atk"]], axis=0)
        per_layer = []
        for li, L in enumerate(layers):
            X_layer = all_acts[:, li, :]
            norms = np.linalg.norm(X_layer, axis=1)
            # Eigenvalue spectrum (effective dimensionality)
            X_centered = X_layer - X_layer.mean(axis=0, keepdims=True)
            U, S, _ = np.linalg.svd(
                X_centered[:200] if len(X_centered) > 200 else X_centered,
                full_matrices=False,
            )
            S_squared = S ** 2
            S_squared = S_squared / S_squared.sum()
            # Effective dim: how many components for 90% variance
            cumsum = np.cumsum(S_squared)
            eff_dim_90 = int(np.searchsorted(cumsum, 0.9) + 1)
            participation = float(1.0 / (S_squared ** 2).sum())
            per_layer.append({
                "layer": L,
                "norm_mean": float(norms.mean()),
                "norm_std": float(norms.std()),
                "eff_dim_90pct": eff_dim_90,
                "participation_ratio": participation,
                "d_total": int(X_layer.shape[1]),
            })
        results[name] = per_layer

        avg_norm = np.mean([p["norm_mean"] for p in per_layer])
        avg_pr = np.mean([p["participation_ratio"] for p in per_layer])
        avg_eff_dim = np.mean([p["eff_dim_90pct"] for p in per_layer])
        print(f"  {name}:")
        for p in per_layer:
            print(f"    layer {p['layer']:>2}: |x|≈{p['norm_mean']:7.2f}  "
                  f"eff_dim_90%={p['eff_dim_90pct']:>3}  "
                  f"PR={p['participation_ratio']:.1f}")
        print(f"    avg norm: {avg_norm:.2f}, avg eff_dim: {avg_eff_dim:.0f}, "
              f"avg PR: {avg_pr:.1f}")

    llama_eff = np.mean([p["eff_dim_90pct"] for p in results["Llama-3"]])
    vicuna_eff = np.mean([p["eff_dim_90pct"] for p in results["Vicuna"]])
    print(f"\n  Effective dimensionality (90% variance):")
    print(f"    Llama-3:  {llama_eff:.0f}")
    print(f"    Vicuna:   {vicuna_eff:.0f}")
    if vicuna_eff > llama_eff + 5:
        verdict = "SUPPORTS_H6"
        print(f"  → Vicuna activations have HIGHER effective dim (supports H6 — "
              f"more dims needed)")
    elif vicuna_eff < llama_eff - 5:
        verdict = "CONTRADICTS_H6"
        print(f"  → Vicuna activations are LOWER effective dim "
              f"(contradicts H6)")
    else:
        verdict = "INCONCLUSIVE_H6"
        print(f"  → Comparable effective dimensionality")

    return {
        "per_layer": results,
        "avg_eff_dim_llama3": float(llama_eff),
        "avg_eff_dim_vicuna": float(vicuna_eff),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama3_cache",
                        default="results/llama3_activations_cache.npz")
    parser.add_argument("--vicuna_cache",
                        default="results/vicuna_activations_cache.npz")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output",
                        default="results/vicuna_diagnostic.json")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--skip_h5", action="store_true",
                        help="Skip the HPS optimization comparison (slow)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print("VICUNA FAILURE DIAGNOSTIC")
    print("=" * 78)
    print(f"  Llama-3 cache: {args.llama3_cache}")
    print(f"  Vicuna  cache: {args.vicuna_cache}")
    print(f"  Device:        {device}")
    print()

    # ── Load data ──
    print("Loading activation caches...")
    llama_data = load_cache(args.llama3_cache, LLAMA_LAYERS)
    vicuna_data = load_cache(args.vicuna_cache, VICUNA_LAYERS)
    print(f"  Llama-3 train: {llama_data['X_tr_ben'].shape[0]} ben + "
          f"{llama_data['X_tr_atk'].shape[0]} atk, "
          f"d={llama_data['X_tr_ben'].shape[2]}")
    print(f"  Vicuna  train: {vicuna_data['X_tr_ben'].shape[0]} ben + "
          f"{vicuna_data['X_tr_atk'].shape[0]} atk, "
          f"d={vicuna_data['X_tr_ben'].shape[2]}")

    output = {
        "config": {
            "device": device,
            "llama_layers": LLAMA_LAYERS,
            "vicuna_layers": VICUNA_LAYERS,
            "kappa_init": KAPPA_INIT,
            "epochs": args.epochs,
        },
    }

    # ── Run all hypotheses ──
    output["H1_hyperbolicity"]      = test_h1_hyperbolicity(llama_data, vicuna_data)
    output["H2_capacity"]            = test_h2_capacity(llama_data, vicuna_data)
    output["H3_per_layer_signal"]    = test_h3_per_layer_signal(llama_data, vicuna_data)
    output["H4_refusal_direction"]   = test_h4_refusal_direction(llama_data, vicuna_data)
    if not args.skip_h5:
        output["H5_optimization"]    = test_h5_optimization(
            llama_data, vicuna_data, device, args.epochs)
    output["H6_activation_stats"]    = test_h6_activation_stats(llama_data, vicuna_data)

    # ── Summary verdict table ──
    print("\n" + "=" * 78)
    print("SUMMARY OF HYPOTHESES")
    print("=" * 78)
    print()
    print("  H1: δ-hyperbolicity                  → "
          f"{output['H1_hyperbolicity']['verdict']}")
    print("  H2: Capacity bottleneck (PCA64)      → "
          f"{output['H2_capacity']['verdict']}")
    print("  H3: Per-layer signal concentration   → "
          f"{output['H3_per_layer_signal']['verdict']}")
    print("  H4: Refusal direction strength       → "
          f"{output['H4_refusal_direction']['verdict']}")
    if "H5_optimization" in output:
        print("  H5: HPS optimization quality         → "
              f"{output['H5_optimization']['verdict']}")
    print("  H6: Activation distribution          → "
          f"{output['H6_activation_stats']['verdict']}")
    print()

    # Identify which hypotheses are supported
    supports = []
    for k, v in output.items():
        if isinstance(v, dict) and v.get("verdict", "").startswith("SUPPORTS"):
            supports.append(k.replace("_", " "))
    if supports:
        print(f"  PRIMARY EXPLANATIONS: {', '.join(supports)}")
    else:
        print("  No clear single explanation — inspect individual results")

    # ── Save ──
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=_np_default)
    print(f"\nSaved full results to {output_path}")

    # ── Plots ──
    print("\nGenerating diagnostic plots...")
    figdir = Path("results/figs")
    figdir.mkdir(parents=True, exist_ok=True)

    # Plot 1: per-layer AUROC comparison (H3)
    fig, ax = plt.subplots(figsize=(8, 4))
    h3 = output["H3_per_layer_signal"]["per_layer"]
    for name, color in [("Llama-3", "tab:blue"), ("Vicuna", "tab:orange")]:
        layers_idx = list(range(len(h3[name])))
        aurocs = [p["auroc"] for p in h3[name]]
        layer_labels = [str(p["layer"]) for p in h3[name]]
        ax.plot(layers_idx, aurocs, "o-", label=name, color=color, lw=2)
        for i, (lx, ly, lbl) in enumerate(zip(layers_idx, aurocs, layer_labels)):
            ax.annotate(lbl, (lx, ly), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("Layer index (0=shallow, last=deep)")
    ax.set_ylabel("Single-layer LR probe AUROC")
    ax.set_title("Per-layer separability — Vicuna vs Llama-3")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([0.5, 1.05])
    plt.tight_layout()
    fig.savefig(figdir / "vicuna_diag_h3_per_layer.png", dpi=120)
    plt.close(fig)

    # Plot 2: H5 loss curves (if computed)
    if "H5_optimization" in output:
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, color in [("Llama-3", "tab:blue"), ("Vicuna", "tab:orange")]:
            losses = output["H5_optimization"][name]["loss_history"]
            ax.plot(range(1, len(losses) + 1), losses, label=name,
                    color=color, lw=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Contrastive loss")
        ax.set_title("HPS training loss — Vicuna vs Llama-3")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(figdir / "vicuna_diag_h5_loss.png", dpi=120)
        plt.close(fig)

    # Plot 3: H4 refusal direction strength
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, color in [("Llama-3", "tab:blue"), ("Vicuna", "tab:orange")]:
        per = output["H4_refusal_direction"]["per_layer"][name]
        layers_idx = list(range(len(per)))
        dprimes = [p["d_prime"] for p in per]
        ax.bar(
            [li + (0 if name == "Llama-3" else 0.4) for li in layers_idx],
            dprimes, width=0.4, label=name, color=color,
        )
    ax.set_xlabel("Layer index")
    ax.set_ylabel("d′ (class separation along refusal direction)")
    ax.set_title("Refusal direction strength — Vicuna vs Llama-3")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(figdir / "vicuna_diag_h4_refusal.png", dpi=120)
    plt.close(fig)

    print(f"  saved {figdir}/vicuna_diag_h3_per_layer.png")
    if "H5_optimization" in output:
        print(f"  saved {figdir}/vicuna_diag_h5_loss.png")
    print(f"  saved {figdir}/vicuna_diag_h4_refusal.png")

    print()
    print("=" * 78)
    print("INTERPRETATION GUIDE")
    print("=" * 78)
    print("""
The verdict for each hypothesis tells you WHICH mechanism is plausibly
responsible for HPS failing on Vicuna:

H1 (δ-hyperbolicity):
  SUPPORTS_H1     → Vicuna activations are less tree-like → hyperbolic
                    geometry can't help.  Cure: don't use hyperbolic for
                    Vicuna; use a different geometry.

H2 (capacity bottleneck):
  SUPPORTS_H2     → 64-dim is too small for Vicuna's signal.  Cure: increase
                    projection dim.
  CONTRADICTS_H2  → Capacity isn't the bottleneck; HPS is failing in another
                    specific way.

H3 (per-layer signal):
  VICUNA_MORE_SPREAD → Signal is across many layers; HPS's chosen 6 layers
                       miss most of it.  Cure: rescan layer selection.

H4 (refusal direction):
  SUPPORTS_H4     → Vicuna's refusal direction is intrinsically weaker
                    (consistent with no-RLHF).  Less to detect, regardless of
                    method.

H5 (optimization):
  SUPPORTS_H5     → Same architecture, but Vicuna training converges worse.
                    Cure: better optimizer / longer training / regularization.

H6 (activation stats):
  SUPPORTS_H6     → Vicuna activations have higher intrinsic dimensionality
                    → 64-dim projection inevitably loses more info.

If H1+H6 both support, the diagnosis is "Vicuna is just harder for any 64-dim
projection."  If H2 contradicts and H4 supports, the issue is more fundamental
(weak signal, not bottleneck).  Etc.
""")


if __name__ == "__main__":
    main()
