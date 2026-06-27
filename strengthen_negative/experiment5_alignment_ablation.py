"""
Experiment 5 — Alignment ablation: Llama-3-8B-base vs Llama-3-8B-Instruct.

Goal: Disentangle the alignment-mediation hypothesis from confounds. The
Vicuna result (HPS fails on GCG; C4 succeeds) was confounded with model
size (13B vs 8B), layer choice, and data source. Here we hold all of these
constant and vary ONLY the alignment training.

Hypothesis: HPS depends on alignment-shaped activation structure. If alignment
creates concentrated discriminative signatures (RLHF), HPS works. If
activations have only pre-training structure (base model), HPS fails on
gradient-optimized attacks like GCG.

Setup:
  - Model A: meta-llama/Meta-Llama-3-8B (base, no SFT, no RLHF)
  - Model B: meta-llama/Meta-Llama-3-8B-Instruct (SFT + RLHF, already cached)
  - Same architecture (32 layers, 4096-dim), same layer indices, same data,
    same training procedure, same hyperparameters
  - Trains HPS, HPS-Euclidean, C4 on each model's activations
  - Compares per-attack TPR

Predictions:
  If alignment-mediation hypothesis is true:
    - HPS Instruct GCG TPR ~= 1.000 (already observed)
    - HPS base GCG TPR << 1.000 (signal not concentrated)
    - C4 base GCG TPR ~= 1.000 (full activation space preserves signal)

  If alignment-mediation is false:
    - HPS base GCG TPR ~= 1.000 (signal exists pre-alignment)
    - The Vicuna failure was due to other confounds

Caveat: Base Llama-3 will not refuse harmful prompts; we are testing whether
the activation pattern that DISTINGUISHES harmful from benign prompts exists,
not whether the model behaves safely.

Usage:
  # Step 1: Extract base Llama-3 activations (~2-3 hours on DGX)
  python strengthen_negative/experiment5_alignment_ablation.py --extract

  # Step 2: Run comparison (uses cached Instruct activations from hps_llama3.py)
  python strengthen_negative/experiment5_alignment_ablation.py --compare
"""

import os
import sys
import json
import time
import argparse
import hashlib

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from helpers.lorentz_ops import (
    LorentzProjection, lorentz_distance, to_hyperboloid
)

# ----------------------------- Config -----------------------------
BASE_MODEL = "meta-llama/Meta-Llama-3-8B"
INSTRUCT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
ALL_LAYERS = [0, 2, 17, 24, 28, 31]
HPS_LAYERS = [0, 2, 17, 24, 28, 31]
KAPPA_INIT = 0.1
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-5
PROJ_DIM = 64
FPR_TARGET = 0.05
N_SEEDS = 3

BASE_CACHE = "results/llama3_base_activations_cache.npz"
INSTRUCT_CACHE = "results/llama3_activations_cache.npz"

device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------- Activation extraction --------------------------
def extract_hidden_states(model, tokenizer, prompt, layers, max_len=512):
    """Forward pass and capture last-token activation at each layer."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=max_len).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    result = {}
    for l in layers:
        # hidden_states[l] is shape (1, seq_len, d_hidden)
        # take last token
        result[l] = outputs.hidden_states[l][0, -1].cpu().float().numpy()
    return result


def extract_all(model, tokenizer, prompts, layers, label):
    results = []
    for i, p in enumerate(prompts):
        try:
            hs = extract_hidden_states(model, tokenizer, p, layers)
            results.append(hs)
        except Exception as e:
            print(f"  WARN: failed on prompt {i}: {type(e).__name__}: {e}")
        if (i + 1) % 100 == 0:
            print(f"    {label}: {i+1}/{len(prompts)}")
    return results


def load_dataset(test_attacks_path, harmless_path, n_calib=100):
    """Load the same dataset used by hps_llama3.py."""
    import pandas as pd

    # Harmless prompts (Alpaca-style)
    df_harmless = pd.read_csv(harmless_path)
    harmless = df_harmless.iloc[:, 0].astype(str).tolist()

    # Attack prompts (categorized JSON)
    with open(test_attacks_path) as f:
        categorized = json.load(f)

    # Flatten attack list and remember category labels
    attack_prompts = []
    attack_methods = []
    for method, prompts in categorized.items():
        for p in prompts:
            if p:
                attack_prompts.append(p)
                attack_methods.append(method)

    # Use same split as hps_llama3.py: 80/20 stratified by method
    rng = np.random.RandomState(42)
    atk_idx = rng.permutation(len(attack_prompts))
    n_train_atk = int(0.8 * len(atk_idx))
    train_atk_idx = atk_idx[:n_train_atk]
    test_atk_idx = atk_idx[n_train_atk:]

    train_atk = [attack_prompts[i] for i in train_atk_idx]
    test_atk = [attack_prompts[i] for i in test_atk_idx]
    train_methods = [attack_methods[i] for i in train_atk_idx]
    test_methods = [attack_methods[i] for i in test_atk_idx]

    # Match benign size
    n_train_ben = len(train_atk)
    n_test_ben = len(test_atk)
    ben_idx = rng.permutation(len(harmless))
    train_ben = [harmless[i] for i in ben_idx[:n_train_ben]]
    test_ben = [harmless[i] for i in ben_idx[n_train_ben:n_train_ben + n_test_ben]]

    return {
        "train_ben": train_ben,
        "train_atk": train_atk,
        "train_atk_methods": train_methods,
        "test_ben": test_ben,
        "test_atk": test_atk,
        "test_atk_methods": test_methods,
    }


def extract_and_cache(model_name, dataset, cache_path):
    """Extract activations and save to disk."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n[extract] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    print("[extract] Model loaded.")

    cfg_str = f"{model_name}|{ALL_LAYERS}|{len(dataset['train_atk'])}"
    cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()[:8]

    print(f"[extract] Extracting (hash={cfg_hash})...")
    print("  Train benign...")
    hs_train_ben = extract_all(model, tokenizer, dataset["train_ben"], ALL_LAYERS, "train_ben")
    print("  Train attacks...")
    hs_train_atk = extract_all(model, tokenizer, dataset["train_atk"], ALL_LAYERS, "train_atk")
    print("  Test benign...")
    hs_test_ben = extract_all(model, tokenizer, dataset["test_ben"], ALL_LAYERS, "test_ben")
    print("  Test attacks...")
    hs_test_atk = extract_all(model, tokenizer, dataset["test_atk"], ALL_LAYERS, "test_atk")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path,
             hs_train_ben=np.array(hs_train_ben, dtype=object),
             hs_train_atk=np.array(hs_train_atk, dtype=object),
             hs_test_ben=np.array(hs_test_ben, dtype=object),
             hs_test_atk=np.array(hs_test_atk, dtype=object),
             train_atk_methods=np.array(dataset["train_atk_methods"], dtype=object),
             test_atk_methods=np.array(dataset["test_atk_methods"], dtype=object),
             cfg_hash=np.array(cfg_hash))
    print(f"[extract] Cached to {cache_path}")


# -------------------------- Loading helpers --------------------------
def to_arr(hs_list, layers):
    return np.array([[hs[l] for l in layers] for hs in hs_list])


def load_cached(cache_path):
    """Load activations from cache. Works for both base and instruct caches."""
    if not os.path.exists(cache_path):
        return None

    cache = np.load(cache_path, allow_pickle=True)
    hs_train_ben = cache["hs_train_ben"].tolist()
    hs_train_atk = cache["hs_train_atk"].tolist()
    hs_test_ben = cache["hs_test_ben"].tolist()
    hs_test_atk = cache["hs_test_atk"].tolist()

    # Both caches store dict-of-layer or array per sample. Handle both:
    if isinstance(hs_train_ben[0], dict):
        # Dict format: {layer_idx: vector}
        X_tr_ben = to_arr(hs_train_ben, HPS_LAYERS)
        X_tr_atk = to_arr(hs_train_atk, HPS_LAYERS)
        X_te_ben = to_arr(hs_test_ben, HPS_LAYERS)
        X_te_atk = to_arr(hs_test_atk, HPS_LAYERS)
    else:
        # Already-array format from hps_llama3 (last-token):
        # the existing cache is dict-like via [hs[l][-1] for l in layers]
        X_tr_ben = np.array([[hs[l][-1] for l in HPS_LAYERS] for hs in hs_train_ben])
        X_tr_atk = np.array([[hs[l][-1] for l in HPS_LAYERS] for hs in hs_train_atk])
        X_te_ben = np.array([[hs[l][-1] for l in HPS_LAYERS] for hs in hs_test_ben])
        X_te_atk = np.array([[hs[l][-1] for l in HPS_LAYERS] for hs in hs_test_atk])

    test_methods = cache["test_atk_methods"].tolist() if "test_atk_methods" in cache.files else None

    return {
        "X_tr_ben": X_tr_ben, "X_tr_atk": X_tr_atk,
        "X_te_ben": X_te_ben, "X_te_atk": X_te_atk,
        "test_methods": test_methods,
    }


# -------------------------- Methods (HPS, HPS-Euc, C4) --------------------------
def eval_features(features_train, y_train, features_te_ben, features_te_atk,
                  per_method_test_idx=None, methods_unique=None, seed=42):
    """Standard eval + optional per-method TPR breakdown."""
    if features_train.ndim == 1:
        features_train = features_train.reshape(-1, 1)
        features_te_ben = features_te_ben.reshape(-1, 1)
        features_te_atk = features_te_atk.reshape(-1, 1)

    sc = StandardScaler()
    Xtr = sc.fit_transform(features_train)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(Xtr, y_train)

    n_calib = max(1, len(features_te_ben) // 2)
    s_calib = clf.predict_proba(sc.transform(features_te_ben[:n_calib]))[:, 1]
    s_ben = clf.predict_proba(sc.transform(features_te_ben[n_calib:]))[:, 1]
    s_atk = clf.predict_proba(sc.transform(features_te_atk))[:, 1]

    thr = float(np.quantile(s_calib, 1.0 - FPR_TARGET))
    tpr = float((s_atk > thr).mean())
    fpr = float((s_ben > thr).mean())
    auroc = roc_auc_score(
        np.array([0] * len(s_ben) + [1] * len(s_atk)),
        np.concatenate([s_ben, s_atk])
    )

    per_method_tpr = {}
    if per_method_test_idx and methods_unique:
        atk_correct = (s_atk > thr)
        for m in methods_unique:
            idx = per_method_test_idx.get(m, [])
            if len(idx) > 0:
                method_tpr = float(atk_correct[idx].mean())
                per_method_tpr[m] = {"tpr": method_tpr, "n": len(idx)}

    return {"auroc": auroc, "tpr": tpr, "fpr": fpr, "per_method": per_method_tpr,
            "threshold": thr}


def method_hps(data, seed=42, kappa=KAPPA_INIT, epochs=EPOCHS,
               per_method_test_idx=None, methods_unique=None):
    """HPS: Lorentz projection + contrastive + mean radial feature."""
    n_layers = data["X_tr_ben"].shape[1]
    d_hidden = data["X_tr_ben"].shape[2]

    X_train = np.concatenate([data["X_tr_ben"], data["X_tr_atk"]])
    y_train = np.array([0] * len(data["X_tr_ben"]) + [1] * len(data["X_tr_atk"]))

    torch.manual_seed(seed)
    proj = LorentzProjection(d_hidden, PROJ_DIM, k_init=kappa, freeze_kappa=True).to(device)
    opt = optim.Adam([p for p in proj.parameters() if p.requires_grad],
                     lr=LR, weight_decay=WEIGHT_DECAY)
    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.long, device=device)

    for _ in range(epochs):
        loss = torch.tensor(0.0, device=device)
        for l in range(n_layers):
            h = proj(Xt[:, l, :])
            d = lorentz_distance(h.unsqueeze(0), h.unsqueeze(1), k=proj.k.item())
            same = (yt.unsqueeze(0) == yt.unsqueeze(1)).float()
            diff = 1.0 - same
            triu = torch.triu(torch.ones_like(d), diagonal=1)
            ns = (same * triu).sum().clamp(min=1)
            nd = (diff * triu).sum().clamp(min=1)
            margin = 2.0
            loss = loss + ((d ** 2 * same * triu).sum() / ns +
                           (torch.clamp(margin - d, min=0) ** 2 * diff * triu).sum() / nd) / 2
        loss = loss / n_layers
        opt.zero_grad()
        loss.backward()
        opt.step()

    proj.eval()

    def feats(X):
        out = []
        with torch.no_grad():
            for i in range(len(X)):
                xi = torch.tensor(X[i], dtype=torch.float32, device=device)
                h = proj(xi)
                out.append(float(h[:, 0].mean().cpu()))
        return np.array(out).reshape(-1, 1)

    f_tr = feats(X_train)
    f_be = feats(data["X_te_ben"])
    f_at = feats(data["X_te_atk"])
    return eval_features(f_tr, y_train, f_be, f_at,
                         per_method_test_idx=per_method_test_idx,
                         methods_unique=methods_unique, seed=seed)


def method_hps_euclidean(data, seed=42, epochs=EPOCHS,
                          per_method_test_idx=None, methods_unique=None):
    """Parameter-matched Euclidean variant: same architecture, flat space."""
    n_layers = data["X_tr_ben"].shape[1]
    d_hidden = data["X_tr_ben"].shape[2]

    X_train = np.concatenate([data["X_tr_ben"], data["X_tr_atk"]])
    y_train = np.array([0] * len(data["X_tr_ben"]) + [1] * len(data["X_tr_atk"]))

    torch.manual_seed(seed)
    proj = nn.Linear(d_hidden, PROJ_DIM, bias=False).to(device)
    nn.init.xavier_uniform_(proj.weight)
    scale_per_layer = nn.Parameter(torch.ones(n_layers, device=device) / 8.0)
    log_margin = nn.Parameter(torch.tensor(np.log(2.0), device=device))
    opt = optim.Adam(list(proj.parameters()) + [scale_per_layer, log_margin],
                     lr=LR, weight_decay=WEIGHT_DECAY)
    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.long, device=device)

    for _ in range(epochs):
        loss = torch.tensor(0.0, device=device)
        margin = torch.exp(log_margin).clamp(0.5, 5.0)
        for l in range(n_layers):
            h = proj(Xt[:, l, :]) * scale_per_layer[l]
            d = torch.cdist(h, h)
            same = (yt.unsqueeze(0) == yt.unsqueeze(1)).float()
            diff = 1.0 - same
            triu = torch.triu(torch.ones_like(d), diagonal=1)
            ns = (same * triu).sum().clamp(min=1)
            nd = (diff * triu).sum().clamp(min=1)
            loss = loss + ((d ** 2 * same * triu).sum() / ns +
                           (torch.clamp(margin - d, min=0) ** 2 * diff * triu).sum() / nd) / 2
        loss = loss / n_layers
        opt.zero_grad()
        loss.backward()
        opt.step()

    proj.eval()

    def feats(X):
        out = []
        with torch.no_grad():
            for i in range(len(X)):
                xi = torch.tensor(X[i], dtype=torch.float32, device=device)
                pts = []
                for l in range(n_layers):
                    pts.append(proj(xi[l:l+1]) * scale_per_layer[l])
                pts = torch.cat(pts)
                out.append(float(torch.norm(pts, dim=1).mean().cpu()))
        return np.array(out).reshape(-1, 1)

    f_tr = feats(X_train)
    f_be = feats(data["X_te_ben"])
    f_at = feats(data["X_te_atk"])
    return eval_features(f_tr, y_train, f_be, f_at,
                         per_method_test_idx=per_method_test_idx,
                         methods_unique=methods_unique, seed=seed)


def method_c4(data, seed=42, per_method_test_idx=None, methods_unique=None):
    """C4: linear probe on mean-pooled activations."""
    f_tr_ben = data["X_tr_ben"].mean(axis=1)
    f_tr_atk = data["X_tr_atk"].mean(axis=1)
    f_te_ben = data["X_te_ben"].mean(axis=1)
    f_te_atk = data["X_te_atk"].mean(axis=1)
    f_tr = np.concatenate([f_tr_ben, f_tr_atk])
    y_train = np.array([0] * len(f_tr_ben) + [1] * len(f_tr_atk))
    return eval_features(f_tr, y_train, f_te_ben, f_te_atk,
                         per_method_test_idx=per_method_test_idx,
                         methods_unique=methods_unique, seed=seed)


# -------------------------- Comparison --------------------------
def evaluate_model(name, data, methods_unique, seed=42):
    """Run all 3 methods and return per-method TPR breakdown."""
    print(f"\n  [{name}] Building per-method test indices...")
    per_method_test_idx = {m: [] for m in methods_unique}
    for i, m in enumerate(data["test_methods"]):
        if m in per_method_test_idx:
            per_method_test_idx[m].append(i)
    for m in methods_unique:
        per_method_test_idx[m] = np.array(per_method_test_idx[m])
        print(f"    {m}: {len(per_method_test_idx[m])} test samples")

    results = {}

    print(f"\n  [{name}] Training HPS...")
    results["HPS"] = method_hps(data, seed=seed,
                                 per_method_test_idx=per_method_test_idx,
                                 methods_unique=methods_unique)

    print(f"  [{name}] Training HPS-Euclidean...")
    results["HPS-Euclidean"] = method_hps_euclidean(
        data, seed=seed,
        per_method_test_idx=per_method_test_idx,
        methods_unique=methods_unique
    )

    print(f"  [{name}] Training C4...")
    results["C4"] = method_c4(data, seed=seed,
                               per_method_test_idx=per_method_test_idx,
                               methods_unique=methods_unique)

    return results


def print_comparison(base_results, instruct_results, methods_unique):
    """Compare base vs Instruct results."""
    print("\n" + "=" * 78)
    print(" SAME-DISTRIBUTION COMPARISON: Base vs Instruct")
    print("=" * 78)
    print(f"\n  {'Method':<15} | {'Base AUROC':>11} | {'Inst AUROC':>11} | {'Δ':>6}")
    print(f"  {'-'*15}-+-{'-'*11}-+-{'-'*11}-+-{'-'*6}")
    for m in ["HPS", "HPS-Euclidean", "C4"]:
        b = base_results[m]["auroc"]
        i = instruct_results[m]["auroc"]
        print(f"  {m:<15} | {b:>11.4f} | {i:>11.4f} | {b-i:>+6.4f}")

    print(f"\n  {'Method':<15} | {'Base TPR':>9} | {'Inst TPR':>9} | {'Δ':>6}")
    print(f"  {'-'*15}-+-{'-'*9}-+-{'-'*9}-+-{'-'*6}")
    for m in ["HPS", "HPS-Euclidean", "C4"]:
        b = base_results[m]["tpr"]
        i = instruct_results[m]["tpr"]
        print(f"  {m:<15} | {b:>9.4f} | {i:>9.4f} | {b-i:>+6.4f}")

    print("\n" + "=" * 78)
    print(" PER-ATTACK TPR COMPARISON (the alignment-mediation test)")
    print("=" * 78)
    for m in ["HPS", "HPS-Euclidean", "C4"]:
        print(f"\n  Method: {m}")
        print(f"  {'Attack':<15} | {'Base TPR':>9} (n) | {'Inst TPR':>9} (n) | {'Δ':>6}")
        print(f"  {'-'*15}-+-{'-'*15}-+-{'-'*15}-+-{'-'*6}")
        for atk in methods_unique:
            b_data = base_results[m]["per_method"].get(atk, {"tpr": -1, "n": 0})
            i_data = instruct_results[m]["per_method"].get(atk, {"tpr": -1, "n": 0})
            if b_data["n"] > 0 and i_data["n"] > 0:
                delta = b_data["tpr"] - i_data["tpr"]
                print(f"  {atk:<15} | {b_data['tpr']:>7.3f} ({b_data['n']:>3}) | "
                      f"{i_data['tpr']:>7.3f} ({i_data['n']:>3}) | {delta:>+6.3f}")

    print("\n" + "=" * 78)
    print(" CRITICAL TEST: HPS GCG performance (the Vicuna confound check)")
    print("=" * 78)
    for m in ["HPS", "HPS-Euclidean", "C4"]:
        b_gcg = base_results[m]["per_method"].get("gcg", {"tpr": -1, "n": 0})
        i_gcg = instruct_results[m]["per_method"].get("gcg", {"tpr": -1, "n": 0})
        if b_gcg["n"] > 0 and i_gcg["n"] > 0:
            print(f"\n  {m} GCG:")
            print(f"    Llama-3-Instruct: TPR = {i_gcg['tpr']:.3f} (n={i_gcg['n']})")
            print(f"    Llama-3-base:     TPR = {b_gcg['tpr']:.3f} (n={b_gcg['n']})")
            print(f"    Δ (base-Inst):    {b_gcg['tpr']-i_gcg['tpr']:+.3f}")

    # Verdict
    print("\n" + "=" * 78)
    print(" VERDICT")
    print("=" * 78)
    hps_gcg_b = base_results["HPS"]["per_method"].get("gcg", {}).get("tpr", -1)
    hps_gcg_i = instruct_results["HPS"]["per_method"].get("gcg", {}).get("tpr", -1)
    c4_gcg_b = base_results["C4"]["per_method"].get("gcg", {}).get("tpr", -1)

    if hps_gcg_b > 0 and hps_gcg_i > 0:
        if hps_gcg_b < hps_gcg_i - 0.20:
            print(f"\n  HPS GCG drops by {hps_gcg_i - hps_gcg_b:.3f} from Instruct to base.")
            if c4_gcg_b > hps_gcg_b + 0.20:
                print(f"  C4 still works on base (TPR={c4_gcg_b:.3f}).")
                print(f"\n  -> ALIGNMENT-MEDIATION HYPOTHESIS SUPPORTED.")
                print(f"     HPS depends on alignment-shaped activation structure.")
                print(f"     The Vicuna failure is consistent with this mechanism.")
            else:
                print(f"  C4 also drops on base (TPR={c4_gcg_b:.3f}).")
                print(f"\n  -> Signal genuinely weakens without alignment.")
                print(f"     Both methods affected; not a method-specific finding.")
        else:
            print(f"\n  HPS GCG is comparable on base ({hps_gcg_b:.3f}) and Instruct ({hps_gcg_i:.3f}).")
            print(f"\n  -> ALIGNMENT-MEDIATION HYPOTHESIS NOT SUPPORTED.")
            print(f"     HPS works without alignment training.")
            print(f"     The Vicuna failure must have other confounds (model size, layers, data).")


# -------------------------- Main --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true",
                        help="Extract base Llama-3 activations (~2-3 hours)")
    parser.add_argument("--compare", action="store_true",
                        help="Run methods on base+instruct and compare")
    parser.add_argument("--test-attacks", default="llama3_attacks.json")
    parser.add_argument("--harmless", default="data_harmless_6500.csv")
    args = parser.parse_args()

    if not args.extract and not args.compare:
        parser.error("Specify --extract and/or --compare")

    if args.extract:
        print("\n" + "=" * 78)
        print(" STEP 1: Extract base Llama-3-8B activations")
        print("=" * 78)
        if os.path.exists(BASE_CACHE):
            print(f"\n  Cache already exists at {BASE_CACHE}")
            print(f"  Skipping extraction. Delete the file to re-extract.")
        else:
            dataset = load_dataset(args.test_attacks, args.harmless)
            print(f"\n  Train: {len(dataset['train_ben'])} ben + {len(dataset['train_atk'])} atk")
            print(f"  Test:  {len(dataset['test_ben'])} ben + {len(dataset['test_atk'])} atk")
            extract_and_cache(BASE_MODEL, dataset, BASE_CACHE)

    if args.compare:
        print("\n" + "=" * 78)
        print(" STEP 2: Compare base vs Instruct")
        print("=" * 78)

        # Load both caches
        base_data = load_cached(BASE_CACHE)
        instruct_data = load_cached(INSTRUCT_CACHE)
        if base_data is None:
            print(f"\nERROR: base cache not found at {BASE_CACHE}")
            print("Run with --extract first.")
            return
        if instruct_data is None:
            print(f"\nERROR: instruct cache not found at {INSTRUCT_CACHE}")
            print("Run hps_llama3.py first to generate the instruct cache.")
            return

        # Get attack methods from instruct cache (they should match)
        if instruct_data["test_methods"] is None:
            print("\nERROR: instruct cache doesn't have test_atk_methods.")
            print("This is expected if the cache was created before alignment ablation work.")
            print("Either re-run hps_llama3.py (which now saves methods) or supply methods manually.")
            # Try to reconstruct from llama3_attacks.json
            print("\nAttempting to reconstruct test_methods from llama3_attacks.json...")
            with open(args.test_attacks) as f:
                categorized = json.load(f)
            attack_prompts = []
            attack_methods_full = []
            for method, prompts in categorized.items():
                for p in prompts:
                    if p:
                        attack_prompts.append(p)
                        attack_methods_full.append(method)
            rng = np.random.RandomState(42)
            atk_idx = rng.permutation(len(attack_methods_full))
            n_train = int(0.8 * len(atk_idx))
            test_idx = atk_idx[n_train:]
            instruct_data["test_methods"] = [attack_methods_full[i] for i in test_idx]
            print(f"  Reconstructed {len(instruct_data['test_methods'])} test methods.")

        if base_data["test_methods"] is None:
            base_data["test_methods"] = instruct_data["test_methods"]

        methods_unique = sorted(set(instruct_data["test_methods"]))
        print(f"\n  Attack methods: {methods_unique}")
        print(f"  Base shapes:     X_tr_ben={base_data['X_tr_ben'].shape}, "
              f"X_te_atk={base_data['X_te_atk'].shape}")
        print(f"  Instruct shapes: X_tr_ben={instruct_data['X_tr_ben'].shape}, "
              f"X_te_atk={instruct_data['X_te_atk'].shape}")

        # Evaluate both
        print("\n" + "-" * 78)
        print(" Evaluating Llama-3-8B base...")
        print("-" * 78)
        base_results = evaluate_model("base", base_data, methods_unique, seed=42)

        print("\n" + "-" * 78)
        print(" Evaluating Llama-3-8B Instruct...")
        print("-" * 78)
        instruct_results = evaluate_model("instruct", instruct_data, methods_unique, seed=42)

        # Compare
        print_comparison(base_results, instruct_results, methods_unique)

        # Save
        out = {
            "base_model": BASE_MODEL,
            "instruct_model": INSTRUCT_MODEL,
            "layers": HPS_LAYERS,
            "kappa": KAPPA_INIT,
            "epochs": EPOCHS,
            "base_results": {k: {kk: vv for kk, vv in v.items() if kk != "threshold"}
                             for k, v in base_results.items()},
            "instruct_results": {k: {kk: vv for kk, vv in v.items() if kk != "threshold"}
                                  for k, v in instruct_results.items()},
        }
        out_path = "results/strengthen_exp5_alignment_ablation.json"
        os.makedirs("results", exist_ok=True)

        def _np_default(o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            raise TypeError(f"Type {type(o)} not serializable")

        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, default=_np_default)
        print(f"\n  Saved -> {out_path}")


if __name__ == "__main__":
    main()
