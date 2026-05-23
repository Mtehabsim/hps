"""
HPS on Llama-3-8B — Same-distribution and cross-attack evaluation.
Uses JBShield attack data for direct comparison with RTV.

Usage:
  python hps_llama3.py --test-attacks llama3_attacks.json
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from rtv_standalone import load_model, extract_hidden_states, FPR_TARGET
from experiment7 import LorentzProjection, contrastive_loss, extract_trajectory_features

# Llama-3-8B: 32 layers. Fisher-selected layers TBD; use early+late heuristic
HPS_LAYERS = [0, 1, 2, 28, 29, 30, 31]  # early + late for 32-layer model


def extract_hps(model, tokenizer, prompts, label=""):
    acts = []
    for i, p in enumerate(prompts):
        hs = extract_hidden_states(model, tokenizer, p, HPS_LAYERS)
        acts.append(np.array([hs[l][-1] for l in HPS_LAYERS]))
        if (i+1) % 50 == 0:
            print(f"    {label}: {i+1}/{len(prompts)}")
    return np.array(acts)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--test-attacks", required=True)
    parser.add_argument("--benign-csv", default="data_harmless_100.csv")
    parser.add_argument("--n-benign", type=int, default=100)
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"  HPS on Llama-3-8B")
    print(f"  Layers: {HPS_LAYERS}")
    print(f"{'═'*60}\n")

    # Load attacks
    with open(args.test_attacks) as f:
        categorized = json.load(f)
    attack_prompts, attack_methods = [], []
    for method, prompts in categorized.items():
        for p in prompts:
            if p:
                attack_prompts.append(p)
                attack_methods.append(method)
    print(f"  Attacks: {len(attack_prompts)} across {len(set(attack_methods))} methods")

    # Load benign
    import pandas as pd
    df = pd.read_csv(args.benign_csv)
    for col in ["prompt", "goal", "text", "instruction", "query"]:
        if col in df.columns:
            benign_prompts = df[col].dropna().tolist(); break
    else:
        benign_prompts = df.iloc[:, 0].dropna().tolist()
    benign_prompts = benign_prompts[:args.n_benign]
    print(f"  Benign: {len(benign_prompts)}")

    # 80/20 split
    rng = np.random.RandomState(42)
    ben_idx = rng.permutation(len(benign_prompts))
    atk_idx = rng.permutation(len(attack_prompts))
    n_ben_tr = int(0.8 * len(ben_idx))
    n_atk_tr = int(0.8 * len(atk_idx))

    train_ben = [benign_prompts[i] for i in ben_idx[:n_ben_tr]]
    test_ben = [benign_prompts[i] for i in ben_idx[n_ben_tr:]]
    train_atk = [attack_prompts[i] for i in atk_idx[:n_atk_tr]]
    test_atk = [attack_prompts[i] for i in atk_idx[n_atk_tr:]]
    test_methods = [attack_methods[i] for i in atk_idx[n_atk_tr:]]

    print(f"  Train: {len(train_ben)} benign + {len(train_atk)} attacks")
    print(f"  Test: {len(test_ben)} benign + {len(test_atk)} attacks")

    # Load model
    model, tokenizer = load_model(args.model)

    # Extract activations
    print(f"\n  Extracting activations...")
    X_tr_ben = extract_hps(model, tokenizer, train_ben, "train benign")
    X_tr_atk = extract_hps(model, tokenizer, train_atk, "train attacks")
    X_te_ben = extract_hps(model, tokenizer, test_ben, "test benign")
    X_te_atk = extract_hps(model, tokenizer, test_atk, "test attacks")

    X_train = np.concatenate([X_tr_ben, X_tr_atk])
    y_train = np.array([0]*len(X_tr_ben) + [1]*len(X_tr_atk))
    n_layers = len(HPS_LAYERS)
    d_hidden = X_train.shape[2]

    # Train HPS
    print(f"\n  Training Lorentz projection (d={d_hidden}, layers={n_layers})...")
    torch.manual_seed(42)
    proj = LorentzProjection(d_hidden, 64, 1.0, n_layers=n_layers).to("cuda")
    opt = optim.Adam(proj.parameters(), lr=1e-3, weight_decay=1e-5)
    X_t = torch.tensor(X_train, dtype=torch.float32, device="cuda")
    y_t = torch.tensor(y_train, dtype=torch.long, device="cuda")

    for epoch in range(120):
        loss = torch.tensor(0.0, device="cuda")
        for l in range(n_layers):
            h = proj(X_t[:, l, :])
            loss = loss + contrastive_loss(h, y_t, k=proj.k, tau=proj.tau(l))
        loss = loss / n_layers
        opt.zero_grad(); loss.backward(); opt.step()
        if (epoch+1) % 40 == 0:
            print(f"    Epoch {epoch+1}/120 loss={loss.item():.4f} k={proj.k.item():.3f}")
    proj.eval()

    # Features
    feats_train = extract_trajectory_features(proj, X_train)
    feats_te_ben = extract_trajectory_features(proj, X_te_ben)
    feats_te_atk = extract_trajectory_features(proj, X_te_atk)

    # Classifier
    sc = StandardScaler()
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(sc.fit_transform(feats_train), y_train)

    scores_ben = clf.predict_proba(sc.transform(feats_te_ben))[:, 1]
    scores_atk = clf.predict_proba(sc.transform(feats_te_atk))[:, 1]
    threshold = float(np.quantile(scores_ben, 1.0 - FPR_TARGET))

    tpr = float((scores_atk > threshold).mean())
    auroc = roc_auc_score(
        np.array([0]*len(scores_ben) + [1]*len(scores_atk)),
        np.concatenate([scores_ben, scores_atk])
    )

    # Results
    print(f"\n{'─'*60}")
    print(f"  HPS RESULTS (Llama-3-8B, same-distribution)")
    print(f"{'─'*60}")
    print(f"  AUROC: {auroc:.3f}")
    print(f"  TPR@5%FPR: {tpr:.3f}")

    print(f"\n  Per-attack-type:")
    print(f"  {'Method':<15} | {'N':>5} | {'TPR':>6}")
    print(f"  {'─'*15}─┼─{'─'*5}─┼─{'─'*6}")
    methods_unique = sorted(set(test_methods))
    per_method = {}
    for m in methods_unique:
        idx = [i for i, x in enumerate(test_methods) if x == m]
        mt = float((scores_atk[idx] > threshold).mean())
        print(f"  {m:<15} | {len(idx):>5} | {mt:>6.3f}")
        per_method[m] = {"n": len(idx), "tpr": mt}

    # Comparison
    print(f"\n{'─'*60}")
    print(f"  COMPARISON (Llama-3-8B)")
    print(f"{'─'*60}")
    print(f"  {'Method':<20} | {'AUROC':>6} | {'TPR@5%':>7}")
    print(f"  {'─'*20}─┼─{'─'*6}─┼─{'─'*7}")
    print(f"  {'RTV (our impl)':<20} | {'0.893':>6} | {'0.683':>7}")
    print(f"  {'HPS (this run)':<20} | {auroc:>6.3f} | {tpr:>7.3f}")

    # Save
    results = {
        "model": args.model, "layers": HPS_LAYERS,
        "auroc": float(auroc), "tpr": float(tpr),
        "per_method": per_method,
        "n_train_ben": len(train_ben), "n_train_atk": len(train_atk),
        "n_test_ben": len(test_ben), "n_test_atk": len(test_atk),
    }
    out = "results/hps_llama3_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {out}")
    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    main()
