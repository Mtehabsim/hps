"""
Find refusal-sensitive layers for RTV on Vicuna-13B.
Method: For each layer, compute the separation between harmful and harmless
cosine similarities with the refusal direction. The layers with maximum
separation are the best candidates.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import pandas as pd
import argparse

from rtv_standalone import load_model, extract_hidden_states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lmsys/vicuna-13b-v1.5")
    parser.add_argument("--harmless", required=True)
    parser.add_argument("--harmful", required=True)
    parser.add_argument("--n", type=int, default=30, help="Samples per class")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)

    # Determine total layers
    if hasattr(model.config, "num_hidden_layers"):
        n_layers = model.config.num_hidden_layers
    else:
        n_layers = 40
    all_layers = list(range(n_layers))

    # Load data
    df_harmless = pd.read_csv(args.harmless)
    df_harmful = pd.read_csv(args.harmful)
    for col in ["prompt", "goal", "text", "instruction", "query"]:
        if col in df_harmless.columns:
            harmless = df_harmless[col].dropna().tolist(); break
    else:
        harmless = df_harmless.iloc[:, 0].dropna().tolist()
    for col in ["prompt", "goal", "text", "instruction", "query"]:
        if col in df_harmful.columns:
            harmful = df_harmful[col].dropna().tolist(); break
    else:
        harmful = df_harmful.iloc[:, 0].dropna().tolist()

    n = min(args.n, len(harmless), len(harmful))
    print(f"Using {n} samples per class, {n_layers} layers")

    # Extract last-token hidden states at ALL layers
    print("Extracting harmful...")
    harmful_acts = {l: [] for l in all_layers}
    for i in range(n):
        hs = extract_hidden_states(model, tokenizer, harmful[i], all_layers)
        for l in all_layers:
            if l in hs:
                harmful_acts[l].append(hs[l][-1])
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n}")

    print("Extracting harmless...")
    harmless_acts = {l: [] for l in all_layers}
    for i in range(n):
        hs = extract_hidden_states(model, tokenizer, harmless[i], all_layers)
        for l in all_layers:
            if l in hs:
                harmless_acts[l].append(hs[l][-1])
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n}")

    # For each layer: compute refusal direction, then measure separation
    print(f"\n{'Layer':>5} | {'Separation':>10} | {'Harmful cos':>11} | {'Harmless cos':>12}")
    print(f"{'─'*5}─┼─{'─'*10}─┼─{'─'*11}─┼─{'─'*12}")

    separations = []
    for l in all_layers:
        if len(harmful_acts[l]) == 0 or len(harmless_acts[l]) == 0:
            continue
        h_arr = np.array(harmful_acts[l])
        b_arr = np.array(harmless_acts[l])

        # Refusal direction at this layer
        mu_h = h_arr.mean(axis=0)
        mu_b = b_arr.mean(axis=0)
        r = mu_h - mu_b
        r_norm = r / (np.linalg.norm(r) + 1e-8)

        # Cosine similarity of each sample with r
        cos_harmful = np.array([np.dot(v, r_norm) / (np.linalg.norm(v) + 1e-8) for v in h_arr])
        cos_harmless = np.array([np.dot(v, r_norm) / (np.linalg.norm(v) + 1e-8) for v in b_arr])

        # Separation = difference in means / pooled std
        mean_diff = cos_harmful.mean() - cos_harmless.mean()
        pooled_std = np.sqrt((cos_harmful.std()**2 + cos_harmless.std()**2) / 2 + 1e-8)
        sep = mean_diff / pooled_std

        separations.append((l, sep, cos_harmful.mean(), cos_harmless.mean()))
        print(f"{l:>5} | {sep:>10.3f} | {cos_harmful.mean():>11.4f} | {cos_harmless.mean():>12.4f}")

    # Rank
    separations.sort(key=lambda x: -x[1])
    print(f"\nTop 10 layers by separation:")
    for l, sep, ch, cb in separations[:10]:
        print(f"  Layer {l:>2}: separation={sep:.3f}  (harmful_cos={ch:.4f}, harmless_cos={cb:.4f})")

    # Recommend 3 layers: best from early/mid/late thirds
    third = n_layers // 3
    best_early = max([x for x in separations if x[0] < third], key=lambda x: x[1], default=None)
    best_mid = max([x for x in separations if third <= x[0] < 2*third], key=lambda x: x[1], default=None)
    best_late = max([x for x in separations if x[0] >= 2*third], key=lambda x: x[1], default=None)

    recommended = []
    if best_early: recommended.append(best_early[0])
    if best_mid: recommended.append(best_mid[0])
    if best_late: recommended.append(best_late[0])

    print(f"\nRecommended RTV layers for {args.model}: {recommended}")
    print(f"(Best from each third: early={best_early}, mid={best_mid}, late={best_late})")


if __name__ == "__main__":
    main()
