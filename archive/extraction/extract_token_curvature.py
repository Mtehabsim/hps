#!/usr/bin/env python3
"""
WS6 — All-token (vs last-token) curvature diagnosis.

Tests the hypothesis: the last-token summary is flat, but the *token-level* manifold
(esp. embedding / early layers) may be more hyperbolic (vocabulary/semantic hierarchy).

Loads the model, forwards prompts (harmful + benign), collects SAMPLED token-level
hidden states at every layer (incl. embedding layer 0), and computes relative Gromov
delta-hyperbolicity per layer for:
  - all-token manifold (pooled across prompts)
  - last-token only (for comparison with WS1)
each vs the dimension-matched random baseline (so "low" is judged correctly).

Usage:
  python extract_token_curvature.py --model_path $MP \
      --harmful_csv datasets/harmful_dataset/harmful_train_no_spec_tokens.csv \
      --benign_csv  datasets/harmful_dataset/benign_train_no_spec_tokens.csv \
      --n_prompts 120 --tokens_per_prompt 40 --output results/ws6_token_curvature
Run from obfuscated-activations/inference_time_experiments (so the dataset paths resolve),
or pass absolute CSV paths.
"""
import argparse, json, os, sys
import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", os.environ.get("MPLCONFIGDIR", "/tmp/mpl"))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# reuse the delta-hyperbolicity estimator + baseline logic
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from analyze_curvature import delta_rel
except Exception:
    # fallback: repo root
    sys.path.insert(0, os.getcwd())
    from analyze_curvature import delta_rel


def read_prompts(csv_path, n, text_col=None):
    import pandas as pd
    df = pd.read_csv(csv_path)
    if text_col is None:
        for c in ["prompt", "text", "behavior", "query", "goal"]:
            if c in df.columns:
                text_col = c; break
        if text_col is None:
            text_col = df.columns[0]
    print(f"[ws6] {csv_path}: cols={list(df.columns)} -> using '{text_col}'", flush=True)
    vals = df[text_col].dropna().astype(str).tolist()
    return vals[:n]


@torch.no_grad()
def collect_token_acts(model, tok, prompts, tokens_per_prompt, device):
    """Return dict: layer -> list of token vectors (sampled), plus last-token vectors."""
    rng = np.random.default_rng(0)
    per_layer_all, per_layer_last = None, None
    for p in prompts:
        msg = [{"role": "user", "content": p}]
        ids = tok.apply_chat_template(msg, return_tensors="pt", add_generation_prompt=True).to(device)
        out = model(ids, output_hidden_states=True)
        hs = out.hidden_states  # tuple (n_layers+1) of [1, seq, hidden]
        nL = len(hs)
        if per_layer_all is None:
            per_layer_all = [[] for _ in range(nL)]
            per_layer_last = [[] for _ in range(nL)]
        seq = hs[0].shape[1]
        # sample token positions (exclude none; include last)
        k = min(tokens_per_prompt, seq)
        pos = rng.choice(seq, k, replace=False)
        for L in range(nL):
            h = hs[L][0].float().cpu().numpy()      # [seq, hidden]
            per_layer_all[L].append(h[pos])
            per_layer_last[L].append(h[-1])
    all_stacked = [np.concatenate(x, 0) for x in per_layer_all]      # [Ntok, hidden] per layer
    last_stacked = [np.stack(x, 0) for x in per_layer_last]          # [Nprompt, hidden] per layer
    return all_stacked, last_stacked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--harmful_csv", required=True)
    ap.add_argument("--benign_csv", required=True)
    ap.add_argument("--text_col", default=None)
    ap.add_argument("--n_prompts", type=int, default=120)
    ap.add_argument("--tokens_per_prompt", type=int, default=40)
    ap.add_argument("--n_sample", type=int, default=800)
    ap.add_argument("--output", default="results/ws6_token_curvature")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16,
                                                 device_map=device).eval()

    har = read_prompts(args.harmful_csv, args.n_prompts, args.text_col)
    ben = read_prompts(args.benign_csv, args.n_prompts, args.text_col)
    print(f"[ws6] forwarding {len(har)} harmful + {len(ben)} benign prompts...", flush=True)
    har_all, har_last = collect_token_acts(model, tok, har, args.tokens_per_prompt, device)
    ben_all, ben_last = collect_token_acts(model, tok, ben, args.tokens_per_prompt, device)
    nL = len(har_all)

    # dimension-matched random baseline (4096-D Gaussian) computed once
    rngg = np.random.default_rng(1)
    base = delta_rel(rngg.standard_normal((args.n_sample, har_all[0].shape[1])),
                     n_sample=args.n_sample, standardize=True)["delta_rel_p999"]
    print(f"[ws6] dimension-matched random baseline delta_rel = {base:.3f}", flush=True)

    out = {"baseline": float(base), "layers": {}}
    for L in range(nL):
        rec = {
            "alltok_atk": delta_rel(har_all[L], args.n_sample)["delta_rel_p999"],
            "alltok_ben": delta_rel(ben_all[L], args.n_sample)["delta_rel_p999"],
            "lasttok_atk": delta_rel(har_last[L], args.n_sample)["delta_rel_p999"],
            "lasttok_ben": delta_rel(ben_last[L], args.n_sample)["delta_rel_p999"],
        }
        out["layers"][str(L)] = rec
        print(f"[ws6] layer {L:2d}  all-tok(atk/ben)={rec['alltok_atk']:.3f}/{rec['alltok_ben']:.3f}  "
              f"last-tok(atk/ben)={rec['lasttok_atk']:.3f}/{rec['lasttok_ben']:.3f}  base={base:.3f}", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump(out, open(args.output + ".json", "w"), indent=2)
    Ls = list(range(nL))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(Ls, [out["layers"][str(L)]["alltok_atk"] for L in Ls], "o-", c="tab:red", label="all-token (harmful)")
    ax.plot(Ls, [out["layers"][str(L)]["alltok_ben"] for L in Ls], "o-", c="tab:green", label="all-token (benign)")
    ax.plot(Ls, [out["layers"][str(L)]["lasttok_atk"] for L in Ls], "s--", c="darkred", alpha=0.6, label="last-token (harmful)")
    ax.axhline(base, ls=":", c="black", label=f"random 4096-D baseline ({base:.3f})")
    ax.set_xlabel("layer (0 = embedding)"); ax.set_ylabel("relative δ-hyperbolicity")
    ax.set_title("All-token vs last-token curvature (below baseline = genuinely hyperbolic)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(args.output + ".png", dpi=140)
    print(f"[ws6] wrote {args.output}.json/.png", flush=True)


if __name__ == "__main__":
    main()
