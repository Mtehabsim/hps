#!/usr/bin/env python3
"""
Per-prompt, per-layer delta-hyperbolicity (DESCRIPTIVE).

For each prompt we compute delta-hyperbolicity over ITS OWN tokens at each layer,
so we can see whether any SPECIFIC prompt is tree-like (delta below the matched
random baseline) and at which layers. No classifier, no confound regression --
just the raw per-prompt geometry vs the baseline.

Outputs:
  - <output>.npz : full [n_prompts, n_layers] delta matrices (harmful, benign) + token counts
  - <output>.json: per-prompt summary (prompt text, min delta + layer, #layers below baseline)
  - <output>.png : heatmap (prompts x layers), prompts sorted by min delta, baseline contour

Usage (GPU):
  python per_prompt_hyperbolicity.py --model_path $MP \
    --harmful_csv .../harmful_train_no_spec_tokens.csv \
    --benign_csv  .../benign_train_no_spec_tokens.csv \
    --K 24 --n_prompts 200 --output results/perprompt_hyp
"""
import argparse, json, os, sys
import numpy as np, torch
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
from analyze_curvature import delta_rel


def read_prompts(csv, n, col=None):
    import pandas as pd
    df = pd.read_csv(csv)
    if col is None:
        col = next((c for c in ["prompt","text","behavior","query","goal"] if c in df.columns), df.columns[0])
    return df[col].dropna().astype(str).tolist()[:n], col


@torch.no_grad()
def per_prompt_delta(model, tok, prompts, K, device):
    rng = np.random.default_rng(0)
    deltas, ntok, kept = [], [], []
    for p in prompts:
        ids = tok.apply_chat_template([{"role":"user","content":p}], return_tensors="pt",
                                      add_generation_prompt=True).to(device)
        seq = ids.shape[1]
        if seq < K:
            continue
        hs = model(ids, output_hidden_states=True).hidden_states  # tuple nL of [1,seq,h]
        pos = rng.choice(seq, K, replace=False)
        row = [delta_rel(hs[L][0, pos].float().cpu().numpy(), n_sample=K, n_quad=50000)["delta_rel_p999"]
               for L in range(len(hs))]
        deltas.append(row); ntok.append(int(seq)); kept.append(p)
    return np.array(deltas), np.array(ntok), kept


def baseline_K(K, D, seed=1):
    rng = np.random.default_rng(seed)
    return float(delta_rel(rng.standard_normal((K, D)), n_sample=K, n_quad=50000)["delta_rel_p999"])


def summarize(deltas, ntok, prompts, base):
    out = []
    for i in range(len(deltas)):
        row = deltas[i]
        out.append({
            "prompt": prompts[i][:160],
            "ntok": int(ntok[i]),
            "min_delta": float(row.min()),
            "min_delta_layer": int(row.argmin()),
            "n_layers_below_baseline": int((row < base).sum()),
            "delta_per_layer": [round(float(x), 4) for x in row],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--harmful_csv", required=True); ap.add_argument("--benign_csv", required=True)
    ap.add_argument("--col", default=None); ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--n_prompts", type=int, default=200); ap.add_argument("--output", default="results/perprompt_hyp")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map=device).eval()
    har, col = read_prompts(args.harmful_csv, args.n_prompts, args.col)
    ben, _ = read_prompts(args.benign_csv, args.n_prompts, args.col)
    print(f"[pp] col='{col}', K={args.K}; forwarding...", flush=True)
    dh, nh, ph = per_prompt_delta(model, tok, har, args.K, device)
    db, nb, pb = per_prompt_delta(model, tok, ben, args.K, device)
    D = model.config.hidden_size
    base = baseline_K(args.K, D)
    base_lo = baseline_K(args.K, D, seed=7)  # second seed, to show baseline noise
    print(f"[pp] usable prompts: harmful={len(dh)} benign={len(db)} (need >= K tokens)", flush=True)
    print(f"[pp] matched random baseline (K={args.K} points): delta_rel ~ {base:.3f} (2nd seed {base_lo:.3f})", flush=True)
    print(f"[pp]   -> a prompt-layer is 'tree-like' only if its delta is clearly BELOW ~{base:.3f}", flush=True)

    har_s = summarize(dh, nh, ph, base); ben_s = summarize(db, nb, pb, base)
    np.savez(args.output + ".npz", harmful=dh, benign=db, ntok_harmful=nh, ntok_benign=nb,
             baseline=base, K=args.K)
    json.dump({"K": args.K, "baseline": base, "baseline_seed2": base_lo,
               "harmful": har_s, "benign": ben_s}, open(args.output + ".json", "w"), indent=2)

    # how many prompts are EVER tree-like (any layer below baseline), and how far below
    def report(tag, S):
        ever = sum(1 for r in S if r["n_layers_below_baseline"] > 0)
        mins = sorted(S, key=lambda r: r["min_delta"])
        print(f"\n[pp] === {tag}: {len(S)} prompts ===", flush=True)
        print(f"[pp] prompts with >=1 layer below baseline ({base:.3f}): {ever}/{len(S)}", flush=True)
        print(f"[pp] global min delta = {mins[0]['min_delta']:.3f} (layer {mins[0]['min_delta_layer']})", flush=True)
        print(f"[pp] 5 most tree-like prompts (lowest min-delta across layers):", flush=True)
        for r in mins[:5]:
            print(f"      min_delta={r['min_delta']:.3f} @L{r['min_delta_layer']:2d} "
                  f"({r['n_layers_below_baseline']} layers<base, ntok={r['ntok']})  {r['prompt'][:80]!r}", flush=True)
    report("HARMFUL", har_s); report("BENIGN", ben_s)

    # heatmap: prompts (sorted by min delta) x layers
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=False)
    for ax, M, tag in [(axes[0], dh, "harmful"), (axes[1], db, "benign")]:
        order = np.argsort(M.min(axis=1))
        im = ax.imshow(M[order], aspect="auto", cmap="viridis", vmin=0, vmax=max(0.2, float(M.max())))
        ax.set_title(f"{tag}: per-prompt δ (baseline≈{base:.3f})"); ax.set_xlabel("layer"); ax.set_ylabel("prompt (sorted by min δ)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(args.output + ".png", dpi=140)
    print(f"\n[pp] wrote {args.output}.npz/.json/.png", flush=True)


if __name__ == "__main__":
    main()
