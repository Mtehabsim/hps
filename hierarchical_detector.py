#!/usr/bin/env python3
"""
Stage 1: hierarchical-hyperbolic detector vs Euclidean control vs C4 — with
multi-seed/embed_dim robustness sweep and diagnostic visualizations.

Identical architecture for H and E (Linear encoder + per-class prototypes + distance
softmax over 14 harm leaves + benign); ONLY the distance (Lorentz vs Euclidean) differs.
This is the rpf-vs-rpfnox0 control at the detector level. C4 = linear LR on raw reps.

Reports (mean±std over seeds×embed_dims):
  - typed_acc : 14-way harm-leaf classification (the hierarchical task)
  - binary    : harm-vs-benign AUROC + TPR@1%FPR
Viz (one representative run): score distributions, embedding-radius distributions, 2D scatter.

Usage:
  python hierarchical_detector.py --harmful_npz results/harm_taxonomy_llm_reps.npz \
    --benign_reps_npz results/hier_detector_benign.npz \
    --embed_dims 8 16 32 --seeds 5 --epochs 300 --output results/hier_detector
"""
import argparse, json, os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def expmap0(v, clip=12.0):
    n = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    vn = n.clamp(max=clip)
    return torch.cat([torch.cosh(vn), torch.sinh(vn) * (v / n)], dim=-1)

def lorentz_dist(x, y):
    inner = -x[:, None, 0] * y[None, :, 0] + (x[:, None, 1:] * y[None, :, 1:]).sum(-1)
    return torch.acosh((-inner).clamp_min(1 + 1e-6))


class ProtoNet(torch.nn.Module):
    def __init__(self, d_in, d_emb, n_class, hyperbolic):
        super().__init__()
        self.enc = torch.nn.Linear(d_in, d_emb)
        self.proto = torch.nn.Parameter(torch.randn(n_class, d_emb) * 0.1)
        self.hyp = hyperbolic
    def encode(self, x): return self.enc(x)
    def logits(self, x):
        e = self.enc(x)
        if self.hyp: return -lorentz_dist(expmap0(e), expmap0(self.proto))
        return -torch.cdist(e, self.proto)
    def proto_dist(self):
        if self.hyp:
            P = expmap0(self.proto); return lorentz_dist(P, P)
        return torch.cdist(self.proto, self.proto)


def tpr_at_fpr(y, s, fpr=0.01):
    neg = np.sort(s[y == 0]); thr = neg[max(0, int(np.ceil((1 - fpr) * len(neg))) - 1)]
    return float((s[y == 1] > thr).mean())


def extract_benign(csv, model_path, layer, n, seed=0):
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    df = pd.read_csv(csv); col = "prompt" if "prompt" in df.columns else df.columns[0]
    prompts = df[col].dropna().astype(str).tolist()
    rng = np.random.default_rng(seed)
    if len(prompts) > n: prompts = [prompts[i] for i in rng.choice(len(prompts), n, replace=False)]
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="cuda").eval()
    reps = []
    with torch.no_grad():
        for p in prompts:
            ids = tok.apply_chat_template([{"role": "user", "content": p}], return_tensors="pt",
                                          add_generation_prompt=True).to("cuda")
            reps.append(model(ids, output_hidden_states=True).hidden_states[layer][0, -1].float().cpu().numpy())
    return np.array(reps)


def train_eval(Xtr, ytr, Xte, n_class, d_emb, hyperbolic, tree_D, epochs, beta, seed):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = ProtoNet(Xtr.shape[1], d_emb, n_class, hyperbolic).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-4)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev); ytr_t = torch.tensor(ytr, device=dev)
    tD = torch.tensor(tree_D, dtype=torch.float32, device=dev); ce = torch.nn.CrossEntropyLoss()
    nH = tD.shape[0]
    for _ in range(epochs):
        opt.zero_grad(); loss = ce(net.logits(Xtr_t), ytr_t)
        if beta > 0:
            pd_ = net.proto_dist()[:nH, :nH]; m = tD > 0
            loss = loss + beta * ((pd_[m] / pd_[m].mean() - tD[m] / tD[m].mean()) ** 2).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
    net.eval()
    with torch.no_grad():
        Xte_t = torch.tensor(Xte, dtype=torch.float32, device=dev)
        return net.logits(Xte_t).cpu().numpy(), net.encode(Xte_t).cpu().numpy()


def run_once(Xh, yh, Xb, nL, tree_D, d_emb, epochs, beta, seed):
    rng = np.random.default_rng(seed); benign_cls = nL
    def split(n): idx = rng.permutation(n); c = int(0.7 * n); return idx[:c], idx[c:]
    htr, hte = split(len(Xh)); btr, bte = split(len(Xb))
    Xtr = np.concatenate([Xh[htr], Xb[btr]], 0)
    ytr = np.concatenate([yh[htr], np.full(len(btr), benign_cls)])
    Xte = np.concatenate([Xh[hte], Xb[bte]], 0); nHte = len(hte)
    yte_h = yh[hte]; y_bin = np.concatenate([np.ones(nHte), np.zeros(len(bte))])
    res, viz = {}, {}
    for geo, hyp in [("hyperbolic", True), ("euclidean", False)]:
        lg, enc = train_eval(Xtr, ytr, Xte, nL + 1, d_emb, hyp, tree_D, epochs, beta, seed)
        hs = lg[:, :nL].max(1) - lg[:, benign_cls]
        res[geo] = {"typed_acc": float((lg[:nHte, :nL].argmax(1) == yte_h).mean()),
                    "binary_auroc": float(roc_auc_score(y_bin, hs)),
                    "binary_tpr": tpr_at_fpr(y_bin, hs)}
        viz[geo] = {"score": hs, "radius": np.linalg.norm(enc, axis=1), "enc": enc}
    clf = LogisticRegression(max_iter=2000).fit(Xtr, (ytr != benign_cls).astype(int))
    s = clf.decision_function(Xte)
    res["c4_linear"] = {"typed_acc": None, "binary_auroc": float(roc_auc_score(y_bin, s)), "binary_tpr": tpr_at_fpr(y_bin, s)}
    return res, viz, y_bin, nHte


def make_viz(viz, y_bin, nL, output):
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    for r, geo in enumerate(["hyperbolic", "euclidean"]):
        sc, rad, enc = viz[geo]["score"], viz[geo]["radius"], viz[geo]["enc"]
        hm, bn = y_bin == 1, y_bin == 0
        ax[r, 0].hist(sc[bn], 30, alpha=0.6, label="benign", color="tab:blue", density=True)
        ax[r, 0].hist(sc[hm], 30, alpha=0.6, label="harmful", color="tab:red", density=True)
        ax[r, 0].set_title(f"{geo}: harm-score dist"); ax[r, 0].legend(fontsize=7)
        ax[r, 1].hist(rad[bn], 30, alpha=0.6, label="benign", color="tab:blue", density=True)
        ax[r, 1].hist(rad[hm], 30, alpha=0.6, label="harmful", color="tab:red", density=True)
        ax[r, 1].set_title(f"{geo}: embedding radius ‖enc‖"); ax[r, 1].legend(fontsize=7)
        Z = enc - enc.mean(0); U, S, Vt = np.linalg.svd(Z, full_matrices=False); P = Z @ Vt[:2].T
        ax[r, 2].scatter(P[bn, 0], P[bn, 1], s=6, alpha=0.5, label="benign", color="tab:blue")
        ax[r, 2].scatter(P[hm, 0], P[hm, 1], s=6, alpha=0.5, label="harmful", color="tab:red")
        ax[r, 2].set_title(f"{geo}: 2D PCA of embeddings"); ax[r, 2].legend(fontsize=7)
    fig.suptitle("H vs E diagnostics: does benign separate from harmful (by score / radius / layout)?")
    fig.tight_layout(); fig.savefig(output + "_viz.png", dpi=140)
    print(f"[hier] wrote {output}_viz.png", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harmful_npz", required=True)
    ap.add_argument("--benign_reps_npz"); ap.add_argument("--benign_csv"); ap.add_argument("--model_path")
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--embed_dim", type=int, default=16); ap.add_argument("--embed_dims", type=int, nargs="*")
    ap.add_argument("--seeds", type=int, default=1); ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--beta", type=float, default=1.0); ap.add_argument("--output", default="results/hier_detector")
    args = ap.parse_args()
    dims = args.embed_dims if args.embed_dims else [args.embed_dim]

    d = np.load(args.harmful_npz, allow_pickle=True)
    Xh, labels = d["reps"].astype(np.float32), d["labels"]
    leaves = sorted(set(labels.tolist())); leaf_ix = {l: i for i, l in enumerate(leaves)}
    yh = np.array([leaf_ix[l] for l in labels]); nL = len(leaves)
    if args.benign_reps_npz and os.path.exists(args.benign_reps_npz):
        Xb = np.load(args.benign_reps_npz)["reps"].astype(np.float32)
    else:
        Xb = extract_benign(args.benign_csv, args.model_path, args.layer, len(Xh)); np.savez(args.output + "_benign.npz", reps=Xb)
    parse = lambda s: tuple(s.split("/"))
    tree_D = np.array([[0 if a == b else (2 if parse(a)[0] == parse(b)[0] else 4) for b in leaves] for a in leaves], float)
    allX = np.concatenate([Xh, Xb], 0); mu, sd = allX.mean(0), allX.std(0) + 1e-6
    Xh = (Xh - mu) / sd; Xb = (Xb - mu) / sd

    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list)); viz0 = None
    for D in dims:
        for s in range(args.seeds):
            res, viz, y_bin, _ = run_once(Xh, yh, Xb, nL, tree_D, D, args.epochs, args.beta, s)
            for model in res:
                for k, v in res[model].items():
                    if v is not None: agg[model][k].append(v)
            if viz0 is None: viz0 = (viz, y_bin)
    n_runs = len(dims) * args.seeds
    print(f"\n[hier] layer {args.layer}, embed_dims {dims} x {args.seeds} seeds = {n_runs} runs, {nL} harm leaves + benign", flush=True)
    print(f"{'model':12s}  typed_acc       binary_AUROC     binary_TPR@1%FPR", flush=True)
    out = {}
    for model in ["hyperbolic", "euclidean", "c4_linear"]:
        m = agg[model]; out[model] = {k: [float(np.mean(v)), float(np.std(v))] for k, v in m.items()}
        ta = "   n/a      " if "typed_acc" not in m else f"{np.mean(m['typed_acc']):.3f}±{np.std(m['typed_acc']):.3f}"
        print(f"{model:12s}  {ta}   {np.mean(m['binary_auroc']):.3f}±{np.std(m['binary_auroc']):.3f}   {np.mean(m['binary_tpr']):.3f}±{np.std(m['binary_tpr']):.3f}", flush=True)
    h, e = agg["hyperbolic"], agg["euclidean"]
    print(f"\n[hier] controlled H vs E (mean Δ over {n_runs} runs):", flush=True)
    print(f"  typed_acc:  Δ(H−E) = {np.mean(h['typed_acc'])-np.mean(e['typed_acc']):+.3f}", flush=True)
    print(f"  binary TPR: Δ(H−E) = {np.mean(h['binary_tpr'])-np.mean(e['binary_tpr']):+.3f}  (and vs C4: H−C4={np.mean(h['binary_tpr'])-np.mean(agg['c4_linear']['binary_tpr']):+.3f})", flush=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump(out, open(args.output + ".json", "w"), indent=2)
    if viz0: make_viz(viz0[0], viz0[1], nL, args.output)
    print(f"[hier] wrote {args.output}.json", flush=True)


if __name__ == "__main__":
    main()
