#!/usr/bin/env python3
"""
Curvature sweep: does ANY hyperbolic curvature help the harm detector?

Uses a curvature-c Lorentz model where the distance smoothly interpolates:
   c -> 0   == Euclidean   (the control, for free)
   c = 1    == our standard hyperbolic
   c large  == strongly hyperbolic
So sweeping c tests every geometry between Euclidean and strongly-hyperbolic. If typed/
binary performance peaks at c->0 (Euclidean) and doesn't improve for c>0, hyperbolic
curvature provides no benefit -- robust to "we picked a bad curvature/projection".

Same prototype detector as hierarchical_detector (14 harm leaves + benign), beta=0
(pure classification, to isolate curvature). C4 linear baseline shown for reference.

Usage:
  python curvature_sweep.py --harmful_npz results/harm_taxonomy_llm_reps.npz \
    --benign_reps_npz results/hier_detector_benign.npz \
    --curvatures 0.05 0.25 0.5 1 2 4 --embed_dim 16 --seeds 5 --output results/curv_sweep
"""
import argparse, json, os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def expmap0_c(v, c, clip=12.0):
    u = (c ** 0.5) * v
    n = u.norm(dim=-1, keepdim=True).clamp_min(1e-8); un = n.clamp(max=clip)
    return torch.cat([torch.cosh(un), torch.sinh(un) * (u / n)], dim=-1)

def lorentz_dist_c(x, y, c):
    inner = -x[:, None, 0] * y[None, :, 0] + (x[:, None, 1:] * y[None, :, 1:]).sum(-1)
    return torch.acosh((-inner).clamp_min(1 + 1e-6)) / (c ** 0.5)


class ProtoNetC(torch.nn.Module):
    def __init__(self, d_in, d_emb, n_class, c):
        super().__init__()
        self.enc = torch.nn.Linear(d_in, d_emb)
        self.proto = torch.nn.Parameter(torch.randn(n_class, d_emb) * 0.1)
        self.c = c
    def logits(self, x):
        e = self.enc(x)
        return -lorentz_dist_c(expmap0_c(e, self.c), expmap0_c(self.proto, self.c), self.c)


def tpr_at_fpr(y, s, fpr=0.01):
    neg = np.sort(s[y == 0]); thr = neg[max(0, int(np.ceil((1 - fpr) * len(neg))) - 1)]
    return float((s[y == 1] > thr).mean())


def run(Xh, yh, Xb, nL, c, d_emb, epochs, seed):
    rng = np.random.default_rng(seed); bcls = nL
    def sp(n): idx = rng.permutation(n); k = int(0.7 * n); return idx[:k], idx[k:]
    htr, hte = sp(len(Xh)); btr, bte = sp(len(Xb))
    Xtr = np.concatenate([Xh[htr], Xb[btr]], 0); ytr = np.concatenate([yh[htr], np.full(len(btr), bcls)])
    Xte = np.concatenate([Xh[hte], Xb[bte]], 0); nHte = len(hte)
    yte_h = yh[hte]; y_bin = np.concatenate([np.ones(nHte), np.zeros(len(bte))])
    torch.manual_seed(seed); dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = ProtoNetC(Xtr.shape[1], d_emb, nL + 1, c).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-4)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev); ytr_t = torch.tensor(ytr, device=dev)
    ce = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad(); ce(net.logits(Xtr_t), ytr_t).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
    net.eval()
    with torch.no_grad():
        lg = net.logits(torch.tensor(Xte, dtype=torch.float32, device=dev)).cpu().numpy()
    hs = lg[:, :nL].max(1) - lg[:, bcls]
    return float((lg[:nHte, :nL].argmax(1) == yte_h).mean()), tpr_at_fpr(y_bin, hs), float(roc_auc_score(y_bin, hs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harmful_npz", required=True); ap.add_argument("--benign_reps_npz", required=True)
    ap.add_argument("--curvatures", type=float, nargs="+", default=[0.05, 0.25, 0.5, 1, 2, 4])
    ap.add_argument("--embed_dim", type=int, default=16); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--output", default="results/curv_sweep")
    args = ap.parse_args()
    d = np.load(args.harmful_npz, allow_pickle=True)
    Xh, labels = d["reps"].astype(np.float32), d["labels"]
    leaves = sorted(set(labels.tolist())); ix = {l: i for i, l in enumerate(leaves)}
    yh = np.array([ix[l] for l in labels]); nL = len(leaves)
    Xb = np.load(args.benign_reps_npz)["reps"].astype(np.float32)
    allX = np.concatenate([Xh, Xb], 0); mu, sd = allX.mean(0), allX.std(0) + 1e-6
    Xh = (Xh - mu) / sd; Xb = (Xb - mu) / sd

    # C4 baseline (binary), seed-averaged
    c4t, c4a = [], []
    rng = np.random.default_rng(0)
    for s in range(args.seeds):
        r = np.random.default_rng(s); 
        hi = r.permutation(len(Xh)); bi = r.permutation(len(Xb)); kh = int(0.7*len(Xh)); kb=int(0.7*len(Xb))
        Xtr = np.concatenate([Xh[hi[:kh]], Xb[bi[:kb]]]); ytr = np.concatenate([np.ones(kh), np.zeros(kb)])
        Xte = np.concatenate([Xh[hi[kh:]], Xb[bi[kb:]]]); yte = np.concatenate([np.ones(len(Xh)-kh), np.zeros(len(Xb)-kb)])
        clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr); ss = clf.decision_function(Xte)
        c4t.append(tpr_at_fpr(yte, ss)); c4a.append(roc_auc_score(yte, ss))

    out = {"curvatures": args.curvatures, "typed": {}, "binary_tpr": {}, "binary_auroc": {},
           "c4_binary_tpr": [float(np.mean(c4t)), float(np.std(c4t))]}
    print(f"\n[curv] sweep (c→0 = Euclidean). embed_dim {args.embed_dim}, {args.seeds} seeds, {nL} leaves+benign", flush=True)
    print(f"{'curvature c':12s}  typed_acc       binary_TPR@1%FPR   binary_AUROC", flush=True)
    for c in args.curvatures:
        T, B, A = [], [], []
        for s in range(args.seeds):
            t, b, a = run(Xh, yh, Xb, nL, c, args.embed_dim, args.epochs, s); T.append(t); B.append(b); A.append(a)
        out["typed"][str(c)] = [float(np.mean(T)), float(np.std(T))]
        out["binary_tpr"][str(c)] = [float(np.mean(B)), float(np.std(B))]
        out["binary_auroc"][str(c)] = [float(np.mean(A)), float(np.std(A))]
        tag = " (≈Euclidean)" if c <= 0.06 else (" (standard hyp)" if abs(c-1) < 1e-6 else "")
        print(f"{c:<12.3g}  {np.mean(T):.3f}±{np.std(T):.3f}   {np.mean(B):.3f}±{np.std(B):.3f}      {np.mean(A):.3f}{tag}", flush=True)
    print(f"{'C4 (linear)':12s}      n/a         {np.mean(c4t):.3f}±{np.std(c4t):.3f}      {np.mean(c4a):.3f}", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump(out, open(args.output + ".json", "w"), indent=2)
    cs = args.curvatures
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].errorbar(cs, [out["typed"][str(c)][0] for c in cs], yerr=[out["typed"][str(c)][1] for c in cs], fmt="o-")
    ax[0].set_xscale("log"); ax[0].set_xlabel("curvature c (log; →0 = Euclidean)"); ax[0].set_ylabel("typed_acc"); ax[0].set_title("Typed accuracy vs curvature")
    ax[1].errorbar(cs, [out["binary_tpr"][str(c)][0] for c in cs], yerr=[out["binary_tpr"][str(c)][1] for c in cs], fmt="o-", label="hyperbolic-c")
    ax[1].axhline(np.mean(c4t), ls="--", c="k", label=f"C4 linear ({np.mean(c4t):.3f})")
    ax[1].set_xscale("log"); ax[1].set_xlabel("curvature c (log; →0 = Euclidean)"); ax[1].set_ylabel("binary TPR@1%FPR"); ax[1].set_title("Binary TPR vs curvature"); ax[1].legend(fontsize=8)
    fig.suptitle("Does any curvature help? (peak at c→0 = Euclidean wins; flat/down = hyperbolic no help)")
    fig.tight_layout(); fig.savefig(args.output + ".png", dpi=140)
    print(f"[curv] wrote {args.output}.json/.png", flush=True)


if __name__ == "__main__":
    main()
