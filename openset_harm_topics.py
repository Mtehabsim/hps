#!/usr/bin/env python3
"""
openset_detection.py — does retrieval-based detection beat a linear probe on UNSEEN attack types?

Context: the dissociation work showed curvature helps RANKING (mAP) but NOT the in-distribution
binary decision (top-1 tied; the harm-vs-benign axis is linearly separable, so a linear probe C4 is
already at ceiling). The ONE place retrieval could still beat a parametric probe is OPEN-SET / novel
attacks: a non-parametric "is this near any KNOWN-harm cluster?" can generalize to attack categories
it never trained on, whereas a linear probe fit on known categories may fail on a new one.

Protocol — leave-one-category-out (LOCO):
  For each harm category H (held out):
    GALLERY (train) = benign reps + harm reps from all categories EXCEPT H
    TEST            = held-out-category H harm reps  (UNSEEN attack type)  +  fresh benign reps
    Detectors (all trained/built ONLY on the gallery; H is never seen):
      - knn_hyperbolic : score = (mean dist to k nearest BENIGN) - (mean dist to k nearest HARM)
                         in the curved (Lorentz, expmap0) geometry on a learned encoder
      - knn_euclidean  : same, Euclidean distance (the geometry control)
      - c4_linear      : logistic regression harm-vs-benign on raw reps (the parametric baseline)
    Metric: AUROC + TPR@1%FPR for flagging the UNSEEN harm category vs benign.

If knn_hyperbolic >= knn_euclidean > c4_linear on the HELD-OUT category, retrieval detection (and
curved geometry) genuinely helps novel-attack detection — a real detection win the negative didn't
cover. If all tie / C4 wins, retrieval-as-detection adds nothing and the negative extends to open-set.

Reuses extract_benign + geometry primitives. GPU only for the one-off benign extraction; the LOCO
loop is CPU-light. CPU --selftest validates the pipeline on synthetic data.

Usage:
  python openset_detection.py \
    --harmful_npz results/harm_taxonomy_deep_llm_reps.npz \
    --benign_reps_npz results/hier_detector_benign.npz \
    --benign_csv obfuscated-activations/inference_time_experiments/datasets/harmful_dataset/benign_train_no_spec_tokens.csv \
    --model_path $MP --layer 24 --d_emb 32 --k 10 --seeds 5 \
    --output results/openset_detection
"""
import argparse, json, os, sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
from hyperbolic_retrieval import expmap0_c, lorentz_dist_c, ProtoNet
from hierarchical_detector import tpr_at_fpr


def encode_and_train(Xtr, ytr, d_emb, hyp, c, epochs, seed):
    """Train a ProtoNet leaf-classifier on the GALLERY to get a geometry-aware encoder, then return
    the encoder so we can embed test points. We DON'T use the held-out category at all (open-set)."""
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = ProtoNet(Xtr.shape[1], d_emb, int(ytr.max()) + 1, hyp, c).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-4)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    ytr_t = torch.tensor(ytr, device=dev)
    ce = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad(); ce(net.logits(Xtr_t), ytr_t).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
    net.eval()
    return net


def embed(net, X):
    dev = next(net.parameters()).device
    with torch.no_grad():
        return net.encode(torch.tensor(X, dtype=torch.float32, device=dev)).cpu().numpy()


def knn_harm_score(E_query, E_harm_gallery, E_benign_gallery, k, hyp, c):
    """Harm score = (mean dist to k nearest BENIGN) - (mean dist to k nearest HARM gallery point).
    High => closer to harmful cluster than benign => flag harmful. Geometry set by `hyp`."""
    def dmat(A, B):
        if hyp:
            PA = expmap0_c(torch.tensor(A, dtype=torch.float32), c)
            PB = expmap0_c(torch.tensor(B, dtype=torch.float32), c)
            return lorentz_dist_c(PA, PB, c).numpy()
        return np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
    Dh = np.sort(dmat(E_query, E_harm_gallery), axis=1)[:, :k].mean(1)
    Db = np.sort(dmat(E_query, E_benign_gallery), axis=1)[:, :k].mean(1)
    return Db - Dh                      # higher = more harmful


def run_loco(Xh, cat, Xb, d_emb, c, k, epochs, seed):
    """One seed: leave-one-category-out open-set detection for every category. Returns per-category
    AUROC/TPR for knn_hyperbolic, knn_euclidean, c4_linear on the UNSEEN category vs benign."""
    rng = np.random.default_rng(seed)
    cats = np.unique(cat)
    # split benign into gallery/test halves (so test benign is fresh, not in the gallery)
    bidx = rng.permutation(len(Xb)); bcut = len(Xb) // 2
    Xb_gal, Xb_test = Xb[bidx[:bcut]], Xb[bidx[bcut:]]
    out = {}
    for held in cats:
        tr_mask = cat != held
        Xh_gal, yh_gal = Xh[tr_mask], cat[tr_mask]
        Xh_test = Xh[cat == held]               # UNSEEN attack type
        if len(Xh_test) < 3 or len(np.unique(yh_gal)) < 2:
            continue
        # standardize on the GALLERY only (benign+seen-harm); apply to everything
        gal_all = np.concatenate([Xh_gal, Xb_gal], 0)
        mu, sd = gal_all.mean(0), gal_all.std(0) + 1e-6
        Xh_gal_n, Xb_gal_n = (Xh_gal - mu) / sd, (Xb_gal - mu) / sd
        Xh_test_n, Xb_test_n = (Xh_test - mu) / sd, (Xb_test - mu) / sd
        # relabel seen-harm categories to 0..K-1 for the classifier
        cmap = {cc: i for i, cc in enumerate(np.unique(yh_gal))}
        ytr = np.array([cmap[v] for v in yh_gal])
        y_test = np.concatenate([np.ones(len(Xh_test_n)), np.zeros(len(Xb_test_n))])

        res = {}
        for geo, hyp in [("knn_hyperbolic", True), ("knn_euclidean", False)]:
            net = encode_and_train(Xh_gal_n, ytr, d_emb, hyp, c, epochs, seed)
            Eh_gal = embed(net, Xh_gal_n); Eb_gal = embed(net, Xb_gal_n)
            Eq = embed(net, np.concatenate([Xh_test_n, Xb_test_n], 0))
            s = knn_harm_score(Eq, Eh_gal, Eb_gal, k, hyp, c)
            res[geo] = {"auroc": float(roc_auc_score(y_test, s)), "tpr": tpr_at_fpr(y_test, s)}
        # C4 linear: harm(any seen category)=1 vs benign=0, on RAW gallery reps
        Xc = np.concatenate([Xh_gal_n, Xb_gal_n], 0)
        yc = np.concatenate([np.ones(len(Xh_gal_n)), np.zeros(len(Xb_gal_n))])
        clf = LogisticRegression(max_iter=2000).fit(Xc, yc)
        sc = clf.decision_function(np.concatenate([Xh_test_n, Xb_test_n], 0))
        res["c4_linear"] = {"auroc": float(roc_auc_score(y_test, sc)), "tpr": tpr_at_fpr(y_test, sc)}
        out[str(held)] = res
    return out


def make_synthetic(seed=0, d_in=128):
    rng = np.random.default_rng(seed)
    ncat, per = 5, 50
    centers = rng.standard_normal((ncat, d_in)) * 4
    Xh = np.concatenate([centers[i] + rng.standard_normal((per, d_in)) for i in range(ncat)]).astype("float32")
    cat = np.repeat(np.arange(ncat), per)
    Xb = (rng.standard_normal((200, d_in)) * 4 - 8).astype("float32")  # benign offset cluster
    return Xh, cat, Xb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harmful_npz"); ap.add_argument("--benign_reps_npz")
    ap.add_argument("--benign_csv"); ap.add_argument("--model_path"); ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--n_benign", type=int, default=1500)
    ap.add_argument("--d_emb", type=int, default=32); ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=10); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--output", default="results/openset_detection")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        print("[openset] synthetic selftest (no GPU)")
        Xh, cat, Xb = make_synthetic()
        args.seeds = 2; args.epochs = 120
        cat_names = [f"cat{i}" for i in range(len(np.unique(cat)))]
    else:
        if not args.harmful_npz:
            ap.error("--harmful_npz required (or --selftest)")
        d = np.load(args.harmful_npz, allow_pickle=True)
        Xh = d["reps"].astype(np.float32)
        labels = [str(s) for s in d["labels"]]
        cat_strs = [l.split("/")[0] for l in labels]          # top category = the open-set unit
        cat_names = sorted(set(cat_strs))
        cix = {c: i for i, c in enumerate(cat_names)}
        cat = np.array([cix[c] for c in cat_strs])
        # benign reps: load cache or extract via the existing helper
        if args.benign_reps_npz and os.path.exists(args.benign_reps_npz):
            Xb = np.load(args.benign_reps_npz)["reps"].astype(np.float32)
        elif args.benign_csv and args.model_path:
            from hierarchical_detector import extract_benign
            Xb = extract_benign(args.benign_csv, args.model_path, args.layer, args.n_benign)
            np.savez(args.output + "_benign.npz", reps=Xb)
        else:
            ap.error("need --benign_reps_npz OR (--benign_csv and --model_path)")

    print(f"[openset] {len(Xh)} harm reps in {len(np.unique(cat))} categories + {len(Xb)} benign; "
          f"d_emb={args.d_emb} c={args.c} k={args.k} seeds={args.seeds}", flush=True)

    # aggregate over seeds, per held-out category
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # cat -> detector -> metric -> [vals]
    for s in range(args.seeds):
        r = run_loco(Xh, cat, Xb, args.d_emb, args.c, args.k, args.epochs, s)
        for held, dets in r.items():
            for det, m in dets.items():
                for mk, mv in m.items():
                    agg[held][det][mk].append(mv)

    results = {"config": vars(args), "by_heldout_category": {}, "macro": {}}
    dets = ["knn_hyperbolic", "knn_euclidean", "c4_linear"]
    print(f"\n{'held-out category':28s} {'detector':16s} {'AUROC':>14} {'TPR@1%FPR':>12}", flush=True)
    macro = {d: {"auroc": [], "tpr": []} for d in dets}
    for held in sorted(agg):
        results["by_heldout_category"][held] = {}
        name = cat_names[int(held)] if held.isdigit() else held
        for det in dets:
            au = agg[held][det]["auroc"]; tp = agg[held][det]["tpr"]
            if not au: continue
            results["by_heldout_category"][held][det] = {
                "auroc": [float(np.mean(au)), float(np.std(au))],
                "tpr": [float(np.mean(tp)), float(np.std(tp))]}
            macro[det]["auroc"].append(np.mean(au)); macro[det]["tpr"].append(np.mean(tp))
            print(f"{name:28s} {det:16s} {np.mean(au):>8.3f}±{np.std(au):.3f} {np.mean(tp):>8.3f}", flush=True)
        print("", flush=True)

    print("=== MACRO (mean over held-out categories) — the open-set headline ===", flush=True)
    for det in dets:
        ma = float(np.mean(macro[det]["auroc"])); mt = float(np.mean(macro[det]["tpr"]))
        results["macro"][det] = {"auroc": ma, "tpr": mt}
        print(f"  {det:16s} AUROC={ma:.3f}  TPR@1%FPR={mt:.3f}", flush=True)
    h, e, c4 = results["macro"]["knn_hyperbolic"], results["macro"]["knn_euclidean"], results["macro"]["c4_linear"]
    print(f"\n  Δ AUROC: knn_hyp - knn_euc = {h['auroc']-e['auroc']:+.3f}   "
          f"knn_hyp - c4 = {h['auroc']-c4['auroc']:+.3f}", flush=True)
    verdict = ("OPEN-SET WIN: retrieval (esp. hyperbolic) beats the linear probe on unseen attacks"
               if h["auroc"] > c4["auroc"] + 0.01 else
               "no open-set win: retrieval ties/loses to the linear probe -> negative extends to open-set")
    print(f"  VERDICT: {verdict}", flush=True)
    results["verdict"] = verdict

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump(results, open(args.output + ".json", "w"), indent=2)
    print(f"\n[openset] wrote {args.output}.json", flush=True)


if __name__ == "__main__":
    main()
