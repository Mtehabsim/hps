#!/usr/bin/env python3
"""
hyperbolic_retrieval.py — the POSITIVE experiment (sibling to the negative paper).

Question: does hyperbolic geometry help HIERARCHICAL harm-taxonomy modeling at LOW
dimension — i.e. retrieval / parent-accuracy / zero-shot novel-leaf placement / tree-
distance fidelity — even though it does NOT help flat binary jailbreak detection?

This deliberately leaves the linearly-separable binary task (where C4 wins and no geometry
can help) and tests distance/containment tasks on the harm tree, on the d in {2,3,4,...}
frontier the negative paper never swept. See positive_experiment_prereg.md.

Design (the fair, isometry-controlled contrast — H vs E differ ONLY in the distance metric):
  - reuse ProtoNet / expmap0 / lorentz_dist from hierarchical_detector.py
  - reuse fit_all (Sala h-MDS distortion) from embedding_distortion.py
  - train a leaf classifier; evaluate RETRIEVAL on the learned embedding space
  - FIX the train+test standardization leak (fit mu/sd on train rows only)

Tasks / metrics (harm reps only; benign has no place in the harm tree):
  - mAP@all, recall@k        : same-leaf / same-category retrieval
  - 1-NN leaf acc, parent acc: geometry-faithful nearest-neighbour
  - tree-distance distortion : embedding pairwise dists vs taxonomy tree dists (0/2/4)
  - novel-leaf placement     : hold out a whole leaf; does it land near a sibling leaf?

Usage (GPU, real data — after extracting reps with harm_taxonomy.py extract --layer 24):
  python hyperbolic_retrieval.py \
    --harmful_npz results/harm_taxonomy_llm_reps.npz \
    --dims 2 3 4 6 8 16 32 --seeds 5 --epochs 300 \
    --output results/hyperbolic_retrieval

  # curvature sweep at the best dim (does retrieval peak at c>0 = genuine hyperbolic?):
  python hyperbolic_retrieval.py --harmful_npz results/harm_taxonomy_llm_reps.npz \
    --dims 3 --curvatures 0.05 0.25 0.5 1 2 4 --seeds 5 --output results/hyp_retr_curv

Usage (CPU, no GPU, validate the pipeline on synthetic tree-structured data):
  python hyperbolic_retrieval.py --selftest
"""
import argparse, json, os, sys
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())

# ---------------------------------------------------------------------------
# Geometry primitives — reuse the repo's own, with a curvature-c generalization
# (matches hierarchical_detector.expmap0 at c=1 and curvature_sweep.expmap0_c).
# ---------------------------------------------------------------------------
import torch


def expmap0_c(v, c=1.0, clip=12.0):
    """Exp-map at the origin onto the curvature-c Lorentz hyperboloid. c->0 ~ Euclidean."""
    u = (c ** 0.5) * v
    n = u.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    un = n.clamp(max=clip)
    return torch.cat([torch.cosh(un), torch.sinh(un) * (u / n)], dim=-1)


def lorentz_dist_c(x, y, c=1.0):
    """Pairwise Lorentz geodesic distance, curvature c. x:(n,1+d) y:(m,1+d) -> (n,m)."""
    inner = -x[:, None, 0] * y[None, :, 0] + (x[:, None, 1:] * y[None, :, 1:]).sum(-1)
    return torch.acosh((-inner).clamp_min(1 + 1e-6)) / (c ** 0.5)


class ProtoNet(torch.nn.Module):
    """Linear encoder + per-leaf prototypes; Hyperbolic (Lorentz) or Euclidean read-out.

    Identical architecture across geometries — only `hyp` (and curvature c) differ.
    This is the isometry-controlled contrast at the detector level.
    """
    def __init__(self, d_in, d_emb, n_class, hyp, c=1.0):
        super().__init__()
        self.enc = torch.nn.Linear(d_in, d_emb)
        self.proto = torch.nn.Parameter(torch.randn(n_class, d_emb) * 0.1)
        self.hyp, self.c = hyp, c

    def encode(self, x):
        return self.enc(x)

    def logits(self, x):
        e = self.enc(x)
        if self.hyp:
            return -lorentz_dist_c(expmap0_c(e, self.c), expmap0_c(self.proto, self.c), self.c)
        return -torch.cdist(e, self.proto)

    def emb_dists(self, x):
        """Pairwise distances between encoded points, in the model's geometry."""
        e = self.enc(x)
        if self.hyp:
            P = expmap0_c(e, self.c)
            return lorentz_dist_c(P, P, self.c)
        return torch.cdist(e, e)


# ---------------------------------------------------------------------------
# Retrieval / hierarchy metrics (numpy; operate on a precomputed distance matrix)
# ---------------------------------------------------------------------------
def _ap(rel_sorted):
    """Average precision given a boolean array of relevance ordered by ascending distance."""
    if rel_sorted.sum() == 0:
        return 0.0
    hits = np.cumsum(rel_sorted)
    ranks = np.arange(1, len(rel_sorted) + 1)
    return float((hits / ranks * rel_sorted).sum() / rel_sorted.sum())


def retrieval_metrics(D, leaf, parent, ks=(1, 5, 10), return_pq=False):
    """D: (n,n) query x gallery distances (same set; self excluded). leaf/parent: (n,) int labels.
    If return_pq, also return per-query average-precision arrays (length n, NaN where undefined) so
    a paired bootstrap / permutation test can run on the curved-vs-flat gap (stats hardening, #3)."""
    n = D.shape[0]
    Dq = D.copy()
    np.fill_diagonal(Dq, np.inf)              # exclude self
    order = np.argsort(Dq, axis=1)            # nearest first
    ap_leaf, ap_parent = [], []
    rec = {k: [] for k in ks}
    nn_leaf, nn_parent = [], []
    pq_parent = np.full(n, np.nan); pq_leaf = np.full(n, np.nan)
    for i in range(n):
        o = order[i]
        rel_leaf = (leaf[o] == leaf[i])
        rel_par = (parent[o] == parent[i])
        # AP only defined when at least one relevant gallery item exists
        if rel_leaf.sum() > 0:
            a = _ap(rel_leaf); ap_leaf.append(a); pq_leaf[i] = a
            for k in ks:
                rec[k].append(float(rel_leaf[:k].any()))
        if rel_par.sum() > 0:
            a = _ap(rel_par); ap_parent.append(a); pq_parent[i] = a
        nn_leaf.append(float(leaf[o[0]] == leaf[i]))
        nn_parent.append(float(parent[o[0]] == parent[i]))
    out = {
        "mAP_leaf": float(np.mean(ap_leaf)) if ap_leaf else 0.0,
        "mAP_parent": float(np.mean(ap_parent)) if ap_parent else 0.0,
        "nn_leaf_acc": float(np.mean(nn_leaf)),
        "nn_parent_acc": float(np.mean(nn_parent)),
        **{f"recall@{k}": float(np.mean(rec[k])) if rec[k] else 0.0 for k in ks},
    }
    if return_pq:
        out["_pq_parent"], out["_pq_leaf"] = pq_parent, pq_leaf
    return out


def paired_bootstrap(diff, n_boot=10000, seed=0):
    """Paired bootstrap over per-query gap values `diff` (NaNs dropped). Returns mean, 95% CI,
    and a two-sided p-value (fraction of resampled means that cross zero, x2)."""
    d = np.asarray(diff, float); d = d[~np.isnan(d)]
    if len(d) < 2:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "p_value": float("nan"), "n": int(len(d))}
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    p = 2.0 * min(float((means <= 0).mean()), float((means >= 0).mean()))
    return {"mean": float(d.mean()), "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5)), "p_value": min(p, 1.0), "n": int(len(d))}


def _pairwise(metric, Xtr, Xte):
    """Distance matrix among TEST points under a model-free metric. Some metrics are fit on
    TRAIN (whitening covariance, radial scale) and applied to TEST — no leakage into labels."""
    Xte = Xte.astype(np.float64)
    if metric == "l2":
        return np.linalg.norm(Xte[:, None] - Xte[None, :], axis=-1)
    if metric == "cosine":
        Xn = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-9)
        return 1.0 - Xn @ Xn.T
    if metric == "whitened":
        # Mahalanobis with covariance estimated on TRAIN; strips per-dim scale + correlation,
        # the lever behind high-d L2 concentration. Diagonal-shrinkage for stability.
        Xc = Xtr.astype(np.float64) - Xtr.mean(0)
        cov = (Xc.T @ Xc) / max(len(Xtr) - 1, 1)
        cov += 1e-3 * np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0]
        try:
            L = np.linalg.cholesky(np.linalg.inv(cov))
        except np.linalg.LinAlgError:
            L = np.diag(1.0 / (np.sqrt(np.diag(cov)) + 1e-9))
        Y = (Xte - Xtr.mean(0)) @ L
        return np.linalg.norm(Y[:, None] - Y[None, :], axis=-1)
    if metric == "radial_reweight":
        # ABLATION for "is it really curvature, or just radius recalibration?":
        # split each rep into magnitude r and direction u; rescale magnitude by a fixed
        # log-compression (the dominant effect of an exp-map at the origin), then plain L2.
        # If THIS reproduces the hyperbolic gain, "curvature" collapses to distance recalibration.
        r = np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-9
        u = Xte / r
        r2 = np.log1p(r)                          # monotone radial compression (no learned curvature)
        Y = u * r2
        return np.linalg.norm(Y[:, None] - Y[None, :], axis=-1)
    raise ValueError(metric)


def baseline_metrics(Xtr, Xte, yte, par_te, metrics):
    """Model-free retrieval baselines on raw test reps. These adjudicate the #1 reviewer
    objection: does HYPERBOLIC win, or does plain-L2 Euclidean merely DEGRADE in high d?
    cosine + whitened are concentration-immune; if H still beats them the curvature claim
    is earned, if not the gap was high-d L2 concentration."""
    out = {}
    for mname in metrics:
        D = _pairwise(mname, Xtr, Xte)
        out[mname] = retrieval_metrics(D, yte, par_te)
    return out


def tree_distortion(D_emb, leaf, tree_D_leaf):
    """Distortion of embedding pairwise distances vs taxonomy tree distances.

    Aggregate points to leaf centroids in distance space (median pairwise) then compare to
    the 0/2/4 tree-distance matrix via the repo's _distortion (scale-free, |a-b|/b)."""
    from embedding_distortion import _distortion
    leaves = np.unique(leaf)
    K = len(leaves)
    Dc = np.zeros((K, K))
    for ai, a in enumerate(leaves):
        for bi, b in enumerate(leaves):
            if ai == bi:
                continue
            sub = D_emb[np.ix_(leaf == a, leaf == b)]
            Dc[ai, bi] = np.median(sub) if sub.size else 0.0
    Dc = Dc / (np.median(Dc[Dc > 0]) + 1e-9)
    Dt = tree_D_leaf[np.ix_(leaves, leaves)].astype(float)
    return _distortion(Dc, Dt)


# ---------------------------------------------------------------------------
# Train + evaluate one (geometry, dim, curvature, seed) configuration
# ---------------------------------------------------------------------------
def train_embed(Xtr, ytr, Xte, n_class, d_emb, hyp, c, epochs, beta, tree_D, seed):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = ProtoNet(Xtr.shape[1], d_emb, n_class, hyp, c).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-4)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    ytr_t = torch.tensor(ytr, device=dev)
    ce = torch.nn.CrossEntropyLoss()
    tD = torch.tensor(tree_D, dtype=torch.float32, device=dev) if beta > 0 else None
    for _ in range(epochs):
        opt.zero_grad()
        loss = ce(net.logits(Xtr_t), ytr_t)
        if beta > 0:                                  # optional tree regularizer on prototypes
            e = net.proto
            P = expmap0_c(e, c) if hyp else None
            pd_ = lorentz_dist_c(P, P, c) if hyp else torch.cdist(e, e)
            m = tD > 0
            loss = loss + beta * ((pd_[m] / pd_[m].mean() - tD[m] / tD[m].mean()) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
    net.eval()
    with torch.no_grad():
        Xte_t = torch.tensor(Xte, dtype=torch.float32, device=dev)
        D_emb = net.emb_dists(Xte_t).cpu().numpy()
        typed = net.logits(Xte_t).cpu().numpy()
    return D_emb, typed


def run_config(Xh, leaf, parent, tree_D, d_emb, c, epochs, beta, seed, baselines=(), want_pq=False):
    """One split. Returns retrieval metrics for H and E (+ typed acc, tree distortion) and,
    optionally, model-free baselines (l2/cosine/whitened/radial_reweight) on the SAME test reps.
    If want_pq, also returns per-query AP for H and trained-E so the paired bootstrap (#3) can run
    on the FAIR gap (curved minus identically-trained Euclidean), the correct baseline."""
    rng = np.random.default_rng(seed)
    n = len(Xh)
    idx = rng.permutation(n); cut = int(0.7 * n)
    tr, te = idx[:cut], idx[cut:]
    # standardization fit on TRAIN ONLY (fixes the leak in the older scripts)
    mu, sd = Xh[tr].mean(0), Xh[tr].std(0) + 1e-6
    Xtr, Xte = (Xh[tr] - mu) / sd, (Xh[te] - mu) / sd
    ytr, yte = leaf[tr], leaf[te]
    par_te = parent[te]
    n_class = int(leaf.max()) + 1
    out = {}; pq = {}
    for geo, hyp in [("hyperbolic", True), ("euclidean", False)]:
        D_emb, typed = train_embed(Xtr, ytr, Xte, n_class, d_emb, hyp, c, epochs, beta, tree_D, seed)
        m = retrieval_metrics(D_emb, yte, par_te, return_pq=want_pq)
        m["typed_acc"] = float((typed.argmax(1) == yte).mean())
        m["tree_distortion"] = tree_distortion(D_emb, yte, tree_D)
        if want_pq:
            pq[geo] = {"parent": m.pop("_pq_parent"), "leaf": m.pop("_pq_leaf")}
        out[geo] = m
    if baselines:
        # baselines operate on the RAW 4096-d standardized test reps (no projection/training):
        # context for the concentration objection — NOT the baseline for the curvature claim.
        for bname, bm in baseline_metrics(Xtr, Xte, yte, par_te, baselines).items():
            out[f"baseline_{bname}"] = bm
    if want_pq:
        return out, pq
    return out


def run_novel_leaf(Xh, leaf, parent, leaves_arr, d_emb, c, epochs, beta, tree_D, seed):
    """Zero-shot: hold out each placeable leaf (parent has >=2 leaves); does it land near a
    sibling? Reports per-geometry parent-placement accuracy averaged over held-out leaves."""
    rng = np.random.default_rng(seed + 1000)
    # placeable = leaves whose parent has >= 2 leaves
    par_of = {}
    for lf in np.unique(leaf):
        par_of[lf] = parent[leaf == lf][0]
    from collections import Counter
    par_count = Counter(par_of.values())
    placeable = [lf for lf in np.unique(leaf) if par_count[par_of[lf]] >= 2]
    if not placeable:
        return {"hyperbolic": float("nan"), "euclidean": float("nan"), "n_leaves": 0}
    acc = {"hyperbolic": [], "euclidean": []}
    n_class = int(leaf.max()) + 1
    for held in placeable:
        tr = np.where(leaf != held)[0]
        te = np.where(leaf == held)[0]
        mu, sd = Xh[tr].mean(0), Xh[tr].std(0) + 1e-6
        Xtr, Xte = (Xh[tr] - mu) / sd, (Xh[te] - mu) / sd
        ytr = leaf[tr]
        for geo, hyp in [("hyperbolic", True), ("euclidean", False)]:
            torch.manual_seed(seed)
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            net = ProtoNet(Xtr.shape[1], d_emb, n_class, hyp, c).to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-4)
            Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev)
            ytr_t = torch.tensor(ytr, device=dev)
            ce = torch.nn.CrossEntropyLoss()
            for _ in range(epochs):
                opt.zero_grad(); ce(net.logits(Xtr_t), ytr_t).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
            net.eval()
            with torch.no_grad():
                etr = net.encode(Xtr_t).cpu().numpy()
                ete = net.encode(torch.tensor(Xte, dtype=torch.float32, device=dev)).cpu().numpy()
            # nearest TRAIN neighbour (in the geometry) -> does it share the held leaf's parent?
            if hyp:
                Ptr = expmap0_c(torch.tensor(etr), c); Pte = expmap0_c(torch.tensor(ete), c)
                D = lorentz_dist_c(Pte, Ptr, c).numpy()
            else:
                D = np.linalg.norm(ete[:, None, :] - etr[None, :, :], axis=-1)
            nn = D.argmin(1)
            nn_parent = parent[tr][nn]
            acc[geo].append(float((nn_parent == par_of[held]).mean()))
    return {"hyperbolic": float(np.mean(acc["hyperbolic"])),
            "euclidean": float(np.mean(acc["euclidean"])),
            "n_leaves": len(placeable)}


# ---------------------------------------------------------------------------
# Data loading: reps + leaf/parent integer labels + tree-distance matrix
# ---------------------------------------------------------------------------
def _prefix_match(label, prefix):
    """True if `prefix` is a path-segment prefix of `label` (so 'cyber/intrusion' matches
    'cyber/intrusion' and 'cyber/intrusion/credential_theft' but NOT 'cyber/intrusionX')."""
    lp, pp = label.split("/"), prefix.split("/")
    return len(lp) >= len(pp) and lp[:len(pp)] == pp


def apply_label_ops(Xh, labels, drop_leaves=None, merge_groups=None):
    """Robustness re-run support (#2 follow-up): DROP unreliable leaves and/or MERGE
    systematically-confused leaf pairs flagged by label_agreement.py, then re-derive everything.
      drop_leaves : list of path-prefixes; matching prompts are removed (Xh filtered too).
      merge_groups: list of lists of path-prefixes; each group is relabeled to a single merged
                    prefix '<commonparent>/<a__b>' preserving any deeper subleaf segment.
    Returns (Xh', labels')."""
    labels = list(labels)
    keep = np.ones(len(labels), dtype=bool)
    if drop_leaves:
        for i, lb in enumerate(labels):
            if any(_prefix_match(lb, p) for p in drop_leaves):
                keep[i] = False
    if merge_groups:
        for group in merge_groups:
            # merged prefix: shared top segment + joined leaf names (sorted for determinism)
            leaf_names = sorted(p.split("/")[-1] for p in group)
            top = group[0].split("/")[0]
            merged_prefix = f"{top}/" + "__".join(leaf_names)
            for i, lb in enumerate(labels):
                for p in group:
                    if _prefix_match(lb, p):
                        rest = lb.split("/")[len(p.split("/")):]   # deeper segments (e.g. subleaf)
                        labels[i] = "/".join([merged_prefix] + rest)
                        break
    Xh = Xh[keep]; labels = [l for l, k in zip(labels, keep) if k]
    return Xh, labels


def load_reps(npz, drop_leaves=None, merge_groups=None):
    """Labels are '/'-separated paths: 2-level 'category/leaf' OR 3-level 'category/leaf/subleaf'.
    The finest segment-path is the LEAF (retrieval class); its PARENT is the path minus the last
    segment. Tree distance = 2 x (depth - length of shared prefix), so deeper shared structure =
    closer (2-level: same-parent 2 / diff 4; 3-level: same-subparent 2 / same-top 4 / diff 6).
    Optional drop_leaves / merge_groups clean the labels before deriving anything (robustness re-run)."""
    d = np.load(npz, allow_pickle=True)
    Xh = d["reps"].astype(np.float32)
    labels = [str(s) for s in d["labels"]]
    if drop_leaves or merge_groups:
        n0 = len(labels)
        Xh, labels = apply_label_ops(Xh, labels, drop_leaves, merge_groups)
        print(f"[label-ops] dropped/merged: {n0} -> {len(labels)} prompts "
              f"(drop={drop_leaves}, merge={merge_groups})", flush=True)
    leaves = sorted(set(labels))
    leaf_ix = {lf: i for i, lf in enumerate(leaves)}
    leaf = np.array([leaf_ix[l] for l in labels])
    # PARENT = label with the last path segment removed (the level just above the leaf).
    def parent_of(lbl):
        parts = lbl.split("/")
        return "/".join(parts[:-1]) if len(parts) > 1 else lbl
    parents = sorted(set(parent_of(l) for l in leaves))
    par_ix = {p: i for i, p in enumerate(parents)}
    parent = np.array([par_ix[parent_of(l)] for l in labels])
    # shared-prefix tree distance between two leaf paths
    def tdist(a, b):
        if a == b: return 0
        pa, pb = leaves[a].split("/"), leaves[b].split("/")
        shared = 0
        for x, y in zip(pa, pb):
            if x == y: shared += 1
            else: break
        depth = max(len(pa), len(pb))
        return 2 * (depth - shared)
    K = len(leaves)
    tree_D = np.array([[tdist(a, b) for b in range(K)] for a in range(K)], float)
    return Xh, leaf, parent, np.arange(K), tree_D, leaves, parents


def corrupt_labels(leaf, leaves, frac, seed):
    """LABEL-NOISE ROBUSTNESS (#1): reassign a fraction `frac` of points to a RANDOM DIFFERENT leaf,
    simulating 'what if frac of our labels were wrong all along'. Returns a new (leaf, parent) with
    parent recomputed consistently from each corrupted point's new leaf path. The reps are untouched;
    only the ground-truth labels move — so retrieval relevance now scores against partly-wrong labels.
    If the curved-minus-Euclidean gap survives high frac, mislabeling cannot explain the gap."""
    rng = np.random.default_rng(seed + 99991)
    leaf = leaf.copy()
    K = len(leaves)
    n = len(leaf)
    n_corrupt = int(round(frac * n))
    if n_corrupt > 0 and K > 1:
        idx = rng.choice(n, size=n_corrupt, replace=False)
        for i in idx:
            # pick a different leaf uniformly at random
            new = rng.integers(0, K - 1)
            if new >= leaf[i]:
                new += 1
            leaf[i] = new
    # recompute parent from the (possibly new) leaf path
    def parent_of(lbl):
        parts = lbl.split("/")
        return "/".join(parts[:-1]) if len(parts) > 1 else lbl
    parents = sorted(set(parent_of(l) for l in leaves))
    par_ix = {p: i for i, p in enumerate(parents)}
    parent = np.array([par_ix[parent_of(leaves[l])] for l in leaf])
    return leaf, parent


# Benign-topic CONTROL: group the 9 diverse-benign sources into a 2-level topic tree that
# MIRRORS the harm tree's shape (some parents multi-leaf, some single). This lets us ask the
# decisive question: is hyperbolic's curvature-driven retrieval gain HARM-specific, or does it
# show up on ANY topic hierarchy? (reconciliation_memo doubt #1). leaf = source, parent = group.
SOURCE_GROUPS = {
    "humaneval": "code", "mbpp": "code",
    "gsm8k": "math",
    "wikitext_long": "text", "wikitext_short": "text",
    "mmlu": "knowledge",
    "wildchat": "chat", "alpaca_control": "chat",
    "or_bench_hard": "safety_adjacent",
}


def build_benign_topic_json(csv_path, out_json, per_source_cap=120, seed=0):
    """CPU-only: read data_harmless_diverse.csv (prompt,source) -> write an assignments JSON in
    the SAME format harm_taxonomy.py `extract` consumes ({prompt, category=group, leaf=source}).
    Caps per source so leaves are size-balanced (the harm tree is imbalanced; we keep the topic
    control balanced to be conservative). Then run the SAME extract + experiment on its reps."""
    import csv, json as _json
    rng = np.random.default_rng(seed)
    by_src = {}
    with open(csv_path) as f:
        r = csv.reader(f); header = next(r)
        pi, si = header.index("prompt"), header.index("source")
        for row in r:
            if len(row) <= max(pi, si):
                continue
            src = row[si]
            if src not in SOURCE_GROUPS:
                continue
            by_src.setdefault(src, []).append(row[pi])
    rows = []
    for src, prompts in sorted(by_src.items()):
        if len(prompts) > per_source_cap:
            prompts = [prompts[i] for i in rng.choice(len(prompts), per_source_cap, replace=False)]
        for p in prompts:
            rows.append({"prompt": p, "category": SOURCE_GROUPS[src], "leaf": src})
    from collections import Counter
    cov = Counter((r["category"], r["leaf"]) for r in rows)
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    _json.dump({"taxonomy": "benign_topic_control", "assignments": rows}, open(out_json, "w"), indent=2)
    print(f"[benign-topic] wrote {out_json}: {len(rows)} prompts, "
          f"{len(set(SOURCE_GROUPS.values()))} groups / {len(by_src)} sources", flush=True)
    for (c, l), k in sorted(cov.items()):
        print(f"    {c:16s} / {l:16s} : {k}", flush=True)
    print(f"[benign-topic] next: python harm_taxonomy.py extract --taxonomy_json {out_json} "
          f"--model_path $MP --layer 24 --output {out_json.replace('.json','')}", flush=True)


def make_synthetic(seed=0, d_in=128, per_leaf=40):
    """Synthetic tree-structured reps for --selftest: 3 parents x 3 leaves, hierarchical means.
    Hyperbolic-friendly by construction (clear parent/leaf structure) so the pipeline can be
    validated without a GPU. NOT evidence — only checks the metric code runs and is sane."""
    rng = np.random.default_rng(seed)
    cats, leaves_per = 3, 3
    parent_means = rng.standard_normal((cats, d_in)) * 5.0
    X, leaf, parent, names = [], [], [], []
    li = 0
    for c in range(cats):
        for l in range(leaves_per):
            leaf_mean = parent_means[c] + rng.standard_normal(d_in) * 1.0
            pts = leaf_mean + rng.standard_normal((per_leaf, d_in)) * 0.5
            X.append(pts); leaf += [li] * per_leaf; parent += [c] * per_leaf
            names.append(f"cat{c}/leaf{l}"); li += 1
    X = np.concatenate(X).astype(np.float32)
    leaf = np.array(leaf); parent = np.array(parent)
    K = li
    leaves = sorted(set(range(K)))
    def tdist(a, b):
        if a == b: return 0
        return 2 if (a // leaves_per) == (b // leaves_per) else 4
    tree_D = np.array([[tdist(a, b) for b in range(K)] for a in range(K)], float)
    return X, leaf, parent, np.arange(K), tree_D, names, [f"cat{c}" for c in range(cats)]


def run_label_noise_sweep(args):
    """#1 LABEL-NOISE ROBUSTNESS: at each corruption fraction, reassign that fraction of labels to
    random different leaves, then re-run the curved-vs-trained-Euclidean retrieval + paired bootstrap.
    Reports the parent-mAP gap and its 95% CI vs noise. The gap surviving high noise => the +0.13
    effect is not an artifact of imperfect (8B-model) labels."""
    Xh, leaf0, parent0, leaves_arr, tree_D, names, cats = load_reps(
        args.harmful_npz, drop_leaves=getattr(args, "_drop_leaves", None),
        merge_groups=getattr(args, "_merge_groups", None))
    leaves = names
    baselines = ()  # not needed here; the comparator is the trained-Euclidean ProtoNet
    print(f"[noise] {len(Xh)} reps, {len(leaves)} leaves; dims={args.dims} curvatures={args.curvatures} "
          f"seeds={args.seeds}; noise levels={args.label_noise}", flush=True)
    results = {"config": vars(args), "leaves": names, "by_level": {}}
    print(f"\n{'noise':>6} {'dim':>4} {'c':>5} {'H_mAP_par':>10} {'E_mAP_par':>10} {'gap':>7} "
          f"{'95%CI':>20} {'p':>8}  verdict", flush=True)
    for frac in args.label_noise:
        # corrupt ONCE per seed (different corruption per seed for averaging), inside the seed loop
        for d_emb in args.dims:
            for c in args.curvatures:
                Hs, Es, pq_gaps = [], [], []
                for s in range(args.seeds):
                    leaf, parent = corrupt_labels(leaf0, leaves, frac, s)
                    ret = run_config(Xh, leaf, parent, tree_D, d_emb, c, args.epochs, args.beta, s,
                                     baselines, want_pq=True)
                    r, pq = ret
                    Hs.append(r["hyperbolic"]["mAP_parent"]); Es.append(r["euclidean"]["mAP_parent"])
                    pq_gaps.append(pq["hyperbolic"]["parent"] - pq["euclidean"]["parent"])
                Hm, Em = float(np.mean(Hs)), float(np.mean(Es))
                bs = paired_bootstrap(np.concatenate(pq_gaps), seed=0)
                sig = "SURVIVES" if (bs["ci_low"] > 0) else ("FLIPPED" if bs["ci_high"] < 0 else "n.s.")
                key = f"noise{frac}_d{d_emb}_c{c}"
                results["by_level"][key] = {"frac": frac, "dim": d_emb, "c": c,
                    "H_mAP_parent": Hm, "E_mAP_parent": Em, "gap": Hm - Em, "bootstrap": bs}
                print(f"{frac:>6.2f} {d_emb:>4} {c:>5.2g} {Hm:>10.3f} {Em:>10.3f} {Hm-Em:>+7.3f} "
                      f"[{bs['ci_low']:+.3f},{bs['ci_high']:+.3f}] {bs['p_value']:>8.4f}  {sig}", flush=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump(results, open(args.output + ".json", "w"), indent=2)
    print(f"\n[noise] wrote {args.output}.json", flush=True)
    # plot gap vs noise (one line per dim/c)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        from collections import defaultdict
        series = defaultdict(list)
        for v in results["by_level"].values():
            series[(v["dim"], v["c"])].append((v["frac"], v["gap"], v["bootstrap"]["ci_low"], v["bootstrap"]["ci_high"]))
        for (d_emb, c), pts in series.items():
            pts.sort()
            fr = [p[0] for p in pts]; gp = [p[1] for p in pts]
            lo = [p[1] - p[2] for p in pts]; hi = [p[3] - p[1] for p in pts]
            ax.errorbar(fr, gp, yerr=[lo, hi], fmt="o-", capsize=3, label=f"d={d_emb}, c={c}")
        ax.axhline(0, ls="--", c="k", lw=1)
        ax.set_xlabel("fraction of labels corrupted"); ax.set_ylabel("parent-mAP gap (H − trained-Euclidean)")
        ax.set_title("Label-noise robustness of the curved retrieval gain\n(gap > 0 with CI above 0 = survives)")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(args.output + ".png", dpi=140)
        print(f"[noise] wrote {args.output}.png", flush=True)
    except Exception as e:
        print(f"[noise] plot skipped: {e}", flush=True)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harmful_npz", help="reps npz from harm_taxonomy.py extract (layer 24)")
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 3, 4, 6, 8, 16, 32])
    ap.add_argument("--curvatures", type=float, nargs="+", default=[1.0],
                    help="curvature c per dim; c->0 ~ Euclidean. Use a sweep at one dim to find the peak.")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--beta", type=float, default=0.0, help="tree-regularizer weight (0 = pure CE)")
    ap.add_argument("--novel_leaf", action="store_true", help="also run zero-shot novel-leaf placement")
    ap.add_argument("--baselines", default="",
                    help="comma list of model-free retrieval baselines on raw test reps: "
                         "l2,cosine,whitened,radial_reweight. CONTEXT for the concentration objection.")
    ap.add_argument("--stats", action="store_true",
                    help="paired per-query bootstrap on the FAIR gap (curved minus trained-Euclidean "
                         "ProtoNet). The decisive significance test for the +0.13 retrieval claim (#3).")
    ap.add_argument("--output", default="results/hyperbolic_retrieval")
    ap.add_argument("--selftest", action="store_true", help="run on synthetic data, no GPU needed")
    ap.add_argument("--build_benign_topic", metavar="CSV",
                    help="CPU-only: build the benign-topic CONTROL assignments JSON from "
                         "data_harmless_diverse.csv, then exit (feed to harm_taxonomy.py extract).")
    ap.add_argument("--label_noise", type=float, nargs="+", default=None,
                    help="LABEL-NOISE ROBUSTNESS (#1): fractions to corrupt, e.g. 0 0.05 0.1 0.2 0.3. "
                         "Re-runs the H-vs-trainedE bootstrap gap at each noise level. If the gap "
                         "survives high noise, mislabeling cannot explain it. Uses --dims/--curvatures.")
    ap.add_argument("--drop_leaves", default="",
                    help="comma list of leaf path-prefixes to REMOVE (robustness re-run after the "
                         "agreement check flags unreliable leaves), e.g. "
                         "'regulatory_evasion/illegal_operation'. Matching prompts are dropped.")
    ap.add_argument("--merge_leaves", default="",
                    help="semicolon-separated groups, each a comma list of leaf prefixes to MERGE into "
                         "one class, e.g. 'cyber/intrusion,cyber/malware'. Use for systematically-"
                         "confused pairs. Multiple groups separated by ';'.")
    args = ap.parse_args()
    # parse the cleanup ops once
    drop_leaves = [s.strip() for s in args.drop_leaves.split(",") if s.strip()]
    merge_groups = [[p.strip() for p in g.split(",") if p.strip()]
                    for g in args.merge_leaves.split(";") if g.strip()]
    args._drop_leaves, args._merge_groups = drop_leaves, merge_groups

    if args.build_benign_topic:
        build_benign_topic_json(args.build_benign_topic, args.output + ".json")
        return

    if args.label_noise is not None:
        if not args.harmful_npz:
            ap.error("--harmful_npz required for --label_noise")
        run_label_noise_sweep(args)
        return

    if args.selftest:
        print("[selftest] synthetic tree-structured data (no GPU); validates the metric pipeline.")
        Xh, leaf, parent, leaves_arr, tree_D, names, cats = make_synthetic()
        args.dims = [2, 4, 16]; args.seeds = 2; args.epochs = 150; args.novel_leaf = True
    else:
        if not args.harmful_npz:
            ap.error("--harmful_npz required (or use --selftest)")
        Xh, leaf, parent, leaves_arr, tree_D, names, cats = load_reps(
            args.harmful_npz, drop_leaves=args._drop_leaves, merge_groups=args._merge_groups)

    print(f"[retr] {len(Xh)} harm reps, {len(set(leaf.tolist()))} leaves, {len(set(parent.tolist()))} parents, "
          f"d_in={Xh.shape[1]}; dims={args.dims} curvatures={args.curvatures} seeds={args.seeds}", flush=True)

    baselines = tuple(b.strip() for b in args.baselines.split(",") if b.strip())
    results = {"config": vars(args), "leaves": names, "cats": cats, "by_dim": {}}
    print(f"\n{'dim':>4} {'c':>5} {'geo':14} {'mAP_leaf':>9} {'mAP_par':>8} {'nnLeaf':>7} {'nnPar':>7} "
          f"{'typed':>6} {'treeDist':>8}", flush=True)
    for d_emb in args.dims:
        for c in args.curvatures:
            agg = {"hyperbolic": [], "euclidean": []}
            base_agg = {f"baseline_{b}": [] for b in baselines}
            pq_gap_parent, pq_gap_leaf = [], []   # pooled per-query (H - trained-E) AP gaps across seeds
            for s in range(args.seeds):
                ret = run_config(Xh, leaf, parent, tree_D, d_emb, c, args.epochs, args.beta, s,
                                 baselines, want_pq=args.stats)
                r, pq = ret if args.stats else (ret, None)
                agg["hyperbolic"].append(r["hyperbolic"]); agg["euclidean"].append(r["euclidean"])
                for bk in base_agg:
                    base_agg[bk].append(r[bk])
                if args.stats:
                    # paired per-query: same test queries scored by H and by trained-E in this split
                    pq_gap_parent.append(pq["hyperbolic"]["parent"] - pq["euclidean"]["parent"])
                    pq_gap_leaf.append(pq["hyperbolic"]["leaf"] - pq["euclidean"]["leaf"])
            key = f"d{d_emb}_c{c}"
            results["by_dim"][key] = {}
            for geo in ["hyperbolic", "euclidean"]:
                keys = agg[geo][0].keys()
                m = {k: [float(np.mean([a[k] for a in agg[geo]])),
                         float(np.std([a[k] for a in agg[geo]]))] for k in keys}
                results["by_dim"][key][geo] = m
                print(f"{d_emb:>4} {c:>5.2g} {geo:14} {m['mAP_leaf'][0]:>9.3f} {m['mAP_parent'][0]:>8.3f} "
                      f"{m['nn_leaf_acc'][0]:>7.3f} {m['nn_parent_acc'][0]:>7.3f} {m['typed_acc'][0]:>6.3f} "
                      f"{m['tree_distortion'][0]:>8.3f}", flush=True)
            # model-free baselines (raw test reps; same across geometry — depend only on d via nothing,
            # but recomputed per seed so the split/standardization matches)
            for bk, runs in base_agg.items():
                keys = runs[0].keys()
                m = {k: [float(np.mean([a[k] for a in runs])), float(np.std([a[k] for a in runs]))] for k in keys}
                results["by_dim"][key][bk] = m
                print(f"{'':>4} {'':>5} {bk:14} {m['mAP_leaf'][0]:>9.3f} {m['mAP_parent'][0]:>8.3f} "
                      f"{m['nn_leaf_acc'][0]:>7.3f} {m['nn_parent_acc'][0]:>7.3f} {'':>6} {'':>8}", flush=True)
            H, E = results["by_dim"][key]["hyperbolic"], results["by_dim"][key]["euclidean"]
            results["by_dim"][key]["delta_H_minus_E"] = {
                k: round(H[k][0] - E[k][0], 4) for k in ["mAP_leaf", "mAP_parent", "nn_parent_acc", "typed_acc", "tree_distortion"]}
            d = results["by_dim"][key]["delta_H_minus_E"]
            print(f"{'':>4} {'':>5} {'Δ(H−E)':14} {d['mAP_leaf']:>+9.3f} {d['mAP_parent']:>+8.3f} "
                  f"{'':>7} {d['nn_parent_acc']:>+7.3f} {d['typed_acc']:>+6.3f} {d['tree_distortion']:>+8.3f}  "
                  f"(treeDist: negative Δ = H more faithful)", flush=True)
            # #3 STATS: paired per-query bootstrap on the FAIR gap (curved minus trained-Euclidean).
            # This is the decider — if +0.13 is not significant per-query, the positive collapses.
            if args.stats:
                gp = np.concatenate(pq_gap_parent); gl = np.concatenate(pq_gap_leaf)
                bp = paired_bootstrap(gp, seed=0); bl = paired_bootstrap(gl, seed=0)
                results["by_dim"][key]["paired_bootstrap_H_minus_trainedE"] = {
                    "mAP_parent_gap": bp, "mAP_leaf_gap": bl}
                sig = "SIGNIFICANT (CI excludes 0)" if (bp["ci_low"] > 0 or bp["ci_high"] < 0) else "n.s. (CI spans 0)"
                print(f"{'':>4} {'':>5} {'bootstrap':14} parent-mAP gap (H−trainedE) = {bp['mean']:+.3f} "
                      f"[95% CI {bp['ci_low']:+.3f},{bp['ci_high']:+.3f}] p={bp['p_value']:.4f} n={bp['n']} -> {sig}", flush=True)
            # CONTEXT comparison: H vs the BEST model-free baseline. If H beats even the
            # concentration-immune baselines (cosine/whitened), the curvature claim is earned;
            # if a raw baseline ties/beats H, the "hyperbolic retrieval" advantage was an artifact.
            if baselines:
                best_b, best_v = None, -1.0
                for b in baselines:
                    v = results["by_dim"][key][f"baseline_{b}"]["mAP_parent"][0]
                    if v > best_v: best_b, best_v = b, v
                gap = H["mAP_parent"][0] - best_v
                verdict = "H beats best raw baseline -> curvature EARNED" if gap > 0 else \
                          "raw baseline >= H -> advantage is NOT curvature (concentration/artifact)"
                results["by_dim"][key]["H_minus_best_baseline_mAP_parent"] = {
                    "best_baseline": best_b, "best_baseline_mAP_parent": round(best_v, 4),
                    "H_mAP_parent": round(H["mAP_parent"][0], 4), "gap": round(gap, 4)}
                print(f"{'':>4} {'':>5} {'H−bestBase':14} mAP_parent gap = {gap:+.3f} "
                      f"(best raw = {best_b} @ {best_v:.3f})  -> {verdict}", flush=True)

            if args.novel_leaf and c == args.curvatures[0]:
                nls = [run_novel_leaf(Xh, leaf, parent, leaves_arr, d_emb, c, args.epochs, args.beta, tree_D, s)
                       for s in range(args.seeds)]
                nh = float(np.mean([x["hyperbolic"] for x in nls]))
                ne = float(np.mean([x["euclidean"] for x in nls]))
                results["by_dim"][key]["novel_leaf_parent_acc"] = {
                    "hyperbolic": nh, "euclidean": ne, "delta_H_minus_E": round(nh - ne, 4),
                    "n_placeable_leaves": nls[0]["n_leaves"]}
                print(f"{d_emb:>4} {c:>5.2g} {'novelLeaf':10} H={nh:.3f} E={ne:.3f}  Δ(H−E)={nh-ne:+.3f} "
                      f"(parent placement of held-out leaves; {nls[0]['n_leaves']} placeable)", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump(results, open(args.output + ".json", "w"), indent=2)
    print(f"\n[retr] wrote {args.output}.json", flush=True)

    # plot: Δ(H−E) on the key metrics vs dim (only when a single curvature was swept over dims)
    if len(args.curvatures) == 1 and len(args.dims) > 1:
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            c = args.curvatures[0]
            dims = sorted(args.dims)
            def delta(metric):
                return [results["by_dim"][f"d{d}_c{c}"]["delta_H_minus_E"][metric] for d in dims]
            fig, ax = plt.subplots(1, 2, figsize=(13, 5))
            for metric, lab in [("mAP_parent", "parent mAP"), ("nn_parent_acc", "parent NN acc"), ("mAP_leaf", "leaf mAP")]:
                ax[0].plot(dims, delta(metric), "o-", label=lab)
            ax[0].axhline(0, ls="--", c="k", lw=1); ax[0].set_xscale("log", base=2)
            ax[0].set_xlabel("embedding dim d (log2)"); ax[0].set_ylabel("Δ(H − E)")
            ax[0].set_title("Hyperbolic advantage vs dim\n(>0 and growing as d↓ = POSITIVE per H1)"); ax[0].legend(fontsize=8)
            ax[1].plot(dims, delta("tree_distortion"), "s-", c="tab:purple")
            ax[1].axhline(0, ls="--", c="k", lw=1); ax[1].set_xscale("log", base=2)
            ax[1].set_xlabel("embedding dim d (log2)"); ax[1].set_ylabel("Δ(H − E) tree distortion")
            ax[1].set_title("Tree-distance fidelity\n(negative = H embeds the tree more faithfully)")
            fig.suptitle("Positive experiment: does hyperbolic help hierarchical harm modeling at low d?")
            fig.tight_layout(); fig.savefig(args.output + ".png", dpi=140)
            print(f"[retr] wrote {args.output}.png", flush=True)
        except Exception as e:
            print(f"[retr] plot skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
