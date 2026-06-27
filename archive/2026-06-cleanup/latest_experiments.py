#!/usr/bin/env python3
"""
mentor_experiments.py — ONE script answering Dr Nour's three points, on the same reps.

Her message had three parts; this runs all three:

  EXP 1 — "is it secretly flat?" (her critique: HS mapped to tangent → flat vector → linear probe).
          Take the SAME trained ProtoNet embedding and decide TWO ways on the IDENTICAL points:
            (a) geodesic distance on the hyperboloid   (what we actually do)
            (b) LOG-MAP back to tangent space → linear/Euclidean classifier  (her described pipeline)
          If her claim is true, (a) == (b). If they differ, the curvature is doing real work.

  EXP 2 — her idea: explicit FEATURES as an anomaly score (radial x0 / curvature / displacement),
          each STANDALONE and combined, harm-vs-benign AUROC. We never measured them in isolation.

  EXP 3 — "look at them from 3D/2D space, can we see them?": train at d=2 and plot the actual
          Poincaré disk, colored by category, benign overlaid — the visualization she asked for.

Reuses expmap0_c / lorentz_dist_c / ProtoNet / load_reps from hyperbolic_retrieval.py and the 12
trajectory-feature primitives from hps_core.py. GPU only for the (already-cached) reps; the analysis
is CPU-light. `--selftest` validates the whole pipeline on synthetic data with no GPU.

Usage (real, on the DGX after reps are extracted):
  python mentor_experiments.py \
    --harmful_npz results/harm_taxonomy_deep_llm_reps.npz \
    --benign_reps_npz results/hier_detector_benign.npz \
    --d_emb 32 --c 1.0 --seeds 5 --output results/mentor_experiments

CPU validation (no GPU, synthetic):
  python mentor_experiments.py --selftest
"""
import argparse, json, os, sys
import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
from hyperbolic_retrieval import expmap0_c, lorentz_dist_c, ProtoNet, load_reps
from hps_core import LorentzProjection, extract_trajectory_features
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from itertools import combinations


# ---------------------------------------------------------------------------
# log-map: inverse of expmap0_c — maps a point ON the hyperboloid back to the
# tangent space at the origin (a flat vector). This is literally "flatten it".
# expmap0_c(v): u=sqrt(c)v, n=|u|, returns [cosh(n), sinh(n) u/n].
# logmap0_c(p): given p=[p0, ps], recover v.  n = arccosh(p0); v = (n/ sinh(n)) ps / sqrt(c).
# ---------------------------------------------------------------------------
def logmap0_c(p, c=1.0):
    p0 = p[:, :1].clamp_min(1.0 + 1e-6)
    ps = p[:, 1:]
    n = torch.acosh(p0)                       # geodesic radius
    sinhn = torch.sinh(n).clamp_min(1e-8)
    v = (n / sinhn) * ps / (c ** 0.5)
    return v                                   # flat tangent vector


def train_protonet(Xtr, ytr, n_class, d_emb, hyp, c, epochs, seed):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = ProtoNet(Xtr.shape[1], d_emb, n_class, hyp, c).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=dev); yt = torch.tensor(ytr, device=dev)
    ce = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad(); ce(net.logits(Xt), yt).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
    net.eval()
    return net


# ---------------------------------------------------------------------------
# EXP 1 — geodesic-on-manifold  vs  log-mapped-to-flat-then-linear  (her critique)
# ---------------------------------------------------------------------------
def exp1_flatten_test(Xh, yh, Xb, d_emb, c, epochs, seeds):
    """Train the hyperbolic ProtoNet; then classify harm-vs-benign TWO ways on the SAME embedding:
       (a) geodesic distance to a benign/harm prototype (manifold decision)
       (b) log-map points to flat tangent space, fit a LINEAR probe there (her 'flat' pipeline).
    Report harm-vs-benign AUROC for each + Spearman(geodesic, euclidean) pair-distance to show the
    metrics are not a monotone relabel. If (a)~=(b) AND Spearman~=1, her claim holds; else it fails."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    aucs_geo, aucs_flat, spears = [], [], []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        # standardize on train; simple 70/30 split per class
        def split(n): idx = rng.permutation(n); k = int(0.7 * n); return idx[:k], idx[k:]
        htr, hte = split(len(Xh)); btr, bte = split(len(Xb))
        mu = np.concatenate([Xh[htr], Xb[btr]]).mean(0); sd = np.concatenate([Xh[htr], Xb[btr]]).std(0) + 1e-6
        Xh_n, Xb_n = (Xh - mu) / sd, (Xb - mu) / sd
        # train the hyperbolic encoder as a 2-class (harm/benign) protonet
        Xtr = np.concatenate([Xh_n[htr], Xb_n[btr]]); ytr = np.concatenate([np.zeros(len(htr)), np.ones(len(btr))]).astype(int)
        net = train_protonet(Xtr, ytr, 2, d_emb, True, c, epochs, s)
        with torch.no_grad():
            Xte = np.concatenate([Xh_n[hte], Xb_n[bte]]); yte = np.concatenate([np.ones(len(hte)), np.zeros(len(bte))])
            e_te = net.encode(torch.tensor(Xte, dtype=torch.float32, device=dev))
            # (a) GEODESIC decision: harm-score = (dist to benign proto) - (dist to harm proto)
            P = expmap0_c(e_te, c)
            protoP = expmap0_c(net.proto, c)
            D = lorentz_dist_c(P, protoP, c).cpu().numpy()      # cols: [harm_proto, benign_proto]
            geo_score = D[:, 1] - D[:, 0]                        # high => closer to harm
            aucs_geo.append(roc_auc_score(yte, geo_score))
            # (b) FLATTEN (log-map) then LINEAR probe — her described pipeline
            e_tr = net.encode(torch.tensor(Xtr, dtype=torch.float32, device=dev))
            flat_tr = logmap0_c(expmap0_c(e_tr, c), c).cpu().numpy()
            flat_te = logmap0_c(expmap0_c(e_te, c), c).cpu().numpy()
            clf = LogisticRegression(max_iter=2000).fit(flat_tr, ytr == 0)  # 0 was harm in ytr
            flat_score = clf.decision_function(flat_te)
            aucs_flat.append(roc_auc_score(yte, flat_score))
            # Spearman: are geodesic and euclidean pair-distances the same ordering?
            Pn = P.cpu().numpy(); en = e_te.cpu().numpy()
            iu = np.triu_indices(len(en), 1)
            Dgeo = lorentz_dist_c(P, P, c).cpu().numpy()[iu]
            Deuc = np.linalg.norm(en[:, None] - en[None, :], axis=-1)[iu]
            spears.append(spearmanr(Dgeo, Deuc).statistic)
    return {
        "geodesic_AUROC": [float(np.mean(aucs_geo)), float(np.std(aucs_geo))],
        "flattened_linear_AUROC": [float(np.mean(aucs_flat)), float(np.std(aucs_flat))],
        "spearman_geodesic_vs_euclidean": [float(np.mean(spears)), float(np.std(spears))],
        "verdict": ("curvature is FLATTENED (her claim holds)" if abs(np.mean(spears)) > 0.999
                    else "curvature is REAL in the decision (geodesic != flat relabel)"),
    }


# ---------------------------------------------------------------------------
# EXP 2 — explicit features as an anomaly score (her idea), each standalone
# radial = x0 of the projected point; curvature/displacement from the layer trajectory if available,
# else computed on the single-layer embedding's geometry.
# ---------------------------------------------------------------------------
def exp2_feature_anomaly(Xh, Xb, c, seeds):
    """Per-feature harm-vs-benign AUROC, on the lifted single-layer reps.
       radial  = x0 = sqrt(1/c + |x|^2)  (her 'distance from apex')
       norm    = |x| (euclidean magnitude, the thing radial is monotone in)
       For a single layer we don't have a trajectory, so 'curvature/displacement' (which need the
       per-layer path) are reported only if the npz has a layer axis; otherwise flagged N/A."""
    y = np.concatenate([np.ones(len(Xh)), np.zeros(len(Xb))])
    X = np.concatenate([Xh, Xb]).astype(np.float64)
    feats = {}
    norm = np.linalg.norm(X, axis=1)
    feats["norm |x|"] = norm
    feats["radial x0 = sqrt(1/c+|x|^2)"] = np.sqrt(1.0 / c + norm ** 2)
    # mean abs activation & std as cheap 'shape' proxies
    feats["mean|act|"] = np.abs(X).mean(1)
    feats["act std"] = X.std(1)
    out = {}
    for name, f in feats.items():
        a = roc_auc_score(y, f)
        out[name] = round(max(a, 1 - a), 4)   # report separating power (flip-invariant)
    # combined unsupervised-ish anomaly score: distance from benign mean in standardized feat space
    F = np.stack(list(feats.values()), 1)
    Fz = StandardScaler().fit(F[y == 0]).transform(F)     # fit on benign only (anomaly = far from benign)
    anom = np.linalg.norm(Fz, axis=1)
    out["COMBINED anomaly (dist from benign)"] = round(max(roc_auc_score(y, anom), 1 - roc_auc_score(y, anom)), 4)
    out["_note"] = ("curvature/displacement need the per-layer trajectory (multi-layer cache); "
                    "single-layer reps give radial/magnitude/shape only")
    return out


# ---------------------------------------------------------------------------
# EXP 3 — Poincaré-disk visualization at d=2 (what she asked to 'see')
# ---------------------------------------------------------------------------
def exp3_poincare_disk(Xh, yh, leaves, Xb, c, epochs, seed, output):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_class = int(yh.max()) + 1
    mu = Xh.mean(0); sd = Xh.std(0) + 1e-6
    net = train_protonet((Xh - mu) / sd, yh, n_class, 2, True, c, epochs, seed)
    with torch.no_grad():
        e = net.encode(torch.tensor((Xh - mu) / sd, dtype=torch.float32, device=dev))
        P = expmap0_c(e, c).cpu().numpy()        # [n, 3]: (x0, x1, x2)
        eb = net.encode(torch.tensor((Xb - mu) / sd, dtype=torch.float32, device=dev))
        Pb = expmap0_c(eb, c).cpu().numpy()
    # map hyperboloid -> Poincaré disk: y_i = x_i / (1 + x0)
    def to_disk(Q): return Q[:, 1:] / (1.0 + Q[:, :1])
    Dh, Db = to_disk(P), to_disk(Pb)
    cats = [l.split("/")[0] for l in leaves]
    fig, ax = plt.subplots(figsize=(7, 7))
    circ = plt.Circle((0, 0), 1.0, fill=False, color="gray", lw=1); ax.add_patch(circ)
    ax.scatter(Db[:, 0], Db[:, 1], s=6, c="lightgray", alpha=0.5, label="benign")
    uniq = sorted(set(cats))
    cmap = plt.cm.tab10(np.linspace(0, 1, len(uniq)))
    for ci, cat in enumerate(uniq):
        m = np.array([cats[yh[i]] == cat for i in range(len(yh))])
        ax.scatter(Dh[m, 0], Dh[m, 1], s=10, color=cmap[ci], label=cat, alpha=0.7)
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_aspect("equal")
    ax.set_title("Harm taxonomy on the Poincaré disk (d=2)\nparents central, sub-categories toward the edge")
    ax.legend(fontsize=7, markerscale=2, loc="upper right", bbox_to_anchor=(1.35, 1.0))
    fig.tight_layout(); fig.savefig(output + "_poincare.png", dpi=140, bbox_inches="tight")
    print(f"[exp3] wrote {output}_poincare.png", flush=True)


# ---------------------------------------------------------------------------
# EXP 4 — HER ACTUAL EXPERIMENT: trajectory features (radial / curvature / displacement)
# → anomaly score, ALL combinations (each alone, each pair, all three).
# Needs MULTI-LAYER reps: X shape (N, n_layers, d). Features come from hps_core's 12-dim extractor,
# split into the 3 groups she named:  radial = idx 0:5, curvature = 5:9, displacement = 9:12.
# ---------------------------------------------------------------------------
FEATURE_GROUPS = {"radial": slice(0, 5), "curvature": slice(5, 9), "displacement": slice(9, 12)}


class IdentityLift(torch.nn.Module):
    """NO-COMPRESSION lift: map the FULL d-dim activation onto the hyperboloid directly (no learned
    4096->64 squeeze). forward(x) returns [x0, x] with x0=sqrt(1/k+|x|^2) — same x0 formula as
    LorentzProjection but on the full vector, so extract_trajectory_features works unchanged."""
    def __init__(self, k=0.1):
        super().__init__()
        self.k_val = float(k)
        self._dev = torch.nn.Parameter(torch.zeros(1))   # dummy param so .parameters()/.device work
    @property
    def k(self):
        return torch.tensor(self.k_val)
    def forward(self, x):
        x = x.float()
        x0 = torch.sqrt(1.0 / self.k_val + (x ** 2).sum(-1, keepdim=True))
        return torch.cat([x0, x], dim=-1)


def exp4_feature_combinations(Xh_ml, Xb_ml, kappa, seeds, epochs, no_compression=False):
    """Xh_ml, Xb_ml: (N, n_layers, d) multi-layer reps for harm and benign.
    Train a LorentzProjection (or, if no_compression, lift the FULL d-dim activation with no squeeze),
    extract the 12 trajectory features, split into radial/curvature/displacement, and for EVERY
    non-empty subset of the 3 groups build a harm-vs-benign score two ways:
      - supervised: logistic regression on the subset's features
      - anomaly:    distance from the benign mean in standardized subset-feature space (unsupervised)
    Report mean AUROC over seeds for each subset. -> a clean 7-row table."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_layers, d_hidden = Xh_ml.shape[1], Xh_ml.shape[2]
    groups = list(FEATURE_GROUPS.keys())
    subsets = [c for r in range(1, len(groups) + 1) for c in combinations(groups, r)]  # 7 subsets
    agg_sup = {s: [] for s in subsets}; agg_anom = {s: [] for s in subsets}
    for seed in range(seeds):
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        def split(n): idx = rng.permutation(n); k = int(0.7 * n); return idx[:k], idx[k:]
        htr, hte = split(len(Xh_ml)); btr, bte = split(len(Xb_ml))
        Xtr = np.concatenate([Xh_ml[htr], Xb_ml[btr]]); ytr = np.concatenate([np.ones(len(htr)), np.zeros(len(btr))])
        if no_compression:
            # NO learned projection: lift the full d-dim activation directly (identity, no squeeze)
            proj = IdentityLift(kappa).to(dev)
        else:
            proj = LorentzProjection(d_hidden, 64, kappa, n_layers=n_layers).to(dev)
            # light training so the projection isn't random: contrastive on train (harm vs benign)
            from hps_core import contrastive_loss
            opt = torch.optim.Adam(proj.parameters(), lr=1e-2)
            Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev); ytr_t = torch.tensor(ytr, device=dev)
            for _ in range(epochs):
                opt.zero_grad(); loss = 0.0
                for l in range(n_layers):
                    loss = loss + contrastive_loss(proj(Xtr_t[:, l, :]), ytr_t, k=proj.k)
                (loss / n_layers).backward(); torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0); opt.step()
        # extract the 12 features for train + test (clean NaN/inf the geodesic extractor can emit)
        Ftr = np.nan_to_num(extract_trajectory_features(proj, Xtr), nan=0.0, posinf=0.0, neginf=0.0)
        Xte = np.concatenate([Xh_ml[hte], Xb_ml[bte]]); yte = np.concatenate([np.ones(len(hte)), np.zeros(len(bte))])
        Fte = np.nan_to_num(extract_trajectory_features(proj, Xte), nan=0.0, posinf=0.0, neginf=0.0)
        for sub in subsets:
            cols = np.r_[tuple(np.arange(FEATURE_GROUPS[g].start, FEATURE_GROUPS[g].stop) for g in sub)]
            sc = StandardScaler().fit(Ftr[:, cols])
            Ztr, Zte = sc.transform(Ftr[:, cols]), sc.transform(Fte[:, cols])
            # supervised
            clf = LogisticRegression(max_iter=2000).fit(Ztr, ytr)
            a_sup = roc_auc_score(yte, clf.decision_function(Zte))
            agg_sup[sub].append(max(a_sup, 1 - a_sup))
            # anomaly: fit benign mean/scale on TRAIN benign, score = distance from it
            scb = StandardScaler().fit(Ftr[:, cols][ytr == 0])
            anom = np.linalg.norm(scb.transform(Fte[:, cols]), axis=1)
            a_an = roc_auc_score(yte, anom)
            agg_anom[sub].append(max(a_an, 1 - a_an))
    table = {}
    for sub in subsets:
        table[" + ".join(sub)] = {
            "n_features": int(sum(FEATURE_GROUPS[g].stop - FEATURE_GROUPS[g].start for g in sub)),
            "supervised_AUROC": [round(float(np.mean(agg_sup[sub])), 4), round(float(np.std(agg_sup[sub])), 4)],
            "anomaly_AUROC": [round(float(np.mean(agg_anom[sub])), 4), round(float(np.std(agg_anom[sub])), 4)],
        }
    return table


def load_multilayer(cache_path, hps_layers=(0, 2, 17, 24, 28, 31)):
    """Load the HPS multi-layer activation cache → (Xh, Xb) each (N, n_layers, d).
    Matches the format used by statistical_tests.py: keys hs_{train,test}_{ben,atk}, each an array of
    per-sample objects where hs[layer][-1] = last-token activation at that layer. We pool train+test
    (we only need the features, not the original split) and slice the standard HPS layers."""
    cache = np.load(cache_path, allow_pickle=True)
    keys = list(cache.keys())

    def feat(hs, l):
        a = np.asarray(hs[l], dtype=np.float32)
        return a[-1] if a.ndim == 2 else a   # (seq,d) -> last token; (d,) -> already last-token

    def to_array(hs_list):
        arr = np.array([[feat(hs, l) for l in hps_layers] for hs in hs_list], dtype=np.float32)
        assert arr.ndim == 3, f"expected (N,n_layers,d), got {arr.shape} (per-element shape mismatch)"
        return arr

    # primary: the hs_{train,test}_{ben,atk} layout
    if {"hs_train_ben", "hs_train_atk"} <= set(keys):
        ben = list(cache["hs_train_ben"].tolist())
        atk = list(cache["hs_train_atk"].tolist())
        if "hs_test_ben" in cache: ben += list(cache["hs_test_ben"].tolist())
        if "hs_test_atk" in cache: atk += list(cache["hs_test_atk"].tolist())
        Xb, Xh = to_array(ben), to_array(atk)
        print(f"[exp4] loaded multilayer: harm {Xh.shape}, benign {Xb.shape} "
              f"(layers {list(hps_layers)})", flush=True)
        return Xh, Xb

    # fallback: explicit 3-D arrays
    def pick(*names):
        for n in names:
            if n in cache and np.asarray(cache[n]).ndim == 3: return np.asarray(cache[n], np.float32)
        return None
    Xh = pick("X_atk", "harm_reps", "attack_reps", "Xh"); Xb = pick("X_ben", "benign_reps", "Xb")
    if Xh is not None and Xb is not None:
        return Xh, Xb
    raise SystemExit(f"[exp4] unrecognized cache layout in {cache_path}. keys={keys}.")


def make_synth():
    rng = np.random.default_rng(0); d = 128
    cats, per = 4, 60
    ctr = rng.standard_normal((cats, d)) * 4
    Xh = np.concatenate([ctr[i] + rng.standard_normal((per, d)) for i in range(cats)]).astype("float32")
    yh = np.repeat(np.arange(cats), per)
    leaves = [f"cat{i}/leaf{i}" for i in range(cats)]
    Xb = (rng.standard_normal((150, d)) * 4 - 6).astype("float32")
    return Xh, yh, leaves, Xb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harmful_npz"); ap.add_argument("--benign_reps_npz")
    ap.add_argument("--d_emb", type=int, default=32); ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--output", default="results/mentor_experiments")
    ap.add_argument("--multilayer_npz", help="multi-layer cache (N,n_layers,d) for EXP 4 — her actual "
                    "experiment: radial/curvature/displacement feature combinations -> anomaly score")
    ap.add_argument("--kappa", type=float, default=0.1)
    ap.add_argument("--no_compression", action="store_true",
                    help="EXP 4: lift the FULL d-dim activation onto the hyperboloid (no learned "
                         "4096->64 squeeze) — tests whether compression is what kills the features.")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        print("[mentor-exp] SELFTEST (synthetic, no GPU)\n")
        Xh, yh, leaves, Xb = make_synth(); args.seeds = 2; args.epochs = 120
    else:
        if not args.harmful_npz: ap.error("--harmful_npz required (or --selftest)")
        Xh, leaf, parent, leaves_arr, tree_D, leaves, parents = load_reps(args.harmful_npz)
        yh = leaf
        if args.benign_reps_npz and os.path.exists(args.benign_reps_npz):
            Xb = np.load(args.benign_reps_npz)["reps"].astype(np.float32)
        else:
            ap.error("--benign_reps_npz required (benign reps for harm-vs-benign)")

    results = {"config": vars(args)}
    print("=" * 70)
    print("EXP 1 — Is the decision secretly flat? (geodesic vs log-mapped-linear)")
    print("=" * 70)
    e1 = exp1_flatten_test(Xh, yh, Xb, args.d_emb, args.c, args.epochs, args.seeds)
    results["exp1_flatten_test"] = e1
    print(f"  geodesic-on-manifold AUROC : {e1['geodesic_AUROC'][0]:.3f} ± {e1['geodesic_AUROC'][1]:.3f}")
    print(f"  flattened(log-map)+linear  : {e1['flattened_linear_AUROC'][0]:.3f} ± {e1['flattened_linear_AUROC'][1]:.3f}")
    print(f"  Spearman(geodesic,euclid)  : {e1['spearman_geodesic_vs_euclidean'][0]:.3f}  (1.0 => secretly flat)")
    print(f"  VERDICT: {e1['verdict']}\n")

    print("=" * 70)
    print("EXP 2 — Her idea: explicit FEATURES as an anomaly score (standalone AUROC)")
    print("=" * 70)
    e2 = exp2_feature_anomaly(Xh, Xb, args.c, args.seeds)
    results["exp2_feature_anomaly"] = e2
    for k, v in e2.items():
        if not k.startswith("_"): print(f"  {k:38s}: AUROC {v}")
    print(f"  note: {e2['_note']}\n")

    print("=" * 70)
    print("EXP 3 — Poincaré disk visualization (d=2)")
    print("=" * 70)
    try:
        exp3_poincare_disk(Xh, yh, leaves, Xb, args.c, args.epochs, 0, args.output)
        results["exp3_poincare"] = args.output + "_poincare.png"
    except Exception as ex:
        print(f"  [exp3] skipped: {ex}")

    # EXP 4 — her ACTUAL experiment: radial/curvature/displacement combinations (needs multi-layer)
    Xh_ml = Xb_ml = None
    if args.selftest:
        # synthetic multi-layer trajectory: 6 layers, harm has a distinctive cross-layer path
        rng = np.random.default_rng(1); d = 64; nl = 6
        def traj(n, drift):
            base = rng.standard_normal((n, 1, d)) * 2
            steps = np.cumsum(rng.standard_normal((n, nl, d)) * 0.3 + drift, axis=1)
            return (base + steps).astype("float32")
        Xh_ml = traj(180, 0.15); Xb_ml = traj(150, -0.05)
    elif args.multilayer_npz:
        Xh_ml, Xb_ml = load_multilayer(args.multilayer_npz)

    if Xh_ml is not None:
        mode = "NO COMPRESSION (full d-dim lift)" if args.no_compression else "HPS projection (4096->64)"
        print("\n" + "=" * 70)
        print(f"EXP 4 — radial/curvature/displacement → anomaly, ALL combinations  [{mode}]")
        print("=" * 70)
        e4 = exp4_feature_combinations(Xh_ml, Xb_ml, args.kappa, args.seeds, min(args.epochs, 60),
                                       no_compression=args.no_compression)
        results["exp4_feature_combinations"] = e4
        print(f"  {'feature subset':30s} {'#f':>3} {'supervised':>14} {'anomaly':>14}")
        for name, v in e4.items():
            print(f"  {name:30s} {v['n_features']:>3} "
                  f"{v['supervised_AUROC'][0]:>8.3f}±{v['supervised_AUROC'][1]:.3f} "
                  f"{v['anomaly_AUROC'][0]:>8.3f}±{v['anomaly_AUROC'][1]:.3f}")
        print("  (AUROC reported as separating power, flip-invariant; both supervised LR and "
              "unsupervised anomaly-from-benign shown)")
    else:
        print("\n[exp4] skipped — pass --multilayer_npz <cache> to run the radial/curvature/"
              "displacement feature-combination experiment (needs multi-layer reps).")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump(results, open(args.output + ".json", "w"), indent=2)
    print(f"\n[mentor-exp] wrote {args.output}.json")


if __name__ == "__main__":
    main()
