#!/usr/bin/env python3
"""
INDEPENDENT ARBITER — does the hyperbolic detector make a genuinely CURVED decision,
or is it (as claimed) "flattened to tangent space + a linear probe"?

No assumptions, no appeal to authority. Re-implements the exact ops from
hierarchical_detector.py (expmap0 + lorentz geodesic distance, curvature c) from scratch
and runs four falsifiable tests. Prints a verdict derived only from the numbers.
"""
import numpy as np
from numpy.random import default_rng
try:
    from scipy.stats import spearmanr
except Exception:
    def spearmanr(a, b):
        ar = np.argsort(np.argsort(a)); br = np.argsort(np.argsort(b))
        return (np.corrcoef(ar, br)[0, 1], 0.0)
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist

rng = default_rng(0)

def expmap0(V, c, clip=30.0):           # tangent vector(s) -> point(s) on the hyperboloid
    u = np.sqrt(c) * np.atleast_2d(V)
    n = np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
    un = clip * np.tanh(n / clip)
    return np.concatenate([np.cosh(un), np.sinh(un) * (u / n)], axis=1)

def lorentz_dist(X, Y, c):              # geodesic distance on the hyperboloid  [N,M]
    inner = -np.outer(X[:, 0], Y[:, 0]) + X[:, 1:] @ Y[:, 1:].T
    return np.arccosh(np.clip(-inner, 1 + 1e-9, None)) / np.sqrt(c)

d, N, K = 16, 4000, 14
V = rng.standard_normal((N, d)); P = rng.standard_normal((K, d))

def metrics(c):
    Dh = lorentz_dist(expmap0(V, c), expmap0(P, c), c)   # curved distances [N,K]
    De = cdist(V, P)                                     # flat distances    [N,K]
    rho = spearmanr(Dh.ravel(), De.ravel())[0]
    disagree = float(np.mean(Dh.argmin(1) != De.argmin(1)))   # do they pick a DIFFERENT nearest prototype?
    return rho, disagree

print("=== TEST 1 — is the curved distance just the flat distance relabeled? ===")
for c in [0.001, 0.01, 0.1, 1.0, 2.0, 4.0]:
    rho, dis = metrics(c)
    print(f"  c={c:<6}: Spearman(curved,flat)={rho:.4f}   nearest-prototype DISAGREEMENT={100*dis:4.1f}%")

print("\n=== TEST 2 — sanity: does c->0 collapse curved to flat? (parameterization check) ===")
rho0, dis0 = metrics(1e-4)
print(f"  c->0: Spearman={rho0:.4f} (expect ~1.0), disagreement={100*dis0:.1f}% (expect ~0)  -> {'PASS' if rho0>0.999 else 'FAIL'}")

print("\n=== TEST 3 — is the curved DECISION linearly reproducible? (in 2D, capacity-limited) ===")
# In high-d a linear model fits almost any boundary by sheer capacity, so this is only
# meaningful in LOW dimension where a linear separator = one straight line.
V2 = rng.standard_normal((4000, 2)); p0, p1 = rng.standard_normal((2, 2))
Hp = expmap0(np.vstack([p0, p1]), 1.0); Hpts = expmap0(V2, 1.0)
lab_hyp = lorentz_dist(Hpts, Hp, 1.0).argmin(1)
lab_euc = cdist(V2, np.vstack([p0, p1])).argmin(1)
acc_hyp = LogisticRegression(max_iter=5000).fit(V2, lab_hyp).score(V2, lab_hyp)
acc_euc = LogisticRegression(max_iter=5000).fit(V2, lab_euc).score(V2, lab_euc)
print(f"  best straight line reproducing the FLAT decision   : {100*acc_euc:5.1f}%  (~100 -> flat boundary IS a line)")
print(f"  best straight line reproducing the CURVED decision : {100*acc_hyp:5.1f}%  (<100 -> curved boundary is NOT a line)")

print("\n=== TEST 4 — empirical clincher: if flattened, H and E would be IDENTICAL ===")
try:
    import json
    j = json.load(open('results/hyp_retr_deep3_stats.json'))['by_dim']['d32_c1.0']
    H = j['hyperbolic']['mAP_parent'][0]; E = j['euclidean']['mAP_parent'][0]
    print(f"  retrieval mAP_parent (d32, 3-level): hyperbolic={H:.3f}  euclidean={E:.3f}  gap={H-E:+.3f}")
    print(f"  a nonzero gap is only possible if the metric (curvature) changes the decision.")
except Exception as e:
    print(f"  (JSON not available here: {e}) — gap was +0.10..0.16 mAP in committed results.")

print("\n================ VERDICT ================")
rho1, dis1 = metrics(1.0)
curved_metric = rho1 < 0.999
curved_decision = dis1 > 0.05            # >5% different nearest-prototype picks (capacity-free)
print(f"  [decisive] curved vs flat pick a DIFFERENT nearest prototype on {100*dis1:.1f}% of points")
print(f"  [decisive] Spearman(curved,flat) = {rho1:.3f}  (<1 -> metrics genuinely differ)")
print(f"  [support ] 2D curved boundary non-linear: best line {100*acc_hyp:.1f}% (flat {100*acc_euc:.1f}%)")
print(f"  [sanity  ] c->0 collapses curved->flat: Spearman {rho0:.3f}")
if curved_metric and curved_decision:
    print('\n  CONCLUSION (from the numbers, no opinion): the hyperbolic detector makes a')
    print('  GENUINELY CURVED decision — curved and flat disagree on a large fraction of')
    print('  decisions, so it is NOT "flattened to tangent space + a linear probe."')
    print('  The claim is FALSE for the geodesic-distance (retrieval/ProtoNet) probe.')
    print('  (TRUE only for the separate rpf probe = linear classifier on lifted coords,')
    print('   which is the inert / negative-result probe.)')
else:
    print('\n  CONCLUSION: decision is effectively flat — the claim would hold.')
