"""
Experiment 1 — Test multiple hyperbolic methods, not just HPS.

Goal: Strengthen the claim "hyperbolic doesn't help" by showing that several
different hyperbolic frameworks all fail to beat C4. If one specific approach
(HPS) fails it could be implementation-specific. If five different approaches
all fail, the conclusion is much stronger.

Methods compared:
  HPS (Lorentz projection + contrastive + trajectory features) -- existing
  Hyperbolic Autoencoder (Lorentz reconstruction, latent as features)
  Hyperbolic Multi-Layer Perceptron (stacked Lorentz layers + nonlinearity)
  Lorentz Centroid (distance to class centroids in Lorentz space)
  Hyperbolic Linear Probe (PCA -> hyperboloid -> distance features)
  Euclidean (matched parameter count, flat space)
  C4 (linear probe on mean-pooled raw activations)

Evaluation:
  Llama-3-8B and Vicuna-13B, same-distribution full-data,
  with 5% FPR threshold protocol (held-out calibration split).

Skip Poincare ball: Lorentz and Poincare are isometric models of the same
hyperbolic space. Any result on Lorentz transfers to Poincare. Including both
would be redundant.

Usage:
  python strengthen_negative/experiment1_hyperbolic_methods.py
"""

import os
import sys
import json

# Allow running from either the project root or the strengthen_negative folder.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from helpers.lorentz_ops import (
    LorentzProjection, lorentz_distance, lorentz_inner,
    to_hyperboloid, hyperbolic_origin
)

# ----------------------------- Config -----------------------------
HPS_LAYERS_LLAMA = [0, 2, 17, 24, 28, 31]
HPS_LAYERS_VICUNA_DEFAULT = [0, 2, 22, 31, 35, 39]
KAPPA_INIT = 0.1
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-5
PROJ_DIM = 64
FPR_TARGET = 0.05
device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------- Data loading --------------------------
def to_arr(hs_list, layers):
    return np.array([[hs[l][-1] for l in layers] for hs in hs_list])


def load_llama3():
    """Load Llama-3 cache."""
    path = "results/llama3_activations_cache.npz"
    if not os.path.exists(path):
        return None
    cache = np.load(path, allow_pickle=True)
    layers = HPS_LAYERS_LLAMA
    X_tr_ben = to_arr(cache["hs_train_ben"].tolist(), layers)
    X_tr_atk = to_arr(cache["hs_train_atk"].tolist(), layers)
    X_te_ben = to_arr(cache["hs_test_ben"].tolist(), layers)
    X_te_atk = to_arr(cache["hs_test_atk"].tolist(), layers)
    return {
        "name": "Llama-3-8B",
        "X_tr_ben": X_tr_ben, "X_tr_atk": X_tr_atk,
        "X_te_ben": X_te_ben, "X_te_atk": X_te_atk,
        "layers": layers, "d_hidden": X_tr_ben.shape[2],
    }


def load_vicuna():
    """Load Vicuna cache (different format, see extract_vicuna_activations.py)."""
    path = "results/vicuna_activations_cache.npz"
    if not os.path.exists(path):
        return None
    cache = np.load(path, allow_pickle=True)
    X_benign = cache["X_benign"]   # (N_ben, n_layers, d_hidden)
    X_attack = cache["X_attack"]   # (N_atk, n_layers, d_hidden)
    layers = list(cache["layers"]) if "layers" in cache.files else HPS_LAYERS_VICUNA_DEFAULT

    # Standard 80/20 split (matches existing Vicuna pipeline)
    rng = np.random.RandomState(42)
    ben_idx = rng.permutation(len(X_benign))
    atk_idx = rng.permutation(len(X_attack))
    n_tr_b = int(0.8 * len(X_benign))
    n_tr_a = int(0.8 * len(X_attack))
    X_tr_ben = X_benign[ben_idx[:n_tr_b]]
    X_te_ben = X_benign[ben_idx[n_tr_b:]]
    X_tr_atk = X_attack[atk_idx[:n_tr_a]]
    X_te_atk = X_attack[atk_idx[n_tr_a:]]
    return {
        "name": "Vicuna-13B",
        "X_tr_ben": X_tr_ben, "X_tr_atk": X_tr_atk,
        "X_te_ben": X_te_ben, "X_te_atk": X_te_atk,
        "layers": layers, "d_hidden": X_tr_ben.shape[2],
    }


# -------------------------- Evaluation --------------------------
def eval_features(features_train, y_train, features_te_ben, features_te_atk, seed=42):
    """Standard eval protocol: standardize -> LR -> calib/heldout split -> 5% FPR."""
    if features_train.ndim == 1:
        features_train = features_train.reshape(-1, 1)
        features_te_ben = features_te_ben.reshape(-1, 1)
        features_te_atk = features_te_atk.reshape(-1, 1)

    sc = StandardScaler()
    Xtr = sc.fit_transform(features_train)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(Xtr, y_train)

    n_calib = max(1, len(features_te_ben) // 2)
    s_calib = clf.predict_proba(sc.transform(features_te_ben[:n_calib]))[:, 1]
    s_ben = clf.predict_proba(sc.transform(features_te_ben[n_calib:]))[:, 1]
    s_atk = clf.predict_proba(sc.transform(features_te_atk))[:, 1]

    thr = float(np.quantile(s_calib, 1.0 - FPR_TARGET))
    tpr = float((s_atk > thr).mean())
    fpr = float((s_ben > thr).mean())
    auroc = roc_auc_score(
        np.array([0] * len(s_ben) + [1] * len(s_atk)),
        np.concatenate([s_ben, s_atk])
    )
    return {"auroc": auroc, "tpr": tpr, "fpr": fpr}


def _contrastive_pairs_minibatch(h_proj, y_t, k, margin=2.0, batch_size=512, seed=0):
    """Compute contrastive loss over a sampled minibatch of pairs.

    h_proj: (N, d_proj+1) on the hyperboloid
    y_t:    (N,) labels
    Returns scalar loss tensor.
    """
    n = h_proj.shape[0]
    if n <= batch_size:
        idx = torch.arange(n, device=h_proj.device)
    else:
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(n, generator=g)[:batch_size].to(h_proj.device)
    h = h_proj[idx]
    y = y_t[idx]
    d = lorentz_distance(h.unsqueeze(0), h.unsqueeze(1), k=k)
    same = (y.unsqueeze(0) == y.unsqueeze(1)).float()
    diff = 1.0 - same
    triu = torch.triu(torch.ones_like(d), diagonal=1)
    ns = (same * triu).sum().clamp(min=1)
    nd = (diff * triu).sum().clamp(min=1)
    return ((d ** 2 * same * triu).sum() / ns +
            (torch.clamp(margin - d, min=0) ** 2 * diff * triu).sum() / nd) / 2


def _contrastive_pairs_minibatch_l2(h, y_t, margin, batch_size=512, seed=0):
    """L2 contrastive loss minibatch (for Euclidean and HMLP variants)."""
    n = h.shape[0]
    if n <= batch_size:
        idx = torch.arange(n, device=h.device)
    else:
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(n, generator=g)[:batch_size].to(h.device)
    h_b = h[idx]
    y = y_t[idx]
    d = torch.cdist(h_b, h_b)
    same = (y.unsqueeze(0) == y.unsqueeze(1)).float()
    diff = 1.0 - same
    triu = torch.triu(torch.ones_like(d), diagonal=1)
    ns = (same * triu).sum().clamp(min=1)
    nd = (diff * triu).sum().clamp(min=1)
    return ((d ** 2 * same * triu).sum() / ns +
            (torch.clamp(margin - d, min=0) ** 2 * diff * triu).sum() / nd) / 2


# -------------------------- Method 1: HPS (existing) --------------------------
def method_hps(data, seed=42):
    """The Lorentz projection + contrastive + radial mean feature."""
    n_layers = data["X_tr_ben"].shape[1]
    d_hidden = data["X_tr_ben"].shape[2]

    X_train = np.concatenate([data["X_tr_ben"], data["X_tr_atk"]])
    y_train = np.array([0] * len(data["X_tr_ben"]) + [1] * len(data["X_tr_atk"]))

    torch.manual_seed(seed)
    proj = LorentzProjection(d_hidden, PROJ_DIM, k_init=KAPPA_INIT, freeze_kappa=True).to(device)
    opt = optim.Adam([p for p in proj.parameters() if p.requires_grad],
                     lr=LR, weight_decay=WEIGHT_DECAY)
    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.long, device=device)

    for ep in range(EPOCHS):
        loss = torch.tensor(0.0, device=device)
        for l in range(n_layers):
            h = proj(Xt[:, l, :])  # on the hyperboloid
            loss = loss + _contrastive_pairs_minibatch(
                h, yt, k=proj.k.item(), margin=2.0, batch_size=512, seed=ep
            )
        loss = loss / n_layers
        opt.zero_grad()
        loss.backward()
        opt.step()

    proj.eval()

    def feats(X):
        out = []
        with torch.no_grad():
            for i in range(len(X)):
                xi = torch.tensor(X[i], dtype=torch.float32, device=device)
                h = proj(xi)  # (n_layers, d_proj+1)
                # Single feature: mean radial position
                out.append(float(h[:, 0].mean().cpu()))
        return np.array(out).reshape(-1, 1)

    f_tr = feats(X_train)
    f_be = feats(data["X_te_ben"])
    f_at = feats(data["X_te_atk"])
    return eval_features(f_tr, y_train, f_be, f_at, seed=seed)


# -------------------------- Method 2: Hyperbolic Autoencoder --------------------------
class HyperbolicAutoencoder(nn.Module):
    """Encoder-decoder where the latent lives on the Lorentz hyperboloid.

    The latent radius (time coordinate) is then used as a single discriminative
    feature. Trained end-to-end with reconstruction loss (no labels, unlike
    HPS's contrastive). This is purely unsupervised representation learning.
    """

    def __init__(self, d_in, d_latent=64, k=0.1):
        super().__init__()
        self.k = k
        self.encoder_lin = nn.Linear(d_in, d_latent)
        self.encoder_scale = nn.Parameter(torch.tensor(1.0 / 8.0))
        self.decoder_lin = nn.Linear(d_latent, d_in)
        nn.init.xavier_uniform_(self.encoder_lin.weight)
        nn.init.xavier_uniform_(self.decoder_lin.weight)

    def encode(self, x):
        h_spatial = self.encoder_lin(x) * self.encoder_scale
        return to_hyperboloid(h_spatial, k=self.k)

    def decode(self, h):
        return self.decoder_lin(h[..., 1:])  # use only spatial part

    def forward(self, x):
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h


def method_hyperbolic_autoencoder(data, seed=42):
    """Train Hyperbolic AE per layer, use mean radius as feature."""
    n_layers = data["X_tr_ben"].shape[1]
    d_hidden = data["X_tr_ben"].shape[2]

    X_train = np.concatenate([data["X_tr_ben"], data["X_tr_atk"]])
    y_train = np.array([0] * len(data["X_tr_ben"]) + [1] * len(data["X_tr_atk"]))

    torch.manual_seed(seed)
    ae = HyperbolicAutoencoder(d_hidden, d_latent=PROJ_DIM, k=KAPPA_INIT).to(device)
    opt = optim.Adam(ae.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)

    for _ in range(EPOCHS):
        opt.zero_grad()
        # Pool over layers for the AE objective (a single AE for all layers)
        x_in = Xt.reshape(-1, d_hidden)
        x_hat, _ = ae(x_in)
        loss = ((x_hat - x_in) ** 2).mean()
        loss.backward()
        opt.step()

    ae.eval()

    def feats(X):
        out = []
        with torch.no_grad():
            for i in range(len(X)):
                xi = torch.tensor(X[i], dtype=torch.float32, device=device)
                h = ae.encode(xi)  # (n_layers, d_proj+1)
                out.append(float(h[:, 0].mean().cpu()))
        return np.array(out).reshape(-1, 1)

    f_tr = feats(X_train)
    f_be = feats(data["X_te_ben"])
    f_at = feats(data["X_te_atk"])
    return eval_features(f_tr, y_train, f_be, f_at, seed=seed)


# -------------------------- Method 3: Hyperbolic Multi-Layer Perceptron --------------------------
class HyperbolicMLP(nn.Module):
    """Stacked Lorentz projections with intermediate non-linearity in tangent
    space. The non-linearity is applied between projections to match the
    standard Ganea et al. style hyperbolic NN.
    """

    def __init__(self, d_in, d_hidden=128, d_out=64, k=0.1):
        super().__init__()
        self.k = k
        self.proj1 = LorentzProjection(d_in, d_hidden, k_init=k, freeze_kappa=True)
        self.proj2 = LorentzProjection(d_hidden, d_out, k_init=k, freeze_kappa=True)
        self.activation = nn.Tanh()

    def forward(self, x):
        # x: (..., d_in) Euclidean input
        h1 = self.proj1(x)        # to first hyperboloid
        # Use spatial part as Euclidean input to next projection,
        # with tanh non-linearity (a common HNN choice).
        z1 = self.activation(h1[..., 1:])
        h2 = self.proj2(z1)
        return h2


def method_hyperbolic_mlp(data, seed=42):
    """Train HMLP with the same contrastive loss as HPS for a fair comparison."""
    n_layers = data["X_tr_ben"].shape[1]
    d_hidden = data["X_tr_ben"].shape[2]

    X_train = np.concatenate([data["X_tr_ben"], data["X_tr_atk"]])
    y_train = np.array([0] * len(data["X_tr_ben"]) + [1] * len(data["X_tr_atk"]))

    torch.manual_seed(seed)
    hmlp = HyperbolicMLP(d_hidden, d_hidden=128, d_out=PROJ_DIM, k=KAPPA_INIT).to(device)
    params = [p for p in hmlp.parameters() if p.requires_grad]
    opt = optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)
    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.long, device=device)

    for ep in range(EPOCHS):
        loss = torch.tensor(0.0, device=device)
        for l in range(n_layers):
            h = hmlp(Xt[:, l, :])
            loss = loss + _contrastive_pairs_minibatch(
                h, yt, k=KAPPA_INIT, margin=2.0, batch_size=512, seed=ep
            )
        loss = loss / n_layers
        opt.zero_grad()
        loss.backward()
        opt.step()

    hmlp.eval()

    def feats(X):
        out = []
        with torch.no_grad():
            for i in range(len(X)):
                xi = torch.tensor(X[i], dtype=torch.float32, device=device)
                h = hmlp(xi)  # (n_layers, d_out+1)
                out.append(float(h[:, 0].mean().cpu()))
        return np.array(out).reshape(-1, 1)

    f_tr = feats(X_train)
    f_be = feats(data["X_te_ben"])
    f_at = feats(data["X_te_atk"])
    return eval_features(f_tr, y_train, f_be, f_at, seed=seed)


# -------------------------- Method 4: Lorentz Centroid distance --------------------------
def method_lorentz_centroid(data, seed=42):
    """Class centroids on the hyperboloid. Feature = log(d_atk / d_ben)
    where d_atk, d_ben are Lorentz distances to attack vs benign centroids,
    computed per layer and averaged.

    No training of a deep network. Just centroid computation. This is the
    minimal possible "use hyperbolic geometry" baseline.
    """
    n_layers = data["X_tr_ben"].shape[1]
    d_hidden = data["X_tr_ben"].shape[2]
    k = KAPPA_INIT

    # Use a fixed random projection to a manageable dimension
    rng = np.random.RandomState(seed)
    W = rng.randn(d_hidden, PROJ_DIM).astype(np.float32) / np.sqrt(d_hidden)
    Wt = torch.tensor(W, device=device)

    def to_hyper(X):
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        # Project per layer and lift to hyperboloid
        spatial = Xt @ Wt   # (N, n_layers, d_proj)
        return to_hyperboloid(spatial, k=k)

    H_ben = to_hyper(data["X_tr_ben"])  # (N_ben, L, d+1)
    H_atk = to_hyper(data["X_tr_atk"])

    # Per-layer centroid via Frechet mean approximation: lift mean(spatial)
    cent_ben_spatial = H_ben[:, :, 1:].mean(dim=0)   # (L, d_proj)
    cent_atk_spatial = H_atk[:, :, 1:].mean(dim=0)
    cent_ben = to_hyperboloid(cent_ben_spatial, k=k)
    cent_atk = to_hyperboloid(cent_atk_spatial, k=k)

    def feats(X):
        H = to_hyper(X)
        # Distance to each centroid per layer
        N = H.shape[0]
        scores = []
        for i in range(N):
            d_be = lorentz_distance(H[i], cent_ben, k=k)
            d_at = lorentz_distance(H[i], cent_atk, k=k)
            score = float((torch.log(d_be + 1e-8) - torch.log(d_at + 1e-8)).mean().cpu())
            scores.append(score)
        return np.array(scores).reshape(-1, 1)

    X_train = np.concatenate([data["X_tr_ben"], data["X_tr_atk"]])
    y_train = np.array([0] * len(data["X_tr_ben"]) + [1] * len(data["X_tr_atk"]))
    f_tr = feats(X_train)
    f_be = feats(data["X_te_ben"])
    f_at = feats(data["X_te_atk"])
    return eval_features(f_tr, y_train, f_be, f_at, seed=seed)


# -------------------------- Method 5: PCA -> hyperboloid -> distance --------------------------
def method_pca_hyperbolic(data, seed=42):
    """PCA reduces to PROJ_DIM, then lift to hyperboloid, then distance to
    centroids. Tests whether *any* hyperbolic processing helps when the
    projection isn't trained.
    """
    from sklearn.decomposition import PCA
    n_layers = data["X_tr_ben"].shape[1]
    d_hidden = data["X_tr_ben"].shape[2]
    k = KAPPA_INIT

    # Fit one PCA on training activations (concat across layers)
    X_all_train = np.concatenate([
        data["X_tr_ben"].reshape(-1, d_hidden),
        data["X_tr_atk"].reshape(-1, d_hidden),
    ])
    pca = PCA(n_components=PROJ_DIM, random_state=seed)
    pca.fit(X_all_train)

    def to_hyper(X):
        N, L, _ = X.shape
        flat = X.reshape(-1, d_hidden)
        proj = pca.transform(flat).reshape(N, L, PROJ_DIM)
        Pt = torch.tensor(proj, dtype=torch.float32, device=device)
        return to_hyperboloid(Pt, k=k)

    H_ben = to_hyper(data["X_tr_ben"])
    H_atk = to_hyper(data["X_tr_atk"])

    cent_ben_spatial = H_ben[:, :, 1:].mean(dim=0)
    cent_atk_spatial = H_atk[:, :, 1:].mean(dim=0)
    cent_ben = to_hyperboloid(cent_ben_spatial, k=k)
    cent_atk = to_hyperboloid(cent_atk_spatial, k=k)

    def feats(X):
        H = to_hyper(X)
        N = H.shape[0]
        scores = []
        for i in range(N):
            d_be = lorentz_distance(H[i], cent_ben, k=k)
            d_at = lorentz_distance(H[i], cent_atk, k=k)
            score = float((torch.log(d_be + 1e-8) - torch.log(d_at + 1e-8)).mean().cpu())
            scores.append(score)
        return np.array(scores).reshape(-1, 1)

    X_train = np.concatenate([data["X_tr_ben"], data["X_tr_atk"]])
    y_train = np.array([0] * len(data["X_tr_ben"]) + [1] * len(data["X_tr_atk"]))
    f_tr = feats(X_train)
    f_be = feats(data["X_te_ben"])
    f_at = feats(data["X_te_atk"])
    return eval_features(f_tr, y_train, f_be, f_at, seed=seed)


# -------------------------- Method 6: C4 baseline --------------------------
def method_c4(data, seed=42):
    """Linear probe on mean-pooled activations. The reference simple baseline."""
    f_tr_ben = data["X_tr_ben"].mean(axis=1)   # (N, d_hidden)
    f_tr_atk = data["X_tr_atk"].mean(axis=1)
    f_te_ben = data["X_te_ben"].mean(axis=1)
    f_te_atk = data["X_te_atk"].mean(axis=1)
    f_tr = np.concatenate([f_tr_ben, f_tr_atk])
    y_train = np.array([0] * len(f_tr_ben) + [1] * len(f_tr_atk))
    return eval_features(f_tr, y_train, f_te_ben, f_te_atk, seed=seed)


# -------------------------- Method 7: Euclidean (matched) --------------------------
def method_euclidean_matched(data, seed=42):
    """Same architecture as HPS but flat space. Trained with L2 contrastive."""
    n_layers = data["X_tr_ben"].shape[1]
    d_hidden = data["X_tr_ben"].shape[2]

    X_train = np.concatenate([data["X_tr_ben"], data["X_tr_atk"]])
    y_train = np.array([0] * len(data["X_tr_ben"]) + [1] * len(data["X_tr_atk"]))

    torch.manual_seed(seed)
    proj = nn.Linear(d_hidden, PROJ_DIM, bias=False).to(device)
    nn.init.xavier_uniform_(proj.weight)
    scale_per_layer = nn.Parameter(torch.ones(n_layers, device=device) / 8.0)
    log_margin = nn.Parameter(torch.tensor(np.log(2.0), device=device))
    opt = optim.Adam(list(proj.parameters()) + [scale_per_layer, log_margin],
                     lr=LR, weight_decay=WEIGHT_DECAY)
    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.long, device=device)

    for ep in range(EPOCHS):
        loss = torch.tensor(0.0, device=device)
        margin = torch.exp(log_margin).clamp(0.5, 5.0)
        for l in range(n_layers):
            h = proj(Xt[:, l, :]) * scale_per_layer[l]
            loss = loss + _contrastive_pairs_minibatch_l2(
                h, yt, margin=margin.item(), batch_size=512, seed=ep
            )
        loss = loss / n_layers
        opt.zero_grad()
        loss.backward()
        opt.step()

    proj.eval()

    def feats(X):
        out = []
        with torch.no_grad():
            for i in range(len(X)):
                xi = torch.tensor(X[i], dtype=torch.float32, device=device)
                # Mean L2 norm across layers as the single feature
                pts = []
                for l in range(n_layers):
                    pts.append(proj(xi[l:l+1]) * scale_per_layer[l])
                pts = torch.cat(pts)
                out.append(float(torch.norm(pts, dim=1).mean().cpu()))
        return np.array(out).reshape(-1, 1)

    f_tr = feats(X_train)
    f_be = feats(data["X_te_ben"])
    f_at = feats(data["X_te_atk"])
    return eval_features(f_tr, y_train, f_be, f_at, seed=seed)


# -------------------------- Main --------------------------
METHODS = {
    "C4 (linear probe on mean-pooled)": method_c4,
    "Euclidean (matched parameters)":   method_euclidean_matched,
    "HPS (Lorentz + contrastive)":      method_hps,
    "Hyperbolic AE (reconstruction)":   method_hyperbolic_autoencoder,
    "Hyperbolic MLP (stacked Lorentz)": method_hyperbolic_mlp,
    "Lorentz Centroid (random proj)":   method_lorentz_centroid,
    "Hyperbolic + PCA (no training)":   method_pca_hyperbolic,
}


def main():
    print("\n" + "=" * 78)
    print(" EXPERIMENT 1 — Test multiple hyperbolic methods")
    print("=" * 78 + "\n")

    datasets = []
    for loader in (load_llama3, load_vicuna):
        d = loader()
        if d is None:
            print(f"[skip] {loader.__name__} cache not found")
            continue
        datasets.append(d)
        print(f"  Loaded {d['name']}: train {len(d['X_tr_ben'])} ben + "
              f"{len(d['X_tr_atk'])} atk, test {len(d['X_te_ben'])} ben + "
              f"{len(d['X_te_atk'])} atk, layers {d['layers']}, d={d['d_hidden']}")

    if not datasets:
        print("ERROR: no caches found. Run hps_llama3.py and "
              "extract_vicuna_activations.py first.")
        return

    all_results = {}
    for ds in datasets:
        print("\n" + "-" * 78)
        print(f"  {ds['name']}")
        print("-" * 78)
        print(f"  {'Method':<40} | {'AUROC':>6} | {'TPR@5%':>7} | {'FPR':>5}")
        print(f"  {'-'*40}-+-{'-'*6}-+-{'-'*7}-+-{'-'*5}")
        ds_results = {}
        for name, fn in METHODS.items():
            try:
                r = fn(ds, seed=42)
                ds_results[name] = r
                print(f"  {name:<40} | {r['auroc']:>6.3f} | "
                      f"{r['tpr']:>7.3f} | {r['fpr']:>5.3f}")
            except Exception as e:
                print(f"  {name:<40} | FAILED: {type(e).__name__}: {e}")
                ds_results[name] = {"error": str(e)}
        all_results[ds["name"]] = ds_results

    # Save
    out_path = "results/strengthen_exp1_hyperbolic_methods.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved -> {out_path}")

    # Conclusion summary
    print("\n" + "=" * 78)
    print(" CONCLUSION")
    print("=" * 78)
    for ds_name, ds_results in all_results.items():
        c4 = ds_results.get("C4 (linear probe on mean-pooled)", {})
        if "tpr" not in c4:
            continue
        print(f"\n  {ds_name}: C4 baseline TPR = {c4['tpr']:.3f}")
        any_beats = False
        for name, r in ds_results.items():
            if name == "C4 (linear probe on mean-pooled)" or "tpr" not in r:
                continue
            if r["tpr"] > c4["tpr"] + 0.01:
                any_beats = True
                print(f"    {name} beats C4 by {r['tpr']-c4['tpr']:+.3f}")
        if not any_beats:
            print(f"    No hyperbolic method beats C4 by more than 0.01 TPR.")
            print(f"    The 'hyperbolic doesn't help' claim is supported on "
                  f"{ds_name}.")


if __name__ == "__main__":
    main()
