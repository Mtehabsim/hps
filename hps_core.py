"""
hps_core.py — Self-contained HPS primitives without transformer dependencies.

Extracted from experiment7.py and utils.py so analysis scripts (statistical
tests, radial distribution check, etc.) can run without loading the LLM.

Provides:
  - LorentzProjection (nn.Module): learned projection to Lorentz hyperboloid
  - lorentz_distance: geodesic distance on hyperboloid (numpy)
  - hyperbolic_curvature: discrete curvature along a trajectory (numpy)
  - contrastive_loss: per-layer contrastive loss in Lorentz space (torch)
  - extract_trajectory_features: compute the 12-dim trajectory features
"""

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Numpy primitives (geometric calculations)
# ---------------------------------------------------------------------------

def lorentz_inner(x: np.ndarray, y: np.ndarray) -> float:
    """Minkowski inner product: ⟨x, y⟩_L = -x₀y₀ + x₁y₁ + ... + xₙyₙ"""
    return float(-x[0] * y[0] + np.dot(x[1:], y[1:]))


def lorentz_distance(x: np.ndarray, y: np.ndarray, k: float = 1.0) -> float:
    """Geodesic distance on the Lorentz hyperboloid with curvature k."""
    inner = lorentz_inner(x, y)
    inner = min(inner, -1.0 / k - 1e-8)
    return float((1.0 / np.sqrt(k)) * np.arccosh(-k * inner))


def hyperbolic_curvature(lorentz_pts, k: float = 1.0) -> np.ndarray:
    """
    Discrete curvature using Lorentz geodesic distances.
    Measures the triangle-inequality deviation at each interior point.
    """
    kappas = []
    for i in range(1, len(lorentz_pts) - 1):
        d_prev = lorentz_distance(lorentz_pts[i], lorentz_pts[i - 1], k)
        d_next = lorentz_distance(lorentz_pts[i + 1], lorentz_pts[i], k)
        d_span = lorentz_distance(lorentz_pts[i + 1], lorentz_pts[i - 1], k)
        denom = d_prev + d_next
        kappas.append(
            0.0 if denom < 1e-8
            else float(abs(d_prev + d_next - d_span) / denom)
        )
    return np.array(kappas)


# ---------------------------------------------------------------------------
# Torch module: Lorentz projection
# ---------------------------------------------------------------------------

class LorentzProjection(nn.Module):
    """Learned linear projection of LLM activations onto the Lorentz hyperboloid.

    Args:
        d_in: input dimension (LLM hidden dim)
        d_proj: projection dimension (default 64)
        k: initial curvature (default 1.0; learnable via log_k)
        n_layers: number of layers (for per-layer temperature)
    """

    def __init__(self, d_in, d_proj=64, k=1.0, n_layers=8):
        super().__init__()
        self.proj = nn.Linear(d_in, d_proj, bias=False)
        self.scale = nn.Parameter(torch.tensor(1.0 / np.sqrt(d_proj)))
        self.log_k = nn.Parameter(torch.tensor(np.log(k)))
        self.log_tau = nn.Parameter(torch.zeros(n_layers))
        nn.init.xavier_uniform_(self.proj.weight)

    @property
    def k(self):
        return torch.exp(self.log_k).clamp(min=0.1, max=10.0)

    def tau(self, layer_idx):
        return torch.exp(self.log_tau[layer_idx]).clamp(min=0.01, max=10.0)

    def forward(self, x):
        x_fp32 = x.float()
        x_proj = self.proj(x_fp32) * self.scale
        norm_sq = (x_proj ** 2).sum(dim=-1, keepdim=True)
        x0 = torch.sqrt(1.0 / self.k + norm_sq)
        return torch.cat([x0, x_proj], dim=-1)


# ---------------------------------------------------------------------------
# Contrastive loss
# ---------------------------------------------------------------------------

def contrastive_loss(anchors, labels, k=1.0, margin=2.0, tau=1.0):
    """Vectorized contrastive loss with balanced weighting and per-layer τ."""
    n = anchors.shape[0]
    anchors = anchors.float()
    inner = -anchors[:, 0:1] @ anchors[:, 0:1].T \
        + anchors[:, 1:] @ anchors[:, 1:].T

    if isinstance(k, torch.Tensor):
        clamp_val = (-1.0 / k - 1e-6).detach().item()
        inner = torch.clamp(inner, max=clamp_val)
        dists = (1.0 / torch.sqrt(k)) * torch.acosh(-k * inner)
    else:
        inner = torch.clamp(inner, max=-1.0 / k - 1e-6)
        dists = (1.0 / np.sqrt(k)) * torch.acosh(-k * inner)

    dists = dists / tau

    same_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    diff_mask = 1.0 - same_mask
    triu = torch.triu(torch.ones(n, n, device=anchors.device), diagonal=1)

    same_loss = (dists ** 2 * same_mask * triu).sum()
    diff_loss = (torch.clamp(margin - dists, min=0) ** 2 * diff_mask * triu).sum()

    n_same = (same_mask * triu).sum().clamp(min=1)
    n_diff = (diff_mask * triu).sum().clamp(min=1)
    return (same_loss / n_same + diff_loss / n_diff) / 2.0


# ---------------------------------------------------------------------------
# Trajectory features (12-dim)
# ---------------------------------------------------------------------------

def extract_trajectory_features(proj, X_all, k=None):
    """
    Compute 12-dim trajectory features per sample:
      Radial (5): mean, max, min, std, range of x_0
      Curvature (4): max, mean, std of triangle-inequality bending,
                     argmax position
      Displacement (3): start-end distance, path length, ratio
    """
    if k is None:
        k = proj.k.item()
    device = next(proj.parameters()).device
    n_samples, n_layers, d_hidden = X_all.shape
    features = []
    proj.eval()
    with torch.no_grad():
        for i in range(n_samples):
            x = torch.tensor(X_all[i], dtype=torch.float32, device=device)
            h = proj(x)
            h_np = h.cpu().numpy()
            radii = h_np[:, 0]
            curv = hyperbolic_curvature(
                [h_np[j] for j in range(n_layers)], k=k)
            d_total = lorentz_distance(h_np[0], h_np[-1], k=k)
            path_len = sum(
                lorentz_distance(h_np[j], h_np[j+1], k=k)
                for j in range(n_layers - 1)
            )
            feat = np.array([
                np.mean(radii),
                np.max(radii),
                np.min(radii),
                np.std(radii),
                float(np.max(radii) - np.min(radii)),
                float(curv.max()) if len(curv) > 0 else 0,
                float(curv.mean()) if len(curv) > 0 else 0,
                float(curv.std()) if len(curv) > 0 else 0,
                float(np.argmax(curv) / max(len(curv), 1)) if len(curv) > 0 else 0,
                d_total,
                path_len,
                d_total / (path_len + 1e-8),
            ])
            features.append(feat)
    return np.array(features)
