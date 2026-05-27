"""
Lorentz manifold operations for hyperbolic neural networks.

Self-contained implementation. No external geometry libraries required.

The Lorentz model represents hyperbolic space H^n as a hyperboloid embedded
in (n+1)-dimensional Minkowski space:
    H^n = {x in R^(n+1) : <x,x>_L = -1/k, x_0 > 0}
where k is the curvature parameter.

Lorentz inner product:
    <x, y>_L = -x_0 * y_0 + sum(x_i * y_i for i=1..n)

Geodesic distance:
    d(x, y) = (1/sqrt(k)) * arccosh(-k * <x, y>_L)

The "time coordinate" x_0 = sqrt(1/k + ||x_spatial||^2) gives the radial
position; smaller x_0 = closer to the origin = more "central".
"""

import torch
import torch.nn as nn
import numpy as np


def lorentz_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Lorentz inner product: -x_0*y_0 + sum_i x_i * y_i.

    Args:
        x, y: tensors of shape (..., d+1) where index 0 is the time component.

    Returns:
        Lorentz inner product of shape (...,).
    """
    return -x[..., 0:1] * y[..., 0:1] + (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)


def lorentz_distance(x: torch.Tensor, y: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """Geodesic distance on the Lorentz hyperboloid.

    Args:
        x, y: points on the hyperboloid, shape (..., d+1).
        k: curvature parameter.

    Returns:
        Distance, shape (...,).
    """
    inner = lorentz_inner(x, y).squeeze(-1)
    # Argument to arccosh must be >= 1; clamp for numerical safety.
    arg = (-k * inner).clamp(min=1.0 + 1e-7)
    return torch.acosh(arg) / np.sqrt(k)


def to_hyperboloid(x_spatial: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """Lift a Euclidean vector to the Lorentz hyperboloid.

    Given x_spatial in R^d, computes x_0 such that <x,x>_L = -1/k:
        x_0 = sqrt(1/k + ||x_spatial||^2)

    Args:
        x_spatial: shape (..., d).
        k: curvature.

    Returns:
        Point on hyperboloid, shape (..., d+1) with x[0] = time coordinate.
    """
    x_0 = torch.sqrt(1.0 / k + (x_spatial ** 2).sum(dim=-1, keepdim=True))
    return torch.cat([x_0, x_spatial], dim=-1)


def exp_map(v: torch.Tensor, x: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """Lorentz exponential map.

    Maps a tangent vector v at point x to a point on the hyperboloid.
    Used to do "addition" in hyperbolic space.

    Args:
        v: tangent vector at x, shape (..., d+1).
        x: base point on the hyperboloid, shape (..., d+1).
        k: curvature.

    Returns:
        Point on hyperboloid, shape (..., d+1).
    """
    v_norm = torch.sqrt(lorentz_inner(v, v).clamp(min=1e-9)).squeeze(-1).unsqueeze(-1)
    sqrt_k = np.sqrt(k)
    return torch.cosh(sqrt_k * v_norm) * x + torch.sinh(sqrt_k * v_norm) * v / (sqrt_k * v_norm + 1e-9)


def log_map(y: torch.Tensor, x: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """Lorentz logarithmic map.

    Inverse of exp_map. Computes the tangent vector at x that points to y.

    Args:
        y: target point on hyperboloid, shape (..., d+1).
        x: base point on hyperboloid, shape (..., d+1).
        k: curvature.

    Returns:
        Tangent vector at x, shape (..., d+1).
    """
    inner = lorentz_inner(x, y).squeeze(-1).unsqueeze(-1)
    arg = (-k * inner).clamp(min=1.0 + 1e-7)
    dist = torch.acosh(arg) / np.sqrt(k)
    diff = y + k * inner * x
    diff_norm = torch.sqrt(lorentz_inner(diff, diff).clamp(min=1e-9)).squeeze(-1).unsqueeze(-1)
    return dist * diff / (diff_norm + 1e-9)


def hyperbolic_origin(d: int, k: float = 1.0, device: str = "cuda") -> torch.Tensor:
    """The origin of the Lorentz hyperboloid: (1/sqrt(k), 0, 0, ..., 0)."""
    o = torch.zeros(d + 1, device=device)
    o[0] = 1.0 / np.sqrt(k)
    return o


class LorentzProjection(nn.Module):
    """Standard projection from Euclidean R^d to Lorentz H^d_proj.

    This is the same projection used by HPS in experiment7.py.
    """

    def __init__(self, d_in: int, d_proj: int, k_init: float = 1.0,
                 freeze_kappa: bool = True):
        super().__init__()
        self.proj = nn.Linear(d_in, d_proj, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(1.0 / 8.0))
        self.log_k = nn.Parameter(torch.tensor(float(np.log(k_init))))
        if freeze_kappa:
            self.log_k.requires_grad = False

    @property
    def k(self) -> torch.Tensor:
        return torch.exp(self.log_k).clamp(0.01, 100.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_spatial = self.proj(x) * self.scale
        return to_hyperboloid(h_spatial, k=self.k.item())
