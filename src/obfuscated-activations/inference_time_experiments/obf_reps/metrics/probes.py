import torch
from sparsify import Sae  # For Eleuther SAEs for Llama-3-8b


class LogisticRegression(torch.nn.Module):
    """Differentiable Logistic Regression."""

    def __init__(self, input_dim, dtype=torch.float16):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1, dtype=dtype)

    def forward(self, x):
        return self.linear(x)


class SAEClassifier(torch.nn.Module):
    def __init__(self, sae):
        super().__init__()
        self.encoder = sae
        self.linear = torch.nn.Linear(self.encoder.num_latents, 1)

    def forward(self, hidden_states):
        return self.linear(self.encoder.encode_no_topk(hidden_states))


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dtype=torch.float16):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim, dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1, dtype=dtype),
        )

    def forward(self, x):
        return self.model(x)


class VAE(torch.nn.Module):
    """Simple VAE with MLP encoder and decoder.

    Adapted from PyTorch VAE (Apache-2.0) but with a different architecture.
    https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
    Original copyright notice:
                             Apache License
                       Version 2.0, January 2004
                    http://www.apache.org/licenses/
                    Copyright Anand Krishnamoorthy Subramanian 2020
                               anandkrish894@gmail.com
    """

    def __init__(self, input_dim: int, latent_dim: int, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dtype = dtype

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 2 * input_dim, dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * input_dim, 2 * self.latent_dim, dtype=dtype),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, 2 * input_dim, dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * input_dim, input_dim, dtype=dtype),
        )

    def encode(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes the input by passing through the encoder network and returns the latent codes.

        :param input: (Tensor) Input tensor to encoder [N x C] or [N x T x C] where N is batch
            size, T is the number of time steps (optional position dimension), and C is the number
            of channels.
        :return: (Tensor) Tuple of latent codes (mu, log_var), shaped as input but with C =
            latent_dim
        """
        original_shape = input.shape

        input = input.to(self.dtype)

        if input.ndim == 2:
            # Input is already 2D, use as is
            reshaped_input = input
        elif input.ndim == 3:
            # Reshape 3D input to 2D: [N*T x C]
            N, T, C = input.shape
            reshaped_input = input.reshape(-1, C)
        else:
            raise ValueError(f"Input must be 2D or 3D, got shape {original_shape}")

        # Pass through encoder
        result = self.encoder(reshaped_input)

        assert result.ndim == 2, "Encoder output must be 2-dimensional"
        assert (
            result.shape[1] == 2 * self.latent_dim
        ), f"Encoder output shape mismatch. Expected {2 * self.latent_dim} features, got {result.shape[1]}"

        # Split the result into mu and log_var components
        mu = result[:, : self.latent_dim]
        log_var = result[:, self.latent_dim :]

        # Reshape mu and log_var to match input shape
        if input.ndim == 3:
            mu = mu.reshape(N, T, self.latent_dim)
            log_var = log_var.reshape(N, T, self.latent_dim)

        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Maps the given latent codes onto the image space.

        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from N(mu, var) from N(0,1).

        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(
        self, input: torch.Tensor, noise: bool = True, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        mu, log_var = self.encode(input)
        if noise:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
        return self.decode(z), mu, log_var

    def loss_function(
        self, reconstruction, input, mu, log_var, kld_weight=1.0, reduce: bool = True
    ) -> dict[str, torch.Tensor]:
        """Computes the VAE loss function.

        KL(N(\\mu, \\sigma), N(0, 1)) =
        \\log \frac{1}{\\sigma} + \frac{\\sigma^2 + \\mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        assert input.ndim == 2
        assert reconstruction.ndim == 2
        assert mu.ndim == 2
        assert log_var.ndim == 2

        input = input.to(self.dtype)

        recons_loss = torch.nn.functional.mse_loss(reconstruction, input, reduction="none")
        # Reduce over all but first dimension
        recons_loss = recons_loss.view(recons_loss.shape[0], -1).mean(dim=1)

        kld_loss = -0.5 * torch.sum(
            1 + log_var - mu**2 - log_var.exp(), dim=tuple(range(1, mu.dim()))
        )

        if reduce:
            recons_loss = recons_loss.mean()
            kld_loss = kld_loss.mean()

        loss = recons_loss + kld_weight * kld_loss

        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": kld_loss.detach(),
        }


# ============================================================
#  HPS (Hyperbolic Projection Sentinel) primitives
#  Ported from the HPS codebase (hps_core.py / adaptive_attack.py).
#  Differentiable so the obfuscation attack can backprop through them.
# ============================================================
import math


class LorentzProjection(torch.nn.Module):
    """Learned linear projection of activations onto the Lorentz hyperboloid."""

    def __init__(self, d_in, d_proj=64, k_init=0.1):
        super().__init__()
        self.proj = torch.nn.Linear(d_in, d_proj, bias=False)
        self.scale = torch.nn.Parameter(torch.tensor(1.0 / math.sqrt(d_proj)))
        self.log_k = torch.nn.Parameter(torch.tensor(float(math.log(k_init))))
        torch.nn.init.xavier_uniform_(self.proj.weight)

    @property
    def k(self):
        return torch.exp(self.log_k).clamp(min=0.05, max=10.0)

    def forward(self, x):
        x = x.float()
        xp = self.proj(x) * self.scale
        x0 = torch.sqrt(1.0 / self.k + (xp**2).sum(dim=-1, keepdim=True))
        return torch.cat([x0, xp], dim=-1)


def lorentz_distance(x, y, k: float):
    """Geodesic distance on the Lorentz hyperboloid. Broadcasts over leading dims.
    x, y: (..., 1 + d_proj) Lorentz coordinates. k: scalar float."""
    inner = -(x[..., 0] * y[..., 0]) + (x[..., 1:] * y[..., 1:]).sum(dim=-1)
    # acosh domain is [1, inf). Clamp the ARGUMENT (not `inner`) to 1+eps with an
    # fp32-safe margin, otherwise rounding can push it just below 1.0 -> NaN.
    arg = torch.clamp(-k * inner, min=1.0 + 1e-5)
    return (1.0 / math.sqrt(k)) * torch.acosh(arg)


def hps_trajectory_features(z, k: float):
    """12 trajectory features computed ALONG axis 0 (the layer trajectory).

    z: (n_layers, batch, 1 + d_proj) Lorentz coordinates.
    Returns: (batch, 12) = 5 radial + 4 curvature + 3 displacement.
    """
    radial = z[..., 0]  # (n_layers, batch)
    n = z.shape[0]
    mean_r = radial.mean(0)
    max_r = radial.max(0).values
    min_r = radial.min(0).values
    std_r = radial.std(0, unbiased=False)
    range_r = max_r - min_r
    if n < 3:
        zero = torch.zeros_like(mean_r)
        max_k = mean_k = std_k = spike = zero
    else:
        ks = []
        for i in range(1, n - 1):
            d_prev = lorentz_distance(z[i - 1], z[i], k)
            d_next = lorentz_distance(z[i], z[i + 1], k)
            d_skip = lorentz_distance(z[i - 1], z[i + 1], k)
            ks.append((d_prev + d_next - d_skip).clamp(min=0))
        ks = torch.stack(ks, dim=0)
        max_k = ks.max(0).values
        mean_k = ks.mean(0)
        std_k = ks.std(0, unbiased=False)
        spike = ks.argmax(0).float() / max(ks.shape[0] - 1, 1)
    disp = lorentz_distance(z[0], z[-1], k)
    path = sum(lorentz_distance(z[i], z[i + 1], k) for i in range(n - 1))
    prog = disp / (path + 1e-9)
    return torch.stack(
        [mean_r, max_r, min_r, std_r, range_r,
         max_k, mean_k, std_k, spike, disp, path, prog], dim=-1
    )
