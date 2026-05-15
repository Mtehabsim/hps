"""
HPS Sentinel - Shared Utilities
Covers: model loading, activation extraction, hyperbolic operations, curvature.
"""

import json
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(model_name: str, device: str, dtype):
    """Load a HuggingFace causal LM and its tokenizer."""
    print(f"[utils] Loading model: {model_name} on {device} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check if 4-bit quantization is requested
    use_4bit = getattr(__import__('config'), 'USE_4BIT', False)

    if use_4bit:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=None,
            trust_remote_code=True,
        )
        model = model.to(device)

    model.eval()
    print(f"[utils] Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
#  ACTIVATION EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _get_transformer_layers(model):
    """Return the list of transformer decoder layers, model-agnostic."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers          # Llama, Qwen, Gemma, Mistral
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h         # GPT-2 style
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers       # Pythia / GPT-NeoX
    raise ValueError(
        "Cannot find transformer layers. Supported: model.model.layers, "
        "model.transformer.h, model.gpt_neox.layers"
    )


def extract_activations(
    model,
    tokenizer,
    prompt: str,
    layer_indices: list[int],
    device: str,
) -> dict[int, np.ndarray]:
    """
    Run one forward pass and return the last-token hidden state at each
    requested layer index.

    Returns
    -------
    dict mapping layer_idx → numpy array of shape (hidden_dim,)
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    captured: dict[int, np.ndarray] = {}
    hooks = []
    transformer_layers = _get_transformer_layers(model)

    def make_hook(idx: int):
        def hook(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            # Shape: (batch=1, seq_len, hidden_dim) → take last token
            captured[idx] = hidden[0, -1, :].detach().cpu().float().numpy()
        return hook

    for idx in layer_indices:
        if idx < len(transformer_layers):
            hooks.append(transformer_layers[idx].register_forward_hook(make_hook(idx)))

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    return {idx: captured[idx] for idx in layer_indices if idx in captured}


def extract_activations_batch(
    model,
    tokenizer,
    prompts: list[str],
    layer_indices: list[int],
    device: str,
) -> list[dict[int, np.ndarray]]:
    """Run extract_activations for every prompt. Returns a list of dicts."""
    results = []
    for i, p in enumerate(prompts):
        print(f"  Extracting {i+1}/{len(prompts)}: {p[:60]}…")
        results.append(extract_activations(model, tokenizer, p, layer_indices, device))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  POINCARÉ BALL OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def project_to_poincare(x: np.ndarray, c: float = 1.0, max_norm: float = 0.95) -> np.ndarray:
    """
    Project a Euclidean vector onto the Poincaré ball of curvature c.
    Uses the exponential map at the origin: expmap_0(v) = tanh(√c·‖v‖/2) · v/‖v‖·√c

    CRITICAL: We scale x by 1/sqrt(d) BEFORE computing the norm to prevent
    all vectors from saturating to the same radius. This preserves relative
    magnitude differences between vectors.
    """
    sqrt_c = np.sqrt(c)
    # Scale by 1/sqrt(dim) to keep norms in a reasonable range for tanh
    d = len(x)
    x_scaled = x / np.sqrt(d)
    norm = np.linalg.norm(x_scaled)
    if norm < 1e-10:
        return np.zeros_like(x_scaled)
    unit = x_scaled / norm
    r = np.tanh(sqrt_c * norm / 2) / sqrt_c
    r = min(r, max_norm / sqrt_c)
    return r * unit


def poincare_radius(x: np.ndarray) -> float:
    """Euclidean norm of a point already on the Poincaré ball (proxy for depth)."""
    return float(np.linalg.norm(x))


def poincare_distance(x: np.ndarray, y: np.ndarray, c: float = 1.0) -> float:
    """Geodesic distance between two points in the Poincaré ball."""
    sqrt_c = np.sqrt(c)
    x_sq = np.dot(x, x)
    y_sq = np.dot(y, y)
    diff_sq = np.dot(x - y, x - y)
    denom = (1 - c * x_sq) * (1 - c * y_sq)
    if denom < 1e-10:
        return float("inf")
    arg = 1 + 2 * c * diff_sq / denom
    arg = max(arg, 1.0 + 1e-10)
    return float((2 / sqrt_c) * np.arctanh(sqrt_c * np.sqrt((arg - 1) / (arg + 1))))


# ═══════════════════════════════════════════════════════════════════════════════
#  LORENTZ (HYPERBOLOID) OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def to_lorentz(x: np.ndarray, k: float = 1.0) -> np.ndarray:
    """
    Lift a Euclidean vector x ∈ R^d onto the Lorentz hyperboloid L^d_k.
    The hyperboloid satisfies <v, v>_L = -1/k.
    Returns a (d+1)-dimensional vector where index 0 is the time coordinate.

    CRITICAL: Scale x by 1/sqrt(d) to preserve relative magnitude differences.
    Without this, all high-dimensional vectors have similar norms (~sqrt(d))
    and collapse to the same region on the hyperboloid.
    """
    d = len(x)
    x_scaled = x / np.sqrt(d)
    norm_sq = np.dot(x_scaled, x_scaled)
    x0 = np.sqrt(1.0 / k + norm_sq)          # time coordinate
    return np.concatenate([[x0], x_scaled])


def lorentz_inner(x: np.ndarray, y: np.ndarray) -> float:
    """Minkowski inner product: ⟨x, y⟩_L = -x₀y₀ + x₁y₁ + … + xₙyₙ"""
    return float(-x[0] * y[0] + np.dot(x[1:], y[1:]))


def lorentz_distance(x: np.ndarray, y: np.ndarray, k: float = 1.0) -> float:
    """Geodesic distance on the Lorentz hyperboloid."""
    inner = lorentz_inner(x, y)
    # Clamp: ⟨x,y⟩_L ≤ -1/k for points on the manifold
    inner = min(inner, -1.0 / k - 1e-8)
    return float((1.0 / np.sqrt(k)) * np.arccosh(-k * inner))


# ═══════════════════════════════════════════════════════════════════════════════
#  CURVATURE ALONG A TRAJECTORY
# ═══════════════════════════════════════════════════════════════════════════════

def euclidean_curvature(points: list[np.ndarray]) -> np.ndarray:
    """
    Discrete Frenet-Serret curvature along a sequence of Euclidean points.
    κ_i = ‖v_i × a_i‖ / ‖v_i‖³   (generalised to any dimension)

    Returns an array of length len(points) - 2.
    """
    kappas = []
    for i in range(1, len(points) - 1):
        v = points[i]     - points[i - 1]
        a = points[i + 1] - 2 * points[i] + points[i - 1]
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-8:
            kappas.append(0.0)
            continue
        cross_sq = np.dot(v, v) * np.dot(a, a) - np.dot(v, a) ** 2
        kappas.append(float(np.sqrt(max(cross_sq, 0.0))) / v_norm ** 3)
    return np.array(kappas)


def hyperbolic_curvature(lorentz_pts: list[np.ndarray], k: float = 1.0) -> np.ndarray:
    """
    Discrete curvature using Lorentz geodesic distances.
    Measures the triangle-inequality deviation at each interior point.
    """
    kappas = []
    for i in range(1, len(lorentz_pts) - 1):
        d_prev = lorentz_distance(lorentz_pts[i], lorentz_pts[i - 1], k)
        d_next = lorentz_distance(lorentz_pts[i + 1], lorentz_pts[i],     k)
        d_span = lorentz_distance(lorentz_pts[i + 1], lorentz_pts[i - 1], k)
        denom  = d_prev + d_next
        kappas.append(0.0 if denom < 1e-8 else float(abs(d_prev + d_next - d_span) / denom))
    return np.array(kappas)


def compute_displacement(points: list[np.ndarray]) -> tuple[float, float]:
    """
    Returns (total_displacement, path_length).
    displacement = straight-line distance from first to last point.
    path_length  = sum of consecutive step lengths.
    """
    disp = float(np.linalg.norm(points[-1] - points[0]))
    path = float(sum(np.linalg.norm(points[i + 1] - points[i]) for i in range(len(points) - 1)))
    return disp, path


# ═══════════════════════════════════════════════════════════════════════════════
#  I/O HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def save_json(data, filename: str, directory: str) -> str:
    path = os.path.join(directory, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"[utils] Saved → {path}")
    return path


def load_json(filename: str, directory: str):
    path = os.path.join(directory, filename)
    with open(path) as f:
        return json.load(f)
