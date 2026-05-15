"""
HPS Sentinel - Phase 1 Configuration
Target: Vicuna-13B-v1.5 (40 layers)
"""

import os
import torch

# ── GPU ───────────────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "lmsys/vicuna-13b-v1.5"

# Fallback layers (evenly spaced for 40-layer model). Overridden by experiment7 discovery.
TARGET_LAYERS = [3, 8, 13, 18, 23, 28, 33, 38]

# ── Hardware ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if torch.cuda.is_available() else torch.float32

# ── Hyperbolic geometry ───────────────────────────────────────────────────────
HYPERBOLIC_K = 1.0        # initial curvature (learnable during training)
PROJECTION_DIM = 64       # d_p: 5120 → 64 (80x compression, ~330K params)
MAX_NORM = 0.95           # Poincaré ball clamp

# ── Output directories ────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)
