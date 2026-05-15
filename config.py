"""
HPS Sentinel - Phase 1 Configuration
Target: Llama-3-8B-Instruct (32 layers, safety-aligned)
"""

import os
import torch

# ── GPU ───────────────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# 32 layers → sample 8 points spread across depth
TARGET_LAYERS = [2, 6, 10, 14, 18, 22, 26, 30]

# ── Hardware ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if torch.cuda.is_available() else torch.float32

# ── Hyperbolic geometry ───────────────────────────────────────────────────────
HYPERBOLIC_K = 1.0   # curvature constant (positive)
MAX_NORM     = 0.95  # Poincaré ball clamp

# ── Output directories ────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)
