"""
HPS Sentinel — Interactive Chat with Detection
═══════════════════════════════════════════════
Chat with the model. Each prompt is scored by the trained HPS-Full detector.

Usage:
  python chat.py                    # uses saved projection head
  python chat.py --train            # retrain projection head first

Type 'quit' or 'exit' to stop.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import config
from utils import (
    load_model, extract_activations,
    to_lorentz, hyperbolic_curvature, lorentz_distance,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  Projection Head (same as test5)
# ═══════════════════════════════════════════════════════════════════════════════

class LorentzProjection(nn.Module):
    def __init__(self, d_in, d_proj=256, k=1.0):
        super().__init__()
        self.proj = nn.Linear(d_in, d_proj, bias=False)
        self.scale = nn.Parameter(torch.tensor(1.0 / np.sqrt(d_proj)))
        self.k = k
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x):
        x_proj = self.proj(x) * self.scale
        norm_sq = (x_proj ** 2).sum(dim=-1, keepdim=True)
        x0 = torch.sqrt(1.0 / self.k + norm_sq)
        return torch.cat([x0, x_proj], dim=-1)


def lorentz_distance_torch(x, y, k=1.0):
    inner = -x[:, 0] * y[:, 0] + (x[:, 1:] * y[:, 1:]).sum(dim=-1)
    inner = torch.clamp(inner, max=-1.0 / k - 1e-6)
    return (1.0 / np.sqrt(k)) * torch.acosh(-k * inner)


def contrastive_loss(anchors, labels, k=1.0, margin=2.0):
    n = anchors.shape[0]
    loss = torch.tensor(0.0, device=anchors.device)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = lorentz_distance_torch(anchors[i:i+1], anchors[j:j+1], k=k).squeeze()
            if labels[i] == labels[j]:
                loss = loss + d ** 2
            else:
                loss = loss + torch.clamp(margin - d, min=0) ** 2
            count += 1
    return loss / max(count, 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Train or Load
# ═══════════════════════════════════════════════════════════════════════════════

PROJ_PATH = os.path.join(config.RESULTS_DIR, "hps_projection_head.pt")
THRESH_PATH = os.path.join(config.RESULTS_DIR, "hps_threshold.npy")


def train_detector(model, tokenizer):
    """Train the projection head and compute decision threshold."""
    from dataset import BENIGN, ADVERSARIAL
    import torch.optim as optim

    layers = config.TARGET_LAYERS
    prompts = BENIGN[:50] + ADVERSARIAL[:50]
    labels = np.array([0] * min(50, len(BENIGN)) + [1] * min(50, len(ADVERSARIAL)))

    print("[chat] Extracting training activations...")
    all_acts = []
    for i, p in enumerate(prompts):
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(prompts)}")
        all_acts.append(extract_activations(model, tokenizer, p, layers, config.DEVICE))

    d_hidden = all_acts[0][layers[0]].shape[0]
    n_layers = len(layers)
    X = np.zeros((len(prompts), n_layers, d_hidden))
    for i, act_dict in enumerate(all_acts):
        for j, l in enumerate(layers):
            if l in act_dict:
                X[i, j] = act_dict[l]

    device = config.DEVICE
    proj = LorentzProjection(d_hidden, 256, config.HYPERBOLIC_K).to(device)
    optimizer = optim.Adam(proj.parameters(), lr=1e-3, weight_decay=1e-5)

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(labels, dtype=torch.long, device=device)

    print("[chat] Training projection head (80 epochs)...")
    proj.train()
    mid = n_layers // 2
    for epoch in range(80):
        h = proj(X_t[:, mid, :])
        loss = contrastive_loss(h, y_t, k=config.HYPERBOLIC_K)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/80 — loss: {loss.item():.4f}")

    # Compute scores and find threshold
    proj.eval()
    with torch.no_grad():
        all_scores = []
        for i in range(len(prompts)):
            score = _compute_score(proj, X[i], layers, device)
            all_scores.append(score)

    all_scores = np.array(all_scores)
    benign_scores = all_scores[labels == 0]
    adv_scores = all_scores[labels == 1]

    # Threshold: maximize separation (midpoint between max benign and min adversarial)
    threshold = (benign_scores.max() + adv_scores.min()) / 2

    print(f"[chat] Benign scores:  mean={benign_scores.mean():.3f}, max={benign_scores.max():.3f}")
    print(f"[chat] Adversarial:    mean={adv_scores.mean():.3f}, min={adv_scores.min():.3f}")
    print(f"[chat] Threshold: {threshold:.3f}")

    # Save
    torch.save(proj.state_dict(), PROJ_PATH)
    np.save(THRESH_PATH, threshold)
    print(f"[chat] Saved projection head → {PROJ_PATH}")

    return proj, threshold


def _compute_score(proj, act_array, layers, device):
    """Compute anomaly score for a single prompt's activations."""
    n_layers = len(layers)
    d_hidden = act_array.shape[-1] if act_array.ndim == 2 else len(act_array[layers[0]])

    if act_array.ndim == 1:
        return 0.0

    with torch.no_grad():
        x = torch.tensor(act_array, dtype=torch.float32, device=device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        h = proj(x)  # (n_layers, d_proj+1)
        h_np = h.cpu().numpy()

    # Features: radius at mid layer + curvature + displacement
    radii = h_np[:, 0]
    curv = hyperbolic_curvature([h_np[j] for j in range(len(h_np))], k=config.HYPERBOLIC_K)

    # Score = combination of radius (higher = more suspicious) and curvature
    score = np.mean(radii) + 2.0 * (curv.max() if len(curv) > 0 else 0)
    return float(score)


def score_prompt(proj, model, tokenizer, prompt, layers, device, pool_mode="last"):
    """Score a single prompt in real-time."""
    # Import extract_all_layers from experiment7 if available, else use basic extraction
    try:
        from experiment7 import extract_all_layers
        act_dict = extract_all_layers(model, tokenizer, prompt, device, pool_mode)
    except ImportError:
        act_dict = extract_activations(model, tokenizer, prompt, layers, device)

    n_layers = len(layers)
    d_hidden = act_dict[layers[0]].shape[0]
    X = np.zeros((n_layers, d_hidden))
    for j, l in enumerate(layers):
        if l in act_dict:
            X[j] = act_dict[l]

    return _compute_score(proj, X, layers, device)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Chat Loop
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Force retrain projection head")
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"  HPS Sentinel — Interactive Chat")
    print(f"  Model: {config.MODEL_NAME}")
    print(f"{'═'*60}\n")

    model, tokenizer = load_model(config.MODEL_NAME, config.DEVICE, config.DTYPE)
    layers = config.TARGET_LAYERS
    pool_mode = "last"  # default, overridden by checkpoint if available

    # Load or train projection head
    if args.train or not os.path.exists(PROJ_PATH):
        proj, threshold = train_detector(model, tokenizer)
    else:
        # Load from experiment6/7
        checkpoint = torch.load(PROJ_PATH, map_location=config.DEVICE)
        d_in = checkpoint["d_in"]
        d_proj = checkpoint["d_proj"]
        k = checkpoint["k"]
        proj = LorentzProjection(d_in, d_proj, k).to(config.DEVICE)
        proj.load_state_dict(checkpoint["state_dict"])
        proj.eval()
        threshold = float(np.load(THRESH_PATH))
        # Override layers/pool if saved by experiment7
        if "layers" in checkpoint:
            layers = checkpoint["layers"]
        if "pool_mode" in checkpoint:
            pool_mode = checkpoint["pool_mode"]
        else:
            pool_mode = "last"
        print(f"[chat] Loaded (d={d_in}→{d_proj}, layers={layers}, pool={pool_mode}, threshold={threshold:.3f})")

    print(f"\n{'─'*60}")
    print("  Type a prompt. The sentinel will score it.")
    print("  Type 'quit' to exit.")
    print(f"{'─'*60}\n")

    while True:
        try:
            prompt = input("\033[1mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not prompt or prompt.lower() in ("quit", "exit", "q"):
            break

        # Score
        score = score_prompt(proj, model, tokenizer, prompt, layers, config.DEVICE, pool_mode)
        is_adversarial = score > threshold

        # Display
        if is_adversarial:
            status = "\033[91m⚠ BLOCKED — Adversarial detected\033[0m"
        else:
            status = "\033[92m✓ Safe\033[0m"

        bar_len = 30
        normalized = min(score / (threshold * 2), 1.0)
        filled = int(normalized * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        print(f"  [{bar}] score={score:.3f}  threshold={threshold:.3f}  {status}")

        # Generate response if safe
        if not is_adversarial:
            inputs = tokenizer(prompt, return_tensors="pt").to(config.DEVICE)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=150, do_sample=True,
                    temperature=0.7, top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"\033[1mAssistant:\033[0m {response.strip()}\n")
        else:
            print(f"\033[1mAssistant:\033[0m I cannot help with that request.\n")


if __name__ == "__main__":
    main()
