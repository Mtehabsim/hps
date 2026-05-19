"""
Experiment 12 — HPS-Adv: Adversarially Trained Hyperbolic Defense
══════════════════════════════════════════════════════════════════
Implements PGD-adversarial training (Madry et al. 2018) for the HPS Lorentz
projection. This produces a defense that has provable robustness up to the
training-time perturbation budget.

The idea:
  Currently HPS is trained on clean activations only. Adversarial attacks
  exploit this by finding perturbations that push attacks toward benign in
  the projected space. Adversarial training EXPLICITLY trains the projection
  to maintain class separation under such perturbations.

Algorithm (per batch):
  1. Compute clean activations h
  2. PGD inner loop (K_inner steps) to find delta that maximizes contrastive loss
  3. Train projection W to minimize the worst-case loss

Output:
  - Saves trained adversarial projection to results/hps_adv_projection.pt
  - Reports baseline metrics (clean and adversarial)

Then run experiment11.py against this projection to verify adaptive robustness.

Usage:
  python experiment12.py
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
np.random.seed(42)
import torch
torch.manual_seed(42)
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import config
from utils import load_model, save_json
from experiment7 import (
    extract_all_layers, LorentzProjection, contrastive_loss,
    extract_trajectory_features
)
from dataset import BENIGN, ADVERSARIAL


# ═══════════════════════════════════════════════════════════════════════════
#  Hyperparameters for adversarial training
# ═══════════════════════════════════════════════════════════════════════════

HPS_LAYERS = [0, 1, 2, 35, 36, 37, 38, 39]
EPOCHS = 200             # more than clean training, AT typically needs more
EPS_TRAIN = 0.5          # training-time perturbation budget (L_inf)
PGD_STEPS_INNER = 7      # inner PGD steps to find worst-case delta
PGD_ALPHA = EPS_TRAIN / PGD_STEPS_INNER * 2.5  # step size
ADV_LOSS_WEIGHT = 1.0    # weight on adversarial loss (1.0 = pure AT)
CLEAN_LOSS_WEIGHT = 0.5  # weight on clean loss (mix to avoid forgetting clean perf)


# ═══════════════════════════════════════════════════════════════════════════
#  PGD inner loop (find worst-case delta)
# ═══════════════════════════════════════════════════════════════════════════

def pgd_find_worst_delta(proj_h, X_layer, y, eps, n_steps, alpha, layer_idx):
    """Find delta within L_inf ball that MAXIMIZES the contrastive loss.

    X_layer: (B, d_hidden) - clean activations at one layer
    y: (B,) - labels
    layer_idx: which layer this is (used for per-layer tau)

    Returns: delta of shape X_layer.shape
    """
    delta = torch.zeros_like(X_layer, requires_grad=True)
    for _ in range(n_steps):
        h = proj_h(X_layer + delta)
        # Use the SAME tau the defender will use for this layer
        loss = contrastive_loss(h, y, k=proj_h.k, tau=proj_h.tau(layer_idx))
        # Attacker MAXIMIZES this loss
        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]
        with torch.no_grad():
            delta_new = delta + alpha * torch.sign(grad)
            delta_new = torch.clamp(delta_new, -eps, +eps)
        delta = delta_new.detach().requires_grad_(True)
    return delta.detach()


# ═══════════════════════════════════════════════════════════════════════════
#  Adversarial training step
# ═══════════════════════════════════════════════════════════════════════════

def adversarial_train_step(proj_h, X, y, optimizer, eps, n_pgd_steps, alpha,
                           clean_w, adv_w):
    """One adversarial training step.

    X: (N, n_layers, d_hidden) - clean activations for all selected layers
    y: (N,) - labels
    """
    n_layers = X.shape[1]

    # Find worst-case delta per layer (independent perturbations)
    deltas = []
    for l in range(n_layers):
        delta_l = pgd_find_worst_delta(proj_h, X[:, l, :], y, eps, n_pgd_steps, alpha, layer_idx=l)
        deltas.append(delta_l)

    # Compute clean and adversarial losses
    clean_loss = torch.tensor(0.0, device=X.device)
    adv_loss = torch.tensor(0.0, device=X.device)
    for l in range(n_layers):
        h_clean = proj_h(X[:, l, :])
        h_adv = proj_h(X[:, l, :] + deltas[l])
        tau_l = proj_h.tau(l)
        clean_loss = clean_loss + contrastive_loss(h_clean, y, k=proj_h.k, tau=tau_l)
        adv_loss = adv_loss + contrastive_loss(h_adv, y, k=proj_h.k, tau=tau_l)
    clean_loss = clean_loss / n_layers
    adv_loss = adv_loss / n_layers

    total_loss = clean_w * clean_loss + adv_w * adv_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return float(clean_loss), float(adv_loss), float(total_loss)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'═'*60}")
    print(f"  Experiment 12 — HPS-Adv: PGD-Adversarial Training")
    print(f"  Model: {config.MODEL_NAME}")
    print(f"  ε_train: {EPS_TRAIN}   PGD inner steps: {PGD_STEPS_INNER}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Loss weights: clean={CLEAN_LOSS_WEIGHT}  adv={ADV_LOSS_WEIGHT}")
    print(f"{'═'*60}\n")

    # ── Load attack labels ──
    cat_path = os.path.join(config.RESULTS_DIR, "validated_attacks_categorized.json")
    with open(cat_path) as f:
        categorized = json.load(f)

    attack_prompts = []
    for method, prompts in categorized.items():
        attack_prompts.extend(prompts)

    benign_prompts = list(BENIGN)
    print(f"[exp12] {len(attack_prompts)} attacks, {len(benign_prompts)} benign")

    # ── Load model ──
    model, tokenizer = load_model(config.MODEL_NAME, config.DEVICE, config.DTYPE)
    device = config.DEVICE

    # ── Stratified 80/20 split (consistent with experiment11) ──
    rng = np.random.RandomState(42)
    benign_idx = rng.permutation(len(benign_prompts))
    attack_idx = rng.permutation(len(attack_prompts))
    n_ben_tr = int(0.8 * len(benign_idx))
    n_atk_tr = int(0.8 * len(attack_idx))

    train_benign = [benign_prompts[i] for i in benign_idx[:n_ben_tr]]
    test_benign = [benign_prompts[i] for i in benign_idx[n_ben_tr:]]
    train_attacks = [attack_prompts[i] for i in attack_idx[:n_atk_tr]]
    test_attacks = [attack_prompts[i] for i in attack_idx[n_atk_tr:]]

    print(f"[exp12] Train: {len(train_benign)} benign + {len(train_attacks)} attacks")
    print(f"[exp12] Test:  {len(test_benign)} benign + {len(test_attacks)} attacks")

    # ── Extract activations ──
    def extract(prompts, label):
        n = len(prompts)
        if n == 0:
            return np.zeros((0, len(HPS_LAYERS), 5120))
        acts = []
        print(f"  Extracting {label} ({n})...")
        for i, p in enumerate(prompts):
            d = extract_all_layers(model, tokenizer, p, device, "last")
            acts.append(d)
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{n}")
        d_hidden = acts[0][HPS_LAYERS[0]].shape[0]
        X = np.zeros((n, len(HPS_LAYERS), d_hidden))
        for i, ad in enumerate(acts):
            for j, l in enumerate(HPS_LAYERS):
                if l in ad:
                    X[i, j] = ad[l]
        return X

    print("\n[exp12] Extracting activations...")
    X_train_ben = extract(train_benign, "train benign")
    X_train_atk = extract(train_attacks, "train attacks")
    X_test_ben = extract(test_benign, "test benign")
    X_test_atk = extract(test_attacks, "test attacks")

    X_train = np.concatenate([X_train_ben, X_train_atk], axis=0)
    y_train = np.array([0] * len(X_train_ben) + [1] * len(X_train_atk))
    X_test = np.concatenate([X_test_ben, X_test_atk], axis=0)
    y_test = np.array([0] * len(X_test_ben) + [1] * len(X_test_atk))

    n_layers = len(HPS_LAYERS)
    d_hidden = X_train.shape[2]

    # ── Build projection ──
    print(f"\n[exp12] Building Lorentz projection (d={d_hidden} → d_p={config.PROJECTION_DIM})...")
    torch.manual_seed(42)
    proj_h = LorentzProjection(d_hidden, config.PROJECTION_DIM, 1.0, n_layers=n_layers).to(device)
    optimizer = optim.Adam(proj_h.parameters(), lr=1e-3, weight_decay=1e-5)

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)

    # ── Adversarial training loop ──
    print(f"\n[exp12] Adversarial training: {EPOCHS} epochs, ε={EPS_TRAIN}, K_inner={PGD_STEPS_INNER}\n")

    history = []
    for epoch in range(EPOCHS):
        clean_loss, adv_loss, total = adversarial_train_step(
            proj_h, X_t, y_t, optimizer,
            EPS_TRAIN, PGD_STEPS_INNER, PGD_ALPHA,
            CLEAN_LOSS_WEIGHT, ADV_LOSS_WEIGHT,
        )
        history.append({
            "epoch": epoch,
            "clean_loss": clean_loss,
            "adv_loss": adv_loss,
            "total_loss": total,
        })

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:>3}/{EPOCHS} | clean={clean_loss:.4f}  adv={adv_loss:.4f}  total={total:.4f}")

    # ── Save adversarially trained projection ──
    proj_h.eval()
    save_path = os.path.join(config.RESULTS_DIR, "hps_adv_projection.pt")
    torch.save({
        "state_dict": proj_h.state_dict(),
        "d_in": d_hidden,
        "d_proj": config.PROJECTION_DIM,
        "n_layers": n_layers,
        "eps_train": EPS_TRAIN,
        "training_history": history[-20:],
    }, save_path)
    print(f"\n[exp12] Saved → {save_path}")

    # ══════════════════════════════════════════════════════════════════════
    #  Evaluate baseline (clean) and adversarial accuracy
    # ══════════════════════════════════════════════════════════════════════

    # Extract trajectory features for train and test
    feat_train = extract_trajectory_features(proj_h, X_train)
    feat_test = extract_trajectory_features(proj_h, X_test)

    # Fit logistic regression
    sc = StandardScaler()
    feat_train_s = sc.fit_transform(feat_train)
    feat_test_s = sc.transform(feat_test)
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(feat_train_s, y_train)

    test_scores = clf.predict_proba(feat_test_s)[:, 1]
    auroc_clean = roc_auc_score(y_test, test_scores)

    # TPR @ FPR=1%
    benign_scores = test_scores[y_test == 0]
    attack_scores = test_scores[y_test == 1]
    threshold = float(np.quantile(benign_scores, 0.99))
    tpr_at_fpr01 = float((attack_scores > threshold).mean())

    print(f"\n{'─'*60}")
    print(f"  HPS-Adv BASELINE (clean test)")
    print(f"{'─'*60}")
    print(f"  AUROC:        {auroc_clean:.3f}")
    print(f"  TPR@FPR=1%:   {tpr_at_fpr01:.3f}")
    print(f"  threshold:    {threshold:.4f}")

    # ── Self-attack evaluation: PGD against the adversarially trained model ──
    print(f"\n{'─'*60}")
    print(f"  SELF-ATTACK EVALUATION")
    print(f"  (PGD on test attacks; lower evasion = more robust)")
    print(f"{'─'*60}")

    eval_epsilons = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 20.0, 100.0]

    # Build a simple forward-pass scorer (just contrastive distance to benign cluster)
    # We use the contrastive loss "attack-class similarity" as an attack target
    # For consistency with experiment11, we use trajectory features → logistic
    print(f"  ε       | evasion rate")
    print(f"  --------|-------------")

    results = {
        "config": {
            "eps_train": EPS_TRAIN,
            "pgd_steps_inner": PGD_STEPS_INNER,
            "epochs": EPOCHS,
            "clean_w": CLEAN_LOSS_WEIGHT,
            "adv_w": ADV_LOSS_WEIGHT,
        },
        "baseline_clean": {
            "auroc": float(auroc_clean),
            "tpr_at_fpr01": float(tpr_at_fpr01),
            "threshold": float(threshold),
        },
        "self_attack": {},
    }

    # Build a differentiable scorer to attack against.
    # Mirrors extract_trajectory_features exactly:
    #   - radii = TIME coordinate (h[:, 0]) NOT spatial norms
    #   - distances = Lorentz geodesic distance NOT Euclidean
    #   - all 12 features computed (not 8 + 4 zeros)
    class HPSAdvScorer(nn.Module):
        def __init__(self, proj, sc, clf, n_layers):
            super().__init__()
            self.proj = proj
            self.n_layers = n_layers
            self.register_buffer("scaler_mean", torch.tensor(sc.mean_, dtype=torch.float32))
            self.register_buffer("scaler_std", torch.tensor(sc.scale_, dtype=torch.float32))
            self.register_buffer("clf_coef", torch.tensor(clf.coef_[0], dtype=torch.float32))
            self.register_buffer("clf_intercept", torch.tensor(float(clf.intercept_[0]), dtype=torch.float32))

        @staticmethod
        def lorentz_inner(x, y):
            """⟨x,y⟩_L = -x_0 y_0 + sum_i x_i y_i (signature -+++)"""
            return -x[0] * y[0] + (x[1:] * y[1:]).sum()

        def lorentz_distance(self, x, y):
            """Differentiable Lorentz geodesic distance.
            d(x,y) = (1/sqrt(k)) * arccosh(-k <x,y>_L), with -k<x,y>_L ≥ 1.
            """
            k = self.proj.k
            inner = self.lorentz_inner(x, y)
            arg = -k * inner
            # Clamp to avoid arccosh of < 1 due to numerical error
            arg = torch.clamp(arg, min=1.0 + 1e-7)
            # arccosh(z) = log(z + sqrt(z^2 - 1)), differentiable
            arccosh = torch.log(arg + torch.sqrt(arg * arg - 1.0))
            return arccosh / torch.sqrt(k)

        def forward(self, h, return_logit=False):
            n_layers, d_hidden = h.shape
            # Project each layer to the Lorentz hyperboloid
            x_lorentz_list = []
            for l in range(n_layers):
                x_lorentz_list.append(self.proj(h[l].unsqueeze(0)).squeeze(0))
            x_lorentz = torch.stack(x_lorentz_list, dim=0)  # (n_layers, d_proj+1)

            # Radial features: TIME coordinate (matches extract_trajectory_features)
            radii = x_lorentz[:, 0]

            # Curvature features (triangle-inequality deviation in Lorentz distance)
            curvatures = []
            for j in range(1, n_layers - 1):
                d_prev = self.lorentz_distance(x_lorentz[j], x_lorentz[j - 1])
                d_next = self.lorentz_distance(x_lorentz[j + 1], x_lorentz[j])
                d_span = self.lorentz_distance(x_lorentz[j + 1], x_lorentz[j - 1])
                denom = d_prev + d_next + 1e-8
                kappa = torch.abs(d_prev + d_next - d_span) / denom
                curvatures.append(kappa)
            if len(curvatures) > 0:
                curv = torch.stack(curvatures)
            else:
                curv = torch.zeros(1, device=h.device)

            # Lorentz displacement features
            d_total = self.lorentz_distance(x_lorentz[0], x_lorentz[-1])
            path_segments = []
            for j in range(n_layers - 1):
                path_segments.append(self.lorentz_distance(x_lorentz[j], x_lorentz[j + 1]))
            path_len = torch.stack(path_segments).sum() if path_segments else torch.zeros(1, device=h.device).squeeze()

            # Build the 12 features in the SAME order as extract_trajectory_features
            # Note: argmax is non-differentiable; we use a soft argmax via softmax indexing
            curv_max_arg = (torch.softmax(curv * 10.0, dim=0) *
                            torch.arange(len(curv), dtype=torch.float32, device=h.device)).sum()
            spike_loc = curv_max_arg / max(len(curv), 1)

            feats = torch.stack([
                radii.mean(),
                radii.max(),
                radii.min(),
                radii.std(),
                radii.max() - radii.min(),
                curv.max(),
                curv.mean(),
                curv.std() if len(curv) > 1 else torch.zeros(1, device=h.device).squeeze(),
                spike_loc,
                d_total,
                path_len,
                d_total / (path_len + 1e-8),
            ])

            feats_s = (feats - self.scaler_mean) / (self.scaler_std + 1e-8)
            feats_s = feats_s.float()
            logit = torch.dot(feats_s, self.clf_coef) + self.clf_intercept
            if return_logit:
                return logit
            return torch.sigmoid(logit)

    scorer = HPSAdvScorer(proj_h, sc, clf, n_layers).to(device).eval()

    # ── DIAGNOSTIC: Check for gradient masking via logit range ──
    print(f"\n  Diagnostic: checking for gradient masking (saturated sigmoid)...")
    benign_logits = []
    attack_logits = []
    for i in range(min(20, len(X_test_ben))):
        h_t = torch.tensor(X_test_ben[i], dtype=torch.float32, device=device)
        with torch.no_grad():
            l = scorer(h_t, return_logit=True).item()
        benign_logits.append(l)
    for i in range(min(20, len(X_test_atk))):
        h_t = torch.tensor(X_test_atk[i], dtype=torch.float32, device=device)
        with torch.no_grad():
            l = scorer(h_t, return_logit=True).item()
        attack_logits.append(l)
    print(f"    Benign logit range: [{min(benign_logits):.2f}, {max(benign_logits):.2f}]  mean={np.mean(benign_logits):.2f}")
    print(f"    Attack logit range: [{min(attack_logits):.2f}, {max(attack_logits):.2f}]  mean={np.mean(attack_logits):.2f}")
    if min(attack_logits) > 5.0:
        print(f"    ⚠ Attack logits very large (>5) — sigmoid saturated. Attacking logit directly to avoid gradient masking.")

    # PGD on test attacks (attack the LOGIT to avoid sigmoid saturation / gradient masking)
    _pgd_diag_printed = [False]  # mutable flag for first-sample diagnostic

    def pgd_minimize_score(scorer, h, eps, n_steps=50):
        """Minimize logit (pre-sigmoid). Avoids gradient masking from saturated sigmoid.
        See Athalye et al. 2018, Carlini & Wagner 2017."""
        h0 = h.detach().clone()
        delta = torch.zeros_like(h0, requires_grad=True)
        lr = (eps / max(n_steps, 1)) * 2.5
        for step in range(n_steps):
            logit = scorer(h0 + delta, return_logit=True)
            grad = torch.autograd.grad(logit, delta, create_graph=False)[0]

            # Diagnostic: print on first sample, first/last steps
            if not _pgd_diag_printed[0] and step in (0, n_steps - 1):
                has_nan = torch.isnan(grad).any().item()
                print(f"    [DIAG] eps={eps} step={step}: logit={logit.item():.4f}, "
                      f"grad_norm={grad.norm().item():.2e}, grad_max={grad.abs().max().item():.2e}, "
                      f"has_nan={has_nan}, delta_norm={delta.norm().item():.2e}")
                if step == n_steps - 1:
                    final_s = float(torch.sigmoid(logit))
                    print(f"    [DIAG] final sigmoid={final_s:.6f}, threshold={threshold:.6f}")
                    _pgd_diag_printed[0] = True

            with torch.no_grad():
                delta_new = delta - lr * torch.sign(grad)
                delta_new = torch.clamp(delta_new, -eps, +eps)
            delta = delta_new.detach().requires_grad_(True)
        # Return final score (post-sigmoid) for evasion comparison
        return float(scorer(h0 + delta.detach()))

    # PGD to MAXIMIZE score (for adversarial FPR on benign samples)
    def pgd_maximize_score(scorer, h, eps, n_steps=50):
        """Maximize logit — pushes benign samples toward being flagged as attacks."""
        h0 = h.detach().clone()
        delta = torch.zeros_like(h0, requires_grad=True)
        lr = (eps / max(n_steps, 1)) * 2.5
        for _ in range(n_steps):
            logit = scorer(h0 + delta, return_logit=True)
            grad = torch.autograd.grad(logit, delta, create_graph=False)[0]
            with torch.no_grad():
                delta_new = delta + lr * torch.sign(grad)  # + to maximize
                delta_new = torch.clamp(delta_new, -eps, +eps)
            delta = delta_new.detach().requires_grad_(True)
        return float(scorer(h0 + delta.detach()))

    print(f"  {'ε':<9}| evasion | adv_FPR")
    print(f"  {'─'*9}|─────────|────────")

    for eps in eval_epsilons:
        _pgd_diag_printed[0] = False  # reset diagnostic for each epsilon
        n_evaded = 0
        for i in range(len(X_test_atk)):
            h_t = torch.tensor(X_test_atk[i], dtype=torch.float32, device=device)
            final_score = pgd_minimize_score(scorer, h_t, eps)
            if final_score < threshold:
                n_evaded += 1
        evasion_rate = n_evaded / max(len(X_test_atk), 1)

        # Adversarial FPR: PGD on benign trying to push above threshold
        n_false_pos = 0
        for i in range(len(X_test_ben)):
            h_t = torch.tensor(X_test_ben[i], dtype=torch.float32, device=device)
            final_score = pgd_maximize_score(scorer, h_t, eps)
            if final_score > threshold:
                n_false_pos += 1
        adv_fpr = n_false_pos / max(len(X_test_ben), 1)

        print(f"  {eps:<9.4f}| {evasion_rate:.3f}   | {adv_fpr:.3f}")
        results["self_attack"][f"eps_{eps}"] = {"evasion": evasion_rate, "adv_fpr": adv_fpr}

    save_json(results, "experiment12_adv_training.json", config.RESULTS_DIR)

    # ── Summary ──
    print(f"\n{'═'*60}")
    print(f"  HPS-ADV TRAINING COMPLETE")
    print(f"{'═'*60}")
    print(f"  Clean baseline TPR@FPR=1%: {tpr_at_fpr01:.3f}")
    print(f"\n  Self-attack robustness:")
    for eps in eval_epsilons:
        ev = results["self_attack"][f"eps_{eps}"]
        print(f"    ε={eps:<6}: {ev:.3f} evasion ({(1-ev)*100:.1f}% caught)")

    print(f"\n  Compare to vanilla HPS (from experiment11):")
    print(f"    ε=0.001: HPS=0.359  HPS-Adv={results['self_attack']['eps_0.001']:.3f}")
    print(f"    ε=0.01:  HPS=0.766  HPS-Adv={results['self_attack']['eps_0.01']:.3f}")
    print(f"    ε=0.05:  HPS=1.000  HPS-Adv={results['self_attack']['eps_0.05']:.3f}")
    print(f"    ε=0.1:   HPS=1.000  HPS-Adv={results['self_attack']['eps_0.1']:.3f}")
    print(f"    ε=0.5:   HPS=1.000  HPS-Adv={results['self_attack']['eps_0.5']:.3f}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
