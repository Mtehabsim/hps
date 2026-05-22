# Beyond the Refusal Direction: Hyperbolic Geometry as Inductive Bias for Cross-Attack Jailbreak Detection

## Abstract

Representation-level jailbreak defenses monitor LLM hidden states to detect adversarial prompts. The current state-of-the-art, Representation Trajectory Verification (RTV), uses cosine similarity with an analytical refusal direction across multiple layers, achieving strong detection against attacks that explicitly suppress refusal propagation. However, RTV's reliance on a single fixed direction leaves it blind to attacks that preserve refusal-direction alignment while achieving jailbreak through other mechanisms — notably role-play templates (JBC) and semantic rewrites (PAIR).

We propose HPS (Hyperbolic Projection Sentinel), a learned multi-layer detector that projects activations into hyperbolic space via a trained Lorentz projection with contrastive loss. The hyperbolic geometric prior provides an inductive bias that Euclidean projections lack: the radial coordinate's exponential volume growth enforces consistent semantic meaning across attack types, enabling cross-attack generalization.

On Vicuna-13B with four attack families (GCG, PAIR, JBC, prompt\_with\_random\_search), HPS achieves:
- **Same-distribution:** AUROC=0.970, TPR=85.9% at 5% FPR
- **Cross-attack generalization:** Mean TPR=68.5% (vs Euclidean 0.0% at FPR=1%)
- **Ensemble (HPS+RTV):** Mean cross-attack TPR=73.0% (+4.5% over HPS alone)

We further show that (1) hyperbolic geometry provides +0.302 AUROC over identically-trained Euclidean projections in cross-attack evaluation, (2) the dominant detection signal is radial position (98.7% feature importance), consistent with the geometric prior's design, (3) jailbreak activations occupy a subspace geometrically orthogonal to the harmful-harmless axis, explaining why single-direction defenses have fundamental blind spots, and (4) the refusal suppression hypothesis ("early activation, late suppression") is empirically rejected on Vicuna-13B.

---

## 1. Introduction

Large language models aligned via RLHF can be "jailbroken" — tricked into producing harmful content through adversarial prompts that bypass safety training. Defenses operate at different levels: input filtering (perplexity, classifiers), output filtering (Llama Guard), and representation-level monitoring (JBShield, RTV, GradSafe).

Representation-level defenses are attractive because they observe the model's internal computation — the actual decision-making process — rather than surface-level text. However, recent work establishes that these defenses face fundamental challenges under adaptive threat models:

- Bailey et al. (2026) demonstrate that Mahalanobis-based detectors on raw activations are systematically bypassed by gradient-based obfuscation attacks, regardless of dimensionality.
- Derya & Sunar (2026) show that JBShield's single-layer AND-gate detection is structurally vulnerable, achieving 0% detection against their JB-GCG adaptive attack.
- The same work proposes RTV, which achieves 7% ASR under full white-box adaptive attack by exploiting the tension between refusal suppression and fingerprint evasion.

RTV's robustness derives from a key insight: the refusal direction (Arditi et al., 2024) provides a mechanistically grounded detection signal that creates tension for adaptive attackers. However, RTV operates on a fixed analytical direction and assumes all jailbreaks suppress refusal propagation. Attacks that achieve jailbreak through other mechanisms — role-play contextualization (JBC), semantic rewriting (PAIR) — may not suppress the refusal direction at all, leaving RTV blind.

We propose an alternative approach: rather than relying on a fixed analytical direction, learn a multi-layer geometric projection that captures the structural signature of jailbreak activations. We use hyperbolic geometry (the Lorentz hyperboloid) as the target space, motivated by:

1. **Exponential volume growth** separates hierarchical levels — the radial coordinate provides consistent semantic meaning regardless of attack type.
2. **Contrastive training** in curved space learns attack-specific structure that a fixed direction cannot capture.
3. **Multi-layer trajectory features** (radial position, curvature, displacement) encode cross-layer dynamics that single-layer methods miss.

### 1.1 Contributions

1. We introduce HPS, a jailbreak detector based on learned Lorentz projection of multi-layer activations with contrastive loss, achieving AUROC=0.970 on Vicuna-13B across four attack families.

2. We demonstrate that hyperbolic geometry provides a +0.302 AUROC improvement over identically-trained Euclidean projections in cross-attack generalization — the setting that matters for deployment against unknown future attacks.

3. We provide the first empirical comparison between learned geometric projections (HPS) and analytical refusal-direction methods (RTV) on the same model and attack set, showing complementary strengths: HPS excels on role-play attacks (JBC: 85.7% vs RTV: 9.5%), while RTV provides marginal signal on semantic attacks (PAIR).

4. We show that an ensemble combining HPS trajectory features with RTV refusal-direction fingerprints achieves +4.5% mean cross-attack TPR over HPS alone, demonstrating that the two signals are complementary rather than redundant.

5. We empirically reject the "refusal suppression" hypothesis on Vicuna-13B and show that jailbreak activations are geometrically orthogonal to the harmful-harmless axis — a finding that explains the fundamental limitations of single-direction defenses.

---

## 2. Related Work

### 2.1 Jailbreak Attacks

- **GCG** (Zou et al., 2023): Gradient-based adversarial suffix optimization. Produces gibberish tokens that suppress refusal.
- **PAIR** (Chao et al., 2023): Iterative semantic rewriting via attacker LLM. Produces fluent, natural-sounding jailbreaks.
- **JBC** (jailbreak chat templates): Role-play prompts ("Act as Niccolo Machiavelli...") that contextualize harmful content.
- **prompt\_with\_random\_search** (Andriushchenko et al., 2024): Adaptive suffix search via logprob feedback.

### 2.2 Representation-Level Defenses

- **JBShield** (Zhang et al., USENIX Security 2025): AND-gate over toxic and jailbreak concept directions at a single layer. Reports 0% ASR against non-adaptive GCG. Broken by JB-GCG adaptive attack (53.4% ASR).
- **RTV** (Derya & Sunar, 2026): Mahalanobis outlier detection over 15-dim refusal-direction fingerprint (3 layers × 5 token positions). AUROC=0.99 against JB-GCG; 7% ASR under full adaptive attack.
- **GradSafe** (Xie et al., 2024): Gradient cosine similarity with harmful-response target.
- **HiddenDetect** (Jiang et al., 2025): Cosine similarity with vocabulary-space refusal vector.
- **Circuit Breakers** (Zou et al., 2024): Training-time representation rerouting. Bypassed by Schwinn & Geisler (2024).

### 2.3 Refusal Direction Analysis

Arditi et al. (2024) show that refusal in aligned LLMs is mediated by a single direction in the residual stream, estimated as `r_l = μ_harmful - μ_harmless`. GCG-style attacks succeed by suppressing propagation along this direction. This makes refusal-direction alignment a natural detection feature — but one that is blind to attacks achieving jailbreak through mechanisms other than refusal suppression.

### 2.4 Hyperbolic Geometry for Language

- **HELM** (He et al., NeurIPS 2025): First fully hyperbolic LLM at billion scale; token embeddings exhibit negative Ricci curvature.
- **HypLoRA** (Yang et al., NeurIPS 2025): Token embeddings have measurable δ-hyperbolicity and power-law radial structure.
- **HyPE** (Maljkovic et al., 2026): Hyperbolic SVDD on prompt embeddings for VLM safety — input-side only.

### 2.5 Adaptive Evaluation

Bailey et al. (2026) establish that representation-space defenses must be evaluated under gradient-based adaptive attacks. We follow this methodology using PGD on activations (Section 5.3).

---

## 3. Method

### 3.1 Overview

```
Input prompt
    ↓
Forward pass → extract activations at N selected layers
    ↓
Learned Lorentz projection → points on hyperboloid
    ↓
12 trajectory features (radial, curvature, displacement)
    ↓
Logistic regression → detection score
    ↓
SAFE / FLAGGED
```

### 3.2 Layer Selection

Layers are selected via Fisher separation ratio on a held-out calibration split (30 benign + 30 attacks):

```
Fisher(l) = ||μ_attack(l) − μ_benign(l)|| / (σ_attack(l) + σ_benign(l))
```

Top-8 layers by Fisher score are selected. For Vicuna-13B (40 layers): `[0, 1, 2, 35, 36, 37, 38, 39]` — early embedding layers + late semantic layers.

### 3.3 Lorentz Projection

Each hidden state `h ∈ ℝ^d` (d=5120 for Vicuna-13B) is projected onto the Lorentz hyperboloid:

```
x_proj = W · h · λ                    (W ∈ ℝ^(64×5120), ~330K params)
x₀ = √(1/κ + ||x_proj||²)            (time coordinate)
point = [x₀, x_proj] ∈ L^64_κ        (on the hyperboloid)
```

Learnable parameters:
- `W`: linear projection (5120 → 64)
- `λ`: scale factor, initialized to 1/√64
- `κ`: curvature, learnable in log-space, clamped [0.1, 10.0]
- `τ_l`: per-layer temperature (8 scalars)

### 3.4 Training Loss

Per-layer supervised contrastive loss in hyperbolic space:

```
L = (1/N_L) Σ_l [same_loss(l) / n_same + diff_loss(l) / n_diff] / 2

same_loss(l) = Σ_{i,j same class} d_L(x_i^l, x_j^l)²
diff_loss(l) = Σ_{i,j diff class} max(0, M − d_L(x_i^l, x_j^l))²
```

where `d_L` is the Lorentz geodesic distance, scaled by per-layer temperature `τ_l`.

### 3.5 Trajectory Features

From the projected points across N layers, we extract 12 statistics:

| # | Feature | Category |
|---|---|---|
| 1–5 | mean/max/min/std/range of radial position (x₀) | Radial |
| 6–9 | max/mean/std curvature, spike location | Curvature |
| 10–12 | total displacement, path length, progress ratio | Displacement |

### 3.6 Why Hyperbolic?

The Lorentz hyperboloid has exponential volume growth with radius. This means:
- Points near the origin (low radial position) are "general" — benign prompts cluster here.
- Points far from the origin (high radial position) are "specific/extreme" — attacks are pushed here by contrastive training.
- The radial coordinate provides a **monotonic depth axis** that is consistent across attack types — unlike Euclidean space where different attack families can scatter in arbitrary directions.

This geometric prior is what enables cross-attack generalization: even unseen attack types tend to produce high radial position because they represent "extreme" semantic content.

---

## 4. Experimental Setup

### 4.1 Model and Data

**Target model:** Vicuna-13B-v1.5 (40 layers, 5120 hidden dim, 4-bit quantization)

**Benign prompts (520):**
- Alpaca instructions (100), HumanEval code (50), GSM8K math (100), WritingPrompts (50), WinoGrande (50)
- Hard benign: security education, edgy roleplay, sensitive professional (130)
- Short chat: "hi", "ok", "thanks", etc. (40)

**Attack prompts (316 confirmed jailbreaks):**

| Method | Tested | Confirmed | ASR |
|---|---|---|---|
| prompt\_with\_random\_search | 100 | 98 | 98.0% |
| JBC (role-play templates) | 100 | 84 | 84.0% |
| GCG (gradient suffixes) | 100 | 68 | 68.0% |
| PAIR (semantic rewrites) | 82 | 66 | 80.5% |

**RTV calibration:** 30 harmful + 30 harmless from JBShield repository (Alpaca + AdvBench).

### 4.2 Evaluation Protocol

- **Same-distribution:** 80/20 stratified split, all attack types in both train and test.
- **Cross-attack:** Train on 3 attack methods, test on held-out 4th. Benign 80/20 split shared.
- **Metrics:** AUROC, TPR at 5% FPR, per-method breakdown.
- **Baselines:** Euclidean projection (identical architecture, flat-space contrastive loss), RTV (Derya & Sunar 2026, our reimplementation with empirically-discovered layers).

### 4.3 RTV Reimplementation

We reimplemented RTV following Section 7.2 of Derya & Sunar (2026):
- Refusal direction: `r_l = μ_harmful - μ_harmless` (difference-of-means, normalized)
- Fingerprint: `F[l,p] = cos(r_l, h_l^(p))` for K=3 layers, P=5 token positions → 15-dim
- Detection: `M(x) = min(d_harmless(x), d_harmful(x))` via Mahalanobis with Ledoit-Wolf shrinkage
- Layers empirically discovered for Vicuna-13B: `[12, 16, 26]` (peak refusal sensitivity at layer 16, separation=4.38)

---

## 5. Results

### 5.1 Same-Distribution Detection

[TODO: Table showing HPS vs Euclidean vs RTV on same-distribution test set]

| Method | AUROC | TPR@5%FPR | Training data |
|---|---|---|---|
| RTV (zero-shot) | 0.843* | 0.566* | 30 harmful + 30 harmless |
| Euclidean (supervised) | ~0.85 | ~0.22 | 416 benign + 252 attacks |
| **HPS (supervised)** | **0.970** | **0.859** | 416 benign + 252 attacks |
| **HPS+RTV Ensemble** | **0.975** | **0.891** | Same + 30 harmful/harmless |

*RTV evaluated standalone with matched calibration data (JBShield 30 samples).

### 5.2 Cross-Attack Generalization (Key Result)

Train on 3 attack methods, test on held-out 4th. TPR at 5% FPR:

| Held-out | HPS | RTV | Ensemble | Euclidean |
|---|---|---|---|---|
| GCG | 0.779 | 0.000 | **0.824** | 0.000 |
| JBC | **0.857** | 0.024 | **0.881** | 0.000 |
| PAIR | 0.288 | 0.045 | **0.348** | 0.000 |
| random\_search | 0.816 | 0.194 | **0.867** | 0.000 |
| **MEAN** | **0.685** | 0.066 | **0.730** | 0.000 |

[TODO: Bar chart comparing per-method cross-attack TPR for HPS vs Euclidean vs Ensemble]

**Key findings:**
- HPS generalizes to 3/4 unseen attack families at >77% TPR.
- Euclidean projection achieves 0% TPR on ALL held-out methods — catastrophic negative transfer.
- The ensemble provides consistent +4.5% improvement over HPS alone.
- PAIR remains the hardest attack (34.8% ensemble TPR), consistent with RTV paper's findings (58% on Llama-3-8B).

### 5.3 Hyperbolic vs Euclidean Ablation

Cross-attack mean AUROC (from experiment 8):

| Method | Mean AUROC | Mean TPR@FPR=1% |
|---|---|---|
| Euclidean (trained, 12 features) | 0.513 | 0.000 |
| **Hyperbolic (trained, 12 features)** | **0.815** | **0.236** |
| Δ | **+0.302** | **+0.236** |

The only difference between these two methods is the geometry of the projection space. Architecture, training procedure, features, and classifier are identical. The +0.302 AUROC gap is entirely attributable to the hyperbolic inductive bias.

[TODO: Side-by-side PCA plots showing Euclidean vs Hyperbolic feature spaces with attack clusters]

### 5.4 Per-Method Detection Analysis

Same-distribution per-method TPR at 5% FPR:

| Attack | HPS | RTV | Ensemble |
|---|---|---|---|
| GCG | 0.929 | 0.000 | 0.929 |
| JBC | 0.778 | 0.111 | 0.833 |
| PAIR | 0.800 | 0.067 | 0.800 |
| random\_search | 0.941 | 0.176 | 1.000 |

HPS has no method-specific blind spot below 77.8%. RTV (with mismatched calibration) fails across the board in this evaluation setting.

### 5.5 Adversarial Training (HPS-Adv)

PGD-adversarial training (Madry et al., 2018) applied to the Lorentz projection:

| Metric | Vanilla HPS | HPS-Adv |
|---|---|---|
| AUROC | 0.962 | 0.978 |
| TPR@FPR=1% | 0.672 | 0.812 |

Self-attack robustness (PGD on test attacks):

| ε | Vanilla evasion | HPS-Adv evasion |
|---|---|---|
| 0.001 | 0.359 | 0.266 |
| 0.01 | 0.766 | 0.531 |
| 0.05 | 1.000 | 0.969 |

Adversarial training improves both baseline detection and small-ε robustness. The defense breaks at ε≈0.05 on activations.

### 5.6 Feature Importance

Permutation importance analysis on the 12 trajectory features:

| Category | % Importance |
|---|---|
| **Radial (x₀ statistics)** | **98.7%** |
| Curvature | 0.0% |
| Displacement | 1.3% |

The detection signal is almost entirely radial — the hyperbolic projection's time coordinate (x₀) carries the information. This is consistent with the geometric prior: the radial coordinate encodes "depth/extremity" which is the relevant axis for separating attacks from benign.

### 5.7 Geometric Orthogonality of Jailbreak Activations

[TODO: PCA plot showing attacks displaced along PC2 (orthogonal to harmful-harmless PC1 axis)]

PCA of HPS trajectory features reveals that attacks are displaced along PC2 (8.4% variance) while harmful and harmless prompts separate along PC1 (88.9% variance). This means jailbreaks don't simply move along the harmful-harmless axis — they create activations in a geometrically distinct subspace.

This finding explains why:
- Single-direction methods (RTV, HiddenDetect) have fundamental blind spots — they only monitor PC1.
- Learned multi-dimensional projections (HPS) can capture the orthogonal attack signal.

### 5.8 Negative Results

**Refusal suppression hypothesis rejected.** We tested whether jailbreaks show "early activation, late suppression" of the refusal direction across layers. Empirical measurement:

```
Suppression score (early_mean - late_mean cosine with refusal direction):
  Benign:     +0.058
  Refused:    +0.040
  Jailbreak:  +0.028
```

Jailbreaks show LESS early-late gap than benign prompts. The suppression narrative is not supported on Vicuna-13B.

**Multi-turn drift detection fails with synthetic data.** AUROC=0.593 (barely above random). Synthetic multi-turn conversations don't produce meaningful drift separation. Real multi-turn attack data (actual PAIR conversation logs) would be needed.

**Hyperbolic geometry does not help in zero-shot regime.** With only 30 harmful + 30 harmless calibration samples (no attacks), a learned Lorentz projection achieves CV AUROC=0.845±0.036 — statistically equivalent to RTV's 0.843. The geometric prior requires labeled attack data (~250+ examples) to provide advantage over analytical methods.

---

## 6. Discussion

### 6.1 When Does Hyperbolic Geometry Help?

Our experiments reveal a clear threshold:
- **Zero-shot (30-100 samples, no attacks):** Hyperbolic ties analytical methods. Not enough data to learn useful geometry.
- **Supervised (250+ attacks):** Hyperbolic provides +0.302 AUROC over Euclidean. The contrastive loss in curved space learns structure that flat geometry cannot represent.

The practical implication: use RTV if you have no attack data; switch to HPS once you have ~100+ labeled jailbreak examples.

### 6.2 Complementarity of Detection Signals

HPS and RTV detect different attack signatures:
- **HPS** detects radial displacement in learned hyperbolic space — catches attacks that change the "depth" of representations (JBC, GCG, random\_search).
- **RTV** detects refusal-direction suppression — catches attacks that explicitly suppress the refusal mechanism (GCG, partially PAIR).

The ensemble combining both signals achieves +4.5% mean cross-attack TPR over HPS alone, confirming they are complementary rather than redundant.

### 6.3 Limitations

- **Single model tested.** All experiments on Vicuna-13B. Generalization to Llama-3, Qwen, Mistral is untested.
- **PAIR remains hard.** 34.8% cross-attack TPR on semantic rewrites. This is a shared limitation of all representation-level defenses (RTV reports 58% on Llama-3-8B).
- **Adaptive robustness is limited.** HPS breaks at ε≈0.05 under PGD on activations. Adversarial training provides modest improvement but does not solve the problem.
- **Requires labeled attack data.** Unlike RTV (zero-shot), HPS needs ~250 labeled jailbreak examples for training. This is a deployment constraint.
- **White-box access required.** Cannot protect closed-weight APIs.

### 6.4 Comparison with RTV Paper

| Setting | RTV (paper, Llama-3-8B) | RTV (ours, Vicuna-13B) | HPS (ours, Vicuna-13B) |
|---|---|---|---|
| AUROC vs JB-GCG | 0.9946 | N/A | N/A |
| GCG detection | 0.88 | 0.735* | 0.929 |
| PAIR detection | 0.58 | 0.545* | 0.800 |
| Adaptive ASR | 7% | N/A | breaks at ε≈0.05 |

*RTV standalone with matched calibration (JBShield 30 samples, layers [12,16,26]).

Direct comparison is limited by different models. RTV's headline numbers (0.99 AUROC) are against their custom JB-GCG attack on Llama-3-8B. Against standard diverse attacks, both methods show similar limitations on PAIR.

---

## 7. Conclusion

We introduced HPS, a jailbreak detector that projects multi-layer LLM activations into hyperbolic space via a learned Lorentz projection. The hyperbolic geometric prior provides a +0.302 AUROC improvement over Euclidean projections in cross-attack generalization — the evaluation setting that matters for deployment against unknown future attacks.

Our key finding is that jailbreak detection requires monitoring multiple orthogonal signals. The refusal direction (RTV) and hyperbolic trajectory (HPS) capture different aspects of attack signatures. Neither alone achieves complete coverage, but their ensemble provides consistent improvement across all tested attack families.

We further establish that jailbreak activations are geometrically orthogonal to the harmful-harmless axis, explaining the fundamental limitations of single-direction defenses. This geometric characterization suggests that future defenses should explicitly model the multi-dimensional structure of jailbreak representations rather than relying on a single analytical direction.

---

## References

- Arditi, O., et al. (2024). Refusal in language models is mediated by a single direction. NeurIPS 37.
- Andriushchenko, M., et al. (2024). Simple adaptive jailbreaks bypass leading safety-aligned LLMs. arXiv:2404.02151.
- Bailey, L., et al. (2026). Obfuscated activations bypass LLM latent-space defenses. ICLR 2026. arXiv:2603.10484.
- Chao, P., et al. (2023). Jailbreaking black box large language models in twenty queries. arXiv:2310.08419.
- Derya, K. & Sunar, B. (2026). Revisiting JBShield: Breaking and rebuilding representation-level jailbreak defenses. arXiv:2605.03095.
- He, B., et al. (2025). HELM: Hyperbolic LLM embedding at billion scale. NeurIPS 2025.
- Jiang, F., et al. (2025). HiddenDetect: Detecting jailbreak attacks against LLMs using hidden representations.
- Lee, K., et al. (2018). A simple unified framework for detecting out-of-distribution samples and adversarial attacks. NeurIPS 2018.
- Ledoit, O. & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. Journal of Multivariate Analysis.
- Madry, A., et al. (2018). Towards deep learning models resistant to adversarial attacks. ICLR 2018.
- Maljkovic, D., et al. (2026). HyPE: Hyperbolic prompt embedding for VLM safety. arXiv:2604.06285.
- Schwinn, L. & Geisler, S. (2024). Bypassing circuit breakers. arXiv.
- Wu, Z., et al. (2026). Knowing without acting: The disentangled geometry of safety mechanisms in LLMs. arXiv:2603.05773.
- Yang, M., et al. (2025). HypLoRA: Hyperbolic low-rank adaptation. NeurIPS 2025.
- Zhang, S., et al. (2025). JBShield: Defending LLMs from jailbreak attacks through activated concept analysis. USENIX Security 2025.
- Zou, A., et al. (2023a). Representation engineering: A top-down approach to AI transparency. arXiv:2310.01405.
- Zou, A., et al. (2023b). Universal and transferable adversarial attacks on aligned language models. arXiv:2307.15043.
- Zou, A., et al. (2024). Circuit breakers: Representation rerouting for safety. arXiv.
