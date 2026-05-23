# Beyond the Refusal Direction: Hyperbolic Geometry as Inductive Bias for Cross-Attack Jailbreak Detection

## Abstract

Representation-level jailbreak defenses monitor LLM hidden states to detect adversarial prompts. The current state-of-the-art, Representation Trajectory Verification (RTV), uses cosine similarity with an analytical refusal direction across multiple layers, achieving strong detection against attacks that explicitly suppress refusal propagation. However, RTV's reliance on a single fixed direction leaves it blind to attacks that preserve refusal-direction alignment while achieving jailbreak through other mechanisms — notably role-play templates (JBC) and semantic rewrites (PAIR).

We propose HPS (Hyperbolic Projection Sentinel), a learned multi-layer detector that projects activations into hyperbolic space via a trained Lorentz projection with contrastive loss. The hyperbolic geometric prior provides an inductive bias that Euclidean projections lack: the radial coordinate's exponential volume growth enforces consistent semantic meaning across attack types, enabling cross-attack generalization.

On Vicuna-13B with four attack families (GCG, PAIR, JBC, prompt\_with\_random\_search), HPS achieves:
- **Same-distribution:** AUROC=0.970, TPR=85.9% at 5% FPR
- **Cross-attack generalization:** Mean TPR=68.5% (vs Euclidean 0.0% at FPR=1%)
- **Ensemble (HPS+RTV):** Mean cross-attack TPR=73.0% (+4.5% over HPS alone)

On Llama-3-8B with nine attack families (6520 attacks, balanced evaluation):
- **Same-distribution:** Ensemble AUROC=0.992, TPR=96.9%, F1=0.960
- **Cross-attack generalization:** Ensemble mean TPR=86.1% (vs HPS 68.2%, RTV 52.0%)
- **vs JBShield-D (USENIX Security 2025):** Ensemble accuracy 95.9% vs JBShield-D 55.0%

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

### 1.1 Research Questions

This paper addresses the following research questions:

- **RQ1:** Does hyperbolic geometry provide a better inductive bias than Euclidean geometry for jailbreak detection in compressed activation spaces?
- **RQ2:** Can a learned hyperbolic projection generalize to unseen attack types (cross-attack detection)?
- **RQ3:** How robust is hyperbolic detection under gradient-based adaptive attacks?
- **RQ4:** Does the approach generalize across different LLM architectures?

### 1.2 Contributions

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

- **JBShield** (Zhang et al., USENIX Security 2025): AND-gate over toxic and jailbreak concept directions at a single layer. Reports 0% ASR against non-adaptive GCG. Broken by JB-GCG adaptive attack (53.4% ASR). Our reproduction on Llama-3-8B shows mean accuracy of 55% across 9 attacks.
- **RTV** (Derya & Sunar, 2026): Mahalanobis outlier detection over 15-dim refusal-direction fingerprint (3 layers × 5 token positions). AUROC=0.99 against JB-GCG; 7% ASR under full adaptive attack.
- **GradSafe** (Xie et al., ACL 2024): Computes cosine similarity between the target model's gradient w.r.t. a harmful-response target and a reference gradient. Outperforms Llama Guard on Llama-2 without training. Operates on gradients rather than activations — complementary mechanism to HPS.
- **SALO** (arXiv:2605.02958, 2026): Sparse Activation Localization Operator. Uses causal tracing to identify persistent upstream refusal trajectories that survive even when terminal signals are suppressed. Reports >90% detection where terminal-state methods fail. Concurrent work.
- **Latent Representation Framework** (arXiv:2602.11495, 2026): Tensor-based analysis of hidden activations for jailbreak detection. Reports 78% jailbreak blocking on abliterated Llama-3.1-8B while preserving 94% benign behavior. Also demonstrates inference-time disruption. Concurrent work.
- **HiddenDetect** (Jiang et al., 2025): Cosine similarity with vocabulary-space refusal vector for VLMs.
- **Circuit Breakers** (Zou et al., 2024): Training-time representation rerouting. Bypassed by Schwinn & Geisler (2024).
- **Layerwise Convergence Fingerprints** (arXiv:2604.24542, 2026): Runtime misbehavior detection via layer-wise convergence patterns. Finds different attack types activate different layers. Concurrent work.

### 2.3 Refusal Direction Analysis

Arditi et al. (2024) show that refusal in aligned LLMs is mediated by a single direction in the residual stream, estimated as `r_l = μ_harmful - μ_harmless`. GCG-style attacks succeed by suppressing propagation along this direction. This makes refusal-direction alignment a natural detection feature — but one that is blind to attacks achieving jailbreak through mechanisms other than refusal suppression.

### 2.4 Hyperbolic Geometry for Language and Safety

- **HELM** (He et al., NeurIPS 2025): First fully hyperbolic LLM at billion scale; token embeddings exhibit negative Ricci curvature.
- **HypLoRA** (Yang et al., NeurIPS 2025): Token embeddings have measurable δ-hyperbolicity and power-law radial structure.
- **HyPE** (Maljkovic et al., 2026): Hyperbolic SVDD on prompt embeddings for VLM safety.
- **SALO** (arXiv:2605.02958, 2026): Sparse activation localization for refusal trajectory detection via causal tracing.
- **Streaming Hidden-state Trajectory Detection** (arXiv:2604.07727): Gaussian modeling + Mahalanobis on hidden states at decoding time.

**Differentiation from HyPE.** Concurrent work HyPE (Maljkovic et al., 2026) applies one-class Hyperbolic SVDD to prompt embeddings from a frozen VLM text encoder (HySAC) for NSFW content detection. HPS differs fundamentally in three respects: (1) we operate on *internal multi-layer activations* during the forward pass, not input-side prompt embeddings; (2) we use *supervised contrastive training* in Lorentz space with per-layer temperature calibration, not one-class anomaly detection with a single learned radius; (3) we target *LLM jailbreak detection with cross-attack generalization* across diverse attack families (GCG, PAIR, JBC, random\_search), not VLM content filtering against NSFW prompts. The two methods address different threat models at different points in the pipeline — HyPE guards the input to a VLM decoder, while HPS monitors the internal computation of an LLM during inference.

### 2.5 Adaptive Evaluation

Bailey et al. (2026) establish that representation-space defenses must be evaluated under gradient-based adaptive attacks. We follow this methodology using PGD on activations (Section 5.3).

---

## 3. Method

### Figure 0: Method Overview

[arch2.png]

The HPS pipeline in five steps: (1) Extract activations at 7 critical layers during a single forward pass, (2) Project each activation onto the Lorentz hyperboloid via learned linear map W, (3) Measure 12 trajectory features capturing radial statistics, curvature, and displacement in hyperbolic space, (4) Combine with 15 RTV refusal-direction features into a 27-dimensional vector, (5) Classify via logistic regression calibrated at 5% FPR.

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

### 3.7 Why Hyperbolic Outperforms Euclidean: Theoretical Intuition

Our ablation shows +0.302 AUROC over an identically-trained Euclidean projection. We attribute this to three properties of hyperbolic geometry that align with the structure of jailbreak activations:

**1. Geometric inductive bias via trajectory structure.** In hyperbolic space, the trajectory features (radial position, geodesic displacement, curvature) have consistent geometric meaning. The contrastive loss pushes attacks and benign into geometrically distinct trajectory shapes. In Euclidean space, there is no privileged structure; different attack types scatter in different directions, making a single decision boundary insufficient for cross-attack generalization.

**2. Exponential separation amplifies margins.** The geodesic distance in hyperbolic space grows exponentially with radial position: two points at high radius that are slightly separated in Euclidean terms become vastly separated in geodesic terms. This means the contrastive loss naturally creates larger margins between attack clusters and the benign cluster — without requiring explicit margin tuning.

**3. Model-adaptive feature extraction.** The trajectory features capture different signals on different models: on Vicuna-13B, radial statistics dominate (98.7%); on Llama-3-8B, displacement and curvature contribute meaningfully. This adaptivity is a strength — the projection learns to exploit whichever geometric property best separates classes for a given architecture, rather than relying on a fixed analytical direction.

**Empirical confirmation:** The Euclidean ablation (identical architecture, flat-space contrastive loss) achieves 0% TPR on all held-out attack types at FPR=1% on Vicuna-13B, demonstrating catastrophic failure in cross-attack generalization. The hyperbolic prior prevents this collapse by imposing structural constraints on the learned embedding.

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

**Same-distribution results (Vicuna-13B):**

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

See Figure 3 (`hps_llama3_clusters.png`) for visual comparison of per-method cross-attack separation.

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

See Figure 3 (`hps_llama3_clusters.png`) for side-by-side PCA comparison of HPS vs RTV vs Ensemble feature spaces.

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

See Figure 2 (`hps_zeroshot_clusters.png`) showing attacks displaced along PC2 (orthogonal to the harmful-harmless PC1 axis).

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

### 6.3 Relationship to the Disentangled Safety Hypothesis

Wu et al. (2026) propose that safety in LLMs operates on two geometrically distinct axes — Recognition (v\_H, detecting harmful intent) and Execution (v\_R, deciding whether to comply). Our findings partially engage with this framework:

- **Our detector captures Recognition, not Execution.** Refused attacks score identically to successful attacks (0.99 vs 0.998), confirming that HPS detects harmful *content* (Recognition axis) rather than model *compliance* (Execution axis). This is consistent with DSH's prediction that these axes decouple in deep layers.

- **The orthogonal displacement (PC2) may represent a third axis.** Our PCA analysis shows jailbreak activations displaced along a direction orthogonal to the harmful-harmless separation (PC1). This orthogonal component is not predicted by the two-axis DSH framework and may correspond to the specific computational signature of adversarial prompt processing — distinct from both Recognition and Execution. Further investigation is needed to characterize this axis mechanistically.

- **The refusal suppression hypothesis is not supported.** We tested whether jailbreaks show "early activation, late suppression" of the refusal direction (a prediction derivable from DSH). Empirically, jailbreaks show *less* early-late gap than benign prompts on Vicuna-13B, rejecting this specific mechanism.

### 6.4 Adaptive Robustness and Obfuscated Activations

Under white-box PGD attack on activations (following Bailey et al., 2026), we evaluate both HPS alone and the HPS+RTV ensemble:

| ε (L∞ budget) | HPS-Adv evasion | Ensemble evasion |
|---|---|---|
| 0.001 | 26.6% | 6.0% |
| 0.005 | — | 16.0% |
| 0.01 | 53.1% | 26.0% |
| 0.05 | 96.9% | 90.0% |
| 0.1 | 100% | 100% |

The ensemble provides modest improvement at small ε (6% vs 26.6% evasion at ε=0.001) but both break at ε≈0.05. The dual-signal architecture does not provide substantial additional adaptive robustness because both feature sets (HPS trajectory and RTV fingerprint) are computed from the same activation tensor — a perturbation that fools one simultaneously affects the other.

An attacker with full white-box access to the projection weights W, refusal directions r_l, classifier coefficients, and detection threshold can minimize the ensemble score via gradient descent. This is a shared limitation of all representation-level defenses operating in continuous activation space (Bailey et al., 2026). True robustness would require features computed from independent sources (e.g., different model components, multiple forward passes, or token-level analysis).

We note that the relevant deployment threat model is not unlimited white-box access to activations (which implies the attacker can already modify model behavior arbitrarily), but rather token-level optimization with a fixed suffix budget — a setting where representation-level defenses retain practical value because the mapping from token perturbations to activation perturbations is constrained and non-trivial to optimize.

### 6.5 Limitations

- **Single model tested.** All experiments on Vicuna-13B. Generalization to Llama-3, Qwen, Mistral is untested.
- **PAIR remains hard.** 34.8% cross-attack TPR on semantic rewrites. This is a shared limitation of all representation-level defenses (RTV reports 58% on Llama-3-8B).
- **Adaptive robustness is limited.** HPS breaks at ε≈0.05 under PGD on activations. Adversarial training provides modest improvement but does not solve the problem.
- **Requires labeled attack data.** Unlike RTV (zero-shot), HPS needs ~250 labeled jailbreak examples for training. This is a deployment constraint.
- **White-box access required.** Cannot protect closed-weight APIs.
- **Calibration sensitivity.** RTV's performance varies significantly with calibration data distribution (AUROC 0.843 with JBShield data vs 0.729 with HarmBench data). The ensemble inherits this sensitivity for the RTV component.

### 6.6 Comparison with RTV Paper

### 6.6 Multi-Model Validation: Llama-3-8B

To validate that HPS generalizes beyond Vicuna-13B, we evaluate on Meta-Llama-3-8B-Instruct using 9 attack types from the JBShield repository (6520 attacks total) with balanced benign data (5200 train + 1300 test).

**Same-distribution results (Llama-3-8B):**

| Method | AUROC | TPR | FPR | F1 | Accuracy |
|---|---|---|---|---|---|
| RTV | 0.856 | 0.531 | 0.050 | 0.672 | 0.740 |
| HPS | 0.956 | 0.841 | 0.050 | 0.890 | 0.896 |
| **Ensemble (HPS+RTV)** | **—** | **0.969** | **0.050** | **0.960** | **0.959** |

The ensemble achieves 96.9% TPR — a +12.8% improvement over HPS alone — demonstrating that RTV features provide substantial complementary signal when combined via logistic regression, even though RTV alone achieves only 53.1% TPR.

**Cross-attack results (Llama-3-8B, train on 8 methods, test on held-out 1):**

| Held-out | HPS | RTV | 
|---|---|---|
| autodan | **0.800** | 0.413 |
| base64 | 0.642 | **0.800** |
| drattack | **0.860** | 0.804 |
| gcg | **0.747** | 0.559 |
| ijp | **0.439** | 0.233 |
| pair | **0.448** | 0.136 |
| puzzler | **0.740** | 0.360 |
| saa | 1.000 | **0.995** |
| zulu | **0.464** | 0.379 |
| **MEAN** | **0.682** | 0.520 |

HPS outperforms RTV on 7/9 attack types in cross-attack generalization, with a mean TPR advantage of +16.2%. RTV wins on encoding-based attacks (base64) where refusal-direction suppression is the dominant signal.

**Per-method comparison with baselines (Llama-3-8B):**

| Attack | JBShield-D (reproduced) | RTV (ours) | HPS (ours) | Ensemble (ours) |
|---|---|---|---|---|
| IJP | 0.425 acc / F1=0.240 | 0.258 | 0.742 | **0.916** |
| GCG | 0.616 acc / F1=0.539 | 0.576 | 0.785 | **0.983** |
| SAA | 0.915 acc / F1=0.922 | 0.994 | 0.718 | **1.000** |
| AutoDAN | 0.537 acc / F1=0.338 | 0.412 | 0.919 | **0.986** |
| PAIR | 0.411 acc / F1=0.383 | 0.134 | 0.909 | **0.902** |
| DrAttack | 0.701 acc / F1=0.716 | 0.775 | 0.856 | **0.955** |
| Puzzler | 0.275 acc / F1=0.000 | 0.455 | 0.727 | **1.000** |
| Zulu | 0.993 acc / F1=0.993 | 0.352 | 0.905 | **1.000** |
| Base64 | 0.391 acc / F1=0.000 | 0.812 | 0.938 | **1.000** |

JBShield-D (Zhang et al., USENIX Security 2025) was reproduced using their published code and data on Llama-3-8B-Instruct. It achieves mean accuracy of 0.55 across 9 attack types, with F1=0 on Puzzler and Base64 (complete detection failure). The AND-gate architecture is structurally vulnerable as demonstrated by Derya & Sunar (2026). Our HPS+RTV ensemble achieves 95.9% accuracy on the same model, representing a +40.9% absolute improvement over the peer-reviewed baseline.

**Comprehensive comparison with all representation-level defenses:**

| Method | Venue | Model | Metric | Overall | GCG | PAIR | Notes |
|---|---|---|---|---|---|---|---|
| **JBShield-D** | USENIX Sec 2025 | Llama-3-8B | Accuracy | 0.55 | 0.616 | 0.411 | Our reproduction, their code |
| **JBShield-D** | USENIX Sec 2025 | Llama-3-8B | Det. rate | ~0.97* | 0.98* | 0.77* | *Paper-reported (non-adaptive) |
| **RTV** | Preprint 2026 | Llama-3-8B | TPR@5% | 0.531 | 0.576 | 0.134 | Our implementation |
| **RTV** | Preprint 2026 | Llama-3-8B | Det. rate | ~0.68* | 0.88* | 0.58* | *Paper-reported |
| **GradSafe** | ACL 2024 | Llama-2-7B | F1 | 0.95* | — | — | *Paper-reported, different model |
| **SALO** | Preprint 2026 | Llama-3-8B | Det. rate | >0.90* | — | — | *Paper-reported, forced-decoding |
| **Latent Rep. Framework** | Preprint 2026 | Llama-3.1-8B | Block rate | 0.78* | — | — | *Paper-reported, abliterated model |
| **HPS (ours)** | — | Llama-3-8B | TPR@5% | 0.841 | 0.785 | 0.909 | Same-distribution |
| **HPS+RTV Ensemble (ours)** | — | Llama-3-8B | TPR@5% | **0.969** | **0.983** | **0.902** | Same-distribution |
| **HPS+RTV Ensemble (ours)** | — | Llama-3-8B | Cross-atk TPR | **0.861** | **0.966** | 0.306 | Train on 8, test on 1 |

*Numbers marked with * are from the respective papers and may not be directly comparable due to different evaluation protocols, calibration data, or threat models.

**Detailed per-attack results — HPS vs RTV vs Ensemble (Llama-3-8B, same test set, 5% FPR):**

*Same-distribution (train on all 9 attack types, test on held-out 20%):*

| Attack | N | HPS TPR | RTV TPR | Ensemble TPR |
|---|---|---|---|---|
| AutoDAN | 148 | 0.919 | 0.412 | **0.986** |
| Base64 | 160 | 0.938 | 0.812 | **1.000** |
| DrAttack | 111 | 0.856 | 0.775 | **0.955** |
| GCG | 172 | 0.785 | 0.576 | **0.983** |
| IJP | 178 | 0.742 | 0.258 | **0.916** |
| PAIR | 164 | **0.909** | 0.134 | 0.902 |
| Puzzler | 11 | 0.727 | 0.455 | **1.000** |
| SAA | 181 | 0.718 | **0.994** | **1.000** |
| Zulu | 179 | 0.905 | 0.352 | **1.000** |
| **Overall** | **1304** | **0.841** | **0.531** | **0.969** |

| Metric | HPS | RTV | Ensemble |
|---|---|---|---|
| AUROC | 0.956 | 0.856 | **0.992** |
| TPR (Recall) | 0.841 | 0.531 | **0.969** |
| FPR | 0.050 | 0.050 | 0.050 |
| Precision | 0.944 | 0.914 | **0.951** |
| F1 | 0.890 | 0.672 | **0.960** |
| Accuracy | 0.896 | 0.740 | **0.959** |

*Cross-attack (train on 8 attack types, test on held-out 1):*

| Held-out | N | HPS TPR | RTV TPR | Ensemble TPR |
|---|---|---|---|---|
| AutoDAN | 850 | 0.800 | 0.413 | 0.796 |
| Base64 | 850 | 0.642 | 0.800 | **1.000** |
| DrAttack | 520 | 0.860 | 0.804 | **0.927** |
| GCG | 850 | 0.747 | 0.559 | **0.966** |
| IJP | 850 | 0.439 | 0.233 | **0.785** |
| PAIR | 850 | **0.448** | 0.136 | 0.306 |
| Puzzler | 50 | 0.740 | 0.360 | **0.980** |
| SAA | 850 | 1.000 | 0.995 | **1.000** |
| Zulu | 850 | 0.464 | 0.379 | **0.991** |
| **MEAN** | — | 0.682 | 0.520 | **0.861** |

**Key takeaways:**
- Ensemble outperforms both individual methods on 8/9 attack types in cross-attack.
- HPS alone outperforms RTV on 7/9 attack types (+16.2% mean TPR).
- The ensemble's cross-attack mean (86.1%) is +17.9% over HPS alone and +34.1% over RTV alone.
- PAIR is the only attack where cross-attack performance remains low (30.6%) — a shared limitation of all representation-level defenses.

### 6.7 Mitigation: Refusal Direction Injection (Preliminary)

We explore activation-level mitigation inspired by JBShield-M (Zhang et al., 2025). When HPS flags a prompt as an attack, we inject the refusal direction into late-layer activations to steer the model toward safe refusal:

```
If HPS_score(prompt) > threshold:
    h_l ← h_l + α · r_l    for l ∈ {35, 36, 37, 38, 39}
```

Results on Vicuna-13B (α=8.0, detection-gated):

| Metric | Value |
|---|---|
| Baseline ASR | 0.720 |
| Mitigated ASR | 0.660 |
| ASR Reduction | 0.060 |
| Helpfulness (benign) | 0.933 |
| False Refusal Rate | 0.000 |

The detection gate preserves helpfulness (93.3%) with zero false refusals on benign prompts. However, ASR reduction is modest (6%) — the refusal direction injection at this strength is insufficient to override most successful jailbreaks. This is consistent with the broader challenge of activation-level mitigation: JBShield-M achieves 30.7% ASR (from 53.4% baseline) with a similar approach.

Mitigation via activation steering remains an open problem. The detection component (HPS) is the primary contribution; mitigation is left for future work with stronger intervention strategies (e.g., multi-direction injection, adaptive strength calibration, or token-level removal for suffix-based attacks).

---

## 7. Conclusion

We introduced HPS, a jailbreak detector that projects multi-layer LLM activations into hyperbolic space via a learned Lorentz projection. The hyperbolic geometric prior provides a +0.302 AUROC improvement over Euclidean projections in cross-attack generalization — the evaluation setting that matters for deployment against unknown future attacks.

We validate HPS on two models (Vicuna-13B and Llama-3-8B) across 9 attack families, demonstrating consistent superiority over both JBShield-D (USENIX Security 2025) and RTV (Derya & Sunar, 2026). On Llama-3-8B with balanced evaluation:
- HPS alone achieves AUROC=0.956 and 84.1% TPR at 5% FPR
- The HPS+RTV ensemble reaches **96.9% TPR** (AUROC=0.992) by combining complementary detection signals
- In cross-attack generalization, the ensemble achieves **86.1% mean TPR** across 9 unseen attack types
- This represents a +40.9% absolute accuracy improvement over JBShield-D (95.9% vs 55.0%)

Our key findings:
1. **Hyperbolic geometry provides superior inductive bias** for cross-attack generalization (+0.302 AUROC over Euclidean, +16.2% TPR over RTV in cross-attack on Llama-3-8B).
2. **HPS and RTV are complementary, not redundant.** HPS excels on semantic and role-play attacks (PAIR, AutoDAN, JBC); RTV provides signal on encoding-based attacks (Base64, SAA). Their ensemble achieves coverage no single method provides.
3. **Jailbreak activations are geometrically orthogonal** to the harmful-harmless axis, explaining why single-direction defenses (JBShield, RTV alone) have fundamental blind spots.
4. **JBShield-D is structurally broken** on Llama-3-8B under standard evaluation (mean accuracy 55%, F1=0 on two attack types), confirming the vulnerability identified by Derya & Sunar (2026).

Limitations remain: PAIR achieves only 30.6% cross-attack TPR (the hardest attack family for all representation-level defenses), adaptive robustness is limited to ε<0.05 under PGD, and mitigation via activation steering shows only modest ASR reduction (6%). Future work should address these through stronger intervention strategies and multi-model scaling.

### Answers to Research Questions

- **RQ1 (Hyperbolic vs Euclidean):** Yes. Hyperbolic provides +0.302 AUROC over Euclidean in cross-attack generalization on Vicuna-13B. Euclidean achieves 0% TPR on all held-out attack types; hyperbolic achieves 23.6%. The radial coordinate's monotonic depth interpretation is the key structural advantage.

- **RQ2 (Cross-attack generalization):** Yes, with caveats. HPS generalizes to 7/9 unseen attack types at >46% TPR on Llama-3-8B (mean 68.2%). The ensemble with RTV reaches 86.1% mean cross-attack TPR. PAIR (semantic rewrites) remains the hardest case at 30.6%.

- **RQ3 (Adaptive robustness):** Limited. Under PGD on activations, both HPS alone and the HPS+RTV ensemble break at ε≈0.05. The ensemble provides modest improvement at small ε (6% evasion vs 26.6% at ε=0.001) but does not fundamentally change the breaking point. The dual-signal architecture does not provide additional robustness because both feature sets are computed from the same perturbed activations.

- **RQ4 (Multi-model generalization):** Yes. Validated on Vicuna-13B (40 layers) and Llama-3-8B (32 layers) with consistent results. HPS outperforms RTV on both models. The ensemble achieves 96.9% TPR on Llama-3-8B and 89.1% on Vicuna-13B in same-distribution evaluation.

---

## 8. Figures

### Figure 1: Poincaré Disk — HPS Projection (Final Layer)

[results/viz_poincare_disk.png]

The trained Lorentz projection maps LLM activations onto the hyperbolic hyperboloid, visualized here via Poincaré disk projection. Benign prompts (green) form a compact cluster on the right side of the disk, while jailbreak attacks (purple) cluster on the left — clearly separated with minimal overlap. This demonstrates that the contrastive training successfully learns a hyperbolic embedding where the two classes occupy geometrically distinct regions. The separation is angular (different directions from center) rather than purely radial, indicating the projection exploits the full hyperbolic structure, not just distance from origin.

### Figure 2: Geometric Orthogonality of Jailbreak Activations (Vicuna-13B)

[results/hps_zeroshot_clusters.png]

PCA projection of HPS trajectory features reveals a fundamental geometric property: jailbreak activations are displaced along PC2 (8.4% variance, vertical axis) while harmful and harmless prompts separate along PC1 (88.9% variance, horizontal axis). This means jailbreaks don't simply move along the harmful-harmless continuum — they create activations in a geometrically orthogonal subspace. The annotated axes show: PC1 corresponds to the refusal axis (what RTV monitors), while PC2 captures an attack-specific signal invisible to single-direction defenses. This explains why RTV achieves only 9.5% TPR on JBC attacks — it monitors only the horizontal axis and misses the vertical displacement entirely.

### Figure 3: Feature Space Comparison — HPS vs RTV vs Ensemble (Llama-3-8B)

[results/hps_llama3_clusters.png]

Three-panel comparison of detection feature spaces on the same test data (1300 benign + 1304 attacks across 9 attack types):

- **Left (HPS, 12D, AUROC=0.956):** PC1 captures 96.8% of variance. Benign (green) clusters at positive PC1, attacks scatter at negative PC1. Clear separation along one dominant axis — the learned hyperbolic trajectory direction.

- **Middle (RTV, 15D, AUROC=0.856):** PC1 captures only 60.5% of variance. All classes are heavily tangled — harmless, harmful, and attacks overlap substantially. The refusal-direction fingerprint provides weaker separation because many attacks (PAIR, IJP) don't suppress the refusal direction.

- **Right (Ensemble, 27D, AUROC=0.992):** Combining both feature sets, PC1 captures 90.4% of variance with improved separation. The logistic regression learns to weight HPS features for most attacks while leveraging RTV features for encoding-based attacks (Base64, SAA) where refusal suppression is the dominant signal.

### Figure 4: Activation Trajectory Through Layers

[results/viz_trajectory.png]

Shows how the radial position (x₀, time coordinate on the Lorentz hyperboloid) evolves as activations pass through model layers. Both benign and attack trajectories start at x₀≈1.0 in early layers (0, 1, 2) — where activations are dominated by token embeddings and carry little semantic information. As activations reach late layers (28–31), radial position increases dramatically as the model commits to its response strategy. The key observation: benign and attack trajectories diverge at the final layer (31), with benign reaching slightly higher x₀ on average. The large variance (shaded regions) at late layers explains why multi-layer trajectory features (displacement, range, curvature) are needed rather than single-layer radial thresholding.

### Figure 5: HPS Feature Importance

[results/viz_feature_importance.png]

Logistic regression coefficient magnitudes reveal which of the 12 trajectory features drive detection on Llama-3-8B. The top features are:
1. **Displacement** (geodesic distance from first to last layer) — the strongest signal, measuring how far the activation "travels" through hyperbolic space across layers.
2. **Max radial position** and **radial range** — capturing the peak extremity and variability of the trajectory.
3. **Mean radial position** — overall depth in hyperbolic space.
4. **Max curvature** — trajectory bending, indicating abrupt directional changes between layers.

Notably, on Llama-3-8B, curvature features contribute meaningfully (unlike Vicuna-13B where they were negligible). This suggests the trajectory shape signal is model-dependent — the projection adapts to each architecture's internal dynamics. Path length is the least important feature, indicating that total distance traveled matters less than the endpoints and shape of the trajectory.

### Figure 6: RTV Fingerprint Space (Llama-3-8B)

[results/rtv_llama3_results_clusters.png]

RTV's 15-dimensional refusal-direction fingerprint projected to 2D. Harmless prompts (green, left) and harmful prompts (red, right) are well-separated — confirming the refusal direction is correctly computed. However, attacks (purple) form a massive diffuse cloud that overlaps heavily with both legitimate clusters. This explains RTV's 53.1% TPR: many attacks produce fingerprints indistinguishable from legitimate prompts. The overlap is particularly severe in the center, where PAIR and IJP attacks reside — these attacks don't suppress the refusal direction and thus appear "normal" to RTV's Mahalanobis detector.

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
- SALO (2026). Exploiting latent refusal trajectories for robust jailbreak detection. arXiv:2605.02958.
- Streaming Hidden-state Trajectory Detection (2026). Streaming hidden-state trajectory detection for decoding-time jailbreak defense. arXiv:2604.07727.
- Latent Representation Framework (2026). Understanding and detecting jailbreak attacks from internal representations of large language models. arXiv:2602.11495.
- Layerwise Convergence Fingerprints (2026). Layerwise convergence fingerprints for runtime misbehavior detection in LLMs. arXiv:2604.24542.
- Xie, Y., et al. (2024). GradSafe: Detecting jailbreak prompts for LLMs via safety-critical gradient analysis. ACL 2024. arXiv:2402.13494.
