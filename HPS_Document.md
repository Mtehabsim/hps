# Hyperbolic Physiological Sentinel (HPS): Detecting Jailbreaks via Internal Trajectory Geometry

## One-Sentence Summary

HPS intercepts jailbreaks by detecting the layer-wise geodesic pivot that occurs when a model transitions from processing a benign wrapper to resolving a harmful payload — a structural signature invisible to input-side classifiers and output-side filters.

---

## 1. The Idea

Large language models can be tricked into producing harmful content through "jailbreak" attacks — carefully crafted prompts that bypass safety alignment. Current defenses either inspect the input text (easily fooled by obfuscation) or check the output (too late for agentic systems that act immediately).

We propose monitoring the model's internal reasoning trajectory during inference. When a model processes a jailbreak, its hidden states undergo a characteristic geometric deformation — a "pivot" from the benign wrapper to the harmful payload. We detect this pivot by projecting the model's layer-by-layer activations into hyperbolic space and measuring trajectory anomalies.

**Mechanistic grounding (DSH, Wu et al. 2026):** Wu et al. ("Knowing without Acting: The Disentangled Geometry of Safety Mechanisms in Large Language Models", arXiv:2603.05773) empirically demonstrate via linear probes that safety in LLMs operates on two geometrically distinct axes — a Recognition axis (v_H, encoding harmful semantics) and an Execution axis (v_R, driving refusal). They measure cosine similarity between these axes across layers and observe a universal "Reflex-to-Dissociation" trajectory: strong antagonistic coupling in early layers (sim ≈ -0.9) decaying to statistical independence in deep layers.

**The bridge from DSH to curvature (hypothesis):** When the axes are coupled (early layers), the activation trajectory moves smoothly — refusal is automatically co-activated with recognition. When they decouple (deep layers), a jailbreak must redirect the trajectory from the Recognition subspace toward compliant generation without triggering the now-independent Execution axis. While this decoupling is gradual across layers, we hypothesize that the jailbreak's semantic commitment — the point at which the model resolves the obfuscated instruction into a concrete harmful directive — is discrete. If correct, this discrete resolution would manifest as a concentrated curvature spike at the specific layer where the decision crystallizes. This is a testable prediction: if curvature is instead distributed uniformly, the method may still work via radial and displacement features, but the "pivot detection" narrative would need revision.

**Why hyperbolic space:** Hyperbolic geometry's exponential volume growth naturally separates hierarchical levels that Euclidean space compresses. A jailbreak forces traversal from abstract to specific, producing measurable radial displacement and curvature. Whether this separation is amplified by metric stretching (which requires projected points to land near the boundary) is an empirical question we verify by measuring the norm distribution of projected activations.

---

## 2. Related Work

### Jailbreak Attacks
- **GCG** (Zou et al., 2023): Gradient-based suffix optimization producing gibberish tokens that trigger compliance
- **PAIR** (Chao et al., 2023): Iterative semantic rewriting producing fluent jailbreaks
- **TAP** (Mehrotra et al., 2024): Tree-of-attacks pruning for efficient jailbreak search
- **AutoDAN, multi-turn, role-play**: Various automated methods producing diverse attack surfaces

### Input-Side Defenses
- **HyPE** (Maljkovic et al., arXiv:2604.06285, 2026): Hyperbolic SVDD anomaly detection on VLM prompt embeddings. Achieves 0.98 F1 but only sees the input — cannot detect obfuscated intent resolved internally.
- **LatentGuard, GuardT2I**: Embedding-based classifiers that flag harmful prompts before processing.
- **Limitation**: All input-side methods fail against attacks where surface text appears benign but internal semantic resolution produces harmful compliance.

### Activation-Based Safety Monitoring
- **Representation Engineering (RepE)** (Zou et al., 2024): Safety concepts are linearly separable at specific "collapse point" layers.
- **Disentangled Safety Hypothesis (DSH)** (Wu et al., arXiv:2603.05773, 2026): Empirically proves via Double-Difference Extraction that safety operates on two geometrically distinct axes (Recognition v_H and Execution v_R) that decouple in deep layers. Their Refusal Erasure Attack achieves SOTA jailbreak success by surgically removing v_R.
- **LatentBiopsy** (arXiv:2603.27412, 2026): Training-free jailbreak detection via angular deviation in residual streams. Detects harmful intent as a geometric feature without any learned parameters. A direct competitor — we differentiate by using trajectory dynamics across layers (curvature, displacement) rather than angular deviation at individual layers, and by learning a hyperbolic projection that amplifies the signal.
- **Latent Representation Framework** (arXiv:2602.11495, 2026): Tensor-based analysis of hidden activations for jailbreak detection. Identifies layer-wise patterns and achieves 78% jailbreak blocking on abliterated models. Operates in Euclidean space without geometric projection.
- **AdaSteer** (Zhou et al., EMNLP 2025): Dynamically adjusts steering intensity based on activation thresholds.
- **Google AMS** (2026): Validates activation-based scanning for open-weight LLM safety verification.

### Hyperbolic Geometry for Language
- **HELM** (He et al., NeurIPS 2025): First fully hyperbolic LLM at billion scale. Token embeddings exhibit wide-ranging negative Ricci curvature.
- **HypLoRA** (Yang et al., NeurIPS 2025): LLM token embeddings exhibit measurable δ-hyperbolicity and power-law radial structure.
- **HyperRealm** (CVPR 2026): Poincaré ball VLM with entropy-driven entailment loss — trained hyperbolic projections produce meaningful hierarchy.
- **HyCon** (arXiv:2603.14093, 2026): Hyperbolic concept steering for T2I models via parallel transport. Demonstrates that hyperbolic steering is more stable than Euclidean under strong intervention — relevant as evidence that hyperbolic geometry provides structural advantages for safety-related manipulation.

### Our Position

| Defense Type | Example | What It Sees | Blind Spot |
|---|---|---|---|
| Input filter | HyPE, LatentGuard | Prompt text/embedding | Obfuscated intent decoded internally |
| Output filter | Llama Guard | Generated text | Too late for agentic systems |
| Single-layer probe | RepE, ITI | One layer snapshot | Misses dynamics across layers |
| Angular deviation | LatentBiopsy | Per-layer angle from mean | No learned projection, no trajectory structure |
| **Ours (HPS)** | — | Multi-layer hyperbolic trajectory | Requires white-box access |

HyPE guards the door. We guard the mind.

---

## 3. Method

### Overview

```
Input Prompt → [Activation Extraction] → [Hyperbolic Projection] → [Trajectory Features] → [Classification]
                                                                                              ↓
                                                                                    SAFE / BLOCKED
```

### 3.1 Targeted Activation Extraction

During the forward pass, we extract hidden states from N_L selected layers (we use N_L = 8).

**Layer selection:** Performed via Fisher-ratio separation scoring on a held-out calibration split (disjoint from both training and test data). We measure the ratio of between-class distance to within-class spread at each layer and select the N_L layers with maximum separation. This identifies model-specific "decision layers" where safety-relevant computation concentrates.

**Calibration split:** The calibration set (used only for layer selection) is drawn separately from the training set (used for projection head training) and the test set (used for evaluation). We allocate 15% of data to calibration (~15 adversarial + 30 benign). This is thin but sufficient for Fisher-ratio estimation (which only requires class means and variances, not complex decision boundaries). Alternative: layer selection via unsupervised activation variance across layers (no labels needed), which would eliminate the calibration split entirely. We compare both approaches.

**Token pooling:** We ablate three strategies and select empirically:
- Last token (has attended to full context)
- Mean of all tokens (captures distributed signal)
- Mean of last 5 tokens (compromise)

The choice is a hyperparameter determined by ablation, not a design assumption.

This produces a trajectory: h₁, h₂, ..., h_{N_L} ∈ ℝᵈ

### 3.2 Hyperbolic Projection

Each hidden state is projected onto the Lorentz hyperboloid through a learned projection head:

```
x_proj = W · h · λ                         (linear projection, W ∈ ℝ^(d×d_p), learnable scale λ)
x₀ = sqrt(1/κ + ||x_proj||²)               (time coordinate, κ = curvature constant)
point = [x₀, x_proj] ∈ L^{d_p}_κ           (on the hyperboloid)
```

**Notation:** κ denotes the curvature constant of the hyperboloid (κ = 1.0). N_L denotes the number of selected layers. These are distinct quantities.

**Projection dimension d_p:** We use d_p = 256 (16x compression from d = 4096). Sensitivity analysis over d_p ∈ {64, 128, 256, 512} is included in ablations.

**Shared projection head:** A single W is applied across all selected layers. Our hypothesis is that Fisher-ratio selection picks layers with sufficiently compatible statistics that a shared projection is not a bottleneck — and that sharing acts as regularization, forcing the head to find a universal safety-relevant projection rather than overfitting to layer-specific artifacts. We test this against per-layer heads (N_L separate W matrices) in ablation; if per-layer heads significantly outperform, the shared design should be abandoned.

**Training objective:** Supervised contrastive loss computed in Lorentz geodesic distance, applied independently at each layer and summed:

```
L = (1/N_L) Σ_{l=1}^{N_L} L_layer(l)

L_layer(l) = (1/|P|) Σ_{(i,j) ∈ P} [ 𝟙[y_i = y_j] · d_L(x_i^l, x_j^l)²
                                      + 𝟙[y_i ≠ y_j] · max(0, M - d_L(x_i^l, x_j^l))² ]
```

where x_i^l is the Lorentz projection of sample i at layer l, P is the set of all pairs in the batch, and M is the margin hyperparameter.

**Per-layer loss (Option C):** By applying the contrastive loss at each layer independently and summing, the projection head is trained to produce discriminative geometry at every layer — not just a single representative layer.

**Pair sampling:** With imbalanced classes (e.g., 100 adversarial + 200 benign), same-class benign pairs dominate the loss. We use balanced pair sampling: equal numbers of same-class and cross-class pairs per batch, with same-class pairs drawn equally from both classes.

**On "trajectory" framing:** The per-layer loss does not model inter-layer relationships (no temporal smoothness or transition terms). Sequential structure is captured only at the feature extraction stage via curvature and displacement statistics computed on the projected points. This is deliberate: an end-to-end sequential model (RNN/transformer over N_L points) would be more expressive but prone to overfitting on ~100 adversarial samples, and would sacrifice interpretability. More precisely, HPS is a "multi-layer contrastive projection with geometric summary statistics" — the trajectory metaphor describes the feature extraction, not the training objective.

**Margin M:** Sensitivity analysis over M ∈ {0.5, 1.0, 2.0, 4.0} is included. The appropriate value depends on the typical scale of Lorentz distances between projected vectors.

**Regularization:** Weight decay (1e-5) on the projection head, early stopping on validation loss (patience=10 epochs), and cosine learning rate decay from 1e-3 to 1e-5. Train/val loss curves are reported to verify convergence without overfitting.

**Geometry-aware gradient flow:** The projection head W is Euclidean, but gradients flow through the differentiable Lorentz lifting (x₀ = sqrt(1/κ + ||x_proj||²)). This means W is optimized to produce vectors whose norms and directions yield good Lorentz distances after lifting — the geometry influences the linear layer during training, not just at inference. This cross-boundary gradient flow is the mechanism by which hyperbolic structure shapes the learned representation, and is what we hypothesize distinguishes HPS-Full from a purely Euclidean trained baseline.

**Training data:** Only validated prompts — confirmed jailbreaks (model actually complied) vs confirmed benign (model responded normally). Most prior work trains on nominal labels, contaminating the positive class with failed attacks that are geometrically indistinguishable from benign prompts. Our validation step is a methodological contribution.

**Generalization protocol:** Train/test split by attack type (e.g., train on PAIR+WildGuard, evaluate on GCG+TAP).

### 3.3 Trajectory Feature Extraction

From the N_L projected points on the hyperboloid, we compute 12 features:

| # | Feature | Category | What It Measures |
|---|---|---|---|
| 1 | Mean radius | Radial | Projected norm magnitude (empirically correlates with class) |
| 2 | Max radius | Radial | Peak projected norm across layers |
| 3 | Min radius | Radial | Minimum projected norm |
| 4 | Std radius | Radial | Radial variation across layers |
| 5 | Radius range | Radial | Total radial traversal |
| 6 | Max curvature | Curvature | Sharpest trajectory bend (the "pivot") |
| 7 | Mean curvature | Curvature | Overall trajectory smoothness |
| 8 | Std curvature | Curvature | Curvature consistency |
| 9 | Spike location | Curvature | Which layer the pivot occurs (normalized 0–1) |
| 10 | Total displacement | Displacement | Geodesic distance first→last layer |
| 11 | Path length | Displacement | Total distance traveled along trajectory |
| 12 | Progress ratio | Displacement | Directness (displacement / path_length) |

**Note on radial interpretation:** The contrastive loss does not explicitly enforce hierarchy (unlike entailment losses in HyperRealm/HELM). After training, radial features capture depth-related variation in the learned projection that empirically separates classes — but the "abstraction depth" interpretation is a hypothesis, not a guaranteed property. Adding an entailment regularizer to enforce explicit hierarchy is a Phase 2 extension.

**Curvature computation:** Discrete curvature at interior point i is computed via the triangle inequality deviation in Lorentz geodesic distance:

```
κ_i = |d_L(p_{i-1}, p_i) + d_L(p_i, p_{i+1}) - d_L(p_{i-1}, p_{i+1})| / (d_L(p_{i-1}, p_i) + d_L(p_i, p_{i+1}))
```

This measures how much the path deviates from a geodesic at each point. κ_i ∈ [0, 1], with 0 meaning the three points are collinear on a geodesic. Note: the absolute value discards the direction of curvature (left vs. right turns are identical), which is acceptable for detection but limits mechanistic interpretation. On the Lorentz hyperboloid, geodesic triangles are "thinner" than Euclidean triangles (negative curvature), so high κ_i indicates the trajectory deviates from the manifold's natural hierarchical structure at that layer — consistent with the jailbreak pivot hypothesis, but not uniquely so.

**Inter-layer spacing:** Fisher-selected layers may have non-uniform gaps (e.g., layers 3, 7, 14, 22 have gaps of 4, 7, 8). Curvature between closely-spaced layers and widely-spaced layers has different meaning. We normalize curvature by inter-layer distance: κ_i^norm = κ_i / (layer_idx[i+1] - layer_idx[i-1]). This ensures curvature features are not confounded by the arbitrary spacing of selected layers.

**Progress ratio and dual-use:** Feature 12 (progress ratio) may be vulnerable to dual-use confusion — benign prompts about sensitive topics might also show low progress ratio due to internal deliberation. The 12-feature probe learns appropriate weighting, but we report per-feature importance to identify which features are most susceptible to dual-use false positives.

**12 features vs direct classification:** The 12-feature probe is an interpretable summary of the 8×257 = 2056-dimensional trajectory. We ablate against a direct classifier (logistic regression on the flattened trajectory) to verify the hand-crafted features are not a bottleneck. If the 12-feature probe matches the direct classifier, this is a strong result: the geometric summary statistics capture all discriminative information in an interpretable form.

### 3.4 Classification

Logistic regression on the 12-dimensional feature vector. Threshold τ calibrated for target FPR on validation set.

### 3.5 Verifying Each Component Contributes

To isolate what drives detection and rule out the possibility that simpler methods suffice, we include a comprehensive ablation:

**Core question: Does hyperbolic geometry add value, or is the signal already trivially available?**

| Config | Description | What It Tests |
|---|---|---|
| **Fisher-8 Raw** | Concatenate activations from 8 selected layers → logistic regression (no projection, no geometry) | Is the signal already linearly separable from layer selection alone? |
| **Euclidean-Trained** | Same d→d_p linear head, contrastive loss in L2 distance | Does training a projection help beyond raw concatenation? |
| **Nonlinear-Euclidean** | Same linear head + LayerNorm + tanh, contrastive in L2 | Does any nonlinearity work, or is Lorentz specifically needed? |
| **Hyperbolic-Naive** | Raw exponential map, no training (HPS-Lite) | Does geometry alone help without training? |
| **Hyperbolic-Trained** | d→d_p + Lorentz lift + contrastive in d_L (HPS-Full) | Full system |

**Possible outcomes and their implications:**

| Outcome | Implication | Paper Framing |
|---|---|---|
| Fisher-8 Raw ≈ all others | Signal is trivially separable; HPS adds engineering value only | Pivot to "trajectory monitoring is the contribution, geometry is optional" |
| Euclidean-Trained > Fisher-8 Raw, ≈ Hyperbolic-Trained | Contrastive projection helps, but geometry doesn't | Paper about trajectory detection with learned projection (drop hyperbolic framing) |
| Nonlinear-Euclidean ≈ Hyperbolic-Trained | Any nonlinearity works; Lorentz is not special | Weaken geometry claims, present Lorentz as one valid choice |
| Hyperbolic-Trained > Nonlinear-Euclidean > Euclidean-Trained | Lorentz geometry specifically helps beyond generic nonlinearity | Full hyperbolic narrative justified |

**Architectural ablations:**

| Ablation | Variants | What It Tests |
|---|---|---|
| Shared vs per-layer W | 1 shared head vs N_L separate heads | Is shared W a bottleneck? |
| 12 features vs direct classifier | 12-dim probe vs 2056-dim flattened logistic regression | Is the feature bottleneck losing signal? |
| Projection dimension | d_p ∈ {32, 64, 128, 256} | Overfitting risk at small sample sizes |
| Margin M | M ∈ {0.5, 1.0, 2.0, 4.0} | Sensitivity to contrastive margin |
| Token pooling | last vs mean vs last-5 | Best extraction strategy |

**Feature importance:** We report permutation importance on the 12 features to determine which geometric properties actually drive detection. If curvature features (6–9) contribute <20% of total importance, the "curvature spike" narrative is not supported by the data — the method works via other geometric properties (radial, displacement), and the framing should be revised accordingly.

(Section 5, Claim 2 is tested by the core ablation above.)

### 3.6 Configurations

| Configuration | Projection | Training | Purpose |
|---|---|---|---|
| **HPS-Lite** | Naive exponential map | Probe only | Baseline: does signal exist? |
| **HPS-Full** | Learned head (per-layer contrastive) | Head + probe | Full system |

---

## 4. Datasets

### Training Data (Validated)

All prompts validated against the target model before use:
- **Benign (class 0):** Only prompts where the model produces a normal, helpful response
- **Adversarial (class 1):** Only prompts where the model actually complies with the harmful request

**Target training set size:** 100 confirmed adversarial + 200 confirmed benign (balanced via oversampling during training). The projection head has ~1M parameters (4096×256); 100 adversarial examples is sufficient for the contrastive loss given the low-rank structure, but we acknowledge this is a limitation and report learning curves showing performance vs training set size.

| Source | Role | Notes |
|---|---|---|
| Alpaca-Cleaned | Benign | General instructions |
| GSM8K | Benign | Math word problems |
| WritingPrompts | Benign | Creative/stories |
| WildGuard (unharmful) | Benign | Real-world safe queries |
| Built-in emotional/code | Benign | Emotional support, technical |
| JailbreakBench artifacts (GCG) | Adversarial | Gradient-optimized suffixes |
| JailbreakBench artifacts (PAIR) | Adversarial | Semantically rewritten attacks |
| JailbreakBench artifacts (TAP) | Adversarial | Tree-of-attacks pruning |
| WildGuard (harmful) | Adversarial | Real-world attacks |
| AdvBench | Adversarial | Harmful requests (caveat: many are refused by well-aligned models even without safety training; our validation step filters these out, so AdvBench may contribute fewer confirmed breaks than expected) |

### Cross-Attack Generalization Protocol

| Split | Train | Test |
|---|---|---|
| A | PAIR + WildGuard | GCG + TAP |
| B | GCG + AdvBench | PAIR + TAP |
| C | TAP + WildGuard | GCG + PAIR |

Report mean ± std across splits.

### Dual-Use Evaluation Set

100+ sensitive-but-legitimate prompts drawn from WildGuard's "unharmful" subset filtered for security/medical/chemistry topics, supplemented with hand-curated examples. Used only for FPR evaluation, never training. We report both point estimate and 95% Clopper-Pearson confidence interval.

---

## 5. Key Claims (Falsifiable)

1. Jailbreaks produce geometrically distinct activation trajectories detectable before output generation.
2. Hyperbolic projection adds statistically significant discriminative power beyond (a) raw concatenated activations from selected layers, (b) a Euclidean-trained projection, and (c) a generic nonlinear projection. If any of these baselines match HPS-Full, the hyperbolic framing is not justified and the paper pivots to "trajectory-based detection with contrastive projection."
3. Trained projection separates adversarial intent from sensitive topics (target: <5% dual-use FPR).
4. Detection generalizes across attack types not seen during training.
5. Empirical layer selection via Fisher-ratio outperforms uniform spacing.
6. The curvature spike is a primary detection signal (testable via feature importance — if curvature features contribute <20%, this claim is falsified and the narrative shifts to radial/displacement features).

---

## 6. Limitations

- **White-box requirement**: Needs access to internal activations. Cannot protect closed-weight APIs.
- **Validated data dependency**: Requires finding prompts that actually break the target model. Well-aligned models resist most attacks.
- **Adaptive attacks**: Not yet tested against attackers who know the sentinel exists and optimize against it.
- **Multi-model generalization**: Pending validation on Qwen, Mistral.
- **Scale**: Current validation on ~100 prompts per class. Larger-scale evaluation needed.
- **Latency**: Theoretical <20ms. Wall-clock benchmarks pending.
- **Deployment base rates**: Like all safety classifiers, HPS's operational precision depends on the base rate of adversarial queries. At low base rates (<1%), even a low FPR produces many false positives in absolute terms (e.g., 95% TPR + 5% FPR at 0.1% base rate → ~1.9% positive predictive value). Deployment should combine HPS with other signals (input filters, output checks) or reserve it for high-risk contexts where the base rate is elevated (red-team monitoring, agentic tool-use pipelines).
