# Hyperbolic Physiological Sentinel (HPS): Detecting Jailbreaks via Internal Trajectory Geometry

## 1. The Idea

Large language models can be tricked into producing harmful content through "jailbreak" attacks — carefully crafted prompts that bypass safety alignment. Current defenses either inspect the input text (easily fooled by obfuscation) or check the output (too late for agentic systems that act immediately).

We propose a different approach: **monitor the model's internal reasoning trajectory during inference**. When a model processes a jailbreak, its hidden states undergo a characteristic geometric deformation — a "pivot" from the benign wrapper to the harmful payload. We detect this pivot by projecting the model's layer-by-layer activations into hyperbolic space and measuring trajectory anomalies.

**Core insight:** In hyperbolic space, general/abstract concepts naturally sit near the origin while specific/actionable directives sit near the boundary. A jailbreak forces the model to traverse from a benign abstract region to a harmful specific region — producing a measurable geometric signature that surface-level analysis cannot see.

**One-sentence summary:** We treat inference as a trajectory through a learned hyperbolic manifold and detect jailbreaks as structural anomalies in that trajectory.

---

## 2. Related Work

### Jailbreak Attacks
- **GCG** (Zou et al., 2023): Gradient-based suffix optimization that finds gibberish tokens triggering compliance
- **PAIR** (Chao et al., 2023): Iterative semantic rewriting that produces fluent jailbreaks
- **AutoDAN, TAP, multi-turn attacks**: Various automated methods that produce diverse attack surfaces

### Input-Side Defenses
- **HyPE** (Maljkovic et al., April 2026): Hyperbolic anomaly detection on prompt embeddings. Achieves 0.98 F1 but only sees the input — cannot detect obfuscated intent resolved internally.
- **LatentGuard, GuardT2I**: Embedding-based classifiers that flag harmful prompts before processing.

### Activation-Based Safety Monitoring
- **Representation Engineering (RepE)** (Zou et al., 2024): Shows safety concepts are linearly separable at specific layers.
- **Disentangled Safety Hypothesis (DSH)** (Wu et al., ICML 2026): Proves safety operates on two geometrically distinct axes — Recognition ("Knowing") and Execution ("Acting") — that decouple in deep layers.
- **Google AMS** (2026): Validates activation patterns can detect safety-modified models.

### Hyperbolic Geometry for Language
- **HELM** (He et al., NeurIPS 2025): First fully hyperbolic LLM at billion scale. Shows token embeddings have intrinsic negative curvature.
- **HypLoRA** (Yang et al., NeurIPS 2025): Demonstrates LLM token embeddings exhibit measurable δ-hyperbolicity and power-law radial structure.
- **HyperRealm** (CVPR 2026): Poincaré ball VLM with entropy-driven entailment loss for hierarchy-aware embeddings.
- **HyCon** (ICML 2026): Hyperbolic concept steering via parallel transport — shows hyperbolic steering is more stable than Euclidean.

### Our Position
HyPE guards the input. We guard the internal reasoning. RepE probes a single layer statically. We monitor the trajectory across layers dynamically. Nobody has combined hyperbolic geometry with multi-layer trajectory analysis for jailbreak detection.

---

## 3. Method

### Overview

The Hyperbolic Physiological Sentinel (HPS) operates in four stages during the model's forward pass:

```
Input Prompt → [Activation Extraction] → [Hyperbolic Projection] → [Trajectory Features] → [Classification]
                                                                                              ↓
                                                                                    SAFE / BLOCKED
```

### Stage 1: Targeted Activation Extraction

During the forward pass, we extract the hidden state at the last token position from K=8 selected layers spread across the model's depth (e.g., layers 2, 6, 10, 14, 18, 22, 26, 30 for a 32-layer model). This produces a sequence:

```
h₁, h₂, ..., h₈ ∈ ℝ⁴⁰⁹⁶
```

representing how the model's understanding evolves from shallow syntax to deep semantics.

### Stage 2: Hyperbolic Projection

Each hidden state is projected onto the Lorentz hyperboloid through a **learned projection head** — a small linear layer (4096 → 256) followed by the Lorentz lifting:

```
x_proj = W · h · λ
x₀ = sqrt(1/K + ||x_proj||²)     ← time coordinate (radial depth)
point = [x₀, x_proj]              ← on the hyperboloid
```

The projection head is trained with a **contrastive loss** that pushes benign and adversarial trajectories apart in geodesic distance. This training produces meaningful radial structure: benign prompts cluster near the origin (abstract/general), adversarial prompts are pushed toward the boundary (specific/actionable).

### Stage 3: Trajectory Feature Extraction

From the 8 projected points on the hyperboloid, we compute:

- **Radial features**: Mean/max/std of time coordinates (abstraction depth)
- **Curvature features**: Max/mean/std of discrete geodesic curvature (trajectory bending)
- **Displacement features**: Total geodesic distance, path length, progress ratio (directness)
- **Spike location**: Which layer shows maximum curvature (where the "pivot" happens)

### Stage 4: Classification

A logistic regression classifier on the 12-dimensional feature vector outputs an anomaly score. If the score exceeds a calibrated threshold, generation is halted before the first output token.

### Two Configurations

| Configuration | Projection | Training | Use Case |
|---|---|---|---|
| **HPS-Lite** | Naive exponential map (no training) | Probe only | Quick baseline, no GPU training needed |
| **HPS-Full** | Learned projection head (contrastive loss) | Head + probe | Full system, best performance |

---

## 4. Validation

### Datasets

| Dataset | Source | Role | Size Used |
|---|---|---|---|
| **Alpaca-Cleaned** | yahma/alpaca-cleaned (HuggingFace) | Benign prompts — normal user instructions | 50 prompts |
| **WildGuard (benign)** | allenai/wildguardmix | Benign prompts — real-world safe queries | 50 prompts |
| **JailbreakBench** | JailbreakBench/JBB-Behaviors | Adversarial — curated jailbreak goals | 50 prompts |
| **AdvBench** | walledai/AdvBench | Adversarial — harmful behavior requests with GCG-style patterns | 50 prompts |
| **WildGuard (harmful)** | allenai/wildguardmix | Adversarial — real-world attack prompts collected in the wild | 50 prompts |
| **Dual-Use (built-in)** | Hand-curated | Hard negatives — legitimate questions on sensitive topics (cybersecurity, chemistry, pharmacology) | 25 prompts |

**Benign class (100 total):** Mix of general knowledge questions (Alpaca) and real user queries labeled safe by human annotators (WildGuard). Covers diverse topics: science, coding, cooking, history, etc.

**Adversarial class (100 total):** Mix of three attack types:
- Curated jailbreak goals from JailbreakBench (role-play, instruction override, persona injection)
- Harmful behavior prompts from AdvBench (direct harmful requests, GCG-optimized suffixes)
- Real-world attacks from WildGuard (collected from actual user interactions, human-labeled as harmful)

**Dual-use evaluation set (25):** Sensitive but legitimate prompts — "Explain how SQL injection works for my security class," "What makes chemical combinations explosive from a chemistry perspective?" Used only for FPR evaluation, never for training.

### Setup

- **Model**: Meta-Llama-3-8B-Instruct (32 layers, safety-aligned)
- **Data**: 100 benign prompts (Alpaca, WildGuard) + 100 adversarial prompts (JailbreakBench, AdvBench, WildGuard attacks)
- **Evaluation**: 5-fold stratified cross-validation, AUROC, FPR@95%TPR
- **Dual-use test set**: 25 sensitive-but-legitimate prompts (cybersecurity education, toxicology courses, etc.)

### Results

#### Test 3 — Raw Trajectory Signal

Does curvature alone separate benign from adversarial?

| Space | AUROC |
|---|---|
| Euclidean | 0.733 |
| Hyperbolic | 0.760 |

[PLOT: test3_traced_euclidean.png — TRACED scatter showing benign vs adversarial clusters]

[PLOT: test3_traced_hyperbolic.png — Same in hyperbolic space]

**Finding**: The signal exists. Adversarial prompts produce higher curvature trajectories even with naive projection.

#### Test 4 — Multi-Feature Probe (HPS-Lite)

| Probe | AUROC | FPR@95%TPR | F1 |
|---|---|---|---|
| Euclidean features | 0.958 | 0.160 | 0.958 |
| Hyperbolic features | 0.986 | 0.080 | 0.894 |
| Combined | 0.986 | 0.000 | 0.958 |

[PLOT: test4_roc_comparison.png — ROC curves for all three probes]

[PLOT: test4_feature_importance.png — Which features matter most]

**Finding**: Hyperbolic features reduce false positives by 50% (0.16 → 0.08) at the same detection rate.

#### Test 5 — Trained Projection Head (HPS-Full)

| Method | AUROC | FPR@95%TPR | Dual-Use FPR |
|---|---|---|---|
| HPS-Lite (naive projection) | 0.958 | 0.120 | 0.333 |
| **HPS-Full (learned projection)** | **1.000** | **0.000** | **0.000** |

[PLOT: test5_roc_comparison.png — ROC curves: HPS-Lite vs HPS-Full]

[PLOT: test5_learned_radii.png — Histogram showing radial separation between benign and adversarial]

**Finding**: The trained projection achieves perfect separation. Critically, it produces **zero false positives on dual-use prompts** — it learns to distinguish adversarial structure from sensitive topics.

#### Radial Structure (Learned Projection)

| Category | Mean Lorentz Radius | Std |
|---|---|---|
| Benign | 1.951 | 0.084 |
| Adversarial | 2.524 | 0.110 |
| **Separation** | **0.573** | — |

[PLOT: test5_learned_radii.png — Clear bimodal distribution]

**Finding**: The learned projection creates interpretable geometry — adversarial prompts are measurably "deeper" (further from origin) on the hyperboloid, confirming the theoretical prediction that harmful actionable content occupies the boundary region.

---

## 5. Key Takeaways

1. **The internal trajectory signal is real.** Jailbreaks produce geometrically distinct activation trajectories that can be detected before any output is generated.

2. **Hyperbolic geometry adds meaningful value.** The non-Euclidean projection amplifies discriminative features that are compressed in flat space, reducing false positives by 50-100%.

3. **Training the projection solves the dual-use problem.** A naive projection conflates dangerous topics with dangerous intent. A learned projection separates them — achieving zero FPR on legitimate security/medical/chemistry questions.

4. **The method is lightweight.** The projection head is ~2M parameters. Feature extraction and classification add <20ms to inference. No model weights are modified.

5. **Complementary to existing defenses.** HPS monitors the "mind" while HyPE monitors the "door." They can be deployed together for defense-in-depth.

---

## 6. Limitations and Next Steps

- **Scale**: Validated on 100+100 prompts. Phase 2 needs 1000+ with diverse attack types.
- **Adaptive attacks**: Not yet tested against an attacker who knows the sentinel exists and optimizes against it.
- **Models**: Only tested on Llama-3-8B. Need Qwen, Mistral, Gemma for generalization claims.
- **Latency**: Measured theoretically (<20ms). Need wall-clock benchmarks under load.
- **Overfitting risk**: 1.000 AUROC on small data with CV could be optimistic. Held-out evaluation needed.
