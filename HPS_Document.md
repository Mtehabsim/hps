# Hyperbolic Physiological Sentinel (HPS): Detecting Jailbreaks via Internal Trajectory Geometry

## One-Sentence Summary

HPS monitors the internal activation trajectory of an LLM during inference and detects jailbreak compliance via geometric features in hyperbolic space — a structural signature of the model's commitment to harmful generation that is invisible to input-side classifiers and output-side filters.

---

## 1. The Idea

Large language models can be tricked into producing harmful content through "jailbreak" attacks — carefully crafted prompts that bypass safety alignment. Current defenses either inspect the input text (easily fooled by obfuscation) or check the output (too late for agentic systems that act immediately).

We propose monitoring the model's internal reasoning trajectory during inference. When a model processes a jailbreak, its hidden states undergo a characteristic geometric deformation. We detect this deformation by projecting layer-by-layer activations into hyperbolic space and measuring trajectory features (radial position, curvature, displacement).

**Mechanistic grounding (DSH, Wu et al. 2026):** Wu et al. ("Knowing without Acting: The Disentangled Geometry of Safety Mechanisms in Large Language Models", arXiv:2603.05773) empirically demonstrate that safety in LLMs operates on two geometrically distinct axes — Recognition (v_H) and Execution (v_R) — that decouple in deep layers. A jailbreak succeeds by traversing from Recognition to compliant generation without triggering Execution.

**Why hyperbolic space:** Hyperbolic geometry's exponential volume growth naturally separates hierarchical levels — general/abstract concepts near the origin, specific/actionable directives near the boundary. Critically, hyperbolic geometry imposes a **structural inductive bias** that a Euclidean projection lacks: the radial coordinate has consistent semantic meaning (depth/specificity) regardless of attack type. This bias is what we hypothesized would help cross-distribution generalization.

---

## 2. Related Work

### Jailbreak Attacks
- **GCG** (Zou et al., 2023): Gradient-based suffix optimization → gibberish tokens
- **PAIR** (Chao et al., 2023): Iterative semantic rewriting → fluent attacks
- **JBC/AIM** (jailbreak chat templates): Role-play prompts ("act as Niccolo Machiavelli...")
- **prompt_with_random_search** (Andriushchenko et al., 2024): Adaptive suffix search via logprobs

### Input-Side Defenses
- **HyPE** (Maljkovic et al., arXiv:2604.06285, 2026): Hyperbolic SVDD on prompt embeddings (VLM safety)
- **LatentGuard, GuardT2I**: Embedding-based prompt classifiers
- **Limitation**: All input-side methods see only the surface prompt

### Activation-Based Safety Monitoring
- **Representation Engineering (RepE)** (Zou et al., 2024): Safety concepts are linearly separable at specific layers
- **Disentangled Safety Hypothesis (DSH)** (Wu et al., arXiv:2603.05773, 2026): Recognition and Execution axes decouple
- **LatentBiopsy** (arXiv:2603.27412, 2026): Training-free angular deviation detection
- **Latent Representation Framework** (arXiv:2602.11495, 2026): Tensor analysis of hidden states
- **AdaSteer** (Zhou et al., EMNLP 2025): Activation-threshold-based steering
- **Google AMS** (2026): Activation-based scanner for open-weight LLMs

### Hyperbolic Geometry for Language
- **HELM** (He et al., NeurIPS 2025): First fully hyperbolic LLM at billion scale; token embeddings exhibit negative Ricci curvature
- **HypLoRA** (Yang et al., NeurIPS 2025): Token embeddings have measurable δ-hyperbolicity, power-law radial structure
- **HyperRealm** (CVPR 2026): Poincaré ball VLM with entropy-driven entailment loss
- **HyCon** (arXiv:2603.14093, 2026): Hyperbolic concept steering via parallel transport (T2I)
- **ATLAS / Mixed-Curvature VLM** (CSS 2026): Different concept types prefer different geometries

### Our Position

| Defense Type | Example | What It Sees | Blind Spot |
|---|---|---|---|
| Input filter | HyPE, LatentGuard | Prompt text/embedding | Internally-resolved intent |
| Output filter | Llama Guard | Generated text | Too late for agents |
| Single-layer probe | RepE, ITI | One snapshot | Misses cross-layer dynamics |
| Angular deviation | LatentBiopsy | Per-layer angles | No learned projection |
| **Ours (HPS)** | — | Multi-layer trajectory in learned hyperbolic space | Requires white-box access |

---

## 3. Method

### Pipeline

```
Input prompt
    ↓
[Forward pass] → extract activations at N selected layers
    ↓
[Hyperbolic projection] → learned linear head + Lorentz lift
    ↓
[Trajectory features] → 12 geometric statistics
    ↓
[Classifier] → logistic regression
    ↓
SAFE / BLOCKED
```

### 3.1 Layer Selection (Fisher-Ratio)

Layers are selected empirically using a held-out calibration split (30 benign + 30 attacks). For each layer, we compute:

```
Fisher_score = ||μ_attack − μ_benign|| / (σ_attack + σ_benign)
```

The top 8 layers by Fisher score are selected. **Empirically discovered for Vicuna-13B**: layers `[0, 1, 2, 35, 36, 37, 38, 39]` — early embedding layers + late semantic layers.

### 3.2 Token Pooling

Three strategies tested via ablation: last token, mean over all tokens, mean of last 5 tokens. Empirical winner: **last token**.

### 3.3 Hyperbolic Projection (HPS-Full)

Each hidden state is projected onto the Lorentz hyperboloid:

```
x_proj = W · h · λ                         (linear projection, W ∈ ℝ^(d × 64))
x₀ = sqrt(1/κ + ||x_proj||²)               (time coordinate)
point = [x₀, x_proj] ∈ L^64_κ              (on the hyperboloid)
```

**Learnable parameters:**
- `W`: d → 64 linear projection (d=5120 for Vicuna-13B; ~330K params)
- `λ`: scale factor, initialized to 1/sqrt(64)
- `κ`: curvature, learnable in log-space, clamped [0.1, 10.0]
- `τ_l`: per-layer temperature for distance calibration (8 scalars)

**Loss:** Per-layer supervised contrastive loss with balanced pair sampling, summed across all selected layers:

```
L = (1/N_L) Σ_l L_layer(l)

L_layer(l) = (same_loss / n_same + diff_loss / n_diff) / 2
  where same_loss = Σ d_L²(x_i^l, x_j^l) for same-class pairs
        diff_loss = Σ max(0, M − d_L(x_i^l, x_j^l))² for diff-class pairs
```

Distance scaled by per-layer temperature `τ_l`. FP32 throughout for numerical stability.

### 3.4 Trajectory Features (12 Statistics)

| # | Feature | Category |
|---|---|---|
| 1–5 | mean/max/min/std radius, radius range | Radial |
| 6–9 | max/mean/std curvature, spike location | Curvature |
| 10–12 | total displacement, path length, progress ratio | Displacement |

### 3.5 Configurations Tested

| Config | Projection | Loss | Purpose |
|---|---|---|---|
| **Raw** | None (concatenate activations) | — | High-dim baseline (40,960 features, C=0.01 regularization) |
| **HPS-Lite** | Naive Lorentz lift, no training | — | Geometry alone |
| **Euclidean-Trained** | Linear head + L2 contrastive | Same loss in flat space | Training alone |
| **Nonlinear-Euclidean** | LayerNorm + tanh + L2 contrastive | Same loss in flat space | Generic nonlinearity |
| **Hyperbolic-Trained (HPS-Full)** | Linear head + Lorentz lift + geodesic contrastive | Loss in hyperbolic space | Full system |

---

## 4. Datasets

**Target model:** lmsys/vicuna-13b-v1.5 (40 layers, 5120 hidden dim, 4-bit quantization for memory)

### Benign Prompts (350 total — diverse domains)

| Source | Count | Type |
|---|---|---|
| tatsu-lab/alpaca | 100 | General instructions |
| openai_humaneval | 50 | Code completion |
| gsm8k | 100 | Math word problems |
| euclaise/writingprompts | 50 | Creative writing |
| allenai/winogrande | 50 | Tricky sentence completions |

### Adversarial Prompts (316 confirmed jailbreaks across 4 methods)

Pulled from JailbreakBench artifacts, then validated against Vicuna-13B (only kept prompts where the model actually complied):

| Method | Tested | Confirmed | ASR |
|---|---|---|---|
| prompt_with_random_search | 100 | 98 | 98.0% |
| JBC (role-play templates) | 100 | 84 | 84.0% |
| GCG (gradient suffixes) | 100 | 68 | 68.0% |
| PAIR (semantic rewrites) | 82 | 66 | 80.5% |
| **TOTAL** | **382** | **316** | **82.7%** |

### Refused Attacks (66 — analysis only, not used in training)

Prompts the model refused. Used to test the DSH trajectory hypothesis: if the trajectory theory is correct, refused attacks should score between benign and successful attacks.

### Dual-Use Set (20 — sensitive but legitimate)

Hand-curated cybersecurity/chemistry/medical educational queries. Evaluation only.

### Calibration Split

15% of attacks + benign reserved for layer selection only (Fisher-ratio computation). Disjoint from training and test sets.

---

## 5. Validation Results

### Setup
- **Model**: Vicuna-13B-v1.5 (40 layers)
- **Evaluation**: 5-fold stratified cross-validation + cross-attack generalization
- **Layers**: Top-8 by Fisher score = `[0, 1, 2, 35, 36, 37, 38, 39]`
- **Pooling**: Last token (best by ablation)
- **d_p**: 64 (16x compression from 5120)

### Result 1: In-Distribution Performance (Same-Type Train/Test)

All trained methods achieve perfect AUROC = 1.000. The dataset is linearly separable in the selected layer space — any reasonable method solves it.

| Method | AUROC | FPR@95TPR | F1 | Dual-Use FPR |
|---|---|---|---|---|
| Fisher-8 Raw | 1.000 | 0.000 | 0.991 | — |
| Euclidean-Trained | 1.000 | 0.000 | 1.000 | — |
| Nonlinear-Euclidean | 1.000 | 0.000 | 0.998 | — |
| HPS-Lite (naive) | 0.911 | 0.303 | 0.547 | — |
| **HPS-Full** | **1.000** | **0.000** | **0.998** | **0.250** |

**Interpretation:** In-distribution performance saturates. Cannot distinguish methods on this evaluation.

### Result 2: Cross-Attack Generalization (THE CRITICAL TEST)

Train on 3 attack methods, test on the held-out 4th. This breaks the saturation and reveals which methods truly generalize.

| Held-Out Method | n_test | Raw AUROC | Euclidean AUROC | **Hyperbolic AUROC** |
|---|---|---|---|---|
| GCG | 68 | 1.000 | 1.000 | 0.970 |
| JBC | 84 | 1.000 | 0.048 | 0.761 |
| PAIR | 66 | 0.993 | 0.176 | 0.639 |
| prompt_with_random_search | 98 | 1.000 | 0.003 | 0.997 |
| **MEAN** | — | **0.998** | **0.307** | **0.842** |

**Critical findings:**
- **Raw generalizes near-perfectly.** High-dim activations with strong L2 regularization learn universal features.
- **Euclidean catastrophically fails** (0.307 mean — worse than random on JBC, PAIR, Random Search). The 12-feature compression loses generalizable information; the classifier overfits to method-specific surface features.
- **Hyperbolic recovers most of the performance** (0.842). The hyperbolic geometric prior provides a regularizing structure that Euclidean lacks. **Δ = +0.535 over Euclidean.**

**Conclusion:** Hyperbolic geometry's value is specifically as a **structural prior for compressed representations**. When features are compressed (12-dim from 40,960), the inductive bias matters enormously. With raw high-dim features, the bias provides no advantage because regularization alone suffices.

### Result 3: Layer Ablation

| Layer Config | AUROC | F1 |
|---|---|---|
| Early only (0–2) | 0.988 | 0.955 |
| Late only (35–39) | 1.000 | 0.992 |
| Mid only (20–30) | 1.000 | 0.992 |
| Late+Mid (20–39) | 1.000 | 0.992 |
| Combined (0–2 + 35–39) | 1.000 | 0.985 |

**Finding:** Late layers (semantic processing) and mid layers (deeper computation) both carry sufficient signal independently. The signal is NOT purely lexical — it persists in semantic-processing layers even without early embedding layers.

### Result 4: Per-Method Activation Norms

| Layer Group | Benign | GCG | JBC | PAIR | Random Search |
|---|---|---|---|---|---|
| Early (0–2) | 28.47 ± 52.98 | 6.96 ± 0.48 | 6.60 ± 0.35 | 5.22 ± 0.24 | 6.34 ± 0.42 |
| Late (35–39) | 313.57 ± 34.38 | 293.92 ± 25.44 | 309.51 ± 12.44 | 317.36 ± 16.52 | 335.28 ± 33.54 |

**Finding:** At early layers, attack methods cluster TIGHTLY (std ~0.4) at much smaller norms than benign (std ~53). The signal at early layers is "anomalously low diversity" — attacks share a distinctive activation pattern that benign prompts don't. At late layers, distributions overlap significantly, suggesting the lexical detection at early layers explains most of the signal.

### Result 5: Refused Attack Scoring (DSH Test)

| Category | Mean Score | Std | Range |
|---|---|---|---|
| Benign | 0.0016 | 0.0029 | [0.000, 0.038] |
| **Refused attacks** | **0.9906** | 0.0179 | [0.890, 1.000] |
| Successful attacks | 0.9983 | 0.0022 | [0.978, 1.000] |

**Finding:** Refused attacks score **identically** to successful attacks (0.99 vs 0.998). The detector cannot distinguish "model recognized harmful intent and refused" from "model recognized harmful intent and complied."

**Conclusion (DSH hypothesis falsified for this method):** The classifier is detecting **harmful prompt content**, not **the model's compliance trajectory**. The trajectory pivot story does not hold — the model's internal resistance is geometrically invisible to this approach.

### Result 6: Feature Importance (Permutation, Held-Out Split)

| Category | % of Total Importance |
|---|---|
| Radial (mean_r dominates at 98.7%) | **99.0%** |
| Curvature | **0.0%** |
| Displacement | **1.0%** |

**Finding:** The "curvature spike" narrative is **falsified by the data**. Curvature features contribute zero to detection. The signal is entirely in radial position (mean activation norm in projected space). The "trajectory bending" story does not hold; the method is a "compressed multi-layer norm classifier."

### Result 7: Direct Classifier vs 12-Feature Probe

| | AUROC |
|---|---|
| 12-feature probe | 1.000 |
| Direct classifier (8×65 = 520 features) | 1.000 |

**Finding:** The 12 hand-crafted features capture all discriminative information. No bottleneck.

---

## 6. What These Results Actually Show

### What's Supported

✅ **Activation monitoring detects jailbreaks reliably.** Cross-attack mean AUROC of 0.998 with Raw features. The signal is real and universal across attack types.

✅ **Hyperbolic geometric prior aids cross-distribution generalization in compressed representations.** When the projection is bottlenecked to 12 features, hyperbolic dramatically outperforms Euclidean (0.842 vs 0.307). The geometry imposes a useful inductive bias.

✅ **Layer selection via Fisher-ratio identifies meaningful detection layers** that span both lexical (early) and semantic (late) processing.

### What's NOT Supported

❌ **Curvature spike / trajectory pivot narrative.** Curvature contributes 0% to detection. The signal is radial, not bent.

❌ **DSH compliance vs recognition distinction.** Refused attacks score identically to successful ones. The detector finds "harmful content," not "model is about to comply."

❌ **HPS-Full > Raw.** Raw activation features generalize better than any trained low-dimensional projection.

❌ **HPS-Full > simple methods on in-distribution data.** All methods saturate at 1.000.

---

## 7. Honest Paper Framing

The original "geodesic pivot" thesis is not supported. The defensible findings are:

> "We demonstrate that LLM jailbreaks produce universal activation signatures detectable via simple linear classification on Fisher-selected layer activations, with cross-attack generalization AUROC of 0.998 averaged across four attack methods (GCG, PAIR, JBC, prompt_with_random_search). When compressing to a low-dimensional projection, however, generalization collapses for Euclidean methods (mean AUROC 0.307) but is largely preserved by a hyperbolic projection (mean AUROC 0.842), demonstrating that hyperbolic geometric priors provide a meaningful inductive bias for cross-distribution generalization in compressed representations. Notably, the signal is dominated by radial features (98.7% of importance) rather than trajectory curvature (0%), and the detector cannot distinguish successful jailbreaks from prompts the model refused — indicating that current activation-based detection captures the harmful content of prompts rather than the model's compliance decision."

This is a **real publishable finding**. Three contributions:

1. **Engineering contribution:** Universal jailbreak detector via simple regularized regression on selected activations (0.998 AUROC, generalizes across attack types).
2. **Scientific contribution:** Hyperbolic geometric prior aids generalization in compressed representations — first empirical demonstration of this for safety detection.
3. **Negative result:** Activation-based detection in this form detects content, not compliance. The DSH-inspired "trajectory pivot" hypothesis is not supported by the data.

---

## 8. Key Implementation Details

### Numerical Stability
- All geometric operations in FP32 (Lorentz lift, distance computation)
- Curvature κ clamped [0.1, 10.0] in log-space
- Per-layer temperature τ clamped [0.01, 10.0]
- Gradient clipping in Adam with weight decay 1e-5

### Training
- Adam, lr=1e-3, weight decay=1e-5
- Early stopping on training loss, patience=20, max 200 epochs
- Per-layer balanced contrastive loss with margin M=2.0
- Calibration split: 15% of data, disjoint from train/test

### Memory
- Vicuna-13B in 4-bit quantization (BitsAndBytesConfig, NF4)
- Activations extracted in fp16 compute, stored as fp32 numpy
- 14 layers × 666 prompts × 5120 dim ≈ 200 MB cache

---

## 9. Limitations

- **White-box requirement:** Needs access to internal activations. Cannot protect closed-weight APIs.
- **Trained on saturated dataset:** All in-distribution evaluations hit AUROC=1.000, limiting what the in-distribution comparison can show.
- **Single model tested:** Vicuna-13B only. Generalization to Llama-3, Qwen, Mistral pending.
- **Adaptive attacks not tested:** No evaluation against attackers who know HPS exists and optimize against it.
- **Compliance vs recognition:** Cannot distinguish refused attacks from successful ones — the DSH hypothesis is not empirically supported by this method.
- **Dual-use FPR is high in some configurations:** 25% on dual-use prompts at the calibrated threshold (in-sample). Need broader dual-use evaluation.
- **Deployment base rates:** At low real-world jailbreak rates, even low FPR produces many false positives in absolute terms.

---

## 10. Execution Phases

### Phase 1 (COMPLETED) — Go/No-Go

| Step | Result |
|---|---|
| Validate attacks against target model | 316 confirmed jailbreaks across 4 methods |
| Layer discovery via Fisher-ratio | Top-8 = [0, 1, 2, 35, 36, 37, 38, 39] |
| Pooling ablation | Last token wins |
| In-distribution evaluation | All trained methods AUROC=1.000 (saturated) |
| Cross-attack generalization | Raw 0.998, Hyperbolic 0.842, Euclidean 0.307 |
| Feature importance | Radial 99%, Curvature 0% |
| Refused attack analysis | Score identical to successful (DSH falsified) |

**Decision:** GO with revised framing (engineering + generalization + negative result)

### Phase 2 (PROPOSED) — Publication-Ready

1. **Multi-model validation** (Llama-3, Qwen, Mistral)
2. **Adaptive attack evaluation** (attacker optimizes against HPS)
3. **Larger dual-use evaluation** (500+ prompts, Clopper-Pearson CI)
4. **Mixed-curvature ablation** (Euclidean + Spherical + Hyperbolic with router, à la ATLAS)
5. **Latency benchmarks**
6. **Out-of-distribution attacks** (multi-turn, encoded payloads, etc.)
