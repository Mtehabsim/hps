# Post-Meeting Plan: Mentor Draft + Adaptive Attack Investigation

**Date:** 2026-06-05 (post-meeting)
**Goal:** Two parallel deliverables based on meeting outcomes
**Timeline:** 3-4 weeks total

---

# Task A — Comprehensive Mentor Draft

**Deliverable:** A focused research document covering everything you've done, suitable for a mentor who has read the 4-paper Tier 1 reading list. Length: 15-25 pages.

**Format:** Markdown (easy to iterate, convert to LaTeX later for paper)

**Estimated time:** 8-12 hours of focused writing, 1-2 days elapsed

## A.1 Document Structure (9 sections)

The structure follows a natural research narrative arc:

> Setup (Sec 1-2) → Tension (Sec 3) → Investigation (Sec 4) → Resolution (Sec 5) → Surprise (Sec 6) → Limitations (Sec 7) → Future (Sec 8) → Reference (Sec 9)

This is the classical scientific story structure: hypothesis, suspicious results, audit, fixes, re-evaluation, unexpected findings, honest limitations.

### Section 1: Hypothesis & Motivation (1 page)

**Goal:** Set up the research question and theoretical justification

Content:
- The original idea: why hyperbolic geometry should help for jailbreak detection
- HypLoRA (Yang et al., NeurIPS 2025) and HELM (He et al., NeurIPS 2025) as foundation
  - LLM token embeddings exhibit empirical δ-hyperbolicity
  - Power-law radial structure / negative Ricci curvature in embeddings
  - Hyperbolic spaces have exponential volume growth (matches tree branching)
- The naive prediction: HPS should beat Euclidean if safety has hierarchical structure
- Specific testable claim: harmful content should occupy higher radial position in Lorentz space

### Section 2: System Description (2-3 pages)

**Goal:** Describe what we built in technical detail

Content:
- **HPS architecture in detail:**
  - Lorentz projection W ∈ ℝ^(d×64)
  - 12 trajectory features (5 radial: mean/max/min/std/range; 4 curvature: triangle inequality bending; 3 displacement: distance/path length/straightness)
  - Per-layer-temperature contrastive loss
  - Layer selection rationale: [0, 2, 17, 24, 28, 31] from systematic ablation (TEST 7 finding — spread layers achieve AUROC=1.000 vs Fisher-discovered layers at 0.925)
  - Curvature κ=0.1 frozen (TEST 9 finding — learnable κ unstable, sweep over {0.1, 0.5, 1.0, 2.0, 10.0})
  - Training: 50 epochs, lr=1e-3, weight_decay=1e-5

- **HPS-Euclidean ablation:**
  - SAME architecture, FLAT geometry
  - Per-layer scale + learnable margin to match HPS parameter count exactly
  - 262K parameters (matched)
  - Critical control: rules out "more parameters helped"

- **C4 baseline:**
  - Mean-pool 6 layers' last-token activations + StandardScaler + LR
  - 4,097 parameters
  - Adapted from Anthropic Cheap Monitors with axis swap

- **MTP baseline (Anthropic exact reproduction):**
  - Faithful reproduction of Cunningham et al. mean-token probe
  - Single layer probed (we test layers [0, 2, 17, 24, 28, 31] separately)
  - L2 regularization λ=1e4 (their Appendix D)
  - Standardized features

**Comparison table:**

| Method | Parameters | Geometry | Aggregation | Training |
|---|---|---|---|---|
| HPS | 262K | Lorentz hyperboloid | Layer trajectory + 12 features | Contrastive |
| HPS-Euclidean | 262K | Flat | Layer trajectory + 12 features | Contrastive |
| C4 | 4,097 | Flat | Mean across layers, last token | Discriminative LR |
| MTP @ Lk | 4,097 | Flat | Mean across tokens, layer k | Discriminative LR |

### Section 3: Initial Results & Why We Didn't Trust Them (1 page)

**Goal:** The dramatic turning point — show the suspicious pattern that triggered the audit

Content:
- All methods hit AUROC > 0.99 on standard benchmark
- Per-method initial numbers (HPS=1.000, C4=1.000, etc.)
- "These results looked too good to be true"
- Permutation test passed (random labels → AUROC=0.48), so it wasn't pure noise
- But three signs of trouble:
  1. Methods couldn't be distinguished (saturation)
  2. Small variations in benign data caused large changes
  3. Even controls (HPS-Euclidean) hit ceiling
- Decision: audit methodology before accepting results

### Section 4: Methodology Audit — The Three Confounds (3-4 pages)

**Goal:** The methodology contribution — three field-wide issues we identified

#### 4.1 Length confound (1.5 pages)

- **Discovery:** Length-only classifier achieves AUROC = **0.992** on standard benchmark (Alpaca harmless ~60 chars vs attacks ~218 chars weighted mean)
- **Per-attack analysis:**

| Attack | Mean length | Length-only AUROC |
|---|---|---|
| puzzler | 2,193 chars | 1.000 |
| saa | 473 chars | 1.000 |
| drattack | 437 chars | 1.000 |
| base64 | 124 chars | 1.000 |
| ijp | 460 chars | 0.999 |
| autodan | 77 chars | 0.999 |
| pair | 72 chars | 0.997 |
| zulu | 50 chars | 0.976 |
| gcg | 35 chars | 0.971 |

- **Affected papers:** HSF (WWW 2025), JBShield (USENIX 2025), GradSafe (ACL 2024), RTV — all use Alpaca + AdvBench-style setup
- **Anthropic Cheap Monitors did this right:** WildChat + OR-Bench Hard (varied lengths)
- **Fix:** Built diverse benign dataset (5,905 prompts from 9 sources)
- **After fix:** Length-only AUROC drops to **0.318**

#### 4.2 Train/test contamination (0.5 pages)

- **Hash-based analysis:** SHA-256 of prompts identifies duplicates
- **Found:** 15/1304 test attack prompts (1.15%) appear in training set
- **Source:** ijp (8 prompts), saa (6), zulu (1)
- **Fix:** Deduplication via `investigate_contamination.py` → `llama3_attacks_clean.json` (6,474 prompts after dedup)

#### 4.3 max_length truncation confound (1.5 pages)

- **The bug:**
  ```
  Original cache:    benign max_length=512    attacks max_length=512    consistent
  Diverse cache:     benign max_length=2048   attacks REUSED at 512     MISMATCH
  ```
- **Symptom:** Norm-only AUROC = **1.000** on Llama-3 fixed cache (vs 0.917 baseline)
- **Layer 5 norm comparison:**

| | Old (consistent) | Diverse (mismatch) |
|---|---|---|
| Benign mean norm | 155.6 | **35.5** |
| Attack mean norm | 153.0 | 153.0 |
| Ratio | 1.0× | **4.3×** |

- **Mechanism explanation:**
  - At token position 50 vs 2048: different positional embeddings (Llama-3 uses RoPE)
  - At token position 50 vs 2048: different cumulative attention integration
  - Long benign sequences truncated at 2048 produce different last-token representations than short benign at 512
- **Fix:** Re-extracted attacks at max_length=2048 → norm-only AUROC drops to **0.761**
- **Permutation test:** Random labels → AUROC=0.498 (chance), real labels → AUROC=0.998 — confirms genuine semantic signal beyond confound
- **Methodology lesson:** All cached activations must use consistent tokenization parameters (max_length, padding, chat template)

### Section 5: Re-evaluation After Fixes (3-4 pages)

**Goal:** The actual findings after methodology corrections

#### 5.1 Geometric hypothesis test

- **Setup:** 13 configurations tested (5 seeds × 4 epoch checkpoints × 4 κ values)
- **Result:** **0/13 inversions** — all configurations show ben_median < atk_median

| Configuration | Benign median | Attack median | Δ (ben - atk) |
|---|---|---|---|
| Seed 42, κ=0.1, ep=50 | 3.20 | 3.50 | -0.30 |
| Seed 43-46 | 3.20 | 3.50 | -0.30 to -0.32 |
| Epoch 5 | 3.23 | 3.27 | -0.04 |
| Epoch 50 | 3.20 | 3.50 | -0.30 |
| κ=0.5 | 1.57 | 1.96 | -0.39 |
| κ=1.0 | 1.36 | 1.53 | -0.17 |
| κ=2.0 | 1.37 | 1.42 | -0.05 |

- **Interpretation:** Hyperbolic prior is theoretically validated. Original "13/13 inversion" was a length-confound artifact.

#### 5.2 Performance comparison (4-method results)

- **Setup:** n=5 random seeds, n_bootstrap=10,000 iterations

| Method | AUROC | TPR @ 5%FPR | TPR @ 1%FPR |
|---|---|---|---|
| **MTP @ L17** (Anthropic exact, best layer) | 0.9988 | 0.9946 | 0.9799 |
| **C4** (cross-layer, our variant) | 0.9986 | 0.9954 | 0.9776 |
| **HPS** (Lorentz contrastive) | 0.9971 ± 0.0001 | 0.9914 ± 0.0003 | — |
| **HPS-Euclidean** (parameter-matched flat) | 0.9680 | 0.9311 | — |

- **Statistical tests:**
  - ΔAUROC (HPS - C4) = -0.0010, p=0.036 (significant but trivially small)
  - ΔTPR5 (HPS - C4) = -0.0019, p=0.601 (NOT SIGNIFICANT)
  - McNemar's test: p=0.755 (NOT SIGNIFICANT)
  - Cohen's d: -0.039 (negligible)

#### 5.3 Cold-start sweep

| N | HPS | HPS-Euclidean | C4 |
|---|---|---|---|
| 45 | 0.70 | 0.61 | 0.96 |
| 90 | 0.82 | 0.61 | 0.98 |
| 225 | 0.95 | 0.89 | 0.98 |
| 450 | 0.90 | 0.83 | 0.99 |
| 900 | 0.98 | 0.94 | 0.99 |
| 2250 | 0.97 | 0.93 | 0.99 |

- **HPS - HPS-Euclidean:** +0.049 TPR average (geometric prior provides real benefit)
- **HPS - C4:** -0.015 TPR (linear probe still wins)

#### 5.4 Prediction agreement

- **Test set n=1,295 attacks, mean of 3 seeds:**
  - HPS catches that C4 misses: **0 examples**
  - C4 catches that HPS misses: 21 examples
  - Pearson(HPS, C4) score correlation: 0.958
- **Ensemble test:**
  - OR-gate ensemble TPR = 0.995 (= C4 alone)
  - OR-gate ensemble FPR = 0.103 (above 5% target — strictly worse)
- **Implication:** HPS is empirically a noisy subset of C4.

### Section 6: Cross-Model Evaluation (2 pages)

**Goal:** The most striking finding — Vicuna catastrophic failure

#### 6.1 Vicuna-13B failure

| Attack | N | HPS rate | C4 rate | gap |
|---|---|---|---|---|
| **gcg** | 171 | **0.076** | 0.994 | **+0.92** |
| ijp | 164 | 0.329 | 0.933 | +0.60 |
| pair | 167 | 0.389 | 0.958 | +0.57 |
| puzzler | 13 | 0.462 | 1.000 | +0.54 |
| zulu | 185 | 0.632 | 1.000 | +0.37 |
| autodan | 167 | 0.701 | 1.000 | +0.30 |
| base64 | 166 | 0.922 | 1.000 | +0.08 |
| drattack | 91 | 0.956 | 1.000 | +0.04 |
| saa | 170 | 0.994 | 1.000 | +0.01 |

- **Mean detection across 9 attacks:** HPS = 0.61, C4 = 0.99
- **Compare to Llama-3-Instruct:** HPS = 0.99, C4 = 0.99 (same data, same code)

**Imbalance ablation (controls for sample size):**
- Vicuna BALANCED 253+253: HPS TPR=0.67, C4 TPR=0.99 (gap persists)
- Llama-3 with Vicuna-style imbalance: HPS TPR=0.999 (Vicuna-specific failure)

#### 6.2 Alignment-mediated failure hypothesis

- **Critical comparison:** Same architecture, same hyperparameters, only LLM differs
  - Llama-3-Instruct (RLHF aligned): 100% GCG detection
  - Vicuna-13B-v1.5 (SFT only): 7.6% GCG detection

- **Mechanism:** RLHF concentrates harm-discriminative features
  - With strong alignment: harm signal compresses into compact regions
  - HPS's 12-feature compression preserves concentrated signal ✓
  - Without RLHF: harm features remain diffuse
  - HPS's compression FILTERS OUT diffuse signal ✗
  - C4's 4096-dim representation preserves both ✓

- **Why this is novel evidence:**
  - First explicit demonstration that activation-based defense effectiveness depends on alignment training strength
  - Has implications for deploying detection on weakly-aligned models

### Section 7: Adversarial Robustness Limitation (1-2 pages)

**Goal:** Honest acknowledgment of what we DIDN'T test

Content:
- **Bailey et al. (ICLR 2025):** Latent-space defenses can be reduced from 100% → 0% recall via 16-token adversarial suffix optimization on Llama-3-8B-Instruct
- **Bailey et al.'s adversarial training failure:**
  - 70 rounds of attacker-defender alternation
  - Defender always perfect on past attacks
  - Attacker always finds new bypass each round
  - **Geometric implication:** "There is no single 'harmfulness direction'"

- **Bailey et al.'s transfer findings:**
  - Suffix trained against LR probe also bypasses MLP probes
  - Suffix trained against generation-based bypasses input-based
  - Suffix trained against OOD detector bypasses input-based probes

- **Field-wide confirmations:**
  - Nasr et al. (2025): None of 12 jailbreak defenses robust to adaptive attacks
  - Gupta & Jenner (2025): RL-trained models can learn to evade probes
  - DeepMind (2026): Probes fail under production distribution shifts

- **Why we don't claim robustness:**
  - We tested only standard (non-adaptive) attacks
  - HPS's 12-feature compression likely makes it MORE vulnerable, not less
  - Our PGD-on-activations finding (HPS 96% evasion vs C4 2%) supports this

- **Lead-in to Task B:** We will run Bailey-style adaptive attacks (Section 8 details the plan)

### Section 8: Open Questions & Next Steps (1-2 pages)

Content:
- **Adaptive attack experiment proposal** (links to Task B plan)
- **Llama-2-7b-chat ablation:**
  - SAME base model as Vicuna (which has SFT only)
  - Has SFT + RLHF (like Llama-3-Instruct)
  - Comparing them isolates RLHF as the variable

- **Multi-turn pivot consideration:**
  - Multi-turn conversations have genuine hierarchical structure
  - 6-month commitment vs 6 weeks to TMLR submission

- **Other open questions:**
  - Generation-based probing variants
  - Defense-in-depth combining input/output classifiers
  - HSF/JBShield reproductions on diverse benign

- **Recommendation for paper venue:**
  - TMLR (60-65% acceptance): rigorous methodology paper
  - AAAI/IJCAI 2027 (35-50%): with methodology framing
  - Lead recommendation: TMLR

### Section 9: Appendices (3-5 pages)

#### Appendix A: Full Hyperparameter Specification

```python
# HPS hyperparameters
HPS_LAYERS_LLAMA3 = [0, 2, 17, 24, 28, 31]
HPS_LAYERS_VICUNA = [0, 2, 22, 31, 35, 39]
KAPPA_INIT = 0.1
FREEZE_KAPPA = True
EPOCHS = 50
PROJ_DIM = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# Anthropic MTP
MTP_L2_LAMBDA = 1e4
LR_L2_LAMBDA = 1e2
STANDARDIZE = True

# Evaluation
TARGET_FPR = 0.05
```

#### Appendix B: Dataset Construction

- **Diverse benign sources (9):** WildChat, OR-Bench Hard, MMLU, GSM8K, HumanEval, MBPP, WikiText long-form, multilingual, Alpaca control. Total: 5,905 prompts.
- **Length distribution:** Mean 1,076 chars; median 179; max 18,205; >2,000 chars: 752 prompts; >5,000 chars: 354 prompts.
- **JBShield attack source:** 9 categories from public release at github.com/NISPLab/JBShield
- **Train/test split:** 80/20 with seed=42

#### Appendix C: All Experimental Result Tables

Reference detailed JSON files:
- `results/statistical_tests.json` — HPS vs C4 statistics
- `results/radial_distribution_check.json` — 0/13 inversions detail
- `results/gcg_specific_test.json` — Vicuna catastrophic failure
- `results/hyperbolic_vs_euclidean_diverse.json` — Cold-start + 4-way comparison
- `results/prediction_agreement.json` — HPS-only catches = 0
- `results/verify_saturation_fixed.json` — Length AUROC 0.992

#### Appendix D: Code Organization

**Main scripts (5-7 most important):**
- `hps_core.py` — Self-contained HPS implementation
- `extract_diverse_benign_activations.py` — Activation extraction
- `fix_cache_max_length.py` — Re-extract attacks at consistent max_length
- `verify_saturation.py` — 6-check methodology audit
- `statistical_tests.py` — Bootstrap CIs + paired tests
- `radial_distribution_check.py` — Multi-config inversion check
- `prediction_agreement.py` — Per-example agreement analysis
- `anthropic_mean_token_probe.py` — Faithful Anthropic MTP reproduction

**Pipeline orchestration:**
- `run_overnight_pipeline.sh` — Full 7-phase pipeline
- `run_diverse_benign_pipeline.sh` — Earlier diverse-benign-focused pipeline

## A.2 Writing Workflow

### Day 1 (Morning, 4 hours): Sections 1-3
- Section 1: Hypothesis + theoretical foundation (1 page) — easiest entry point
- Section 2: System description with comparison table (2-3 pages)
- Section 3: Initial results + suspicion (1 page) — narrative pivot

### Day 1 (Afternoon, 4 hours): Section 4 (Methodology Audit)
- This is the heart of the paper
- 4.1 Length confound
- 4.2 Train/test contamination
- 4.3 max_length confound
- Reference: `verify_saturation_fixed.json`, `norm_check_diverse.json`

### Day 2 (Morning, 4 hours): Section 5 (Re-evaluation)
- Reference: `statistical_tests.json`, `radial_distribution_check.json`, `prediction_agreement.json`

### Day 2 (Afternoon, 3 hours): Sections 6-7
- Section 6: Vicuna failure (per-attack table + mechanism)
- Section 7: Adversarial limitation

### Day 3 (1-2 hours): Sections 8-9 + polish

## A.3 Source Materials Already Available

- `mentor_briefing.md` — high-level overview
- `team_meeting_slides.md` — slides with all numbers (use as scaffold)
- `mentor_reading_list.md` — citations
- `literature_review_activation_defenses.md` — related work descriptions

## A.4 Citation Anchors

Cite the 4 must-read papers (from `mentor_reading_list.md`):
- Anthropic Cheap Monitors (Cunningham et al. 2025)
- Bailey et al. (ICLR 2025)
- JBShield (Zhang et al. USENIX 2025)
- HypLoRA + HELM (NeurIPS 2025)

Plus 2 second-tier:
- Wollschläger et al. (ICML 2025) — geometry of refusal
- Arditi et al. (NeurIPS 2024) — single-direction refusal


# Task B — Bailey Deep Dive + Adaptive Attack on HPS/C4

**Goal:** Determine empirically whether HPS provides any adversarial robustness advantage over C4/MTP, OR confirm it suffers the same vulnerabilities. Either result is publishable.

**Estimated time:** 2 weeks (1 week deep read + planning, 1 week implementation)

## B.1 Phase 1: Deep Read of Bailey et al. (3 days)

### B.1.1 What Bailey et al. Tested

**Setup (verified from paper):**
- Model: Llama-3-8B-Instruct (same as us)
- Datasets: UltraChat (benign), Circuit Breakers (harmful)
- Standard jailbreaks tested: GCG, PAIR, MSJ, AutoDAN, Multi-Turn, Prefill, Misc

**Probe types they tested:**
1. Logistic Regression (LR) — input-based and generation-based
2. MLP probes (single hidden layer, 64 dim) — input-based and generation-based
3. Mean Difference probes — basic baseline
4. Sparse Autoencoder (SAE) probes — at SQL detection
5. OOD detectors:
   - Gaussian (Mahalanobis distance)
   - Variational Autoencoder (VAE)
   - Beatrix (state-of-the-art backdoor detector)
   - TED (topological evolution dynamics)

**Attacks they tested:**
1. **Embedding suffix attack (white-box):**
   - 16-token universal suffix
   - Adam optimizer
   - Joint loss: behavior loss (CrossEntropy to harmful target) + obfuscation loss (probe score)
   - Universal across all harmful inputs (held-out test)

2. **Hard prompt suffix (gray-box):**
   - Modified FLRT optimizer (gradient-free)
   - Per-input optimization (not universal)
   - Only access to model logits + probe score

3. **Adversarial training:**
   - 70 rounds of attacker-defender alternation
   - Defender always perfect on past attacks; attacker always finds new bypass

4. **Cross-defense transfer:**
   - Attack trained against one probe, tested on another
   - Found: LR ↔ MLP transfer perfectly; OOD → probes transfers; probes → OOD doesn't

**Key results:**
- Standard jailbreaks: probes work (90-100% recall)
- Obfuscation attacks: probes fail catastrophically (100% → 0% recall)
- 90% jailbreaking success rate maintained throughout

### B.1.2 What Bailey et al. DID NOT Test (gaps we can exploit)

This is the key question. Let me identify gaps systematically:

#### Gap 1: Hyperbolic projections
- They tested LR, MLP, SAE, OOD — all flat geometry
- They did NOT test: Lorentz projections, Poincaré ball probes, hyperbolic neural networks
- **Question:** Does the curvature constraint of hyperbolic space change attack difficulty?
- **Hypothesis:** HPS might be MORE vulnerable (12-feature compression = single point of failure) OR LESS vulnerable (manifold constraint creates regularization)

#### Gap 2: Trajectory features (multi-layer)
- They tested single-layer and generation-based probes
- They did NOT test: trajectory features across multiple layers (HPS-style)
- HPS's "trajectory" = sequence of activations across [0, 2, 17, 24, 28, 31]
- **Question:** Does multi-layer trajectory information provide redundancy that resists obfuscation?
- **Hypothesis:** Maybe the attacker has to fool MORE layers simultaneously

#### Gap 3: Aggregation axis
- They tested mean-token (Anthropic-style) at one layer
- They did NOT test: mean-over-layers at one token (our C4 variant)
- **Question:** Does aggregation axis matter for adversarial robustness?
- **Hypothesis:** Probably not (both are linear projections), but worth verifying

#### Gap 4: Cross-LLM adversarial
- They tested only Llama-3-8B-Instruct
- They did NOT test: Vicuna-13B-v1.5, Llama-2, base models
- **Question:** Do obfuscation attacks transfer across LLMs?
- **Hypothesis:** Should transfer (similar vulnerabilities)

#### Gap 5: Number of attack prompts in obfuscation training
- They optimized one suffix against ALL harmful prompts (universal)
- They did NOT test: optimizing against a small subset
- **Question:** Is robust attack development sample-efficient?

#### Gap 6: Per-attack-category robustness
- They reported overall recall reduction
- They did NOT report: per-attack-category breakdown of obfuscation effectiveness
- **Question:** Are some attack types harder to obfuscate against than others?
- **Relevance to us:** GCG and PAIR have very different structures; HPS might have different vulnerabilities

### B.1.3 What Bailey et al. Could Have Done Differently (their limitations)

From their own limitations section:
- They acknowledge hard prompt attacks are weaker
- They acknowledge cross-LLM transfer not fully tested
- They acknowledge OAT (their proposed defense) increases attack budget but doesn't achieve robustness
- They focus on Llama-3-8B-Instruct only

### B.1.4 Specifically What We Should Test

Combining all gaps, here's our research question:

**"Do hyperbolic geometric features provide adversarial robustness over linear probes when subjected to Bailey-style obfuscation attacks?"**

Sub-questions:
1. Per-method robustness: HPS vs C4 vs MTP vs HPS-Euclidean
2. Per-attack-method susceptibility: which attacks are hardest to obfuscate against?
3. Cross-method transfer: does suffix optimized against C4 transfer to HPS?
4. Cross-LLM transfer: does suffix transfer Llama-3 → Vicuna?
5. Generation-based variants: probes that look at output tokens too

## B.2 Phase 2: Implementation Plan (1 week)

### B.2.1 Use Bailey et al.'s Released Code as Starting Point

🔗 https://github.com/LukeBailey181/obfuscated-activations

Pros: Working implementation of Bailey-style attacks. Saves us 1-2 weeks.
Cons: Need to adapt to our HPS/C4/MTP architectures.

### B.2.2 Adaptation Strategy

```python
# Skeleton of adaptive attack against any probe:

def obfuscation_attack(model, probe, harmful_prompts, harmful_targets,
                       n_steps=512, suffix_length=16, lambda_obf=1.0):
    # Initialize 16-token universal embedding suffix
    suffix_embed = init_random_embedding(suffix_length, model.hidden_size)

    for step in range(n_steps):
        for harmful_prompt, harmful_target in zip(harmful_prompts, harmful_targets):
            # Construct input: prompt + suffix
            input_embeds = concat([prompt_embeds(harmful_prompt), suffix_embed])

            # Forward through model
            outputs = model(input_embeds, output_hidden_states=True)

            # Behavior loss: cross-entropy to harmful target
            beh_loss = cross_entropy(outputs.logits, harmful_target)

            # Obfuscation loss: probe score on suffix-augmented input
            probe_score = probe(outputs.hidden_states)  # method-specific
            obf_loss = probe_score  # we want this LOW

            # Combined loss
            total_loss = beh_loss + lambda_obf * obf_loss

            # Update suffix embedding
            suffix_embed = adam_step(suffix_embed, total_loss.grad)

    return suffix_embed

# Method-specific probe wrappers:

def c4_probe_score(hidden_states, layers=[0,2,17,24,28,31]):
    feats = [hidden_states[l][-1] for l in layers]  # last-token at each layer
    feats = mean(feats)  # aggregate across layers
    return c4_logistic_regression(feats)

def mtp_probe_score(hidden_states, layer=17):
    return mtp_logistic_regression(mean(hidden_states[layer]))  # mean over tokens

def hps_probe_score(hidden_states, layers=[0,2,17,24,28,31], kappa=0.1):
    proj = lorentz_projection(W, hidden_states[layers, -1, :])  # (n_layers, 65)
    traj_feats = extract_trajectory_features(proj, kappa)  # (12,)
    return hps_logistic_regression(traj_feats)
```

### B.2.3 Experiment Design

**Test 1: Standalone obfuscation attack on each method (Llama-3-8B-Instruct)**
- 4 methods: HPS, HPS-Euclidean, C4, MTP @ L17
- For each method:
  - Train suffix on training attack set
  - Test on held-out attack set
  - Measure: jailbreak success rate, probe recall reduction
- Predicted: all methods show 100% → 0% recall (matching Bailey)

**Test 2: Per-attack-category breakdown**
- For each method × each attack type (9 types from JBShield):
  - Test if obfuscation suffix is more/less effective per attack
- Predicted: some attacks (e.g., GCG) more amenable to obfuscation than others

**Test 3: Cross-method transfer**
- Train suffix against C4 only
- Test on HPS, HPS-Euclidean, MTP separately
- Vice versa for HPS
- Predicted: full transfer (per Bailey's findings on probe transfer)

**Test 4: Cross-LLM transfer**
- Train suffix on Llama-3-8B-Instruct (HPS)
- Test on Vicuna-13B-v1.5 (HPS)
- Reverse direction too
- Predicted: ~50-70% transfer (different LLM, similar vulnerabilities)

**Test 5: Hyperbolic-specific defense ideas (extension)**
- Test 5a: Increase number of layers in HPS (current 6, try 12 or 24)
- Test 5b: Test generation-based variant of HPS
- Test 5c: Test Hyperbolic MLP (already shown to work on Vicuna)
- Predicted: more layers might help marginally; generation-based might help more (per Bailey)

### B.2.4 Expected Timeline

- **Day 1-2:** Clone Bailey's code, study their attack implementation
- **Day 3:** Implement HPS probe wrapper (the trickiest — need to differentiate through trajectory features)
- **Day 4:** Implement C4, MTP, HPS-Euclidean probe wrappers
- **Day 5:** Run Test 1 on Llama-3 (4 methods, ~2 hours each)
- **Day 6:** Run Tests 2-3 (per-attack breakdown, transfer)
- **Day 7:** Analysis + initial writeup

## B.3 Phase 3: Possible Outcomes & Decision Tree

### Outcome A: All methods fail similarly (most likely, per Bailey)
- HPS: 100% → 0% recall
- C4: 100% → 0% recall
- MTP: 100% → 0% recall
- HPS-Euclidean: 100% → 0% recall

**What to do:** Confirm Bailey's finding extends to hyperbolic projections. Cite as field-wide vulnerability, no exception.

**Paper section:** "We empirically verify that HPS, despite its compressed feature space, suffers the same adversarial vulnerability as flat linear probes. The geometric prior provides no robustness advantage."

### Outcome B: HPS marginally more robust (e.g., 100% → 30%)
**What to do:** Investigate why. Possible mechanism: trajectory features create constraints attacker must satisfy across all layers.

**Paper section:** "While not adversarially robust, HPS's multi-layer trajectory features provide marginal robustness gain over single-layer probes. Attackers must obfuscate across more layers."

### Outcome C: HPS catastrophically MORE vulnerable (e.g., reaches 0% faster)
**What to do:** This is consistent with our PGD finding (HPS 96% evasion vs C4 2%). Confirms compressed feature space = single point of failure.

**Paper section:** "HPS's 12-feature compression provides an attacker-friendly target: gradient flows directly to the dominant trajectory feature (mean radial position), making attacks more efficient than against C4's 4096-dim representation."

### Outcome D: Method-specific vulnerabilities
- E.g., GCG-style attacks specifically obfuscate against HPS but not C4
- Or vice versa

**What to do:** Frame as "different methods have different attack surfaces." This is a finer-grained finding.

## B.4 What This Adds to the Paper

Currently the paper says:
> "We do not test adaptive attacks. Bailey et al. (ICLR 2025) demonstrated that latent-space defenses can be reduced from 100% to 0% recall via obfuscation attacks. We do not claim adversarial robustness."

After Task B, we can say:
> "We empirically verify Bailey et al.'s findings extend to hyperbolic geometric defenses. HPS is reduced from [X%] to [Y%] recall under [embedding suffix / hard prompt] obfuscation attacks. This [confirms / refutes] the hypothesis that geometric priors provide adversarial robustness. [Outcome-specific implications.]"

This is a much stronger paper position with concrete numbers instead of citing.

## B.5 Risk Assessment

**Risks:**
1. **Implementation complexity** — HPS gradients through trajectory features may have numerical issues
2. **Compute cost** — each attack run is ~2-4 hours; 4 methods × 5 tests = ~40 hours
3. **Disk space** — Vicuna full-sequence cache + attack experiments need ~50 GB free
4. **Bailey's code may not adapt cleanly** — might need significant rewriting

**Mitigations:**
1. Start with simpler probes (C4, MTP) before HPS
2. Use shorter attack runs first to verify correctness
3. Free disk space first (delete failed extraction caches)
4. Implement core attack from scratch using Bailey's paper as reference

**Failure mode:** If implementation takes >1 week, fall back to:
- Run attacks only on C4 + MTP (simpler implementations)
- Cite Bailey for HPS as "expected to fail similarly based on architectural similarity"

---

# Combined Workflow

## Week 1: Parallel Start
- **Mornings:** Write Sections 1-3 of mentor draft (Task A)
- **Afternoons:** Read Bailey et al. paper, identify gaps (Task B Phase 1)

## Week 2: Continue Both
- **Mornings:** Write Sections 4-7 of mentor draft (methodology + findings)
- **Afternoons:** Implement adaptive attack code (Task B Phase 2)

## Week 3: Finish Draft + Run Experiments
- **Mornings:** Sections 8-12 of mentor draft (limitations + code + recommendations)
- **Afternoons:** Run adaptive attack experiments (Task B Phase 2 continued)

## Week 4: Iterate + Polish
- Get mentor feedback on draft
- Analyze adaptive attack results
- Update draft with Task B findings
- Polish for paper submission

---

# Quick Decision Points

## For Task A
- **Format:** Markdown vs LaTeX? **Recommend: Markdown** (faster to iterate, easy to convert later)
- **Length:** 25-page version vs 40-page version? **Recommend: 30 pages** (comprehensive but readable)
- **Audience:** Just mentor or wider team? **Recommend: Write for mentor, but make publishable**

## For Task B
- **Implementation:** From scratch vs adapt Bailey's code? **Recommend: Adapt Bailey's code** (faster, more reliable)
- **Methods to test:** All 4 (HPS, HPS-Euc, C4, MTP) or just 2? **Recommend: All 4** (proper comparison)
- **Attack types:** Embedding only or also hard prompts? **Recommend: Embedding only first** (simpler, sufficient for paper)

---

# Files to Produce

**Task A:**
- `mentor_draft.md` — 30-page comprehensive document (NEW)

**Task B:**
- `bailey_deep_read.md` — notes on Bailey et al. (NEW)
- `adaptive_attack_hps_vs_c4.py` — implementation (NEW)
- `adaptive_attack_results.json` — experiment results (NEW)
- Update `mentor_draft.md` with adaptive attack section (UPDATE)

---

# Recommendation

**Start with Task A.** Spend 3-4 days writing the comprehensive draft skeleton with everything you currently have. This forces you to organize your thinking and identify any gaps in your own understanding.

**While drafting, alternate with reading Bailey et al.** Take notes on what they did and didn't test. By end of week 1, you'll have a solid draft skeleton AND a clear plan for the adaptive attack experiments.

**Week 2 onwards:** implement and run.

**Don't try to do both at full speed simultaneously.** Writing forces clear thinking; experiments give you new data. Let them inform each other.

Want me to start drafting Section 1 of the mentor draft right now? Or write the Bailey deep-read notes first?
