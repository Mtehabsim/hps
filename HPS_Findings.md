# HPS Findings — Complete Empirical Record

This document records every experiment run, the results, and the implications for the paper.

## Status Summary

| Question | Answer |
|----------|--------|
| Does HPS work? | Yes — 1.000 AUROC same-dist, 0.997 cross-attack |
| Is it novel vs Euclidean projections? | Only in low-data regime (N ≤ 100 per method) |
| Is it novel vs linear probes on raw activations? | **No** — C4 (mean-pool + LR) matches HPS at all scales |
| Is HPS more robust to PGD? | **No** — HPS is *less* robust at moderate ε |
| Is the framing publishable? | At workshop or mid-tier venue. Not USENIX-tier. |

---

## 1. Final Production Configuration

After extensive ablation, the canonical HPS configuration is:

- **Layers:** `[0, 2, 17, 24, 28, 31]` (spread, derived from TEST 7)
- **Curvature κ:** 0.1, frozen (TEST 9 showed learnable κ doesn't escape initialization)
- **Training:** 50 epochs, no early stopping (TEST 5 showed overfitting past 50)
- **Features:** 12 trajectory features, but **`mean_r` alone matches all 12** (feature ablation)
- **Classifier:** Logistic regression on standardized features

---

## 2. Headline Results (Llama-3-8B)

### 2.1 Same-Distribution Detection

| Method | AUROC | TPR@5%FPR | F1 | Accuracy |
|--------|-------|-----------|-----|----------|
| HPS    | 1.000 | 1.000     | 0.989 | 0.985 |
| RTV    | 0.854 | 0.551     | 0.696 | 0.679 |
| Ensemble | 1.000 | 1.000   | 0.989 | 0.985 |
| Euclidean (param-matched) | 0.999 | 0.998 | — | — |
| C4 (LR on mean-pooled, 4096-dim) | 1.000 | 1.000 | — | — |
| C5 (LR on flattened, 24576-dim) | 1.000 | 1.000 | — | — |

**Per-attack TPR (HPS, all 9 methods):** 100% on autodan, base64, drattack, gcg, ijp, pair, puzzler, saa, zulu.

### 2.2 Cross-Attack Generalization (leave-one-out, 3 seeds)

| Method | Mean TPR (9-fold LOO) | std |
|--------|----------------------|-----|
| HPS | 0.997 | 0.005 |
| RTV | 0.549 | — |
| Ensemble | 0.997 | — |
| Euclidean | 0.994 | 0.002 |

### 2.3 Cold-Start Cross-Attack (the unique HPS regime)

The discriminating result. Train on N attacks per method, test on held-out method.

| N/method | HPS TPR | Euclidean TPR | C4 TPR | Δ (HPS - Euc) |
|----------|---------|---------------|--------|---------------|
| 5  | 0.996 | 0.244 | 0.992 | +0.752 |
| 10 | 0.989 | 0.411 | — | +0.578 |
| 15 | 0.985 | 0.785 | — | +0.200 |
| 20 | 0.994 | 0.833 | — | +0.161 |
| 25 | 0.996 | 0.867 | 0.996 | +0.129 |
| 50 | 0.998 | 0.989 | — | +0.009 |
| 100 | 0.998 | 0.971 | 0.999 | +0.027 |
| 250 | 0.997 | 0.978 | — | +0.019 |
| 500 | 0.997 | 0.992 | — | +0.005 |

**Crossover point:** ~50 examples per method. Below this, HPS dominates Euclidean by 50-75% TPR.

**However:** C4 (linear probe on mean-pooled activations) also achieves 0.992-0.999 TPR at N=5-100. **HPS does not beat C4 in cold-start.**

### 2.4 Method Diversity Effect (fixed N=25)

| #methods | HPS TPR | Euclidean TPR | Δ |
|----------|---------|---------------|---|
| 2 | 0.500 | 0.000 | +0.500 |
| 3 | 0.680 | 0.253 | +0.427 |
| 4 | 0.990 | 0.250 | +0.740 |
| 5 | 1.000 | 0.608 | +0.392 |
| 6 | 0.993 | 0.720 | +0.273 |
| 7 | 0.989 | 0.760 | +0.229 |
| 9 | 0.996 | 0.867 | +0.129 |

Hyperbolic advantage shrinks as attack diversity grows — consistent with regularization theory.

### 2.5 Multi-Seed Stability (5 seeds, full data)

| Method | Mean AUROC ± std | Mean TPR ± std |
|--------|------------------|-----------------|
| HPS | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Euclidean | 0.999 ± 0.000 | 0.998 ± 0.000 |

### 2.6 Feature Ablation (which of 12 features matter?)

Permutation importance ranking on same-dist:
1. `mean_r` (radial) — 0.0013
2. `progress` (displacement) — 0.0009
3. `max_r` (radial) — 0.0001
4. `range_r` (radial) — 0.0001
5. `displacement` — 0.0001
6. `path_len` — 0.0001
7-12. All near-zero (curvature features useless)

**Single-feature performance (`mean_r` alone):**
| Regime | TPR |
|--------|-----|
| Same-dist | 1.000 |
| Cold-start N=5 | 0.996 |
| Cold-start N=25 | 0.998 |
| Vicuna-like (4 methods × 25) | 1.000 |

The `mean_r` scalar matches the full 12-feature model within 0.002 TPR. Curvature features add no signal.

### 2.7 Learning Curve

| N attacks | AUROC | TPR@5% |
|-----------|-------|--------|
| 50 | 0.995 | 0.984 |
| 100 | 0.999 | 0.997 |
| 200 | 1.000 | 0.998 |
| 500 | 1.000 | 0.999 |
| 1000+ | 1.000 | 1.000 |

HPS is useful from ~50 attacks; saturates at 200.

### 2.8 Computational Cost

- Projection + features + classification: **0.81 ms/prompt**
- LLM forward pass (activation extraction): bottleneck, not measured
- Total inference overhead beyond LLM forward: <1ms

---

## 3. Critical Findings vs Baselines

### 3.1 Controls That Confirm HPS Is Doing Something Real

| Control | TPR (CS N=5) | Δ vs HPS | Interpretation |
|---------|--------------|----------|----------------|
| C1: Mean L2 norm scalar (1-dim) | 0.321 | -0.676 | Activation magnitude alone fails |
| C2: Per-layer L2 norms (6-dim) | 0.658 | -0.338 | Multi-layer norms fail |
| C3: Random untrained Lorentz | 0.099 | -0.897 | Random projection is useless |

These rule out "HPS is just detecting magnitude" or "any projection works."

### 3.2 Controls That Match HPS

| Control | TPR (CS N=5) | Δ vs HPS | Interpretation |
|---------|--------------|----------|----------------|
| C4: LR on mean-pooled (4096-dim) | 0.992 | -0.004 | Tied with HPS |
| C5: LR on flattened (24576-dim) | 0.986 | -0.010 | Tied with HPS |

C4 is essentially equivalent to **Latent Sentinel** (OpenReview 2025) and **Anthropic's representation re-use** approach. **HPS does not beat established prior art.**

### 3.3 Adversarial Robustness Comparison (PGD)

| ε | HPS evasion | C4 evasion | Δ |
|---|-------------|------------|---|
| 0.001 | 0.000 | 0.000 | +0.000 |
| 0.005 | 0.000 | 0.000 | +0.000 |
| 0.01 | 0.000 | 0.000 | +0.000 |
| 0.05 | **0.660** | **0.020** | **-0.640** |
| 0.1 | 1.000 | 0.350 | -0.650 |

**HPS is dramatically less robust than C4 at moderate ε.** The 1-feature bottleneck is a vulnerability — perturbing `mean_r` requires only one direction in activation space, while breaking C4 requires fighting 4096 features simultaneously.

### 3.4 Layer Norm Analysis (Part D)

L2 norms across layers for benign vs attacks:

| Layer | Benign mean | Attack mean | Δ |
|-------|-------------|-------------|---|
| 0 | 0.60 | 0.80 | +0.20 |
| 2 | 1.49 | 1.66 | +0.17 |
| 17 | 10.57 | 10.04 | -0.52 |
| 24 | 19.39 | 19.14 | -0.26 |
| 28 | 28.36 | 28.45 | +0.09 |
| 31 | 155.62 | 153.03 | -2.58 |

The signal is small (0.1-2.5 in raw norm) and noisy (std 1-4). Confirms why C1 fails — raw norms don't separate cleanly.

---

## 4. Vicuna-13B Results

| Method | AUROC | FPR@95 | F1 |
|--------|-------|--------|-----|
| HPS-Full (Hyperbolic-Trained) | 1.000 | 0.000 | 1.000 |
| Euclidean-Trained (param-matched) | 1.000 | 0.000 | 1.000 |
| Nonlinear-Euclidean (LayerNorm + tanh) | 1.000 | 0.000 | 0.995 |
| Fisher-8 Raw | 1.000 | 0.000 | 0.995 |
| HPS-Lite (untrained) | 0.774 | 0.637 | 0.561 |
| Dual-use FPR | 0.000 | — | — |

On Vicuna with 252 attacks, all trained methods saturate at 1.000 AUROC. **No advantage of HPS over Euclidean here.**

The previously reported "+0.302 AUROC over Euclidean" was an artifact of the bad Euclidean baseline (single global scale instead of per-layer scale + learnable margin).

---

## 5. Adversarial Training (HPS-Adv, Vicuna)

| ε | Vanilla HPS evasion | HPS-Adv evasion |
|---|---------------------|-----------------|
| 0.001 | 0.359 | 0.484 |
| 0.01 | 0.766 | 0.828 |
| 0.05 | 1.000 | 0.984 |
| 0.1 | 1.000 | 1.000 |

Adversarial training provides marginal benefit at ε≤0.05, but breaks at ε≥0.05 same as vanilla. Honest limitation.

---

## 6. Adaptive PGD on Ensemble (Vicuna)

| ε | Evasion | Mean Score |
|---|---------|-----------|
| 0.001 | 0.100 | 0.737 |
| 0.005 | 0.200 | 0.623 |
| 0.01 | 0.280 | 0.536 |
| 0.05 | **0.940** | 0.075 |
| 0.1 | 1.000 | 0.006 |
| 0.5 | 1.000 | 0.000 |

Ensemble breaks at ε≈0.05. Same as HPS alone.

---

## 7. Honest Assessment of Novelty

### What's Genuinely Novel in This Work

1. **Hyperbolic Lorentz projection for jailbreak detection** — no prior work uses this geometry
2. **Cold-start empirical finding** — HPS reaches 99.6% TPR at N=5 examples per method, where Euclidean projection collapses to 24.4%
3. **The crossover analysis** — first systematic study showing where geometric priors help vs Euclidean alternatives
4. **Curvature features useless** — empirical finding that 11 of 12 features add no value

### What's NOT Novel (Prior Art Exists)

1. **Linear probes on hidden states for jailbreak detection** — Latent Sentinel (2025), Anthropic Constitutional Classifiers (2025), arXiv:2411.03343, multiple others
2. **Multi-layer activation features** — HiddenDetect, Layerwise Convergence Fingerprints, multiple others
3. **Trajectory/path features for jailbreak detection** — SALO (ICML 2026), Harmfulness Propagation Dynamics (OpenReview 2025)
4. **Contrastive learning on representations for safety** — SupCon (Khosla 2020), CRAFT, Chen et al. 2025
5. **Refusal direction monitoring** — Arditi et al. 2024, RTV (Derya & Sunar 2026)
6. **Per-attack-type breakdown** — Standard in JBShield, ALERT, others

### What's Refuted by Our Own Experiments

1. **"Hyperbolic geometry universally improves jailbreak detection"** — false. HPS = Euclidean at scale.
2. **"Geometric prior provides robustness"** — false. HPS is *less* robust under PGD than linear probe.
3. **"+0.302 AUROC over Euclidean"** — was an artifact of unfair Euclidean baseline. With matched parameters, the gap is +0.001 same-dist, +0.005-0.045 cross-attack at scale.
4. **"Trajectory features (curvature, displacement) matter"** — they don't. `mean_r` alone matches the full 12-feature model.

---

## 8. Implications for Paper Submission

### Original Framing (REJECTED)

> "Beyond the Refusal Direction: Hyperbolic Geometry as Inductive Bias for Cross-Attack Jailbreak Detection"
> "+0.302 AUROC over Euclidean projections"

This framing does not survive the controls:
- Euclidean (parameter-matched) achieves nearly identical performance at scale
- C4 (linear probe on raw activations) matches HPS in cold-start
- HPS is less robust than C4 under PGD

### Revised Framing (DEFENSIBLE)

> "Hyperbolic Geometry as a Cold-Start Inductive Bias for Jailbreak Detection"

**Contribution scope:**
- Empirical finding: hyperbolic projection enables 99.6% TPR with N=5 attacks per method (Euclidean: 24.4%)
- Crossover analysis: when geometric priors help vs when they don't
- Negative results: curvature features useless, robustness no better than linear probes
- Single-feature compression: `mean_r` is sufficient

**Honest scope:** This is a low-data inductive bias paper, not a universal improvement paper.

### Realistic Venues

| Venue | Fit | Notes |
|-------|-----|-------|
| AAAI, IJCAI | Good | ML focus, accepts nuanced empirical work |
| ACSAC, RAID | Good | Mid-tier security, values honest limitations |
| TMLR | Good | Accepts thorough empirical contributions |
| NeurIPS/ICLR Workshop | Good | Safe ML / interpretability workshops |
| **USENIX Security** | **Poor** | Needs strong universal claim, not low-data niche |
| **CCS, NDSS** | **Poor** | Same as USENIX |
| **ICML / NeurIPS main** | **Poor** | Method too simple for main track |

### Cross-Model Transfer Test (Optional, Strengthens Paper)

If HPS's projection trained on Vicuna transfers to Llama-3 while C4's linear probe doesn't, this strengthens the paper. Not yet tested. Requires extracting Vicuna activations at matched layer indices.

---

## 9. Recommended Next Steps

1. **Run cross-model transfer experiment** (1-2 hours) — could provide a strong differentiator
2. **Rewrite paper** around cold-start framing
3. **Add comprehensive baseline table** including C4, C5, Latent Sentinel, Anthropic representation re-use
4. **Add explicit honest limitations section** (curvature ablation, robustness comparison, no advantage over linear probe at scale)
5. **Submit to AAAI / ACSAC / TMLR** rather than USENIX

---

## 10. Updated Reference List

### Activation-Based Jailbreak Detection (Direct Competitors)

- **Latent Sentinel** (OpenReview 2025) — linear probes on frozen LLM hidden layers for real-time jailbreak detection. arXiv: pending.
- **Anthropic Constitutional Classifiers via Representation Re-use** (2025) — linear probes on intermediate activations for cost-effective jailbreak detection. https://alignment.anthropic.com/2025/cheap-monitors/
- **JBShield** (USENIX Security 2025) — toxic + jailbreak concept activation analysis. arXiv:2502.07557.
- **HiddenDetect** (Jiang et al., 2025) — cosine similarity with vocabulary-space refusal vector for VLMs. arXiv:2502.14744.
- **SALO** (ICML 2026) — Sparse Activation Localization Operator using causal tracing for refusal trajectory detection. arXiv:2605.02958.
- **Harmfulness Propagation Dynamics** (OpenReview 2025) — multi-layer trajectory analysis for adversarial intent detection.
- **Layerwise Convergence Fingerprints** (arXiv:2604.24542, 2026) — runtime misbehavior detection via layer-wise convergence patterns.
- **What Features in Prompts Jailbreak LLMs?** (arXiv:2411.03343, Wang et al. 2024) — linear and non-linear probes; shows attack-specific features fail to generalize.
- **Latent Representation Framework** (arXiv:2602.11495, 2026) — tensor-based hidden activation analysis.
- **Hidden State Forensics** (arXiv:2504.00446, 2025) — abnormal behavior detection via layer-specific activation patterns.
- **Linear Probe Accuracy Scales with Model Size** (arXiv:2604.13386, 2026) — multi-layer probe ensembles.
- **Stable Jailbreak Detection via Token-Level Logits** (arXiv:2604.01473) — alternative output-side detection.
- **Pretrained Embeddings for Jailbreak Detection** (arXiv:2412.01547, 2024) — text embedding + classical ML.
- **Investigating Coverage Criteria for Jailbreak Detection** (arXiv:2408.15207, 2024) — neural activation features classifier.
- **Defending LLMs Against Jailbreaks Comprehensive Framework** (arXiv:2511.18933) — taxonomy of defense strategies.

### Refusal Direction Analysis

- **Arditi et al. 2024** — Refusal in Language Models is mediated by a single direction. arXiv:2406.11717.
- **RTV: Refusal Trajectory Vector** (Derya & Sunar 2026, arXiv:2605.03095) — multi-layer Mahalanobis on refusal directions.
- **Wollschläger et al.** (ICML 2025) — multiple independent refusal directions and concept cones.
- **Guo et al. 2025** — refusal behaviors as 11 geometrically distinct directions.
- **Linearly Decoding Refused Knowledge** (arXiv:2507.00239, 2025) — refused information is linearly decodable.

### Hyperbolic Geometry in ML (Theoretical Foundation)

- **Hyperbolic vs Euclidean Embeddings in Few-Shot Learning** (arXiv:2309.10013) — direct comparison showing hyperbolic excels in low-data.
- **Is Hyperbolic Space All You Need for Medical Anomaly Detection?** (arXiv:2505.21228) — "hyperbolic space exhibits resilience to parameter variations and excels in few-shot scenarios."
- **HCFSLN: Adaptive Hyperbolic Few-Shot Learning** (arXiv:2511.06988) — hyperbolic few-shot for multimodal tasks.
- **HyperVD** (arXiv:2305.18797) — hyperbolic embeddings for violence detection.
- **HypAD** (CVPR Workshop 2023) — anomaly detection with hyperbolic embeddings.
- **Poincaré Embeddings for Learning Hierarchical Representations** (Nickel & Kiela, NeurIPS 2017) — foundational.
- **Hyperbolic Active Learning for Semantic Segmentation under Domain Shift** (arXiv:2306.11180) — HALO for active learning.

### Concurrent Work in Contrastive / Geometric Detection

- **Representational Contrastive Scoring for VLMs** (Wang et al., arXiv:2512.12069, 2025) — closest methodological predecessor; learned projection with contrastive scoring for VLM jailbreak detection.
- **CRAFT** (arXiv:2603.17305, 2026) — contrastive representation learning + RL for safe reasoning trajectories.
- **Improving LLM Safety with Contrastive Representation Learning** (Chen et al. 2025, arXiv:2506.11938).
- **Probing Latent Subspaces of LLMs for AI Security** (arXiv:2503.09066) — LDA on activations for jailbreak detection.

### Jailbreak Attacks Cited

- **GCG** (Zou et al., 2023) — Greedy Coordinate Gradient. arXiv:2307.15043.
- **PAIR** (Chao et al., 2023) — Prompt Automatic Iterative Refinement. arXiv:2310.08419.
- **AutoDAN** (Liu et al., 2024) — automated DAN-style jailbreaks. arXiv:2310.04451.
- **TombRaider** (arXiv:2501.18628, EMNLP 2025) — historical knowledge exploitation.
- **ICE: Intent Concealment and divErsion** (arXiv:2505.14316, ACL 2025).
- **SequentialBreak** (arXiv:2411.06426, ACL 2025 SRW) — sequential prompt chains.
- **DrAttack** (arXiv:2402.16914, EMNLP 2024) — prompt decomposition.
- **ArtPrompt** (arXiv:2402.11753, ACL 2024) — ASCII art jailbreaks.
- **Bit-flip Jailbreaks** (arXiv:2412.07192) — physical attacks via bit flips.
- **Improved Few-Shot Jailbreaking** (arXiv:2406.01288).
- **Multilingual Jailbreak Challenges** (arXiv:2310.06474, ICLR 2024) — DAMO MultiJail dataset.
- **Multilingual jailbreaking via low-resource languages** (arXiv:2605.18239) — Afrikaans, Kiswahili, isiXhosa, isiZulu.
- **BVS Visual Safety Bypass** (arXiv:2601.15698) — semantic-agnostic VLM attacks.
- **IDEATOR** (arXiv:2411.00827, ICCV 2025) — VLMs jailbreak themselves.
- **Memory-Augmented Multi-Agent VLM Jailbreaks** (arXiv:2604.12616).
- **Self-Introspection Jailbreaks** (arXiv:2505.11790).
- **Open Sesame Black-Box Jailbreaking** (arXiv:2309.01446).

### Defense Methods (for comparison)

- **Circuit Breakers** (Zou et al., 2024) — training-time representation rerouting. arXiv:2406.04313.
- **Mitigating LLM Jailbreaks with a Few Examples** (Peng et al., Anthropic, 2024, arXiv:2411.07494) — 1 example/strategy → 240× ASR reduction.
- **Self-Reminder** (Wu et al., 2023) — prompt-level defense.
- **AttentionDefense** (arXiv:2504.12321) — system prompt attention monitoring.
- **Uncertainty-Driven Defense** (arXiv:2504.01533) — shifted token distribution.
- **Token Activation Defense for SLMs** (arXiv:2603.28817).
- **Streaming Hidden-state Trajectory Detection** (arXiv:2604.07727) — decoding-time defense.
- **Answer-Then-Check** (arXiv:2509.11629) — reasoning-based defense.
- **Latent Diffusion Safety Defense** (arXiv:2602.18782).

### Adversarial Robustness in LLM Detection

- **Schwinn & Geisler** (2024) — adversarial attacks on circuit breakers. arXiv:2406.18510.
- **Embedding Space Attacks** (arXiv:2402.09063) — direct attack on continuous embeddings.
- **Lagrangian-Optimized Robust Embeddings** (arXiv:2505.18884).
- **Representation Bending for Safety** (arXiv:2504.01550) — RepBend defense.

### Datasets Used / Available

- **JBShield Attack Dataset** (USENIX 2025) — 9 attack methods × thousands of examples.
- **MultiJail** (DAMO-NLP-SG, ICLR 2024) — 315 prompts × 9 languages. Used in our cold-start evaluation.
- **HarmBench** — standard harmful prompt benchmark.
- **AdvBench** (Zou et al. 2023) — adversarial benchmark.
- **BELLS-Operational Jailbreak Dataset** (HuggingFace centrepourlasecuriteia) — 6,406 multilingual jailbreak prompts.
- **Alpaca** — benign prompt source for our 6,500-prompt benign set.

### Foundational ML / Theory

- **SupCon: Supervised Contrastive Learning** (Khosla et al., NeurIPS 2020).
- **Constitutional AI** (Bai et al., Anthropic, 2022).
- **Lorentz Model of Hyperbolic Geometry** (Cannon et al., 1997).
- **Hyperbolic Neural Networks** (Ganea et al., NeurIPS 2018).
- **Mahalanobis Distance OOD Detection** (Lee et al., NeurIPS 2018).
