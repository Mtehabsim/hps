# Paper Outline: Geometric vs Linear-Probe Activation Defenses for Jailbreak Detection

**Status:** Outline addressing all critical issues identified by AI evaluator
**Target venue:** TMLR (primary), NeurIPS Safety Workshop (backup)
**Estimated length:** 10-12 pages + appendices

---

## Working Title Options

1. *"Hyperbolic Geometry for Jailbreak Detection? A Rigorous Comparison Against Established Linear Probe Baselines"*
2. *"Geometric vs Linear Probes for Activation-Based Jailbreak Detection: A Comprehensive Empirical Study"*
3. *"When Geometry Doesn't Help: A Controlled Study of Hyperbolic Priors for LLM Jailbreak Detection"*

Recommendation: Option 1 — frames as honest scientific question (with negative answer), positions HPS as the test, includes "established baselines" to acknowledge prior art upfront.

---

## Abstract Draft

Activation-based jailbreak detection has been explored from multiple angles: linear probes on hidden states (Anthropic Cheap Monitors 2025; Google DeepMind 2026; Bailey et al. ICML 2025), specialized geometric architectures, refusal-direction methods, and gradient-based approaches. In this work, we ask whether hyperbolic geometric priors—motivated by the empirical δ-hyperbolicity of LLM hidden states (HypLoRA, HELM)—provide advantages over established linear probe baselines for jailbreak detection.

We propose **HPS (Hyperbolic Projection Sentinel)**, a novel framework using learned Lorentz projections of multi-layer activations with contrastive loss and trajectory features (radial, curvature, displacement). We compare HPS directly against **C4**, a deliberately minimal mean-pool linear probe baseline (structurally similar to Anthropic Cheap Monitors), as well as reproduced peer-reviewed methods (HSF WWW 2025, GradSafe ACL 2024, JBShield USENIX 2025, RTV preprint 2026).

Across 9 attack families, 2 LLMs (Vicuna-13B, Llama-3-8B), multi-seed evaluation, cold-start regime analysis, and activation-space perturbation testing, we find:
- At saturation, HPS and C4 perform comparably (TPR > 99%); geometric priors add no advantage
- HPS exhibits significant fragility on Vicuna (cold-start TPR drops to 0.068)
- C4 is more robust to activation-space perturbations than HPS
- Single-feature analysis shows HPS's trajectory features collapse to one scalar
- Empirical radial distribution contradicts the geometric hypothesis: benign prompts occupy higher radial position than attacks

We additionally propose **cold-start regime evaluation methodology** (varying N attacks per method, varying #methods, leave-one-out) as a contribution applicable beyond this work.

Our findings extend established results from industry deployments (Anthropic, Google DeepMind) and ICML 2025 to the geometric framework setting, and identify which design choices matter and which don't.

---

## Section 1: Introduction (1.5 pages)

### 1.1 Motivation
- Jailbreak attacks on aligned LLMs are a major safety concern
- Activation-based detection examines internal LLM state, complementing prompt-level defenses
- Multiple approaches: linear probes (Anthropic Cheap Monitors, Google DeepMind, ICML 2025), specialized geometric methods (RTV, HSF, JBShield), gradient-based (GradSafe, Gradient Cuff)

### 1.2 Research Question
Hyperbolic geometry has been shown to capture hierarchical structure in LLM token embeddings (HypLoRA NeurIPS 2025, HELM NeurIPS 2025). Does this property translate to advantages for activation-based jailbreak detection?

### 1.3 Contributions
1. **HPS framework**: Novel architecture combining learned Lorentz projection + multi-layer trajectory features + LR classifier
2. **Comprehensive comparison**: Direct head-to-head of HPS vs C4 (linear probe), HSF, GradSafe, Gradient Cuff, JBShield, RTV with parameter-matched ablations
3. **Cold-start methodology**: Novel evaluation protocol for low-data regimes
4. **Multi-LLM fragility analysis**: Documenting HPS's catastrophic Vicuna failure
5. **Honest negative result**: Geometric priors do not help on standard activation-based jailbreak detection benchmarks; established linear probes (Anthropic, Google) are matched, not exceeded
6. **Statistical rigor**: Multi-seed, parameter-matched, threshold-leakage-free protocol with formal statistical tests

### 1.4 Roadmap

---

## Section 2: Related Work (1.5 pages)

### 2.1 Linear-probe activation monitoring (acknowledge prior art prominently)
- Anthropic Cheap Monitors (Cunningham et al. 2025) — mean-token, EMA, suffix probes for Claude
- Google DeepMind Production-Ready Probes for Gemini (arXiv:2601.11516) — MultiMax, attention probes
- Detecting High-Stakes Interactions with Activation Probes (ICML 2025)
- Activation Monitoring (OpenReview 2025)
- Bricken et al. 2024 (Features as Classifiers)
- Linear Probe Accuracy Scales (arXiv:2604.13386)
- "What Features in Prompts Jailbreak LLMs?" (arXiv:2411.03343) — caveat: probes fail to generalize OOD
- "When Benchmarks Lie" (ICLR 2026 AIWILD) — caveat: AUC inflation under distribution shift
- Latent Sentinel (ICLR 2026 withdrawn) — concurrent

### 2.2 Specialized activation-based jailbreak defenses
- HSF (WWW 2025) — last-layer + last-k tokens linear classifier
- JBShield (USENIX 2025) — concept activation analysis
- GradSafe (ACL 2024) — gradient-based detection
- Gradient Cuff (NeurIPS 2024) — refusal loss landscape
- Token Highlighter (AAAI 2025) — affirmation loss gradients
- RTV (preprint 2026) — refusal direction Mahalanobis
- Coverage Criteria (arXiv 2024) — neuron coverage features

### 2.3 Hyperbolic geometry for representations
- Nickel & Kiela 2017 — Poincaré embeddings for trees
- HypLoRA (Yang et al. NeurIPS 2025) — empirical δ-hyperbolicity in LLM embeddings
- HELM (He et al. NeurIPS 2025) — radial structure in token embeddings
- HyPE (ICLR 2026) — hyperbolic prompt embedding for harm detection (concurrent)

### 2.4 Adversarial robustness against activation-based defenses
- Bailey et al. 2024 (arXiv:2412.09565) — obfuscated activations bypass latent-space defenses (realistic threat model)
- Wollschläger et al. ICML 2025 — geometry of refusal

---

## Section 3: Methods (2 pages)

### 3.1 HPS: Hyperbolic Projection Sentinel

**Architecture:**
1. Extract activations at N=6 layers (selected by Fisher score or domain knowledge)
2. Apply learned linear projection W ∈ ℝ^(d×64) to each layer
3. Map to Lorentz hyperboloid via exponential map at origin
4. Compute trajectory features:
   - Radial features (5): mean/max/min/std/range of x_0
   - Curvature features (4): triangle-inequality bending
   - Displacement features (3): start-end distance, path length, progress ratio
5. Logistic regression on 12 trajectory features

**Training:** Per-layer-temperature contrastive loss in Lorentz space, 50 epochs, AdamW.

### 3.2 C4: Mean-Pool Linear Probe Baseline

A deliberately minimal baseline structurally similar to Anthropic Cheap Monitors mean-token probes:

```
1. Extract last-token activations at same N=6 layers
2. Mean-pool across layers: feature = (1/6) Σ h_l ∈ ℝ^4096
3. StandardScaler
4. Logistic regression with L2 regularization
```

Difference from Anthropic Cheap Monitors: we mean-pool over layers (last-token), they mean-pool over tokens (per-layer).

### 3.3 Reproduced Baselines

- **HSF** (Qian et al. WWW 2025): last decoder layer + last-k tokens concatenated → linear classifier. k=7 for Llama-3 (their best).
- **GradSafe** (Xie et al. ACL 2024): gradient norm of compliance loss w.r.t. safety-critical params.
- **Gradient Cuff** (Hu et al. NeurIPS 2024): two-step refusal loss landscape analysis.
- **JBShield-D** (Zhang et al. USENIX 2025): concept activation AND-gate.
- **RTV** (Derya & Sunar 2026): refusal direction cosine fingerprint + Mahalanobis.

### 3.4 Methodology Protocol

- **Threshold calibration**: held-out split, never test set
- **Parameter matching**: HPS Euclidean baseline has same parameter count
- **Multi-seed**: 5 seeds same-distribution, 3+ seeds cross-attack
- **Statistical tests**: paired bootstrap CIs, McNemar's test for HPS vs C4
- **Cold-start regime**: vary N attacks/method ∈ {5, 10, 25, 50, 100, 250, 500}, vary #methods ∈ {2, 3, 4, 5, 7, 9}

---

## Section 4: Experimental Setup (1 page)

### 4.1 Models
- Vicuna-13B-v1.5
- Llama-3-8B-Instruct

### 4.2 Attacks (9 families)
autodan, base64, drattack, gcg, ijp, pair, puzzler, saa, zulu — based on EasyJailbreak / similar to JailbreakBench attacks.

### 4.3 Benign Data
WildChat, Alpaca, OR-bench (matching Anthropic Cheap Monitors' setup where possible).

### 4.4 Evaluation Metrics
- AUROC (full curve)
- TPR @ 5% FPR (deployment-relevant)
- Cross-attack TPR (leave-one-out)
- PGD evasion rate at varying ε (activation-space perturbation analysis — see caveat in Section 6.4)

---

## Section 5: Main Results (3 pages)

### 5.1 Same-distribution detection

Table: HPS, C4, Euclidean-matched, HSF (reproduced), GradSafe (reproduced), JBShield (reproduced), RTV across 9 attacks on both LLMs.

**Key findings:**
- HPS, C4, Euclidean tie at TPR≈1.000 on Llama-3 (no statistically significant difference)
- HPS underperforms C4 on Vicuna (TPR=0.746 vs 1.000)
- HSF, GradSafe, JBShield, RTV achieve lower TPR than HPS/C4 on these benchmarks

### 5.2 Cold-start regime

**Plot:** TPR vs N per method, with 95% bootstrap CIs

**Findings:**
- Llama-3: HPS shows narrow advantage over Euclidean projection at extreme low-N (Δ=+0.733 at N=5), but C4 also benefits from same-N (Δ=+0.752 vs Euclidean)
- Vicuna: HPS catastrophically fails at low-N (TPR=0.068 at N=25); C4 maintains TPR > 0.93 throughout

### 5.3 Cross-attack generalization

**Caveat upfront:** Leave-one-out within our 9-attack benchmark is generous; not true distribution shift (cite "When Benchmarks Lie", "What Features...").

**Findings:**
- HPS = 0.997, C4 = 0.992 on Llama-3 (within statistical noise)
- HPS-ensemble + RTV achieves marginal improvement on per-attack coverage

### 5.4 Multi-LLM fragility analysis

**Key novel finding (no prior work tests this):**
- HPS optimal κ=0.1 on Llama-3, κ=2.0 on Vicuna — hyperparameters do NOT transfer
- 24-config hyperparameter sweep on Vicuna: best HPS = 0.769 vs C4 = 0.918
- This suggests geometric priors are model-fragile while linear probes are not

### 5.5 Activation-space perturbation analysis (renamed from "adversarial robustness")

**Caveat upfront:** This is NOT a realistic threat model. Real adversarial attacks operate in input space (cite Bailey et al. 2024). This analysis tests robustness *property* of features, not actual security.

**Findings:**
- HPS evasion 96% at ε=0.05; C4 evasion 4%
- HPS-Adv (PGD adversarial training) does not improve robustness
- HPS's single-feature bottleneck (mean_r alone) is exploitable in activation space
- C4's higher dimensionality (4096) is not exploitable in activation space

**This finding does not establish C4 is "more secure"** — only that HPS's compressed feature space has a vulnerability under this idealized attack model. Future work needs realistic input-space adaptive attacks.

### 5.6 Mechanistic analysis: radial distribution

**Critical finding (contradicts hypothesis):**
- Benign prompts: median radial position = 3.71
- Attack prompts: median radial position = 3.24

**Interpretation:** The contrastive loss finds an arbitrary discriminative direction. The Lorentz geometry constrains this to be radial. The semantic interpretation that "attacks are pushed to extreme periphery" is NOT supported by the data. The geometric prior provides class separation but not the hypothesized hierarchical semantics.

**This is direct mechanistic evidence supporting the negative finding** that geometric priors don't capture meaningful structure for this task.

---

## Section 6: Ablations and Analysis (1.5 pages)

### 6.1 Geometry: Hyperbolic vs Euclidean (parameter-matched)
- At saturation: no significant difference
- At cold-start: marginal difference, attributable to projection dimensionality not geometry

### 6.2 Trajectory features
- Single feature (mean_r) suffices
- Curvature features add no signal
- Single-feature ablation matches all-feature performance

### 6.3 Layer selection
- Spread layers > Fisher-discovered > late only > shallow only
- C4 is more robust to layer selection than HPS

### 6.4 Curvature κ
- Llama-3: optimal κ=0.1 (high curvature)
- Vicuna: optimal κ=2.0 (low curvature)
- κ acts more as regularization than as hierarchical prior

---

## Section 7: Limitations and Threats to Validity (0.75 pages)

### 7.1 Threat model: activation-space perturbations are not realistic
- Real attacks operate in input space (Bailey et al. 2024)
- Our PGD-on-activations is a feature robustness analysis, not security evaluation

### 7.2 Cross-attack generalization may reflect benchmark saturation
- Leave-one-out within shared benchmark family
- Cite "What Features in Prompts Jailbreak LLMs?" (probes fail OOD)
- Cite "When Benchmarks Lie" (8.4 pp AUC inflation under true distribution shift)

### 7.3 Non-standardized attack benchmark
- Our 9 attacks differ from JailbreakBench / HarmBench
- Document data sources and per-attack statistics
- Future work: standardize on community benchmarks

### 7.4 Linear probes are not novel; we test geometry on top of established baselines
- Anthropic Cheap Monitors, Google DeepMind probes, ICML 2025 establish the linear probe approach
- Our contribution is geometric vs linear comparison + cold-start methodology + multi-LLM fragility

### 7.5 Single-attack PGD is idealized
- Adaptive attackers would use input-space optimization
- Future work: realistic adaptive attacks à la Bailey et al.

### 7.6 Two LLMs is not enough
- Concurrent work (Gemini probes, etc.) tests on closed models we cannot access
- Future work: Mistral, Qwen, additional architectures

---

## Section 8: Discussion (0.75 pages)

### 8.1 What the empirical findings tell us
- Geometric priors don't help on current activation-based jailbreak benchmarks
- The activation space is highly linearly separable for current attacks
- Multiple architectures find discriminative directions; specifics don't matter at saturation

### 8.2 Implications for the field
- Specific peer-reviewed jailbreak defenses (HSF, JBShield, GradSafe, Gradient Cuff, Token Highlighter) should include linear probe baselines
- Industry deployments (Anthropic, Google) confirm linear probes work; specialized methods need to demonstrate clear advantage
- Cold-start methodology is needed for honest evaluation of low-data regimes

### 8.3 Open questions
- Multi-turn jailbreaks: do geometric priors help when conversations have hierarchical structure? (untested)
- Agentic/tool-use jailbreaks: do tree-structured action sequences benefit from hyperbolic? (untested)
- Cross-model transferable concept embeddings: can a defense work across LLMs without retraining?

---

## Section 9: Conclusion (0.5 pages)

We tested whether hyperbolic geometric priors help activation-based jailbreak detection. Through rigorous comparison against established linear probe baselines (Anthropic Cheap Monitors-style C4) and reproduced peer-reviewed methods (HSF, GradSafe, JBShield, RTV), we find that geometric priors provide no consistent advantage and exhibit fragility (Vicuna failure, perturbation vulnerability). Our cold-start regime methodology and multi-LLM analysis are contributions applicable beyond this paper. We hope this rigorous negative result helps the field focus on harder benchmarks (multi-turn, agentic, novel attacks) where geometric priors might genuinely help.

---

## Appendices

- A. Implementation details
- B. Reproduction notes for baselines
- C. Per-attack breakdown
- D. Hyperparameter sweep details
- E. Statistical test details
- F. Additional ablations
- G. Data card for benchmark

---

## Key Departures from Earlier Drafts

This outline addresses every issue raised by the AI evaluator:

| Issue | How addressed |
|---|---|
| C1: paper_draft.md stale | Full rewrite with this outline |
| C2: Missing Anthropic Cheap Monitors | Section 2.1 leads with this; abstract acknowledges; reframes contribution |
| C3: Cross-attack generalization claims | Section 5.3 and 7.2 add caveats; cite "When Benchmarks Lie" |
| C4: No CIs | Section 3.4 specifies; all results have bootstrap CIs |
| C5: Unrealistic adversarial | Section 5.5 renamed to "perturbation analysis"; 7.1 caveat |
| C6: Radial distribution contradicts hypothesis | Section 5.6 honest discussion as evidence for negative finding |
| N1-N6: Various | Addressed in Limitations and Discussion |

The negative finding is stronger when honestly presented than when artificially defended.
