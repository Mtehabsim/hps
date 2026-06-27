# Literature Review: Activation-Based Jailbreak Defenses (2024-2026)

**Purpose:** Comprehensive analysis of peer-reviewed and preprint activation-based jailbreak detection methods to identify which are direct competitors to C4 and HPS, and to validate the paper's central claim that "the field has not systematically compared against strong activation-level linear probes."

**Date:** May 2026
**Status:** Research complete; ready for reproduction prioritization

---

## Executive Summary (REVISED after evaluator feedback)

After thorough literature search across arxiv, ACL/EMNLP/NeurIPS/ICML/USENIX/WWW proceedings, OpenReview, Anthropic's alignment.anthropic.com blog, and Google DeepMind publications:

### Key findings (REVISED — earlier draft significantly overstated novelty)

1. **Linear probes / mean-pool probes on hidden states for jailbreak/harm detection are an ESTABLISHED technique with multiple peer-reviewed and industry-deployed instances:**
   - **Anthropic "Cheap Monitors"** (Cunningham et al. 2025) — mean-token probes, EMA probes, suffix probes; deployed for Claude 3 Sonnet against bioweapons content
   - **Google DeepMind "Production-Ready Probes for Gemini"** (arXiv:2601.11516, Jan 2026) — MultiMax and Max of Rolling Means Attention Probes; deployed for Gemini cyber-offensive monitoring
   - **"Detecting High-Stakes Interactions with Activation Probes"** (ICML 2025) — peer-reviewed at top ML venue
   - **"Activation Monitoring: Advantages of Using Internal Representations for LLM Oversight"** (OpenReview 2025) — explicit advocacy for activation probes
   - **Bricken et al. 2024** ("Features as Classifiers") — established baseline
   - **Kantamneni et al. 2025** — uses linear probes as baseline
   - **Latent Sentinel** (ICLR 2026 withdrawn) — per-layer probes
   - **When Benchmarks Lie** (ICLR 2026 AIWILD Workshop) — activation linear probes

2. **C4 (mean-pool 6 layers + LR) cannot be claimed as a novel "missed baseline".** The general approach is well-established, peer-reviewed at ICML 2025, and deployed in production at both Anthropic and Google DeepMind. As one industry summary put it: *"Activation probes achieve production-ready jailbreak robustness at orders-of-magnitude lower cost than LLM classifiers, with probe-first cascades now deployed at both Anthropic and Google DeepMind."*

3. **What IS still novel about our work:**
   - **Direct head-to-head comparison against geometric methods (HPS)** with parameter-matched ablation — unique
   - **Cold-start regime evaluation methodology** (varying N attacks/method, varying #methods, leave-one-out) — distinct from existing protocols
   - **Multi-LLM fragility analysis** of geometric vs linear methods — unique
   - **Multi-layer mean-pool variant** (vs single-layer or per-token aggregation in Anthropic/Google work) — minor architectural variation
   - **Methodology contributions** (threshold leakage protocol, parameter-matched baseline ablation) — applicable beyond this paper
   - **Confirmation that established findings extend to the geometric-framework setting** — incremental

4. **Most peer-reviewed JAILBREAK-SPECIFIC defense papers (HSF, JBShield, GradSafe, Gradient Cuff, Token Highlighter) STILL did not include strong activation-level linear-probe baselines** in their direct comparisons. This narrower claim remains true, but must be qualified: "Specific peer-reviewed jailbreak defense papers (HSF WWW 2025, JBShield USENIX 2025, etc.) did not test against the activation-probe approach that has since been validated in deployment by Anthropic and Google DeepMind."

5. **Updated paper framing:**
   - ✅ "We provide the first comprehensive comparison of activation-based geometric methods (HPS) against established linear-probe baselines (C4, in the spirit of Anthropic Cheap Monitors and Google DeepMind probes), demonstrating that geometry adds no advantage on standard jailbreak benchmarks."
   - ✅ "Specific peer-reviewed jailbreak defense papers did not include this baseline comparison."
   - ❌ "We discovered that linear probes work for jailbreak detection." (FALSE — Anthropic, Google, ICML 2025)
   - ❌ "C4 is the first linear probe tested for jailbreak detection." (FALSE)

---

## Comprehensive Method Inventory (REVISED)

### Category A1: Industry-deployed activation probe approaches (NEW — was missing)

These are the most damaging gap in our prior literature review. C4's approach was already established in industry.

| Method | Source | Year | Architecture | Status |
|---|---|---|---|---|
| **Anthropic Cheap Monitors** | Cunningham et al., Anthropic Alignment Blog | 2025 | Mean-token + EMA + suffix probes on Claude 3 Sonnet activations | Industry-deployed |
| **Gemini Production Probes** | Google DeepMind, arXiv:2601.11516 | Jan 2026 | MultiMax, Max of Rolling Means Attention Probes for cyber-offensive monitoring | Industry-deployed |
| **Detecting High-Stakes Interactions** | Bailey et al., ICML 2025 | 2025 | Activation probes for harm-pressure detection | **Peer-reviewed top venue** |
| **Activation Monitoring** | OpenReview 2025 | 2025 | Internal representation oversight | OpenReview |
| **Features as Classifiers** | Bricken et al. (Anthropic) | 2024 | Linear probes as baseline + SAE features | Industry research |
| **Red-teaming Activation Probes** | arXiv:2511.00554 | 2025 | Adversarial probing of activation probes | Recent |
| **Linear Probes Scale** | Nordby et al., arXiv:2604.13386 | Apr 2026 | Multi-layer probe ensemble for deception | Recent (different task) |

### Category A2: Peer-reviewed activation-based JAILBREAK-SPECIFIC defenses

These are the direct competitors for HPS. They work on LLM hidden states for jailbreak detection specifically.

| Method | Venue | Year | Architecture | Tested linear probe baselines? |
|---|---|---|---|---|
| **HSF (Hidden State Filter)** | WWW Companion 2025 | 2024-2025 | Last decoder layer + last-k tokens concatenated → MLP/linear classifier | No — only PPL, Self-Examination, Paraphrase, etc. |
| **JBShield** | USENIX Security 2025 | 2025 | Concept activation analysis + AND-gate detection | No — only PAPI, PPL, Llama Guard, Self-Ex, GradSafe |
| **RTV** | Preprint 2026 | 2026 | Refusal direction cosine fingerprint + Mahalanobis | No — only ablations |
| **GradSafe** | ACL 2024 | 2024 | Gradient norm of refusal-loss for safety-critical params | Compared mainly to Llama Guard |
| **Gradient Cuff** | NeurIPS 2024 | 2024 | Two-step refusal loss landscape analysis | No — focused on refusal landscape |
| **Token Highlighter** | AAAI 2025 | 2024-2025 | Affirmation Loss gradient over input tokens | No — different signal type |
| **Coverage Criteria** | arXiv 2408.15207 | 2024 | Neuron coverage (TKNC, NBC) features | No — different feature engineering |
| **Latent Sentinel** | **ICLR 2026 (WITHDRAWN)** | 2025 | Per-layer linear probes + cross-layer aggregation | This IS a linear probe approach — closest competitor |
| **HiddenDetect** | ACL 2025 (Long) | 2025 | VLM-only — internal layer activation patterns | VLM-specific, not directly comparable |
| **JailNeurons** | ICLR 2026 Poster | 2026 | LVLM hidden state neuron-level detection | VLM-specific |
| **STShield** | Preprint 2025 | 2025 | Single-token sentinel appended to response | Different paradigm (output-based) |
| **DELMAN** | (referenced) | 2024-2025 | Hidden-state intervention/detection | Need direct verification |
| **AdaSteer** | EMNLP 2025 | 2025 | LR-based steering coefficients on activations | Steering, not pure detection |
| **Jailbreaking Leaves a Trace** | arXiv 2602.11495 | 2026 | Tensor-based latent representation framework | Direct competitor (concurrent) |
| **Do Internal Layers Reveal Patterns?** | arXiv 2510.06594 (NeurIPS 2025 workshop) | 2025 | Layer-wise behavior analysis (preliminary) | Concurrent workshop paper |
| **When Benchmarks Lie** | ICLR 2026 AIWILD Workshop | 2026 | Activation linear probes + LODO evaluation | Direct competitor (concurrent) |
| **Linear Probe Accuracy Scales** | arXiv 2604.13386 | 2026 | Multi-layer probe ensemble for deception (not jailbreak) | Multi-layer approach but different task |

### Category B: Probe-based interpretability work (related but different focus)

| Method | Venue | Year | What they do |
|---|---|---|---|
| **Probing Latent Subspaces** | arXiv 2503.09066 | 2025 | LDA on activations to reveal latent subspaces |
| **What Features in Prompts Jailbreak LLMs?** | arXiv 2411.03343 | 2024 | Per-layer linear and non-linear probes |
| **Linearly Decoding Refused Knowledge** | arXiv 2507.00239 | 2025 | Linear probes to extract refused information (attack, not defense) |

### Category C: Different defense categories (NOT direct competitors to C4)

Comparing C4 to these would be unfair — different problem categories.

| Method | Type | Why not directly comparable |
|---|---|---|
| Llama Guard 1/2/3 | External classifier model | Separate fine-tuned LLM, different paradigm |
| WildGuard | External classifier | Same |
| PromptGuard 2 | External classifier | Same |
| Perspective API | External text classifier | Different paradigm |
| ShieldGemma | External classifier | Same |
| SmoothLLM | Output-based perturbation | Different paradigm |
| PARDEN | Paraphrase + detection | Output-based |
| Self-Examination | LLM self-judges | Output-based |

---

## Detailed Method Analysis

### 1. HSF (Hidden State Filter) — Qian et al., WWW Companion 2025

**Architecture (verified by reading the paper):**
- Uses **only the last decoder layer**
- Extracts hidden states from the **last k tokens** (k tested 1-8, k=7 best for Llama2-7B-chat)
- Concatenates last k tokens with **zero padding between them**: T_k = [t_k, 0, t_2, 0, ..., 0, t_1]
- Final classifier is: f_k(x) = w_k^T · T_k + b_k → sigmoid → harmfulness score
- Despite calling it "MLP" the equation shows pure linear classifier

**Training data:**
- 3,000 harmful from UltraSafety
- 3,000 harmful from PKU-SafeRLHF-prompt
- 6,000 benign from databricks-dolly-15k
- Plus jailbreak augmentation for Mistral

**Stated baselines (their Section 5.1):**
- PPL (input perplexity)
- Self-Examination (Phute et al. 2023)
- Paraphrase (Jain et al. 2023)
- Retokenization (Jain et al. 2023)
- Self-Reminder (Xie et al. 2023)
- DRO (safety prompt-based)
- SPD (single-pass logit detection)

**Critical observation:** **HSF did NOT compare against:**
- Multi-layer linear probes
- Mean-pool baselines
- GradSafe, Gradient Cuff
- JBShield (concurrent work)

**HSF vs C4 differences:**

| Aspect | HSF | C4 |
|---|---|---|
| Layers | Last decoder layer only | 6 layers (spread) |
| Token aggregation | Last k tokens concatenated | Last token only |
| Multi-layer | ❌ | ✅ (mean-pool) |
| Classifier | Linear (sigmoid) | Logistic regression |
| Generalization | "Does not generalize across models" (paper's own admission) | Tested across 2 LLMs |

**Reproducibility:** Code at https://anonymous.4open.science/r/Hidden-State-Filtering-8652/

**Reproduction priority: HIGH** — closest peer-reviewed competitor, must be in paper for USENIX/CCS targets.

### 2. JBShield — Zhang et al., USENIX Security 2025

**Architecture:** Concept activation analysis identifying "toxic" and "jailbreak" concepts in activations; AND-gate decision (input must activate both concepts to be flagged).

**Stated baselines:** PAPI (Perspective API), PPL, Llama Guard 7B fine-tuned, Self-Ex, GradSafe.

**Reported F1:** 0.94 average across 9 attacks on 5 models.

**Did NOT compare against:** Linear probes, mean-pool baselines, HSF, RTV.

**Reproducibility:** GitHub at https://github.com/NISPLab/JBShield

**Reproduction priority: COMPLETE** — already tested, reproduced at 55% accuracy.

### 3. GradSafe — Xie et al., ACL 2024

**Architecture:** Computes gradient of compliance loss w.r.t. safety-critical model parameters; magnitude indicates jailbreak likelihood.

**Stated baselines:** Compared mainly against Llama Guard (showed GradSafe outperforms Llama Guard without finetuning).

**Did NOT compare against:** Activation linear probes, HSF.

**Reproducibility:** GitHub at https://github.com/xyq7/GradSafe

**Reproduction priority: HIGH** — different signal type (gradient vs activation), good comparison for showing C4 generalizes.

### 4. Latent Sentinel — Kim & Lee, ICLR 2026 (WITHDRAWN)

**This is the closest direct competitor to C4 conceptually.**

**Architecture (from abstract):**
- "Tiny linear probes" on **frozen LLM hidden states of multiple Transformer layers**
- "Aggregate their scores in real time"
- Trained on JailbreakV-28k (50k adversarial + 50k benign)
- Tested on JailbreakBench, AdvBench, MultiJail
- Models: Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct
- "<0.003% parameters" added
- 98-100% detection on JailbreakBench/AdvBench

**Key claim:** "Adversarial intent is approximately linearly separable in LLM latent space."

**Status:** WITHDRAWN from ICLR 2026 (not peer-reviewed accepted)

**Differences from C4:**

| Aspect | Latent Sentinel | C4 |
|---|---|---|
| Probe type | Per-layer probes, then aggregated | Single mean-pool probe |
| Training data | JailbreakV-28k (large-scale) | Smaller curated set |
| Aggregation | Per-layer scores aggregated | Layer activations averaged before classifier |

**Implication for our paper:** Cannot claim C4 is the first to test linear probes on hidden states for jailbreak detection. Latent Sentinel is a concurrent withdrawn submission with similar conclusions.

**Reproduction priority: MEDIUM** — preprint only, not peer-reviewed. Cite as concurrent work supporting our claim.

### 5. When Benchmarks Lie — Fomin, ICLR 2026 AIWILD Workshop (ACCEPTED)

**Architecture:** Activation-based linear probes on LLM hidden states.

**Key findings:**
- Trains linear probes on activations across 18 datasets
- Proposes Leave-One-Dataset-Out (LODO) evaluation
- Shows aggregate AUC overestimates true OOD performance by 8.4 percentage points
- 28% of top SAE features are dataset-specific shortcuts
- All production guardrails (PromptGuard, Llama Guard, LLM-as-judge) fail on indirect attacks (7-37% detection)

**Implication for our paper:** Our cold-start regime evaluation is conceptually similar to LODO. We should cite this work and note our methodology aligns with it.

**Reproduction priority: LOW** — workshop paper, primarily methodological. Cite for methodology.

### 6. Jailbreaking Leaves a Trace — Kadali & Papalexakis, arXiv 2602.11495v2 (Feb 2026)

**Architecture:** Tensor-based latent representation framework on hidden activations.

**Key findings:**
- Layer-wise analysis across GPT-J, LLaMA, Mistral, Mamba
- 78% block rate of jailbreak attempts on LLaMA-3.1-8B
- Lightweight detection without fine-tuning

**Difference from C4:** Tensor-based features (more elaborate than mean-pooling). Different feature engineering.

**Reproduction priority: MEDIUM** — peer review status unclear (arXiv only). Cite as concurrent work.

### 7. Gradient Cuff — Hu et al., NeurIPS 2024

**Architecture:** Two-step detection using refusal loss landscape — checks both functional values and smoothness gradients.

**Stated baselines:** Compared against PPL, SmoothLLM, Erase-and-Check, Self-Reminder, RAIN.

**Did NOT compare against:** Activation-based linear probes.

**Reproducibility:** GitHub at https://github.com/IBM/Gradient-Cuff

**Reproduction priority: MEDIUM** — peer-reviewed at NeurIPS 2024, but uses different signal (gradients) so less directly comparable to C4.

---

## Comparison Matrix

The critical question: did each method test against multi-layer linear probes on activations?

| Method | Venue | Tested mean-pool LR? | Tested per-layer LR? | Tested concat-layer LR? | Tested HSF or similar? |
|---|---|---|---|---|---|
| HSF | WWW 2025 | ❌ | ❌ | ❌ | self |
| JBShield | USENIX 2025 | ❌ | ❌ | ❌ | ❌ |
| RTV | preprint | ❌ | ❌ | ❌ | ❌ |
| GradSafe | ACL 2024 | ❌ | ❌ | ❌ | ❌ |
| Gradient Cuff | NeurIPS 2024 | ❌ | ❌ | ❌ | ❌ |
| Token Highlighter | AAAI 2025 | ❌ | ❌ | ❌ | ❌ |
| Coverage Criteria | arXiv 2024 | ❌ | ❌ | ❌ | ❌ |
| Latent Sentinel | ICLR 2026 withdrawn | ❌ | ✅ self | ❌ | ❌ |
| When Benchmarks Lie | ICLR 2026 workshop | ✅ self | ❌ | ❌ | ❌ |
| Jailbreaking Leaves a Trace | arXiv 2026 | ❌ | ❌ | ❌ | ❌ |
| **HPS (us)** | **target paper** | **✅ (C4)** | **✅** | **✅ (Fisher-8)** | **❌** |

**Conclusion:** Among **peer-reviewed** activation-based defenses, none have tested mean-pool linear probes on multi-layer activations. The two preprints/workshop papers that test something similar (Latent Sentinel, When Benchmarks Lie) are concurrent with our work and have been published in 2025-2026. We are the first peer-reviewed-quality study with comprehensive comparison if we publish soon.

---

## Updated Paper Framing

### Strong claim (defensible after evaluator feedback)

> "We provide the first comprehensive empirical comparison of geometric activation-based methods (HPS, our novel framework) against established linear-probe baselines (C4, structurally similar to Anthropic's mean-token probes and Google DeepMind's Gemini production probes). Across 9 attack families, 2 LLMs, multiple data regimes, and adversarial settings, we find that geometric priors provide no consistent advantage over established linear probe baselines. Our work confirms findings from concurrent industry research (Anthropic Cheap Monitors 2025; Google DeepMind 2026) extend to the geometric framework setting. Among peer-reviewed *jailbreak-specific* defenses (HSF WWW 2025, JBShield USENIX 2025, GradSafe ACL 2024, Gradient Cuff NeurIPS 2024, Token Highlighter AAAI 2025), none have included strong linear probe baselines in their comparisons — our work closes this gap."

### Claims to avoid (these are now FALSE given Anthropic/Google work)

- ❌ "C4 is the first linear probe tested for jailbreak detection." (FALSE — Anthropic Cheap Monitors)
- ❌ "No paper has tested linear probes on activations." (FALSE — multiple, including ICML 2025)
- ❌ "We discovered linear probes work for jailbreak detection." (FALSE — established in industry)
- ❌ "C4 reveals a gap the field missed." (FALSE — it's deployed at Anthropic and Google)

### What to ACTUALLY claim (defensible novel contributions)

1. **First comprehensive head-to-head comparison of geometric (HPS) vs linear-probe (C4) approaches** — existing papers test one or the other, not both with parameter matching
2. **Cold-start regime methodology** — varying N attacks/method, varying #methods, leave-one-out — is genuinely novel methodology
3. **Multi-LLM HPS fragility analysis** — others don't test geometric methods across multiple LLMs
4. **Methodology fixes** (threshold leakage protocol, parameter-matched baseline ablation, multi-seed reporting) — applicable beyond this paper
5. **Field-level observation specific to peer-reviewed jailbreak defenses** — HSF, JBShield, GradSafe, etc. did not include linear probe baselines (true even given Anthropic/Google work, since those are different problem categories)

---

## Reproduction Recommendations (REVISED)

### Tier 1: Must reproduce for any venue above TMLR
1. **HSF** (WWW 2025) — closest peer-reviewed jailbreak-specific competitor; ~1-2 weeks effort. Code available.
2. **GradSafe** (ACL 2024) — different signal type (gradient), well-cited; ~1 week effort. Code available.

These two reproductions close the biggest gap for jailbreak-specific peer-reviewed comparison. Total: 2-3 weeks.

### Tier 2: Acknowledge but cannot reproduce (industry/closed)
1. **Anthropic Cheap Monitors** — Claude 3 Sonnet specific; we can't reproduce on closed model. **Must cite prominently.**
2. **Gemini Production Probes** — Gemini specific; we can't reproduce on closed model. **Must cite.**

### Tier 3: Strong stretch additions
1. **Gradient Cuff** (NeurIPS 2024) — peer-reviewed at top venue; ~1 week. Code available.
2. **Coverage Criteria** (arXiv) — different feature engineering; ~3-5 days.

### Tier 4: Cite as concurrent/related (no reproduction needed)
- **Latent Sentinel** (withdrawn from ICLR 2026; cite as concurrent)
- **When Benchmarks Lie** (workshop paper; cite for LODO methodology)
- **Jailbreaking Leaves a Trace** (arXiv only; cite as concurrent)
- **Detecting High-Stakes Interactions** (ICML 2025; cite as established peer-reviewed prior art)

### Don't reproduce (different categories)
- Llama Guard / WildGuard / PromptGuard (external classifiers, different paradigm)
- SmoothLLM / PARDEN / Self-Examination (output-based, different paradigm)
- Token Highlighter (gradient-based on tokens, different paradigm)

---

## Revised Venue Strategy (DOWNGRADED after evaluator feedback)

Given the now-clearer literature picture (Anthropic + Google deployments, ICML 2025 prior art):

### TMLR submission (highest probability)
- **60-65% acceptance probability** (down from 65-70% in earlier draft)
- Existing data sufficient
- Reframe to acknowledge ALL prior art (Anthropic, Google, ICML 2025, plus concurrent preprints)
- Position contribution as: **comprehensive empirical comparison of geometric vs linear-probe methods + cold-start methodology + multi-LLM HPS fragility analysis**

### NeurIPS Safety Workshop / ICML Safety
- **50-60% acceptance probability**
- Existing data sufficient

### USENIX Security 2027 / NDSS / CCS
- **Probability: 20-30%**
- The "rethinking baselines" framing is significantly weaker now that Anthropic/Google have published this approach
- Possible only with strong "first comprehensive jailbreak-specific comparison + adversarial analysis" framing
- Risk: reviewer points to industry deployments as prior art

### AAAI / IJCAI 2027
- **35-45% with HSF + GradSafe reproduction**
- Better fit than USENIX given prior-art landscape

---

## Bottom Line for the Mentor (REVISED)

1. **C4 is NOT novel as a concept.** Linear probes / mean-token probes on hidden states for harm/jailbreak detection are an established technique with industry deployments at Anthropic and Google DeepMind, and peer-reviewed publication at ICML 2025. Our earlier framing claiming the field "missed simple baselines" was overstated.

2. **What IS novel about our work:**
   - Direct head-to-head comparison of geometric (HPS) vs linear (C4) — unique
   - Cold-start regime methodology — genuinely new
   - Multi-LLM HPS fragility analysis — others don't test geometric methods on multiple LLMs
   - Adversarial PGD comparison (with appropriate caveats about threat model)
   - Methodology fixes (threshold leakage, parameter matching) applicable beyond this paper

3. **Reproducing HSF + GradSafe is moderately important.** Without these, reviewers can ask "did you actually test against any peer-reviewed jailbreak-specific activation defense?" Currently the answer is "JBShield (with reproduction issues at 55% accuracy) and RTV (concurrent preprint)."

4. **Realistic best path:**
   - **Spend 2-3 weeks reproducing HSF + GradSafe** for jailbreak-specific peer-reviewed comparison
   - **Update paper to acknowledge Anthropic and Google work prominently** in related work
   - Submit to TMLR with the corrected framing
   - In parallel, prepare a USENIX/NDSS submission with sharper framing if relevant

5. **The honest paper title is something like:**
   - "Activation-Based Jailbreak Detection Revisited: Linear Probes, Geometric Methods, and the Cold-Start Regime"
   - Or: "Hyperbolic Geometry for Jailbreak Detection? A Rigorous Comparison Against Established Linear Probe Baselines"

   Both clearly position the work as comparative/empirical rather than "new SOTA defense" or "we discovered linear probes."

3. **Reproducing HSF + GradSafe is moderately important for any venue above TMLR.** Without these, reviewers can ask "did you actually test against any peer-reviewed activation-based defense?" Currently the answer is "JBShield (with reproduction issues at 55% accuracy) and RTV."

4. **Realistic best path:**
   - **Spend 2-3 weeks reproducing HSF + GradSafe**
   - Submit to TMLR with these reproductions included
   - In parallel, prepare a USENIX/NDSS submission with sharper framing
   - If TMLR accepts, that's the publication
   - If TMLR rejects, iterate based on feedback before USENIX

5. **The honest paper title is something like:**
   - "Activation-Based Jailbreak Detection Revisited: Linear Probes, Geometric Methods, and the Cold-Start Regime"
   - Or: "When Hyperbolic Geometry Meets Linear Probes: A Comprehensive Study of Activation-Based Jailbreak Detection"

Both clearly position the work as comparative/empirical rather than "new SOTA defense."

---

## Sources

**INDUSTRY-DEPLOYED ACTIVATION PROBE APPROACHES (added after evaluator feedback):**

- ⚠️ External link — [Anthropic: Cost-Effective Constitutional Classifiers via Representation Re-use ("Cheap Monitors")](https://alignment.anthropic.com/2025/cheap-monitors/) — accessed 2026-05-25 — **CRITICAL: established prior art for C4 approach**
- [Building Production-Ready Probes For Gemini (Google DeepMind)](https://arxiv.org/abs/2601.11516) — accessed 2026-05-25 — **CRITICAL: established prior art**
- [Detecting High-Stakes Interactions with Activation Probes (ICML 2025)](https://arxiv.org/abs/2506.10805) — accessed 2026-05-25 — **CRITICAL: peer-reviewed prior art at top ML venue**
- [Activation Monitoring: Advantages of Using Internal Representations for LLM Oversight](https://openreview.net/forum?id=qbvtwhQcH5) — accessed 2026-05-25
- [Red-teaming Activation Probes using Prompted LLMs](https://arxiv.org/abs/2511.00554) — accessed 2026-05-25
- [Investigating task-specific prompts and SAEs for activation monitoring](https://arxiv.org/abs/2504.20271) — accessed 2026-05-25
- [Linearly Decoding Refused Knowledge in Aligned Language Models](https://arxiv.org/abs/2507.00239) — accessed 2026-05-25

**JAILBREAK-SPECIFIC ACTIVATION DEFENSES:**

- [HSF: Defending against Jailbreak Attacks with Hidden State Filtering](https://arxiv.org/abs/2409.03788) — accessed 2026-05-25
- [GradSafe: Detecting Jailbreak Prompts via Safety-Critical Gradient Analysis](https://arxiv.org/abs/2402.13494) — accessed 2026-05-25
- [Latent Sentinel: Real-Time Jailbreak Detection with Layer-wise Probes](https://openreview.net/forum?id=tuFRx6Ww2n) — accessed 2026-05-25 (WITHDRAWN ICLR 2026)
- [AdaSteer: Your Aligned LLM is Inherently an Adaptive Jailbreak Defender](https://aclanthology.org/2025.emnlp-main.1248/) — accessed 2026-05-25
- [Gradient Cuff: Detecting Jailbreak Attacks by Exploring Refusal Loss Landscapes](https://arxiv.org/abs/2403.00867) — accessed 2026-05-25
- [Token Highlighter: Inspecting and Mitigating Jailbreak Prompts](https://arxiv.org/abs/2412.18171) — accessed 2026-05-25
- [Coverage Criteria in LLMs](https://arxiv.org/abs/2408.15207) — accessed 2026-05-25
- [JBShield: Defending LLMs through Activated Concept Analysis](https://arxiv.org/abs/2502.07557) — accessed 2026-05-25
- [Jailbreaking Leaves a Trace: Internal Representations](https://arxiv.org/abs/2602.11495) — accessed 2026-05-25
- [Do Internal Layers of LLMs Reveal Patterns for Jailbreak Detection?](https://arxiv.org/abs/2510.06594) — accessed 2026-05-25
- [When Benchmarks Lie: Evaluating Malicious Prompt Classifiers](https://openreview.net/forum?id=jWIOJOQqne) — accessed 2026-05-25 (ICLR 2026 AIWILD Workshop)
- [Linear Probe Accuracy Scales with Model Size](https://arxiv.org/abs/2604.13386) — accessed 2026-05-25
- [HiddenDetect: Detecting Jailbreak Attacks against LVLMs via Hidden States](https://arxiv.org/abs/2502.14744) — accessed 2026-05-25 (ACL 2025)
- [JailNeurons: Detecting Jailbreak in LVLMs](https://www.iclr.cc/virtual/2026/poster/10007389) — accessed 2026-05-25 (ICLR 2026)
- [Probing Latent Subspaces of LLMs for AI Security](https://arxiv.org/abs/2503.09066) — accessed 2026-05-25
- [What Features in Prompts Jailbreak LLMs?](https://arxiv.org/abs/2411.03343) — accessed 2026-05-25
- [STShield: Single-Token Sentinel for Real-Time Jailbreak Detection](https://arxiv.org/abs/2503.17932) — accessed 2026-05-25
- ⚠️ External link — [Bailey et al.: Obfuscated Activations Bypass LLM Latent-Space Defenses (ICLR 2026)](https://arxiv.org/abs/2412.09565) — accessed 2026-05-25
