# HPS Research Opportunities — Roadmap

**Last updated:** 2026-05-24
**Current decision:** Pursue Opportunity 1 (Cold-Start) + Opportunity 2 (Multilingual) as the primary paper framing.

---

## Background: What the Diagnostic Revealed

After fixing methodology bugs (threshold leakage, parameter-fair Euclidean baseline, train/test overlap), the original "+0.302 AUROC always wins" claim did not hold up on Llama-3-8B with full data (5216 attacks, 9 methods). However, the diagnostic (`diagnostic_hps_vs_euc.py`) revealed a **clear inflection point**:

| N attacks/method | HPS TPR | Euclidean TPR | Δ |
|---|---|---|---|
| 25 | 0.996 | 0.440 | **+0.556** |
| 50 | 0.993 | 0.580 | **+0.413** |
| 100 | 0.990 | 0.731 | **+0.259** |
| 250 | 0.983 | 0.938 | +0.045 |
| 500 | 0.755 | 0.974 | -0.219 |

**Vicuna-like conditions on Llama-3 (4 methods, 60 attacks/method):** HPS=0.971, Euc=0.337, **Δ=+0.633** — replicates the original Vicuna result.

**Conclusion:** Hyperbolic geometric priors are a **low-data regularizer**. They dominate when labeled attack data is scarce (≤100 examples per method, ≤4 attack methods) and yield to Euclidean approaches as data scales.

This is consistent with established findings in the hyperbolic few-shot learning literature (see Theoretical Backing section below).

---

## Theoretical Backing — Existing Literature

The empirical inflection point connects directly to a decade of work on hyperbolic embeddings as inductive bias for few-shot learning:

| Paper | Key Finding |
|---|---|
| Hyperbolic vs Euclidean Embeddings in Few-Shot Learning (arXiv:2309.10013) | Direct comparison shows hyperbolic excels in few-shot |
| Is Hyperbolic Space All You Need for Medical Anomaly Detection? (arXiv:2505.21228) | "Hyperbolic space exhibits resilience to parameter variations and excels in few-shot scenarios, where healthy images are scarce" |
| HCFSLN: Adaptive Hyperbolic Few-Shot Learning (arXiv:2511.06988) | Same pattern in multimodal anxiety detection |
| HyperVD (arXiv:2305.18797) | Hyperbolic embeddings for weakly-supervised violence detection |
| HypAD (CVPR Workshop 2023) | Anomaly detection with hyperbolic embeddings |

**Implication for the paper:** The geometric prior story has theoretical motivation independent of jailbreak detection. We should cite this literature in Section 3 as motivation rather than presenting the inflection point as a novel discovery.

---

## Opportunity 1: Cold-Start Detection for Emerging Attacks 🏆 PRIMARY

### The Story
When a new jailbreak family emerges (TombRaider EMNLP 2025, ICE ACL 2025, SequentialBreak ACL 2025 SRW), defenders face a cold-start problem. They have days to weeks before enough examples accumulate. Existing supervised approaches (input classifiers, embedding-based detectors) need hundreds of examples to generalize. Zero-shot approaches sacrifice accuracy.

HPS gives 99%+ cross-attack TPR with as few as 25 examples per method, where parameter-matched Euclidean approaches achieve only 44%.

### Verified Emerging Attacks (2024-2025)

| Attack | Year | Venue | Mechanism |
|---|---|---|---|
| TombRaider | 2025 | EMNLP 2025 | Two-agent (inspector + attacker) extracting historical knowledge |
| ICE | 2025 | ACL 2025 Findings | Single-query Intent Concealment + Diversion |
| SequentialBreak | 2024-2025 | ACL 2025 SRW | Embeds harmful prompt within benign sequence chains |
| DIA (Dialogue Injection) | 2025 | arXiv:2503.08195 | Leverages dialogue history |
| EquaCode | 2025 | arXiv:2512.23173 | Multi-strategy single-query (math equations + code) |
| StructAttack | 2025 | arXiv:2603.07590 | Semantic blueprint assembly |
| Bit-flip jailbreaks | 2024-2025 | arXiv:2412.07192 | Hardware fault injection on weights |

### Comparison to Existing Cold-Start Defenses

| Method | Approach | Limitation |
|---|---|---|
| ALERT (arXiv:2601.03600) | Zero-shot via internal discrepancy | Imprecise, often overestimates risk |
| Anthropic's "Rapid Response" (arXiv:2411.07494) | Few-shot via input-classifier proliferation | Text-level features only; needs LLM-generated proliferation |
| GradSafe (ACL 2024) | Gradient-based zero-shot | Strong baseline but no cross-attack generalization analysis |
| **HPS (proposed)** | **Activation-based, 25-example few-shot** | **Requires white-box access** |

### Effort to Add to Paper
- Reproduce TombRaider, ICE, SequentialBreak attacks on Llama-3 (~50 examples each): **2-3 days**
- Include as held-out attack methods in cross-attack evaluation: **1 day**
- Write up cold-start framing in Section 5: **1-2 days**

**Total: ~1 week**

### Venue Fit
- USENIX Security ⭐⭐⭐ (cold-start is a deployment story they value)
- NDSS ⭐⭐⭐
- CCS ⭐⭐⭐

### Realistic Acceptance Probability
**35-45%** at top security venue with the cold-start framing.

---

## Opportunity 2: Multilingual / Low-Resource Language Jailbreaks 🏆 PRIMARY (combine with #1)

### The Story
Low-resource language jailbreaks (Zulu, Xhosa, Afrikaans, Arabic transliteration, Scots Gaelic) achieve 75-80% ASR on commercial LLMs. Safety alignment is concentrated in high-resource languages, leaving non-English attacks under-defended. Defenders have very limited labeled data per language. The cold-start framing extends naturally to "cold-start per language."

The dataset already includes "zulu" attacks; HPS detects them at 90.5% same-distribution TPR (Llama-3 results from main pipeline).

### Established Literature
- Multilingual Jailbreak Challenges (arXiv:2310.06474, ICLR 2024) — original demonstration of low-resource attack effectiveness
- Multilingual Jailbreaking with Low-Resource Languages (arXiv:2605.18239) — tested on isiZulu, isiXhosa, Afrikaans, Kiswahili
- Jailbreaking LLMs with Arabic Transliteration and Arabizi (arXiv:2406.18725)
- The State of Multilingual LLM Safety Research (arXiv:2505.24119) — documents the language gap
- Multilingual Safety Alignment via Self-Distillation (arXiv:2605.02971)

### Effort to Add
- Add 2-3 more low-resource languages with ~50 attack examples each: **1-2 days**
- Cross-language transfer evaluation (train on EN, test on Zulu, etc.): **1 day**

**Total: ~3 days**

### Venue Fit
- USENIX Security ⭐⭐ (security relevance)
- ACL / EMNLP ⭐⭐⭐ (multilingual NLP focus)

### Realistic Acceptance Probability
**40-50%** at security venue, **50-60%** at *ACL.

### Why Combine with Opportunity 1
Both share the cold-start framing. A combined paper:
- Headline: data-efficient jailbreak detection
- Two case studies: emerging attack families + low-resource languages
- Stronger evaluation breadth → stronger paper

---

## Opportunity 3: Agentic / Tool-Use Jailbreak Detection 🥈 FUTURE WORK

### The Story
LLM agents with tool access (code execution, web browsing, API calls) face a growing threat from indirect prompt injection (IPI). Examples are scarce because they require complex environments. Current defenses are mostly input-filtering; activation-level monitoring is an open area.

### Verified Attack Landscape
- InjecAgent benchmark (arXiv:2403.02691) — 1,054 test cases across 17 user tools, 62 attacker tools
- AgentDojo benchmark — standard evaluation
- AgentVigil (arXiv:2505.05849) — black-box red-teaming, 71% success against o3-mini
- Adaptive Tool-Based IPI (arXiv:2602.20720) — exploiting MCP integrations
- IPIGuard (arXiv:2508.15310) — defense via tool-dependency graph
- Targeted Bit-Flip Attacks on LLM-based Agents (arXiv:2603.10042)

### Why Defer
- High engineering cost (set up AgentDojo, generate attacks, adapt HPS to multi-turn activation patterns)
- Crowded space (multiple defenses being proposed)
- Better as a **follow-up paper** after the cold-start paper is established

### Estimated Effort: 4-8 weeks

### Venue Fit (when ready)
- USENIX Security ⭐⭐⭐
- CCS ⭐⭐⭐
- NDSS ⭐⭐⭐
- IEEE S&P ⭐⭐

---

## Opportunity 4: VLM / Multimodal Extension 🥈 FUTURE WORK

### The Story
Extend HPS to vision-language model jailbreak detection (IDEATOR, BVS, Bi-Modal Adversarial Prompt). Multimodal attacks have enormous attack surface and sparse data per modality combination.

### Direct Competitor — Critical Caveat
**arXiv:2512.12069 (December 2025)** — "Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring" — already does contrastive scoring on VLMs. This is direct prior art.

To publish in this space, must demonstrate clear differentiation:
- Few-shot advantage they don't evaluate
- Multi-layer trajectory framework vs. single-layer scoring
- Cross-attack vs. their same-distribution evaluation

### Verified VLM Attacks
- IDEATOR (ICCV 2025, arXiv:2411.00827) — VLM jailbreaks VLM
- BVS (arXiv:2601.15698) — semantic-agnostic for harmful image generation
- Bi-Modal Adversarial Prompt (arXiv:2406.04031)
- Universal Transferable VLM Jailbreak (arXiv:2602.01025)

### Estimated Effort: 4-6 weeks

### Venue Fit (when ready)
- ICCV / CVPR (vision focus)
- NeurIPS / ICLR (ML focus)
- USENIX (less natural fit due to ML emphasis)

---

## Opportunity 5: Continual / Streaming Detection 🥉 SUPPORTING WORK

### The Story
Hybrid system that switches geometry based on data availability:
- HPS as initial detector when N < 100 per method
- Euclidean takeover as N grows
- Automatic regime detection

### Why Lower Priority
This is more of an engineering paper than a research contribution. The interesting science is the inflection point itself (Opportunity 1 covers this). The hybrid system is an obvious application.

### Estimated Effort: 1-2 weeks

### Venue Fit
- Workshop papers (NeurIPS Safety, ICLR Safety)
- Applied AI venues
- As a paragraph in the cold-start paper rather than its own paper

---

## Opportunity 6: Enterprise / Custom Model Deployment 🥉 PRACTICAL BUT HARD

### The Story
Custom fine-tuned LLMs in regulated domains (medical, legal, financial) often have weakened safety alignment and unique attack surfaces. Each enterprise has different threats based on their system prompt and tools. Privacy prevents data sharing across organizations.

### Why Hard
- Hard to publish without enterprise data access
- Privacy concerns limit empirical validation
- Each deployment has unique threat model

### When Useful
- As a deployment case study **after** publishing the academic paper
- For industry impact metrics rather than academic publication

### Venue Fit
- USENIX SOUPS (security in practice)
- Industry tracks at security conferences
- Trade publications

---

## Recommended Path Forward (Combined Opportunity 1 + 2)

### Paper Title (working)
**"Data-Efficient Jailbreak Detection via Hyperbolic Geometric Priors: Defending Against Emerging Attacks and Low-Resource Languages"**

Or, more concise:

**"Geometric Priors for the Cold-Start Problem in Jailbreak Detection"**

### Paper Structure

1. **Introduction**
   - Cold-start problem when novel jailbreak families emerge
   - Multilingual safety gap as a parallel cold-start scenario
   - Existing solutions: zero-shot (imprecise) or supervised (data-hungry)
   - Contribution: hyperbolic geometric prior as data-efficient alternative

2. **Related Work**
   - Jailbreak detection (JBShield, RTV, GradSafe, ALERT)
   - Few-shot defense (Anthropic's Rapid Response)
   - Hyperbolic embeddings for few-shot learning (theoretical foundation)
   - Multilingual safety (the language gap)

3. **Method**
   - HPS architecture (Lorentz projection, trajectory features)
   - Why hyperbolic — connect to few-shot learning literature
   - RTV ensemble for complementary signals

4. **Experimental Setup**
   - Models: Vicuna-13B, Llama-3-8B
   - Attack families:
     - **Established**: GCG, PAIR, AutoDAN, JBC, SAA, IJP, Puzzler, DrAttack, Base64
     - **Emerging (Opportunity 1)**: TombRaider, ICE, SequentialBreak
     - **Multilingual (Opportunity 2)**: Zulu, Arabic, Afrikaans
   - Cross-attack evaluation protocol

5. **Main Results**
   - Headline plot: TPR vs N attacks/method (TEST 2 inflection point)
   - Headline number: HPS 99.6% TPR @ 25 examples vs Euclidean 44%
   - Same-distribution and cross-attack tables
   - Per-method breakdown including emerging and multilingual attacks

6. **Analysis**
   - When does the geometric prior help? (data regime, method diversity)
   - SAA failure mode investigation
   - Layer selection ablation
   - Theoretical motivation from few-shot learning literature

7. **Comparison with Existing Defenses**
   - JBShield (USENIX Security 2025) — reproduce and break
   - RTV (Derya & Sunar 2026) — direct comparison
   - Anthropic's Rapid Response (arXiv:2411.07494) — apples-to-apples on emerging attacks
   - ALERT zero-shot — when no training data is available
   - GradSafe — gradient-based comparison

8. **Adaptive Evaluation**
   - PGD on activations breaks at ε=0.05
   - Honest discussion of limitations
   - Token-level adaptive attacks (if time)

9. **Limitations and Discussion**
   - White-box access required
   - Per-language activation extraction needed
   - Adaptive robustness limited

10. **Conclusion**

### Headline Numbers
- HPS 99.6% TPR @ 25 examples per method (cold-start)
- HPS+RTV ensemble: 96.2% same-distribution TPR on Llama-3 (full data)
- 91.2% mean cross-attack TPR (Llama-3, full data, ensemble)
- vs JBShield-D: +40.9% absolute accuracy

### Effort Estimate
- Diagnostic completion (TEST 7+9): 1 hour
- Re-run main pipeline with finalized hyperparameters (50 epochs, spread layers): 6-9 hours
- Reproduce TombRaider, ICE, SequentialBreak attacks: 2-3 days
- Add multilingual attacks (Arabic, Afrikaans): 1-2 days
- Run cold-start ablation with new attacks: 1 day
- Update paper text: 2-3 days
- Total to submission-ready: **~2 weeks of focused work**

### Target Submission
- USENIX Security 2027 (deadlines: Feb 2027 spring, Aug 2027 fall)
- CCS 2027 as backup
- AAAI 2027 as fallback

---

## Decision Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-24 | Pursue Opportunity 1 (Cold-Start) + 2 (Multilingual) as combined paper | Strongest empirical evidence (inflection point at 25-100 examples), theoretical backing from hyperbolic few-shot literature, deployment-relevant story for security venues |
| 2026-05-24 | Defer Opportunity 3 (Agentic) and 4 (VLM) to follow-up papers | Higher engineering cost, crowded competition (4), better as second papers after establishing the cold-start framing |

---

## Sources

- [TombRaider: Entering the Vault of History to Jailbreak LLMs (EMNLP 2025)](https://arxiv.org/abs/2501.18628) — accessed 2026-05-24
- [Exploring Jailbreak Attacks on LLMs through Intent Concealment and Diversion (ACL 2025 Findings)](https://arxiv.org/abs/2505.14316) — accessed 2026-05-24
- [SequentialBreak: Embedding Jailbreak Prompts into Sequential Prompt Chains (ACL 2025 SRW)](https://arxiv.org/abs/2411.06426) — accessed 2026-05-24
- [Jailbreaking LLMs with Fewer Than Twenty-Five Targeted Bit-flips](https://arxiv.org/abs/2412.07192) — accessed 2026-05-24
- [Mitigating LLM Jailbreaks with a Few Examples (Anthropic, 2024)](https://arxiv.org/abs/2411.07494) — accessed 2026-05-24
- [ALERT: Zero-shot LLM Jailbreak Detection via Internal Discrepancy Amplification](https://arxiv.org/abs/2601.03600) — accessed 2026-05-24
- [IDEATOR: Jailbreaking VLMs Using VLMs (ICCV 2025)](https://arxiv.org/abs/2411.00827) — accessed 2026-05-24
- [BVS: Jailbreaking MLLMs for Harmful Image Generation via Semantic-Agnostic Inputs](https://arxiv.org/abs/2601.15698) — accessed 2026-05-24
- [Multilingual Jailbreak Challenges in Large Language Models (ICLR 2024)](https://arxiv.org/abs/2310.06474) — accessed 2026-05-24
- [Multilingual Jailbreaking of LLMs Using Low-Resource Languages](https://arxiv.org/abs/2605.18239) — accessed 2026-05-24
- [Jailbreaking LLMs with Arabic Transliteration and Arabizi](https://arxiv.org/abs/2406.18725) — accessed 2026-05-24
- [The State of Multilingual LLM Safety Research](https://arxiv.org/abs/2505.24119) — accessed 2026-05-24
- [InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated LLM Agents](https://arxiv.org/abs/2403.02691) — accessed 2026-05-24
- [Adaptive Tool-Based Indirect Prompt Injection Attacks on Agentic LLMs](https://arxiv.org/html/2602.20720) — accessed 2026-05-24
- [Targeted Bit-Flip Attacks on LLM-based Agents](https://arxiv.org/html/2603.10042) — accessed 2026-05-24
- [Hyperbolic vs Euclidean Embeddings in Few-Shot Learning](https://arxiv.org/abs/2309.10013) — accessed 2026-05-24
- [Is Hyperbolic Space All You Need for Medical Anomaly Detection?](https://arxiv.org/abs/2505.21228) — accessed 2026-05-24
- [HCFSLN: Adaptive Hyperbolic Few-Shot Learning](https://arxiv.org/html/2511.06988) — accessed 2026-05-24
- [Rethinking Jailbreak Detection of Large VLMs with Representational Contrastive Scoring](https://arxiv.org/abs/2512.12069) — accessed 2026-05-24
- [GradSafe: Detecting Jailbreak Prompts via Safety-Critical Gradient Analysis (ACL 2024)](https://arxiv.org/abs/2402.13494) — accessed 2026-05-24
- [Dialogue Injection Attack: Jailbreaking LLMs through Context Manipulation](https://arxiv.org/abs/2503.08195) — accessed 2026-05-24
- [JBShield: Defending LLMs from Jailbreak Attacks (USENIX Security 2025)](https://arxiv.org/abs/2502.07557) — accessed 2026-05-24
- [Revisiting JBShield: Breaking and Rebuilding Representation-Level Jailbreak Defenses (RTV)](https://arxiv.org/abs/2605.03095) — accessed 2026-05-24
