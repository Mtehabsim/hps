# Reading List for Mentor: HPS Research Context

**Purpose:** Bring mentor up to speed on the literature underlying HPS research. Organized by priority (must-read first) with paragraphs explaining each paper's relevance.

---

## TIER 1: MUST READ (5 papers)

These are essential to understand the paper. Reading these is sufficient for the meeting.

### 1. Anthropic Cheap Monitors (Cunningham et al., 2025)

**Title:** "Cost-Effective Constitutional Classifiers via Representation Re-use"
**Authors:** Hoagy Cunningham, Alwin Peng, Jerry Wei, et al. (Anthropic)
**Link:** https://alignment.anthropic.com/2025/cheap-monitors/

**What they did:** Anthropic's production approach to LLM jailbreak detection. They trained linear probes on intermediate activations of Claude 3 Sonnet for safety classification, instead of using a separate jailbreak classifier. They tested three probe variants: mean-token probe (mean-pool tokens at one layer), EMA probe (per-token scoring with exponential moving average), and suffix probe (probe a specific token after appending a query suffix). They reported that their EMA probe matches a dedicated classifier with 2% of the policy model's parameters at minimal computational overhead. Their paper has a famous quote: "Linear probing as a baseline is difficult to beat."

**Why it matters to us:** This is THE paper our C4 baseline is adapted from. They use mean-token-probe (pool over tokens at one layer); we made a variant called C4 that pools over layers at the last token. We then ran their exact mean-token probe (MTP) and showed it equals our C4 variant (AUROC 0.9988 vs 0.9986). This means we cite Anthropic, not claim novelty. They also explicitly acknowledge they did NOT test against adaptive adversarial attacks — leaving the door open for Bailey et al.'s work to identify this gap.

---

### 2. Obfuscated Activations Bypass LLM Latent-Space Defenses (Bailey et al., ICLR 2025)

**Title:** "Obfuscated Activations Bypass LLM Latent-Space Defenses"
**Authors:** Luke Bailey, Alex Serrano, Abhay Sheshadri, Mikhail Seleznyov, Jordan Taylor, Erik Jenner, Jacob Hilton, Stephen Casper, Carlos Guestrin, Scott Emmons (Stanford / UC Berkeley / MIT)
**arxiv:** 2412.09565
**Link:** https://arxiv.org/abs/2412.09565

**What they did:** They demonstrated that ALL latent-space defenses (linear probes, MLP probes, sparse autoencoders, OOD detectors) are vulnerable to "obfuscation attacks" — adversarial 16-token suffixes optimized to (a) make the model output harmful content AND (b) make activations look benign to the probe. Their results: probes drop from 100% recall to 0% recall while maintaining 90% jailbreak success. They tested on Llama-3-8B-Instruct (same model as us). Even 70 rounds of adversarial training (defender keeps retraining against new attacks) failed — the attacker always found new bypasses. They also showed obfuscation attacks transfer between probe types.

**Why it matters to us:** This is the field-wide adversarial limitation. We do NOT claim adversarial robustness for HPS; we cite Bailey et al. as the reason no latent-space defense (including ours) can be claimed robust without explicit adaptive testing. Their finding affects HPS, C4, MTP, JBShield, HSF, GradSafe — every linear probe approach. This is the single most important "what doesn't work" paper for our space.

---

### 3. JBShield (Zhang et al., USENIX Security 2025)

**Title:** "JBShield: Defending Large Language Models from Jailbreak Attacks through Activated Concept Analysis and Manipulation"
**Authors:** Shenyi Zhang, Yuchen Zhai, Keyan Guo, Hongxin Hu, et al. (Wuhan University, Buffalo, Xi'an Jiaotong)
**arxiv:** 2502.07557
**Link:** https://arxiv.org/abs/2502.07557

**What they did:** They proposed two ideas: (1) "toxic concepts" and "jailbreak concepts" as linear subspaces in activation space, extracted via SVD on differences between paired benign/harmful prompts. (2) JBShield-D detector + JBShield-M mitigation. They report F1=0.94 average on 5 LLMs (Mistral-7B, Llama-2, Llama-3, Vicuna-7B, Vicuna-13B) × 9 attacks (autodan, base64, drattack, gcg, ijp, pair, puzzler, saa, zulu).

**Why it matters to us:** Two reasons.

1. **Their dataset is exactly what we use.** They released their attacks publicly at github.com/NISPLab/JBShield. We use their 9-attack categorization on Vicuna-13B-v1.5 — same as theirs. So our HPS Vicuna failure (gcg=7.6% vs C4=99.4%) is directly comparable to their reported numbers.

2. **They have the same length confound issue we identified.** Their data: 850 Alpaca harmless (~60 chars) + 9-attack mixture (some up to 2,193 chars for puzzler). Their paper does NOT analyze prompt-length distributions. They report F1=0.94 but never check if length alone could be the discriminator. Our finding (length-only AUROC=0.992 on this kind of data) suggests their numbers are partially length-driven. This is a methodology critique we apply field-wide, not just to them.

---

### 4. Hyperbolic Geometry of Latent Representations (HypLoRA + HELM)

**Title 1:** "HypLoRA: Boosting LoRA Efficiency with Hyperbolic Geometry" (Yang et al., NeurIPS 2025, arxiv:2405.18515)
**Title 2:** "HELM: Hyperbolic Embeddings for LLMs Reveal Hierarchical Geometry" (He et al., NeurIPS 2025, arxiv:2505.24722)
**Combined Link:** https://arxiv.org/abs/2405.18515, https://arxiv.org/abs/2505.24722

**What they did:** Both papers empirically demonstrate that LLM token embeddings have hierarchical/tree-like structure. HypLoRA shows δ-hyperbolicity (a quantitative measure of tree-likeness) in LLM representations. HELM finds power-law radial distributions and negative Ricci curvature in token embeddings. Together these papers establish that hyperbolic geometry is the theoretically appropriate space for LLM representations, since hyperbolic spaces have exponential volume growth (matching tree branching).

**Why it matters to us:** This is the THEORETICAL motivation for HPS. Hyperbolic geometry isn't an arbitrary choice — it's the principled choice if LLM activations have hierarchical structure. The geometric hypothesis we tested (attacks at higher radial position in Lorentz space) follows directly from this theoretical setup: harmful content is more "specific" (deeper in hierarchy) than benign content, so it should sit further from the origin. Our finding that 0/13 configurations show inversion (attacks > benign in radial position) confirms this prediction empirically.

---

### 5. Are GANs Created Equal? (Lucic et al., NeurIPS 2018)

**Title:** "Are GANs Created Equal? A Large-Scale Study"
**Authors:** Mario Lucic, Karol Kurach, Marcin Michalski, Sylvain Gelly, Olivier Bousquet
**arxiv:** 1711.10337
**Link:** https://arxiv.org/abs/1711.10337

**What they did:** They ran a large-scale controlled comparison of GAN variants (DCGAN, WGAN, WGAN-GP, BEGAN, etc.) using the same model architectures, hyperparameter tuning protocol, and evaluation metrics. They found that no GAN variant consistently outperformed the others; differences in published results came from differences in protocol, not from fundamental architectural superiority. The paper has 1,800+ citations and forced the field to use better evaluation protocols.

**Why it matters to us:** This is our methodological precedent. Our paper is in the same spirit: rigorous controlled comparison of activation-based jailbreak defenses, identifying that methodology issues (length confound, max_length, contamination) explain a large fraction of "improvements" in the field. We don't propose a winning method; we identify what's been confounded in the comparisons. TMLR specifically welcomes papers in this tradition. Mentor should read this to understand the framing.

---

## TIER 2: HIGHLY RECOMMENDED (5 papers)

These add important context. Read these if time permits.

### 6. Geometry of Refusal in Language Models (Wollschläger et al., ICML 2025)

**Title:** "Geometry of Refusal in Language Models"
**Authors:** Tobias Wollschläger, et al.
**arxiv:** 2502.17420
**Link:** https://arxiv.org/abs/2502.17420

**What they did:** Showed that refusal behavior in aligned LLMs is mediated by structured geometric directions in activation space. They identified specific directions in activation space that correspond to "refusing" vs "complying" with requests. By manipulating activations along these directions, they can either bypass refusals or strengthen them.

**Why it matters to us:** This is the geometric counterpart to our work. They show refusal has clean geometric structure; we test whether harm has clean geometric structure. They use Euclidean directions; we hypothesize hyperbolic radial position. Our findings (0/13 inversions, attacks at high radial) extend their refusal-direction work to harm representation. Cite as related work.

---

### 7. Refusal in Language Models is Mediated by a Single Direction (Arditi et al., NeurIPS 2024)

**Title:** "Refusal in Language Models is Mediated by a Single Direction"
**Authors:** Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, Neel Nanda
**arxiv:** 2406.11717
**Link:** https://arxiv.org/abs/2406.11717

**What they did:** They demonstrate that LLM refusal behavior is mediated by a single linear direction in activation space. Adding or subtracting this direction can make models refuse benign requests or comply with harmful ones. This provides the foundational empirical evidence that "refusal" is a clean, low-dimensional concept in activation space.

**Why it matters to us:** Foundational evidence that LLM activations encode safety-relevant concepts as identifiable subspaces. Our work investigates whether adding hyperbolic geometry on top of this insight provides additional benefit. Their result is consistent with our finding that linear probes (C4, MTP) suffice on aligned LLMs — if refusal is just one direction, probing for it doesn't need complex geometry.

---

### 8. Constitutional Classifiers (Sharma et al., Anthropic 2025)

**Title:** "Constitutional Classifiers: Defending against Universal Jailbreaks across Thousands of Hours of Red Teaming"
**Authors:** Mrinank Sharma, et al. (Anthropic)
**arxiv:** 2501.18837
**Link:** https://arxiv.org/abs/2501.18837

**What they did:** Anthropic's approach to deploying jailbreak defenses in production. They train classifiers using synthetic data generated by prompting LLMs with natural-language rules ("constitution"). They red-teamed extensively with thousands of hours of human attackers and showed robustness. This is the system Cheap Monitors aims to make more efficient.

**Why it matters to us:** Context for Cheap Monitors. Our paper compares against Anthropic's deployed approach (not the constitutional classifier itself but the cheaper variant). Demonstrates Anthropic's rigor with adversarial evaluation — and explicitly notes the cheap monitors variant has NOT been red-teamed yet (which Bailey et al. addressed).

---

### 9. The Attacker Moves Second (Nasr et al., 2025)

**Title:** "The Attacker Moves Second: Stronger Adaptive Attacks Bypass Defenses Against LLM Jailbreaks and Prompt Injections"
**Authors:** Milad Nasr, Nicholas Carlini, Chawin Sitawarin, Sander V. Schulhoff, Jamie Hayes, Florian Tramèr, et al. (Google DeepMind / ETH / others)
**arxiv:** 2510.09023
**Link:** https://arxiv.org/abs/2510.09023

**What they did:** They tested 12 published LLM jailbreak/prompt-injection defenses against adaptive attackers (where the attacker adapts strategy to bypass each specific defense). All 12 defenses fail. They argue that evaluation against static benchmarks gives a false sense of security. The paper has a quote: "None of the 12 defenses across four common techniques is robust to strong adaptive attacks."

**Why it matters to us:** Reinforces Bailey et al.'s finding. The adversarial robustness gap is field-wide and consistent across multiple research groups. We cite this alongside Bailey to establish that "no latent-space defense has been shown adversarially robust" is an established fact, not our personal opinion.

---

### 10. HSF: Defending against Jailbreak Attacks with Hidden State Filtering (Qian et al., WWW 2025)

**Title:** "Defending against Jailbreak Attacks with Hidden State Filtering"
**arxiv:** 2409.03788
**Link:** https://arxiv.org/abs/2409.03788

**What they did:** A jailbreak detector that uses a single layer's activations + concatenated last-k tokens to classify benign vs harmful. They report AUC ≥ 0.998 across multiple LLMs and attack types. They use the standard Alpaca harmless + AdvBench harmful setup.

**Why it matters to us:** Direct competitor we don't reproduce but cite. Same field-wide methodology issues we identify (length confound likely affects them too). They have similar reported numbers but didn't test diverse benign or adaptive attacks. Demonstrates that high reported numbers in the field are the norm, but might not survive better evaluation.

---

## TIER 3: USEFUL FOR DEPTH (8 papers)

If mentor has time after Tiers 1-2, these provide additional context.

### 11. GCG (Zou et al., 2023) — The attack we test most

**Title:** "Universal and Transferable Adversarial Attacks on Aligned Language Models"
**arxiv:** 2307.15043
**Link:** https://arxiv.org/abs/2307.15043

**What they did:** Introduced Greedy Coordinate Gradient (GCG), an optimization-based jailbreak attack that appends an adversarial 15-20 token suffix to harmful prompts. The suffix is optimized to maximize the likelihood of an affirmative response ("Sure, here is..."). Universal and transferable across LLMs.

**Why it matters to us:** GCG is one of the 9 attacks we test. Critical because the catastrophic Vicuna failure happens specifically on GCG (7.6% detection vs 99.4% on C4). GCG produces concentrated activation patterns due to gradient optimization, which is exactly what HPS's compression filters out — this is the mechanistic reason for the alignment-mediated failure.

### 12. PAIR (Chao et al., 2023)

**Title:** "Jailbreaking Black Box Large Language Models in Twenty Queries"
**arxiv:** 2310.08419
**Link:** https://arxiv.org/abs/2310.08419

**What they did:** Uses GPT-4 as an attacker LLM to iteratively rewrite harmful prompts until target LLM complies. Black-box, query-efficient, doesn't require gradient access.

**Why it matters to us:** One of our 9 attacks. Represents the "natural language" jailbreak category — different threat model from GCG's gradient-based suffixes.

### 13. AutoDAN (Liu et al., 2023)

**Title:** "AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models"
**arxiv:** 2310.04451
**Link:** https://arxiv.org/abs/2310.04451

**What they did:** Genetic algorithm for jailbreak prompt optimization. Uses a hierarchical mutation strategy (sentence and word level).

**Why it matters to us:** One of our 9 attacks. Represents "template-based" jailbreak category.

### 14. Bricken et al. "Features as Classifiers" (Anthropic 2024)

**Title:** "Features as Classifiers" (Transformer Circuits Thread)
**Link:** https://transformer-circuits.pub/2024/features-as-classifiers/index.html

**What they did:** Investigated using sparse autoencoder (SAE) features as classifiers for downstream tasks like harm detection. Found that SAE-based probes are difficult to beat with linear probes as a baseline.

**Why it matters to us:** Foundational evidence that "linear probing as a baseline is difficult to beat" — which is the central empirical observation our paper confirms. Cite alongside Anthropic Cheap Monitors.

### 15. Simple Probes Can Catch Sleeper Agents (MacDiarmid et al., Anthropic 2024)

**Title:** "Simple Probes Can Catch Sleeper Agents"
**Link:** https://www.anthropic.com/news/probes-catch-sleeper-agents

**What they did:** Showed that linear probes on activations can detect "sleeper agent" backdoored models that produce malicious behavior on triggers. Simple probes work surprisingly well in this setting.

**Why it matters to us:** More evidence of linear probe effectiveness in safety contexts. Provides cross-validation for the "linear probes work for activation-based detection" claim.

### 16. RTV / Refusal-Trajectory Vectors (Derya & Sunar, 2026)

**Title:** "Refusal-Trajectory Vectors for Activation-Based Jailbreak Detection"
**arxiv:** TBD (recent preprint)

**What they did:** Computes a "refusal direction" from paired harmless/harmful prompts at the last-token activation, then uses Mahalanobis distance for classification.

**Why it matters to us:** Another competitor we cite but don't reproduce. Same Alpaca-style benchmark issues. Listed in our comparison table.

### 17. GradSafe (Xie et al., ACL 2024)

**Title:** "GradSafe: Detecting Jailbreak Prompts for LLMs via Safety-Critical Gradient Analysis"
**arxiv:** 2402.13494
**Link:** https://arxiv.org/abs/2402.13494

**What they did:** Detection method using gradient patterns of safety-critical parameters. Different threat model than activation probes — looks at gradients during inference.

**Why it matters to us:** Another field-wide competitor. We don't reproduce but cite as comparison. Same Alpaca-style benchmark, same methodology questions.

### 18. RL-Obfuscation (Gupta & Jenner, 2025)

**Title:** "RL-Obfuscation: Can Language Models Learn to Evade Latent-Space Monitors?"
**arxiv:** 2506.14261
**Link:** https://arxiv.org/abs/2506.14261

**What they did:** Trained LLMs via RL to evade activation-based safety monitors. Models successfully learned to generate harmful content while keeping activations benign-looking. Direct evidence that probes can be evaded not just at inference time (Bailey et al.) but also via training (Gupta & Jenner).

**Why it matters to us:** Strengthens our adversarial limitation. Bailey et al. showed inference-time evasion; this paper shows training-time evasion. Together they make a complete case: latent-space defenses have a fundamental weakness, regardless of how the attacker accesses the model.

---

## TIER 4: BACKGROUND (Optional, for context)

Skip these unless mentor has specific interest.

### 19. Poincaré Embeddings for Hierarchical Data (Nickel & Kiela, NeurIPS 2017)
**arxiv:** 1705.08039 — Foundational hyperbolic representation learning paper

### 20. The Geometry of Truth (Marks & Tegmark, 2024)
**arxiv:** 2310.06824 — LLM activations encode true/false as linear directions

### 21. JailbreakBench (Chao et al., NeurIPS 2024 D&B Track)
**arxiv:** 2404.01318 — Standard benchmark + leaderboard for jailbreaks

### 22. Building Production-Ready Probes For Gemini (Google DeepMind, 2026)
**arxiv:** 2601.11516 — DeepMind's industry-deployed activation probe approach. Reports that probes fail under distribution shift.

### 23. Detecting High-Stakes Interactions with Activation Probes (Bailey et al., ICML 2025)
**arxiv:** 2506.10805 — Different Bailey et al. paper, on detecting "high-stakes" model interactions via probes.

### 24. Red-teaming Activation Probes using Prompted LLMs (Blandfort & Graham, 2025)
**arxiv:** 2511.00554 — Black-box LLM-based red-teaming of activation probes. Shows probes can be broken with prompted attackers, no gradient access needed.

### 25. The Attacker's Cookbook: TombRaider (EMNLP 2025), FlipAttack (ICML 2025)
**arxiv:** 2501.18628 (TombRaider), 2410.02832 (FlipAttack) — Recent novel attack methods

---

## What to Tell Your Mentor

**For 1-hour prep:** Read just Tier 1 (papers 1-5). That's enough to understand the meeting.

**For 3-hour prep:** Add Tier 2 (papers 6-10). You'll understand the broader field.

**For full context:** Add a few from Tier 3 based on interest (papers 11-18 cover specific topics).

## How These Papers Map to Our Findings

| Our finding | Most relevant papers |
|---|---|
| HPS = C4 = MTP statistically | #1 (Anthropic), #14 (Bricken) |
| Geometric hypothesis confirmed (0/13 inversions) | #4 (HypLoRA, HELM), #6 (Wollschläger), #7 (Arditi) |
| Vicuna catastrophic failure (alignment-mediated) | #3 (JBShield data), Anthropic alignment work |
| Length confound in standard benchmarks | #1 (Anthropic — they did it right), #3, #10, #17 (didn't) |
| max_length confound | Methodology — no direct precedent |
| Adversarial robustness limitation | #2 (Bailey), #9 (Nasr), #18 (Gupta & Jenner) |
| Methodology paper framing | #5 (Are GANs Created Equal) |

---

## Key Quotes for Mentor

To save mentor time, here are the key quotes from the most important papers:

**Anthropic Cheap Monitors (Cunningham et al. 2025):**
> "Linear probing as a baseline is difficult to beat... These methods could dramatically reduce the computational overhead of jailbreak detection, though further testing with adaptive adversarial attacks is needed."

**Bailey et al. (2025):**
> "Our attacks can often reduce recall from 100% to 0% while retaining a 90% jailbreaking rate... Obfuscated activations are not rare exceptions but rather appear to be widespread in the latent space... an attacker can always find new activations that bypass the monitor."

**Nasr et al. (2025):**
> "We argue that this evaluation process is flawed. None of the 12 defenses across four common techniques is robust to strong adaptive attacks."

**JBShield (Zhang et al. 2025):**
> "JBShield-D achieves an average detection accuracy of 0.95 and an average F1-Score of 0.94" (no length analysis in their paper)

**HypLoRA (Yang et al. 2025):**
> "LLM token embeddings exhibit empirical δ-hyperbolicity, indicating that hyperbolic geometry is the natural space for LLM representations."

These quotes anchor each part of our paper's narrative.

---

## Sources

- ⚠️ External link — [Anthropic Cheap Monitors](https://alignment.anthropic.com/2025/cheap-monitors/) — accessed 2026-05-29
- ⚠️ External link — [Bailey et al. ICLR 2025 (arxiv 2412.09565)](https://arxiv.org/abs/2412.09565) — accessed 2026-05-29
- ⚠️ External link — [JBShield USENIX 2025 (arxiv 2502.07557)](https://arxiv.org/abs/2502.07557) — accessed 2026-05-29
- ⚠️ External link — [HypLoRA NeurIPS 2025 (arxiv 2405.18515)](https://arxiv.org/abs/2405.18515) — accessed 2026-05-29
- ⚠️ External link — [HELM NeurIPS 2025 (arxiv 2505.24722)](https://arxiv.org/abs/2505.24722) — accessed 2026-05-29
- ⚠️ External link — [Lucic et al. NeurIPS 2018 (arxiv 1711.10337)](https://arxiv.org/abs/1711.10337) — accessed 2026-05-29
- ⚠️ External link — [Wollschläger ICML 2025 (arxiv 2502.17420)](https://arxiv.org/abs/2502.17420) — accessed 2026-05-29
- ⚠️ External link — [Arditi NeurIPS 2024 (arxiv 2406.11717)](https://arxiv.org/abs/2406.11717) — accessed 2026-05-29
- ⚠️ External link — [Constitutional Classifiers (arxiv 2501.18837)](https://arxiv.org/abs/2501.18837) — accessed 2026-05-29
- ⚠️ External link — [Nasr et al. (arxiv 2510.09023)](https://arxiv.org/abs/2510.09023) — accessed 2026-05-29
- ⚠️ External link — [HSF (arxiv 2409.03788)](https://arxiv.org/abs/2409.03788) — accessed 2026-05-29
- ⚠️ External link — [GCG (arxiv 2307.15043)](https://arxiv.org/abs/2307.15043) — accessed 2026-05-29
- ⚠️ External link — [PAIR (arxiv 2310.08419)](https://arxiv.org/abs/2310.08419) — accessed 2026-05-29
- ⚠️ External link — [AutoDAN (arxiv 2310.04451)](https://arxiv.org/abs/2310.04451) — accessed 2026-05-29
- ⚠️ External link — [Bricken Features as Classifiers](https://transformer-circuits.pub/2024/features-as-classifiers/index.html) — accessed 2026-05-29
- ⚠️ External link — [Probes Catch Sleeper Agents](https://www.anthropic.com/news/probes-catch-sleeper-agents) — accessed 2026-05-29
- ⚠️ External link — [GradSafe ACL 2024 (arxiv 2402.13494)](https://arxiv.org/abs/2402.13494) — accessed 2026-05-29
- ⚠️ External link — [RL-Obfuscation (arxiv 2506.14261)](https://arxiv.org/abs/2506.14261) — accessed 2026-05-29
- ⚠️ External link — [Poincaré Embeddings NeurIPS 2017 (arxiv 1705.08039)](https://arxiv.org/abs/1705.08039) — accessed 2026-05-29
- ⚠️ External link — [Geometry of Truth (arxiv 2310.06824)](https://arxiv.org/abs/2310.06824) — accessed 2026-05-29
- ⚠️ External link — [JailbreakBench (arxiv 2404.01318)](https://arxiv.org/abs/2404.01318) — accessed 2026-05-29
- ⚠️ External link — [DeepMind Gemini Probes (arxiv 2601.11516)](https://arxiv.org/abs/2601.11516) — accessed 2026-05-29
- ⚠️ External link — [Bailey ICML 2025 (arxiv 2506.10805)](https://arxiv.org/abs/2506.10805) — accessed 2026-05-29
- ⚠️ External link — [Red-teaming Activation Probes (arxiv 2511.00554)](https://arxiv.org/abs/2511.00554) — accessed 2026-05-29
- ⚠️ External link — [TombRaider EMNLP 2025 (arxiv 2501.18628)](https://arxiv.org/abs/2501.18628) — accessed 2026-05-29
- ⚠️ External link — [FlipAttack ICML 2025 (arxiv 2410.02832)](https://arxiv.org/abs/2410.02832) — accessed 2026-05-29
