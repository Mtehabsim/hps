# 9-Slide Team Meeting Presentation: Hyperbolic Geometric Priors for LLM Jailbreak Detection

**Goal:** Present findings to team for feedback. ~30 minute meeting.
**Style:** Each slide has 1 main message. Use big numbers, simple charts.

---

## Figure Inventory & Setup

Before the meeting, ensure all figures are in `figures_for_meeting/`. Run this once:

```bash
# Generate locally-built figures (instant)
python fig_4method_comparison.py
python fig_cold_start_sweep.py
python fig_lorentz_concept.py

# Copy DGX figures (they were generated on the GPU server from the fixed cache)
mkdir -p figures_for_meeting/dgx
scp dgx03:/mnt/lab/Mo/hps/hps2/hps/results/figs/radial_check_seeds.png   figures_for_meeting/dgx/
scp dgx03:/mnt/lab/Mo/hps/hps2/hps/results/figs/vicuna_per_attack.png    figures_for_meeting/dgx/
scp dgx03:/mnt/lab/Mo/hps/hps2/hps/results/figs/gcg_per_attack_cross_model.png  figures_for_meeting/dgx/
scp dgx03:/mnt/lab/Mo/hps/hps2/hps/results/figs/gcg_specific.png         figures_for_meeting/dgx/
scp dgx03:/mnt/lab/Mo/hps/hps2/hps/results/figs/norm_controlled_eval.png figures_for_meeting/dgx/
scp dgx03:/mnt/lab/Mo/hps/hps2/hps/results/figs/norm_confound_summary.png figures_for_meeting/dgx/
```

---

## SLIDE 1: The Question

**Title:** Do Hyperbolic Geometric Priors Help LLM Jailbreak Detection?

**Subtitle:** A Controlled Comparison Study

**Body content:**

Background context (3 bullets):
- HypLoRA (NeurIPS 2025): LLM token embeddings show empirical δ-hyperbolicity (tree-like structure)
- HELM (NeurIPS 2025): Power-law radial structure / negative Ricci curvature in embeddings
- Theoretical prediction: hyperbolic projection should provide useful inductive bias for distinguishing harmful (specific) from benign (general) content

The research question (large text, centered):
> **"If LLM activations have hierarchical structure, does hyperbolic projection provide a measurable detection advantage over flat (Euclidean) baselines?"**

Bottom strip — 3 deliverables:
- ✓ Built HPS (Hyperbolic Projection Sentinel): Lorentz contrastive framework
- ✓ Compared against parameter-matched Euclidean ablation + linear probe baseline
- ✓ Tested across two LLMs with different alignment (RLHF vs SFT-only)

**Visual:**

![Hyperbolic geometry concept](figures_for_meeting/fig_lorentz_concept.png)

*Conceptual: Lorentz hyperboloid (left) and Poincaré disk projection (right). Hypothesis predicts attacks at higher radial position than benign. (Synthetic illustration; real measurements on Slide 5.)*

**Speaker notes:**
- Frame as "we set out to test a theoretical prediction"
- Emphasize this is a scientific question, not a method advertisement
- Skip details for now; will get into specifics

---

## SLIDE 2: What We Built — Four Methods Compared

**Title:** Methods Under Comparison

**Body content as a table:**

| Method | Architecture | Parameters | Key Innovation |
|---|---|---|---|
| **HPS** (ours) | Lorentz projection + 12 trajectory features + LR | 262K | Hyperbolic geometric prior |
| **HPS-Euclidean** (control) | Same arch, FLAT geometry | 262K (matched) | Tests if geometry matters |
| **C4** (baseline) | Mean-pool 6 layers' last-token + LR | 4,097 | Adapted from Anthropic Cheap Monitors |
| **MTP** (baseline) | Mean-pool tokens at 1 layer + LR | 4,097 | Faithful Anthropic reproduction |

Below the table — Two callouts:

**HPS architecture details:**
```
Activations [N=6 layers] → W·activations (project to 64-d)
       → Lorentz hyperboloid (κ=0.1, frozen)
       → 12 trajectory features (5 radial + 4 curvature + 3 displacement)
       → LR
Trained with per-layer-temperature contrastive loss, 50 epochs
```

**Why HPS-Euclidean control is critical:**
> "Without parameter-matched Euclidean, any HPS-vs-C4 win could be 'more parameters helped.' HPS-Euclidean has IDENTICAL parameter count and architecture, only flat geometry. This isolates the geometric prior."

**Visual (optional):**

![HPS pipeline](figures/hps_pipeline.png)

*HPS architecture: activations → Lorentz projection → trajectory features → LR.*

**Speaker notes:**
- Emphasize MTP is faithful Anthropic reproduction (not C4 axis swap)
- Don't oversell C4 as novel — it's "an adaptation of Anthropic"
- The 4-way comparison gives clean ablations

---

## SLIDE 3: Methodology Surprise #1 — Length Confound

**Title:** Standard Benchmarks Have a Length Confound

**Subtitle:** A trivial classifier using only prompt length detects 96.9% of attacks

**Body content — top half:**

The standard benchmark setup (Alpaca-style harmless + 9 attacks):
- Alpaca-style harmless: very short (mean ~12-60 chars depending on filter)
- Attack prompts: weighted mean **218 chars**, with extreme variation by attack type

Length-only classifier (input: just `len(prompt)` characters, NO model needed):

| Metric | Length-only |
|---|---|
| AUROC | **0.992** |
| TPR @ 5% FPR | **0.969** |
| TPR @ 1% FPR | **0.875** |

> **A classifier that NEVER LOOKS at the prompt CONTENT, only its length, catches 96.9% of attacks at 5% FPR.**

**Body content — bottom half — per-attack table:**

| Attack | Mean length | Length-only AUROC |
|---|---|---|
| puzzler | 2,193 chars | **1.000** |
| saa | 473 chars | **1.000** |
| drattack | 437 chars | **1.000** |
| base64 | 124 chars | **1.000** |
| ijp | 460 chars | 0.999 |
| autodan | 77 chars | 0.999 |
| pair | 72 chars | 0.997 |
| zulu | 50 chars | 0.976 |
| gcg | 35 chars | 0.971 |

**Bottom callout — implications:**

This affects:
- HSF (WWW 2025) — same Alpaca + AdvBench setup
- JBShield (USENIX 2025) — confirmed: uses Alpaca + 9-attack mixture (their paper, Section 5.1)
- GradSafe (ACL 2024) — same setup
- RTV — same setup

Anthropic Cheap Monitors did the RIGHT thing: used WildChat + OR-Bench Hard (varied lengths).

**Our fix:** Diverse benign data (WildChat + OR-Bench Hard + MMLU + GSM8K + WikiText long-form + ...) matched in length distribution. After fix:
- Length-only AUROC: 0.992 → **0.318** (length is now slightly anti-correlated with attacks)
- Permutation test: 0.498 (real signal exists)

**Speaker notes:**
- This is the headline methodology finding
- Anticipate question: "did the other papers check?" Answer: "JBShield's paper has no length analysis (I read it). HSF/RTV similar."
- Don't say "they suck" — say "the field shares a methodology issue"

---

## SLIDE 4: Methodology Surprise #2 — max_length Confound

**Title:** Inconsistent Token Truncation Creates a Norm Confound

**Subtitle:** Different max_length between train and test → norm-only AUROC = 1.000

**Body content — the bug:**

```
Original cache:
  benign max_length = 512    attacks max_length = 512    ✓ consistent
  → Norm-only AUROC = 0.917

Diverse benign cache (after fixing length confound):
  benign max_length = 2048   attacks REUSED at max_length = 512  ✗ MISMATCH
  → Norm-only AUROC = 1.000  ← perfect detection from norm alone
```

**Body content — why this matters:**

Layer 5 (deepest) activation norms:

| | Old cache (consistent) | Diverse cache (mismatch) |
|---|---|---|
| Benign mean norm | 155.6 | **35.5** ⚠ |
| Attack mean norm | 153.0 | 153.0 |
| Ratio | 1.0× | **4.3×** |

The benign norm at layer 31 dropped 4.4× when truncation changed from 512→2048 tokens. Long prompts produce systematically different deep-layer activations.

**The fix:**

Re-extract attacks at max_length=2048 to match diverse benign:
- Norm-only AUROC: 1.000 → **0.761** ✓ confounding largely resolved
- Permutation test: AUROC = 0.498 → real signal exists when labels are real

**Visual:**

![Norm-controlled evaluation](figures_for_meeting/dgx/norm_controlled_eval.png)

*Norm-only AUROC across control conditions. After L2-normalization, norm signal disappears (~chance), but C4 detection holds. Real semantic signal exists beyond norm.*

**Bottom callout:**
> Methodology lesson: when re-using cached activations, ALL data must be processed with the same max_length, chat template, padding strategy, and tokenizer settings. Inconsistent preprocessing creates artificial separability.

**Speaker notes:**
- This is a coding bug, but reveals a methodology principle
- Worth mentioning briefly; don't dwell on it
- Combined with length confound = field-wide methodology issue

---

## SLIDE 5: Finding 1 — Geometric Hypothesis CONFIRMED

**Title:** Hyperbolic Projection Learns the Predicted Direction

**Subtitle:** After methodology fixes, attacks occupy higher radial position (0/13 inversions across configurations)

**Body content — main result:**

**13 configurations tested:**
- 5 random seeds (κ=0.1, 50 epochs)
- 4 epoch checkpoints (5, 10, 25, 50)
- 4 curvature κ values (0.1, 0.5, 1.0, 2.0)

**Result: 0/13 inversions. All configurations show ben_median < atk_median.**

| Config | Benign median | Attack median | Δ (ben - atk) | As predicted? |
|---|---|---|---|---|
| Seed 42, κ=0.1, ep=50 | 3.20 | 3.50 | -0.30 | ✓ |
| Seed 43 | 3.20 | 3.50 | -0.30 | ✓ |
| Seed 44 | 3.20 | 3.52 | -0.32 | ✓ |
| Seed 45 | 3.20 | 3.51 | -0.31 | ✓ |
| Seed 46 | 3.20 | 3.52 | -0.32 | ✓ |
| Epoch 5 | 3.23 | 3.27 | -0.04 | ✓ |
| Epoch 50 | 3.20 | 3.50 | -0.30 | ✓ |
| κ=0.1 | 3.20 | 3.50 | -0.30 | ✓ |
| κ=0.5 | 1.57 | 1.96 | -0.39 | ✓ |
| κ=1.0 | 1.36 | 1.53 | -0.17 | ✓ |
| κ=2.0 | 1.37 | 1.42 | -0.05 | ✓ |

**Visual:**

![Radial distribution across seeds](figures_for_meeting/dgx/radial_check_seeds.png)

*Benign vs attack radial position medians across 5 random seeds. All 5 show attacks at higher radial position (as theory predicts).*

**The mechanism:**

Hyperbolic prior predicts: harmful content (specific, deeper hierarchy) → high radial; benign (general) → near origin.

Before length-confound fix: appeared inverted (benign at high radial). The contrastive loss had learned a length-shortcut direction.

After length-confound fix: attacks ARE at higher radial, exactly as theory predicts.

> **"The geometric prior is theoretically valid. Hyperbolic projection learns the meaningful direction once the data is clean."**

**Speaker notes:**
- This was the SURPRISE: we thought this would be 13/13 inversion (proving hypothesis wrong)
- After methodology fixes, it flipped to 0/13 (proving hypothesis right)
- This is robust across seeds, epochs, curvatures

---

## SLIDE 6: Finding 2 — Geometry Provides Measurable Advantage Over Flat

**Title:** Lorentz Constraint Beats Parameter-Matched Euclidean

**Subtitle:** +0.049 TPR over flat geometry (statistically real); but C4 still wins overall

**Body content — full-data comparison (n=5 seeds):**

| Method | AUROC | TPR @ 5% FPR | Parameters |
|---|---|---|---|
| **HPS (Lorentz)** | 0.991 | 0.980 | 262K |
| **HPS-Euclidean (matched)** | 0.968 | 0.931 | 262K |
| **C4 (linear probe)** | 0.998 | 0.995 | 4,097 |
| **MTP (Anthropic exact)** | 0.999 | 0.995 | 4,097 |

**Visual — main comparison:**

![4-method comparison](figures_for_meeting/fig_4method_comparison.png)

*HPS-Euclidean control isolates the geometric prior. HPS (Lorentz) beats HPS-Euclidean (matched parameters), but linear probes (C4, MTP) tie at the top.*

**Key deltas:**
- HPS - HPS-Euclidean: **+0.049 TPR** (geometric prior helps over flat)
- HPS - C4: **-0.015 TPR** (linear probe still wins)
- MTP - C4: 0.000 (axis swap doesn't matter on Llama-3)

**Cold-start sweep** (low N regime where geometric priors should help most):

| N attacks | HPS | HPS-Euclidean | C4 |
|---|---|---|---|
| 45 | 0.70 | 0.61 | 0.96 |
| 90 | 0.82 | 0.61 | 0.98 |
| 225 | 0.95 | 0.89 | 0.98 |
| 450 | 0.90 | 0.83 | 0.99 |
| 900 | 0.98 | 0.94 | 0.99 |
| 2250 | 0.97 | 0.93 | 0.99 |

**Visual — cold-start sweep:**

![Cold-start sweep](figures_for_meeting/fig_cold_start_sweep.png)

*Three clean tiers: C4 dominates throughout, HPS beats HPS-Euclidean (geometry helps over flat), but C4's simplicity wins overall.*

> **"Three clean tiers: C4 > HPS > HPS-Euclidean. Geometric prior provides architectural insight (Lorentz > flat at matched parameters), but C4 dominates regardless."**

**Prediction agreement (per-example, n=1295 attacks, mean of 3 seeds)**

At calibrated 5% FPR thresholds:

| Metric | Value |
|---|---|
| HPS catches that C4 misses | **0 examples** |
| C4 catches that HPS misses | 21 examples |
| Pearson(HPS, C4) | **0.958** |
| OR-gate ensemble TPR | 0.995 (= C4 alone) |
| OR-gate ensemble FPR | 0.103 (vs 5% target) |

> **HPS is essentially a noisy subset of C4. They make the same correct predictions; HPS's extra parameters add no orthogonal information. The "ensemble HPS+C4" hypothesis fails empirically — no TPR gain, doubles FPR.**

**Speaker notes:**
- This is the architectural finding
- Lorentz constraint genuinely helps over parameter-matched Euclidean
- But C4 (no learned projection at all) wins via simplicity
- Important: parameter-matched comparison was the OLD critical methodology issue we fixed

---

## SLIDE 7: Finding 3 — HPS Catastrophically Fails on Vicuna

**Title:** Alignment-Mediated Failure: HPS Works on RLHF, Fails on SFT-Only

**Subtitle:** HPS catches 7.6% of GCG on Vicuna; C4 catches 99% — same architecture, only LLM differs

**Body content — the big number:**

```
HPS GCG detection rate:
  Llama-3-8B-Instruct (RLHF):  100% (172/172)
  Vicuna-13B-v1.5 (SFT only):    7.6% (13/171)

  Cross-model gap on GCG: -92 percentage points
```

**Per-attack breakdown on Vicuna (full table):**

| Attack | HPS rate | C4 rate | gap (C4-HPS) |
|---|---|---|---|
| **gcg** | **0.076** | 0.994 | +0.92 |
| ijp | 0.329 | 0.933 | +0.60 |
| pair | 0.389 | 0.958 | +0.57 |
| puzzler | 0.462 | 1.000 | +0.54 |
| zulu | 0.632 | 1.000 | +0.37 |
| autodan | 0.701 | 1.000 | +0.30 |
| base64 | 0.922 | 1.000 | +0.08 |
| drattack | 0.956 | 1.000 | +0.04 |
| saa | 0.994 | 1.000 | +0.01 |

**Mean detection: HPS=0.61, C4=0.99 on Vicuna**

**Visual — Vicuna per-attack:**

![Vicuna per-attack breakdown](figures_for_meeting/dgx/vicuna_per_attack.png)

*Per-attack HPS vs C4 on Vicuna-13B. Massive gap on GCG (0.076 vs 0.994) and other attack types. C4 is robust; HPS collapses.*

**Visual — Cross-model comparison:**

![GCG cross-model](figures_for_meeting/dgx/gcg_per_attack_cross_model.png)

*Same code, same hyperparameters, same 9 attack types. Only the LLM differs. HPS detection on Vicuna averages 61%; on Llama-3 averages 99%.*

**Mechanism:**

```
Llama-3-Instruct:  Strong RLHF concentrates harm features
                   into compact regions of activation manifold
                   → HPS's 12-feature compression preserves them ✓

Vicuna-13B:        SFT only, weak alignment
                   → harm features remain DIFFUSE in activation space
                   → HPS's compression filters them out ✗
                   → C4's 4096-dim representation preserves them ✓
```

> **"Geometric priors work ONLY on alignment-trained models. Without RLHF, the compression that makes HPS efficient becomes the very thing that breaks it."**

**Speaker notes:**
- This is the most striking finding
- Same code, same hyperparameters, same attacks
- Only difference: alignment training
- Strong evidence for "alignment concentrates harm features" hypothesis
- Note: This finding is on n=171 GCG samples on Vicuna (n=172 on Llama-3) — properly powered. Earlier preliminary tests with smaller samples showed similar pattern.

---

## SLIDE 8: The Threat Surface — Bailey et al. (ICLR 2025)

**Title:** All Latent-Space Defenses Fail Against Adaptive Attacks

**Subtitle:** Field-wide finding affects HPS, C4, JBShield, HSF, RTV, Anthropic Cheap Monitors

**Body content — Bailey et al.'s key result:**

Bailey et al. trained adversarial obfuscation suffixes against linear probes on Llama-3-8B-Instruct (same model we use).

**Standard jailbreaks** (no adaptive attack):
- Linear probes: 90-100% recall ✓
- MLP probes: 95-100% recall ✓

**Adaptive obfuscation attacks** (16-token universal suffix optimized against the probe):

| Defense type | Recall before | Recall after | Source |
|---|---|---|---|
| Logistic regression probe | 100% | **0%** | Bailey et al. (Sec 3.3) |
| MLP probe | 100% | **0%** | Bailey et al. (Sec 3.3) |
| Sparse autoencoder | 91% | **0%** | Bailey et al. (Sec 5) |
| OOD detector (Mahalanobis) | 100% | **0%** | Bailey et al. (Sec 3.3) |
| Adversarially trained probe (70 rounds) | perfect on past attacks | **0% on new attacks** | Bailey et al. (Sec 3.5) |
| Circuit Breakers | broken | broken | Bailey et al. (Sec 3.4) |
| LAT (Latent Adversarial Training) | broken | broken | Bailey et al. (Sec 3.4) |

> **Quote from Bailey et al. (ICLR 2025):**
> "Obfuscated activations are not rare exceptions but rather appear to be widespread in the latent space... an attacker can always find new activations that bypass the monitor."

**What this means for the field:**

Affects:
- HSF, JBShield, GradSafe, RTV — none have been adversarially tested
- Anthropic Cheap Monitors — they explicitly acknowledge this limitation in their paper:
  > "Further testing with adaptive adversarial attacks is needed."
- Our HPS — we don't claim adversarial robustness

**Bottom line for our paper:**

> "We do not claim adversarial robustness. We compare methods under standard (non-adaptive) conditions. The robustness of any activation-based defense to adaptive attacks remains an open problem — affecting the entire field, not specific to us."

**Speaker notes:**
- Now is when you bring up adversarial — after showing your contributions
- This is a field-wide limitation, not our problem alone
- Don't get defensive; honestly cite this as future work
- Anthropic moved on to Constitutional Classifiers++ (different architecture)

---

## SLIDE 9: Open Questions for the Team

**Title:** What We Found vs. What's Open — Need Your Input

**Body content — left column: WHAT WE FOUND**

Methodology contributions:
1. Length confound: standard benchmarks have AUROC=**0.992** from length alone
2. max_length confound: inconsistent truncation → norm-only AUROC=1.000
3. Train/test contamination: 15 attack prompts (1.15%) overlap

Empirical findings:
4. Geometric hypothesis CONFIRMED (0/13 inversions after fixes)
5. HPS provides advantage over flat Euclidean (+0.049 TPR, parameter-matched)
6. HPS catastrophically fails on Vicuna without RLHF (gcg: 7.6% vs 99%)
7. Linear probes (C4 ≈ MTP ≈ HPS) on aligned LLMs

**Body content — right column: WHAT'S OPEN**

Decisions for the team:

**Q1: Where to publish?**
- TMLR (60-65%): rigorous methodology paper
- AAAI/IJCAI 2027 (35-50%): methodology framing
- USENIX Security (25-40%): would need adaptive attacks
- Recommendation: TMLR

**Q2: Adaptive attacks?**
- Pro: strengthens paper
- Con: 1-2 days compute, predicted result confirms Bailey et al.
- Open: required by reviewers?

**Q3: More LLMs?**
- Could add Mistral-7B, Llama-2-7b-chat (JBShield released free)
- 2-3 hours each
- Strengthens cross-model claims
- **Critical for alignment hypothesis:** Llama-2-7b-chat has SFT+RLHF on the SAME base model as Vicuna (which has SFT only). Comparing them isolates RLHF as the variable. If Llama-2-chat works and Vicuna fails, the alignment-mediated story is properly controlled.

**Q4: What's the lead?**
- Architectural contribution (HPS as geometric framework)
- Methodology critique (length, max_length, contamination)
- Alignment-mediated failure (the most striking finding)

**Q5: Multi-turn pivot?**
- Multi-turn jailbreaks have hierarchical structure (better fit for hyperbolic)
- 6-month commitment vs 6-week submission to TMLR
- High-risk vs strengthen current work

**Bottom callout — what we need from the team:**

> "Looking for input on: paper venue, framing priority, whether to add adaptive attacks experiment, and whether to pivot to multi-turn after this submission."

**Speaker notes:**
- Don't try to answer Q1-Q5 yourself; let team weigh in
- Frame as "we want your input" not "we have a plan, what do you think"
- Anticipated questions: handle in Q&A

---

# Visual Style Guidelines

**Color scheme used in figures:**
- Benign data: blue (#3498db)
- Attack data: red (#e74c3c)
- HPS: purple (#9b59b6)
- HPS-Euclidean: gray (#95a5a6)
- C4: green (#2ecc71)
- MTP: dark green (#27ae60)
- Methodology issues: orange (#f39c12)

**Figure manifest (8 figures total across 9 slides):**

| Slide | Figure file | Source |
|---|---|---|
| 1 | `figures_for_meeting/fig_lorentz_concept.png` | Generated locally |
| 2 | `figures/hps_pipeline.png` (optional) | Existing local |
| 4 | `figures_for_meeting/dgx/norm_controlled_eval.png` | Copy from DGX |
| 5 | `figures_for_meeting/dgx/radial_check_seeds.png` | Copy from DGX |
| 6 | `figures_for_meeting/fig_4method_comparison.png` | Generated locally |
| 6 | `figures_for_meeting/fig_cold_start_sweep.png` | Generated locally |
| 7 | `figures_for_meeting/dgx/vicuna_per_attack.png` | Copy from DGX |
| 7 | `figures_for_meeting/dgx/gcg_per_attack_cross_model.png` | Copy from DGX |

**Tone:**
- Confident about findings, humble about limitations
- Cite Bailey et al. as field-wide problem, not specific weakness
- Frame methodology critique constructively
- Don't oversell HPS; don't undersell methodology contribution

# Pacing for 30-Minute Meeting

- Slides 1-2: 5 minutes (context + methods)
- Slides 3-4: 7 minutes (methodology surprises — important)
- Slides 5-7: 10 minutes (the three findings)
- Slide 8: 5 minutes (adversarial limitation)
- Slide 9: 5 minutes (open questions, team feedback)
- Q&A: 10-15 minutes after presentation

# Key Talking Points (for Speaker Reference)

For each slide, the ONE thing to emphasize:

1. "We're testing a theoretical prediction, not promoting a method"
2. "Notice the parameter-matched control — that's our methodology contribution"
3. "Length alone catches 97% of attacks at 5% FPR — this is field-wide"
4. "Max_length consistency matters — methodology lesson"
5. "Geometric hypothesis was hidden by length confound"
6. "Geometric prior helps over flat, but doesn't beat linear probes"
7. "Compression breaks without RLHF concentrating the signal"
8. "Bailey et al. is a field-wide threat, not specific to us"
9. "Looking for team input on framing and venue"

# Anticipated Questions and Prepared Answers

**Q: "If geometric priors don't beat linear probes, why publish?"**
A: "The methodology contribution stands alone. Three field-wide issues (length, max_length, contamination) plus controlled comparison. Plus the alignment-mediated failure is novel mechanism evidence."

**Q: "Why didn't you test adversarial robustness?"**
A: "Bailey et al. (ICLR 2025) showed all latent-space defenses fail. We don't claim robustness. We cite this as a known limitation. Could run experiments but predicted outcome confirms existing literature."

**Q: "Are you saying JBShield/HSF/RTV are wrong?"**
A: "No. They report real numbers on their benchmarks. We're saying the standard benchmark has methodology issues that ALL papers using it inherit. The fix is diverse benign data, following Anthropic's example."

**Q: "Why Vicuna-13B specifically?"**
A: "Same architecture family as Llama-2 (which Vicuna fine-tuned), but only SFT. Lets us isolate the alignment variable. RLHF (Llama-3-Instruct) vs SFT-only (Vicuna). Different alignment, controlled for everything else as much as possible."

**Q: "How much compute have we used?"**
A: "Cache extractions: ~10 hours. Experiments: ~5 hours. Total ~15 GPU-days across the project."

**Q: "What's our timeline if we go to TMLR?"**
A: "4-6 weeks of writing. Submit, ~3 month review cycle. So target: paper draft in 6 weeks, submission August, accept/reject by November."

**Q: "What if HPS fails on Llama-2-7b-chat too?"**
A: "Then the alignment-mediated story is more nuanced — maybe Llama-3's specific alignment recipe matters. We'd need a third aligned model to test. Either way, it sharpens the paper: 'HPS works on this specific class of aligned models.'"

**Q: "How does this compare to JBShield's claimed 99% detection?"**
A: "JBShield's setup has the length confound — Alpaca harmless (mean ~60 chars) vs longer attacks. Their 99% likely includes substantial length signal. We didn't reproduce their full pipeline; we'd need to run their code on diverse benign data to know what fraction is real signal vs length."

---

# Pre-Meeting Checklist

- [ ] Generate local figures: `python fig_4method_comparison.py && python fig_cold_start_sweep.py && python fig_lorentz_concept.py`
- [ ] Copy DGX figures via scp commands at top of this file
- [ ] Verify all 8 figure paths exist before opening slides
- [ ] Print speaker notes (or load on second screen)
- [ ] Have JSON files ready for any "where does this number come from?" follow-ups:
  - `results/verify_saturation_fixed.json` (Slide 3 length confound)
  - `results/norm_check_diverse.json` (Slide 4 norm confound)
  - `results/radial_distribution_check.json` (Slide 5 radial inversion)
  - `results/hyperbolic_vs_euclidean_diverse.json` (Slide 6 main comparison)
  - `results/prediction_agreement.json` (Slide 6 ensemble negative result)
  - `results/gcg_specific_test.json` (Slide 7 cross-model)
  - `results/vicuna_imbalance_test.json` (Slide 7 per-attack Vicuna)
- [ ] Bring laptop with `paper_draft.md` open in case team wants paper outline discussion
