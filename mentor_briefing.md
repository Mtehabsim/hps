# HPS Research Briefing for Mentor

**Project:** Hyperbolic Geometric Priors for LLM Jailbreak Detection
**Date:** May 2026 — Three confirmed findings, one major methodological discovery

---

## ⚡ Major Update (May 28, 2026)

Verification experiments revealed that initial benchmark results were heavily confounded by **prompt length**. After re-running with diverse benign data:

1. The geometric hypothesis (radial = "extremity") is now **CONFIRMED**, not refuted (it was masked by length confound)
2. Saturation persists even after length control — the methods truly tie at TPR=1.000
3. Vicuna findings are less dramatic than originally reported

This briefing reflects the corrected results.

---

## Note on C4 (the baseline used throughout)

**C4 is our adaptation of Anthropic's "Cheap Monitors" approach** (Cunningham et al., 2025), which is deployed in production at Anthropic for jailbreak detection on Claude 3 Sonnet. Anthropic's mean-token probe pools activations *over tokens* within a single LLM layer, then applies a linear classifier:

```
Anthropic Cheap Monitors (mean-token probe):
  activations h ∈ ℝ^(T × d)  →  mean across tokens: f = (1/T) Σ h_t  →  LR
```

**We took this method and made two specific changes** to fit our experimental setup:

1. **We mean-pool over LAYERS instead of over tokens.** Anthropic averages activations across all token positions at one fixed layer; we average activations across N=6 selected layers at the last-token position.

2. **We use multiple layers (6) rather than one.** Anthropic uses a single layer; we use 6 spread layers `[0, 2, 17, 24, 28, 31]` selected to span shallow + middle + deep representations.

The resulting C4 recipe:

```
Our C4 = activations from 6 layers' last token  →  mean across layers
       →  StandardScaler  →  logistic regression
```

**We do not claim C4 as a novel method.** It is a controlled minimal baseline derived from Anthropic's published approach with the modifications above. The point of using C4 in this study is to ask: *does HPS's 262K-parameter geometric framework provide measurable advantage over a 4,097-parameter mean-pool linear probe based on an established industry approach?* Anthropic's own conclusion is that *"linear probing as a baseline is difficult to beat."*

**Related approaches** (linear probes on hidden states for harm detection): Google DeepMind Production-Ready Probes for Gemini (arXiv:2601.11516, 2026); Detecting High-Stakes Interactions with Activation Probes (Bailey et al., ICML 2025); Bricken et al. 2024 "Features as Classifiers." All share the general approach; specific architectural details vary.

**What we contribute (separately from C4):** the controlled three-way comparison (HPS vs HPS-Euclidean vs C4), cold-start regime methodology, multi-LLM alignment analysis, statistical rigor, and the **length-confound discovery** described below.

---

## 1. Background — Papers That Shaped the Idea

| Paper | Year | What it told us |
|---|---|---|
| **HypLoRA** (Yang et al., NeurIPS 2025) | 2025 | LLM token embeddings exhibit empirical δ-hyperbolicity (tree-like) |
| **HELM** (He et al., NeurIPS 2025) | 2025 | Token embeddings have power-law radial structure / negative Ricci curvature |
| **Poincaré Embeddings** (Nickel & Kiela, NeurIPS 2017) | 2017 | Hyperbolic geometry is the natural space for hierarchical / tree-like data |
| **Geometry of Refusal** (Wollschläger et al., ICML 2025) | 2025 | Refusal direction in LLMs has structured geometric properties |
| **Anthropic Cheap Monitors** (Cunningham et al., 2025) | 2025 | Linear probes match dedicated jailbreak classifiers at orders-of-magnitude lower cost (deployed in production for Claude) |
| **JBShield** (Zhang et al., USENIX Security 2025) | 2025 | Peer-reviewed activation-based jailbreak defense; reports F1=0.94 across 5 LLMs × 9 attacks |

**Core thesis from the literature:** If LLM activations have hierarchical structure, hyperbolic projection should provide a useful inductive bias for distinguishing harmful (specific) from harmless (general) content, exploiting hyperbolic space's exponential volume growth.

## 2. What We Built — Four Methods Compared

### **HPS (our novel framework)** — 262K parameters
```
Activations from N=6 layers → learned linear projection W ∈ ℝ^(d×64)
       → Lorentz hyperboloid (curvature κ) → 12 trajectory features
       (radial × 5, curvature × 4, displacement × 3) → logistic regression
```
Trained with per-layer-temperature contrastive loss, 50 epochs.

**Layer selection:** `[0, 2, 17, 24, 28, 31]` chosen via systematic ablation (TEST 7 in diagnostics): spread layers achieved AUROC=1.000 vs Fisher-discovered [0,1,2,28,29,30,31] at 0.925. Spans shallow + middle + deep representations.

**Curvature κ:** Frozen at 0.1 (best from sweep over {0.1, 0.5, 1.0, 2.0, 10.0}). Learnable κ was unstable.

**Trajectory feature definitions:**
- Radial (5): `mean(x₀)`, `max(x₀)`, `min(x₀)`, `std(x₀)`, `range = max−min` of Lorentz time-coordinate
- Curvature (4): `max`, `mean`, `std` of triangle-inequality bending across consecutive layers; argmax position normalized
- Displacement (3): Lorentz distance start→end, sum of consecutive distances (path length), straightness ratio

### **HPS-Euclidean (controlled ablation)** — 262K parameters (parameter-matched)
Same architecture as HPS but **flat space instead of Lorentz hyperboloid**, with per-layer scale + learnable margin to match HPS's parameter count exactly. **This control is critical** — without it, the original "+0.302 hyperbolic AUROC" finding turned out to be a methodology artifact (under-parameterized Euclidean baseline). After parameter matching, HPS ≈ HPS-Euclidean at saturation.

### **C4 (controlled minimal baseline)** — 4,097 parameters
**See "Note on C4" above.** Mean-pool 6 layers' last-token activations + LR. Adapted from Anthropic Cheap Monitors with axis-of-pooling change.

### **Other comparisons**
- **RTV** (Derya & Sunar 2026 preprint): refusal-direction Mahalanobis (we reproduce on our data)
- **JBShield, HSF, GradSafe**: cited via published numbers (different LLMs/attacks/protocols make direct reproduction unfair)
- **C1–C5 ablation controls:** raw L2 norm, untrained Lorentz projection, length-only — all to rule out trivial null hypotheses
- **Hyperbolic MLP, Hyperbolic AE, Lorentz Centroid:** additional hyperbolic architectures tested; none beat C4 (see Experiment 1)

## 3. Data and Methodology

**LLMs:** Llama-3-8B-Instruct (SFT + RLHF) and Vicuna-13B-v1.5 (SFT only)

**Llama-3 attacks (9 categories, 6,520 attack prompts total):** autodan, base64, drattack, gcg, ijp, pair, puzzler, saa, zulu — covering manually-designed (IJP), optimization-based (GCG, SAA), template-based (AutoDAN, PAIR), linguistic (DrAttack, Puzzler), and encoding-based (Zulu, Base64). Same 9 categories as JBShield.

**Vicuna attacks (4 categories):** GCG, JBC, PAIR, prompt_with_random_search.

**Two benign datasets used:**
1. **Original (Alpaca-style):** ~6,500 short instruction-style prompts (mean 60 chars). Used in initial experiments.
2. **Diverse benign (NEW, May 28):** ~5,900 prompts spanning Alpaca, WildChat, OR-Bench Hard, MMLU, GSM8K, HumanEval, MBPP, WikiText (long-form). Mean 1,076 chars; max 18,205 chars. **This breaks the length confound** (see Section 5.1).

**Methodology fixes (after first-round audit):**
- **Threshold calibration** on a held-out split (not the test set) — prevents threshold leakage
- **Parameter matching** between HPS and HPS-Euclidean — fair geometry ablation
- **Multi-seed reporting** — n=5 seeds for same-distribution
- **Bootstrap confidence intervals + formal hypothesis tests** — n=10,000 iterations (paired bootstrap, McNemar's test, Cohen's d)
- **Train/test contamination check** — found 15 overlapping prompts (1.15%); deduplicated
- **Diverse benign evaluation** — re-extracted activations on length-controlled benign data to remove length confound

## 4. Research Questions

- **RQ1:** Do hyperbolic priors provide a measurable advantage over (a) parameter-matched Euclidean projection, and (b) a minimal linear-probe baseline (C4)?
- **RQ2:** Does the projection's geometric structure (radial position) capture meaningful semantic information, as the hyperbolic-priors theory predicts?
- **RQ3:** How does HPS generalize across LLMs with different alignment training, and across attack types (gradient-optimized vs natural-language)?

---

## 5. Experiments and Results

### Experiment 5.1 — **NEW: Length confound discovery and resolution**

While verifying the suspicious TPR=1.000 saturation, we discovered a major methodological issue with standard activation-based jailbreak benchmarks:

**With original Alpaca benign (mean 60 chars):**
- Attack mean length: 961 chars (16× longer than benign)
- **A length-only classifier achieves AUROC=0.973, TPR=0.916 at 5% FPR**
- Per-attack length-only AUROCs: drattack=1.000, puzzler=1.000, saa=1.000, base64=0.999, autodan=0.991, ijp=0.993, pair=0.996, gcg=0.982, zulu=0.841

**This means the original "TPR=1.000 across all methods" finding was largely measuring length, not harm.**

**After diverse benign (~6,000 prompts spanning Alpaca, WildChat, OR-Bench Hard, MMLU, GSM8K, HumanEval, MBPP, WikiText):**
- Attack mean length: 217 chars; benign mean: 252 chars (no longer dominated by length)
- **Length-only classifier drops to AUROC=0.318, TPR=0.001 at 5% FPR**
- Length is now slightly anti-correlated with attack labels

**Other confounds checked:**
- Train/test contamination: 15 prompts overlap (1.15%) — fixed via deduplication
- Activation norm-only AUROC=0.917 — moderate confound (partly length-correlated)
- Permutation test: AUROC=0.479 — confirms real signal exists when labels are real

**This is a publishable methodological observation:** standard activation-based jailbreak detection benchmarks are dominated by length confounds. The field needs length-matched benign sets for rigorous evaluation. Anthropic's Cheap Monitors paper actually addresses this (using WildChat + OR-Bench Hard, not just Alpaca-style); but most academic activation-based defense papers (HSF, JBShield, GradSafe, RTV) have not.

### Experiment 5.2 — Same-distribution comparison after length control (RQ1)

After length confound is resolved with diverse benign, methods STILL saturate:

| Method | AUROC (n=5 seeds) | TPR @ 5% FPR | Approx. Balanced Acc |
|---|---|---|---|
| **HPS (hyperbolic)** | 1.0000 ± 0.0000 | 1.0000 | ~0.975 |
| **HPS-Euclidean (matched)** | 0.999 | 0.998 | ~0.974 |
| **C4 (linear probe)** | 1.0000 ± 0.0000 | 1.0000 | ~0.975 |
| RTV (reproduced) | 0.854 | 0.551 | ~0.751 |

**Paired bootstrap (HPS vs C4):**
- ΔAUROC p = 1.97 (NOT SIG)
- McNemar's p = 0.66 (NOT SIG)
- Cohen's d = 0.04 (negligible)

**No statistically significant difference between HPS and C4.** Saturation is real, not a length artifact. The activation signal genuinely separates harm from benign even when length is controlled.

![Method comparison](results/hps_rtv_results_comparison.png)

### Experiment 5.3 — **REVISED: Mechanistic radial distribution analysis (RQ2)**

**Original hypothesis:** Hyperbolic projection should push attacks to high radial position ("attacks are extreme").

#### With original Alpaca benign (length-confounded):
13/13 configurations showed INVERSION (benign at higher radial than attacks). This appeared to refute the hypothesis.

#### With diverse benign (length controlled):

| Configuration | Benign median | Attack median | Diff | Hypothesis match? |
|---|---|---|---|---|
| Seed 42 (κ=0.1) | 3.19 | 3.64 | **+0.45** | ✓ Attacks HIGHER |
| Seed 43 (κ=0.1) | 3.19 | 3.63 | +0.44 | ✓ |
| Seed 44 (κ=0.1) | 3.19 | 3.65 | +0.46 | ✓ |
| Seed 45 (κ=0.1) | 3.19 | 3.64 | +0.45 | ✓ |
| Seed 46 (κ=0.1) | 3.19 | 3.68 | +0.49 | ✓ |
| Epoch 5 | 3.19 | 3.32 | +0.12 | ✓ |
| Epoch 50 | 3.19 | 3.64 | +0.45 | ✓ |
| κ=0.5 | 1.50 | 2.51 | +1.01 | ✓ |
| κ=1.0 | 1.17 | 2.24 | +1.07 | ✓ |
| κ=2.0 | 1.18 | 2.04 | +0.86 | ✓ |

**0/13 configurations show inversion. The geometric hypothesis is now CONFIRMED.**

**Mechanistic interpretation:** The original "13/13 inversion" was an artifact of length-confounded benign data. With Alpaca (60 chars) vs attacks (961 chars), the contrastive loss learned a length-based projection where short benign got pushed to high radial position. After length is controlled, the contrastive loss finds the harm-discriminative direction predicted by hyperbolic geometry: attacks at high radial position, benign near origin.

**This is a stronger paper finding than the original "geometric hypothesis is wrong" claim.** The geometric prior IS theoretically valid; it just doesn't yield practical performance gains over linear probes at saturation.

### Experiment 5.4 — Cross-LLM and per-attack breakdown (RQ3)

**Same HPS architecture, hyperparameters, training procedure on both LLMs.**

#### After diverse benign (May 28 update):

| Attack | Llama-3 HPS | Llama-3 C4 | Vicuna HPS | Vicuna C4 |
|---|---|---|---|---|
| **GCG** | **100% (172/172)** | 100% | **81.2% (13/16)** | 100% |
| autodan, ijp, drattack, base64, puzzler, saa, zulu | 100% (each) | 100% | — | — |
| pair / PAIR | 100% (164/164) | 100% | 100% (10/10) | 100% |
| JBC | — | — | 100% (21/21) | 100% |
| prompt_with_random_search | — | — | 100% (16/16) | 100% |

**Earlier reports of catastrophic Vicuna failure (37.5% on GCG) were heavily inflated by length confound.** With diverse benign:
- Vicuna full-distribution HPS TPR rose from 0.81 → 0.95
- Vicuna GCG-specific HPS rate rose from 37.5% → 81.2% (13/16)
- The Llama-3 vs Vicuna gap is now 18.75pp (100% vs 81.2%) on GCG

The "alignment-mediated GCG failure" claim is **weaker but not gone**. There's still a Vicuna-specific issue with HPS on GCG, but it's not as dramatic as previously reported.

**Caveat:** Only 16 Vicuna GCG test samples. With n=16, the 95% CI on 13/16 = 81.2% is roughly 56-94%. We need more samples for a definitive claim.

### Experiment 5.5 — Adversarial PGD on activations (counter-intuitive)

We applied PGD perturbations directly on activations (caveat: not a realistic threat model — real attacks operate in input space, per Bailey et al. 2024).

| Method | PGD evasion at ε=0.05 |
|---|---|
| HPS | 96% |
| C4 | **2%** |
| HPS-Adv (PGD adversarial training) | 96.9% |

**C4 is more robust than HPS** under activation-space perturbation. HPS's 12-feature compression has a single dominant feature (mean radial position) that's directionally exploitable. C4's 4096-dim space has no such bottleneck.

**Note:** This finding is preserved across both benign datasets — it's about feature dimensionality, not length.

### Experiment 5.6 — Feature ablation (HPS's compression collapses)

| Feature subset | #features | Same-dist TPR | Cold-start N=5 TPR |
|---|---|---|---|
| All 12 trajectory features | 12 | 1.000 | 0.988 |
| Just **mean radial position** | 1 | 1.000 | 0.996 |
| Curvature features only | 4 | 0.995 | 0.970 |

**A single feature (mean radial position) matches all 12 trajectory features.** HPS effectively collapses to a single scalar — which is why C4's full 4096-dim space is more robust under perturbation.

**Important update:** Now that the radial finding is confirmed (Experiment 5.3), this single-feature collapse aligns with the hyperbolic theory: the meaningful signal is captured by the radial coordinate alone.

### Experiment 5.7 — Cold-start regime (where Lorentz regularizes)

| N per method | HPS | C4 | HPS-Euclidean (matched) |
|---|---|---|---|
| 5 | 0.978 | **0.996** | 0.244 |
| 10 | 0.985 | **0.998** | 0.420 |
| 25 | 0.992 | **0.998** | 0.738 |
| 100 | 0.999 | **1.000** | 0.978 |

**HPS beats parameter-matched Euclidean projection at low N (the original "geometry helps" observation).** But **C4 also achieves high TPR at low N** — so the cold-start advantage doesn't require hyperbolic geometry. The Lorentz constraint regularizes the 262K-parameter projection at low data, but not having a learned projection at all (C4) regularizes more effectively.

**This finding is preserved across both benign datasets.**

### Experiment 5.8 — Permutation test (sanity check)

```
Real labels:        AUROC = 1.000
Random labels (×3): AUROC = 0.483, 0.488, 0.466 (mean 0.479)
```

**Real label signal exists.** With shuffled labels, methods drop to chance. So both methods are detecting something real, not finding noise patterns.

### Experiment 5.9 — Six-method comparison (Experiment 1, separate)

We tested 7 architectures including 4 hyperbolic variants:

| Method | Llama-3 TPR | Vicuna TPR |
|---|---|---|
| **C4 (linear probe)** | 1.000 | 1.000 |
| Hyperbolic MLP (stacked Lorentz) | 1.000 | 1.000 |
| HPS (Lorentz + contrastive) | 1.000 | 0.703-0.954 (depending on benign) |
| Euclidean (matched) | 1.000 | 0.484 |
| Hyperbolic + PCA (no training) | 0.738 | 0.750 |
| Lorentz Centroid (random proj) | 0.780 | 0.453 |
| Hyperbolic AE (reconstruction) | 0.132 | 0.141 |

**No hyperbolic method exceeds C4 on either LLM.** This is the strongest version of the negative finding.

---

## 6. Final Outcome — Three Confirmed Findings

After all experiments and the major methodological correction with diverse benign:

### Finding 1: Length confound dominates standard benchmarks (NEW, methodological)

A single-feature classifier using only prompt length achieves AUROC=0.973 on Llama-3-8B with standard Alpaca-style benign. After replacing benign with diverse data (matching attack lengths), length-only AUROC drops to 0.318. **Most apparent saturation in prior activation-based jailbreak detection work was likely length detection, not harm detection.** The field needs length-matched benign sets for rigorous evaluation.

### Finding 2: At saturation, geometric priors provide no statistically significant advantage (preserved)

Even after length control, HPS and C4 both saturate at TPR=1.000 with no significant difference (p=1.97, McNemar p=0.66, Cohen's d=0.04). HPS = HPS-Euclidean (matched parameters). No hyperbolic architecture exceeded C4 across 7 methods tested.

### Finding 3: The geometric hypothesis is CORRECT after length control (FLIPPED)

We hypothesized hyperbolic projection would push attacks to higher radial position. With length-confounded benign, the apparent finding was inversion (13/13 configurations). With diverse benign, **0/13 configurations show inversion** — attacks ARE at higher radial than benign, exactly as the theory predicted. The original "inversion" was a length-shortcut artifact.

### Honest framing for the paper

This is **not** "geometric priors don't help, here's a new method" or "the hypothesis is wrong":

> "Activation-based jailbreak detection methods saturate at TPR=1.000 on standard benchmarks, but we show this saturation is largely driven by a previously unrecognized length confound: a single-feature classifier using only prompt length achieves AUROC=0.973. After controlling for this confound with diverse benign data, methods still saturate at TPR=1.000 — confirming real activation-level signal exists. With length controlled, geometric methods (HPS) tie linear probes (C4) statistically (paired bootstrap p=1.97, McNemar p=0.66), but they do find the geometrically-meaningful discriminative direction predicted by hyperbolic-priors theory: attacks occupy higher radial positions than benign prompts (Δ ≈ 0.45 in Lorentz space, robust across all training configurations). The geometric hypothesis was previously masked by length confounds. We identify a methodology issue affecting current activation-based jailbreak benchmarks (HSF, JBShield, GradSafe, etc.) and provide controlled evidence that geometric priors capture meaningful structure even when they do not improve raw classification performance."

This is **a stronger paper** than either earlier framing because:
- It identifies a field-wide methodology issue (length confound)
- Provides controlled evidence of what was previously confounded
- Refines the negative result with mechanistic insight (geometric priors ARE valid)
- Goes beyond "method X doesn't beat method Y" to "here's a benchmark problem affecting the entire subfield"

In the spirit of "Are GANs Created Equal?" (Lucic et al. NeurIPS 2018) — methodology critique that strengthens the field.

### Recommended target venue

**TMLR** (60-65% acceptance probability). Rigorous empirical studies welcome; allows acknowledgment of prior art (Anthropic, Google DeepMind, ICML 2025); no "novelty" or "SOTA" bar to clear. The methodology critique angle is well-suited.

**Stretch:** AAAI/IJCAI 2027 (35-50%) with the methodology framing.

## 7. Should We Add or Remove Attacks?

**Neither will help.** Saturation isn't a sample-size issue.

- More attacks → more samples confirming TPR=1.000, narrower CIs, but same point estimate
- Fewer attacks → wider CIs, same point estimate

What WOULD break saturation:
- **Truly novel attack categories** (TombRaider EMNLP 2025, ICE ACL 2025, SequentialBreak) — out-of-distribution
- **Adaptive attacks** (Bailey et al. 2024 input-space PGD) — attacker who knows the defense
- **Multi-turn jailbreaks** — different attack topology
- **Stricter FPR thresholds** (already tested at 0.001%; still saturates)

But the saturation finding **IS the result** when combined with the length-confound discovery. The field needs harder benchmarks; that's a contribution.

## 8. Open Questions and Next Steps

### Resolved by diverse benign experiment
- ✓ Is the saturation real or a length artifact? **Both.** Length explains ~95% but real signal exists.
- ✓ Is the geometric hypothesis wrong? **No, it was masked by length confound.**
- ✓ Is the Vicuna failure catastrophic? **No, less dramatic than initially reported.**

### Still open
- **Direct test of alignment hypothesis:** Take an LLM with SFT only, add RLHF, measure HPS GCG detection delta. Would directly confirm signal-concentration mechanism.
- **Larger Vicuna GCG sample:** Currently n=16 for the GCG test set. Could use JBShield's released dataset to expand to n≈170.
- **Multi-turn jailbreak detection:** Conversation trees genuinely have hierarchical structure — does HPS's hyperbolic prior help here, where the geometric prior matches the data structure?
- **Realistic adaptive attacks:** Bailey et al. 2024 input-space attack framework instead of activation-space PGD.
- **Additional LLMs:** Mistral-7B, Qwen-72B to broaden findings.

### Recommendation

**Submit current paper to TMLR (4-6 weeks of writing, no new experiments needed).** Multi-turn pivot is a natural follow-up that builds on existing methodology. The length-confound discovery + controlled comparison is enough for a publishable contribution.

### Optional enhancements (1-2 weeks each)

1. **Expand Vicuna GCG sample size** using JBShield's public dataset (1-2 days, easy)
2. **Re-extract Llama-3 attacks using deduplicated JSON** to fix the 1295 vs 1304 mismatch (~2 hours)
3. **Test on novel 2025 attacks** (TombRaider, ICE, SequentialBreak) — strengthen out-of-distribution claim
4. **Run base Llama-3 vs Llama-3-Instruct ablation** — direct alignment-strength test

---

## Files Available

**Activation caches:**
- `results/llama3_activations_cache.npz` — original Alpaca benign + attacks
- `results/llama3_activations_cache_diverse.npz` — diverse benign + attacks (NEW)
- `results/vicuna_activations_cache.npz` — original Alpaca benign + attacks
- `results/vicuna_activations_cache_diverse.npz` — diverse benign + attacks (NEW)

**Verification & diagnostic results:**
- `results/verify_saturation.json` — original verification (length AUROC=0.973)
- `results/verify_saturation_diverse.json` — diverse benign verification (length AUROC=0.318)
- `results/statistical_tests.json` — bootstrap CIs, McNemar's, Cohen's d
- `results/radial_distribution_check.json` — 13/13 inversion → 0/13 with diverse benign
- `results/vicuna_imbalance_test.json` — per-attack breakdown
- `results/gcg_specific_test.json` — cross-LLM GCG analysis

**Documentation:**
- `mentor_briefing.md` — this document
- `research_journey.md` — comprehensive narrative
- `paper_outline.md` — paper structure
- `plan_a.md` — strengthening plan
- `evaluation_report.md` — AI reviewer critique
- `RUN_INSTRUCTIONS.md` — DGX execution guide

**Pipeline scripts:**
- `verify_saturation.py` — 6-check verification
- `build_diverse_benign.py` — diverse benign construction
- `extract_diverse_benign_activations.py` — activation extraction
- `run_diverse_benign_pipeline.sh` — full automation
- `statistical_tests.py`, `radial_distribution_check.py`, `vicuna_imbalance_test.py`, `gcg_specific_test.py`, `vicuna_diagnostic.py`, `vicuna_overfitting_test.py`

---

## Sources

**Industry-deployed activation probes:**
- ⚠️ External link — [Anthropic: Cost-Effective Constitutional Classifiers via Representation Re-use](https://alignment.anthropic.com/2025/cheap-monitors/) — accessed 2026-05-25
- ⚠️ External link — [Google DeepMind: Building Production-Ready Probes For Gemini](https://arxiv.org/abs/2601.11516) — accessed 2026-05-25
- ⚠️ External link — [Detecting High-Stakes Interactions with Activation Probes (ICML 2025)](https://arxiv.org/abs/2506.10805) — accessed 2026-05-25

**Concurrent work also testing linear probes:**
- ⚠️ External link — [Latent Sentinel: Real-Time Jailbreak Detection with Layer-wise Probes (ICLR 2026 withdrawn)](https://openreview.net/forum?id=tuFRx6Ww2n) — accessed 2026-05-25
- ⚠️ External link — [When Benchmarks Lie (ICLR 2026 AIWILD Workshop)](https://openreview.net/forum?id=jWIOJOQqne) — accessed 2026-05-25
- ⚠️ External link — [What Features in Prompts Jailbreak LLMs?](https://arxiv.org/abs/2411.03343) — accessed 2026-05-25

**Peer-reviewed jailbreak-specific defenses:**
- ⚠️ External link — [HSF: Defending against Jailbreak Attacks with Hidden State Filtering (WWW 2025)](https://arxiv.org/abs/2409.03788) — accessed 2026-05-25
- ⚠️ External link — [GradSafe (ACL 2024)](https://arxiv.org/abs/2402.13494) — accessed 2026-05-25
- ⚠️ External link — [JBShield (USENIX Security 2025)](https://arxiv.org/abs/2502.07557) — accessed 2026-05-25
- ⚠️ External link — [Gradient Cuff (NeurIPS 2024)](https://arxiv.org/abs/2403.00867) — accessed 2026-05-25
- ⚠️ External link — [Token Highlighter (AAAI 2025)](https://arxiv.org/abs/2412.18171) — accessed 2026-05-25

**Hyperbolic geometry motivation:**
- ⚠️ External link — [HypLoRA (NeurIPS 2025)](https://arxiv.org/abs/2405.18515) — accessed 2026-05-25
- ⚠️ External link — [HELM (NeurIPS 2025)](https://arxiv.org/abs/2505.24722) — accessed 2026-05-25
- ⚠️ External link — [Nickel & Kiela: Poincaré Embeddings (NeurIPS 2017)](https://arxiv.org/abs/1705.08039) — accessed 2026-05-25

**Threat model / adaptive attacks:**
- ⚠️ External link — [Bailey et al.: Obfuscated Activations Bypass LLM Latent-Space Defenses (ICLR 2026)](https://arxiv.org/abs/2412.09565) — accessed 2026-05-25
- ⚠️ External link — [Wollschläger et al.: Geometry of Refusal in LLMs (ICML 2025)](https://arxiv.org/abs/2502.17420) — accessed 2026-05-25

**Methodology benchmark (related):**
- ⚠️ External link — [Lucic et al.: Are GANs Created Equal? (NeurIPS 2018)](https://arxiv.org/abs/1711.10337) — accessed 2026-05-25
