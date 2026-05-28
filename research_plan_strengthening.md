# Research Plan: Strengthening the HPS Paper

**Date:** 2026-05-28
**Goal:** Three additions that strengthen the negative-result paper without requiring new attack generation
**Time budget:** ~1-2 weeks total (compute-bound, not generation-bound)

## Executive Summary

The `mentor_briefing.md` revision identified three concrete next steps that would substantially strengthen the paper. This document scopes each one with feasibility, time estimate, and exact implementation plan.

| Task | Feasibility | Compute time | Wall-clock | Priority |
|---|---|---|---|---|
| **JBShield Vicuna-13B data integration** | ✓ HIGH — exact same LLM, exact same 9 attacks | ~3 hours | 1 day | 🔴 HIGHEST |
| **Alignment ablation (Llama-3 base vs Instruct)** | ✓ HIGH — both models on HF, same architecture | ~6 hours | 2 days | 🟡 HIGH |
| **TombRaider novel attacks** | ⚠ MEDIUM — paper-only, no code yet | ~10-15 hours | 5-7 days | 🟢 MEDIUM |

**Recommendation:** Run JBShield integration first (highest payoff, least risk). Run alignment ablation next (direct mechanism test). Defer TombRaider unless time permits — alternative recent attacks (FlipAttack, GAP, HEA) are easier and still satisfy "out-of-distribution attack" requirement.

---

## Task 1: JBShield Public Vicuna-13B Data Integration

### What this gives us
Currently we have only 4 attack categories on Vicuna-13B (GCG, JBC, PAIR, prompt_with_random_search) with 16 GCG samples in test. After diverse benign, GCG detection at 81.2% has wide CI ([56%, 94%] for n=16).

JBShield released **9 attack categories on Vicuna-13B-v1.5** — the EXACT same model we use. This expands Vicuna evaluation from 4 → 9 attacks (matching Llama-3) and gives us 50-100 samples per attack instead of 16.

### Source

- **GitHub:** https://github.com/NISPLab/JBShield
- **License:** MIT (academic use allowed)
- **Paper:** USENIX Security 2025 (peer-reviewed, citable)
- **Data path:** `./data/jailbreak/{attack_name}/{model_name}_test.json` and `_calibration.json`
- **Attacks:** `autodan, base64, drattack, gcg, ijp, pair, puzzler, saa, zulu` — IDENTICAL to our Llama-3 attack set
- **LLMs:** Includes `vicuna-13b-v1.5` (exact same as ours), `vicuna-7b-v1.5`, `Meta-Llama-3-8B-Instruct` (also same as ours), `Llama-2-7b-chat-hf`, `Mistral-7B-Instruct-v0.2`

### Why this matters

1. **Cross-model finding becomes properly powered.** With ~80-100 GCG samples on Vicuna instead of 16, the alignment-mediated finding gets statistical force. CI for 81% on n=80 is roughly [70%, 89%] vs [56%, 94%] for n=16.

2. **Llama-3 cross-validation.** JBShield's Llama-3-8B-Instruct attacks (same model as ours, same 9 attack categories) provide independent verification. If our cache produces TPR=1.000 on JBShield's attack JSON too, it confirms saturation isn't a benchmark-construction artifact.

3. **Free additional models.** Mistral-7B and Llama-2-7b come for free — turns the cross-model analysis from N=2 to N=5 LLMs. Massively strengthens external validity.

4. **It's the same data their paper claims F1=0.94 on.** Direct comparison to JBShield's published numbers becomes possible.

### Implementation plan

```
Step 1: Download JBShield repo (5 min)
  git clone https://github.com/NISPLab/JBShield.git
  cd JBShield
  ls data/jailbreak/  # confirm 9 attack directories exist

Step 2: Inspect data structure (15 min)
  for attack in autodan base64 drattack gcg ijp pair puzzler saa zulu; do
    echo "=== $attack ==="
    python -c "import json; d=json.load(open('data/jailbreak/$attack/vicuna-13b-v1.5_test.json')); print(type(d), len(d) if hasattr(d,'__len__') else 'scalar')"
  done

Step 3: Build merged Vicuna attack JSON (30 min)
  Write build_jbshield_vicuna_attacks.py:
    - Load all 9 test+calibration JSONs for vicuna-13b-v1.5
    - Combine into single dict {attack_name: [prompts...]}
    - Save as vicuna_attacks_jbshield.json (matches our existing format)

Step 4: Extract Vicuna activations on JBShield attacks (~2 hours)
  Adapt extract_diverse_benign_activations.py:
    - Use existing diverse benign cache for benign side
    - Replace attack side with JBShield prompts
    - Output: results/vicuna_activations_cache_jbshield.npz

Step 5: Re-run experiments on the new cache (~30 min)
  python verify_saturation.py --vicuna_cache results/vicuna_activations_cache_jbshield.npz
  python statistical_tests.py --cache results/vicuna_activations_cache_jbshield.npz
  python vicuna_imbalance_test.py --vicuna_cache results/vicuna_activations_cache_jbshield.npz
  python gcg_specific_test.py --vicuna_cache results/vicuna_activations_cache_jbshield.npz

Step 6: Cross-validate Llama-3 cache against JBShield Llama-3 attacks (~3 hours)
  Same as steps 3-5 but for Llama-3-8B-Instruct
  This is the key cross-validation: if JBShield's data also gives TPR=1.000,
  saturation is robust across data sources.
```

### Risks

- **Prompt format may differ from our attack JSON** — might need wrapping/unwrapping
- **JBShield may have applied its own preprocessing** — check raw vs templated prompts
- **Sample sizes vary by attack** — drattack/puzzler may still be small (<50)

### Success criteria

- ✓ Reproduce JBShield's reported F1=0.94 on Vicuna-13B as sanity check
- ✓ Vicuna GCG sample size ≥ 80 (vs current 16)
- ✓ HPS detection rate confidence interval narrows by ~3×
- ✓ Cross-validate Llama-3 saturation finding on independent attack source

---

## Task 2: Alignment Ablation (Llama-3 Base vs Instruct)

### What this gives us
Direct test of the alignment-mediated hypothesis. Same architecture (8B parameters, identical layers), same attacks, only alignment differs.

**Hypothesis:** RLHF concentrates harm-discriminative features. If true:
- Llama-3-Instruct (RLHF) → high HPS detection (current: 100% on GCG)
- Llama-3-Base (no SFT, no RLHF) → significantly lower HPS detection

If detection drops sharply on base, the mechanism is alignment-induced. If detection stays high, the mechanism is just feature presence in the activation manifold (independent of alignment).

### Source

- **Llama-3-8B Instruct:** `meta-llama/Meta-Llama-3-8B-Instruct` (already have)
- **Llama-3-8B Base:** `meta-llama/Meta-Llama-3-8B` (no SFT, no RLHF) — both gated under same Meta license
- Same architecture: 32 layers, 4096 hidden, identical tokenizer

### Why this matters

This is the **cleanest possible alignment ablation** in the literature:
- Identical base architecture
- Identical training data up to alignment phase
- Only difference: SFT + RLHF vs no SFT/RLHF
- Hugging Face hosts both as separate checkpoints

If HPS drops on base model, your story becomes:

> "Geometric priors detect activation patterns introduced by RLHF. The hyperbolic projection's discriminative power requires alignment training to concentrate harm signal into the geometry. On unaligned models, the same architecture and same projection parameters produce significantly worse detection."

This is a **mechanistic insight that no prior paper provides**. It explains why HPS is fragile on Vicuna (SFT-only, no RLHF) and why C4 (which uses raw activations without geometric projection) is more robust to alignment-strength variation.

### Implementation plan

```
Step 1: Verify access (5 min)
  huggingface-cli login
  python -c "from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B'); print('OK')"

Step 2: Re-extract activations on Llama-3-8B base (~3 hours)
  Adapt extract_diverse_benign_activations.py:
    - --model meta-llama/Meta-Llama-3-8B (base, not Instruct)
    - Same diverse benign + same attacks as cache_diverse
    - --layers [0, 2, 17, 24, 28, 31] (same layer indices)
    - Output: results/llama3_BASE_activations_cache_diverse.npz

  Note: base model has no chat template; use raw prompts directly.
  Important: the [0,2,17,24,28,31] layer split may not be optimal for base.
  Re-run TEST 7 layer ablation? Or just use same layers for fair comparison?
  Recommend: same layers, faithful to "alignment is the only diff" claim.

Step 3: Train HPS on Llama-3 base activations (~30 min)
  python statistical_tests.py --cache results/llama3_BASE_activations_cache_diverse.npz
  Metrics expected:
    - If alignment matters: HPS AUROC drops, ΔAUROC vs Instruct becomes positive
    - If alignment doesn't matter: TPR stays at 1.000

Step 4: Per-attack breakdown (~30 min)
  python vicuna_imbalance_test.py \
    --vicuna_cache results/llama3_BASE_activations_cache_diverse.npz
  Look for: which attacks lose detection on base vs Instruct?
  Hypothesis: GCG should drop more than IJP (GCG signature requires refusal direction)

Step 5: Cross-method comparison (~30 min)
  python hyperbolic_vs_euclidean_diverse.py \
    --llama_cache results/llama3_BASE_activations_cache_diverse.npz \
    --cold_start
  Check: does C4 also drop, or is HPS uniquely vulnerable?
```

### Predicted outcomes (write before running for honest analysis)

**Scenario A (Alignment-mediated):** HPS drops 10-30% on base, C4 drops less, gap between HPS and C4 widens. **Strongest paper story.**

**Scenario B (Architecture-driven):** HPS and C4 both drop similarly. Suggests harm signal exists in pretrained activations; alignment just amplifies. **Weaker but still interesting.**

**Scenario C (Saturation persists):** HPS and C4 both stay at TPR=1.000 on base. Suggests harm signal is already linearly separable in pretrained models. **Surprising and worth reporting.** Would weaken the alignment hypothesis but create a new story: "harm features exist in pretrained representations; RLHF doesn't introduce them, just makes refusal more reliable."

All three outcomes produce publishable findings. **There's no way to lose by running this experiment.**

### Risks

- **Llama-3-8B base has no chat template** — extract activations on raw prompts only
- **Pretrained model's hidden states might be noisier** — HPS may need re-tuning (κ, layers)
- **Gated repo access** — confirm HF token has access before starting the 3-hour run

### Success criteria

- ✓ Extract base model activations cleanly (no chat template artifacts)
- ✓ Train HPS using identical procedure to Instruct version
- ✓ Compare HPS / C4 / RTV detection rates side by side
- ✓ Per-attack breakdown identifies which attacks are alignment-dependent

---

## Task 3: Novel 2025-2026 Attacks (TombRaider + Alternatives)

### What this gives us
Out-of-distribution attack categories not seen during HPS training. Tests whether HPS's learned projection generalizes beyond the 9 attack categories in the cache.

**Hypothesis:** Saturated TPR=1.000 reflects in-distribution mastery. On truly novel attack categories, methods should drop, and the difference between HPS and C4 might widen.

### TombRaider — Status

- **Paper:** EMNLP 2025 main conference (arxiv 2501.18628, last revised Aug 2025)
- **Authors:** Junchen Ding, Jiahao Zhang, Yi Liu, Ziqi Ding, Gelei Deng, Yuekang Li
- **Mechanism:** Two-agent system — inspector retrieves historical knowledge, attacker generates adversarial prompts using historical context
- **Reported ASR:** ~100% on bare models, 55.4% against defenses
- **Public code/data:** ⚠ **NOT FOUND as of 2026-05-28.** Searches returned no GitHub release.
- **Path forward:**
  - **Option A: Email authors** to request prompts/code (3-7 days response time)
  - **Option B: Reproduce attack** (~1 week to implement two-agent loop)
  - **Option C: Skip TombRaider, use alternatives below** ← RECOMMENDED for time

### Recommended alternatives (with public code)

These are 2024-2025 jailbreak papers with **publicly released attack code**, suitable as "novel attacks not in HPS training set":

#### Option 1: FlipAttack (yueliu1999/FlipAttack) — ⭐ RECOMMENDED
- **Paper:** ICML 2025
- **Code:** https://github.com/yueliu1999/FlipAttack
- **Method:** Flip characters/words in harmful prompts; LLMs misread direction-of-text
- **Reported:** 78.97% average ASR across 8 LLMs, 98% bypass against 5 guard models
- **Why this is good:** Linguistic perturbation is qualitatively different from our 9 attacks (no GCG-style suffix, no DRAttack semantic decomposition); truly OOD
- **Sample size:** Authors release ~520 prompts × 4 flipping modes = ~2080 unique attacks
- **Time to integrate:** ~3 hours

#### Option 2: Simple Adaptive Attacks (tml-epfl/llm-adaptive-attacks)
- **Paper:** ICLR 2025 (Andriushchenko et al.)
- **Code:** https://github.com/tml-epfl/llm-adaptive-attacks
- **Method:** Random search over prompt template; achieves 100% ASR on Vicuna-13B
- **Why this is good:** Tests against attack that achieves 100% jailbreak success
- **Sample size:** ~50 attacks per LLM (100% success against Vicuna-13B in their paper)
- **Time to integrate:** ~2 hours

#### Option 3: HEA — Happy Ending Attack (arxiv 2501.13115)
- **Paper:** 2025
- **Method:** Wraps harmful request in a "happy ending" narrative
- **Reported:** 88.79% ASR on GPT-4o, Llama-3-70B, Gemini-Pro
- **Why this is good:** Tests narrative-framing attacks (different from our 9)
- **Code status:** Need to check; paper from Jan 2025

#### Option 4: GAP (arxiv 2501.18638)
- **Paper:** 2025
- **Method:** Stealthy prompt generation, 96% ASR
- **Code status:** Need to check

#### Option 5: JailbreakBench artifacts (RECOMMENDED COMPLEMENT)
- **Repo:** https://github.com/JailbreakBench/artifacts
- **HF dataset:** https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors
- **Coverage:** 100 misuse behaviors × multiple attack methods × multiple target LLMs
- **Includes:** Vicuna-13B GCG/PAIR/JBC artifacts (PRE-COMPUTED; just download)
- **Time to integrate:** ~30 min — these are already extracted prompts ready to evaluate
- **Why this is good:** Independent benchmark, peer-reviewed (NeurIPS 2024 D&B Track), well-documented

### Implementation plan (recommended: FlipAttack + JailbreakBench)

```
Step 1: Download FlipAttack (10 min)
  git clone https://github.com/yueliu1999/FlipAttack.git
  cd FlipAttack
  cat data/*.json | head -100  # inspect format

Step 2: Download JailbreakBench artifacts (10 min)
  git clone https://github.com/JailbreakBench/artifacts.git
  ls artifacts/jailbreaks/  # see GCG, PAIR, etc. by target LLM

Step 3: Build novel-attack JSON (~1 hour)
  Write build_novel_attacks.py:
    - Load FlipAttack prompts (4 modes × 520 = 2080 prompts)
    - Load JailbreakBench Vicuna-13B prompts
    - Save as novel_attacks_2025.json

Step 4: Extract activations on novel attacks (~2-3 hours)
  python extract_diverse_benign_activations.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --attacks_json novel_attacks_2025.json \
    --output results/llama3_novel_attacks.npz
  Same for Vicuna.

Step 5: Train HPS on EXISTING (9-attack) cache, test on NOVEL cache (~30 min)
  This is the OOD generalization test:
    python evaluate_ood_attacks.py \
      --train_cache results/llama3_activations_cache_diverse.npz \
      --test_attacks_cache results/llama3_novel_attacks.npz

  Expected results:
    - HPS detection on novel attacks (in-distribution: 100%)
    - C4 detection on novel attacks
    - Per-novel-attack breakdown (FlipAttack-Mode1, FlipAttack-Mode2, JBB-GCG, JBB-PAIR)
    - Compare cross-distribution generalization
```

### Predicted outcomes

**If saturated:** Both HPS and C4 still detect 90%+ of novel attacks. The activation signal is robust to attack-type changes. Strengthens "harm signal is fundamental, not attack-specific."

**If broken:** Detection drops to 60-80% on novel attacks. Methods diverge meaningfully. Provides the harder benchmark the field needs.

Either result is publishable and answers a critical question.

### Time budget

| Task | Time | Compute |
|---|---|---|
| FlipAttack download + processing | 1 hour | minimal |
| JailbreakBench artifacts | 30 min | minimal |
| Activation extraction (Llama-3 + Vicuna) | 3-4 hours | GPU |
| OOD evaluation script | 1 hour | minimal |
| **Total** | ~6 hours | 4 GPU-hours |

If TombRaider authors respond with code/data within a week, can add as additional check; otherwise FlipAttack + JailbreakBench cover the "novel attacks" requirement.

---

## Combined Implementation Plan (1-2 weeks total)

### Week 1: Run the existing other-AI scripts + JBShield + Alignment ablation

**Day 1 (Monday):**
- Run other AI's three scripts to completion:
  - `python norm_check_diverse.py` (5 min)
  - `python filter_diverse_cache.py` (5 min)
  - `python hyperbolic_vs_euclidean_diverse.py --cold_start` (~60 min)
- Review results; update findings if anything contradicts current narrative

**Day 1-2 (also Monday-Tuesday):**
- Clone JBShield repo, inspect data structure
- Write `build_jbshield_vicuna_attacks.py`
- Extract Vicuna activations on JBShield attacks (~2 hours GPU)
- Run statistical tests + GCG-specific test on new cache

**Day 3 (Wednesday):**
- Repeat for Llama-3-8B Instruct using JBShield Llama-3 data (cross-validation)
- ~3 hours GPU
- Verify saturation on independent attack source

**Day 4-5 (Thursday-Friday):**
- Verify HF access to `meta-llama/Meta-Llama-3-8B` base model
- Extract activations on base model with diverse benign + attacks (~3 hours GPU)
- Train HPS, compare to Instruct version
- Per-attack breakdown
- Document outcome (Scenario A/B/C)

### Week 2: Novel attacks + Paper update

**Day 6-7 (next Monday-Tuesday):**
- Clone FlipAttack + JailbreakBench artifacts
- Build novel attack JSON
- Extract activations on novel attacks (~3-4 hours GPU)
- Run OOD evaluation

**Day 8-10 (Wednesday-Friday):**
- Update `mentor_briefing.md` with all new results
- Update `paper_outline.md` with refined story
- Generate final plots
- Draft paper sections (or send to mentor for review)

**Optional/parallel:**
- Email TombRaider authors for code (response time uncertain)
- If responses arrive, add as additional check

---

## Decision Points

### After Task 1 (JBShield integration)

If JBShield Vicuna data confirms 81% GCG detection at n≈80:
- The alignment-mediated finding is properly powered → keep in paper
- If it shoots up to 95%+: revise — mechanism may be data-scarcity not alignment

If JBShield Llama-3 cross-validation gives TPR=1.000:
- Saturation finding is robust across attack JSONs → strengthens methodology critique

### After Task 2 (Alignment ablation)

If base model HPS drops by ≥10pp:
- **Scenario A confirmed.** Lead the paper with mechanism: "alignment concentrates harm signal."
- Title: "Alignment Concentrates Harm Features: Why Geometric Priors Are Fragile"

If base model HPS stays at 100%:
- **Scenario C.** Lead with: "Harm features exist in pretrained representations; alignment doesn't introduce them."
- Major finding: harm separability is a property of the architecture, not the alignment.

### After Task 3 (Novel attacks)

If saturation persists on novel attacks:
- **Strongest possible methodology critique.** Even out-of-distribution attacks saturate.
- Title: "Activation-Based Jailbreak Detection Saturates on Standard and Novel Attacks: A Methodology Critique"

If novel attacks break saturation:
- Methods diverge meaningfully → can give actual algorithmic recommendation
- Title: "When Does Geometry Help? Hyperbolic Priors Under Distribution Shift"

---

## Existing Scripts (Other AI's — DON'T Duplicate)

These scripts are already written and ready to run. They are part of the diverse-benign re-evaluation pipeline:

```
norm_check_diverse.py             — Re-checks norm confound on diverse cache (~5 min)
filter_diverse_cache.py           — Removes 15 contaminated test rows (~5 min)
hyperbolic_vs_euclidean_diverse.py — Re-tests HPS vs HPS-Euclidean vs C4
                                     on cleaned diverse cache (~30-60 min)
```

Run these FIRST. The results inform whether the three new tasks are needed:
- If `norm_check_diverse.py` shows norm confound persists: lower priority on TombRaider, focus on confound diagnosis
- If `hyperbolic_vs_euclidean_diverse.py` shows HPS still ties C4: continue with Task 1+2 to strengthen the mechanism finding
- If `hyperbolic_vs_euclidean_diverse.py` shows HPS finally beats C4: very different paper, consult mentor

---

## Sources

**JBShield:**
- [JBShield Repository (NISPLab)](https://github.com/NISPLab/JBShield) — accessed 2026-05-28
- [JBShield Data Directory](https://github.com/NISPLab/JBShield/tree/main/data/jailbreak) — accessed 2026-05-28
- ⚠️ External link — [JBShield USENIX 2025 paper (arxiv 2502.07557)](https://arxiv.org/abs/2502.07557) — accessed 2026-05-28

**TombRaider:**
- ⚠️ External link — [TombRaider arxiv 2501.18628](https://arxiv.org/abs/2501.18628) — accessed 2026-05-28
- ⚠️ External link — [TombRaider EMNLP 2025 main conference](https://acl.ldc.upenn.edu/2025.emnlp-main.279/) — accessed 2026-05-28
- ⚠️ External link — [Author contact: Junchen Ding](https://arxiv.org/abs/2501.18628) — accessed 2026-05-28

**Llama-3 base model (alignment ablation):**
- ⚠️ External link — [Llama-3-8B base on HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3-8B) — accessed 2026-05-28
- ⚠️ External link — [Llama-3-8B-Instruct on HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) — accessed 2026-05-28

**Alternative novel attacks:**
- [FlipAttack code (yueliu1999)](https://github.com/yueliu1999/FlipAttack) — accessed 2026-05-28
- ⚠️ External link — [FlipAttack ICML 2025 paper (arxiv 2410.02832)](https://arxiv.org/abs/2410.02832) — accessed 2026-05-28
- ⚠️ External link — [Simple Adaptive Attacks code (tml-epfl)](https://github.com/tml-epfl/llm-adaptive-attacks) — accessed 2026-05-28
- ⚠️ External link — [Simple Adaptive Attacks ICLR 2025 (arxiv 2404.02151)](https://arxiv.org/abs/2404.02151) — accessed 2026-05-28
- ⚠️ External link — [Happy Ending Attack arxiv 2501.13115](https://arxiv.org/abs/2501.13115) — accessed 2026-05-28
- ⚠️ External link — [GAP attack arxiv 2501.18638](https://arxiv.org/abs/2501.18638) — accessed 2026-05-28

**Standard benchmarks:**
- [JailbreakBench artifacts repository](https://github.com/JailbreakBench/artifacts) — accessed 2026-05-28
- ⚠️ External link — [JBB-Behaviors HF dataset](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) — accessed 2026-05-28
- ⚠️ External link — [JailbreakBench NeurIPS 2024 paper (arxiv 2404.01318)](https://arxiv.org/abs/2404.01318) — accessed 2026-05-28
- ⚠️ External link — [Awesome-Jailbreak-on-LLMs collection](https://github.com/yueliu1999/Awesome-Jailbreak-on-LLMs) — accessed 2026-05-28
