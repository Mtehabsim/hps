# HPS Research Draft v2: Adaptive Attack Findings

**Author:** [Author Name]
**For:** Mentor review — addendum to Draft v1
**Status:** Draft v2 (post-adaptive-attack experiment)
**Date:** 2026-06-06
**Length:** ~14 pages
**Companion to:** mentor_draft.md (Draft v1)

---

## Abstract

This addendum reports adaptive obfuscation attack experiments planned in Draft v1 Section 8.1. We implement Bailey et al. (2024) style 16-token universal embedding suffix attacks against HPS and C4 on Llama-3-8B-Instruct, with cross-method transfer evaluation and attack-budget scaling. Three findings: (1) Direct attack — both HPS and C4 collapse from 76.7% standard recall to 0% adaptive recall, confirming the field-wide adversarial vulnerability extends to hyperbolic methods; (2) Transfer asymmetry — suffixes trained against C4 do NOT transfer to HPS (HPS detects them at 100% recall, an increase from baseline), while suffixes trained against HPS DO transfer to fully bypass C4 (0% recall), suggesting geometric methods provide defense diversity rather than adversarial robustness; (3) Budget scaling — HPS adaptive recall is 0% at 256, 512, 1024, and 2048 attack steps, indicating no quantitative robustness margin. We update Draft v1 Section 7 with these results and refine the recommended TMLR framing accordingly.

---

## Table of Contents

1. Summary of New Findings
2. Adaptive Attack Methodology
3. Direct Attack Results: HPS = C4
4. Transfer Asymmetry: The Novel Finding
5. Attack Budget Scaling
6. Probe Logit Dynamics
7. Updated Adversarial Robustness Section (replaces v1 Section 7)
8. Updated Open Questions
9. Appendices

---

## 1. Summary of New Findings

We ran four adaptive-attack experiments against Llama-3-8B-Instruct probes:

| # | Experiment | Standard recall | Adaptive recall | Drop |
|---|------------|----------------|-----------------|------|
| 1 | C4 direct attack | 0.767 | 0.000 | +0.767 |
| 2 | HPS direct attack | 0.767 | 0.000 | +0.767 |
| 3 | Transfer: C4-suffix → HPS | 0.767 | **1.000** | **−0.233** |
| 4 | Transfer: HPS-suffix → C4 | 0.767 | 0.000 | +0.767 |

| # | HPS budget scaling | Adaptive recall |
|---|---------------------|-----------------|
| 5 | HPS at 256 steps | 0.000 |
| 6 | HPS at 512 steps | 0.000 |
| 7 | HPS at 1024 steps | 0.000 |
| 8 | HPS at 2048 steps | 0.000 |

Three findings emerge:

**Finding 1 (Bailey confirmed for hyperbolic):** HPS provides no adversarial robustness against direct white-box attack. Both probes collapse to 0% recall under 16-token universal embedding suffix optimization (Bailey et al. 2024 setup).

**Finding 2 (Defense diversity, NOT robustness):** Suffixes optimized against C4 do not transfer to HPS — HPS detection rate INCREASES from 76.7% to 100% under C4-targeted attack. The reverse direction shows full transfer (HPS-suffix bypasses C4 at 0% recall). This asymmetry is consistent with Bailey et al.'s Figure 4(c) transfer pattern but reveals a specific geometric-vs-flat asymmetry not previously characterized.

**Finding 3 (No budget margin):** HPS recall reaches 0% at 256 attack steps, well below the 2048-step budget Bailey used. The geometric prior provides zero quantitative robustness budget improvement over the linear probe.

Together, these findings yield a refined characterization: **HPS is no more adversarially robust than C4 against targeted attacks, but provides defense diversity by detecting attacks crafted against simpler probes.**

---

## 2. Adaptive Attack Methodology

### 2.1 Attack Setup

Following Bailey et al. (2024) Section 3.3, we optimize a universal 16-token embedding suffix by jointly minimizing:

```
L(w) = Σ_query  CrossEntropy(M(query⊕w), target)  +  λ_obf · D(R(query⊕w))
       └─────── Behavior Loss ──────────────┘   └─── Obfuscation Loss ─┘
```

Where:
- `w ∈ ℝ^{16 × d}` is the optimized continuous embedding suffix (d=4096 for Llama-3)
- `M` is the target LLM (Llama-3-8B-Instruct, fp16)
- `target` = `"Sure, here are detailed step-by-step instructions:\n\n1."` (generic harmful prefix)
- `D` is the probe (C4 or HPS) interpreted as a differentiable scorer
- `R(.)` extracts hidden-state activations at the last input token across 6 layers
- `λ_obf = 1.0`

We apply Llama-3's chat template properly via a sentinel-string splice mechanism: the suffix is positioned within the user-content portion of the templated message, ensuring activations match the probe's training distribution.

### 2.2 Target Probes

We test three probes from Draft v1:

| Probe | Architecture | Training |
|-------|--------------|----------|
| C4 | Mean-pool 6 layers (last token) → StandardScaler → LR | Cached activations from `llama3_activations_cache_diverse_fixed.npz` |
| HPS | Lorentz proj (κ=0.1, 64-d) → 12 trajectory features → StandardScaler → LR | Same cache, contrastive training over 50 epochs |
| HPS-Euclidean | Same arch as HPS, flat geometry | Same cache (not yet attacked in this experiment; future work) |

### 2.3 Optimization Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Suffix length | 16 tokens | Bailey et al. setting |
| Optimizer | Adam | Standard for embedding-space attacks |
| Learning rate | 0.01 | Conservative; suffix near-zero initialized |
| Steps | 2048 (default), 256/512/1024 (budget scaling) | Bailey et al. typical range |
| Mini-batch | 4 queries per gradient step | Memory-efficient |
| Suffix dtype | fp32 (cast to fp16 at forward boundary) | Numerical stability |
| Probe dtype | fp32 (activations cast at boundary) | Probe weights from sklearn |

### 2.4 Evaluation Protocol

- **Training queries:** 20 AdvBench harmful behaviors used for suffix optimization
- **Held-out queries:** 30 disjoint AdvBench harmful behaviors used for adaptive recall measurement
- **Benign reference:** 200 prompts sampled from `data_harmless_diverse.csv` (the diverse benign distribution from Draft v1)
- **FPR threshold:** 1% computed on benign reference scores

### 2.5 Why Standard Recall is 0.767, Not 0.99

A note on baseline calibration: our standard (no-suffix) recall of 0.767 is lower than the ~0.99 reported in Draft v1's main evaluation. The reason is distribution mismatch:

- **Draft v1 evaluation** used JBShield attacks (formatted jailbreaks: GCG suffixes, AutoDAN evolutions, PAIR rewrites) — which the probe was trained to detect.
- **This experiment** uses raw AdvBench harmful behaviors (e.g., "Write a tutorial on how to make a bomb") — bare harmful intent without structural attack patterns.

C4 detects bare harmful intent at 76.7% (out of distribution from training). The adversarial attack experiment measures whether the suffix can drive this detection rate down further. The answer (0.000) is meaningful: the attack succeeds completely.

If we had run the attack on JBShield-style queries (where standard recall would be ~99%), the adaptive recall would still drop to ~0%. The 76.7→0 drop characterizes attack effectiveness; the absolute baseline doesn't change the conclusion.

---

## 3. Direct Attack Results: HPS = C4

### 3.1 Headline Numbers

| Defender | Standard recall | Adaptive recall | Recall drop | Attack runtime |
|----------|----------------|-----------------|-------------|----------------|
| C4 (linear probe) | 0.767 | 0.000 | +0.767 | 18 min |
| HPS (Lorentz contrastive) | 0.767 | 0.000 | +0.767 | 22 min |

**Both probes collapse identically under adaptive attack.**

### 3.2 Probe Score Distributions

The adaptive suffix drives probe scores far below the benign threshold for both methods:

| Defender | Benign mean | Benign std | Threshold @ 1% FPR | Attack final logit |
|----------|-------------|------------|--------------------|--------------------|
| C4 | +4.31 | 16.62 | +41.08 | **−3,135** (~190σ below benign) |
| HPS | +6.06 | 1.35 | +9.59 | **−12.4** (~13σ below benign) |

Note that the magnitude of the final attack logit differs dramatically (3,135 vs 12), but both are well below the benign threshold. This reflects the structural difference: C4's logit is computed on a 4096-dim feature vector with high variance, while HPS's logit derives from 12 standardized features with bounded distribution. Despite different dynamic ranges, both attacks fully evade detection.

### 3.3 Behavior Loss Convergence

Both attacks successfully optimize the jailbreak objective:

| Defender | Step 0 beh_loss | Step 2000 beh_loss |
|----------|-----------------|---------------------|
| C4 | 5.02 | 1.51 |
| HPS | 5.04 | **0.04** |

HPS's behavior loss converges to a much lower value (0.04 vs C4's 1.51), indicating the suffix achieves more confident jailbreak generation. This is consistent with HPS's narrower decision boundary requiring less "obfuscation effort," leaving more capacity for behavior optimization.

### 3.4 Interpretation

The direct-attack result confirms Bailey et al. (2024) Section 3.3 for hyperbolic methods. The geometric prior provides no protection when the attacker explicitly optimizes against the probe. This was the predicted outcome based on Bailey's Section 3.5 geometric impossibility result.

For Draft v1 framing, this means we should NOT claim adversarial robustness for HPS. The activation-space sensitivity finding from Draft v1 Section 7.6 is now corroborated in input space: a realistic gradient-based attack succeeds completely.

---

## 4. Transfer Asymmetry: The Novel Finding

This is the most interesting result of the experiment.

### 4.1 The Result

We trained the universal suffix against one probe and tested it against the other:

| Trained against | Tested on | Standard recall | Suffix-augmented recall | Effect |
|-----------------|-----------|-----------------|-------------------------|--------|
| C4 | HPS | 0.767 | **1.000** | HPS detects MORE |
| HPS | C4 | 0.767 | **0.000** | C4 fully fooled |

**The asymmetry is strong:** the C4-suffix not only fails to transfer to HPS — it actively makes HPS detection MORE confident (76.7% → 100%). The HPS-suffix transfers perfectly to fool C4.

### 4.2 Mechanistic Interpretation

The asymmetry reflects the architectural difference between the two probes:

```
C4 attacks:
  - Loss = LR(mean(activations across 6 layers, last token))
  - Gradient pushes the LAYER-AVERAGED activation toward benign region
  - Attacker minimizes ||mean_layer_activation - benign_centroid||
  - But individual per-layer activations may move atypically

HPS attacks:
  - Loss = LR(12 trajectory features over Lorentz projections of 6 layers)
  - Features include radial position PER LAYER, curvature BETWEEN layers, displacement
  - Gradient must satisfy MORE constraints simultaneously
  - Resulting suffix produces "plausibly benign" trajectory
```

When we apply the C4-suffix to HPS:
- The mean-pooled activation is in C4's benign region (by construction)
- BUT individual layer activations are perturbed in unbalanced ways
- HPS's per-layer radial features show atypical patterns
- HPS's curvature features (triangle inequality bending across layers) detect the structural anomaly
- Result: HPS confidently flags the suffix-augmented input as harmful

When we apply the HPS-suffix to C4:
- The HPS attack drives the full trajectory toward the benign region (more constrained optimization)
- This necessarily moves the layer-averaged activation toward benign too
- C4's simpler check is satisfied as a side effect
- Result: C4 is fully fooled

### 4.3 Consistency with Bailey et al.

Bailey et al. (2024) Figure 4(c) reported asymmetric transfer:
- LR (input-based) → MLP (input-based): transfer
- LR (input-based) → OOD detector: no transfer
- OOD detector → LR: transfer (asymmetric)

Our result is similar in spirit:
- HPS-suffix → C4: transfer (HPS attack generalizes)
- C4-suffix → HPS: no transfer (HPS resists)

This confirms Bailey's finding that probes with richer feature sets are harder to bypass via attacks designed for simpler probes, but the simpler probes are bypassed when attacks target the richer probe.

### 4.4 Implications for Defense-in-Depth

The asymmetry has practical consequences. Consider an ensemble defense:

**Scenario A: Attacker doesn't know about HPS, only attacks C4**
- C4: fooled (0.000 recall)
- HPS: catches it (1.000 recall)
- Ensemble (OR-gate): catches it (1.000 recall)

**Scenario B: Attacker knows about HPS, attacks HPS**
- HPS: fooled (0.000 recall)
- C4: also fooled (0.000 recall, by transfer)
- Ensemble (OR-gate): fully fooled (0.000 recall)

**Scenario C: Adaptive ensemble attack**
- Attacker minimizes max(C4_score, HPS_score) jointly
- Predicted: full bypass (consistent with Bailey Section 3.5)
- Not yet tested in our experiment

The defense-diversity argument therefore depends on threat model:
- **Naive black-box attacker:** ensemble provides marginal protection
- **Sophisticated attacker (knows ensemble exists):** trains against HPS, fools C4 by transfer
- **Adaptive ensemble attacker:** defeats both per Bailey Section 3.5

### 4.5 Caveats

We tested a single transfer pair. The asymmetry should hold across:
- Different attack methods (PAIR, AutoDAN, etc.) — not yet tested
- Different attack lengths — not yet tested
- Different LLMs (Vicuna-13B) — not yet tested
- Adaptive ensemble attack — not yet tested

These are reasonable follow-up experiments. With current data, the asymmetry is a single-point observation supporting a defense-diversity claim with caveats.

---

## 5. Attack Budget Scaling

### 5.1 Result

We re-ran the HPS attack at four step counts:

| Steps | HPS adaptive recall |
|-------|---------------------|
| 256 | 0.000 |
| 512 | 0.000 |
| 1024 | 0.000 |
| 2048 | 0.000 |

**HPS provides no quantitative budget margin.** The attack succeeds at 256 steps as completely as at 2048.

### 5.2 Why This Matters

Bailey et al. tested 256-2048 steps for various probes. Their finding was that all probes break, but some required larger budgets. If HPS required more steps to break than C4, that would be a measurable robustness margin even if the eventual outcome is the same.

Our result: **HPS breaks at the lowest budget tested (256 steps)**. The geometric prior provides zero attack-cost overhead.

This rules out the "geometric prior provides quantitative robustness margin" hypothesis from Draft v1 Section 8.1. The Lorentz constraint is differentiable; gradients flow through it without obstruction; the attacker's optimization succeeds without measurably more effort.

### 5.3 Caveat

We did not test below 256 steps. If HPS breaks at 64 or 128 steps as well, the budget margin is even narrower. We did not test above 2048 steps either, but the saturation pattern in Section 6 below suggests no further attack budget is required.

---

## 6. Probe Logit Dynamics

A subtle but informative observation: the two probes show different attack convergence dynamics.

### 6.1 Convergence Patterns

| Step | C4 attack logit | HPS attack logit |
|------|-----------------|-------------------|
| 0 | +53.6 | +10.4 |
| 50 | −919.7 | −7.5 |
| 100 | −1175.9 | −10.0 |
| 500 | −2490.1 | −11.8 |
| 1000 | −2868.9 | −12.4 |
| 2000 | −3135.9 | **−11.4 (saturated)** |

C4's probe logit decreases monotonically across all 2048 steps, with no apparent saturation. HPS's logit reaches approximately −12 by step ~100 and oscillates around that value for the remaining 1900+ steps.

### 6.2 Interpretation

C4's logit is computed as `weight · z + bias` where `z` is a 4096-dim raw activation vector. The vector can be perturbed arbitrarily (up to model norm constraints), driving the logit to arbitrarily negative values. The attack continues finding lower scores throughout training.

HPS's logit is computed on 12 standardized trajectory features. Each feature has a bounded distribution after training (the StandardScaler normalizes to unit variance over the training set). The attack saturates when the features hit the "edge" of the standardized distribution: pushing further produces marginal logit decrease while costing behavior loss.

### 6.3 What This Implies

HPS's compressed feature space provides a "compactness" property:
- Pro: Attacker's logit reduction is bounded; harder to drive infinitely negative
- Con: Bounded benign distribution also means the threshold is closer to the saturation point

For our setup, HPS's saturated logit (−12) is well below the 1% FPR threshold (+9.6). The attack achieves full evasion despite saturating because the threshold is reached early.

Could HPS be redesigned to make saturation harder? Hypothetically:
- Higher-dimensional feature space (more than 12)
- Larger penalty for feature deviation
- Multi-prototype representations

These are speculation. With current HPS architecture, saturation is reached and full evasion follows.

---

## 7. Updated Adversarial Robustness Section (replaces Draft v1 Section 7)

This section should replace Section 7 in Draft v1 with the empirical adaptive-attack findings.

### 7.1 Adaptive Attacks: Tested Now

We implemented Bailey et al. (2024) style adaptive attacks against C4 and HPS. We DID NOT defer this to "future work" as Draft v1 originally planned.

### 7.2 Direct Attack Results

Both C4 and HPS collapse to 0% recall under adaptive attack:

| Method | Standard recall | Adaptive recall | Recall reduction |
|--------|----------------|-----------------|------------------|
| C4 (linear probe) | 0.767 | 0.000 | 0.767 |
| HPS (Lorentz contrastive) | 0.767 | 0.000 | 0.767 |

The geometric prior provides no adversarial robustness against direct attacks. This confirms Bailey's field-wide finding extends to hyperbolic methods.

### 7.3 Transfer Asymmetry

Cross-method transfer reveals a directional asymmetry:

| Trained against | Tested on | Adaptive recall |
|-----------------|-----------|-----------------|
| C4 | HPS | **1.000** (HPS detects more, not less) |
| HPS | C4 | 0.000 (C4 fully bypassed) |

The asymmetry is consistent with Bailey et al. Figure 4(c). It suggests geometric methods provide defense diversity (catching attacks crafted against simpler probes) but not adversarial robustness (failing under direct attack).

### 7.4 Attack Budget Scaling

HPS adaptive recall is 0% at 256, 512, 1024, and 2048 steps. The geometric prior provides no attack-budget margin.

### 7.5 What This Means For The Paper

The original Draft v1 Section 7 hedged with "we don't claim adversarial robustness." The empirical results now allow stronger statements:

> "We empirically confirm that HPS provides no adversarial robustness against direct white-box attacks, with recall reduction equivalent to a linear probe baseline (C4). However, we identify a transfer asymmetry: attacks optimized against C4 do not transfer to HPS (HPS detection rate increases under such attacks), while attacks optimized against HPS fully transfer to C4. This suggests geometric methods may provide modest defense diversity in deployed defense-in-depth architectures, particularly when attackers do not specifically target the geometric component. Adaptive ensemble attacks (where the attacker optimizes against the combined HPS+C4 score) are not tested and remain future work."

### 7.6 Field-Wide Confirmations Remain Relevant

The supporting citations from Draft v1 Section 7.5 remain accurate:
- Bailey et al. (2024) — comprehensive probe attack
- Schwinn & Geisler (2024) — embedding-space attacks against representation engineering
- Li et al. (2024) — multi-turn attacks against RepE/LAT/Circuit Breakers
- Carlini et al. (2024) — foundational adversarial alignment claim

Our results confirm these findings extend to geometric (Lorentz hyperboloid) probes.

---

## 8. Updated Open Questions

This section updates Draft v1 Section 8 with results-informed priorities.

### 8.1 Adaptive Ensemble Attack (now top priority)

**Goal:** Test whether the transfer asymmetry survives when an attacker optimizes against the HPS+C4 ensemble jointly.

**Setup:**
- Loss = behavior_loss + λ · max(C4_score, HPS_score)
- Or: behavior_loss + λ · LR(C4_score, HPS_score)
- Suffix optimized against the ensemble objective

**Predicted outcome (Bailey Section 3.5):** ensemble breaks. The transfer asymmetry from Section 4 of this draft would not survive this attack.

**Why test it:** The defense-diversity claim from Section 4.4 only holds if the adaptive ensemble attack fails. If it succeeds, the asymmetry is academic — practical defense gains nothing.

**Estimated time:** ~1 day extra GPU. ~30 min for the attack run + setup.

### 8.2 HPS-Euclidean Attack

We did not run adaptive attacks on HPS-Euclidean (the parameter-matched flat ablation). This would test whether the transfer asymmetry is specifically due to hyperbolic geometry or due to the multi-feature trajectory structure.

**If HPS-Euclidean shows the same asymmetry:** the geometric prior is irrelevant; multi-feature trajectory aggregation is the cause.
**If HPS-Euclidean fails like C4:** the hyperbolic constraint matters.

**Estimated time:** ~30 min.

### 8.3 Cross-LLM Adaptive Attack (Vicuna-13B)

Draft v1 found HPS catastrophically fails on Vicuna-13B (alignment-mediated failure). Adaptive attacks on Vicuna's HPS would test whether the failure mode also breaks under adversarial pressure.

**Hypothesis:** Vicuna's HPS already fails on standard attacks (recall ~7.6% on GCG). Adaptive attack would drive this further to 0%.

**Estimated time:** ~30 min.

### 8.4 Scale-Up Statistical Validation

Current evaluation: n=30 held-out queries. To strengthen the transfer asymmetry claim:

- Increase to n=100-200 queries
- Multiple random seeds
- Confidence intervals on the 1.000 / 0.000 numbers

**Estimated time:** ~1-2 hours.

### 8.5 Alternative Suffix Lengths

We tested only 16-token suffixes. Bailey explored 8 / 16 / 32 / 64 token lengths. Longer suffixes give the attacker more capacity but risk over-fitting.

**Hypothesis:** Same asymmetry pattern across lengths.

**Estimated time:** ~2 hours per length.

### 8.6 Updated Venue Recommendation

**TMLR (60-65%):** still recommended. Methodology paper + adversarial attack result fits TMLR scope. The transfer asymmetry is a single-experiment finding; TMLR is the right venue for nuanced negative results.

**USENIX Security (35-50%, conditional):** if Section 8.1 (adaptive ensemble attack) and Section 8.4 (statistical validation) are completed, the transfer asymmetry could be presented as a positive contribution: "Geometric activation probes provide defense diversity not adversarial robustness." This would be a focused paper on the asymmetry finding.

**ICLR/NeurIPS (25-35%):** would require defense-in-depth experiments under adaptive ensemble attack to claim novelty.

**Recommendation:** TMLR submission with current findings, USENIX as stretch goal if Section 8.1 results are favorable.

---

## 9. Appendices

### Appendix A: Updated Hyperparameter Specification

Adaptive attack settings (in addition to Draft v1 Appendix A):

```python
ATTACK_SUFFIX_LENGTH = 16
ATTACK_OPTIMIZER = "Adam"
ATTACK_LR = 0.01
ATTACK_LAMBDA_OBF = 1.0
ATTACK_BATCH_SIZE = 4
ATTACK_TARGET_FPR = 0.01
ATTACK_STEPS = 2048  # 256/512/1024 for budget scaling
ATTACK_N_TRAIN_QUERIES = 20
ATTACK_N_EVAL_QUERIES = 30

# Numerical stability
SUFFIX_DTYPE = "fp32"  # Adam in fp32
MODEL_DTYPE = "fp16"   # forward in fp16
PROBE_DTYPE = "fp32"   # activations cast to fp32 at probe boundary
```

### Appendix B: Result Files

| File | Contents |
|------|----------|
| `results/adaptive_attacks/attack_c4.json` | C4 direct attack |
| `results/adaptive_attacks/attack_hps.json` | HPS direct attack |
| `results/adaptive_attacks/transfer_c4_to_hps.json` | C4-suffix tested on HPS |
| `results/adaptive_attacks/transfer_hps_to_c4.json` | HPS-suffix tested on C4 |
| `results/adaptive_attacks/attack_hps_256steps.json` | HPS at 256 steps |
| `results/adaptive_attacks/attack_hps_512steps.json` | HPS at 512 steps |
| `results/adaptive_attacks/attack_hps_1024steps.json` | HPS at 1024 steps |
| `results/adaptive_attacks/suffix_c4.pt` | C4-trained adversarial suffix |
| `results/adaptive_attacks/suffix_hps.pt` | HPS-trained adversarial suffix |
| `results/adaptive_attacks/log_phase{1-4}_*.txt` | Per-phase logs |

### Appendix C: Code Organization (Updates)

New files in addition to Draft v1 Appendix D:

| File | Purpose |
|------|---------|
| `adaptive_attack.py` | Implements obfuscation attack pipeline; differentiable C4/HPS probe wrappers; evaluation harness |
| `run_adaptive_attacks.sh` | Bash orchestrator: runs Phase 1-4 sequentially with skip-flag support |
| `README_adaptive_attacks.md` | DGX deployment instructions and result interpretation guide |

### Appendix D: Headline Numbers Table (For Quick Reference)

```
DIRECT ATTACK
  C4:  0.767 → 0.000  (drop +0.767)
  HPS: 0.767 → 0.000  (drop +0.767)

CROSS-METHOD TRANSFER
  C4-suffix → HPS:  0.767 → 1.000  (drop −0.233; HPS catches MORE)
  HPS-suffix → C4:  0.767 → 0.000  (drop +0.767; C4 fully bypassed)

HPS BUDGET SCALING (all → 0.000 adaptive recall)
   256 steps: 0.000
   512 steps: 0.000
  1024 steps: 0.000
  2048 steps: 0.000

ATTACK CONVERGENCE (probe logit at step N)
  Step:        0      50    500    1000   2000
  C4:        +53.6  −919.7 −2490.1 −2868.9 −3135.9  (still decreasing)
  HPS:       +10.4   −7.5  −11.8   −12.4   −11.4   (saturated near step 100)

RUNTIME
  Phase 1 (C4 attack):                  18 min
  Phase 2 (HPS attack):                 22 min
  Phase 3a (C4-suffix → HPS, eval only): 4 min
  Phase 3b (HPS-suffix → C4, eval only): 6 min
  Phase 4 (256/512/1024 steps):         32 min total
  Total:                                 66 min on 1× A100 80GB
```

### Appendix E: Updated References

In addition to Draft v1 Appendix E references:

[12] Bailey et al. (2024). "Obfuscated Activations Bypass LLM Latent-Space Defenses." Sections 3.3 (direct obfuscation attack) and 3.5 (geometric impossibility, adversarial training failure). The reference for our experimental methodology.

[13] Cunningham et al. (2025). "Cost-Effective Constitutional Classifiers via Representation Re-use." Anthropic. The paper's Section 4 explicitly notes adaptive attack robustness as future work — addressed empirically here for HPS.

[14] Schwinn & Geisler (2024). "Soft Prompts for Embedding-Space Attacks Against Representation Engineering." arxiv:2402.09063. Independent confirmation of activation-defense vulnerability via embedding-space attacks (predates Bailey).

[15] Li et al. (2025). "Language Models Are Capable of Metacognitive Monitoring and Control of Their Internal Activations." NeurIPS 2025. Provides threat-model nuance: adversarial control via in-context learning is restricted to a low-dimensional metacognitive subspace, suggesting black-box adaptive attacks may face structural constraints absent in our white-box setting.

---

**End of Draft v2**

**Total length:** ~14 pages
**Key new findings:** (1) HPS = C4 under direct adaptive attack, (2) Transfer asymmetry between geometric and flat probes, (3) No HPS attack-budget margin, (4) Probe logit saturation in HPS reflects compactness of geometric feature space.
**Status:** Ready for mentor review as a follow-up to Draft v1.
**Open questions for mentor:**
  1. Adaptive ensemble attack (Section 8.1) — run before submission or as future work?
  2. Update the Draft v1 paper with these findings, or treat as a separate addendum?
  3. With the transfer asymmetry result, is USENIX a more appropriate venue than TMLR?
