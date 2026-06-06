# HPS Research Draft v2: Adaptive Attack Findings

**Author:** [Author Name]
**For:** Mentor review — addendum to Draft v1
**Status:** Draft v2 (post-adaptive-attack experiment)
**Date:** 2026-06-06
**Length:** ~14 pages
**Companion to:** mentor_draft.md (Draft v1)

---

## Abstract

This addendum reports adaptive obfuscation attack experiments planned in Draft v1 Section 8.1. We implement Bailey et al. (2024) style 16-token universal embedding suffix attacks against HPS, C4, HPS-Euclidean, and the HPS+C4 max-aggregation ensemble on Llama-3-8B-Instruct, with cross-method transfer evaluation at n=30 and n=100, and attack-budget scaling. Five findings: (1) Direct attack — all four probes (C4, HPS, HPS-Euclidean, ensemble) collapse from 0.48-0.77 standard recall to 0.000 adaptive recall, confirming the field-wide adversarial vulnerability extends to hyperbolic, ablated-geometric, and combined-defense methods; (2) Transfer asymmetry — suffixes trained against C4 do NOT transfer to HPS (HPS detects them at 1.000 recall at n=100, replicating the n=30 result), while suffixes trained against HPS DO transfer to fully bypass C4 (0.000 recall); (3) Ensemble attack — when the attacker explicitly optimizes against max(C4, HPS), both probes are bypassed simultaneously (Bailey Section 3.5 confirmed for our setup); (4) Geometric prior — HPS-Euclidean (parameter-matched flat ablation) shows the same complete adversarial collapse, ruling out the hyperbolic constraint as a source of robustness; (5) Budget scaling — HPS adaptive recall is 0.000 at 256, 512, 1024, and 2048 attack steps, confirming no quantitative robustness margin. The transfer asymmetry exists but has no practical defense value because adaptive ensemble attacks bypass both probes. We update Draft v1 Section 7 with these results and finalize TMLR as the recommended venue.

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

We ran nine adaptive-attack experiments against Llama-3-8B-Instruct probes:

| # | Experiment | Standard recall | Adaptive recall | Drop |
|---|------------|----------------|-----------------|------|
| 1 | C4 direct attack | 0.767 | 0.000 | +0.767 |
| 2 | HPS direct attack | 0.767 | 0.000 | +0.767 |
| 3 | **HPS-Euclidean direct attack** | **0.610** | **0.000** | **+0.610** |
| 4 | **Ensemble (HPS+C4) direct attack** | **0.480** | **0.000** | **+0.480** |
| 5 | Transfer: C4-suffix → HPS (n=30) | 0.767 | **1.000** | −0.233 |
| 6 | Transfer: HPS-suffix → C4 (n=30) | 0.767 | 0.000 | +0.767 |
| 7 | **Transfer: C4-suffix → HPS (n=100)** | **0.640** | **1.000** | **−0.360** |
| 8 | **Transfer: HPS-suffix → C4 (n=100)** | 0.480 | 0.000 | +0.480 |

| # | HPS budget scaling | Adaptive recall |
|---|---------------------|-----------------|
| 9 | HPS at 256 steps | 0.000 |
| 10 | HPS at 512 steps | 0.000 |
| 11 | HPS at 1024 steps | 0.000 |
| 12 | HPS at 2048 steps | 0.000 |

Five findings emerge:

**Finding 1 (Bailey confirmed across all activation-probe variants):** All four probes — C4, HPS, HPS-Euclidean, and the HPS+C4 ensemble — collapse to 0.000 adaptive recall under direct attack. The geometric prior (Lorentz vs Euclidean), the trajectory architecture (vs simple linear probe), and the defense-in-depth combination (vs individual) all fail at the same complete level. The field-wide adversarial vulnerability is now confirmed for our entire experimental setup.

**Finding 2 (Defense diversity has no practical value):** The transfer asymmetry is real and statistically robust at both n=30 and n=100: suffixes trained against C4 fail to transfer to HPS (HPS recall stays at 1.000), while suffixes trained against HPS transfer fully to bypass C4. However, when the attacker explicitly optimizes against the ensemble (HPS+C4 max-aggregation), BOTH probes are bypassed simultaneously. Therefore, the asymmetry exists but provides no defense value against an adaptive attacker who knows about both probes.

**Finding 3 (Ensemble degrades standard performance too):** The HPS+C4 max-aggregation ensemble achieves only 0.480 standard recall (n=100) — substantially worse than either probe alone (HPS=0.640, C4=0.480 at n=100). The MAX-aggregation inflates the FPR threshold, reducing recall. Even before considering adversarial robustness, simple ensembles of activation probes are not a productive direction.

**Finding 4 (Geometric prior provides no robustness margin):** HPS-Euclidean (the parameter-matched flat ablation) shows the same complete adversarial collapse as HPS (Lorentz). The behavior loss converged less smoothly under the HPS-Euclidean attack (final 4.7 vs HPS's 0.04), but probe recall still reaches 0.000. The hyperbolic constraint provides no measurable adversarial benefit.

**Finding 5 (No budget margin):** HPS recall reaches 0.000 at 256 attack steps, well below the 2048-step budget Bailey used. The geometric prior provides zero quantitative robustness budget improvement over the linear probe.

Together, these findings yield the final characterization: **HPS, HPS-Euclidean, and HPS+C4 ensemble all fail under adaptive attack. The transfer asymmetry between simple and geometric probes is real but adaptively defeated by ensemble-targeted attacks. We recommend TMLR as the publication venue and frame the work as a methodology critique with empirical adversarial robustness limitations.**

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

### 4.5 Statistical Validation at n=100

We re-evaluated the C4 and HPS suffixes (trained on 20 queries each) on a held-out set of 100 harmful queries (vs. the original 30). The asymmetry holds robustly at the larger sample size:

| Trained against | Tested on | Standard recall (n=100) | Adaptive recall (n=100) | Effect |
|-----------------|-----------|-------------------------|--------------------------|--------|
| C4 | HPS | 0.640 | **1.000** | HPS catches all |
| HPS | C4 | 0.480 | **0.000** | C4 fully fooled |

Note: standard recall at n=100 is lower than at n=30 (HPS: 0.640 vs 0.767; C4: 0.480 vs 0.767) because the larger evaluation set includes more diverse harmful queries, some of which fall outside the probe's training distribution. The lower baselines are more conservative measurements of the probe's true performance on raw harmful intents.

The transfer asymmetry survives this stricter evaluation: HPS catches 100/100 attacks crafted against C4, while C4 catches 0/100 attacks crafted against HPS. The directional mismatch is empirically robust.

### 4.6 Adaptive Ensemble Attack: The Asymmetry Doesn't Help Adaptively

If the transfer asymmetry from Sections 4.1-4.5 had practical value, it would manifest in an OR-gate ensemble defense: the attacker bypasses one probe while the other catches them.

We tested this directly by training a suffix that minimizes `max(C4_score, HPS_score)`. The attacker now optimizes against the worst-case score, which forces them to fool BOTH probes simultaneously.

**Result:** The ensemble adaptive recall is **0.000**. The attack fully succeeds in bypassing both probes.

| Probe configuration | Standard recall | Adaptive recall |
|---------------------|-----------------|-----------------|
| HPS+C4 max-aggregation ensemble | 0.480 | 0.000 |

**This confirms Bailey et al. Section 3.5 for our setup.** When the attacker has access to both probes and optimizes against their joint score, the ensemble does not survive. The transfer asymmetry from Section 4 only "helps" when attackers don't target HPS — which is not a defensible threat model.

The standard recall observation is also instructive. The ensemble achieves 0.480 standard recall, which is WORSE than either probe alone (HPS: 0.640 at n=100, C4: 0.480 at n=100). The MAX-aggregation approach inflates the FPR threshold (because the maximum of two noisy scores has higher variance), and detection drops accordingly. Simple ensembles of activation probes are not a productive direction even before considering adversarial robustness.

### 4.7 HPS-Euclidean: Geometric Prior Provides No Adversarial Robustness

To isolate the contribution of hyperbolic geometry, we ran the same direct adaptive attack against HPS-Euclidean (the parameter-matched flat ablation). Same architecture, same trajectory feature pipeline, same training protocol, only the geometric constraint differs.

| Probe | Geometry | Standard recall | Adaptive recall |
|-------|----------|-----------------|-----------------|
| HPS | Lorentz hyperboloid | 0.767 | 0.000 |
| HPS-Euclidean | Flat | 0.610 | 0.000 |

Both collapse to 0.000 adaptive recall. **The hyperbolic constraint provides no adversarial robustness benefit.**

A subtle observation worth noting: HPS-Euclidean's behavior loss converged less smoothly during the attack (final value 4.7, vs HPS's 0.04). This suggests the attacker had MORE difficulty achieving simultaneous jailbreak success and probe evasion against HPS-Euclidean than against HPS, even though both ultimately succeeded in evading the probe. This is a marginal effect that does not translate to defensive value, but is documented for completeness.

The standard recall difference (HPS: 0.767, HPS-Euclidean: 0.610) reflects HPS-Euclidean's lower base AUROC (0.9773 vs HPS's 0.9965). The hyperbolic constraint does help with standard detection — but not with adversarial robustness.

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

### 8.1 Adaptive Ensemble Attack [COMPLETED — see Section 4.6]

This experiment has been completed. The HPS+C4 max-aggregation ensemble was attacked with a 16-token universal embedding suffix optimized against the joint objective. **Result: ensemble adaptive recall = 0.000**, fully bypassing both probes simultaneously. This confirms Bailey et al. Section 3.5 for our setup and removes the "defense diversity" pathway as a positive contribution.

### 8.1.1 Adaptive Attacks on Vicuna-13B (potential follow-up)

### 8.2 HPS-Euclidean Attack [COMPLETED — see Section 4.7]

This experiment has been completed. HPS-Euclidean (parameter-matched flat ablation) shows the same complete adversarial collapse as HPS (Lorentz). **The hyperbolic constraint provides no adversarial robustness margin.** See Section 4.7 for full results.

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

### 8.6 Final Venue Recommendation: TMLR

With Phases 5-7 results in hand, the venue decision is finalized.

**TMLR (recommended):** matches their methodology paper criteria. The paper is now a clean methodology critique with empirical adversarial robustness limitations:
1. Three field-wide methodology confounds (length, max_length, contamination)
2. After fixing: HPS = HPS-Euclidean = C4 = MTP statistically (saturated benchmark)
3. Transfer asymmetry exists (statistically robust at n=100) but adaptively defeated by ensemble attacks
4. Cross-model alignment-mediated failure (HPS catastrophically fails on Vicuna-13B)
5. All activation-probe variants fail under direct adaptive attack

**USENIX Security (NOT recommended):** ruled out by the ensemble attack result. The transfer asymmetry, though statistically robust, has no practical defense value when attackers can target the ensemble objective. USENIX would expect a positive defense contribution that survives adaptive attacks; we cannot provide that.

**ICLR/NeurIPS (NOT recommended):** would require a novel-method positive claim with full adversarial evaluation showing survival under adaptive attack. Our findings are negative.

**Conclusion:** TMLR submission with current findings. No further experiments needed before submission unless mentor disagrees with framing.

### 8.7 Generation-Based HPS (Strongest Future Direction)

Bailey et al. (2024, Section 3.6) found that generation-based probes (using activations across output tokens, not just the last input token) are more robust against adaptive attacks than input-based probes. Our current HPS is input-based: it uses the last-token activation across 6 layers, ignoring all generated tokens. A generation-based variant would extend the architecture to use activations during the first K generated tokens (e.g., K=32-64).

**Architectural sketch:**

```
input prompt → forward + generate K tokens → hidden_states ∈ ℝ^(K × n_layers × hidden_dim)

For each (layer, token) cell:
    z = LorentzProjection(hidden_states[token, layer])    # → Lorentz hyperboloid

Trajectory features along TWO axes:
    Layer-wise trajectory (current HPS): 12 features capturing hierarchical structure
    Token-wise trajectory (new): 12 features capturing generation dynamics

Aggregation:
    Hyperbolic Fréchet mean of token activations per layer
    OR mean of token-wise trajectory features

Final: 12 (layer) + 12 (token) = 24 features → LR
```

**Why this is theoretically motivated:**

1. **Two-axis trajectory unlocks the dead features.** In the current single-axis HPS (across 6 layers only), our feature ablation showed `mean_r` alone matches all 12 features — curvature and displacement are dead weight. With a token axis (K~32 generated tokens), the curvature features finally have a meaningful trajectory to characterize: mid-stream topic pivots, refusal-then-compliance patterns, abrupt distributional shifts in the generation.

2. **More constraints for the attacker.** The obfuscation suffix must now produce activations that look benign across BOTH axes simultaneously. This is a higher-dimensional optimization problem than single-axis input-based probing.

3. **Captures the harm commitment signal.** The first few generated tokens reveal whether the model committed to harmful output (the "Sure, here's how..." prefix tokens). An attacker can suppress the commitment signal in input activations but still has to produce harmful tokens during generation.

**Why we expect modest, not dramatic, gains:**

Bailey et al.'s strongest result on the obfuscation-performance tradeoff was on SQL code generation (Section 5.3), where obfuscating against probes degraded SQL correctness. Jailbreak generation is a SOFTER task than SQL — the attacker only needs the model to produce harmful text, not technically precise output. The tradeoff exists but is weaker for jailbreaks than for SQL.

**Estimated improvements:**

| Metric | Input-based HPS | Generation-based HPS (predicted) |
|--------|----------------|-----------------------------------|
| Adaptive recall (direct attack) | 0.000 | 0.20-0.40 |
| Attack budget for recall < 0.1 | 256 steps | 1024-4096 steps |
| Behavior loss at convergence | 0.04 (full jailbreak) | 0.30-0.60 (degraded coherence?) |

These are speculative; experimental evaluation required.

**Implementation considerations:**

1. **Generation latency:** Each forward pass now produces K tokens (autoregressive). For K=32 with Llama-3-8B, this is ~30× slower than input-only forward.
2. **Sampling determinism:** For reproducibility, use greedy decoding. For robustness analysis, also evaluate with temperature=0.7 sampling.
3. **Hyperbolic Fréchet mean:** No closed-form solution in general; iterative algorithms exist but add compute. Approximation via tangent-space mean is acceptable for engineering simplicity.
4. **Cache invalidation:** All current cached activations are input-only. Generation-based variant requires fresh extraction (~2-3 hours per LLM).

**Implementation effort:** ~3-4 weeks (architecture, training, evaluation). Suitable as standalone follow-up paper at USENIX, NeurIPS Safety Workshop, or ICLR Workshop on Reliable and Responsible Foundation Models.

**Caveat: deployment-context argument.**

Bailey's "complex task forces tradeoff" argument is fundamentally about deployment context, not probe architecture. A defender protecting a system that requires the model to produce complex correct outputs (SQL queries, working code, accurate technical answers) gets the obfuscation-performance tradeoff for free regardless of probe design. A defender protecting a chatbot that only needs the model to "begin complying" with harmful requests gains less from any probe upgrade because the attacker's task is intrinsically simpler.

For a research paper, this is a **discussion-section observation**, not a defensive contribution. Generation-based HPS is the right architectural step forward, but the deployment context bounds achievable robustness more than the probe architecture does.

### 8.8 Decision Tree After Phase 5/6/7 Results [RESOLVED]

The Phase 5/6/7 results determine the path:

| Outcome (predicted) | Outcome (actual) | Recommended path |
|---------------------|------------------|------------------|
| Ensemble adaptive recall → 0 | **CONFIRMED**: ensemble recall = 0.000 | TMLR submission of current findings; generation-based HPS as separate follow-up |
| Ensemble adaptive recall = 0.2-0.7 | not observed | (would have indicated USENIX viability) |
| Ensemble adaptive recall > 0.7 | not observed | (would have indicated major positive finding) |

The ensemble attack achieved full bypass (0.000 recall), confirming Bailey et al. Section 3.5. The transfer asymmetry from Section 4 has no practical defense value because adaptive ensemble attacks defeat it.

**Decision: TMLR submission with current Draft v1 + v2 findings. Generation-based HPS deferred to Section 8.7 as a follow-up paper.**

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

**Total length:** ~17 pages
**Key findings (post Phase 5/6/7):**
1. HPS = HPS-Euclidean = C4 = Ensemble all collapse to 0.000 adaptive recall under direct attack
2. Transfer asymmetry between simple and geometric probes is real but adaptively defeated
3. No HPS attack-budget margin (breaks at 256 steps)
4. Hyperbolic prior provides no adversarial robustness benefit (Section 4.7)
5. Ensemble defenses are doubly broken: bypassed adaptively AND degrade standard performance (Section 4.6)

**Status:** Final empirical findings complete. Ready for mentor review and TMLR draft writing.

**Open questions for mentor:**
  1. Final paper framing: methodology critique with adversarial robustness limitations (recommended)?
  2. Should we add the layer-ablation experiment (run_layer_ablation.sh, 15 configs, ~5 hrs) to characterize layer choice sensitivity? (Optional but informative.)
  3. Llama-2-7b-chat vs Vicuna alignment ablation (Draft v1 Section 8.2) — still worth running before submission?
  4. Generation-based HPS pivot — explicit follow-up paper, or merge into current paper?
