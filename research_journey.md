# HPS Research Journey: From Hyperbolic Hypothesis to Linear Reality

**Status:** Negative result with strong methodology. Pivot direction: multi-turn conversational defense.
**Date:** May 2026

---

## Executive Summary

Original claim: *"Hyperbolic geometric priors enable +0.302 AUROC improvement over Euclidean projections in cross-attack jailbreak detection."*

After rigorous re-evaluation across two LLMs (Vicuna-13B, Llama-3-8B) with 9 attack families, properly parameterized baselines, multi-seed stability, and adversarial controls, the claim **does not hold**. A simple logistic regression on mean-pooled hidden activations (C4) matches or exceeds the hyperbolic projection (HPS) in every regime tested.

The work produced:
- 9 distinct experiments across the codebase
- Identification of methodology issues in the original framing
- A clean characterization of when geometric priors do and don't help
- Strong evidence that current activation-based jailbreak detection is "solved" by linear probes
- A clear path forward: multi-turn / agentic jailbreak detection

---

## 1. Initial Hypothesis (Original Paper Framing)

### Claim
> "Representation-level jailbreak defenses can be improved by projecting LLM activations onto a Lorentz hyperboloid via a learned linear projection. The hyperbolic geometric prior — exponential volume growth, monotonic radial structure — provides an inductive bias that Euclidean projections lack, enabling cross-attack generalization."

### Headline Numbers (claimed)
| Setting | HPS | Euclidean | Δ |
|---|---|---|---|
| Vicuna-13B same-distribution | AUROC=0.970 | ~0.85 | +0.12 |
| Vicuna-13B cross-attack | AUROC=0.815 | 0.513 | **+0.302** |
| Llama-3-8B same-distribution | TPR=0.841 | — | — |
| Cross-attack mean (Llama-3) | TPR=0.682 | TPR=0.520 | +0.162 |

### Architecture
```
Activations (6 layers × 4096 dim)
       ↓
Lorentz Projection (4096 → 64) with learnable κ, per-layer τ
       ↓
12 trajectory features (radial × 5, curvature × 4, displacement × 3)
       ↓
Logistic Regression
       ↓
Score
```

---

## 2. First Verification Round: Did the Code Match the Claims?

**Question:** Are the published numbers reproducible and methodologically sound?

**Found 5 critical methodology issues:**

| Issue | Where | Effect |
|---|---|---|
| Threshold leakage | hps_llama3.py, paper_supplementary.py, experiment12.py | Threshold calibrated on test benign — TPR inflated |
| Train/test overlap | attack_ensemble.py | 63% of "test" attacks were in training |
| Unfair Euclidean baseline | experiment7.py | Euclidean had 8 fewer learnable parameters |
| No multi-seed reporting | All scripts | Single-run numbers reported as facts |
| Inconsistent training | Vicuna vs Llama-3 pipelines used different epochs/early-stopping | |

### Methodology Fixes Applied

```diff
- threshold = quantile(scores_on_test_benign, 0.95)
+ split test_benign into calibration_half + held_out_half
+ threshold = quantile(scores_on_calibration_half, 0.95)
+ FPR measured on held_out_half

- proj_euclidean: Linear(4096→64) + 1 scalar
+ proj_euclidean: Linear(4096→64) + 7 per-layer scales + 1 learnable margin
  # Now matched parameter count with HPS

- 200 epochs without early stopping
+ 50 epochs (TEST 5 finding: HPS overfits past 50)

- Single seed reported
+ 5 seeds with mean ± std
```

---

## 3. Re-Run with Fixes: First Surprise

After methodology fixes, the headline numbers shifted dramatically:

### Llama-3-8B Same-Distribution (Updated)
| Method | AUROC | TPR | FPR |
|---|---|---|---|
| HPS (new config) | 1.000 | 1.000 | 0.046 |
| Euclidean (parameter-matched) | 0.999 | 0.998 | 0.043 |
| **Δ** | **+0.001** | **+0.002** | — |

The +0.302 gap **disappeared**. With matched parameters, both methods saturate at ~1.000.

### What changed?
The original Euclidean baseline was under-parameterized:
- Old Euclidean: single global scale, fixed margin = 2.0
- HPS: per-layer temperatures, learnable curvature κ
- Parameter count: HPS had 8 more learnable parameters

Once the playing field was leveled, the gap evaporated.

📊 **See:** `results/figs/fig_seed_stability.png` — HPS at 1.000 ± 0.000 across 5 seeds, Euclidean at 0.999 ± 0.000.

---

## 4. The Diagnostic: Where Does Hyperbolic Actually Help?

If hyperbolic doesn't help in same-distribution, **does it help in cross-attack?**

### Cross-attack across N attacks per method

| N/method | HPS TPR | Euclidean TPR | Δ |
|---|---|---|---|
| 25 | 0.996 | **0.867** | +0.129 |
| 50 | 0.998 | 0.989 | +0.009 |
| 100 | 0.996 | 0.971 | +0.024 |
| 250 | 0.997 | 0.978 | +0.019 |
| 500 | 0.997 | 0.992 | +0.005 |

**Insight:** Small advantage at very low data, gap shrinks with more data.

### Pushed extreme low-data:

| N/method | HPS | Euclidean | Δ |
|---|---|---|---|
| **5** | **0.978** | 0.244 | **+0.733** |
| 10 | 0.989 | 0.411 | +0.578 |
| 15 | 0.985 | 0.785 | +0.200 |
| 20 | 0.994 | 0.833 | +0.161 |

### Few-method extreme:

| #methods (N=25) | HPS | Euclidean | Δ |
|---|---|---|---|
| 2 | 0.500 | 0.000 | +0.500 |
| 3 | 0.680 | 0.253 | +0.427 |
| **4** | **0.990** | 0.250 | **+0.740** |
| 5 | 1.000 | 0.608 | +0.392 |
| 9 | 0.996 | 0.867 | +0.129 |

### What the inflection point looks like:

```
Cold-Start Regime:
  N=5 / 9 methods:           HPS=0.978   Euc=0.244    Δ=+0.73 ★
  N=25 / 4 methods (Vicuna): HPS=0.990   Euc=0.250    Δ=+0.74 ★
  Extreme (3 meth × 5):      HPS=0.467   Euc=0.067    Δ=+0.40

Saturation Regime:
  N=500 / 9 methods:         HPS=0.997   Euc=0.992    Δ=+0.005
  Full data:                 HPS=1.000   Euc=0.999    Δ=+0.001
```

📊 **See:** `results/figs/fig_cold_start_curve.png` and `fig_heatmap_diversity.png`

**Provisional conclusion at this point:** Hyperbolic priors help in low-data, low-diversity regimes — replicating the original Vicuna result on Llama-3 under matched conditions. *This was thought to be the headline.*

---

## 5. Feature Ablation: Are All 12 Features Needed?

Tested 8 feature subsets across regimes:

| Subset | #feats | Same-dist | CS N=5 | CS N=25 | Vicuna |
|---|---|---|---|---|---|
| all_12 | 12 | 1.000 | 0.988 | 0.996 | 0.990 |
| radial_disp_8 (no curvature) | 8 | 1.000 | 0.994 | 0.996 | 0.990 |
| top6_byimp | 6 | 1.000 | 0.994 | 0.996 | 0.990 |
| **top1_byimp (mean_r alone)** | **1** | **1.000** | **0.996** | **0.998** | **1.000** |
| curvature_4 only | 4 | 0.995 | 0.970 | 0.984 | 0.850 |

**Surprising finding:** A single scalar (mean radial position) matches all 12 features.

**Implication:** The Lorentz projection compresses discriminative info onto the radial axis. The "trajectory features" framework is overengineered.

📊 **See:** `results/figs/fig_feature_ablation.png` and `fig_feature_importance.png`

---

## 6. Control Experiments: Is the Geometric Prior Doing Real Work?

A skeptical reviewer would ask: "Could this just be detecting something trivial like prompt length or activation magnitude?"

### Tested 5 controls vs HPS (mean_r) at cold-start N=5:

| Control | What it tests | TPR | Δ vs HPS |
|---|---|---|---|
| C1: Raw L2 norm scalar | Activation magnitude alone | 0.321 | -0.676 |
| C2: Per-layer L2 norm | Multi-layer raw norm | 0.658 | -0.338 |
| C3: Random untrained Lorentz | Does training matter? | 0.099 | -0.897 |
| **C4: LR on mean-pooled (4096-dim)** | **Linear probe baseline** | **0.992** | **-0.004** |
| C5: LR on flattened (24576-dim) | High-dim probe | 0.986 | -0.010 |

### Activation magnitude analysis (rules out length artifacts):

```
Layer  | Benign norm  | Attack norm | Δ
─────────────────────────────────────────
0      |  0.60 ± 0.18 |  0.80 ± 0.22 | +0.20
2      |  1.49 ± 0.28 |  1.66 ± 0.29 | +0.17
17     | 10.57 ± 0.68 | 10.04 ± 0.76 | -0.52
24     | 19.39 ± 1.04 | 19.14 ± 1.49 | -0.26
28     | 28.36 ± 1.31 | 28.45 ± 2.04 | +0.09
31     |155.62 ± 1.57 |153.03 ± 4.35 | -2.58
```

**Attacks don't have systematically larger activations.** Rules out trivial OOD/length detection.

### Critical finding from controls:

- ✓ **C1, C2, C3 catastrophically fail** → confirms HPS is doing real work (raw magnitude alone is insufficient, and untrained projection is useless)
- ⚠ **C4 (LR on mean-pooled activations) matches HPS** → a simple linear probe with 4096 features ties HPS's 1-feature pipeline

**At this point, the hyperbolic story weakens significantly.** A linear probe on raw activations achieves the same result.

---

## 7. Adversarial Robustness: Is HPS More Robust to Attack?

If a reviewer asks "why care about HPS if C4 matches?", one defense would be **robustness**. Tested PGD evasion at multiple ε:

### PGD evasion comparison on Llama-3:

| ε | HPS evasion | C4 evasion | Δ |
|---|---|---|---|
| 0.001 | 0.000 | 0.000 | tied |
| 0.005 | 0.000 | 0.000 | tied |
| 0.01 | 0.000 | 0.000 | tied |
| **0.05** | **0.660** | **0.020** | **HPS much worse** |
| 0.1 | 1.000 | 0.350 | HPS much worse |
| 0.5 | 1.000 | 1.000 | both broken |

**HPS is LESS robust than C4** at moderate ε. The single-feature bottleneck is exploitable — PGD can efficiently push `mean_r` across threshold.

📊 **See:** `results/figs/fig_adaptive_pgd.png`

**Robustness story is also dead.** HPS provides no robustness advantage over a simple linear probe.

---

## 8. Cross-Model Replication: Does Vicuna Confirm or Refute?

Final check: does HPS work on Vicuna-13B (where original "+0.302" came from)?

### Same-distribution Vicuna:

| Method | AUROC | TPR |
|---|---|---|
| HPS | 0.936 | 0.746 |
| C4 | **1.000** | **1.000** |
| Δ | -0.064 | -0.254 |

### Cold-start cross-attack (Vicuna, leave-one-out):

| N/method | HPS | C4 | Δ |
|---|---|---|---|
| 5 | 0.340 | 0.933 | -0.593 |
| 10 | 0.286 | 0.963 | -0.677 |
| **25** | **0.068** | **0.985** | **-0.917** |
| 50 | 0.246 | 0.974 | -0.727 |

### PGD on Vicuna at ε=0.05:
- HPS evasion: **0.960**
- C4 evasion: **0.040**

**On Vicuna, HPS is catastrophically worse than C4 in every regime.**

The original "+0.302" claim was an artifact of the unfair Euclidean baseline, NOT a property of Vicuna.

---

## 9. Hyperparameter Rescue Attempt

Tried to rescue HPS on Vicuna by sweeping:
- Layer configurations (5 different)
- κ values (0.1, 0.5, 1.0, 2.0)
- Epoch counts (25, 50, 100)
- Frozen vs learnable κ

### Best HPS config result:
- Best: layers=spread, κ=2.0, frozen, epochs=25
- Best HPS TPR: **0.769**
- C4 TPR: **0.918**
- **Gap: -0.149 (HPS still loses by 15%)**

### Best HPS at multiple N:
| N | HPS (best config) | C4 | Δ |
|---|---|---|---|
| 5 | 0.000 | 0.851 | -0.851 |
| 10 | 0.000 | 0.906 | -0.906 |
| 25 | 0.769 | 0.918 | -0.149 |
| 50 | 0.771 | 0.935 | -0.164 |

**No hyperparameter setting rescues HPS.** The negative result is robust.

---

## 10. Final Verdict

### What the data conclusively shows:

1. **Linear probes (C4) match or exceed sophisticated geometric methods** across:
   - 2 LLMs (Vicuna-13B, Llama-3-8B)
   - 13 attack families total (4 in Vicuna, 9 in Llama-3)
   - 5 evaluation regimes (same-dist, cold-start at N=5/25/100, Vicuna-like)
   - PGD adversarial settings

2. **The "+0.302 hyperbolic AUROC" was a methodology artifact** — disappeared with parameter-matched baselines.

3. **Hyperbolic geometric prior provides no consistent advantage:**
   - On Llama-3: small advantage at extreme low data, ties at full data
   - On Vicuna: catastrophic failure at all data sizes
   - Adversarial: actively worse (single-feature bottleneck exploitable)

4. **What HPS does correctly:**
   - Compresses discriminative information into a single scalar
   - But this is also achieved by a simple LR weight vector
   - Compression is real but not unique

5. **Real findings, beyond hyperbolic:**
   - Layer selection matters more than projection geometry
   - Curvature features are redundant (4 of 12 features add nothing)
   - Single-feature detection is sufficient on current benchmarks
   - Cold-start regime has clear inflection point patterns
   - Single-feature methods are LESS adversarially robust

### Why this happens (mechanistic):

The activation space at safety-critical layers is **highly linearly separable**. Any reasonable method finds the discriminative direction. The "geometric prior" doesn't add information — it just constrains where the boundary can be drawn. With enough capacity, simple linear methods find the right boundary anyway.

### What this means for the field:

**Current jailbreak detection benchmarks are saturated.** A logistic regression with 4096 weights achieves >99% TPR. Sophisticated methods (RTV, JBShield, HPS) are over-engineered for current evaluation setups.

The community needs harder benchmarks:
- Multi-turn conversational attacks
- Agentic / tool-use jailbreaks
- Truly novel attack families (TombRaider, ICE)
- Cross-model robustness without retraining
- Attacks adversarially crafted against the defense

---

## 11. Path Forward

### Two complementary directions

**Direction A: Negative-result paper (4-6 weeks)**

Title: *"Activation-Based Jailbreak Detection Is Solved by Linear Probes: A Rigorous Empirical Study"*

Argument: Sophisticated methods (RTV, JBShield, HPS) provide no advantage over a 1-layer linear probe on mean-pooled activations. The community should redirect to harder problems.

Strengths:
- Rigorous methodology (4 control experiments, multi-seed, parameter-matched)
- Provocative thesis with clear evidence
- Negative results with this rigor are publishable
- Existing data is sufficient (no new experiments)

Target: USENIX Security 2027, NDSS 2027, or AAAI 2027

**Direction B: Multi-turn pivot (2-3 months)**

The activation-based single-turn problem is "solved." But multi-turn jailbreaks (DIA, sequential attacks) remain open. Hyperbolic geometry has natural fit for tree-structured conversations.

Concrete approach:
- Embed conversation states in hyperbolic space
- Track radial position across turns
- Detect navigation toward harmful "leaves"
- Compare to single-shot detectors (which fail on multi-turn by design)

Strengths:
- Genuinely open problem
- Hyperbolic has theoretical justification (tree structures)
- Crowded competition lower than single-shot detection
- Reuses existing tooling (HPS framework)

Target: USENIX Security 2027/2028 or NeurIPS Safety track

### Recommendation

**Do both in parallel.** Start writing the negative paper now using existing data. In parallel, begin small-scale multi-turn experiments. The negative paper reframes the field; the multi-turn paper provides the constructive follow-up.

---

## Methodology Notes for Mentor

What was done unusually rigorously:
- Threshold calibration with held-out split (no leakage)
- Parameter-matched baselines (same param count for fair comparison)
- Multi-seed stability with σ reporting
- Adversarial controls (PGD on activations)
- Activation magnitude analysis (rules out trivial detection)
- Ablation across feature subsets (single-feature comparison)
- Hyperparameter sweeps to rule out tuning artifacts
- Cross-model replication (Vicuna and Llama-3)

What this means: the negative result is *very* robust. Multiple independent lines of evidence converge on the same conclusion.

---

## Files Referenced

| Script | Purpose |
|---|---|
| `experiment7.py` | Vicuna pipeline (HPS-Full + ablation baselines) |
| `hps_llama3.py` | Llama-3 main experiment |
| `paper_supplementary.py` | Multi-seed stability, learning curve, Euclidean comparison |
| `attack_ensemble.py` | Adaptive PGD on HPS+RTV ensemble |
| `experiment12.py` | HPS-Adv adversarial training |
| `diagnostic_hps_vs_euc.py` | Comprehensive hyperparameter / regime diagnostic |
| `verify_new_config.py` | Verification of κ=0.1 + spread layers config |
| `feature_ablation.py` | Subset comparison across regimes |
| `control_experiments.py` | C1–C5 controls |
| `adversarial_compare.py` | HPS vs C4 PGD comparison |
| `cross_model_compare.py` | Vicuna replication |
| `vicuna_param_sweep.py` | Hyperparameter rescue attempt |
| `generate_paper_plots.py` | Paper figure generation |

| Result file | Content |
|---|---|
| `results/hps_vs_rtv_llama3.json` | Llama-3 main results |
| `results/paper_supplementary.json` | Multi-seed stability, learning curve |
| `results/verify_new_config.json` | Cold-start regime characterization |
| `results/feature_ablation.json` | Feature subset comparison |
| `results/control_experiments.json` | Controls C1-C5 |
| `results/adversarial_compare.json` | HPS vs C4 robustness |
| `results/cross_model_compare.json` | Vicuna replication |
| `results/vicuna_param_sweep.json` | Hyperparameter rescue attempt |

| Figure | Content |
|---|---|
| `results/figs/fig_cold_start_curve.png` | Headline: TPR vs N per method |
| `results/figs/fig_heatmap_diversity.png` | 2D: Δ vs (#methods, N per method) |
| `results/figs/fig_per_attack_bars.png` | Cross-attack TPR per method |
| `results/figs/fig_seed_stability.png` | HPS multi-seed stability |
| `results/figs/fig_adaptive_pgd.png` | Adversarial PGD curve |
| `results/figs/fig_feature_importance.png` | Feature importance breakdown |
| `results/figs/fig_feature_ablation.png` | Subset TPR across regimes |

---

## Sources Cited

- [Bailey et al. (2026): Obfuscated Activations Bypass LLM Latent-Space Defenses (ICLR 2026)](https://arxiv.org/abs/2412.09565) — accessed 2026-05-24
- [Derya & Sunar (2026): Revisiting JBShield: Breaking and Rebuilding Representation-Level Jailbreak Defenses (RTV)](https://arxiv.org/abs/2605.03095) — accessed 2026-05-24
- [Zhang et al. (USENIX Security 2025): JBShield](https://arxiv.org/abs/2502.07557) — accessed 2026-05-24
- [Wollschläger et al. (ICML 2025): The Geometry of Refusal in LLMs](https://arxiv.org/abs/2502.17420) — accessed 2026-05-24
- [Maljkovic et al. (ICLR 2026): HyPE — Harnessing Hyperbolic Geometry for Harmful Prompt Detection](https://arxiv.org/abs/2604.06285) — accessed 2026-05-24
- [Wang et al. (Dec 2025): Rethinking Jailbreak Detection of VLMs with Representational Contrastive Scoring](https://arxiv.org/abs/2512.12069) — accessed 2026-05-24 — *closest concurrent work*
- [Peng et al. (Anthropic, 2024): Mitigating LLM Jailbreaks with a Few Examples](https://arxiv.org/abs/2411.07494) — accessed 2026-05-24
- [HSF: Hidden State Filter](https://arxiv.org/abs/2409.03788) — accessed 2026-05-25 — *direct C4-style competitor*
- [AdaSteer (EMNLP 2025): Aligned LLM is an Adaptive Jailbreak Defender](https://arxiv.org/abs/2504.09466) — accessed 2026-05-25
