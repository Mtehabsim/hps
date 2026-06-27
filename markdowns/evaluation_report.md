# Research Evaluation: HPS — Activation-Based Jailbreak Detection

**Evaluator Assessment Date:** 2026-05-25  
**Document Evaluated:** `research_journey.md` + code + results + plots  
**Verdict:** Publishable with revisions. Strong empirical work with honest negative findings. Needs tightening for submission.

---

## Executive Assessment

This is a rigorous empirical study that started as a "hyperbolic geometry helps jailbreak detection" paper and evolved into a "simple baselines match sophisticated methods" paper. The intellectual honesty is commendable — the authors discovered their headline claim was artifactual and pivoted to reporting the negative finding constructively. The methodology is sound, the controls are thorough, and the experimental coverage (2 LLMs, 9 attack families, multi-seed, adversarial robustness) exceeds most published work in this area.

**Publishability:** TMLR (65-70%), NeurIPS Safety Workshop (55-65%), ACSAC/RAID (40-50%). Not suitable for top-tier ML venues (ICML/NeurIPS main) in current form.

---

## CRITICAL Issues (Must Address Before Submission)

### C1. The Paper Draft Is Stale and Makes False Claims

The existing `paper_draft.md` claims "+0.302 AUROC hyperbolic advantage" and "AUROC=0.970" as headline results. These are **the exact claims the research journey document proves are artifactual**. The draft must be completely rewritten with the honest framing described in `research_journey.md` Section 10b. Submitting the current draft would be scientific misconduct.

**Action:** Full rewrite with "Reconsidering Activation-Based Jailbreak Detection" framing. The paper_draft.md is unusable.

### C2. Critical Missing Comparison: Anthropic's "Cheap Monitors" (2025)

The Anthropic paper "Cost-Effective Constitutional Classifiers via Representation Re-use" (Cunningham et al., 2025) does **exactly** what C4 does — linear probes on intermediate activations for jailbreak detection — and finds the same thing: probes match or exceed dedicated classifiers at negligible cost. This is a direct competitor that:
- Uses mean-token probes (identical to C4's mean-pooling)
- Uses EMA probes (a variant C4 doesn't test)
- Demonstrates the approach works at production scale on Claude models
- Explicitly acknowledges the adaptive robustness gap (same limitation as C4)

**The C4 "novelty" claim is weakened significantly.** The research_journey.md states "no published paper does exactly C4" — but Anthropic's work does essentially the same thing, just on a different model family and attack set. The paper must cite this and reposition: C4's contribution is not the recipe itself but the **controlled comparison against geometric methods on a standardized jailbreak benchmark**.

**Action:** Add Anthropic cheap monitors to related work. Reframe C4 as "we applied the known linear-probe approach to the specific jailbreak detection benchmark and showed it matches geometric methods" rather than "we invented C4."

### C3. The "When Benchmarks Lie" Paper (OpenReview, 2025) Directly Contradicts C4's Generalization Claims

The OpenReview paper "When Benchmarks Lie: Evaluating Malicious Prompt Classifiers Under True Distribution Shift" trains activation-based classifiers (linear probes on LLM hidden states) and finds they **fail to generalize under true distribution shift**. Similarly, "What Features in Prompts Jailbreak LLMs?" (arXiv:2411.03343) finds linear probes "fail to generalize to unseen attack methods."

This directly challenges C4's cross-attack TPR=0.992 claim. The discrepancy needs explanation:
- Is C4's cross-attack evaluation truly out-of-distribution? (Leave-one-out with 8 training families is generous)
- Would C4 generalize to genuinely novel attack paradigms (multi-turn, agentic)?
- Is the benchmark saturated such that even simple methods work?

**Action:** Add a "Limitations and Threats to Validity" section addressing this. Acknowledge that C4's cross-attack success may reflect benchmark saturation rather than genuine generalization. Cite both papers.

### C4. Statistical Rigor: No Confidence Intervals on Key Claims

The cold-start comparison (HPS=0.996 vs C4=0.992, Δ=0.004) is reported without confidence intervals. With only 3 seeds for cross-attack evaluation, this difference is almost certainly within noise. The paper needs:
- Bootstrap confidence intervals on all TPR comparisons
- A formal statistical test (paired t-test or Wilcoxon) for the HPS vs C4 comparison
- Explicit statement: "the difference is not statistically significant at α=0.05"

The Vicuna results (HPS=0.068 at N=25) are dramatic enough to not need CIs, but the Llama-3 "tie" claims do.

**Action:** Add CIs to all key comparisons. Run paired bootstrap on HPS vs C4 at each regime.

### C5. Adversarial Evaluation Is Unrealistic

The PGD attack operates on the **activation tensor directly** (L∞ perturbation on hidden states). This is not a realistic threat model — an attacker cannot directly perturb internal activations. A realistic adaptive attack would:
- Optimize the input prompt to minimize detection score (prompt-space PGD)
- Use the approach from Bailey et al. (2026) — obfuscated activations via gradient-based prompt optimization
- Test against GCG-style suffix attacks specifically crafted to evade the detector

The current "PGD on activations" evaluation is a useful ablation for understanding robustness properties, but it should NOT be presented as "adversarial robustness evaluation" without qualification. Reviewers familiar with Bailey et al. will flag this immediately.

**Action:** Rename to "Activation-Space Perturbation Analysis" or similar. Add a paragraph acknowledging this is not a realistic threat model. Ideally, add one realistic adaptive attack (GCG optimized against the detector).

### C6. Radial Distribution Plot Contradicts the Hypothesis

The radial distribution plot shows **benign prompts at HIGHER radial position** (median=3.71) than attacks (median=3.24). This is the **opposite** of the stated hypothesis ("adversarial prompts get pushed to high radial position — they're extreme"). The contrastive training has learned to push benign prompts outward and attacks inward.

This means the "hierarchical structure" interpretation is wrong — the projection is just learning an arbitrary discriminative direction, not capturing semantic hierarchy. The paper must:
- Acknowledge this contradicts the original geometric motivation
- Reinterpret: the contrastive loss finds a separating direction; the Lorentz geometry constrains it to be radial
- This further supports the "geometry doesn't matter" conclusion (C4 finds the same direction without geometric constraints)

**Action:** Discuss this honestly in the paper. It's actually a strong piece of evidence for the negative finding.

---

## NORMAL Issues (Adds Delivery Quality)

### N1. Poincaré Disk Visualization Is Misleading

The Poincaré disk plot shows benign (green) points **further from center** (toward boundary) and attacks (purple) **closer to center**. The title says "Benign near center, attacks toward boundary" — this is **backwards** from what the plot shows. Either the title is wrong or the color legend is swapped.

Additionally, the trajectory plot shows trajectories clustered in a narrow band — not using the full disk. This undermines the "exponential volume growth helps separation" argument since both classes occupy a similar radial band.

**Action:** Fix the title/legend inconsistency. Consider whether this visualization helps or hurts the paper's argument.

### N2. Layer Selection Methodology Needs Justification

The "spread layers" [0, 2, 17, 24, 28, 31] were discovered via diagnostic experiments (TEST 7). But C4 also uses these same layers. The comparison is only fair if C4's performance is also reported with different layer selections. The results show C4 is "robust to layer choice" — this should be demonstrated with a table.

**Action:** Add a small ablation showing C4 performance across the same layer configurations tested for HPS.

### N3. Vicuna Results Need More Context

HPS's catastrophic failure on Vicuna (TPR=0.068 at N=25 cold-start) is dramatic but unexplained mechanistically. Why does the same architecture work on Llama-3 but fail on Vicuna? Possible explanations:
- Vicuna's activation space has different geometry (less hierarchical?)
- The contrastive loss converges to a bad local minimum on Vicuna
- Layer selection [0, 2, 22, 31, 35, 39] is suboptimal for Vicuna

Without mechanistic explanation, a reviewer might dismiss this as a hyperparameter issue rather than a fundamental limitation.

**Action:** Add analysis of WHY HPS fails on Vicuna. Compare activation distributions, check if the projection converges, examine per-layer discriminability.

### N4. Missing Comparison to HSF (WWW 2025 Best Paper)

HSF (Hidden State Filter, Qian et al. WWW 2025) is described as "closest in spirit" to C4 but never directly compared. Since HSF won best paper at WWW 2025, reviewers will expect a comparison. Even if you can't reproduce their exact setup, you should:
- Describe the architectural differences clearly
- Explain why a direct comparison wasn't possible
- Discuss whether HSF's "plugin module" is equivalent to C4's logistic regression

**Action:** Add a detailed comparison paragraph in Related Work explaining the HSF relationship.

### N5. The "9 Attack Families" Benchmark Is Not Standard

The attack set (autodan, base64, drattack, gcg, ijp, pair, puzzler, saa, zulu) is custom-assembled. There's no standard jailbreak detection benchmark. This makes cross-paper comparison difficult. The paper should:
- Clearly document the data source and size per attack
- Acknowledge this is not a standardized benchmark
- Discuss how results might differ on other benchmarks (JailbreakBench, HarmBench)

**Action:** Add a data section with per-attack statistics. Acknowledge benchmark limitations.

### N6. Code Quality and Reproducibility

The code is well-structured with clear docstrings and proper train/test splits. However:
- No `requirements.txt` with pinned versions (only a generic one exists)
- No Docker/environment specification
- The `run_all.sh` depends on specific data files not included in the repo
- Activation caches are model-specific and not portable

**Action:** Add a reproducibility section to the paper. Consider releasing a minimal reproduction script.

---

## OPTIONAL Issues (Improves But Doesn't Impact Core)

### O1. The Research Journey Document Is Excellent But Not a Paper

The `research_journey.md` is one of the most honest and thorough research narratives I've seen. It would make an excellent supplementary material or blog post. However, it's 80KB and reads as a lab notebook, not a paper. The actual paper needs to be 8-12 pages with tight structure.

### O2. Feature Importance Plot Contradicts Feature Ablation

The feature importance plot shows `progress` as the most important feature (coefficient ~1.8), followed by `mean_r` (~1.6). But the feature ablation shows `mean_r` alone matches all 12 features. This apparent contradiction (highest-coefficient feature isn't the most important in ablation) should be explained — likely due to multicollinearity between `progress` and `mean_r`.

**Action:** Add a brief note explaining this in the paper.

### O3. The Ensemble (HPS+RTV) Adds Little

The ensemble matches HPS alone on Llama-3 (both at 1.000). On the per-attack breakdown, RTV uniquely helps on SAA (99.4% vs HPS 71.8% under early config), but with the new config HPS already gets 100% on SAA. The ensemble contribution is marginal and adds complexity to the paper without strengthening the story.

**Action:** Consider moving ensemble results to supplementary material.

### O4. Curvature κ Sensitivity Is Under-Explored

The finding that κ=0.1 is optimal on Llama-3 but κ=2.0 is optimal on Vicuna is interesting but unexplained. This suggests the "geometric prior" is actually just a regularization effect (smaller κ = stronger curvature = more regularization). If true, this further undermines the "hyperbolic geometry captures hierarchical structure" narrative.

**Action:** Add a brief discussion of κ as implicit regularization.

### O5. Missing Error Analysis

No qualitative analysis of failure cases. Which specific prompts does HPS miss? Which does C4 miss? Are they the same prompts? A Venn diagram of errors would strengthen the paper.

### O6. Plots Need Publication Quality

The current plots (matplotlib defaults) are adequate for a lab notebook but need polish for publication:
- Consistent color scheme across all figures
- Larger fonts for readability
- Remove redundant titles (use captions instead)
- The Poincaré disk plots waste 80% of the figure area on empty space

---

## Novelty Assessment

| Claim | Novelty | Strength |
|-------|---------|----------|
| HPS framework (Lorentz projection + trajectory features) | Genuine | Moderate — architecturally new but empirically doesn't outperform simple baseline |
| C4 baseline finding | **Weakened** by Anthropic cheap monitors paper | Still valuable as controlled comparison on jailbreak benchmark |
| Cold-start regime evaluation methodology | Genuine | Strong — no prior work evaluates this regime systematically |
| Threshold-leakage correction protocol | Genuine | Moderate — good methodology contribution |
| "Benchmarks are saturating" observation | Genuine | Strong — important field-level finding |
| Vicuna fragility documentation | Genuine | Moderate — useful negative finding |
| Single-feature sufficiency (mean_r) | Genuine | Strong — surprising and actionable |

## Robustness Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Experimental methodology | ★★★★☆ | Excellent controls, multi-seed, parameter-matched. Missing CIs. |
| Statistical rigor | ★★★☆☆ | No formal hypothesis tests, no CIs on key claims |
| Reproducibility | ★★★☆☆ | Code available but environment not fully specified |
| Threat model realism | ★★☆☆☆ | PGD on activations is unrealistic; no prompt-space adaptive attack |
| Cross-model generalization | ★★★★☆ | Two models tested; Vicuna failure honestly reported |
| Benchmark coverage | ★★★☆☆ | 9 attacks is good but non-standard; no HarmBench/JailbreakBench |
| Literature coverage | ★★★☆☆ | Missing Anthropic cheap monitors, OpenReview benchmarks paper |

## Publishability Verdict

### For TMLR (recommended target):
**Probability: 60-65%** (down from the 70% self-assessment due to C2 and C3 issues)

Strengths: Rigorous methodology, honest reporting, clear contribution (cold-start evaluation, benchmark saturation observation). TMLR values rigor over novelty.

Weaknesses: C4 novelty weakened by Anthropic work. Negative result framing is harder to sell. Need to clearly articulate what the reader learns that they didn't know before.

### For NeurIPS Safety Workshop:
**Probability: 50-60%**

Good fit for the community. The "benchmarks are saturating" message is timely. Workshop papers are shorter and can focus on the key finding.

### For USENIX Security:
**Probability: 20-30%**

The "rethinking baselines" framing could work but needs a much sharper security angle. USENIX reviewers will want: (1) realistic adaptive attacks, (2) deployment considerations, (3) clear security implications. Current work is too ML-focused for a security venue.

---

## Recommended Revision Priority

1. **Rewrite paper from scratch** using research_journey.md Section 10b framing (Critical)
2. **Add Anthropic cheap monitors comparison** (Critical)
3. **Add confidence intervals** to all key comparisons (Critical)
4. **Fix radial distribution interpretation** — acknowledge it contradicts hypothesis (Critical)
5. **Reframe adversarial evaluation** as activation-space analysis, not realistic threat model (Critical)
6. **Add "When Benchmarks Lie" discussion** to limitations (Critical)
7. Fix Poincaré disk title/legend (Normal)
8. Add Vicuna failure mechanism analysis (Normal)
9. Add HSF comparison paragraph (Normal)
10. Polish figures for publication (Optional)

---

## Summary for Preprint Preparation

When you ask me to write the preprint, I will:
- Use the "Reconsidering Activation-Based Jailbreak Detection" framing
- Lead with C4 finding as the headline (simple baseline matches sophisticated methods)
- Present HPS as the controlled experiment that makes C4's success interesting
- Honestly report the negative finding (geometry doesn't help at saturation)
- Highlight methodology contributions (cold-start evaluation, threshold-leakage protocol)
- Address all Critical issues above
- Target 10-12 pages (TMLR format)
- Include the Anthropic comparison and "When Benchmarks Lie" discussion

The core story is: **"We built the sophisticated geometric defense the field would build, ran the rigorous comparison the field hadn't done, and found a 4097-parameter linear probe matches it. This exposes a gap in published defense evaluations and suggests current benchmarks are saturating."**

---

## Sources

- [Anthropic: Cost-Effective Constitutional Classifiers via Representation Re-use](https://alignment.anthropic.com/2025/cheap-monitors/) — accessed 2026-05-25
- [When Benchmarks Lie: Evaluating Malicious Prompt Classifiers Under True Distribution Shift](https://openreview.net/forum?id=jWIOJOQqne) — accessed 2026-05-25
- [What Features in Prompts Jailbreak LLMs? Investigating the Mechanisms Behind Attacks](https://arxiv.org/abs/2411.03343) — accessed 2026-05-25
- [JBShield: Defending LLMs from Jailbreak Attacks (USENIX Security 2025)](https://arxiv.org/abs/2502.07557) — accessed 2026-05-25
- [RTV: Revisiting JBShield (Derya & Sunar, 2026)](https://arxiv.org/abs/2605.03095) — accessed 2026-05-25
- [Bailey et al.: Obfuscated Activations Bypass LLM Latent-Space Defenses (ICLR 2026)](https://arxiv.org/abs/2412.09565) — accessed 2026-05-25
- [HyPE: Hyperbolic Geometry for Harmful Prompt Detection (ICLR 2026)](https://arxiv.org/abs/2604.06285) — accessed 2026-05-25
- [Wollschläger et al.: The Geometry of Refusal in LLMs (ICML 2025)](https://arxiv.org/abs/2502.17420) — accessed 2026-05-25
- ⚠️ External link — [Lucic et al.: Are GANs Created Equal? (NeurIPS 2018)](https://arxiv.org/abs/1711.10337) — accessed 2026-05-25
- ⚠️ External link — [Wang et al.: A Frustratingly Simple Baseline for Few-Shot Image Classification (2019)](https://arxiv.org/abs/1909.02729) — accessed 2026-05-25
