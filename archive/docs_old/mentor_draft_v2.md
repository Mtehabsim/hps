# HPS Research Draft v2: Adversarial Robustness (Bailey et al. Reference Implementation)

**Author:** [Author Name]
**For:** Mentor review — addendum to Draft v1
**Status:** Draft v2
**Date:** 2026-06-13
**Companion to:** mentor_draft.md (Draft v1)

---

## Abstract

This addendum reports an adversarial-robustness evaluation of HPS against the C4 linear-probe baseline on Llama-3-8B-Instruct, conducted entirely inside the **Bailey et al. (2024) released reference implementation** — the authors' own universal soft-prompt attack and their **StrongREJECT (Gemma-2) jailbreak judge**. Sweeping the behavior/obfuscation trade-off weight λ, an attacker simultaneously achieves ~85% jailbreak success and 0% post-attack detection against **both** C4 (λ=0.1–0.9) and HPS (λ=0.3–1.0). The model is equally jailbreakable under both probes, and HPS is additionally a **weaker clean detector** (recall@1% FPR 0.16–0.60 vs C4's ~1.0; 0% vs 33% on un-obfuscated jailbreaks). **The hyperbolic geometric prior provides no adversarial-robustness advantage over a plain linear probe — and detects somewhat less.** We recommend TMLR and frame the overall work as a methodology study with empirical adversarial-robustness limitations.

---

## 1. Why the reference implementation

The central question for this addendum is whether HPS's hyperbolic geometry buys any adversarial robustness over a simple linear probe (C4). To answer it in a way that is independent of any attack code we might write, we ran the comparison **directly inside the Bailey et al. (2024) released codebase** (`obfuscated-activations/inference_time_experiments`). The attack (a universal soft-prompt / continuous-embedding obfuscation attack), the probe interface, the optimization loop, and the **StrongREJECT jailbreak judge** are all the original authors' implementation. We supply only the two probes. This makes the result reviewer-proof: a skeptic cannot attribute the outcome to a quirk of our attack harness, because there is no attack harness of ours in the loop, and jailbreak success is verified by the authors' judge rather than inferred.

## 2. Setup

- **Model:** Llama-3-8B-Instruct (fp16).
- **Probes:** C4 (per-layer logistic regression on mean activations) and HPS (Lorentz projection + 12 cross-layer trajectory features + logistic head), both reading the same 6 layers (transformer 0 / 6 / 12 / 18 / 24 / 31).
- **Attack:** a single *universal* continuous soft-prompt, trained on harmful training prompts and evaluated on 200 held-out prompts. The attack minimizes `L = λ · behavior + (1 − λ) · obfuscation`, where the behavior term drives jailbreak success and the obfuscation term drives evasion of the probe. We sweep `λ ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0}` (λ=0 → pure evasion; λ=1 → pure jailbreak, no evasion).
- **Metrics:** `jailbreak` = mean StrongREJECT score on the attacked generations; `detection` = probe recall at 1% FPR on the attacked (obfuscated) examples; `clean detection` = recall at 1% FPR with no attack.

## 3. Results

| λ | C4 jailbreak | C4 detection | HPS jailbreak | HPS detection |
|---|---|---|---|---|
| 0.0 | 0.004 | 0.000 | 0.014 | 1.000 |
| 0.1 | 0.833 | 0.000 | 0.871 | 1.000 |
| 0.3 | 0.868 | 0.000 | 0.853 | 0.000 |
| 0.5 | 0.869 | 0.000 | 0.869 | 0.000 |
| 0.7 | 0.850 | 0.000 | 0.863 | 0.000 |
| 0.9 | 0.862 | 0.000 | 0.846 | 0.000 |
| 1.0 | 0.865 | 0.330 | 0.860 | 0.000 |

Clean (no-attack) detection, recall@1% FPR: **C4 ≈ 0.99–1.00; HPS ≈ 0.16–0.60.**

**Findings:**

1. **The model is equally jailbreakable under both probes** — ~0.85 StrongREJECT at λ ≥ 0.1 in both configurations. The attack reaches the same jailbreak ceiling regardless of which probe is present.
2. **C4 is fully bypassed.** For λ = 0.1–0.9 the attacker obtains ~0.85 jailbreak *and* 0.000 detection simultaneously. C4 retains detection (0.33) only at λ=1.0, where the attacker is not trying to evade at all.
3. **HPS is also fully bypassed.** For λ = 0.3–1.0 the attacker obtains ~0.86 jailbreak *and* 0.000 detection. An adaptive attacker simply selects a λ in this range and evades HPS completely while jailbreaking. HPS retains detection only at the most evasion-weighted points (λ = 0.0, 0.1) — operating points an attacker has no incentive to choose.
4. **HPS detects less than C4 overall.** Its clean recall@1% FPR (0.16–0.60) is well below C4's (~1.0), and on un-obfuscated jailbreaks (λ=1.0) HPS catches 0% versus C4's 33%.

## 4. Conclusion

Under the authors' own attack and judge, **HPS provides no adversarial-robustness advantage over the C4 linear probe**: both are fully bypassed by a universal soft-prompt attack that jailbreaks the model while driving probe detection to zero, and HPS is the weaker detector of the two. The hyperbolic geometric prior does not translate into measurable robustness.

## 5. Relation to Draft v1

This result completes the adversarial-robustness picture and is consistent with the clean-detection findings in Draft v1. Across the project, HPS shows no advantage from four independent angles:

1. **Clean detection** — AUROC parity with C4 (~0.99) but *worse* recall at 1% FPR on out-of-distribution harmful inputs.
2. **Layer sensitivity** — HPS is fragile to layer choice; C4 is robust.
3. **Adversarial robustness** — no advantage (this addendum): fully bypassed, and a weaker detector.
4. **Cross-model generality** — HPS catastrophically fails on Vicuna-13B (alignment-mediated), per Draft v1.

The four angles converge on the same conclusion: a hyperbolic geometric prior on activation trajectories does not improve jailbreak detection or its adversarial robustness over a plain linear probe.

## 6. Recommendation and future work

**Venue — TMLR (recommended).** The combined Draft v1 + v2 story is a clean methodology study: (a) field-wide evaluation confounds (length / max-length / contamination) identified and corrected; (b) after correction, the hyperbolic prior shows no detection advantage; (c) it provides no adversarial robustness under the reference attack; (d) it fails to generalize across models. Negative results on a plausible-sounding idea, established carefully, are a useful contribution. ICLR/NeurIPS would require a positive method claim we cannot make; USENIX would require a defense that survives adaptive attack, which HPS does not.

**Scope / limitations to state explicitly.** (i) The adversarial evaluation is the Bailey soft-prompt (continuous-embedding) attack on Llama-3-8B with this jailbreak dataset; we claim "no advantage under this attack/model/data," not universally. (ii) We did not separately attack a flat (HPS-Euclidean) variant inside the reference framework, so we cannot attribute the null result specifically to curvature versus the trajectory-feature architecture — but for the top-line question (does HPS beat the linear baseline?), the answer is negative regardless.

**Optional follow-up — generation-based HPS.** The only principled direction that might show improvement is extending HPS from input-based (last-token / mean-pooled) to *generation*-based features (trajectory across the first K generated tokens). Bailey et al. report that generation-based probes are harder to attack than input-based ones. However, a fair test would have to beat a generation-based *C4* as well — so the likely finding is that the trajectory/generation structure (not the hyperbolic geometry) is what matters. Best scoped as a separate follow-up, not a blocker for the current negative-results paper.

---

## Appendix: References

[1] Bailey et al. (2024). "Obfuscated Activations Bypass LLM Latent-Space Defenses." Reference implementation used for the attack and StrongREJECT judge in this addendum.

[2] Schwinn & Geisler (2024). "Soft Prompts for Embedding-Space Attacks Against Representation Engineering." arXiv:2402.09063. Independent evidence that activation-space defenses are vulnerable to embedding-space attacks.

[3] Li et al. (2024). Multi-turn attacks against RepE / LAT / Circuit Breakers.

[4] Carlini et al. (2024). Foundational adversarial-alignment results.

---

**End of Draft v2**

**Key finding:** In the Bailey et al. reference implementation, both C4 and HPS are fully bypassed (~0.85 jailbreak with 0.000 detection); HPS is the weaker clean detector. The hyperbolic prior provides no adversarial-robustness advantage.

**Status:** Validated negative result complete. Ready for mentor review and TMLR draft writing.

**Open questions for mentor:**
1. Final paper framing: methodology study with adversarial-robustness limitations (recommended)?
2. Cross-model generality is already evidenced by HPS's failure on Vicuna-13B (Draft v1); an additional model is judged unnecessary — agree?
3. Generation-based HPS — explicit follow-up paper, or a short discussion section in the current paper?
