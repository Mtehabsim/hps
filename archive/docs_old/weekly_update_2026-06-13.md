# Weekly Update — HPS Adversarial Robustness

**Date:** 2026-06-13 (week following the last meeting)
**Focus:** HPS vs. C4 adversarial robustness in the Bailey et al. reference implementation

---

## Goal this week

Test whether the hyperbolic probe (HPS) offers any adversarial-robustness advantage over the C4 linear-probe baseline, evaluated inside the **authors' own released attack framework** (Bailey et al., 2024) — using their attack and a verified jailbreak judge, so the evaluation does not depend on any attack code we wrote.

## What we did

- Integrated **both probes** — C4 (per-layer logistic regression on mean activations) and HPS (Lorentz projection + 12 cross-layer trajectory features + logistic head) — into the Bailey reference codebase, reading the same 6 layers of Llama-3-8B-Instruct.
- Used the reference framework end-to-end: the authors' **universal soft-prompt attack** and their **StrongREJECT (Gemma-2) jailbreak judge**. We supply only the probes.
- Ran a **trade-off sweep** over the attack objective `L = λ·behavior + (1−λ)·obfuscation`, λ ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0}, for both probes (universal attack trained on harmful prompts, evaluated on 200 held-out prompts).

## Results

| λ | C4 jailbreak | C4 detection | HPS jailbreak | HPS detection |
|---|---|---|---|---|
| 0.0 | 0.004 | 0.000 | 0.014 | 1.000 |
| 0.1 | 0.833 | 0.000 | 0.871 | 1.000 |
| 0.3 | 0.868 | 0.000 | 0.853 | 0.000 |
| 0.5 | 0.869 | 0.000 | 0.869 | 0.000 |
| 0.7 | 0.850 | 0.000 | 0.863 | 0.000 |
| 0.9 | 0.862 | 0.000 | 0.846 | 0.000 |
| 1.0 | 0.865 | 0.330 | 0.860 | 0.000 |

*jailbreak = mean StrongREJECT; detection = probe recall @ 1% FPR on attacked examples. Clean (no-attack) recall @ 1% FPR: C4 ≈ 0.99–1.00, HPS ≈ 0.16–0.60.*

**Key observations:**
1. **The model is equally jailbreakable under both probes** (~0.85 StrongREJECT at λ ≥ 0.1).
2. **C4 is fully bypassed** at λ = 0.1–0.9 (jailbreak ~0.85 *and* 0 detection); it only retains detection (0.33) at λ = 1.0, where the attacker isn't trying to evade.
3. **HPS is also fully bypassed** at λ = 0.3–1.0 (jailbreak ~0.86 *and* 0 detection). An adaptive attacker simply picks a λ in this range. HPS retains detection only at the most evasion-weighted points (λ = 0, 0.1), which an attacker has no reason to use.
4. **HPS is the weaker clean detector** — recall @ 1% FPR 0.16–0.60 vs C4's ~1.0, and 0% vs C4's 33% on un-obfuscated jailbreaks.

## Conclusion

Under the authors' own attack and judge, **HPS provides no adversarial-robustness advantage over a plain linear probe — and detects somewhat less.**

This is consistent with the rest of the project: detection parity in AUROC but worse at low FPR, layer fragility, and cross-model failure (Vicuna). Together they give a coherent negative result across four angles — clean detection, low-FPR detection, adversarial robustness, and cross-model generality.

## Recommendation / next steps

- **Finalize as a negative-findings methodology paper (TMLR).** The result is complete and multi-angle; no further experiments are required to support the conclusion.
- Updated `mentor_draft_v2.md` with the reference-implementation result as the headline and reconciled the draft into one coherent story.
- Optional, low priority: a generation-based HPS variant remains the only principled direction that *might* show improvement, but it would most likely demonstrate that trajectory/generation structure (not hyperbolic geometry) is what matters — best scoped as a separate follow-up.
