# Reconciliation Memo: Earlier Project Findings vs the Current Preprint

**Purpose.** Before merging earlier results into `hps_preprint.md` (the "honest negative" paper), classify every prior finding as **VALID** (merge), **SUPERSEDED** (do not merge; corrected later), or **NEEDS-RECHECK** (conflicting across the timeline — resolve before using). Prevents importing a confounded/stale number into the negative-result paper.

**Two framings of the same negative (context).**
- **Earlier arc (May 2026):** "geometric vs linear jailbreak *detection* across 2 LLMs / 9 attack families + confound methodology + cold-start," non-adaptive + activation-space PGD threat model. Authoritative doc: `research_journey.md` (+ `evaluation_report.md` review).
- **Current arc (June 2026, this session):** "does hyperbolic geometry help on Llama-3 under the *realistic* `obf_reps` adaptive threat model," + calibrated geometry measurement + hierarchical reframe + curvature sweep. Authoritative doc: `hps_project_reference.md`. This is the **most recent** and supersedes the earlier arc on threat model and geometry-measurement questions.

The current preprint reflects only the June arc. The memo below recovers the May arc's still-valid contributions.

---

## 1. Classification table

| # | Finding (with key numbers) | Source | Status | Action |
|---|---|---|---|---|
| F1 | **Original "+0.302 AUROC hyperbolic advantage" was ARTIFACTUAL** — under-parameterized Euclidean baseline; after parameter-matching HPS ≈ HPS-Euclidean | research_journey, evaluation_report C1, mentor_briefing | **SUPERSEDED (as a claim) → but the *pivot itself* is VALID and essential** | Merge as the **motivating narrative** ("we expected a win; it dissolved under controls"). Do NOT state +0.302 as a result. |
| F2 | **HPS ≈ C4 at saturation (May JBShield/diverse-fixed cache), C4 significantly ≥ HPS**: AUROC HPS 0.9973 vs C4 0.9981; paired-bootstrap **AUROC p=0.036 (SIGNIFICANT, C4≥HPS)**, McNemar p=0.755 (ns), TPR p=0.601 (ns) | research_journey §1 → **corrected against `statistical_tests.json`** | **VALID** | **Merge the JSON values.** NOTE: research_journey.md mis-stated these as p=0.082/McNemar 0.053/d=0.0148 — those are **phantom** (do not use). Real stats *strengthen* the negative (C4 sig. ≥ HPS). |
| F3 | **Cross-attack leave-one-out (Llama-3):** HPS 0.997, C4 0.992, RTV 0.549 | research_journey §2 | **VALID but caveated** (leave-one-out ≠ true OOD) | Merge with the saturation/OOD caveat. |
| F4 | **Cold-start regime (Llama-3):** HPS narrow edge over *Euclidean-proto* at low N (N=5: 0.978 vs 0.244) **but C4 also high (0.996)** | research_journey §3 | **VALID** | Merge. Note the Euclidean-proto low-N collapse echoes the **single-benign-prototype artifact** found in the June arc (Q5c) — same mechanism. |
| F7 | **Length confound:** length-only AUROC 0.973 → 0.318 after diverse-benign re-extraction | mentor_briefing Update 1; verify_saturation | **VALID** (methodology contribution) | **Merge** as a methodology finding (parallels the curvature-stats caveat). |
| F8 | **Norm confound:** norm-only AUROC ~1.0 → 0.761 after `max_length`/chat-template fix; `norm_controlled_eval` verdict = **REAL signal exists beyond norm** | mentor_briefing Update 2; norm_controlled_eval.py; PROJECT_INVENTORY | **VALID** (methodology contribution, with resolution) | **Merge** — and state the resolution (C4 keeps signal after L2-norm), else it reads as "all artifact." |
| F9 | **Radial distribution** — *conflicting across timeline*: (a) inverted/benign-higher (contradicts hypothesis, 13/13 configs, research_journey); (b) "attacks higher radial after length-fix → confirmed" (mentor_briefing); (c) June arc Q5b viz: "separation is **angular**, radius doesn't separate" | research_journey, mentor_briefing, hps_project_reference Q5b | **NEEDS-RECHECK** | **Do NOT merge** until resolved. Likely resolution: radius is confound/training-direction-dependent; angular is the stable story. Re-run on the canonical cache. |
| F10 | **δ-hyperbolicity per-layer** used as evidence of "tree-likeness" | research_journey §3b | **SUPERSEDED** | Do NOT use δ as evidence — the June arc showed δ is **unreliable at LLM dim** (fails calibration). Mention only as "method we later discarded." |
| F11 | **Activation-space PGD robustness** results | research_journey §5.3; evaluation_report C5 | **SUPERSEDED** | Replaced by the realistic `obf_reps` soft-prompt/GCG adaptive results (current preprint §5.2–5.3). Do NOT merge PGD numbers. |
| F12 | **JBShield reproduction gap:** our repro 0.55 acc vs their published 0.94 | research_journey | **VALID** (transparency note) | Merge as a caveat + the "compare-via-published-numbers" policy. |
| F13 | **SOTA/related work:** RTV (cross-attack 0.549), JBShield (~0.94 published / 0.55 repro), **Anthropic Cheap Monitors**, Google DeepMind Gemini probes, GradSafe, SALO, HyPE | research_journey, paper_draft §2, evaluation_report C2 | **VALID** | Merge into Related Work; **C4 is acknowledged not-novel** — a controlled comparator, not a contribution. |
| F14 | **Benchmark saturation / true-OOD caveat:** "When Benchmarks Lie" (+8.4pp OOD overestimate), "What Features Jailbreak LLMs" (probes fail OOD) | research_journey Limitations; evaluation_report C3 | **VALID** | Merge into Limitations. |
| F15 | **Multi-seed σ=0.000, bootstrap CIs (n=10k)** | research_journey §1 | **VALID** | Merge (rigor). |
| ~~F5, F6, F16~~ | **Vicuna / cross-model / fragility** (Vicuna-GCG collapse, κ non-transfer, 2-LLM breadth) | research_journey §3b | **DROPPED per author decision** | **Do NOT merge.** Paper is single-model (Llama-3); accept the "single model" limitation. |

---

## 2. Conflicts / decisions to resolve before merging

1. **Radial (F9) — the one real conflict.** Three different verdicts exist across time. *Recommended:* treat the **angular-separation / radius-doesn't-separate** (June arc) as current, explain the earlier "inversion" and "confirmed-after-length-fix" as confound-dependent training artifacts, and **re-run `radial_distribution_check.py` on the canonical fixed cache** to lock one story. Until then, omit radial as a positive claim.
2. **δ-hyperbolicity (F10) and PGD (F11)** are *superseded* — the preprint already uses the better replacements (calibrated embedding distortion; `obf_reps` adaptive). Keep them out except as "methods we discarded and why."
3. **C4 novelty (F13).** The preprint implicitly treats C4 as "the strong linear baseline." Must add that linear activation probes are **established prior art** (Anthropic/DeepMind) — C4 is a *comparator*, not a contribution. Avoids an overclaim a reviewer will catch.
4. **Two threat models.** Be explicit: earlier non-adaptive + activation-PGD → *replaced* by `obf_reps` input-space adaptive. The cross-model/cold-start/confound results are about **clean detection**; the adaptive conclusions come from the June arc.

---

## 3. Recommended merge plan into `hps_preprint.md`

Add/extend these sections with **VALID** items only:

- **§1 Introduction → add the pivot (F1):** "We set out to confirm a hyperbolic advantage; under proper controls it dissolved" — frames the negative honestly and motivates the rigor.
- **§4/§5 Clean detection → add F2, F3, F15:** HPS≈C4 with full statistics (p=0.082, McNemar 0.053, d=0.015), cross-attack LOO (Llama-3), multi-seed/bootstrap.
- **New §5.x Cold-start → F4** (with the Euclidean-proto↔single-benign-blob connection to Q5c).
- **§5.4 Methodology → add F7, F8** alongside the curvature-stats caveat: **three** measurement pitfalls (length confound, norm confound, high-D curvature artifacts) → a unified "how activation-defense benchmarks mislead" contribution.
- **§2 Related Work → F13** (RTV, JBShield, Anthropic/DeepMind probes, GradSafe, SALO, HyPE); C4 explicitly **not novel** (comparator).
- **§8 Limitations → F12, F14**; keep "single model (Llama-3)" as an honest limitation (Vicuna dropped by author decision).
- **Hold:** F9 (radial) until rechecked; **exclude:** F5/F6/F16 (Vicuna/cross-model), F10 (δ), F11 (PGD) except δ/PGD as discarded methods.

This upgrades the preprint from "clean Llama-3 slice" to the comprehensive negative the `evaluation_report` says is publishable (TMLR ~60–65%): *comprehensive comparison + cold-start methodology + multi-LLM fragility + multi-confound methodology*.

---

## 4. Recheck checklist (do before importing the numbers)

- [ ] **F9 radial:** re-run `radial_distribution_check.py` on the canonical fixed cache; record final verdict (expect: angular separates, radius doesn't).
- [ ] **F2/F15 stats:** confirm `statistical_tests.json` reflects the current cache (p=0.082, McNemar 0.053).
- [ ] **F7/F8 confounds:** confirm `verify_saturation_fixed.json` and `norm_controlled_eval_*.json` numbers (0.973→0.318; ~1.0→0.761; REAL_SIGNAL verdict).
- [ ] **Cache provenance:** confirm the **June-arc** results (embedding distortion, hierarchical, curvature) were computed on the **confound-fixed canonical** cache, not an older one — else length/norm confounds may still contaminate them.
- [ ] **F13 baselines:** lock published-vs-reproduced policy (cite published; footnote the 0.55 JBShield repro).

---

## 5. What I am genuinely suspicious about (epistemic doubts, ranked)

Beyond the formal status above, these are the things I would not fully trust without more checking — ordered by how much they threaten the core negative.

1. **Is C4 detecting *harm*, or just *dataset/source/topic* differences? (biggest doubt.)** Clean AUROC ≈ 1.0 and clusters ~99% pure is *suspiciously easy*. The length confound (0.973→0.318) and norm confound (~1.0→0.761) were already two instances of "the probe latches onto a superficial artifact, not harm." Even after those fixes, if harmful prompts come from one source (e.g., AdvBench) and benign from another (e.g., Alpaca), the separation may be **distribution/topic separability**, not "harmfulness." This directly affects the central claim "the harm decision is linearly separable." *Resolve:* matched-distribution benign, the `norm_controlled_eval` REAL_SIGNAL verdict, and a topic/format-controlled benign set.

2. **Cache/confound provenance of the June arc.** The embedding-distortion, hierarchical, and curvature-sweep results may have run on `llama3_activations_cache_alllayers.npz`, **not** the confound-fixed canonical cache. If so, the length/norm confounds could still be lurking in those geometry results. *Resolve:* confirm which cache each June script used.

3. **Radial (F9).** Genuinely conflicting verdicts across the timeline. I don't currently know which is true. Flagged NEEDS-RECHECK.

4. **Claims we haven't actually run.** The **HHD adaptive** "no advantage" is *predicted*, not measured (detector built, attack not run). The **GCG** arm didn't jailbreak, so we have *no valid* hard-token adaptive result. Anything stated about these is inference, not evidence.

5. **Noisy metrics under some June conclusions.** (a) Curvature-sweep **typed_acc ≈ 0.70–0.75** rests on **LLM-generated category labels (~81% coverage)** — noisy; the monotonic-with-curvature trend is probably real but the absolute numbers are soft. (b) The sweep's **binary** arm used the **single-benign-prototype** (confounded) setup. (c) **Harm reps at 0.53× the random floor** is a *weak* margin — calling them "mildly hyperbolic" may be over-reading near-noise. The strong embedding-distortion result is only the **token subspace (0.02× floor)**.

6. **"rpf ≡ rpfnox0 ⇒ hyperbolic inert" — check the boring explanation.** If the Lorentz x₀ coordinate is **near-constant** across activations (because norms are similar), then adding it is trivially inert — informative, but for a mundane reason (constant feature), not a deep geometric one. *Resolve:* confirm x₀ actually varies across examples before interpreting "inert" as a geometric statement.

7. **Gradient-masking completeness.** The causal softplus proof is at **λ=0.1** (one point). "No advantage at any λ" leans on the sweep + the inert-lift controls. Solid, but the un-masking is so total (obf_auc 1.0→0.001) that I'd want the softplus sweep across all λ to be sure nothing else is going on.

**Net:** the *adaptive* negative (all probes bypassed; hyperbolic lift inert) and the *gradient-masking* finding are the most solid. The shakiest load-bearing assumption is **#1 — that ceiling-level clean separability reflects harm rather than dataset artifact** — because if separation is artifactual, "linearly separable harm" (the mechanism for the whole negative) needs rewording to "linearly separable *datasets*." That doesn't flip the "hyperbolic doesn't help" conclusion, but it changes the *explanation*.

---

## 6. External-audit corrections (verified against `results/*.json`, 2026-06-21)

An external audit flagged several mis-stated numbers; I re-verified each against the on-disk JSONs. **All confirmed; all either neutral or *strengthen* the negative.**

1. **Phantom stats** (see F2 above): real `statistical_tests.json` = AUROC **p=0.036 (C4 significantly ≥ HPS)**, McNemar 0.755, TPR 0.601. The 0.082/0.053/0.0148 in research_journey.md are wrong — do not propagate.
2. **"rpf ≡ rpfnox0 at every λ" is TPR-only.** On obf-**AUROC** they diverge/cross (rpf 0.34/0.46/0.83… vs rpfnox0 0.58/0.84/0.55…). Restate as "equal on the operational TPR@1%FPR; neither dominates on AUROC." Fix `hps_preprint.md` §5.2 + Fig 1 caption (`plot_lambda_sweep.py`).
3. **Two HPS numbers conflated.** HPS=0.997≈C4 (May JBShield/diverse-fixed saturation, `statistical_tests.json`) vs HPS-12≈0.92 (June obf_reps/HID clean). Label each by setup; the `0.92`/`0.924` also equals the Euclidean-proto in `hier_detector.json` — disambiguate.
4. **Radial (F9):** `radial_distribution_check.json` = **0/5 inverted, attacks higher radial** → radius *does* separate in the trained HPS projection, contradicting the June "angular only" viz. Reconcile (different setups) before any radial claim.
5. **HPS=NaN** in `norm_controlled_eval_llama3.json` — HPS arm failed; the REAL_SIGNAL verdict is **C4-only** (C4 0.998 after L2-norm). Flag the bug; C4's norm-robustness stands.
6. **rpf "inert lift" is a *weak* geometric test** (= suspicion #6): `x0=√(1/k+‖xₚ‖²)` is just the radial/norm coordinate, which we independently showed is uninformative — so "inert" is unsurprising, not a deep geometric result. Retract it as a *headline*; keep as a minor control. **[RESOLVED 2026-06-22]** Ran the clean isometry-matched control on the **non-saturated** `diverse_fixed` cache (AUROC ~0.998, real headroom): **rpf = rpfnox0 = 0.9980 exactly (gap 0.0000), C4 = 0.9981.** So the curved coordinate is *provably inert even off-saturation*, and the HPS (0.99) > HPS-Euclidean (0.968) gap was the **separately-trained contrastive projection (optimization), NOT the geometry** (`rpf_on_cache.py`).
7. **Unbacked sections:** §5.6 curvature-sweep table + Fig 6 and §5.5 taxonomy-gate numbers have **no committed JSON** (only stdout/prose). Save `curv_sweep.json` and a taxonomy-gate JSON when re-run.

**Positive-direction note (does NOT overturn the negative):** hyperbolic was only tested at d=8/16/32 (over-parameterized) and via argmax classification — never the **low-d (d=2–4) retrieval/entailment** regime (mAP, parent-accuracy, zero-shot leaf placement) the design doc proposed. That's the one fair, untested shot at a *positive* — but a *different* (hierarchical-modeling) paper, with real risk of also being negative given the shallow 2-level taxonomy.

---

## Documents reviewed
- `hps_project_reference.md` (June arc master) — current threat model, geometry, hierarchical, curvature.
- `research_journey.md` (May arc narrative) — F1–F16 source.
- `evaluation_report.md` (external-style review) — C1–C6 (stale draft, Anthropic, OOD, CIs, PGD, radial).
- `mentor_briefing.md` — length & norm confound updates.
- `PROJECT_INVENTORY.md` — script→result→verdict map (canonical caches, norm_controlled_eval verdict).
- `paper_draft.md` — the **stale** positive-claims draft (do not reuse numbers).
