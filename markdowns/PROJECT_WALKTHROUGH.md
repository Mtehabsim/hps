# The HPS Project, Explained — From the First Idea to Now

*A plain-language walkthrough for colleagues. You don't need a geometry background. Scientific
names are kept (and explained the first time). Read the one-page summary; dive into chapters as
needed; use the glossary at the end as a reference.*

---

## One-page summary (read this first)

**The question.** AI models like Llama-3 turn each prompt into a long list of numbers (an
"activation") that represents what they understood. A **probe** is a small classifier that reads
those numbers and decides "is this prompt harmful?" We asked one specific question: **does it help
to measure distances in a *curved* (hyperbolic) space instead of the *flat* (Euclidean) space
when building such a probe?** There was a good reason to hope so — curved space is famously good at
representing **hierarchies/trees**, and harm has a natural hierarchy (violence → weapons → bombs).

**The honest negative (the main paper).** For the **yes/no harmful-or-not decision**, curved
geometry gives **no benefit, and at strong curvature a cost**. A plain linear probe is already
near-perfect (it's an "easy," linearly-separable decision), so there is no room for fancy geometry
to help. We proved this carefully and ruled out the usual ways such a study fools itself.

**The twist we found recently (the possible positive).** The *same* probe can be asked a
*different* job: not "yes/no," but **"given this harmful prompt, find and rank the most similar
ones"** (retrieval). For *that* job, the curved ruler **does** help — and the same curvature that
*hurts* the yes/no decision *helps* the ranking. One knob, opposite effects, on the identical data.
That mirror-image (a "discrimination-vs-ranking dissociation") is the candidate positive finding.

**The careful check — and a correction (important).** A first pass compared the trained *curved*
probe against *untrained* flat distance metrics (cosine, whitened) and reported a large **+0.32**
advantage. **That was the wrong baseline** — it is the *same crippled-baseline error that killed our
original +0.302 claim in Chapter 1*: comparing a trained model against an untrained one. The fair
comparison is the **identically-trained Euclidean ProtoNet** (same architecture, only the ruler
differs). Against that fair baseline the gain is **~+0.13 mAP at d=32** (+0.10 at d=16) — real, but
about **3× smaller** than first reported. Two more corrections fall out of using the fair baseline:
(i) the effect is **not harm-specific** — a benign topic hierarchy shows a similar or larger curved
gain; (ii) the advantage **grows with embedding dimension**, which is *backwards* from the textbook
reason hyperbolic was tried (it's supposed to win at *low* dimension), so whatever this is, it is
not the classic "compact tree embedding" effect.

**Bottom line.** The project is no longer just "hyperbolic doesn't help." It's becoming: *"hyperbolic
doesn't help a flat decision, but the curved ruler gives a small, real edge on hierarchical
retrieval (~+0.13 mAP) — a controlled trade-off between discrimination and ranking."* That is a more interesting,
and fully honest story.

---

## Part A — The concepts, in plain language

You only need five ideas to follow everything.

**1. Activations / hidden states.** When Llama-3 reads a prompt, each layer produces ~4096 numbers
— a point in a 4096-dimensional space. Similar prompts land at nearby points. We work with the
point from a chosen layer (usually layer 24) at the last token.

**2. A probe.** A small classifier that reads the activation and outputs a score. Our baseline probe
is **C4** — a plain *linear* probe (logistic regression). Our hyperbolic probes are variations that
measure distance in curved space. A probe is *trained* on labeled examples (harmful vs benign) and
*tested* on held-out ones.

**3. Flat (Euclidean) vs curved (hyperbolic) space.**
- *Euclidean* = ordinary straight-line distance, the geometry of a flat tabletop.
- *Hyperbolic* = a saddle/Pringle-shaped geometry that **curves away from itself**, so there's
  **exponentially more room as you move outward** from the center.
- Why anyone cares: a **tree** needs exploding room (1 root → few children → many grandchildren →
  a flood of great-grandchildren). Flat space runs out of room and things collide; hyperbolic space
  has exactly the room a tree needs. So *if* harm concepts form a tree, curved space *might*
  represent them with less distortion. **Curvature** (the knob `c` or `κ`) controls how curved:
  `c → 0` is basically flat, `c = 1–2` is properly curved.

**4. The two jobs a probe can do (this distinction is the heart of the whole project).**
- **Classification / detection** — draw a boundary, decide which side a prompt is on
  ("harmful?" or "which category?"). Needs a good *dividing line*.
- **Retrieval** — given one prompt, *rank* all others by similarity so that related ones come out
  on top (like a search engine). Needs good *relative distances*, not a boundary.
- These are genuinely different, and — as we'll see — geometry helps them differently.

**5. The attack setting (the threat model).** A jailbreak attacker doesn't just send a harmful
prompt; they *optimize* an input to (a) make the model comply **and** (b) make the probe say
"benign." We use the **obf_reps** framework (from Bailey et al., "Obfuscated Activations") to run
this attacker-vs-probe game. The key dial is **λ (lambda)**: how much the attacker prioritizes
jailbreaking vs evading the probe. The operational success metric is **TPR@1%FPR** — "what fraction
of harmful prompts do we still catch if we're only allowed to false-alarm on 1% of benign ones."

*(Two metric names you'll see for retrieval/ranking: **AUROC** = ranking quality overall; **mAP** =
mean average precision = "are similar items ranked near the top," the search-engine score.)*

---

## Part B — The journey, chapter by chapter

Each chapter follows: **what we believed → what we worried about → how we tested → what we found →
what we did next.**

### Chapter 1 — The exciting first result (that turned out to be wrong)

- **Believed:** Hyperbolic geometry should help detect harmful activations, because harm is
  hierarchical. Our first probe (**HPS** — Hyperbolic Projection Sentinel) seemed to beat the
  Euclidean baseline by a big margin (**+0.302 AUROC**).
- **Worried:** Was that a fair comparison? A big win is suspicious.
- **Tested:** We matched the Euclidean baseline's number of parameters to the hyperbolic one
  (previously the Euclidean version had been unfairly small/weak).
- **Found:** The +0.302 advantage **vanished**. Once the baseline was equally capable, hyperbolic
  ≈ Euclidean. The "win" was an artifact of a crippled baseline.
- **So:** This became the project's founding lesson — *be ruthless about controls; our own exciting
  result was the first thing to fall.* We pivoted from "we found a win" to "let's find out the truth."

### Chapter 2 — Tearing down our own signal: confounds

- **Believed:** Even tied, the probe clearly separates harmful from benign (AUROC near 1.0).
- **Worried:** Maybe it separates them for a *dumb* reason — not "harm," but some superficial
  artifact that happens to correlate with harm.
- **Tested two suspects:**
  - **Length confound:** harmful and benign prompts had different lengths. A "length-only"
    classifier scored AUROC **0.973** — almost as good as the real probe! After we rebuilt the
    benign set to match lengths, length-only dropped to **0.318** (near useless).
  - **Norm confound:** the *magnitude* of the activation vector also leaked the label. A
    "norm-only" classifier hit ~1.0, dropping to **0.761** after a tokenization fix.
- **Found:** Real confounds existed — but after fixing them, the genuine probe (C4) **still**
  worked (AUROC ~0.998 even after we normalized away magnitude). So there *is* real harm signal
  beyond the artifacts — but the episode taught us how easily these benchmarks mislead.
- **So:** We adopted a rule — *every claimed effect must be checked against a dumb baseline that
  could explain it by accident.* (This rule comes back at the very end with the retrieval baselines.)

### Chapter 3 — The honest negative under a realistic attack

- **Believed:** Maybe hyperbolic geometry, even if it doesn't help *clean* detection, makes the
  probe **harder to evade** by an adaptive attacker.
- **Worried:** We need a fair, strong attacker, and a fair comparison between the hyperbolic probe
  and an otherwise-identical Euclidean one.
- **Tested:** The obf_reps attacker game across the full λ-sweep, comparing:
  - **rpf** = the hyperbolic probe (lifts the activation into curved space), vs
  - **rpfnox0** = the *identical* probe with the one hyperbolic coordinate removed (its Euclidean
    twin). Any difference between them is *exactly* the hyperbolic contribution.
- **Found:** By the operational metric (TPR@1%FPR), **every probe gets bypassed** at the λ an
  attacker would choose, and **rpf behaves the same as its Euclidean twin**. The hyperbolic
  coordinate added nothing to robustness.
- **So:** The negative now held under the realistic threat model too. *(Caveat we later flagged: the
  way `rpf` was built, the extra hyperbolic coordinate is a deterministic function of information the
  linear probe already had — so "it added nothing" was partly true by construction. The honest
  detector test in Chapter 6 fixes this.)*

### Chapter 4 — Making the negative bullet-proof

We didn't want a lucky or sloppy negative. Three hardening steps:

- **The "gradient masking" scare.** At one setting (λ=0.1) the hyperbolic probe *looked* unbeatable.
  - **Worried:** Is it truly robust, or is the attacker's math just getting stuck?
  - **Tested:** We swapped the probe's output squashing function from **sigmoid** (which saturates
    and kills the attacker's gradient — "gradient masking," a known false-robustness trap) to
    **softplus** (non-saturating, but ranks scores the same so detection is unchanged).
  - **Found:** The "robustness" **evaporated** (evasion score went from 1.0 to ~0.001). It was a
    measurement artifact, not real security. We proved it causally by toggling the one function.
- **Can we even trust our "is it curved?" measurements?**
  - **Worried:** Popular curvature statistics (δ-hyperbolicity, Ollivier-Ricci) might be unreliable
    in 4096 dimensions.
  - **Tested:** We ran them on *known* shapes (a tree, a sphere, a flat blob). They **failed** —
    e.g. called a sphere "hyperbolic." So we used **embedding distortion** with a **dimension-matched
    random floor** (a random blob run through the same pipeline) — because high-dimensional anything
    looks a bit hyperbolic, only the margin *below the random floor* counts.
  - **Found (where curvature actually lives):** the model's **token/vocabulary space** is *strongly*
    hyperbolic (0.017× the random floor — confirms the HELM paper), but the **harm-decision
    activations are only weakly so** (0.53× floor), and the harm decision itself is **linear**.
- **Was it just an unlucky amount of curvature?**
  - **Tested:** A **curvature sweep** — try every curvature from flat to strongly curved.
  - **Found:** For classification, accuracy is **best at flat (c→0) and gets monotonically worse as
    you add curvature** (typed accuracy 0.753 → 0.548). Remember this curve — it flips in Chapter 6.
- **So:** The negative is robust across attack type, curvature, hierarchy, and layer. This is the
  current main paper: *"When Hyperbolic Geometry Doesn't Help."* Its mechanism: **the harm decision
  is linearly separable, so no curved geometry can improve it; and "a faithful (low-distortion)
  embedding" is not the same as "a discriminative one."**

### Chapter 5 — The audit: was hyperbolic given a *fair* chance?

- **Believed:** The negative is solid (it is).
- **Worried:** "No benefit for detection" is not the same as "hyperbolic is useless." Did we test it
  on the tasks where curved geometry is *supposed* to shine, or only where it was bound to lose?
- **Tested (a careful internal review of our own code and the literature):** We checked each
  hyperbolic thing we built. Findings:
  - Our main probe's "hyperbolic coordinate" was a deterministic function of the norm — so a *linear*
    probe couldn't benefit from it **by construction**. That's a near-tautology, not a discovery.
  - We only ever tested **classification** on a **shallow, flat** decision, at **high embedding
    dimensions** — exactly the regime where flat geometry already wins.
  - We **never** tested the tasks the literature says hyperbolic is *for*: **retrieval, entailment,
    low-dimensional embedding of a hierarchy.**
- **Found:** A genuine, untested opportunity. The harm *taxonomy* itself was already measured to be
  meaningfully hyperbolic (it passed a "green-light" gate: categories fit a tree in curved space far
  better than flat, p=0.003) — we just never built a probe that *used* that for a hierarchy task.
- **So:** We designed a new, **fair** experiment specifically targeting the regime where curvature
  could win — and **pre-registered** it (wrote down the hypotheses and the pass/fail criteria *before*
  running, so we can't fool ourselves after the fact). See `positive_experiment_prereg.md`.

### Chapter 6 — The positive turn: retrieval, not classification

This is the new experiment (`hyperbolic_retrieval.py`). The design is deliberately **fair**: one
small probe (`ProtoNet`) with a learnable encoder and one "landmark" prototype per harm category.
We run it **twice** — once measuring distance to landmarks with a **flat** ruler, once with a
**curved** ruler — and **everything else is identical**. So any difference is *only* the geometry.

We then ask it two jobs on held-out prompts:
- **Classification** ("which harm category?") — the old job.
- **Retrieval** ("rank all other harmful prompts by similarity; do same-category ones come out on
  top?", scored by **mAP**) — the **new** job, the one curved geometry is theoretically good at.

- **Believed (our pre-registered guess):** curved would win at *low* dimension (the classic result).
- **Found — and this is the interesting part:**
  - Curved geometry **helps retrieval** vs the identically-trained Euclidean probe (harm d=32: 0.62
    vs 0.49, **+0.13 mAP**); the curved-vs-Euclidean edge peaks around c=1–2.
  - This is the **mirror image** of Chapter 4: the *same* curvature that made *classification* worse
    makes *retrieval* better — on the *same* representations. One knob, opposite signs.
  - Our specific low-dimension guess was **wrong** (the advantage grew *with* dimension, not shrank) —
    we report that honestly; it means this is not the textbook compact-tree effect. A separate sub-test
    (zero-shot placement of a held-out category) favored flat — also reported honestly.
- **So:** We had a candidate positive — but candidate results that weren't predicted in advance must
  be independently confirmed before we believe them. On to the checks.

### Chapter 7 — Confirming it's real, not an artifact

Two worries, two checks (this is Chapter 2's rule, applied again):

- **Worry 1: Is it really *curvature*, or just the probe's architecture?**
  - **Tested:** the curvature sweep on retrieval. If curvature is the cause, mAP should **peak at
    curved (c≈1–2) and collapse toward flat (c→0)**.
  - **Found:** exactly that (mAP 0.30 at c≈0 → 0.62 at c=1, plateau). **It's the curvature.**
- **Worry 2: Does curved beat *smart* flat methods, or only a weak flat baseline?** (The big one.)
  In high dimensions, ordinary flat distance has a known flaw — everything starts looking equally far
  apart ("distance concentration"). So curved beating *naive* flat could just mean *flat broke down*,
  not *curved is good*.
  - **Tested:** we compared curved retrieval against flat methods that are **immune** to that flaw —
    **cosine** (direction only) and **whitened/Mahalanobis** (rescales the axes) — plus a
    **radial-reweight** ablation that fakes the one simple thing curvature does (compress by radius).
  - **The baseline lesson (we caught ourselves repeating Chapter 1).** Our *first* read compared the
    trained curved probe against the *untrained* model-free metrics (cosine 0.30, whitened 0.20) and
    reported "+0.32, flat stuck at chance." That is the **crippled-baseline error** — a trained model
    vs an untrained one. The fair baseline is the **trained Euclidean ProtoNet** (identical
    architecture, only the ruler differs), which scores **0.49** at d=32. The honest numbers below
    use that fair baseline.
  - **Found (fair baseline = trained Euclidean ProtoNet):**
    - **Harm taxonomy, d=32:** curved 0.62 vs trained-Euclidean 0.49 → **+0.13 mAP** (seed-sd ≈0.04,
      so ~3σ across seeds — likely real, but proper per-query CIs are still owed; see Part D).
    - **Across embedding sizes (d=2…32):** the curved edge over trained-Euclidean **grows with
      dimension** (−0.03 at d=2, ~0 at d=4, +0.10 at d=16, +0.13 at d=32). This is **backwards** from
      the textbook reason we tried hyperbolic (it should win at *low* d for compact tree embedding) —
      so whatever this effect is, it is **not** the classic compact-hierarchy mechanism. We report it
      honestly; it weakens the "we found the expected hyperbolic effect" story.
    - **Benign topics, d=32:** curved 0.94 vs trained-Euclidean 0.78 → **+0.16** — *similar or larger*
      than harm. So the effect is **not harm-specific** (an earlier "rescues semantic hierarchy" claim
      was an artifact of comparing against cosine, which happens to do well on surface-separable
      benign topics — not a fair comparison).
    - **A degeneracy caveat:** at c≈0 (near-flat) the curved arm scores 0.30, *below* the trained
      Euclidean 0.49 — they should match if c→0 truly recovered Euclidean. They don't, so the curved
      arm is **numerically degenerate at very small curvature**, and part of the "mAP climbs as
      curvature rises" is the arm *recovering from degeneracy*. The clean comparison is **curved at
      c≈1 vs trained-Euclidean = +0.13**, not the c→0 ramp.
- **So — the precise, defensible conclusion:** the curved ruler gives a **small, real** edge on
  hierarchical retrieval (**~+0.13 mAP** at d=32) over an identically-trained Euclidean probe, while
  Euclidean ≥ curved on classification — a genuine but modest discrimination-vs-ranking dissociation.
  It is **not** harm-specific and it does **not** match the classic low-dimension hyperbolic story.
  (The model-free baselines remain useful for a *different* point — they show that on harm, neither
  curved nor Euclidean retrieval is just surface topic-matching, since cosine scores only ~0.30 there
  vs ~0.86 on benign — but they are **not** the baseline for the curvature claim.)

---

## Part C — Where we are now

We have **two halves of one story**, measured on the *same* Llama-3 hidden states with the *same*
fair probe:

1. **Negative (solid):** curvature does **not** help — and slightly hurts — the flat harmful/benign
   *decision*. A linear probe is already near-perfect; geometry has no room to help.
2. **Positive (real, modest, and now confirmed against the fair baseline + every stress test):**
   curvature gives an edge on **hierarchical retrieval** vs an *identically-trained* Euclidean probe.
   The same curvature hurts classification and helps retrieval — a controlled
   **discrimination-vs-ranking dissociation**. It has now passed seven checks:
   - **Fair baseline:** **+0.13 mAP** at d=32 on the 2-level tree (vs trained-Euclidean, not the
     untrained cosine that caused our earlier overclaim).
   - **Significance:** paired per-query bootstrap → gap **+0.130, 95% CI [+0.121, +0.139], p<0.0001**.
     (Honest caveat: the bootstrap is over non-independent query pairs and *overstates* significance;
     the conservative measure is the **seed-level ~5–8σ** over 5 seeds, which also holds.)
   - **Depth:** with the deepened 3-level tree (general fallback fixed, 64%→8%, 75 real leaves) the
     *same* category/leaf-retrieval gap grows to **+0.19 mAP (≈+62% relative: 0.49 vs 0.30)** — depth
     nearly doubles the advantage, the predicted direction.
   - **Label-noise robustness:** the gap stays positive and significant under up to **40% random label
     corruption**. So imperfect 8B-model labeling cannot explain it.
   - **Inter-labeler agreement (#2):** two independent labelers (the LLM labels vs a keyword assigner)
     agree strongly — **Cohen's κ = 0.76 leaf / 0.82 category** — and the confusions are
     semantically-adjacent (cyber-intrusion↔malware; the tiny κ=0.33 `regulatory_evasion` leaf).
   - **Drop the unreliable leaves:** re-running with the flagged leaves removed, the gap **survives** —
     **+0.118** (drop-only, d=32, p<0.0001), i.e. removing the worst leaf *strengthened* it. (Merging
     the confused cyber pair lowers it to +0.076, as expected: merging deletes hierarchy the effect
     lives on, so we report **drop-only as the clean robustness number**.)
   - **Depth-generic, on ground-truth labels (the strongest check):** the whole effect replicates on
     **MMLU academic subjects** — a *non-harm*, *ground-truth-labeled* hierarchy (no LLM labeling, no
     `general` fallback, no κ/label-noise caveats). Fair-baseline gap grows with depth exactly as on
     harm: **2-level +0.158 → 3-level +0.191** at d=32 (p<0.0001; +0.138→+0.214 at d=16). Crucially it
     is **not** surface-separability: flat baselines are weak (cosine 0.41/whitened 0.27 at L2) and
     *collapse* as depth increases (0.28/0.10 at L3), while curved holds (~0.54). So "deeper hierarchy
     → bigger curved advantage" is a **general property of LLM hidden-state geometry**, not a harm quirk.

   Caveats we still state up front: the gain is **ranking-only** — top-1 detection metrics are TIED
   (nn_parent H 0.787 vs E 0.789; typed_acc H≈E), so this is **not** a detector improvement (see the
   retrieval-as-detection note below); the edge is **not harm-specific** (it replicates on MMLU
   academic subjects — now a *strength* of the general "curvature embeds hierarchy" claim, not just a
   caveat); it **grows with dimension** (contra the classic low-d hyperbolic story); the absolute mAP
   is modest (~0.5); and it is one model / one layer vs a small ProtoNet.

   **Retrieval-as-detection (asked & answered, twice).** Could we detect harm by retrieving the
   nearest category instead of a linear probe?
   - *In-distribution binary decision:* no — the hyperbolic gain is entirely in mAP (the full ranked
     list), while a detector reads the **top-1** neighbor, exactly where H≈E.
   - *Open-set across harm TOPICS* (hold out a whole harm category, e.g. never-seen "cyber"): we ran
     it (`openset_detection.py`). **C4 linear wins decisively** — macro AUROC **0.999 vs 0.974**
     (hyperbolic-kNN) vs 0.946 (euclidean-kNN); C4 TPR@1%FPR 0.99 vs 0.59. So a linear probe flags
     harm *topics it never trained on* near-perfectly: "harmfulness" is a **single linear,
     topic-general direction**. (Within retrieval, hyperbolic still beats euclidean on every category,
     +0.03 AUROC / ~2× TPR — the dissociation persists, it just never lifts retrieval above linear.)
   - *Important scoping correction:* the taxonomy data are **plain harmful *requests*** (avg 14 words,
     zero jailbreak scaffolding — no DAN/persona, encoding, or adversarial suffixes), so the above is
     "open-set across harm **topics**," NOT "novel **jailbreak** detection."
   - *Open-set across ATTACK TECHNIQUES* (the genuinely safety-relevant test, `openset_attacks.py`):
     held out an entire *evasion method* from 273 validated jailbreaks. **Ran it, with a format/length
     confound control.** The control was essential: two families are near-single-templates and a
     trivial 2-feature length model alone separates them from benign (**format-only AUROC: JBC 0.996,
     prompt_with_random_search 1.000** — flagged ARTIFACT, detection there is template-matching, not
     harm). On the two *clean* families (**GCG format-only 0.41, PAIR 0.67**), detection is real and
     near-perfect — **but retrieval ties the linear probe** (GCG: knn-hyp 0.996 ≈ C4 0.995; PAIR:
     knn-hyp 0.989 vs C4 0.998), both far above the format baseline. So the linear "is-this-an-attack"
     direction **generalizes to unseen attack techniques** (not just unseen topics), and it is **not**
     a length artifact — yet retrieval/curvature again add nothing to *detection*. (Report GCG/PAIR
     per-family, NOT the macro, which is contaminated by the two artifact families.)

   **Both open-set tests agree:** retrieval-as-detection adds nothing over a linear probe for novel
   *topics* or novel *attacks*; the harm/attack direction is linear and generalizes. The hyperbolic
   edge stays confined to *ranking*, never *detection* — the dissociation, confirmed a third way.

The honest framing of the combined paper:

> *"On the same LLM hidden states, a single curvature knob trades discrimination for ranking: it
> hurts flat classification while giving a small but statistically robust gain on hierarchical
> retrieval (+0.13 mAP, 95% CI [+0.12,+0.14], vs an identically-trained Euclidean probe). The effect
> is modest, not harm-specific, and — surprisingly — strengthens with dimension rather than at the
> low dimensions where hyperbolic embeddings are classically motivated."*

This is stronger than either half alone, and every surprising or deflating result (the modest fair
gain, the non-harm-specificity, the backwards dimension trend, the small-c degeneracy) is reported
honestly — including our own initial baseline mistake, which we caught and corrected.

> **Methodology note (for the record).** An initial read of this experiment reported a **+0.32**
> retrieval gain by comparing the trained curved probe against *untrained* distance metrics (cosine,
> whitened). That repeated the project's founding error (Chapter 1's artifactual +0.302 from a
> crippled baseline). The fair comparator is the trained Euclidean ProtoNet; against it the gain is
> **~+0.13**. The correction is logged here rather than hidden — it is the project's own rule working.

---

## Part D — What we are going to do next

**Status update:** the positive has gone from "fragile/unconfirmed" to "survives every check we ran,"
including the full mislabeling battery (random noise, inter-labeler agreement, dropping unreliable
leaves) AND the depth-generic test (MMLU, ground-truth labels — depth effect replicates, +0.16→+0.19).
The "is it harm-specific / label-noise / surface-features?" questions are all now answered *no*. The
main remaining gap is cross-model generalization. In priority order:

1. **Generalize beyond one setting (now the top open item).** Everything is on Llama-3-8B, one layer
   (24). Repeat the retrieval dissociation on 1–2 more models / layers before any "property of LLM
   representations writ large" claim. This is the biggest reviewer ask for the positive half.
2. **Explain the small-c degeneracy.** The curved arm under-performs trained-Euclidean at c≈0; fix or
   characterize it so the curvature-sweep story isn't "recovering from a numerical artifact."
3. **Write it up.** A *modest but well-controlled* discrimination-vs-ranking dissociation, now shown
   general across two hierarchies (harm + MMLU) and depth-scaling (TMLR / representation-geometry
   venue): negative first half + confirmed positive second half. The clean **negative** is the floor
   regardless.

*(Closed since last update: the attack-family retrieval idea — the open-set attack run + format
control already showed attack families are surface-separable and non-hierarchical, so curvature can't
help organize them; not worth a dedicated run. The depth-generic question — answered yes via MMLU.)*

**Done so far:** dimension sweep + model-free baselines (caught the baseline error → fair +0.13);
paired per-query bootstrap (p<0.0001, ~5–8σ seed-level); deepened 3-level taxonomy via data-grounded
sub-categories (general 64%→8%, 75 real leaves; same-task gap +0.10→+0.19); label-noise robustness
(significant to 40% corruption); inter-labeler agreement (κ=0.76/0.82); drop/merge robustness re-run
(drop-only +0.118 survives — the effect does not depend on the unreliable leaves); confirmed the gain
is ranking-only (top-1 tied) so retrieval-as-detection won't help the binary decision; open-set across
harm TOPICS (C4 0.999 ≫ retrieval — "harmfulness" is a topic-general linear direction); open-set across
ATTACK TECHNIQUES with a format-confound control (clean families GCG/PAIR: detection real & ~0.99 but
retrieval ties C4; JBC/random-search flagged as length artifacts — report per-family, not macro);
**depth-generic confirmation on MMLU** (ground-truth labels, non-harm: fair-baseline gap +0.158→+0.191
from 2→3 level at d=32, p<0.0001; flat baselines weak and collapsing with depth — the depth effect is
general, not a harm-data quirk).

---

## Part E — Novelty & related work (is this new?)

A deep literature search (104-agent fan-out, 22 primary sources, adversarial per-claim verification)
graded each finding against prior work. **Verdict: novel as a package; the two most distinctive
claims are genuinely unanticipated.** Per-claim:

| # | Finding | Verdict | Closest prior work / why |
|---|---|---|---|
| **1** | Discrimination-vs-ranking dissociation (one curvature hurts classification, helps retrieval, on identical reps) | 🟢 **FULLY NOVEL** | No paper measures *both* axes on the same reps. Madan et al. (CVPR-W 2026) has a curvature-vs-task tradeoff but between **two retrieval tasks at two curvatures, with NO classification metric**. HIER (CVPR 2023) frames curvature as *improving* discrimination (no tradeoff). **Flagship contribution.** |
| **3** | Hyperbolic advantage GROWS with dimension (d=2→32) | 🟢 **NOVEL (contrarian)** | Inverts an entrenched consensus (De Sa 2018: hyperbolic wins at d=2; Bansal & Benton 2021: advantage gone by d≥50). Strongest novelty signal **and** biggest empirical-burden risk — needs a *mechanism*, not just significance. |
| **4** | Advantage grows with DEPTH (2→3 level, harm + MMLU) | 🟡 **Likely novel, under-corroborated** | No prior depth-scaling result found — but inferred from *absence*, not a positive search. |
| **6** | δ-hyperbolicity / Ollivier-Ricci unreliable at LLM dim | 🟡 **Novel in this form, uncorroborated** | No direct prior art; mild counter-current (HypLoRA uses δ on Llama-3 token embeddings without flagging it). |
| **2** | Hyperbolic helps retrieval of LLM hidden-state categories | 🟠 **PARTIALLY ANTICIPATED — top scoop risk** | Concurrent **Raj, ICLR-2026 GRaM workshop** (`JmWG0P9MDf`, 2 Mar 2026) shows hyperbolic>Euclidean probing of LLM hidden states — but DeepSeek/Qwen, *reasoning* (PrOntoQA), **classification-probe only, no retrieval/mAP/dimension/depth/harm/curvature-tradeoff**. Shares the premise, scoops none of the specifics. |
| **5** | Negative side: harm linearly separable, geometry doesn't help detection | 🔴 **LARGELY KNOWN** | Refusal/harm is an established linear/causal direction (Arditi et al., NeurIPS 2024); separable-cluster detection (HSF); adaptive fragility (Bailey et al. 2024). **Frame as confirmation, cite — not a contribution.** |

**Two scoop risks to manage:** (a) **cite Raj (`JmWG0P9MDf`) proactively** in related work and state the four ways we differ (model, task, retrieval-not-classification, and the dissociation/scaling/depth/curvature results it lacks) — don't let a reviewer surface it first; (b) the field is fast-moving (Raj, Madan, and the negative-side source are all 2026 works), so **move promptly** to an arXiv/workshop stamp.

**The one defense to prepare (Finding 3):** because the dimension-growth direction inverts De Sa/Bansal, a reviewer will ask "is it within noise, and *why* the reversal?" We already have significance (bootstrap p<0.0001, MMLU replication, weak/collapsing flat baselines); what's still owed is a **mechanism** — likely "isometry-matched *trained* probes on *contextual* activations" differs from prior work's fixed-capacity *graph/token reconstruction." State and ideally test this.

**How to frame the paper:** lead with **Finding 1 (dissociation)** — fully novel and it unifies the negative + positive halves; support with **Finding 3 (dimension reversal)** explicitly defended vs the consensus; treat **Finding 5** as confirmation of known linear-separability results; soft-pedal or further-corroborate **4 and 6**. (Closest-prior-work fetched in full: Raj `JmWG0P9MDf`; Madan arXiv:2603.14022 — both confirmed to *not* report the dissociation.)

---

## Glossary (quick reference)

- **Activation / hidden state** — the ~4096 numbers a model produces for a prompt at a given layer.
- **Probe** — a small classifier reading the activation. **C4** = our plain linear baseline.
- **HPS** — Hyperbolic Projection Sentinel, our hyperbolic probe family.
- **rpf / rpfnox0** — the hyperbolic probe and its identical Euclidean twin (their difference =
  exactly the hyperbolic contribution).
- **ProtoNet** — the fair retrieval/classification probe used in the new experiment: encoder +
  one prototype "landmark" per category; flat vs curved differ *only* in the distance ruler.
- **Euclidean (flat) vs hyperbolic (curved)** — straight-line geometry vs a saddle geometry with
  exponentially more room outward (good for trees/hierarchies).
- **Curvature (c / κ)** — how curved; `c→0` ≈ flat, `c=1–2` ≈ properly curved.
- **Classification vs retrieval** — "which side of the boundary?" vs "rank the most similar items."
- **Taxonomy / hierarchy** — the harm tree (category → subcategory → behavior).
- **obf_reps** — the adaptive attacker-vs-probe framework (Bailey et al.).
- **λ (lambda)** — attacker's priority dial: jailbreak vs evade-the-probe.
- **AUROC** — overall ranking/separation quality (1.0 perfect, 0.5 chance).
- **TPR@1%FPR** — operational detection metric: harmful caught while false-alarming on ≤1% of benign.
- **mAP** — mean average precision; the retrieval "are similar items ranked on top" score.
- **Confound** — a superficial feature (length, magnitude) that leaks the label and fakes a real
  signal. We always test against these.
- **Gradient masking** — false robustness where the attacker's math gets stuck (saturated sigmoid),
  not real security. Defeated by switching to a non-saturating function (softplus).
- **Embedding distortion + random floor** — our trustworthy "is it curved?" measure; only the margin
  below a dimension-matched random baseline counts.
- **Distance concentration** — in high dimensions, ordinary flat distances all look similar; why we
  compare against concentration-immune baselines (cosine, whitened).
- **Faithful ≠ discriminative** — a geometry can preserve distances well (faithful) yet not separate
  classes well (discriminative); the core mechanism behind both halves of our story.

---

## Document map (where the details live)

- `hps_preprint.md` — the formal negative-result paper.
- `hps_project_reference.md` — detailed reference (every probe, attack, metric, result table).
- `reconciliation_memo.md` — which older findings are valid/superseded, and our own list of doubts.
- `positive_experiment_prereg.md` — the pre-registered design for the retrieval (positive) experiment.
- `hyperbolic_retrieval.py` — the code for the new experiment (the fair probe + retrieval metrics +
  the concentration-immune baselines).
- `research_journey.md` — the long-form chronological lab notebook (the raw version of Part B).
