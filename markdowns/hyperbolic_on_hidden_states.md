# The Usage of Hyperbolic Geometry on LLM Hidden States: A Discrimination–Ranking Dissociation

*A unified study of when curvature helps representations of harm in Llama-3-8B — and when it doesn't.*

**Authors:** Balhousani Alhussien, *et al.* — [Affiliation] · **Preprint, v1** · **Code:** `Mtehabsim/hps`

> **Status note (honesty).** Part I (the negative) is solid and extensively controlled. Part II (the positive) is **preliminary**: the effect is real but modest (~+0.10 mAP), was *not* the low-dimensional effect we pre-registered, and has open verification items (see §5.3, §8). Numbers herein are verified against the on-disk result JSONs; an "evidence-strength" table is given at the end.

---

## Abstract

Hyperbolic geometry embeds hierarchies with low distortion, and the token subspace of large language models is known to be hyperbolic (HELM). This motivates asking whether curvature helps representation-level probes of harmful content. We study this on Llama-3-8B-Instruct from two angles on the *same* hidden states. **First (negative):** for the flat harmful-vs-benign *decision* — detection and adaptive robustness under the Obfuscated-Activations attack framework — curvature provides **no benefit and, at strong curvature, a cost**; a linear probe is already at ceiling, the harm decision is linearly separable, and apparent positives trace to controllable artifacts (gradient masking, high-dimensional curvature-statistic failure, weak baselines, length/norm confounds). **Second (positive, preliminary):** when the *same* representations are used for **hierarchical retrieval** (rank harmful prompts by taxonomic similarity) rather than classification, a curved distance beats an **architecture-matched, trained Euclidean** prototype network by ~**+0.10 mAP** at higher embedding dimension, while the same curvature *reduces* classification accuracy — a **discrimination-vs-ranking dissociation**. We report honestly that this gain (i) is modest, (ii) *grows* with dimension rather than appearing at low dimension as hyperbolic theory predicts, and (iii) is not specific to the semantic harm hierarchy. The unifying mechanism is **faithful ≠ discriminative**: curvature buys faithful relative distances (good for ranking) at the cost of class separability (bad for a flat boundary). We contribute the dissociation, a calibrated methodology for measuring representation curvature, and a catalogue of the artifacts that make this question easy to get wrong.

---

## 1. Introduction

A *probe* reads a model's hidden activations and scores a property (here: harmfulness). Because hyperbolic space embeds trees compactly and LLM token spaces are hyperbolic, it is natural to ask whether curvature helps probes of harm — which has an obvious taxonomy (violence → weapons → explosives).

We answer with a controlled study that splits into two halves on identical Llama-3-8B hidden states:

- **Part I — the flat decision (negative).** "Is this harmful?" is a *classification* problem needing a dividing line. We show curvature does not help and slightly hurts it, robustly.
- **Part II — the hierarchy (positive, preliminary).** "Rank the most taxonomically-similar harmful prompts" is a *retrieval* problem needing good relative distances. Here a curved ruler gives a modest, real edge over an architecture-matched Euclidean one — and the same curvature that *helped* retrieval *hurt* classification.

**Thesis.** A single curvature knob, on the same representations, **trades discrimination for ranking**. This dissociation — not "hyperbolic is good" or "hyperbolic is useless" — is the contribution, alongside the measurement methodology that lets one state it honestly.

```mermaid
flowchart LR
  R[Same Llama-3 hidden states] --> C[Flat decision:\nharmful? / which class?]
  R --> Ret[Hierarchy:\nrank similar harmful prompts]
  C -->|curvature HURTS\n(typed_acc 0.75→0.55)| N[Part I: negative]
  Ret -->|curvature HELPS\n+0.10 mAP vs trained Euclidean| P[Part II: positive, preliminary]
  N & P --> T["Faithful ≠ discriminative:\none knob trades discrimination for ranking"]
```

---

## 2. Background

- **Hyperbolic / Lorentz model.** Constant-negative-curvature geometry with exponential volume growth; the Lorentz and Poincaré models are isometric (numerically we use Lorentz). Curvature `c` (κ): `c→0` ≈ Euclidean, `c≈1–2` properly curved.
- **HELM (arXiv:2505.24722).** LLM token/vocabulary subspace is hyperbolic; mixture-of-curvature helps language modeling. We reproduce the token-subspace fact with a *calibrated* estimator (§6) — but show it does not extend to the harm decision.
- **Obfuscated Activations (Bailey et al.).** Attacker optimizes an input to jailbreak *and* make the probe read "benign." Operational metric: **TPR@1%FPR** under attack. Dial **λ** = jailbreak-vs-evasion weight.
- **Gradient masking (Athalye et al., 2018).** A saturating output can *fake* robustness by killing the attacker's gradient; defeated by a non-saturating surrogate. We find and causally remove an instance (§4.3).
- **Faithful ≠ discriminative.** Low embedding distortion (faithful relative distances) is a *different* objective from class separability. This distinction is the paper's backbone.

---

## 3. Setup

- **Model / data.** Llama-3-8B-Instruct; last-token activations (layer 24 for geometry; HPS layer set `[0,2,17,24,28,31]` for probes). Harmful/benign datasets from the `obf_reps` HID set and a length-/topic-diverse benign cache (built to defeat the length confound, §6).
- **Probes.** **C4** = per-layer logistic regression (linear baseline; *not novel* — linear activation probes are established, e.g. Anthropic/DeepMind). **HPS-12** = Lorentz-projection + 12 trajectory features. **rpf / rpfnox0** = full-activation hyperbolic lift vs its Euclidean twin (differ only by the radial coordinate). **ProtoNet** = encoder + per-class prototypes; run with a **Lorentz** vs **Euclidean** distance (architecture-matched) for Part II.
- **Geometry measurement.** Embedding distortion into H/E/S with a **known-geometry calibration gate** and a **mandatory dimension-matched random floor** (§6).

---

## 4. Part I — The flat harm decision: curvature does not help

### 4.1 Clean detection
C4 is at ceiling (AUROC ≈ 0.998–1.0, TPR@1%FPR ≈ 0.99). On the saturated cache, HPS ≈ C4 (AUROC 0.9973 vs 0.9981) and **C4 is *significantly* ≥ HPS** (paired-bootstrap AUROC **p=0.036**; McNemar p=0.755, ns; TPR p=0.601, ns). In the obf_reps clean setting, HPS-12 is *weaker* (AUROC ≈ 0.92). Either way, hyperbolic never beats linear.

### 4.2 The "inert lift" — and an honest caveat
`rpf ≈ rpfnox0` on the **operational TPR@1%FPR** at every λ → the hyperbolic radial coordinate adds nothing operationally. (On obf-**AUROC** the two diverge/cross — a ranking artifact, not an operational difference; neither dominates.) **Caveat:** rpf's extra coordinate `x0=√(1/κ+‖x‖²)` is the activation *norm*, which we independently show is uninformative (§6) — so this is a *weak* test of geometry, not a deep result. It is reported as a minor control, not a headline.

### 4.3 Adaptive attacks: all bypassed; one apparent win was masking
Under soft-prompt attack, **every probe is bypassed** (TPR@1%FPR → ~0) at attacker-favorable λ, with rpf ≈ rpfnox0. An apparent HPS "resistance" at λ=0.1 was **gradient masking**, shown **causally**: switching the saturating sigmoid output to a non-saturating softplus (monotone → detection metrics unchanged) collapsed obf-AUROC **1.000 → 0.001** (full evasion). *(Hard-token GCG and the hierarchical detector under attack are unrun — §8.)*

### 4.4 Curvature sweep: classification is best at flat
Sweeping a curvature-`c` Lorentz model (`c→0` = Euclidean), **typed accuracy is best at c→0 and decreases monotonically with curvature** (0.753 → 0.548 at c=8). No curvature beats C4. Since the model is isometric and curvature is swept, the negative is robust to the hyperbolic *configuration*, not one setting.

### 4.5 Hierarchical detector: still no classification payoff
A taxonomy-aware ProtoNet: hyperbolic typed acc 0.701 < Euclidean 0.735; binary TPR@1%FPR C4 0.987 > hyperbolic 0.929 > Euclidean-prototype 0.241. The Euclidean-prototype's low binary TPR is a single-benign-prototype artifact; a **data-driven multi-benign hierarchy** fixes it and **equalizes H≈E (≈0.985), both < C4 (0.996)**. A fairer hierarchy makes Euclidean *catch up*, not hyperbolic win.

**Part I conclusion.** The harm *decision* is linearly separable; no curved geometry improves a problem a line already solves, and strong curvature degrades it.

---

## 5. Part II — The hierarchy: curvature helps retrieval (preliminary)

### 5.1 Reframe
Instead of "harmful?", ask **retrieval**: given a harmful prompt, rank others so same-taxonomy ones come on top (mAP). We use a deeper taxonomy (9 categories / 14 leaves) and the **architecture-matched ProtoNet**, run with a Lorentz vs Euclidean distance — *everything else identical*, so any gap is the geometry (the same isometry-controlled logic as rpf-vs-rpfnox0).

### 5.2 Result — a discrimination–ranking dissociation
Curved (c=1) vs the **trained Euclidean** ProtoNet, harm retrieval (mAP_leaf):

| d | curved | Euclidean (trained) | Δ(H−E) |
|---|---|---|---|
| 2 | 0.254 | 0.278 | −0.024 |
| 8 | 0.484 | 0.450 | +0.034 |
| 16 | 0.554 | 0.474 | +0.080 |
| 32 | **0.591** | 0.486 | **+0.104** |

At d=32, curved also leads on mAP_parent (0.621 vs 0.492). Yet on **classification**, Euclidean ≥ curved (typed acc 0.734 vs 0.721). **Same representations, same architecture: curvature helps ranking and hurts discrimination** — the dissociation, consistent with the curvature sweep of §4.4 in mirror image.

### 5.3 Honest caveats (these bound the claim)
- **Magnitude.** The fair gain is **~+0.10 mAP** vs the *trained* Euclidean ProtoNet — **not** the +0.32 one gets by comparing against *untrained* cosine/whitened (0.26/0.14). The trained-Euclidean baseline is the correct control; the untrained comparison would repeat the weak-baseline error this project was founded on catching.
- **Direction vs theory.** The advantage **grows with dimension** (d2 ≈ tie → d32 +0.10); hyperbolic's textbook win is at **low** d. At d=2–4 Euclidean ties or wins. So this is **not** the "compact tree embedding" mechanism it was motivated by; the mechanism is unsettled.
- **Not semantic-specific.** On a **benign topic** control (code/math/wiki), curved *also* beats trained Euclidean (+0.04) — so the effect is not specific to the harm hierarchy, only larger where Euclidean is below ceiling.
- **`c→0` anomaly.** The hyperbolic arm at c=0.05 underperforms the Euclidean arm (0.24 vs 0.49 harm), although `c→0` should recover Euclidean — suggesting numerical degeneracy at tiny curvature, so within-arm "mAP climbs with c" partly reflects recovery from degeneracy. The clean statement is curved(c=1) vs Euclidean.
- **Significance.** Seed std ≈ 0.03–0.05 (5 seeds) → the +0.10 gap is ≈2σ; **paired CIs/permutation tests are still owed.**
- **Ground truth.** Retrieval targets are LLM-generated taxonomy labels (~81% coverage, tree-alignment ρ=0.244) — noisy.

**Part II conclusion (calibrated).** There is a **real but modest** retrieval advantage for curvature at higher dimension, dissociating from classification — promising, not yet decisive.

---

## 6. Where hyperbolicity lives, and how to measure it without fooling yourself

- **Calibrated estimator.** δ-hyperbolicity and Ollivier-Ricci **fail** known-geometry calibration at LLM dimension (e.g., call a sphere hyperbolic); only baseline-corrected embedding distortion is trusted.
- **Where it lives.** Token/vocabulary subspace: **strongly hyperbolic (0.017× random floor)** — reproduces HELM. Harm-decision reps: **only ~0.53× floor** (weak; we do not over-read this). The harm decision itself is linear.
- **Confounds (a methodology contribution).** A **length confound** (length-only AUROC 0.973 → 0.318 after a length-diverse benign rebuild) and a **norm confound** (norm-only ~1.0 → 0.55 after L2-normalization) — yet **C4 stays 0.998 after L2-normalization**, so real harm signal exists beyond magnitude. Together with the curvature-statistic failure, these give a unified caution: *every effect must be checked against a dimension-matched dumb baseline.*
- **Cross-distribution transfer.** Training C4 on one benign source and testing on another, the **mid-to-late-layer harm direction generalizes (AUROC ≈ 0.88–0.96 at L17/24/28)** while **early layers invert** (surface features), and strict-FPR operating points are distribution-sensitive. So the separability reflects **genuine harm structure in semantic layers**, not pure dataset artifact — though the harmful *source* was held fixed (a held-out harmful dataset is future work).

---

## 7. Unified mechanism: faithful ≠ discriminative

Hyperbolic space embeds the manifold's *relative distances* faithfully (good for **retrieval/ranking**) but its exponential metric **compresses the angular directions that separate classes** (bad for a **flat decision boundary**). Hence one curvature knob *trades* the two: it helps hierarchical retrieval and hurts classification on the *same* reps. The negative and the positive are two faces of this single principle — which is why neither "hyperbolic helps" nor "hyperbolic is useless" is the right summary.

---

## 8. Limitations

- **Single model / layer** (Llama-3-8B, layer 24); generality untested.
- **Part II is preliminary:** modest effect, grows-with-d (contra theory), not harm-specific, c→0 anomaly, CIs owed, noisy LLM labels, shallow 2-level taxonomy.
- **Unrun:** hard-token GCG (did not reach jailbreak), and the hierarchical detector under adaptive attack (built, masking-safe, not yet run) — so adaptive claims are scoped to **soft-prompt**.
- **C4 is not a contribution** (established linear-probe approach); it is a comparator.

---

## 9. Conclusion

On Llama-3-8B hidden states, **hyperbolic geometry does not help — and slightly hurts — the linearly-separable harm decision** (robust across curvature, hierarchy, layer, and soft-prompt attack; apparent wins traced to artifacts), **but a curved distance gives a modest, real advantage for hierarchical retrieval** that dissociates from classification. The honest, unified statement is a **discrimination-vs-ranking trade-off governed by a single curvature knob**, explained by *faithful ≠ discriminative*. The contributions are this dissociation, a calibrated methodology for representation-curvature claims, and a documented set of the artifacts (gradient masking, high-D curvature statistics, weak baselines, length/norm confounds) that make the question easy to answer wrongly.

---

## Evidence-strength table (honesty ledger)

| Claim | Evidence | Strength |
|---|---|---|
| Hyperbolic ⊀ linear on clean detection | HPS 0.997 vs C4 0.998, p=0.036 (C4≥HPS) | **Strong** |
| All probes bypassed adaptively (soft-prompt); lift inert on TPR | λ-sweep; rpf≈rpfnox0 | **Strong** (soft-prompt only) |
| λ=0.1 "resistance" = gradient masking | causal sigmoid→softplus, 1.0→0.001 | **Strong** |
| Curvature monotonically hurts classification | sweep 0.753→0.548 | **Strong** |
| Token subspace hyperbolic; harm decision linear/weakly-hyperbolic | calibrated distortion (0.017× vs 0.53× floor) | **Strong (token) / Moderate (harm)** |
| Harm signal is genuine (not pure dataset artifact) | cross-distribution transfer, L17–28 ≈0.9; C4 survives L2-norm | **Moderate–Strong (benign side)** |
| Curvature helps hierarchical retrieval (dissociation) | +0.10 mAP vs trained Euclidean at d=32 | **Preliminary/Moderate** |
| Effect is the classic low-d hyperbolic mechanism | — | **Refuted** (grows with d) |
| Effect is semantic-specific | benign also +0.04 | **Not supported** |

---

## References
Bailey et al. (*Obfuscated Activations*); HELM (arXiv:2505.24722); Athalye, Carlini & Wagner (2018); Nickel & Kiela (2017, 2018); Sala et al. (2018); Gu et al. (2019); Ganea et al. (2018); Zou et al. (2023, GCG); HyPE (ICLR 2026); Anthropic Cheap Monitors (2025). *Verify venue/year before submission.*
