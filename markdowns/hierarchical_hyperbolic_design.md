# Hierarchical-Hyperbolic Jailbreak Detection — Design Sketch

## Why this is the one direction where hyperbolic could genuinely help

Everything so far shows hyperbolic geometry is **inert for binary harm detection**:
- The harm signal is **flat/linear** (a linear probe C4 hits ~1.0 clean recall@1%FPR).
- The hyperbolic lift is **inert** (`rpf ≡ rpfnox0` at every λ under the soft attack).
- Curvature **statistics** (δ, Ollivier-Ricci) are **unreliable at LLM dimension** (they invert known geometries — see `calibration_panel.py`), so "the activations are hyperbolic" is **not** an evidenced premise.

Hyperbolic space only pays off when the data has a **hierarchy** (its volume grows
exponentially with radius, matching the exponential node count of a tree). A **binary**
label has no hierarchy. **So the only principled way to make hyperbolic matter is to give
the task a hierarchy.**

## The reframe: harm as a taxonomy, not a bit

Replace `harmful ∈ {0,1}` with a **harm taxonomy** (a tree), e.g.:

```
root
├── violence
│   ├── weapons
│   │   ├── firearms
│   │   └── explosives ── synthesis-instructions (leaf behavior)
│   └── ...
├── cyber
│   ├── malware ── ransomware (leaf)
│   └── intrusion ── credential-theft (leaf)
├── fraud ── phishing (leaf)
└── benign  (the "far" subtree)
```

A natural source: HarmBench / StrongREJECT categories already group behaviors → use them
(or LLM-assisted) to build a 2–3 level tree over the existing behaviors.

## The detector

1. Extract hidden states for each behavior (as now, 6 layers).
2. Train a **hyperbolic embedding** (Lorentz) of behaviors that respects the taxonomy:
   parents near the origin (small radius), leaves near the boundary (large radius),
   tree distances ≈ hyperbolic distances. (Entailment-cone / Poincaré-embedding loss.)
3. Detection = "is this point in (or near) a *harmful* subtree" — a hierarchical decision,
   not a single hyperplane.

The **Euclidean control** (mandatory, the `rpf` vs `rpfnox0` lesson): the *identical*
pipeline embedding the taxonomy in Euclidean space. Hyperbolic only "wins" if it beats
this control — measured the correct way:
- **Embedding distortion / mAP of the taxonomy** in H vs E (use `embedding_distortion.py`,
  which passed the calibration gate). If the harm taxonomy embeds with materially lower
  distortion in H than E, hyperbolic has real structure to exploit.

## Why this could raise the adaptive-attack budget (the hyperbolic mechanism)

Against a **flat** binary boundary, the attacker moves the activation across **one**
hyperplane — cheap. Against a **hierarchical hyperbolic** detector, to be scored
"benign" the attacker must move the point **out of the entire harmful subtree to the
benign subtree**, i.e. traverse the tree across a region where the **hyperbolic metric
expands exponentially** → a **larger** required perturbation. Larger perturbation ⇒
(a) higher attacker cost, and (b) more likely to **break the harmful behavior** or be
**flagged as off-manifold**.

This is the falsifiable claim: *hierarchical-hyperbolic detection raises the
evasion budget more than the Euclidean-taxonomy control, and the gain is not gradient
masking* (verify with a gradient-free / GCG attack — see `attack_cost_curve.py`).

## Evaluation protocol (adaptive, honest)

1. **Calibration:** taxonomy embeds with lower distortion in H than E? (gate)
2. **Clean:** hierarchical detection accuracy (per-level), H vs E vs C4.
3. **Adaptive cost curves:** evasion vs attacker budget (λ, GCG steps, perturbation norm)
   for H-taxonomy vs E-taxonomy vs C4. Does H raise the budget?
4. **Gradient-masking control:** repeat with a gradient-free attack; cost increase must survive.
5. **Multi-epoch attacker–defender:** does the H advantage persist as the attacker adapts?

## Honest expectation

- If the **harm taxonomy embeds better in H than E** (plausible — taxonomies are trees,
  and trees are the canonical hyperbolic win), AND the **adaptive cost is higher for H**
  AND it **survives the gradient-free control** → genuine, novel positive result:
  *"hierarchy + hyperbolic raises jailbreak-evasion cost."*
- If H ≈ E even with the taxonomy → strong negative: *hyperbolic doesn't help even when
  the task is made hierarchical* — a clean, complete story alongside the flat-binary result.

Either outcome is a real, hyperbolic-centered contribution, built on evidenced premises
(measured the correct way) rather than the unsupported "activations are hyperbolic."
