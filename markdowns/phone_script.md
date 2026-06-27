# Phone script — what to SAY (screen-share `presentation.md`)

*Terse talking points. One block per slide. Back-pocket answers at the bottom.*

---

### The question
- Models turn prompts into activations; a probe reads them to flag jailbreaks.
- Curved geometry is good at hierarchies; harm has a hierarchy → does curved beat flat for jailbreak detection?

### Main result
- Answer's **no**. Left: pure hyperbolic probe **ties** the linear probe at 1.0 — curvature adds nothing even at ceiling.
- The 0.92 one is a feature-engineered variant — gap is its **features, not geometry**.
- Right, the real test — **adaptive attacker: every probe bypassed**, curved tracks its flat twin exactly.

### Why (the important one)
- It's **not** "linear is perfect, no room" — that's only clean detection.
- For the adaptive problem curvature **could** have helped — that was the hypothesis — so we **tested it**. It doesn't.
- In a harder setting it even beats the flat detector, but **still loses to linear**.
- Real reason: **causal separability** — attacker crosses whatever boundary you draw, curved or straight. Geometry can't raise the attack cost.

### What rpf is (only if asked)
- Same pipeline; HPS squeezes to 64 dims + 12 hand-crafted features = the bottleneck. rpf keeps the full activation → linear-level. rpfnox0 removes the curved coordinate; **rpf = rpfnox0 → curvature inert**.

### PCA *(Setting A — harm monitor, not GCG/PAIR)*
- These are **harmful behaviors vs benign** (a harm monitor), not real attacks.
- Clean harmful behaviors & benign = separate clusters; a line separates them.
- Orange = those harmful behaviors **after the obfuscation attack** (model jailbroken into complying, activation hidden) → dragged into benign cluster. Overlap is **adversarial**, beats every probe regardless of geometry.

### The twist (NOT jailbreak detection)
- Same activations, different job: **rank similar harmful prompts** (like search).
- Curvature **hurts the decision, helps the ranking** (+0.13, more with a deeper tree). One knob trades discrimination for ranking.

### MMLU — general
- Same effect on **MMLU academic subjects, ground-truth labels**; gap **+0.10 → +0.20**, grows with depth.
- **Not surface-level** — plain cosine is weak there → real hierarchy structure.

### Methodology
- Standard "is it curved?" stats are unreliable in high-D (call a sphere hyperbolic). We built a calibrated one.
- Token space **is** hyperbolic, but the **harm direction is linear** → exactly why detection doesn't benefit.

### Bottom line
- Hyperbolic doesn't help jailbreak detection (clean or adaptive) — linear's best. It helps hierarchical **ranking**, generally. One knob, two opposite effects.

### Next steps / my ask
- Publishable now as negative + dissociation. One thing I'd add: a **second model**.
- Attack-family test already done — curved ties the linear probe there too.
- **Ask them: publish now, or invest in a second model first?**

---

## Back-pocket answers
- **Harm detection or jailbreak detection?** → Setting A's probe = harm monitor (harmful vs benign); jailbreak = the obfuscation that evades it. Setting B (real GCG/PAIR) = attack detector. Hyperbolic helps neither.
- **Attack-trained probe = higher budget?** → Open. Might help for known attacks, but overfits to signatures + attacker moves off them (causal separability). Only ran clean open-set on it (~0.99), never adaptive — comparison unrun. Good next experiment.
- **λ=0.5 rpf better?** → Noise — they cross both ways, all near zero. Equal, both bypassed.
- **Is it hyperbolic at all?** → Token space yes, harm direction linear. Both true at once.
- **Isn't 'linear perfect' the reason?** → Only clean. Adaptive: we tested the hypothesis; fails via causal separability.
- **Is the positive a jailbreak result?** → No — content ranking, different task, not a detector.
- **What's it good for?** → Ranking/retrieval over hierarchies — triage, find-similar-cases. Not detection.
- **Is the positive big?** → Modest (~+0.1–0.2 mAP), ranking-only, grows with dimension, one model. Real but modest.
- **Gradient masking?** → One apparent "resistance" (λ=0.1) was a saturating-output artifact; corrected, it's bypassed like the rest.
