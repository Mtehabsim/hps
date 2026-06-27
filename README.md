# HPS — Hyperbolic Geometry for LLM Safety Probes

Does measuring distances in **curved (hyperbolic)** space instead of ordinary **flat (Euclidean)** space
help safety probes on LLM hidden states? Studied on **Llama-3-8B-Instruct**.

## Motivation & idea

A **safety probe** is a small classifier on an LLM's internal activations that flags harmful or jailbroken
behavior. Probes are usually trained in ordinary **flat (Euclidean)** space. Two facts make **curved
(hyperbolic)** space a tempting alternative:

1. **Hierarchies fit hyperbolic space far better than flat space** — negative curvature gives
   exponentially more room as you move outward, which is exactly what a branching tree (a taxonomy)
   needs. (Nickel & Kiela 2017; Sala et al. 2018.)
2. **LLM token spaces are already hyperbolic** — recent work (HELM, 2025) shows the vocabulary subspace
   of LLMs has intrinsic hyperbolic structure.

**The idea:** harm has a natural hierarchy (violence → weapons → bombs), so a probe that measures distance
in *curved* space — respecting that hierarchy — might **detect harm more accurately** and/or be **harder to
evade**. This repo tests that hypothesis rigorously: a purpose-built hyperbolic probe (HPS) and controlled
variants vs. a strong linear baseline (C4), on clean detection, adaptive jailbreak attacks, and — when the
detection idea failed — on the *retrieval* task where curved geometry is theoretically supposed to shine.

**What we found** (full story in `markdowns/PROJECT_WALKTHROUGH.md`): curvature does **not** help the
flat harm *decision* (a linear probe already separates it; strong curvature even hurts), but on the *same*
representations it **does** help *ranking/retrieving* the harm hierarchy — a controlled
**discrimination-vs-ranking dissociation** (one curvature knob, opposite signs), which also replicates on
non-harm MMLU subjects.

**Headline:** No for *detection* (a linear probe wins, and strong curvature hurts) — but on the *same*
representations, the curvature that hurts the flat decision **helps hierarchical retrieval** of harm
categories. One curvature knob, opposite signs: a **discrimination-vs-ranking dissociation**
(*faithful ≠ discriminative*), which also replicates on MMLU subjects. See `markdowns/hps_preprint.md`.

## Key terms

**Probes** (all read last-token hidden states; the standard layer set is **`[0, 2, 17, 24, 28, 31]`** — 6 layers):

- **C4** — the **linear baseline**: an independent **logistic regression per layer** over the 6 layers, scores combined (late fusion). Plain, flat, strong. This is what the geometric probes must beat.
- **HPS** (Hyperbolic Projection Sentinel) — the main hyperbolic probe: project each activation onto a **Lorentz hyperboloid** (4096→64), compute **12 cross-layer "trajectory" features** (radial / curvature / displacement), then an LR head.
- **rpf** (`raw_proj_full`) — the "purest" hyperbolic lift: take the **full 4096-dim** activation, add the Lorentz radial coordinate **`x0 = sqrt(1/k + ‖x‖²)`**, mean-pool over layers, single LR. No 12-feature bottleneck.
- **rpfnox0** — **identical to rpf but without the `x0` coordinate** (so it's Euclidean). `rpf` vs `rpfnox0` isolates *exactly* the hyperbolic contribution (an isometry-matched control); they come out equal → the lift is inert for the flat decision.
- **ProtoNet** — the retrieval/dissociation probe: a linear encoder + one learned **prototype per category**; classifies/ranks by **distance to prototypes**. Run twice — **hyperbolic** (geodesic/Lorentz distance) vs **Euclidean** (flat distance) — *identical except the distance metric*, so any gap is purely the geometry.

**Geometry**

- **Hyperbolic / Lorentz / Poincaré** — negatively-curved space with exponentially more room outward → good for trees/hierarchies. **κ / curvature `c`** = how curved (`c→0` ≈ flat).
- **embedding distortion + dimension-matched random floor** — the *trustworthy* "is this space curved?" test (δ-hyperbolicity and Ollivier-Ricci are unreliable at LLM dimension); only the margin **below a random-Gaussian floor** counts.

**Metrics**

- **AUROC** — ranking/separation quality (1.0 perfect, 0.5 chance).
- **TPR@1%FPR** — the operational detection metric: fraction of harmful caught while false-flagging ≤1% of benign.
- **mAP** — mean average precision; retrieval quality ("do same-category items rank near the top?"). The metric where curvature helps.
- **typed accuracy** — fraction of prompts assigned the correct category (14-way). The classification metric where curvature ties/loses.

**Setting**

- **obf_reps** (Bailey et al., *Obfuscated Activations*) — the adaptive attacker-vs-probe game: an attacker optimizes a prompt to **jailbreak the model AND make the probe score it benign**. **λ** = how much the attacker weights jailbreaking vs evading the probe.
- **gradient masking** — false robustness: a saturating output (sigmoid) kills the attacker's gradient so a probe *looks* unbeatable; fixed by a non-saturating (softplus) signal.

## Layout

```
src/                            all code + data + the obf_reps framework
├── hps_core.py                 HPS / Lorentz projection + 12 trajectory features
├── statistical_tests.py        trains & compares HPS vs C4 (linear) with stats
├── rpf_on_cache.py             the raw-projection-full (rpf) probe
├── hyperbolic_retrieval.py     retrieval/dissociation core (ProtoNet, mAP, dim sweep,
│                               bootstrap, label-noise, concentration-immune baselines)
├── hierarchical_detector.py    prototype detector + shared helpers (tpr_at_fpr, extract_benign)
├── mmlu_generality_check.py    MMLU subject hierarchy — not-harm-specific / depth-generic check
├── openset_harm_topics.py      open-set detection: hold out a whole harm topic
├── openset_attack_families.py  open-set detection: hold out a whole jailbreak family (+ format control)
├── plot_obfuscation.py         PCA figure: the attack drags harm into the benign region
├── dataset.py / utils.py / config.py   data loaders, extraction, constants
├── results/                    cached activations, result JSONs, figures
├── data/                       raw inputs
└── obfuscated-activations/     obf_reps adaptive-attack framework + probe configs (Bailey et al.)

markdowns/
├── hps_preprint.md             the paper (negative detection + positive dissociation + methodology)
├── PROJECT_WALKTHROUGH.md      plain-language narrative + current findings (start here)
└── presentation.md             slide deck

figures/                        figures for talks/paper
archive/                        superseded scripts, the raw lab notebook, strengthen_negative/
```

## Running

Scripts live in `src/` and resolve their data paths relative to themselves, so run from `src/`:

```bash
cd src
pip install -r requirements.txt
# most scripts have a no-GPU sanity check:
python hyperbolic_retrieval.py --selftest
```

The geometry/retrieval scripts read cached activations from `src/results/*.npz`; the obf_reps
adaptive-attack experiments run from `src/obfuscated-activations/` (see its README + `configs/`).

## Start here

1. **`markdowns/PROJECT_WALKTHROUGH.md`** — the story, the findings, and what each script does.
2. **`markdowns/hps_preprint.md`** — the formal write-up.

