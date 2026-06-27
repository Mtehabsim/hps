# How to run the new analysis scripts

Two new analysis scripts have been added that need to run on your **DGX server**
(where the activation cache `results/llama3_activations_cache.npz` lives).

Both scripts have been verified to work end-to-end on synthetic data locally.

---

## Pre-flight check

```bash
# Confirm the cache exists on the DGX
ls -lh results/llama3_activations_cache.npz

# Confirm dependencies (likely already installed)
python -c "import torch, sklearn, scipy, numpy; print('OK')"

# Sanity-check the analysis modules parse
python -c "import ast; ast.parse(open('hps_core.py').read()); print('hps_core OK')"
python -c "import ast; ast.parse(open('statistical_tests.py').read()); print('statistical_tests OK')"
python -c "import ast; ast.parse(open('radial_distribution_check.py').read()); print('radial_distribution_check OK')"
```

If `hps_core.py` is missing, copy it from your local mirror — it contains
the self-contained HPS primitives (`LorentzProjection`, `contrastive_loss`,
`extract_trajectory_features`) without transformer dependencies, so the
analysis scripts run in seconds without loading the LLM.

---

## Script 1 — Statistical significance tests

**Purpose:** Add bootstrap CIs and formal hypothesis tests on HPS vs C4
(addresses evaluator critical issue C4).

**Run:**
```bash
# Recommended (5 seeds, 10K bootstrap — runs in ~5-10 minutes on GPU)
python statistical_tests.py --n_seeds 5 --n_bootstrap 10000

# Quick sanity check (1 seed, 1K bootstrap — runs in ~1 minute)
python statistical_tests.py --n_seeds 1 --n_bootstrap 1000

# Most rigorous (10 seeds, 20K bootstrap — runs in ~15-20 minutes on GPU)
python statistical_tests.py --n_seeds 10 --n_bootstrap 20000
```

**Output:** `results/statistical_tests.json` — full results
- Per-seed AUROC and TPR @ 5% FPR for HPS and C4
- Bootstrap 95% CIs on each metric
- Paired bootstrap on (HPS - C4) difference + p-value
- McNemar's test on per-example correctness
- Cohen's d effect size

**What you're looking for:**
- The `paired_bootstrap.auroc.p_value_two_sided` field — if > 0.05, the
  HPS-vs-C4 difference is **NOT statistically significant**, confirming
  our claim that geometry doesn't help at saturation.
- `mcnemar.p_value` — should also be > 0.05 if methods agree on per-example
  predictions.

---

## Script 2 — Radial distribution verification

**Purpose:** Verify the counterintuitive radial-distribution finding
(benign median = 3.71, attack median = 3.24) is **robust** across
training seeds, epochs, and curvature κ values. Addresses evaluator
critical issue C6.

**Run:**
```bash
# Recommended (5 seeds × 4 epoch checkpoints × 4 κ values — ~15-20 min on GPU)
python radial_distribution_check.py --n_seeds 5 \
    --total_epochs 50 --epochs_to_check 5 10 25 50 \
    --kappas 0.1 0.5 1.0 2.0

# Quick (2 seeds × 2 checkpoints × 2 κ values — ~5 min)
python radial_distribution_check.py --n_seeds 2 \
    --total_epochs 20 --epochs_to_check 5 20 \
    --kappas 0.1 1.0
```

**Output:**
- `results/radial_distribution_check.json` — numeric results
- `results/figs/radial_check_seeds.png` — distributions per seed
- `results/figs/radial_check_epochs.png` — distributions per training epoch
- `results/figs/radial_check_kappas.png` — distributions per κ value

**What you're looking for:**

The `summary` field in the JSON tells you the headline:
- `seeds_inverted`: e.g. `"5/5"` means all 5 seeds show benign at higher
  radial position than attacks (confirms the inversion is robust)
- `epochs_inverted` and `kappas_inverted`: same pattern

**Interpretation:**
- **All N/N inverted** → the inversion is robust; original geometric
  hypothesis is **wrong**; this is mechanistic evidence supporting our
  negative finding
- **Mixed (e.g. 3/5)** → the observation depends on training conditions;
  weaker but still notable
- **0/N inverted** → the original observation was a fluke; need to
  re-examine

---

## Script 3 — Vicuna failure diagnostic

**Purpose:** Identify exactly why HPS fails on Vicuna but works on Llama-3.
Tests 6 distinct hypotheses to pin down the mechanism (addresses the
"why does HPS fail on Vicuna" research question).

**Prerequisite:** You need both `results/llama3_activations_cache.npz` AND
`results/vicuna_activations_cache.npz` (the cache from `cross_model_compare.py
--extract`).

**Run:**
```bash
# Full diagnostic (~5-10 min on GPU)
python vicuna_diagnostic.py

# Skip H5 (HPS retraining) for fast diagnostic
python vicuna_diagnostic.py --skip_h5
```

**Output:**
- `results/vicuna_diagnostic.json` — full numeric results for all 6 hypotheses
- `results/figs/vicuna_diag_h3_per_layer.png` — per-layer separability comparison
- `results/figs/vicuna_diag_h5_loss.png` — HPS training loss curves on both LLMs
- `results/figs/vicuna_diag_h4_refusal.png` — refusal direction strength bar chart

**Hypotheses tested:**

| H | Hypothesis | What it tests |
|---|---|---|
| H1 | Vicuna activations are less tree-like (less hyperbolic) | Gromov δ-hyperbolicity per layer |
| H2 | 64-dim projection bottleneck loses Vicuna signal | C4 forced to 64-dim via PCA — does it still work? |
| H3 | Safety signal is spread across many layers on Vicuna | Per-layer LR probe AUROC distribution |
| H4 | Vicuna's refusal direction is intrinsically weaker (no RLHF) | d′ along mean(harmful)−mean(benign) |
| H5 | HPS contrastive training converges worse on Vicuna | Final loss + loss history |
| H6 | Vicuna activations have higher effective dimensionality | Eigenvalue spectrum / participation ratio |

**Interpreting verdicts:**

The script prints a summary table like:
```
SUMMARY OF HYPOTHESES
  H1: δ-hyperbolicity                  → SUPPORTS_H1   (or CONTRADICTS_H1, INCONCLUSIVE_H1)
  H2: Capacity bottleneck (PCA64)      → CONTRADICTS_H2
  H3: Per-layer signal concentration   → VICUNA_MORE_SPREAD
  ...
  PRIMARY EXPLANATIONS: H1 hyperbolicity, H3 per_layer_signal
```

**Decision rules from the verdicts:**

- **H1 supports + H6 supports** → Vicuna intrinsically harder for any 64-dim
  projection. Curing requires bigger dim or different geometry.
- **H2 supports** → Just increase projection dim. Easy fix.
- **H2 contradicts + H5 supports** → Optimization issue. Try better optimizer,
  longer training, or different κ.
- **H3 says VICUNA_MORE_SPREAD** → Layer selection matters; rescan layers.
- **H4 supports + H6 supports** → Vicuna is intrinsically harder; switch to
  C4 on Vicuna. Inherent property of the model, not method.
- **H1 contradicts + H4 supports** → Safety signal is weak overall on Vicuna,
  not a hyperbolic-specific problem.

**What to do based on results:**
- If H1 supports → write up "hyperbolic doesn't fit Vicuna activation geometry"
- If H2 supports → run an experiment with higher projection dim
- If H3+H4 support → conclude "Vicuna's safety alignment is too weak for
  geometric methods; this is an alignment/RLHF issue, not a method issue"
- If everything is INCONCLUSIVE → the failure may be due to some
  unexamined factor (worth follow-up)

---

## After running both scripts

Once you have `results/statistical_tests.json` and
`results/radial_distribution_check.json`, the paper / `research_journey.md`
can cite them with formal numbers like:

> "On Llama-3-8B (multi-seed, n=5), HPS achieves AUROC = 1.000 ± 0.000,
> C4 achieves AUROC = 1.000 ± 0.000. The paired bootstrap test gives
> ΔAUROC = +0.000 [95% CI: -0.001, +0.001], p = 0.84 (not significant
> at α=0.05). McNemar's test on per-example correctness yields p = 0.92
> (not significant). The observed difference is consistent with statistical
> noise."

> "The empirical radial distribution inversion (benign at higher radial
> position than attacks) is robust across all 5 training seeds, 4 epoch
> checkpoints (5/10/25/50), and 4 curvature values (κ ∈ {0.1, 0.5, 1.0,
> 2.0}). This provides mechanistic evidence that the contrastive loss
> finds an arbitrary discriminative direction, not the hypothesized
> hierarchical structure."

---

## If something fails

**`FileNotFoundError: Cache not found...`**
The activation cache hasn't been generated. Run `hps_llama3.py` first,
or copy the cache from wherever you stored it.

**`ModuleNotFoundError: hps_core`**
Copy `hps_core.py` from your local mirror to the DGX.

**Out of memory:**
Use `--device cpu` to run on CPU (slower but works for small caches).
HPS training on these activation caches is small (4096-dim × 6 layers ×
~5K samples = ~480 MB of float32) and fits easily in CPU RAM.

**`scipy` import error:**
`pip install scipy` (needed for McNemar's test in `statistical_tests.py`).

---

## Files added in this session

| File | Purpose |
|---|---|
| `hps_core.py` | Self-contained HPS primitives (no transformer dependencies) |
| `statistical_tests.py` | Bootstrap CIs + paired tests + McNemar's |
| `radial_distribution_check.py` | Multi-seed/epoch/κ verification of radial inversion |
| `test_synthetic.py` | Local-only sanity test using fake data |
| `RUN_INSTRUCTIONS.md` | This file |
