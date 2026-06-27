# HPS Sentinel — Validation Suite

Empirical validation scripts for the Hyperbolic Physiological Sentinel framework.

## Structure

```
hps_sentinel/
├── config.py               — model, layers, paths (edit this first)
├── utils.py                — shared math: extraction, hyperbolic ops, curvature
├── dataset.py              — built-in prompt dataset (benign / adversarial / dual-use)
├── requirements.txt
│
├── test1_gromov_delta.py   — Is the residual stream intrinsically hyperbolic?
├── test2_hierarchy.py      — Does Poincaré projection encode abstraction depth?
├── test3_traced_space.py   — Do adversarial trajectories look geometrically different?
├── test4_baseline_comparison.py — Does hyperbolic geometry beat a Euclidean probe?
│
├── run_all.py              — Run all tests + summary report
│
├── results/                — JSON outputs (auto-created)
└── plots/                  — PNG visualisations (auto-created)
```

## Quick Start

```bash
pip install -r requirements.txt

# Run the fastest, most informative test first
python test3_traced_space.py

# Run everything
python run_all.py

# Run specific tests
python run_all.py --tests 2,4
```

## Changing the Model

Edit `config.py`:

```python
MODEL_NAME   = "meta-llama/Llama-3.2-1B"   # or any HF causal LM
TARGET_LAYERS = [3, 7, 11, 15, 19]          # sample ~5–8 layers across depth
```

For Llama-3-8B (32 layers), use:
```python
TARGET_LAYERS = [4, 8, 11, 16, 20, 24, 28, 31]
```

## Decision Gates

| Test | Question | Go-criterion |
|------|----------|--------------|
| 1 | Is the space hyperbolic? | avg max-δ < 1.0 for most layers |
| 2 | Does radial coord = depth? | ≥70% concept pairs correctly ordered |
| 3 | Is the trajectory signal real? | Hyperbolic AUROC > 0.70 |
| 4 | Does geometry add value? | Hyperbolic AUROC > Euclidean AUROC + 0.03 |

If Tests 1 and 2 fail → the Poincaré projection may be imposing artificial structure.
If Tests 3 and 4 fail → the curvature signal may not exist in this model.

## Outputs Per Test

**Test 1:**
- `results/test1_gromov_delta.json`
- `plots/test1_gromov_delta_by_layer.png`
- `plots/test1_gromov_delta_heatmap.png`

**Test 2:**
- `results/test2_hierarchy.json`
- `plots/test2_hierarchy_pairs.png`
- `plots/test2_hierarchy_boundary.png`
- `plots/test2_poincare_disk.png`

**Test 3:**
- `results/test3_traced_space.json`
- `plots/test3_traced_euclidean.png`
- `plots/test3_traced_hyperbolic.png`
- `plots/test3_example_trajectories.png`
- `plots/test3_curvature_distribution.png`

**Test 4:**
- `results/test4_baseline_comparison.json`
- `plots/test4_roc_comparison.png`
- `plots/test4_feature_importance.png`
- `plots/test4_fpr_dual_use.png`
- `plots/test4_score_distributions.png`

## Interpretation Guide

### Gromov δ (Test 1)
- δ ≈ 0: perfectly tree-like → hyperbolic projection valid
- δ < 1: mildly hyperbolic → projection useful
- δ > 2: Euclidean-like → projection adds noise

### TRACED Space (Test 3)
- Top-right quadrant (high displacement, high curvature): jailbreak zone
- Bottom-right (high displacement, low curvature): correct reasoning
- Top-left (low displacement, high curvature): hallucination / hesitation loops

### Probe Comparison (Test 4)
- ΔAUROC > 0.03: hyperbolic geometry meaningfully adds signal
- ΔAUROC ≈ 0: Euclidean features capture the same information
- Dual-use FPR > 0.20: too many false blocks on legitimate prompts
