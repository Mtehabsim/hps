# 2026-06 cleanup — archived top-level scripts

Top level was stripped to the active core (5 categories) + the obf_reps framework + data/output dirs.
Moves are non-destructive: restore any file with `git mv archive/2026-06-cleanup/<f>.py .`

## Kept at top level (still active)

| Category | Files |
|---|---|
| HPS + C4 | `hps_core.py`, `statistical_tests.py` |
| RPF | `rpf_on_cache.py` |
| Visualization | `plot_obfuscation.py` |
| Data / extraction | `dataset.py`, `utils.py`, `config.py` |
| Experiment cluster (current findings) | `hyperbolic_retrieval.py` (retrieval/dissociation core), `hierarchical_detector.py` (ProtoNet detector + helpers), `mmlu_generality_check.py` (was `mmlu_taxonomy.py`), `openset_harm_topics.py` (was `openset_detection.py`), `openset_attack_families.py` (was `openset_attacks.py`) |
| Untouched | `obfuscated-activations/` (probe configs + obf_reps framework) |

## Renames (history preserved via git mv)
- `mmlu_taxonomy.py` → `mmlu_generality_check.py`
- `openset_detection.py` → `openset_harm_topics.py`
- `openset_attacks.py` → `openset_attack_families.py`
- (`hyperbolic_retrieval.py` and `hierarchical_detector.py` kept their original names.)

## Archived here (superseded / one-off analyses)
attack_cost_curve, calibration_panel, curvature_sweep, data_driven_hierarchy, embedding_distortion,
harm_taxonomy, harm_vs_dataset_eval, helm_token_curvature, inspect_cache, label_agreement,
latest_experiments, norm_controlled_eval, plot_lambda_sweep, plot_leaves, plot_masking,
radial_distribution_check, verify_curvature_claim.

Verified at archive time: no kept script imports any archived module; all kept scripts parse and
their cross-imports resolve.
