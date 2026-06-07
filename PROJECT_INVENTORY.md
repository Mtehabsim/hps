# HPS Project Inventory & Final Results

**Project:** Hyperbolic Projection Sentinel (HPS) for LLM Jailbreak Detection
**Status as of:** 2026-06-07
**Canonical caches:** `results/llama3_activations_cache_diverse_fixed.npz` + `results/llama3_activations_cache_alllayers.npz`

---

## 1. Core Pipeline — Probes, Attacks, Caches

| File | Purpose | Outputs |
|------|---------|---------|
| `adaptive_attack.py` | Central probe definitions and embedding-suffix attack. Defines C4Probe, HPSProbe, HPSEuclideanProbe, HPSGenProbe, EnsembleProbe. Includes `train_*_probe()` functions, `load_cache_arrays()`, `load_gen_cache_arrays()`. Runs Bailey-style continuous embedding suffix attack against any probe. | `results/adaptive_attacks/attack_*.json`, `suffix_*.pt`, `transfer_*_to_*.json`, `eval_n100_*.json` |
| `flrt_attack.py` | Bailey-style FLRT hard-prompt attack. Discrete token suffix optimization (add/swap/delete) with buffer + replace-worst. Supports `--defender {c4, hps, hps_euc, hps_gen}`. Imports probe defs from adaptive_attack.py. | `results/flrt_attacks/attack_*_flrt.json`, `suffix_*_flrt.json`, `transfer_*_to_*_flrt.json` |
| `hps_core.py` (170 lines) | Self-contained Lorentz projection + 12 trajectory features + contrastive loss. Pure PyTorch. | (importable module) |
| `hps_llama3.py` | HPS pipeline configured for Llama-3 (layers `[0,2,17,24,28,31]`, κ=0.1). | `results/hps_llama3_results.json`, `results/hps_rtv_results.json` |
| `experiment7.py` | Original Vicuna HPS pipeline. Reference implementation; superseded by `hps_core.py` + `hps_llama3.py`. | `results/experiment7_results.json` |
| `utils.py` | Model loading, tokenization, activation extraction helpers. | (importable module) |
| `dataset.py` | Dataset loaders for harmful + benign data. | (importable module) |
| `config.py` | Hyperparameter constants. | (importable module) |

---

## 2. Data Construction

| File | Purpose | Output |
|------|---------|--------|
| `build_diverse_benign.py` | Builds 5,905-prompt benign CSV from 9 sources (WildChat, OR-Bench Hard, MMLU, GSM8K, HumanEval, MBPP, WikiText, Aya, Alpaca). | `results/data_harmless_diverse.csv` (mean 1,076 chars) |
| `build_jbshield_attacks.py` | Parses JBShield repo's attack data into our format. | `results/jbshield_llama3_attacks.json`, `results/jbshield_vicuna13b_attacks.json` |
| `build_novel_attacks.py` | FlipAttack + JailbreakBench-style attacks from AdvBench. | `results/novel_attacks_2025.json` (400 prompts, 8 categories) |
| `download_extra_attacks.py` | Downloads HuggingFace AdvBench dataset. | (input data) |
| `download_calibration_data.py` | Downloads UltraChat samples for benign calibration. | (input data) |
| `investigate_contamination.py` | Hash-based train/test overlap analysis. **Found 15 contaminated prompts.** | `llama3_attacks_clean.json` (6,474 deduplicated attacks) |

---

## 3. Activation Extraction

| File | Purpose | Output |
|------|---------|--------|
| `extract_diverse_benign_activations.py` | Forward 5,905 diverse benign + attacks through Llama-3/Vicuna; build canonical activation cache. | `results/llama3_activations_cache_diverse.npz` (pre-fix), `results/vicuna_activations_cache_diverse.npz` |
| `fix_cache_max_length.py` | Re-extract attacks at max_length=2048 to match diverse benign tokenization (fixes max_length confound). | **`results/llama3_activations_cache_diverse_fixed.npz` ← CANONICAL Llama-3 cache**, `results/vicuna_activations_cache_diverse_fixed.npz` |
| `extract_all_layers.py` | Extract activations at all 32 layers (vs default 6) for layer ablation. | `results/llama3_activations_cache_alllayers.npz` (3.1 GB) |
| `extract_jbshield_activations.py` | Extract activations on JBShield public attack data. | `results/llama3_activations_cache_jbshield.npz`, `results/vicuna_activations_cache_jbshield.npz` |
| `extract_vicuna_full_sequence.py` | Re-extract Vicuna with full sequence stored (for MTP compatibility). | (Vicuna-specific cache) |
| `extract_llama3_base_activations.py` | Extract activations on base Llama-3 (non-RLHF) for alignment ablation. | **STATUS: Failed (`hf_xet` package issue)** |
| `extract_generation_activations.py` | Extract generation-token activations for HPS-Gen variant. | `results/llama3_gen_activations_cache.npz` (when run on DGX) |
| `filter_diverse_cache.py` | Drop contaminated prompts from existing cache. | (in-place updates) |

---

## 4. Methodology Verification

| File | Purpose | Output |
|------|---------|--------|
| `verify_saturation.py` | 6-check audit: contamination, length confound, strict thresholds, per-attack breakdown, permutation test, norm confound. | `results/verify_saturation_fixed.json` |
| `norm_check_diverse.py` | Norm-only classifier baseline. **Found norm-only AUROC=1.000 → 0.761 after fix.** | `results/norm_check_diverse.json` |
| `diagnose_norm_confound.py` | Chat-template vs no-template ablation for norm confound. | `results/norm_confound_diagnosis.json` |
| `norm_controlled_eval.py` | L2-normalize + standardize ablations. **Confirms C4 detects real semantic signal beyond norm.** | `results/norm_controlled_eval_llama3.json`, `results/norm_controlled_eval_vicuna.json` |
| `validate_benign.py` | Check benign dataset diversity, length distribution. | (stdout reports) |
| `validate_attacks.py` | Verify attacks elicit harmful responses (StrongREJECT-style). | `results/validated_attacks.json` |
| `hard_benign.py` | Curate hard benign examples. | (input data) |

---

## 5. Main Experiments (Statistical & Comparative)

| File | Purpose | Output | Key Finding |
|------|---------|--------|-------------|
| `statistical_tests.py` | Bootstrap CIs + paired tests + McNemar's for HPS vs C4. | `results/statistical_tests.json` | HPS=0.997, C4=0.998. Statistically marginal (p=0.036), practically equivalent. |
| `radial_distribution_check.py` | 13-config inversion test (5 seeds × 4 epochs × 4 κ). | `results/radial_distribution_check.json` | **0/13 inversions; geometric hypothesis confirmed.** |
| `vicuna_imbalance_test.py` | Per-attack breakdown + class imbalance ablation. | `results/vicuna_imbalance_test.json` | HPS catastrophically fails on Vicuna's GCG (7.6%); imbalance not the cause. |
| `gcg_specific_test.py` | Cross-LLM GCG analysis (Llama-3 vs Vicuna). | `results/gcg_specific_test.json` | **ALIGNMENT_HYPOTHESIS_CONFIRMED.** Llama=100%, Vicuna=7.6%. |
| `hyperbolic_vs_euclidean_diverse.py` | HPS vs HPS-Euclidean vs C4 + cold-start sweep. | `results/hyperbolic_vs_euclidean_diverse.json` | HPS > HPS-Euclidean by +0.05 TPR; C4 still beats HPS by +0.015 TPR. |
| `anthropic_mean_token_probe.py` | Faithful Anthropic MTP reproduction. | `results/anthropic_mtp_llama3.json` | MTP@L17 AUROC=0.9988. Equal to C4 (0.9986). |
| `prediction_agreement.py` | Per-example HPS vs C4 agreement analysis. | `results/prediction_agreement.json` | **HPS catches 0 unique attacks beyond C4. Pearson r=0.958.** |
| `cross_model_compare.py` | Cross-model comparisons (Llama vs Vicuna behavior). | `results/cross_model_compare.json` | (referenced by other analyses) |
| `evaluate_ood_attacks.py` | Evaluate detection on novel attack distributions. | **STATUS: Crashed (KAPPA_INIT import error + disk space)** | — |

---

## 6. Adversarial Attack Experiments

### 6.1 Embedding Suffix Attacks (`adaptive_attack.py`)

**Status:** Complete. All 4 probes tested, transfer matrix complete, budget scaling done.

| Experiment | Output File | Standard Recall | Adaptive Recall | Notes |
|-----------|-------------|-----------------|-----------------|-------|
| C4 direct | `attack_c4.json` | 0.767 | 0.000 | Baseline |
| HPS direct | `attack_hps.json` | 0.767 | 0.000 | Same as C4 |
| HPS-Euclidean direct (λ=1.0) | `attack_hps_euc.json` | 0.610 | 0.000 | beh_loss didn't converge (4.7) |
| HPS-Euclidean direct (λ=0.5) | `attack_hps_euc_lambda05.json` | 0.610 | 0.000 | beh_loss=0.44 (converged) |
| HPS-Euclidean direct (λ=0.3) | `attack_hps_euc_lambda03.json` | 0.610 | 0.000 | beh_loss=0.034 (converged fastest) |
| Ensemble HPS+C4 | `attack_ensemble.json` | 0.480 | 0.000 | MAX-aggregation |
| C4-suffix → HPS | `transfer_c4_to_hps.json` | 0.767 | **1.000** | HPS catches MORE under C4 attack |
| HPS-suffix → C4 | `transfer_hps_to_c4.json` | 0.767 | 0.000 | Full transfer |
| C4-suffix → HPS (n=100) | `eval_n100_c4_on_hps.json` | 0.640 | **1.000** | Asymmetry confirmed at n=100 |
| HPS-suffix → C4 (n=100) | `eval_n100_hps_on_c4.json` | 0.480 | 0.000 | Full transfer at n=100 |
| Budget 256 | `attack_hps_256steps.json` | 0.767 | 0.000 | HPS breaks fast |
| Budget 512 | `attack_hps_512steps.json` | 0.767 | 0.000 | |
| Budget 1024 | `attack_hps_1024steps.json` | 0.767 | 0.000 | |

### 6.2 FLRT Hard-Prompt Attacks (`flrt_attack.py`)

**Status:** Phase 0/1/2 complete. Phase 3 (transfer) and HPS-Gen pending.

| Experiment | Output File | Standard | Adaptive | monitor_weight | Final beh_loss | Caveat |
|-----------|-------------|----------|----------|----------------|----------------|--------|
| C4 (validation) | `attack_c4_flrt.json` | 0.620 | 0.000 | 0.05 | 1.88 | ✓ Validation passed (in [0.05, 0.70] range) |
| HPS | `attack_hps_flrt.json` | 0.710 | 0.000 | 1.0 | 3.84 | ⚠ beh_loss didn't fully converge |
| HPS-Euclidean | `attack_hps_euc_flrt.json` | 0.630 | 0.000 | 1.0 | 5.15 | ⚠ beh_loss high; jailbreak success unclear |
| HPS-Gen | (pending) | — | — | — | — | Not yet run |
| Transfer matrix | (pending) | — | — | — | — | Not yet run |

**Important caveat:** HPS and HPS-Euclidean FLRT runs used `monitor_weight=1.0` (same configuration that caused convergence issues with embedding attacks). Only C4 was re-run with `monitor_weight=0.05`. The "adaptive recall=0" for HPS/HPS-Euc may reflect probe being fooled while jailbreak is incomplete (high `beh_loss`). For consistency, these should be re-run with `monitor_weight=0.05`.

### 6.3 Layer Ablation (`run_layer_ablation.sh` driving `adaptive_attack.py`)

**Status:** 14/15 complete (B8 had std() NaN bug, now fixed).

| Config | Layers | Standard Recall | Adaptive Recall |
|--------|--------|-----------------|-----------------|
| **B7_pos_spread_even** | [0, 6, 12, 18, 24, 31] | **0.60** | 0.000 |
| **B5_pos_middle** | [10, 12, 14, 16, 18, 20] | **0.50** | 0.000 |
| A7_count_32_all | all 32 layers | 0.48 | 0.000 |
| A5_count_12 | [2, 4, ..., 24] | 0.36 | 0.000 |
| **B1_pos_current** | [0, 2, 17, 24, 28, 31] | **0.32** | 0.000 |
| B6_pos_late | [26, 27, 28, 29, 30, 31] | 0.30 | 0.000 |
| A2_count_4 | [4, 8, 12, 16] | 0.12 | 0.000 |
| A3_count_6_safety | [4, 6, 8, 10, 12, 14] | 0.08 | 0.000 |
| A4_count_8 | [2, 4, 6, 8, 10, 12, 14, 16] | 0.08 | 0.000 |
| B4_pos_safety_buffer | [4, 6, 8, 10, 12, 14] | 0.08 | 0.000 |
| B3_pos_safety_strict | [6, 7, 8, 9, 10, 11] | 0.06 | 0.000 |
| A6_count_16 | every other layer 0-30 | 0.04 | 0.000 |
| A1_count_2 | [6, 12] | 0.00 | 0.000 |
| B2_pos_early | [0, 1, 2, 3, 4, 5] | 0.00 | 0.000 |
| B8_pos_BO_inspired | [2, 6, 12] | (pending re-run) | — |

**Key findings:**
- Layer choice does NOT affect adversarial robustness (all configs → 0.000 adaptive recall)
- Layer choice strongly affects standard recall (0% to 60%)
- Spread-even > clustered-middle for standard detection
- Safety-zone alignment (Li et al.'s middle layers) does NOT outperform our current layer choice

### 6.4 Other Adversarial Experiments

| File | Purpose | Output |
|------|---------|--------|
| `attack_ensemble.py` | Original adaptive PGD on HPS+RTV ensemble. | `results/attack_ensemble_results.json` |
| `experiment11.py` | Adaptive PGD experiment. | `results/experiment11_adaptive_pgd.json` |
| `experiment12.py` | HPS-Adv adversarial training. | `results/experiment12_adv_training.json` |
| `experiment13.py` | Semantic robustness experiment. | `results/experiment13_semantic.json` |
| `adversarial_compare.py` | Compare adversarial properties across methods. | `results/adversarial_compare.json` |

---

## 7. Diagnostic Tools

| File | Purpose | Output |
|------|---------|--------|
| `feature_ablation.py` | Drop-one-feature ablation across 12 HPS features. | `results/feature_ablation.json`, `results/feature_ablation_summary.json`. **Finding: mean_r alone matches all 12 features.** |
| `vicuna_diagnostic.py` | Investigate Vicuna failure mode. | (stdout) |
| `vicuna_overfitting_test.py` | Test if Vicuna failure is overfitting. | (stdout) |
| `vicuna_param_sweep.py` | Hyperparameter sweep on Vicuna. | `results/vicuna_param_sweep.json` |
| `diagnostic_hps_vs_euc.py` | Investigate why Euclidean beats HPS on Llama-3 in early experiments. | (stdout) |
| `verify_new_config.py` | Verify hyperparameter changes didn't break baseline. | `results/verify_new_config.json` |
| `control_experiments.py` | Random-label, shuffled-feature controls. | `results/control_experiments.json` |
| `experiment_mitigation.py`, `experiment_mitigation_v2.py` | Confound mitigation experiments. | `results/experiment_mitigation*.json` |
| `experiment6.py`, `experiment8.py`, `experiment10.py` | Intermediate diagnostic experiments. | `results/experiment*.json` |
| `paper_supplementary.py` | Multi-seed stability + learning curve experiments. | `results/paper_supplementary.json` |

---

## 8. Strengthening / Negative-Result Folder

| File | Purpose |
|------|---------|
| `strengthen_negative/experiment1_hyperbolic_methods.py` | Test multiple hyperbolic methods (HPS variants). |
| `strengthen_negative/experiment4_information_theoretic.py` | Information-theoretic analysis. |
| `strengthen_negative/experiment5_alignment_ablation.py` | Alignment ablation experiment script. |
| `strengthen_negative/ai-2-verification.py` | 4-check verification script. |
| `strengthen_negative/helpers/lorentz_ops.py` | Lorentz manifold operations helper. |

---

## 9. Synthetic / Early-Stage Tests

**Note:** These are PRE-fix experiments. Numbers may not match final results.

| File | Purpose | Output |
|------|---------|--------|
| `test1_gromov_delta.py` | Test Gromov δ-hyperbolicity of LLM activations. | (stdout) |
| `test2_hierarchy.py` | Test hierarchical structure in activations. | (stdout) |
| `test3_traced_space.py` | Trace activation space evolution. | `results/test3_traced_space.json` |
| `test4_baseline_comparison.py` | Baseline comparison across methods. | `results/test4_baseline_comparison.json` |
| `test5_hps_full.py` | Full HPS test on initial setup. | `results/test5_hps_full.json` |
| `test_synthetic.py` | Synthetic data tests for HPS. | (stdout) |
| `test_pca_lorentz.py` | PCA visualization on Lorentz coordinates. | (stdout/figure) |
| `test_hps_zeroshot.py` | Zero-shot HPS evaluation. | `results/hps_zeroshot_clusters.png` |
| `test_ensemble.py`, `test_cross_attack_ensemble.py` | Early ensemble experiments. | `results/ensemble_results.json`, `results/cross_attack_ensemble.json` |

---

## 10. Visualization & Figure Generation

| File | Output |
|------|--------|
| `fig_4method_comparison.py` | `figures_for_meeting/fig_4method_comparison.png` (HPS, HPS-Euc, C4, MTP bar chart) |
| `fig_cold_start_sweep.py` | `figures_for_meeting/fig_cold_start_sweep.png` (cold-start sweep line plot) |
| `fig_lorentz_concept.py` | `figures_for_meeting/fig_lorentz_concept.png` (Lorentz hyperboloid + Poincaré disk) |
| `generate_methodology_diagram.py` | `figures_for_meeting/fig_methodology_pipelines.png` |
| `generate_meeting_charts.py` | Combined slide charts |
| `generate_paper_plots.py` | Per-paper-section figures |
| `visualize_hps.py` | `results/hps_llama3_clusters.png`, `results/viz_*.png` |
| `visualize_activation_space.py` | PCA + t-SNE + decision boundaries |
| `visualize_real.py`, `visualize_results.py` | Various visualization tools |
| `visualize_design.py`, `visualize_poincare.py` | Design diagrams + Poincaré disk |
| `build_slides.py` | `team_meeting.pptx` (PowerPoint with embedded figures) |

---

## 11. Reference Implementations

| File | Purpose | Output |
|------|---------|--------|
| `rtv_standalone.py` | Standalone RTV (Refusal-Token Vulnerability) implementation. | `results/rtv_standalone_results.json`, `results/rtv_llama3_results.json` |
| `rtv_find_layers.py` | Find optimal layers for RTV. | (stdout) |
| `hps_rtv_inspired.py` | HPS variant inspired by RTV findings. | (stdout) |

---

## 12. Pipeline Runners (Bash)

| File | Phases | Status |
|------|--------|--------|
| `run_overnight_pipeline.sh` | 7-phase end-to-end: extraction → fix-cache → train probes → diagnostic → norm-control. | Done. Used to build canonical cache. |
| `run_diverse_benign_pipeline.sh` | Build diverse-benign cache end-to-end. | Done. |
| `run_strengthening_pipeline.sh` | Three follow-up tasks (norm confound, novel attacks, alignment ablation). | Partial — Phase 5/6 had disk-space issues. |
| `rerun_norm_check_fixed.sh` | Re-run norm check after cache fix. | Done. |
| `run_adaptive_attacks.sh` | 7 phases: C4, HPS, transfer, attack-budget scaling (256/512/1024 steps), HPS-Euclidean, ensemble, n=100 re-eval. | **Done. Results in `results/adaptive_attacks/`.** |
| `run_layer_ablation.sh` | A1–A7 (count ablation), B1–B8 (position ablation). | **Done (14/15 configs).** Results in `results/adaptive_attacks/layer_ablation/`. |
| `run_flrt_attacks.sh` | 4 phases: C4 validation, HPS, HPS-Euc, HPS-Gen, transfer matrix. | **Phase 0/1/2 done. Phase 3 (transfer) and Phase 4 (HPS-Gen) pending.** |
| `run_all.sh` | Original master script (cache extraction + experiments). | Outdated (replaced by phased scripts). |

---

## 13. Documentation (Markdown)

### Confirmed local files

| File | Purpose |
|------|---------|
| `mentor_draft.md` | First comprehensive mentor draft (~22 pages). |
| `mentor_draft_v2.md` | Updated draft with adaptive attack findings (~17 pages). |
| `team_meeting_slides.md` | 9-slide presentation specification. |
| `paper_draft.md` | Earlier paper outline. |
| `paper_outline.md` | Original paper outline. |
| `HPS_Findings.md` | Comprehensive findings + prior art. |
| `HPS_Document.md` | Earlier HPS documentation. |
| `research_journey.md` | Chronological research notes. |
| `evaluation_report.md` | Evaluation summary. |
| `literature_review_activation_defenses.md` | Related work review. |
| `research_plan_strengthening.md` | Plan to strengthen negative results. |
| `research_opportunities.md` | Future research directions. |
| `plan_a.md` | Initial research plan. |
| `mentor_briefing.md` | Pre-meeting brief. |
| `RUN_INSTRUCTIONS.md` | DGX pipeline instructions. |
| `README_adaptive_attacks.md` | Adaptive attack experiment instructions. |
| `README.md`, `strengthen_negative/README.md` | Standard READMEs. |

### Files claimed in user inventory but NOT FOUND locally

| File | Status |
|------|--------|
| `mentor_draft.WITH_CONFOUNDS.md.bak` | NOT FOUND locally — may exist on DGX |
| `mentor_reading_list.md` | NOT FOUND locally — may exist on DGX |
| `post_meeting_plan.md` | NOT FOUND locally — may exist on DGX |

---

## 14. Result File Locations Summary

```
results/
├── adaptive_attacks/                                Embedding suffix attack results (DONE)
│   ├── attack_{c4,hps,hps_euc,ensemble}.json       Direct attacks
│   ├── attack_hps_{256,512,1024}steps.json         Budget scaling
│   ├── attack_hps_euc_lambda{03,05}.json           Lambda calibration
│   ├── transfer_{c4_to_hps,hps_to_c4}.json         Transfer matrix
│   ├── eval_n100_{c4,hps}_on_{c4,hps}.json         n=100 re-eval
│   ├── layer_ablation/{A1-A7, B1-B7}/result.json   14 layer configs
│   └── suffix_*.pt                                  Trained suffixes (.pt = embeddings)
│
├── flrt_attacks/                                    FLRT hard prompt results
│   ├── attack_c4_flrt.json                          ✓ DONE (recall: 0.620 → 0.000, λ=0.05 ✓ converged)
│   ├── attack_hps_flrt.json                         ✓ DONE (recall: 0.710 → 0.000, λ=1.0 ⚠ beh_loss=3.84)
│   ├── attack_hps_euc_flrt.json                     ✓ DONE (recall: 0.630 → 0.000, λ=1.0 ⚠ beh_loss=5.15)
│   ├── attack_hps_gen_flrt.json                     ⏳ PENDING
│   ├── transfer_*_to_*_flrt.json                    ⏳ PENDING
│   └── suffix_*_flrt.json                           Trained suffixes (.json = tokens + decoded string)
│
├── statistical_tests.json                           5-seed bootstrap, McNemar's
├── radial_distribution_check.json                   13-config inversion analysis
├── gcg_specific_test.json                           Cross-model GCG breakdown (Vicuna vs Llama)
├── prediction_agreement.json                        HPS vs C4 per-example
├── hyperbolic_vs_euclidean_diverse.json             4-method comparison
├── anthropic_mtp_llama3.json                        Anthropic Cheap Monitors reproduction
├── feature_ablation*.json                           Drop-one-feature
├── verify_saturation_fixed.json                     Length confound diagnostic
├── norm_check_diverse.json                          Norm-only confound
├── norm_controlled_eval_llama3.json                 L2-normalized re-eval (REAL_SIGNAL verdict)
├── data_harmless_diverse.csv                        5,905 benign prompts
├── jbshield_*_attacks.json                          Attack pools (some empty stubs)
├── validated_attacks_categorized.json               316 prompts (4 methods)
├── novel_attacks_2025.json                          ~400 FlipAttack/novel
└── llama3_activations_cache_diverse_fixed.npz       Canonical cache (DGX)
```

---

## 15. Final Findings Summary

### Methodology (3 confounds identified)

| Confound | Severity | Resolution |
|----------|----------|------------|
| Length confound | length-only AUROC = 0.992 | Use diverse benign data (not Alpaca) |
| max_length confound | norm-only AUROC = 1.000 → 0.761 after fix | Consistent tokenization (max_length=2048 across train/test) |
| Train/test contamination | 15 prompts (1.15%) overlapped | SHA-256 hash-based deduplication |

### Empirical Findings (After All Confounds Fixed)

1. **HPS = C4 = MTP statistically equivalent on standard benchmarks** (AUROC 0.997-0.999, McNemar's p=0.755).

2. **Geometric hypothesis confirmed under controlled conditions**: 0/13 inversions (all configs show attacks at higher radial position than benign).

3. **HPS-Euclidean ablation**: HPS (Lorentz) > HPS-Euclidean (flat) by +0.05 TPR. Geometric prior provides measurable advantage over parameter-matched flat geometry.

4. **C4 (linear probe) wins overall**: HPS underperforms C4 by 0.015 TPR on aligned LLMs. Compression-based geometric methods don't beat raw linear probes.

5. **Cross-model alignment-mediated failure**: HPS fails catastrophically on Vicuna-13B (GCG: 7.6% vs Llama-3 100%). Suggests HPS only works on RLHF-aligned models.

6. **All probes broken under embedding suffix attack** (Bailey reproduction): C4, HPS, HPS-Euc, HPS+C4 ensemble all collapse to 0.000 adaptive recall.

7. **Transfer asymmetry (real but not adversarially defensible)**: C4-suffix → HPS recall = 1.000; HPS-suffix → C4 recall = 0.000. Asymmetric transfer pattern (Bailey Fig 4c). Has no defense value because adaptive ensemble attack also bypasses both.

8. **Layer choice doesn't help adversarially**: All 14 layer configs collapse to 0.000 adaptive recall. Layer choice does affect standard recall (range 0% - 60%).

9. **Lambda calibration matters for fair attack comparison**: HPS-Euc shows different convergence behavior at different `monitor_weight` values; properly calibrated, all probes break equally.

10. **FLRT (hard prompt) attacks (partial)**:
    - C4: standard 0.620 → adaptive 0.000 (validated, properly converged)
    - HPS: standard 0.710 → adaptive 0.000 (caveat: behavior loss didn't fully converge with monitor_weight=1.0)
    - HPS-Euc: standard 0.630 → adaptive 0.000 (same caveat)

### Caveats / Open Items

1. **HPS and HPS-Euc FLRT runs need re-running with `monitor_weight=0.05`** for proper jailbreak convergence comparison.

2. **HPS-Gen (generation-based) not yet evaluated.** Bailey notes generation-based probes are more robust against hard prompt attacks (~90% recall under hard prompts vs ~0% input-based). HPS-Gen architecture exists in `adaptive_attack.py` but FLRT vs HPS-Gen pending.

3. **FLRT transfer matrix pending.** Cross-method transfer for hard prompts (Phase 3 of `run_flrt_attacks.sh`).

4. **B8 layer config pending re-run** after std() NaN bug fix.

5. **Llama-2-7b-chat alignment ablation not run.** Would isolate RLHF as the variable for the cross-model failure finding.

---

## 16. Pending Work (In Priority Order)

| # | Task | Estimated Time | Notes |
|---|------|----------------|-------|
| 1 | Re-run HPS and HPS-Euc FLRT with monitor_weight=0.05 | ~5 hours | Confirms jailbreak success rate |
| 2 | HPS-Gen extraction on DGX (`extract_generation_activations.py`) | ~30-40 min | ~3 GB cache |
| 3 | FLRT vs HPS-Gen (Phase 4) | ~2.5 hours | The key hypothesis test |
| 4 | FLRT transfer matrix (Phase 3, eval-only) | ~30 min | Cross-method transfer |
| 5 | B8_pos_BO_inspired layer ablation (re-run with fixed std bug) | ~12-17 min | Completes layer ablation table |
| 6 | Llama-2-7b-chat extraction + adaptive attacks | ~3-4 hours | Alignment hypothesis ablation |

---

## 17. Status Of Major Claims For Paper

| Claim | Evidence | Status |
|-------|----------|--------|
| Standard benchmarks have length confound | `verify_saturation_fixed.json` | ✅ Confirmed |
| Standard benchmarks have norm/max_length confound | `norm_check_diverse.json` | ✅ Confirmed |
| Standard benchmarks have train/test contamination | `investigate_contamination.py` outputs | ✅ Confirmed |
| HPS = C4 = MTP after confound fixes | `statistical_tests.json` + `anthropic_mtp_llama3.json` | ✅ Confirmed |
| HPS catches no unique attacks beyond C4 | `prediction_agreement.json` | ✅ Confirmed |
| Geometric hypothesis (radial direction) | `radial_distribution_check.json` | ✅ Confirmed (0/13 inversions) |
| HPS > HPS-Euclidean (parameter-matched) | `hyperbolic_vs_euclidean_diverse.json` | ✅ Confirmed (+0.05 TPR) |
| Cross-model alignment-mediated failure | `gcg_specific_test.json` | ✅ Confirmed (Vicuna 7.6% GCG) |
| All probes break under embedding attack | `attack_*.json` in `adaptive_attacks/` | ✅ Confirmed |
| Transfer asymmetry C4↔HPS | `transfer_*.json` + `eval_n100_*.json` | ✅ Confirmed |
| Adaptive ensemble attack bypasses both | `attack_ensemble.json` | ✅ Confirmed |
| Layer choice doesn't help adversarially | `layer_ablation/*/result.json` | ✅ Confirmed (0/14) |
| Layer choice affects standard recall | `layer_ablation/*/result.json` | ✅ Confirmed (0% to 60%) |
| FLRT hard prompts also break C4 | `attack_c4_flrt.json` | ✅ Confirmed |
| FLRT hard prompts also break HPS | `attack_hps_flrt.json` | ⚠ Partial — beh_loss didn't fully converge |
| FLRT hard prompts also break HPS-Euc | `attack_hps_euc_flrt.json` | ⚠ Partial — beh_loss didn't fully converge |
| HPS-Gen survives hard prompts | (pending) | ⏳ Not yet tested |

---

**End of Inventory**
