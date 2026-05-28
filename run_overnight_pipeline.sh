#!/bin/bash
# run_overnight_pipeline.sh
# ---------------------------------------------------------------------------
# Single comprehensive pipeline for overnight execution.
# Runs all pending experiments in dependency order with failure recovery.
#
# Total expected time: ~16-20 hours
# Disk usage: ~150-200 GB (multiple activation caches)
#
# Phases:
#   PHASE 1: Cache fix (max_length=2048 consistency)         ~5 hrs
#   PHASE 2: Verification + main experiments rerun           ~1 hr
#   PHASE 3: Anthropic mean-token probe (faithful baseline)  ~10 min
#   PHASE 4: JBShield integration                            ~5 hrs
#   PHASE 5: Alignment ablation (Llama-3 base)               ~4 hrs
#   PHASE 6: Novel attacks (FlipAttack + JBB)                ~4 hrs
#   PHASE 7: Diagnostics (only if confounds persist)         ~1 hr
#
# Usage:
#   ./run_overnight_pipeline.sh
#
#   # Skip specific phases if needed:
#   SKIP_PHASE_4=1 ./run_overnight_pipeline.sh   # skip JBShield
#   SKIP_PHASE_5=1 ./run_overnight_pipeline.sh   # skip alignment
#   SKIP_PHASE_6=1 ./run_overnight_pipeline.sh   # skip novel attacks
#
# All phases use skip-if-exists so the script can be safely re-run if
# interrupted.
# ---------------------------------------------------------------------------

set -u    # error on undefined variables
# NOTE: not using set -e — we want phases to keep running even if one fails

mkdir -p results results/figs

START_TIME=$(date +%s)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DIVERSE_CSV="results/data_harmless_diverse.csv"
ATTACKS_CLEAN="llama3_attacks_clean.json"
ATTACKS_ORIG="llama3_attacks.json"

# Use cleaned attacks if available, else original
if [ ! -f "$ATTACKS_CLEAN" ]; then
    ATTACKS_CLEAN="$ATTACKS_ORIG"
fi

LLAMA_LAYERS="0 2 17 24 28 31"
VICUNA_LAYERS="0 2 22 31 35 39"

JBSHIELD_DIR="${JBSHIELD_DIR:-/tmp/JBShield}"

# Skip flags (override with env vars)
SKIP_PHASE_1="${SKIP_PHASE_1:-0}"
SKIP_PHASE_2="${SKIP_PHASE_2:-0}"
SKIP_PHASE_3="${SKIP_PHASE_3:-0}"
SKIP_PHASE_4="${SKIP_PHASE_4:-0}"
SKIP_PHASE_5="${SKIP_PHASE_5:-0}"
SKIP_PHASE_6="${SKIP_PHASE_6:-0}"
SKIP_PHASE_7="${SKIP_PHASE_7:-0}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log_phase() {
    local phase="$1"
    local title="$2"
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo " PHASE $phase: $title"
    echo " Started: $(date)"
    echo "═══════════════════════════════════════════════════════════════"
}

log_step() {
    local step="$1"
    local title="$2"
    echo ""
    echo "───────────────────────────────────────────────────────────────"
    echo " STEP $step: $title"
    echo "───────────────────────────────────────────────────────────────"
}

skip_if_exists() {
    local file="$1"
    local msg="$2"
    if [ -f "$file" ]; then
        echo "  [SKIP] $msg already exists at $file"
        return 0
    else
        return 1
    fi
}

elapsed_min() {
    local now=$(date +%s)
    echo $(( (now - START_TIME) / 60 ))
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
echo "═══════════════════════════════════════════════════════════════════════"
echo "  OVERNIGHT PIPELINE — HPS RESEARCH"
echo "  Started: $(date)"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "  Estimated total time: 16-20 hours"
echo "  Will save logs to results/log_*.txt"
echo ""
echo "  Configuration:"
echo "    DIVERSE_CSV:   $DIVERSE_CSV"
echo "    ATTACKS_CLEAN: $ATTACKS_CLEAN"
echo "    LLAMA_LAYERS:  $LLAMA_LAYERS"
echo "    VICUNA_LAYERS: $VICUNA_LAYERS"
echo ""

# Sanity check prerequisites
if [ ! -f "$DIVERSE_CSV" ]; then
    echo "  ERROR: $DIVERSE_CSV not found. Run build_diverse_benign.py first."
    exit 1
fi
if [ ! -f "$ATTACKS_CLEAN" ]; then
    echo "  ERROR: No attack JSON found ($ATTACKS_CLEAN or $ATTACKS_ORIG)."
    exit 1
fi
if [ ! -f "results/llama3_activations_cache_diverse.npz" ]; then
    echo "  ERROR: results/llama3_activations_cache_diverse.npz not found."
    echo "  Run extract_diverse_benign_activations.py first."
    exit 1
fi
if [ ! -f "results/vicuna_activations_cache_diverse.npz" ]; then
    echo "  ERROR: results/vicuna_activations_cache_diverse.npz not found."
    echo "  Run extract_diverse_benign_activations.py first."
    exit 1
fi

# ===========================================================================
# PHASE 1 — Cache Fix (max_length consistency)
# ===========================================================================
if [ "$SKIP_PHASE_1" -eq 1 ]; then
    echo "  [SKIP] PHASE 1 disabled by SKIP_PHASE_1=1"
else
log_phase "1" "Cache Fix (max_length=2048 consistency, ~5 hours)"

# Step 1.1: Llama-3 cache fix
log_step "1.1" "Re-extract Llama-3 attacks at max_length=2048"
if skip_if_exists "results/llama3_activations_cache_diverse_fixed.npz" \
    "Llama-3 fixed cache"; then
    :
else
    python fix_cache_max_length.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --diverse_cache results/llama3_activations_cache_diverse.npz \
        --attacks_json "$ATTACKS_CLEAN" \
        --output results/llama3_activations_cache_diverse_fixed.npz \
        --layers $LLAMA_LAYERS \
        --max_length 2048 \
        2>&1 | tee results/log_fix_llama3.txt
fi

# Step 1.2: Vicuna cache fix WITH full sequence storage (for MTP)
log_step "1.2" "Re-extract Vicuna at max_length=2048 with full sequences"
if skip_if_exists "results/vicuna_activations_cache_diverse_fixed.npz" \
    "Vicuna fixed cache"; then
    :
else
    python fix_cache_max_length.py \
        --model lmsys/vicuna-13b-v1.5 \
        --diverse_cache results/vicuna_activations_cache_diverse.npz \
        --attacks_json "$ATTACKS_CLEAN" \
        --output results/vicuna_activations_cache_diverse_fixed.npz \
        --layers $VICUNA_LAYERS \
        --max_length 2048 \
        --store_full_sequence \
        2>&1 | tee results/log_fix_vicuna.txt
fi

echo "  Phase 1 done at $(elapsed_min) min from start"
fi

# ===========================================================================
# PHASE 2 — Verification + main experiments rerun on fixed cache
# ===========================================================================
if [ "$SKIP_PHASE_2" -eq 1 ]; then
    echo "  [SKIP] PHASE 2 disabled by SKIP_PHASE_2=1"
else
log_phase "2" "Verification + Main Experiments on Fixed Cache (~1 hr)"

LLAMA_FIXED="results/llama3_activations_cache_diverse_fixed.npz"
VICUNA_FIXED="results/vicuna_activations_cache_diverse_fixed.npz"

# Sanity: did Phase 1 complete?
if [ ! -f "$LLAMA_FIXED" ]; then
    echo "  WARNING: $LLAMA_FIXED missing. Falling back to original cache."
    LLAMA_FIXED="results/llama3_activations_cache_diverse.npz"
fi
if [ ! -f "$VICUNA_FIXED" ]; then
    echo "  WARNING: $VICUNA_FIXED missing. Falling back to original cache."
    VICUNA_FIXED="results/vicuna_activations_cache_diverse.npz"
fi

# Step 2.1: Norm check on fixed cache
log_step "2.1" "Norm check on fixed cache (should drop from 1.000)"
python norm_check_diverse.py 2>&1 | tee results/log_norm_recheck.txt

# Step 2.2: Verify saturation
log_step "2.2" "verify_saturation on fixed cache"
python verify_saturation.py \
    --llama3_cache "$LLAMA_FIXED" \
    --output results/verify_saturation_fixed.json \
    2>&1 | tee results/log_verify_fixed.txt

# Step 2.3: Statistical tests
log_step "2.3" "Statistical tests (HPS vs C4 with bootstrap CIs)"
python statistical_tests.py \
    --cache "$LLAMA_FIXED" \
    --n_seeds 5 \
    --n_bootstrap 10000 \
    2>&1 | tee results/log_stats_fixed.txt

# Step 2.4: Radial distribution
log_step "2.4" "Radial distribution check"
python radial_distribution_check.py \
    --cache "$LLAMA_FIXED" \
    --n_seeds 5 \
    --total_epochs 50 \
    --epochs_to_check 5 10 25 50 \
    --kappas 0.1 0.5 1.0 2.0 \
    2>&1 | tee results/log_radial_fixed.txt

# Step 2.5: Vicuna imbalance test
log_step "2.5" "Vicuna imbalance + per-attack breakdown"
python vicuna_imbalance_test.py \
    --vicuna_cache "$VICUNA_FIXED" \
    2>&1 | tee results/log_vicuna_imbalance_fixed.txt

# Step 2.6: GCG-specific cross-model
log_step "2.6" "GCG-specific cross-model test"
python gcg_specific_test.py \
    --llama3_cache "$LLAMA_FIXED" \
    --vicuna_cache "$VICUNA_FIXED" \
    --attacks_json "$ATTACKS_CLEAN" \
    2>&1 | tee results/log_gcg_fixed.txt

# Step 2.7: Hyperbolic vs Euclidean re-run
log_step "2.7" "HPS vs HPS-Euclidean vs C4 cold-start sweep"
if [ -f "hyperbolic_vs_euclidean_diverse.py" ]; then
    python hyperbolic_vs_euclidean_diverse.py \
        --llama_cache "$LLAMA_FIXED" \
        --cold_start \
        2>&1 | tee results/log_hps_vs_euc_fixed.txt
else
    echo "  hyperbolic_vs_euclidean_diverse.py not found — skipping"
fi

echo "  Phase 2 done at $(elapsed_min) min from start"
fi

# ===========================================================================
# PHASE 3 — Anthropic Mean-Token Probe (faithful baseline)
# ===========================================================================
if [ "$SKIP_PHASE_3" -eq 1 ]; then
    echo "  [SKIP] PHASE 3 disabled by SKIP_PHASE_3=1"
else
log_phase "3" "Anthropic Mean-Token Probe (~10 min)"

LLAMA_FIXED="results/llama3_activations_cache_diverse_fixed.npz"
[ ! -f "$LLAMA_FIXED" ] && \
    LLAMA_FIXED="results/llama3_activations_cache_diverse.npz"

# Step 3.1: MTP on Llama-3
log_step "3.1" "Anthropic MTP on Llama-3 (vs C4 vs HPS)"
python anthropic_mean_token_probe.py \
    --cache "$LLAMA_FIXED" \
    --layers $LLAMA_LAYERS \
    --hidden_dim 4096 \
    --output results/anthropic_mtp_llama3.json \
    2>&1 | tee results/log_mtp_llama3.txt

# Step 3.2: MTP on Vicuna (only if full-sequence cache exists)
log_step "3.2" "Anthropic MTP on Vicuna (requires full-sequence cache)"
VICUNA_FIXED="results/vicuna_activations_cache_diverse_fixed.npz"
if [ -f "$VICUNA_FIXED" ]; then
    # Need to detect if full-sequence storage was used
    python -c "
import numpy as np
c = np.load('$VICUNA_FIXED', allow_pickle=True)
keys = list(c.keys())
print('Cache keys:', keys)
" 2>&1
    # If dict format with full sequences:
    python anthropic_mean_token_probe.py \
        --cache "$VICUNA_FIXED" \
        --layers $VICUNA_LAYERS \
        --hidden_dim 5120 \
        --output results/anthropic_mtp_vicuna.json \
        2>&1 | tee results/log_mtp_vicuna.txt
else
    echo "  $VICUNA_FIXED not found. Skipping Vicuna MTP."
fi

echo "  Phase 3 done at $(elapsed_min) min from start"
fi

# ===========================================================================
# PHASE 4 — JBShield Integration
# ===========================================================================
if [ "$SKIP_PHASE_4" -eq 1 ]; then
    echo "  [SKIP] PHASE 4 disabled by SKIP_PHASE_4=1"
else
log_phase "4" "JBShield Public Data Integration (~5 hrs)"

# Step 4.1: Clone JBShield
log_step "4.1" "Clone JBShield repo"
if [ ! -d "$JBSHIELD_DIR" ]; then
    git clone https://github.com/NISPLab/JBShield.git "$JBSHIELD_DIR" \
        2>&1 | tee results/log_jbshield_clone.txt
else
    echo "  JBShield already cloned at $JBSHIELD_DIR"
fi

# Step 4.2: Build JBShield attack JSONs
log_step "4.2" "Build JBShield attack JSONs (Vicuna + Llama-3)"
if [ ! -f "results/jbshield_vicuna13b_attacks.json" ] || \
   [ ! -f "results/jbshield_llama3_attacks.json" ]; then
    python build_jbshield_attacks.py \
        --jbshield_dir "$JBSHIELD_DIR" \
        --models vicuna-13b-v1.5 Meta-Llama-3-8B-Instruct \
        2>&1 | tee results/log_jbshield_build.txt
else
    echo "  JBShield attack JSONs already exist"
fi

# Step 4.3: Extract Vicuna activations
log_step "4.3" "Extract Vicuna activations on JBShield attacks (~2 hrs)"
if skip_if_exists "results/vicuna_activations_cache_jbshield.npz" \
    "Vicuna JBShield cache"; then
    :
elif [ -f "results/jbshield_vicuna13b_attacks.json" ]; then
    python extract_jbshield_activations.py \
        --model lmsys/vicuna-13b-v1.5 \
        --attacks_json results/jbshield_vicuna13b_attacks.json \
        --diverse_benign "$DIVERSE_CSV" \
        --output results/vicuna_activations_cache_jbshield.npz \
        --layers $VICUNA_LAYERS \
        --cache_format dict \
        --max_length 2048 \
        2>&1 | tee results/log_jbshield_vicuna_extract.txt
else
    echo "  jbshield_vicuna13b_attacks.json missing — skipping"
fi

# Step 4.4: Extract Llama-3 activations
log_step "4.4" "Extract Llama-3 activations on JBShield attacks (~3 hrs)"
if skip_if_exists "results/llama3_activations_cache_jbshield.npz" \
    "Llama-3 JBShield cache"; then
    :
elif [ -f "results/jbshield_llama3_attacks.json" ]; then
    python extract_jbshield_activations.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --attacks_json results/jbshield_llama3_attacks.json \
        --diverse_benign "$DIVERSE_CSV" \
        --output results/llama3_activations_cache_jbshield.npz \
        --layers $LLAMA_LAYERS \
        --cache_format dict \
        --max_length 2048 \
        2>&1 | tee results/log_jbshield_llama3_extract.txt
else
    echo "  jbshield_llama3_attacks.json missing — skipping"
fi

# Step 4.5: Run main experiments on JBShield caches
log_step "4.5" "Statistical tests on JBShield Vicuna cache"
if [ -f "results/vicuna_activations_cache_jbshield.npz" ]; then
    python statistical_tests.py \
        --cache results/vicuna_activations_cache_jbshield.npz \
        2>&1 | tee results/log_jbshield_vicuna_stats.txt

    python vicuna_imbalance_test.py \
        --vicuna_cache results/vicuna_activations_cache_jbshield.npz \
        2>&1 | tee results/log_jbshield_vicuna_imbalance.txt
fi

log_step "4.6" "Statistical tests on JBShield Llama-3 cache"
if [ -f "results/llama3_activations_cache_jbshield.npz" ]; then
    python verify_saturation.py \
        --llama3_cache results/llama3_activations_cache_jbshield.npz \
        --output results/verify_saturation_jbshield.json \
        2>&1 | tee results/log_jbshield_llama3_verify.txt

    python statistical_tests.py \
        --cache results/llama3_activations_cache_jbshield.npz \
        2>&1 | tee results/log_jbshield_llama3_stats.txt

    # Anthropic MTP on JBShield Llama-3
    python anthropic_mean_token_probe.py \
        --cache results/llama3_activations_cache_jbshield.npz \
        --layers $LLAMA_LAYERS \
        --hidden_dim 4096 \
        --output results/anthropic_mtp_jbshield_llama3.json \
        2>&1 | tee results/log_mtp_jbshield_llama3.txt
fi

# Step 4.7: Cross-LLM GCG test using JBShield data
log_step "4.7" "GCG-specific cross-model test (JBShield data)"
if [ -f "results/llama3_activations_cache_jbshield.npz" ] && \
   [ -f "results/vicuna_activations_cache_jbshield.npz" ]; then
    python gcg_specific_test.py \
        --llama3_cache results/llama3_activations_cache_jbshield.npz \
        --vicuna_cache results/vicuna_activations_cache_jbshield.npz \
        --attacks_json results/jbshield_llama3_attacks.json \
        2>&1 | tee results/log_jbshield_gcg.txt
fi

echo "  Phase 4 done at $(elapsed_min) min from start"
fi

# ===========================================================================
# PHASE 5 — Alignment Ablation (Llama-3 base vs Instruct)
# ===========================================================================
if [ "$SKIP_PHASE_5" -eq 1 ]; then
    echo "  [SKIP] PHASE 5 disabled by SKIP_PHASE_5=1"
else
log_phase "5" "Alignment Ablation (Llama-3 base, ~4 hrs)"

# Step 5.1: Verify HF access to Llama-3-8B base
log_step "5.1" "Verify HF access to base model"
python -c "
from transformers import AutoTokenizer
try:
    tk = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
    print('  HF access OK to Meta-Llama-3-8B (base)')
except Exception as e:
    print(f'  ERROR: {e}')
    print('  Run: huggingface-cli login')
    raise
" 2>&1 | tee results/log_alignment_hf_check.txt

# Step 5.2: Extract base model activations
log_step "5.2" "Extract Llama-3 base activations (~3 hrs)"
if skip_if_exists "results/llama3_BASE_activations_cache_diverse.npz" \
    "Llama-3 base cache"; then
    :
else
    python extract_llama3_base_activations.py \
        --model meta-llama/Meta-Llama-3-8B \
        --diverse_benign "$DIVERSE_CSV" \
        --attacks_json "$ATTACKS_CLEAN" \
        --output results/llama3_BASE_activations_cache_diverse.npz \
        --layers $LLAMA_LAYERS \
        --max_length 2048 \
        2>&1 | tee results/log_alignment_extract.txt
fi

# Step 5.3: Run experiments on base model
log_step "5.3" "Verify saturation on base model"
if [ -f "results/llama3_BASE_activations_cache_diverse.npz" ]; then
    python verify_saturation.py \
        --llama3_cache results/llama3_BASE_activations_cache_diverse.npz \
        --output results/verify_saturation_base.json \
        2>&1 | tee results/log_alignment_verify.txt

    python statistical_tests.py \
        --cache results/llama3_BASE_activations_cache_diverse.npz \
        2>&1 | tee results/log_alignment_stats.txt

    python radial_distribution_check.py \
        --cache results/llama3_BASE_activations_cache_diverse.npz \
        --n_seeds 3 \
        2>&1 | tee results/log_alignment_radial.txt

    # Anthropic MTP on base model
    python anthropic_mean_token_probe.py \
        --cache results/llama3_BASE_activations_cache_diverse.npz \
        --layers $LLAMA_LAYERS \
        --hidden_dim 4096 \
        --output results/anthropic_mtp_base.json \
        2>&1 | tee results/log_mtp_base.txt

    if [ -f "hyperbolic_vs_euclidean_diverse.py" ]; then
        python hyperbolic_vs_euclidean_diverse.py \
            --llama_cache results/llama3_BASE_activations_cache_diverse.npz \
            --cold_start \
            2>&1 | tee results/log_alignment_hps_vs_euc.txt
    fi
fi

echo "  Phase 5 done at $(elapsed_min) min from start"
fi

# ===========================================================================
# PHASE 6 — Novel Attacks (FlipAttack + JailbreakBench)
# ===========================================================================
if [ "$SKIP_PHASE_6" -eq 1 ]; then
    echo "  [SKIP] PHASE 6 disabled by SKIP_PHASE_6=1"
else
log_phase "6" "Novel Attacks 2025 (FlipAttack + JBB-style, ~4 hrs)"

# Step 6.1: Build novel attack JSON
log_step "6.1" "Build novel attacks JSON"
if skip_if_exists "results/novel_attacks_2025.json" "Novel attacks JSON"; then
    :
else
    python build_novel_attacks.py \
        --output results/novel_attacks_2025.json \
        --use_full_advbench \
        2>&1 | tee results/log_novel_build.txt
fi

# Step 6.2: Extract Llama-3 activations on novel attacks
log_step "6.2" "Extract Llama-3 activations on novel attacks (~2 hrs)"
if skip_if_exists "results/llama3_activations_cache_novel.npz" \
    "Llama-3 novel cache"; then
    :
elif [ -f "results/novel_attacks_2025.json" ]; then
    python extract_jbshield_activations.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --attacks_json results/novel_attacks_2025.json \
        --diverse_benign "$DIVERSE_CSV" \
        --output results/llama3_activations_cache_novel.npz \
        --layers $LLAMA_LAYERS \
        --cache_format dict \
        --max_length 2048 \
        2>&1 | tee results/log_novel_llama3_extract.txt
fi

# Step 6.3: Extract Vicuna activations on novel attacks
log_step "6.3" "Extract Vicuna activations on novel attacks (~1.5 hrs)"
if skip_if_exists "results/vicuna_activations_cache_novel.npz" \
    "Vicuna novel cache"; then
    :
elif [ -f "results/novel_attacks_2025.json" ]; then
    python extract_jbshield_activations.py \
        --model lmsys/vicuna-13b-v1.5 \
        --attacks_json results/novel_attacks_2025.json \
        --diverse_benign "$DIVERSE_CSV" \
        --output results/vicuna_activations_cache_novel.npz \
        --layers $VICUNA_LAYERS \
        --cache_format dict \
        --max_length 2048 \
        2>&1 | tee results/log_novel_vicuna_extract.txt
fi

# Step 6.4: OOD evaluation
log_step "6.4" "OOD evaluation (Llama-3)"
LLAMA_FIXED="results/llama3_activations_cache_diverse_fixed.npz"
[ ! -f "$LLAMA_FIXED" ] && \
    LLAMA_FIXED="results/llama3_activations_cache_diverse.npz"

if [ -f "results/llama3_activations_cache_novel.npz" ]; then
    python evaluate_ood_attacks.py \
        --train_cache "$LLAMA_FIXED" \
        --test_cache results/llama3_activations_cache_novel.npz \
        --output results/ood_eval_llama3.json \
        --layers $LLAMA_LAYERS \
        --hidden_dim 4096 \
        2>&1 | tee results/log_ood_llama3.txt
fi

log_step "6.5" "OOD evaluation (Vicuna)"
VICUNA_FIXED="results/vicuna_activations_cache_diverse_fixed.npz"
[ ! -f "$VICUNA_FIXED" ] && \
    VICUNA_FIXED="results/vicuna_activations_cache_diverse.npz"

if [ -f "results/vicuna_activations_cache_novel.npz" ]; then
    python evaluate_ood_attacks.py \
        --train_cache "$VICUNA_FIXED" \
        --test_cache results/vicuna_activations_cache_novel.npz \
        --output results/ood_eval_vicuna.json \
        --layers $VICUNA_LAYERS \
        --hidden_dim 5120 \
        2>&1 | tee results/log_ood_vicuna.txt
fi

echo "  Phase 6 done at $(elapsed_min) min from start"
fi

# ===========================================================================
# PHASE 7 — Confound Diagnostics (only run if confounds persist)
# ===========================================================================
if [ "$SKIP_PHASE_7" -eq 1 ]; then
    echo "  [SKIP] PHASE 7 disabled by SKIP_PHASE_7=1"
else
log_phase "7" "Confound Diagnostics (~1 hr)"

LLAMA_FIXED="results/llama3_activations_cache_diverse_fixed.npz"
[ ! -f "$LLAMA_FIXED" ] && \
    LLAMA_FIXED="results/llama3_activations_cache_diverse.npz"

# Step 7.1: Norm confound diagnosis (chat-template / max_length / etc.)
log_step "7.1" "Norm confound diagnosis (50 samples, ~10 min)"
if [ -f "diagnose_norm_confound.py" ]; then
    python diagnose_norm_confound.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --diverse_benign "$DIVERSE_CSV" \
        --attacks_json "$ATTACKS_CLEAN" \
        --n_samples 50 \
        --layers $LLAMA_LAYERS \
        --output results/norm_confound_diagnosis.json \
        2>&1 | tee results/log_norm_diagnosis.txt
fi

# Step 7.2: Norm-controlled evaluation
log_step "7.2" "Norm-controlled evaluation (4 conditions, ~30 min)"
if [ -f "norm_controlled_eval.py" ]; then
    python norm_controlled_eval.py \
        --cache "$LLAMA_FIXED" \
        --layers $LLAMA_LAYERS \
        --hidden_dim 4096 \
        --output results/norm_controlled_eval_llama3.json \
        2>&1 | tee results/log_norm_controlled_llama3.txt

    VICUNA_FIXED="results/vicuna_activations_cache_diverse_fixed.npz"
    [ ! -f "$VICUNA_FIXED" ] && \
        VICUNA_FIXED="results/vicuna_activations_cache_diverse.npz"

    python norm_controlled_eval.py \
        --cache "$VICUNA_FIXED" \
        --layers $VICUNA_LAYERS \
        --hidden_dim 5120 \
        --output results/norm_controlled_eval_vicuna.json \
        2>&1 | tee results/log_norm_controlled_vicuna.txt
fi

echo "  Phase 7 done at $(elapsed_min) min from start"
fi

# ===========================================================================
# WRAP UP
# ===========================================================================
END_TIME=$(date +%s)
TOTAL_MINUTES=$(( (END_TIME - START_TIME) / 60 ))
TOTAL_HOURS=$(( TOTAL_MINUTES / 60 ))

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  OVERNIGHT PIPELINE COMPLETE"
echo "  Started:  $(date -d @$START_TIME 2>/dev/null || date -r $START_TIME)"
echo "  Finished: $(date)"
echo "  Total:    ${TOTAL_HOURS}h ${TOTAL_MINUTES}m"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "  Key output files (review when you're back):"
echo ""

# Phase 1 outputs
echo "  PHASE 1 (Cache fix):"
[ -f "results/llama3_activations_cache_diverse_fixed.npz" ] && \
    echo "    ✓ results/llama3_activations_cache_diverse_fixed.npz"
[ -f "results/vicuna_activations_cache_diverse_fixed.npz" ] && \
    echo "    ✓ results/vicuna_activations_cache_diverse_fixed.npz"

# Phase 2 outputs
echo ""
echo "  PHASE 2 (Verification + main experiments):"
for f in \
    results/log_norm_recheck.txt \
    results/verify_saturation_fixed.json \
    results/log_stats_fixed.txt \
    results/log_radial_fixed.txt \
    results/log_vicuna_imbalance_fixed.txt \
    results/log_gcg_fixed.txt \
    results/log_hps_vs_euc_fixed.txt; do
    [ -f "$f" ] && echo "    ✓ $f"
done

# Phase 3 outputs
echo ""
echo "  PHASE 3 (Anthropic MTP):"
[ -f "results/anthropic_mtp_llama3.json" ] && \
    echo "    ✓ results/anthropic_mtp_llama3.json"
[ -f "results/anthropic_mtp_vicuna.json" ] && \
    echo "    ✓ results/anthropic_mtp_vicuna.json"

# Phase 4 outputs
echo ""
echo "  PHASE 4 (JBShield integration):"
for f in \
    results/jbshield_vicuna13b_attacks.json \
    results/jbshield_llama3_attacks.json \
    results/vicuna_activations_cache_jbshield.npz \
    results/llama3_activations_cache_jbshield.npz \
    results/log_jbshield_vicuna_stats.txt \
    results/log_jbshield_llama3_verify.txt \
    results/log_jbshield_gcg.txt \
    results/anthropic_mtp_jbshield_llama3.json; do
    [ -f "$f" ] && echo "    ✓ $f"
done

# Phase 5 outputs
echo ""
echo "  PHASE 5 (Alignment ablation):"
[ -f "results/llama3_BASE_activations_cache_diverse.npz" ] && \
    echo "    ✓ results/llama3_BASE_activations_cache_diverse.npz"
for f in \
    results/verify_saturation_base.json \
    results/log_alignment_stats.txt \
    results/log_alignment_radial.txt \
    results/anthropic_mtp_base.json \
    results/log_alignment_hps_vs_euc.txt; do
    [ -f "$f" ] && echo "    ✓ $f"
done

# Phase 6 outputs
echo ""
echo "  PHASE 6 (Novel attacks):"
for f in \
    results/novel_attacks_2025.json \
    results/llama3_activations_cache_novel.npz \
    results/vicuna_activations_cache_novel.npz \
    results/ood_eval_llama3.json \
    results/ood_eval_vicuna.json; do
    [ -f "$f" ] && echo "    ✓ $f"
done

# Phase 7 outputs
echo ""
echo "  PHASE 7 (Diagnostics):"
[ -f "results/norm_confound_diagnosis.json" ] && \
    echo "    ✓ results/norm_confound_diagnosis.json"
[ -f "results/norm_controlled_eval_llama3.json" ] && \
    echo "    ✓ results/norm_controlled_eval_llama3.json"
[ -f "results/norm_controlled_eval_vicuna.json" ] && \
    echo "    ✓ results/norm_controlled_eval_vicuna.json"

echo ""
echo "  All logs in results/log_*.txt"
echo "  All figures in results/figs/*.png"
echo ""
echo "  When you return, the most important things to check:"
echo ""
echo "    1. results/log_norm_recheck.txt"
echo "       → Did norm-only AUROC drop from 1.000 after the cache fix?"
echo "       If YES: methodology bug confirmed, paper survives."
echo "       If NO: deeper issue, check Phase 7 diagnostics."
echo ""
echo "    2. results/anthropic_mtp_llama3.json"
echo "       → Is Anthropic's mean-token probe equal to or better than C4?"
echo "       Determines whether to use MTP or C4 as the primary baseline."
echo ""
echo "    3. results/log_alignment_stats.txt"
echo "       → Did HPS detection drop on the base model?"
echo "       Tests the alignment-mediated hypothesis."
echo ""
echo "    4. results/ood_eval_llama3.json"
echo "       → Does saturation extend to novel attacks?"
echo "       If yes: methodology critique strengthens."
echo ""
