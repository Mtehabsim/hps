#!/bin/bash
# run_strengthening_pipeline.sh — orchestrate the three strengthening tasks
#
#   Task 1: JBShield public Vicuna-13B + Llama-3 data integration
#   Task 2: Alignment ablation (Llama-3 base vs Instruct)
#   Task 3: Novel 2025 attacks (FlipAttack + JBB-style)
#
# Each task can be run independently. Set the corresponding RUN_TASK_N=1
# variables below to enable.
#
# Usage:
#   ./run_strengthening_pipeline.sh
#
# Variables (override on command line):
#   RUN_TASK_1=1 ./run_strengthening_pipeline.sh   # JBShield only
#   RUN_TASK_2=1 ./run_strengthening_pipeline.sh   # Alignment only
#   RUN_TASK_3=1 ./run_strengthening_pipeline.sh   # Novel attacks only

set -e
set -u
mkdir -p results results/figs

# ===========================================================================
# Task selection (default: prompt user)
# ===========================================================================
RUN_TASK_1="${RUN_TASK_1:-prompt}"
RUN_TASK_2="${RUN_TASK_2:-prompt}"
RUN_TASK_3="${RUN_TASK_3:-prompt}"

# ===========================================================================
# Config
# ===========================================================================
DIVERSE_CSV="results/data_harmless_diverse.csv"
ATTACKS_CLEAN="llama3_attacks_clean.json"
JBSHIELD_DIR="${JBSHIELD_DIR:-/tmp/JBShield}"

LLAMA_LAYERS="0 2 17 24 28 31"
VICUNA_LAYERS="0 2 22 31 35 39"

# ===========================================================================
# Header
# ===========================================================================
echo "═══════════════════════════════════════════════════════════════"
echo "  STRENGTHENING PIPELINE — JBShield + Alignment + Novel Attacks"
echo "  Started: $(date)"
echo "═══════════════════════════════════════════════════════════════"

# Sanity check: prerequisites from previous pipelines
if [ ! -f "$DIVERSE_CSV" ]; then
    echo "  ERROR: $DIVERSE_CSV not found"
    echo "  Run build_diverse_benign.py first (Phase A of run_diverse_benign_pipeline.sh)"
    exit 1
fi

if [ ! -f "$ATTACKS_CLEAN" ]; then
    echo "  WARNING: $ATTACKS_CLEAN not found — will use llama3_attacks.json instead"
    ATTACKS_CLEAN="llama3_attacks.json"
fi

# ===========================================================================
# Helper to ask user
# ===========================================================================
ask_run() {
    local var_name=$1
    local task_name=$2
    local time_estimate=$3
    if [ "${!var_name}" == "prompt" ]; then
        read -p "  Run $task_name (~$time_estimate)? [y/N] " response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            return 0
        else
            return 1
        fi
    elif [ "${!var_name}" == "1" ]; then
        return 0
    else
        return 1
    fi
}

# ===========================================================================
# Task 1: JBShield integration
# ===========================================================================
if ask_run RUN_TASK_1 "Task 1 — JBShield Vicuna-13B + Llama-3 integration" "5 hours total"; then

    echo ""
    echo "───────────────────────────────────────────────────────────────"
    echo " TASK 1.1: Clone JBShield repo"
    echo "───────────────────────────────────────────────────────────────"
    if [ ! -d "$JBSHIELD_DIR" ]; then
        git clone https://github.com/NISPLab/JBShield.git "$JBSHIELD_DIR"
    else
        echo "  $JBSHIELD_DIR already exists. Skipping clone."
    fi
    if [ ! -d "$JBSHIELD_DIR/data/jailbreak" ]; then
        echo "  ERROR: $JBSHIELD_DIR/data/jailbreak not found"
        exit 1
    fi

    echo ""
    echo "───────────────────────────────────────────────────────────────"
    echo " TASK 1.2: Build attack JSONs from JBShield data"
    echo "───────────────────────────────────────────────────────────────"
    if [ -f "results/jbshield_vicuna13b_attacks.json" ] && \
       [ -f "results/jbshield_llama3_attacks.json" ]; then
        echo "  Both JBShield attack JSONs already exist. Skipping (delete to rebuild)."
    else
        python build_jbshield_attacks.py \
            --jbshield_dir "$JBSHIELD_DIR" \
            --models vicuna-13b-v1.5 Meta-Llama-3-8B-Instruct \
            2>&1 | tee results/log_jbshield_build.txt
    fi

    echo ""
    echo "───────────────────────────────────────────────────────────────"
    echo " TASK 1.3: Extract Vicuna-13B activations on JBShield attacks"
    echo "───────────────────────────────────────────────────────────────"
    if [ -f "results/vicuna_activations_cache_jbshield.npz" ]; then
        echo "  Vicuna JBShield cache exists. Skipping."
    else
        python extract_jbshield_activations.py \
            --model lmsys/vicuna-13b-v1.5 \
            --attacks_json results/jbshield_vicuna13b_attacks.json \
            --diverse_benign "$DIVERSE_CSV" \
            --output results/vicuna_activations_cache_jbshield.npz \
            --layers $VICUNA_LAYERS \
            --cache_format array \
            2>&1 | tee results/log_jbshield_vicuna_extract.txt
    fi

    echo ""
    echo "───────────────────────────────────────────────────────────────"
    echo " TASK 1.4: Extract Llama-3 activations on JBShield attacks"
    echo "───────────────────────────────────────────────────────────────"
    if [ -f "results/llama3_activations_cache_jbshield.npz" ]; then
        echo "  Llama-3 JBShield cache exists. Skipping."
    else
        python extract_jbshield_activations.py \
            --model meta-llama/Meta-Llama-3-8B-Instruct \
            --attacks_json results/jbshield_llama3_attacks.json \
            --diverse_benign "$DIVERSE_CSV" \
            --output results/llama3_activations_cache_jbshield.npz \
            --layers $LLAMA_LAYERS \
            --cache_format dict \
            2>&1 | tee results/log_jbshield_llama3_extract.txt
    fi

    echo ""
    echo "───────────────────────────────────────────────────────────────"
    echo " TASK 1.5: Run experiments on Vicuna JBShield cache"
    echo "───────────────────────────────────────────────────────────────"
    python statistical_tests.py \
        --cache results/vicuna_activations_cache_jbshield.npz \
        2>&1 | tee results/log_jbshield_vicuna_stats.txt

    python vicuna_imbalance_test.py \
        --vicuna_cache results/vicuna_activations_cache_jbshield.npz \
        2>&1 | tee results/log_jbshield_vicuna_imbalance.txt

    python gcg_specific_test.py \
        --llama3_cache results/llama3_activations_cache_jbshield.npz \
        --vicuna_cache results/vicuna_activations_cache_jbshield.npz \
        --attacks_json results/jbshield_llama3_attacks.json \
        2>&1 | tee results/log_jbshield_gcg.txt

    echo ""
    echo "───────────────────────────────────────────────────────────────"
    echo " TASK 1.6: Run experiments on Llama-3 JBShield cache (cross-validation)"
    echo "───────────────────────────────────────────────────────────────"
    python verify_saturation.py \
        --llama3_cache results/llama3_activations_cache_jbshield.npz \
        --output results/verify_saturation_jbshield.json \
        2>&1 | tee results/log_jbshield_llama3_verify.txt

    python statistical_tests.py \
        --cache results/llama3_activations_cache_jbshield.npz \
        2>&1 | tee results/log_jbshield_llama3_stats.txt

    echo "  Task 1 complete."
fi

# ===========================================================================
# Task 2: Alignment ablation (Llama-3 base vs Instruct)
# ===========================================================================
if ask_run RUN_TASK_2 "Task 2 — Alignment ablation (Llama-3 base)" "4 hours"; then

    echo ""
    echo "───────────────────────────────────────────────────────────────"
    echo " TASK 2.1: Verify HF access to Llama-3-8B base"
    echo "───────────────────────────────────────────────────────────────"
    python -c "from transformers import AutoTokenizer; \
        tk = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B'); \
        print(f'  Tokenizer OK: {len(tk)} tokens')" \
        || { echo "  ERROR: Cannot access meta-llama/Meta-Llama-3-8B"; \
             echo "  Run: huggingface-cli login"; exit 1; }

    echo ""
    echo "───────────────────────────────────────────────────────────────"
    echo " TASK 2.2: Extract base model activations (~3 hrs GPU)"
    echo "───────────────────────────────────────────────────────────────"
    if [ -f "results/llama3_BASE_activations_cache_diverse.npz" ]; then
        echo "  Base cache exists. Skipping."
    else
        python extract_llama3_base_activations.py \
            --model meta-llama/Meta-Llama-3-8B \
            --diverse_benign "$DIVERSE_CSV" \
            --attacks_json "$ATTACKS_CLEAN" \
            --output results/llama3_BASE_activations_cache_diverse.npz \
            --layers $LLAMA_LAYERS \
            2>&1 | tee results/log_alignment_extract.txt
    fi

    echo ""
    echo "───────────────────────────────────────────────────────────────"
    echo " TASK 2.3: Run experiments on base model"
    echo "───────────────────────────────────────────────────────────────"
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

    if [ -f "hyperbolic_vs_euclidean_diverse.py" ]; then
        python hyperbolic_vs_euclidean_diverse.py \
            --llama_cache results/llama3_BASE_activations_cache_diverse.npz \
            --cold_start \
            2>&1 | tee results/log_alignment_hps_vs_euc.txt
    fi

    echo ""
    echo "  Task 2 complete. Compare base vs Instruct:"
    echo "    Instruct: results/statistical_tests.json (rerun on _diverse cache)"
    echo "    Base:     results/log_alignment_stats.txt"
fi

# ===========================================================================
# Task 3: Novel attacks (FlipAttack + JBB-style)
# ===========================================================================
if ask_run RUN_TASK_3 "Task 3 — Novel 2025 attacks (FlipAttack + JBB-style)" "4 hours"; then

    echo ""
    echo "───────────────────────────────────────────────────────────────"
    echo " TASK 3.1: Build novel attacks JSON"
    echo "───────────────────────────────────────────────────────────────"
    if [ -f "results/novel_attacks_2025.json" ]; then
        echo "  Novel attacks JSON exists. Skipping (delete to rebuild)."
    else
        python build_novel_attacks.py \
            --output results/novel_attacks_2025.json \
            --use_full_advbench \
            2>&1 | tee results/log_novel_build.txt
    fi

    echo ""
    echo "───────────────────────────────────────────────────────────────"
    echo " TASK 3.2: Extract Llama-3 activations on novel attacks"
    echo "───────────────────────────────────────────────────────────────"
    if [ -f "results/llama3_activations_cache_novel.npz" ]; then
        echo "  Llama-3 novel cache exists. Skipping."
    else
        python extract_jbshield_activations.py \
            --model meta-llama/Meta-Llama-3-8B-Instruct \
            --attacks_json results/novel_attacks_2025.json \
            --diverse_benign "$DIVERSE_CSV" \
            --output results/llama3_activations_cache_novel.npz \
            --layers $LLAMA_LAYERS \
            --cache_format dict \
            2>&1 | tee results/log_novel_llama3_extract.txt
    fi

    echo ""
    echo "───────────────────────────────────────────────────────────────"
    echo " TASK 3.3: Extract Vicuna activations on novel attacks"
    echo "───────────────────────────────────────────────────────────────"
    if [ -f "results/vicuna_activations_cache_novel.npz" ]; then
        echo "  Vicuna novel cache exists. Skipping."
    else
        python extract_jbshield_activations.py \
            --model lmsys/vicuna-13b-v1.5 \
            --attacks_json results/novel_attacks_2025.json \
            --diverse_benign "$DIVERSE_CSV" \
            --output results/vicuna_activations_cache_novel.npz \
            --layers $VICUNA_LAYERS \
            --cache_format array \
            2>&1 | tee results/log_novel_vicuna_extract.txt
    fi

    echo ""
    echo "───────────────────────────────────────────────────────────────"
    echo " TASK 3.4: OOD evaluation (Llama-3)"
    echo "───────────────────────────────────────────────────────────────"
    python evaluate_ood_attacks.py \
        --train_cache results/llama3_activations_cache_diverse.npz \
        --test_cache results/llama3_activations_cache_novel.npz \
        --output results/ood_eval_llama3.json \
        --layers $LLAMA_LAYERS --hidden_dim 4096 \
        2>&1 | tee results/log_ood_llama3.txt

    echo ""
    echo "───────────────────────────────────────────────────────────────"
    echo " TASK 3.5: OOD evaluation (Vicuna)"
    echo "───────────────────────────────────────────────────────────────"
    python evaluate_ood_attacks.py \
        --train_cache results/vicuna_activations_cache_diverse.npz \
        --test_cache results/vicuna_activations_cache_novel.npz \
        --output results/ood_eval_vicuna.json \
        --layers $VICUNA_LAYERS --hidden_dim 5120 \
        2>&1 | tee results/log_ood_vicuna.txt

    echo "  Task 3 complete."
fi

# ===========================================================================
# Done
# ===========================================================================
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  STRENGTHENING PIPELINE COMPLETE"
echo "  Finished: $(date)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  Key outputs:"
[ -f "results/jbshield_vicuna13b_attacks.json" ] && \
    echo "    JBShield Vicuna attacks:  results/jbshield_vicuna13b_attacks.json"
[ -f "results/jbshield_llama3_attacks.json" ] && \
    echo "    JBShield Llama-3 attacks: results/jbshield_llama3_attacks.json"
[ -f "results/vicuna_activations_cache_jbshield.npz" ] && \
    echo "    Vicuna JBShield cache:    results/vicuna_activations_cache_jbshield.npz"
[ -f "results/llama3_activations_cache_jbshield.npz" ] && \
    echo "    Llama-3 JBShield cache:   results/llama3_activations_cache_jbshield.npz"
[ -f "results/llama3_BASE_activations_cache_diverse.npz" ] && \
    echo "    Llama-3 BASE cache:       results/llama3_BASE_activations_cache_diverse.npz"
[ -f "results/novel_attacks_2025.json" ] && \
    echo "    Novel attacks JSON:       results/novel_attacks_2025.json"
[ -f "results/ood_eval_llama3.json" ] && \
    echo "    OOD eval (Llama-3):       results/ood_eval_llama3.json"
[ -f "results/ood_eval_vicuna.json" ] && \
    echo "    OOD eval (Vicuna):        results/ood_eval_vicuna.json"
echo ""
echo "  Logs in results/log_*.txt"
