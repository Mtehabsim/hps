#!/bin/bash
# run_diverse_benign_pipeline.sh
#
# Full pipeline to address the length-confound finding from verify_saturation.py.
#
# Phase A: Build diverse benign + investigate contamination (~30 min, parallel)
# Phase B: Verify length confound is reduced (~5-10 min)
# Phase C: Re-extract activations on both LLMs (~6-10 hours, gated by Phase B)
# Phase D: Re-run main experiments (~30-60 min, gated by Phase C)
#
# After Phase B, the script asks for confirmation before proceeding to Phase C.
# This is the most expensive phase and should only run if Phase B succeeds.

set -e
set -o pipefail

# ============================================================
# Configuration
# ============================================================
ATTACKS_JSON="llama3_attacks.json"
ATTACKS_CLEAN="llama3_attacks_clean.json"
DIVERSE_CSV="results/data_harmless_diverse.csv"
LLAMA_CACHE_OLD="results/llama3_activations_cache.npz"
LLAMA_CACHE_NEW="results/llama3_activations_cache_diverse.npz"
VICUNA_CACHE_OLD="results/vicuna_activations_cache.npz"
VICUNA_CACHE_NEW="results/vicuna_activations_cache_diverse.npz"

mkdir -p results

START=$(date +%s)
echo "═══════════════════════════════════════════════════════════════"
echo "  DIVERSE BENIGN PIPELINE — addressing length confound"
echo "  Started: $(date)"
echo "═══════════════════════════════════════════════════════════════"

# ============================================================
# PHASE A.1: Build diverse benign
# ============================================================
echo ""
echo "───────────────────────────────────────────────────────────────"
echo " PHASE A.1: Build diverse benign set"
echo "───────────────────────────────────────────────────────────────"
if [ -f "$DIVERSE_CSV" ]; then
    echo "  $DIVERSE_CSV already exists. Skipping (delete to rebuild)."
else
    python build_diverse_benign.py \
        --no_multilingual \
        --tokenizer gpt2 \
        2>&1 | tee results/log_build_diverse.txt
fi

# Sanity check on diverse benign length distribution
echo ""
echo "  Diverse benign length distribution:"
python -c "
import pandas as pd
df = pd.read_csv('$DIVERSE_CSV')
df['len'] = df['prompt'].astype(str).str.len()
print(f'  Total: {len(df)} prompts')
print(f'  Mean: {df.len.mean():.0f} chars')
print(f'  Median: {df.len.median():.0f} chars')
print(f'  Max: {df.len.max()} chars')
print(f'  >1000 chars: {(df.len>1000).sum()}')
print(f'  >2000 chars: {(df.len>2000).sum()}')
print(f'  >5000 chars: {(df.len>5000).sum()}')
"

# ============================================================
# PHASE A.2: Investigate contamination
# ============================================================
echo ""
echo "───────────────────────────────────────────────────────────────"
echo " PHASE A.2: Investigate train/test contamination"
echo "───────────────────────────────────────────────────────────────"
if [ -f "$ATTACKS_CLEAN" ]; then
    echo "  $ATTACKS_CLEAN already exists. Skipping (delete to rerun)."
else
    python investigate_contamination.py \
        --attacks_json "$ATTACKS_JSON" \
        --output "$ATTACKS_CLEAN" \
        2>&1 | tee results/log_contamination.txt
fi

PHASE_A_END=$(date +%s)
echo ""
echo "  Phase A completed in $(( (PHASE_A_END - START) / 60 )) minutes."

# ============================================================
# PHASE B: Verify length confound is reduced
# ============================================================
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " PHASE B: Verify length confound on diverse benign"
echo "═══════════════════════════════════════════════════════════════"
python verify_saturation.py \
    --harmless_csv "$DIVERSE_CSV" \
    --attacks_json "$ATTACKS_CLEAN" \
    --llama3_cache "$LLAMA_CACHE_OLD" \
    --output results/verify_saturation_diverse.json \
    2>&1 | tee results/log_verify_diverse.txt

# Extract length-only AUROC from results to gate Phase C
LENGTH_AUROC=$(python -c "
import json
with open('results/verify_saturation_diverse.json') as f:
    r = json.load(f)
auroc = r.get('check2_length_confound', {}).get('overall_auroc', None)
print(f'{auroc:.4f}' if auroc is not None else 'NaN')
")

echo ""
echo "  Length-only AUROC on diverse benign: $LENGTH_AUROC"

# Decision logic
PROCEED="no"
if python -c "import sys; sys.exit(0 if float('$LENGTH_AUROC') < 0.85 else 1)" 2>/dev/null; then
    PROCEED="yes"
    if python -c "import sys; sys.exit(0 if float('$LENGTH_AUROC') < 0.70 else 1)" 2>/dev/null; then
        echo "  ✓ EXCELLENT: AUROC < 0.70 — length confound mostly resolved."
    else
        echo "  ⚠ MODERATE: 0.70 <= AUROC < 0.85 — partial improvement; document caveat."
    fi
else
    echo "  ✗ FAILED: AUROC >= 0.85 — length confound persists. Phase C not worth running."
    echo ""
    echo "  Recommendation: Stop and reassess. Either:"
    echo "    1. Build a longer-tail benign set (extended WikiText loader)"
    echo "    2. Pivot paper to pure methodology critique"
    echo "    3. Restrict evaluation to length-matched per-attack subsets"
fi

PHASE_B_END=$(date +%s)
echo ""
echo "  Phase A+B total: $(( (PHASE_B_END - START) / 60 )) minutes."

if [ "$PROCEED" != "yes" ]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  STOPPING. Phase C/D would not produce meaningful results"
    echo "  because the length confound is not sufficiently reduced."
    echo "═══════════════════════════════════════════════════════════════"
    exit 0
fi

# ============================================================
# PHASE C: Re-extract activations
# ============================================================
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " PHASE C: Re-extract activations with diverse benign"
echo " (this is the expensive step — 3-5 hours per LLM)"
echo "═══════════════════════════════════════════════════════════════"

# Auto-confirm if running non-interactively
if [ -t 0 ]; then
    read -r -p "  Phase B passed. Proceed with Phase C extraction (6-10 hours)? [y/N] " ans
    case "$ans" in
        [yY]|[yY][eE][sS]) ;;
        *) echo "  Stopping before Phase C."; exit 0 ;;
    esac
fi

# Llama-3 extraction
echo ""
echo "  ── Extracting Llama-3-8B-Instruct activations ──"
if [ -f "$LLAMA_CACHE_NEW" ]; then
    echo "  $LLAMA_CACHE_NEW already exists. Skipping (delete to re-extract)."
else
    python extract_diverse_benign_activations.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --diverse_benign "$DIVERSE_CSV" \
        --existing_cache "$LLAMA_CACHE_OLD" \
        --output "$LLAMA_CACHE_NEW" \
        --layers 0 2 17 24 28 31 \
        2>&1 | tee results/log_extract_llama_diverse.txt
fi

# Vicuna-13B extraction (only if cache exists)
if [ -f "$VICUNA_CACHE_OLD" ]; then
    echo ""
    echo "  ── Extracting Vicuna-13B activations ──"
    if [ -f "$VICUNA_CACHE_NEW" ]; then
        echo "  $VICUNA_CACHE_NEW already exists. Skipping (delete to re-extract)."
    else
        python extract_diverse_benign_activations.py \
            --model lmsys/vicuna-13b-v1.5 \
            --diverse_benign "$DIVERSE_CSV" \
            --existing_cache "$VICUNA_CACHE_OLD" \
            --output "$VICUNA_CACHE_NEW" \
            --layers 0 2 22 31 35 39 \
            2>&1 | tee results/log_extract_vicuna_diverse.txt
    fi
else
    echo "  Vicuna cache not found at $VICUNA_CACHE_OLD; skipping Vicuna extraction."
fi

PHASE_C_END=$(date +%s)
echo ""
echo "  Phase C took $(( (PHASE_C_END - PHASE_B_END) / 60 )) minutes."

# ============================================================
# PHASE D: Re-run main experiments
# ============================================================
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " PHASE D: Re-run main experiments with diverse benign cache"
echo "═══════════════════════════════════════════════════════════════"

# Statistical tests
echo ""
echo "  ── Statistical tests (HPS vs C4 with bootstrap CIs) ──"
python statistical_tests.py \
    --cache "$LLAMA_CACHE_NEW" \
    --n_seeds 5 --n_bootstrap 10000 \
    2>&1 | tee results/log_stats_diverse.txt

# Vicuna imbalance test (only if Vicuna cache exists)
if [ -f "$VICUNA_CACHE_NEW" ]; then
    echo ""
    echo "  ── Vicuna imbalance test ──"
    python vicuna_imbalance_test.py \
        --vicuna_cache "$VICUNA_CACHE_NEW" \
        2>&1 | tee results/log_vicuna_diverse.txt

    # GCG cross-model
    echo ""
    echo "  ── GCG cross-model test ──"
    python gcg_specific_test.py \
        --llama3_cache "$LLAMA_CACHE_NEW" \
        --vicuna_cache "$VICUNA_CACHE_NEW" \
        --attacks_json "$ATTACKS_CLEAN" \
        2>&1 | tee results/log_gcg_diverse.txt
fi

# Radial distribution
echo ""
echo "  ── Radial distribution check ──"
python radial_distribution_check.py \
    --cache "$LLAMA_CACHE_NEW" \
    2>&1 | tee results/log_radial_diverse.txt

PHASE_D_END=$(date +%s)
TOTAL_MIN=$(( (PHASE_D_END - START) / 60 ))

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  PIPELINE COMPLETE"
echo "  Total time: $TOTAL_MIN minutes"
echo "  Finished: $(date)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  Key outputs:"
echo "    Diverse benign:        $DIVERSE_CSV"
echo "    Cleaned attacks:       $ATTACKS_CLEAN"
echo "    Verification (B):      results/verify_saturation_diverse.json"
echo "    Llama-3 cache:         $LLAMA_CACHE_NEW"
echo "    Vicuna cache:          $VICUNA_CACHE_NEW"
echo "    Stats results:         results/log_stats_diverse.txt"
echo "    GCG results:           results/log_gcg_diverse.txt"
echo "    Radial check:          results/log_radial_diverse.txt"
echo ""
echo "  Next: review results and decide paper framing based on Phase B verdict."
echo ""
