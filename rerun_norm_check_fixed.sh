#!/bin/bash
# Quick fix for the Phase 2.1 bug — re-run norm_check on the FIXED cache
# Run this AFTER your pipeline finishes (or now in another terminal)

echo "═══════════════════════════════════════════════════════════════"
echo " RE-RUN NORM CHECK ON FIXED CACHE"
echo "═══════════════════════════════════════════════════════════════"

# Norm check on the FIXED Llama-3 cache (with full sequences for both)
python norm_check_diverse.py \
    --llama_cache results/llama3_activations_cache_diverse_fixed.npz \
    --vicuna_cache results/vicuna_activations_cache_diverse_fixed.npz \
    --output results/norm_check_diverse_fixed.json \
    2>&1 | tee results/log_norm_recheck_fixed.txt

echo ""
echo "Compare to original (broken) check at results/norm_check_diverse.json"
echo "  Old (broken): norm-only AUROC = 1.000 (max_length mismatch)"
echo "  Expected fixed: AUROC ~0.7-0.9 (some norm signal remains, but not perfect)"
