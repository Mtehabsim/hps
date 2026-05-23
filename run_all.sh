#!/bin/bash
set -e  # Exit on first error
set -o pipefail  # Pipeline fails if any command fails

echo "═══════════════════════════════════════════════════════════"
echo "  HPS Paper — Full Re-run (all fixes applied)"
echo "  $(date)"
echo "═══════════════════════════════════════════════════════════"

cd "$(dirname "$0")"

# 0. Backup current paper draft (preserve "before" numbers)
echo ""
echo "── Step 0: Backing up paper_draft.md ──"
cp -n paper_draft.md paper_draft_v_before_rerun.md 2>/dev/null || true
echo "  Saved → paper_draft_v_before_rerun.md"

# 1. Clear stale caches and projections
echo ""
echo "── Step 1: Clearing stale caches and projections ──"
mkdir -p results
rm -f results/llama3_activations_cache.npz
rm -f results/hps_llama3_projection.pt
rm -f results/hps_adv_projection.pt
rm -f results/hps_projection_head.pt
echo "  Done."

# 2. Llama-3 main experiment (extracts activations, trains HPS+RTV+Ensemble, cross-attack 3 seeds)
echo ""
echo "── Step 2: hps_llama3.py (Llama-3-8B main) ──"
python hps_llama3.py \
  --test-attacks llama3_attacks.json \
  --harmless data_harmless_6500.csv \
  --harmful data_harmful_100.csv 2>&1 | tee results/log_hps_llama3.txt
echo "  ✓ hps_llama3.py complete"

# 3. Supplementary (multi-seed stability, Euclidean cross-attack, learning curve, timing)
# Depends on: results/llama3_activations_cache.npz from step 2
echo ""
echo "── Step 3: paper_supplementary.py (multi-seed, Euclidean cross-attack) ──"
python paper_supplementary.py \
  --test-attacks llama3_attacks.json \
  --harmless data_harmless_6500.csv 2>&1 | tee results/log_supplementary.txt
echo "  ✓ paper_supplementary.py complete"

# 4. Vicuna main experiment (HPS + fair Euclidean ablation)
echo ""
echo "── Step 4: experiment7.py (Vicuna-13B main) ──"
python experiment7.py 2>&1 | tee results/log_experiment7.txt
echo "  ✓ experiment7.py complete"

# 5. Adversarial training (HPS-Adv, fixed threshold)
# Depends on: trained projection from experiment7
echo ""
echo "── Step 5: experiment12.py (HPS-Adv adversarial training) ──"
python experiment12.py 2>&1 | tee results/log_experiment12.txt
echo "  ✓ experiment12.py complete"

# 6. Adaptive PGD attack (fixed overlap + threshold)
# Depends on: trained projection from experiment7
echo ""
echo "── Step 6: attack_ensemble.py (adaptive PGD) ──"
python attack_ensemble.py 2>&1 | tee results/log_attack_ensemble.txt
echo "  ✓ attack_ensemble.py complete"

# 7. Regenerate figures
echo ""
echo "── Step 7: visualize_hps.py (paper figures) ──"
python visualize_hps.py 2>&1 | tee results/log_visualize.txt
echo "  ✓ visualize_hps.py complete"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ALL DONE — $(date)"
echo "  Logs saved to results/log_*.txt"
echo "  Update paper_draft.md with new numbers from logs."
echo "═══════════════════════════════════════════════════════════"
