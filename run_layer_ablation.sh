#!/bin/bash
# run_layer_ablation.sh — HPS layer-choice ablation experiment
#
# Tests two questions:
#   A. How many layers is optimal? (count varies, location strategy fixed)
#   B. Which layers matter most? (count fixed at 6, location varies)
#
# Each config produces:
#   - probe with that layer set trained
#   - standard recall measured
#   - adaptive 16-token suffix attack (1024 steps)
#   - adaptive recall measured
#
# Results saved to: results/adaptive_attacks/layer_ablation/<config_name>/
#
# Per-config disk usage: ~270 KB (suffix.pt + json + log)
# Total for 14 configs: ~4 MB
#
# Total estimated runtime on 1×A100 80GB: 5-7 hours
# Reduced from 2048→1024 steps to keep budget reasonable while still adequate
# (Phase 4 confirmed HPS fully breaks at 256 steps already).
#
# Run on DGX:
#   cd /mnt/lab/Mo/hps/hps2/hps
#   bash run_layer_ablation.sh 2>&1 | tee results/adaptive_attacks/layer_ablation/run.log
#
# Skip already-completed configs:
#   The script auto-skips configs where the JSON output already exists.

set -euo pipefail

# ---- Configuration ----
CACHE="${CACHE:-results/llama3_activations_cache_diverse_fixed.npz}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-results/adaptive_attacks/layer_ablation}"

# Reduced steps (1024 instead of 2048) since HPS breaks fast.
# This cuts runtime per config from 22 min to ~12 min.
STEPS="${STEPS:-1024}"
N_TRAIN="${N_TRAIN:-20}"
N_EVAL="${N_EVAL:-50}"            # Reduced from 100 for budget
SUFFIX_LEN="${SUFFIX_LEN:-16}"
LAMBDA_OBF="${LAMBDA_OBF:-1.0}"
TARGET_FPR="${TARGET_FPR:-0.01}"

mkdir -p "$OUTPUT_DIR"

START=$(date +%s)
echo "═══════════════════════════════════════════════════════════════════════"
echo "  HPS LAYER ABLATION"
echo "  Started:  $(date)"
echo "  Steps:    $STEPS  | N_train: $N_TRAIN | N_eval: $N_EVAL"
echo "  Output:   $OUTPUT_DIR"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# Sanity checks
if [ ! -f "$CACHE" ]; then
  echo "ERROR: cache not found: $CACHE"; exit 1
fi
if [ ! -f "adaptive_attack.py" ]; then
  echo "ERROR: adaptive_attack.py not found"; exit 1
fi

nvidia-smi --query-gpu=name,memory.free --format=csv | head -3
echo ""

# ============================================================
#  CONFIGURATION DEFINITIONS
# ============================================================
#
# Each entry maps a name → space-separated layer indices.
# All indices are valid for Llama-3-8B-Instruct (32 layers, indices 0-31).

declare -A CONFIGS

# --- A. LAYER COUNT ABLATION (fixed location strategy: spread within safety + middle) ---
CONFIGS["A1_count_2"]="6 12"                                              # minimal
CONFIGS["A2_count_4"]="4 8 12 16"                                         # safety-anchored
CONFIGS["A3_count_6_safety"]="4 6 8 10 12 14"                             # safety zone aligned
CONFIGS["A4_count_8"]="2 4 6 8 10 12 14 16"                               # safety + buffer
CONFIGS["A5_count_12"]="2 4 6 8 10 12 14 16 18 20 22 24"                  # extended
CONFIGS["A6_count_16"]="0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30"       # half of all layers
CONFIGS["A7_count_32_all"]="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31"

# --- B. LAYER LOCATION ABLATION (fixed count: 6, varying positions) ---
CONFIGS["B1_pos_current"]="0 2 17 24 28 31"                               # current HPS choice (baseline)
CONFIGS["B2_pos_early"]="0 1 2 3 4 5"                                     # all early layers
CONFIGS["B3_pos_safety_strict"]="6 7 8 9 10 11"                           # tightly in Li et al. safety range
CONFIGS["B4_pos_safety_buffer"]="4 6 8 10 12 14"                          # safety + 2-layer buffer (= A3)
CONFIGS["B5_pos_middle"]="10 12 14 16 18 20"                              # middle range
CONFIGS["B6_pos_late"]="26 27 28 29 30 31"                                # all late layers
CONFIGS["B7_pos_spread_even"]="0 6 12 18 24 31"                           # uniform every-5
CONFIGS["B8_pos_BO_inspired"]="2 6 12"                                    # Gao et al. style (3 layers, low+middle)

# Order: A's first (count), then B's (location)
ORDER=(A1_count_2 A2_count_4 A3_count_6_safety A4_count_8 A5_count_12 A6_count_16 A7_count_32_all
       B1_pos_current B2_pos_early B3_pos_safety_strict B4_pos_safety_buffer B5_pos_middle B6_pos_late B7_pos_spread_even B8_pos_BO_inspired)

# ============================================================
#  RUN ALL CONFIGS
# ============================================================

CONFIG_COUNT=${#ORDER[@]}
echo "  Total configs: $CONFIG_COUNT"
echo "  Estimated runtime: $(( CONFIG_COUNT * 12 / 60 ))-$(( CONFIG_COUNT * 18 / 60 )) hours"
echo ""

for i in "${!ORDER[@]}"; do
  name="${ORDER[$i]}"
  layers="${CONFIGS[$name]}"
  layer_count=$(echo $layers | wc -w)

  echo "─────────────────────────────────────────────────────────────"
  echo " [$((i+1))/$CONFIG_COUNT] CONFIG: $name"
  echo " Layers: [$layers] (count=$layer_count)"
  echo " Started: $(date)"
  echo "─────────────────────────────────────────────────────────────"

  CONFIG_DIR="$OUTPUT_DIR/$name"
  mkdir -p "$CONFIG_DIR"
  OUTPUT_JSON="$CONFIG_DIR/result.json"

  # Skip if already done
  if [ -f "$OUTPUT_JSON" ]; then
    echo "  [SKIP] Already done; result.json exists"
    echo ""
    continue
  fi

  python adaptive_attack.py \
      --defender hps \
      --model_name "$MODEL" \
      --cache "$CACHE" \
      --layers $layers \
      --n_train_queries $N_TRAIN \
      --n_eval_queries $N_EVAL \
      --steps $STEPS \
      --suffix_length $SUFFIX_LEN \
      --lambda_obf $LAMBDA_OBF \
      --target_fpr $TARGET_FPR \
      --suffix_save "$CONFIG_DIR/suffix.pt" \
      --output "$OUTPUT_JSON" \
      2>&1 | tee "$CONFIG_DIR/log.txt"

  ELAPSED=$(( ($(date +%s) - START) / 60 ))
  echo "  Config done at $ELAPSED min from start"
  echo ""
done

# ============================================================
#  SUMMARY
# ============================================================
TOTAL_MIN=$(( ($(date +%s) - START) / 60 ))
echo "═══════════════════════════════════════════════════════════════════════"
echo "  LAYER ABLATION COMPLETE"
echo "  Started:  $(date -d @$START 2>/dev/null || echo 'start')"
echo "  Finished: $(date)"
echo "  Total:    ${TOTAL_MIN} min"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# Build summary table
python3 - <<EOF
import json
import os
import glob

ablation_dir = "$OUTPUT_DIR"
configs = sorted(os.listdir(ablation_dir))
configs = [c for c in configs if os.path.isdir(os.path.join(ablation_dir, c))]

print()
print(f"{'CONFIG':<24}  {'#layers':>8}  {'AUROC':>8}  {'StdRecall':>10}  {'AdaptRecall':>12}  {'Drop':>8}")
print("-" * 90)

results = []
for c in configs:
    json_path = os.path.join(ablation_dir, c, "result.json")
    if not os.path.exists(json_path):
        print(f"{c:<24}  (incomplete)")
        continue
    try:
        d = json.load(open(json_path))
        layers = d['config']['layers']
        n = len(layers) if isinstance(layers, list) else len(layers.split())
        auroc = d['baseline_auroc']
        std_r = d['standard_attack']['recall_at_fpr']
        adp_r = d['adaptive_attack']['recall_at_fpr']
        drop = d['recall_reduction']
        results.append((c, n, auroc, std_r, adp_r, drop))
        print(f"{c:<24}  {n:>8d}  {auroc:>8.4f}  {std_r:>10.4f}  {adp_r:>12.4f}  {drop:>+8.4f}")
    except Exception as e:
        print(f"{c:<24}  ERROR: {e}")

print()
print("Headline questions:")
print("  1. Which #layers gives highest standard recall?")
print("  2. Which layer location gives highest standard recall?")
print("  3. Does any config show non-zero adaptive recall (i.e., resists attack)?")
print()

# Find best by metric
if results:
    best_std = max(results, key=lambda r: r[3])
    best_adp = max(results, key=lambda r: r[4])
    print(f"  Best standard recall: {best_std[0]} ({best_std[3]:.4f})")
    print(f"  Best adaptive recall: {best_adp[0]} ({best_adp[4]:.4f})")
    if best_adp[4] > 0.05:
        print(f"  ⚠ {best_adp[0]} shows non-zero adaptive recall ({best_adp[4]:.4f}) — investigate!")

# Save summary table to CSV
csv_path = os.path.join(ablation_dir, "summary.csv")
with open(csv_path, "w") as f:
    f.write("config,n_layers,auroc,standard_recall,adaptive_recall,recall_drop\n")
    for r in results:
        f.write(f"{r[0]},{r[1]},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f},{r[5]:+.4f}\n")
print(f"\n  Summary CSV: {csv_path}")
EOF

echo ""
echo "  Detail of results: $OUTPUT_DIR/<config_name>/result.json"
echo "  Suffix files: $OUTPUT_DIR/<config_name>/suffix.pt"
echo "  Per-config logs: $OUTPUT_DIR/<config_name>/log.txt"
