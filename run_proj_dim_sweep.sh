#!/usr/bin/env bash
# ============================================================
#  HPS PROJECTION-DIMENSION SWEEP
#  Tests the mentor's hypothesis: "C4 wins because it uses the full hidden
#  space without projection." We sweep HPS's projection dim (4096 -> proj_dim)
#  and measure whether HPS climbs toward C4 as the per-layer compression is
#  relaxed. C4 is included once as a (projection-free) reference line.
#
#  NOTE: this relieves ONLY the per-layer projection bottleneck (4096->proj_dim).
#  HPS still summarizes the 6-layer path into 12 trajectory features afterward,
#  so HPS is NOT expected to exactly equal C4 even at proj_dim=4096. If it
#  plateaus below C4, the 12-feature trajectory summarization is the real ceiling.
#
#  Usage:
#    bash run_proj_dim_sweep.sh
#    PROJ_DIMS="64 256 1024 4096" bash run_proj_dim_sweep.sh   # custom sweep
# ============================================================
set -euo pipefail

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
CACHE="${CACHE:-results/llama3_activations_cache_alllayers.npz}"
LAYERS="${LAYERS:-0 6 12 18 24 31}"          # B7 (best layers from ablation)
PROJ_DIMS="${PROJ_DIMS:-64 128 256 512 1024 2048 4096}"
OUTDIR="${OUTDIR:-results/proj_dim_sweep}"
N_TRAIN="${N_TRAIN:-20}"
N_EVAL="${N_EVAL:-50}"
STEPS="${STEPS:-1024}"
TARGET_FPR="${TARGET_FPR:-0.01}"
DEFENDER="${DEFENDER:-hps}"                   # set DEFENDER=hps_euc for the flat control

mkdir -p "$OUTDIR"
LZ_TAG=$(echo "$LAYERS" | tr ' ' '_')

echo "═══════════════════════════════════════════════════════════════════════"
echo "  HPS PROJECTION-DIMENSION SWEEP"
echo "  Defender:  $DEFENDER"
echo "  Cache:     $CACHE"
echo "  Layers:    $LAYERS"
echo "  Proj dims: $PROJ_DIMS"
echo "  Settings:  n_train=$N_TRAIN n_eval=$N_EVAL steps=$STEPS fpr=$TARGET_FPR"
echo "  Output:    $OUTDIR"
echo "═══════════════════════════════════════════════════════════════════════"

# ---- C4 reference (projection-free; run once) ----
C4_OUT="$OUTDIR/c4_ref_${LZ_TAG}.json"
if [ ! -f "$C4_OUT" ]; then
  echo ""
  echo "── C4 reference (no projection) ──"
  python adaptive_attack.py --defender c4 \
      --model_name "$MODEL" --cache "$CACHE" --layers $LAYERS \
      --n_train_queries "$N_TRAIN" --n_eval_queries "$N_EVAL" --steps "$STEPS" \
      --suffix_length 16 --lambda_obf 1.0 --target_fpr "$TARGET_FPR" \
      --output "$C4_OUT"
fi

# ---- HPS sweep over projection dim ----
for PD in $PROJ_DIMS; do
  OUT="$OUTDIR/${DEFENDER}_proj${PD}_${LZ_TAG}.json"
  if [ -f "$OUT" ]; then
    echo "── $DEFENDER proj_dim=$PD already done ($OUT) ──"
    continue
  fi
  echo ""
  echo "── $DEFENDER proj_dim=$PD ──"
  python adaptive_attack.py --defender "$DEFENDER" \
      --model_name "$MODEL" --cache "$CACHE" --layers $LAYERS \
      --proj_dim "$PD" \
      --n_train_queries "$N_TRAIN" --n_eval_queries "$N_EVAL" --steps "$STEPS" \
      --suffix_length 16 --lambda_obf 1.0 --target_fpr "$TARGET_FPR" \
      --output "$OUT"
done

# ---- Readout ----
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  PROJECTION-DIM SWEEP SUMMARY (std/adapt recall @ ${TARGET_FPR} FPR)"
echo "═══════════════════════════════════════════════════════════════════════"
python3 - "$OUTDIR" "$DEFENDER" "$LZ_TAG" <<'PY'
import json, os, sys, glob
outdir, defender, lz_tag = sys.argv[1], sys.argv[2], sys.argv[3]

def load(f):
    d = json.load(open(f))
    return (d.get("baseline_auroc"),
            d["standard_attack"]["recall_at_fpr"],
            d["adaptive_attack"]["recall_at_fpr"],
            d.get("config", {}).get("proj_dim"))

c4 = os.path.join(outdir, f"c4_ref_{lz_tag}.json")
print(f"{'probe':14s} {'proj_dim':>8} {'AUROC':>8} {'std':>6} {'adapt':>6}")
print("-" * 48)
if os.path.exists(c4):
    a, s, ad, _ = load(c4)
    print(f"{'c4 (full)':14s} {'-':>8} {a:8.4f} {s:6.3f} {ad:6.3f}")
rows = []
for f in glob.glob(os.path.join(outdir, f"{defender}_proj*_{lz_tag}.json")):
    a, s, ad, pd = load(f)
    rows.append((pd if pd is not None else 0, a, s, ad))
for pd, a, s, ad in sorted(rows):
    print(f"{defender:14s} {pd:>8} {a:8.4f} {s:6.3f} {ad:6.3f}")
print()
print("Read: does HPS 'std' climb toward C4 'std' as proj_dim grows?")
print("  meets C4   -> projection width was the bottleneck (mentor's hypothesis confirmed)")
print("  plateaus   -> 12-feature trajectory summarization is the real ceiling")
print("  flat       -> the geometry/feature representation limits HPS, not the dimension")
PY
