#!/usr/bin/env bash
# ============================================================
#  run_gen_comparison.sh
#  Generation-based probe comparison + control matrix + jailbreak verification.
#
#  Produces, for each gen probe, BOTH the adaptive-attack metrics AND whether
#  the attack actually jailbroke the model — because adaptive recall is only
#  comparable across suffixes that achieve similar jailbreak success.
#
#  Probes:
#    hps_gen      (hyperbolic, gen)      x {mean, trajectory, both}
#    hps_euc_gen  (flat control, gen)    x {mean, trajectory, both}
#    c4_gen       (full-space control)   (no aggregation)
#
#  Reuses earlier "both" runs (attack_hps_gen_flrt.json / attack_hps_euc_gen_flrt.json)
#  if present, to avoid recompute.
#
#  Usage:
#    bash run_gen_comparison.sh
#    N_JB=20 bash run_gen_comparison.sh
# ============================================================
set -uo pipefail

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
CACHE="${CACHE:-results/llama3_activations_cache_diverse_fixed.npz}"
GEN_CACHE="${GEN_CACHE:-results/llama3_gen_activations_cache.npz}"
LAYERS="${LAYERS:-0 6 12 18 24 31}"
OUTDIR="${OUTDIR:-results/flrt_attacks}"
N_JB="${N_JB:-15}"

COMMON="--model_name $MODEL --cache $CACHE --gen_cache $GEN_CACHE --layers $LAYERS"
mkdir -p "$OUTDIR"

echo "═══════════════════════════════════════════════════════════════════════"
echo "  GENERATION-BASED PROBE COMPARISON"
echo "  Gen cache: $GEN_CACHE   Layers: $LAYERS   JB queries: $N_JB"
echo "═══════════════════════════════════════════════════════════════════════"

# ---- Adopt earlier 'both' runs (avoid recomputing) ----
adopt() {  # canonical_attack canonical_suffix tag
  local cattack="$1" csuffix="$2" tag="$3"
  if [ -f "$cattack" ] && [ ! -f "$OUTDIR/attack_${tag}.json" ]; then
    cp "$cattack" "$OUTDIR/attack_${tag}.json"
    [ -f "$csuffix" ] && cp "$csuffix" "$OUTDIR/suffix_${tag}.json"
    echo "  adopted existing run -> $tag"
  fi
}
adopt "$OUTDIR/attack_hps_gen_flrt.json"     "$OUTDIR/suffix_hps_gen_flrt.json"     hps_gen_both
adopt "$OUTDIR/attack_hps_euc_gen_flrt.json" "$OUTDIR/suffix_hps_euc_gen_flrt.json" hps_euc_gen_both

# ---- Run attacks (skip if output already exists) ----
run_attack() {  # defender aggregation tag
  local def="$1" agg="$2" tag="$3"
  local out="$OUTDIR/attack_${tag}.json"
  if [ -f "$out" ]; then echo "  [skip] $tag (exists)"; return 0; fi
  local aggflag=""
  [ -n "$agg" ] && aggflag="--gen_aggregation $agg"
  echo ""
  echo "── attack: $tag ──"
  python flrt_attack.py --defender "$def" $COMMON $aggflag \
      --suffix_save "$OUTDIR/suffix_${tag}.json" \
      --output "$out" 2>&1 | tee "$OUTDIR/log_${tag}.txt"
}

run_attack hps_gen      mean       hps_gen_mean
run_attack hps_gen      trajectory hps_gen_trajectory
run_attack hps_gen      both       hps_gen_both
run_attack hps_euc_gen  mean       hps_euc_gen_mean
run_attack hps_euc_gen  trajectory hps_euc_gen_trajectory
run_attack hps_euc_gen  both       hps_euc_gen_both
run_attack c4_gen       ""         c4_gen

# ---- Jailbreak verification for every suffix ----
TAGS="hps_gen_mean hps_gen_trajectory hps_gen_both hps_euc_gen_mean hps_euc_gen_trajectory hps_euc_gen_both c4_gen"
for tag in $TAGS; do
  jb="$OUTDIR/jb_${tag}.json"
  suf="$OUTDIR/suffix_${tag}.json"
  if [ -f "$jb" ]; then echo "  [skip] jailbreak-check $tag (exists)"; continue; fi
  if [ ! -f "$suf" ]; then echo "  [skip] jailbreak-check $tag (no suffix)"; continue; fi
  echo ""
  echo "── jailbreak-check: $tag ──"
  python verify_jailbreak.py --suffix "$suf" --model "$MODEL" \
      --n "$N_JB" --output "$jb" 2>&1 | tee "$OUTDIR/log_jb_${tag}.txt"
done

# ---- Final matrix ----
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  FINAL MATRIX (std/adapt recall @1%FPR + jailbreak compliance of suffix)"
echo "═══════════════════════════════════════════════════════════════════════"
python3 - "$OUTDIR" "$TAGS" <<'PY'
import json, os, sys
outdir = sys.argv[1]; tags = sys.argv[2].split()
def g(path, *keys, default=None):
    try:
        d = json.load(open(path))
        for k in keys: d = d[k]
        return d
    except Exception:
        return default
print(f"{'probe/agg':26s} {'AUROC':>7} {'std':>6} {'adapt':>6} {'JB-comply':>9}")
print("-"*62)
for t in tags:
    a = os.path.join(outdir, f"attack_{t}.json")
    jb = os.path.join(outdir, f"jb_{t}.json")
    auroc = g(a, "baseline_auroc"); std = g(a, "standard_attack","recall_at_fpr")
    adp = g(a, "adaptive_attack","recall_at_fpr")
    rr = g(jb, "refusal_rate_suffix")
    comply = (1 - rr) if rr is not None else None
    def f(x, p=3): return f"{x:.{p}f}" if isinstance(x,(int,float)) else "  -  "
    print(f"{t:26s} {f(auroc,4):>7} {f(std):>6} {f(adp):>6} {f(comply):>9}")
print()
print("Valid robustness comparison: look ONLY at probes whose JB-comply is")
print("similar AND non-trivial. High adapt recall + high JB-comply = real robustness.")
print("High adapt recall + low JB-comply = the attack just failed to jailbreak.")
PY
