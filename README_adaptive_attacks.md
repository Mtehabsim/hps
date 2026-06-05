# Adaptive Attack Experiment — DGX Instructions

## What This Tests

Bailey et al. (ICLR 2025) showed that all latent-space defenses fail under adaptive attacks: an attacker optimizes a 16-token universal embedding suffix that simultaneously (a) makes the model output harmful text and (b) makes the probe score it as benign. Linear probes drop from 100% recall to 0% under their attack.

We don't know if HPS is more robust, less robust, or equally vulnerable. This experiment tests it directly.

## Files To Copy To DGX

```bash
# From your laptop, scp these to /mnt/lab/Mo/hps/hps2/hps/ on DGX:
scp adaptive_attack.py        dgx03:/mnt/lab/Mo/hps/hps2/hps/
scp run_adaptive_attacks.sh   dgx03:/mnt/lab/Mo/hps/hps2/hps/
```

## Prerequisites On DGX

These should already exist from your previous pipeline:
- `results/llama3_activations_cache_diverse_fixed.npz` (the canonical cache)
- `results/data_harmless_diverse.csv` (5905 diverse benign prompts)
- Llama-3-8B-Instruct downloaded in HF cache

If any are missing, run the existing pipeline first.

## How To Run

### Quick start (single A100 80GB)

```bash
cd /mnt/lab/Mo/hps/hps2/hps
chmod +x run_adaptive_attacks.sh

# Set HF cache (if you haven't already)
export HF_HOME=/mnt/lab/Mo/hf_cache/

# Run all phases (~12-24 hours total)
nohup bash run_adaptive_attacks.sh > results/adaptive_attacks/run.log 2>&1 &

# Watch progress
tail -f results/adaptive_attacks/run.log
```

### Phase-by-phase (recommended for first run)

Run Phase 1 first as a sanity check before committing to the full ~24 hours:

```bash
# Phase 1 only: attack C4 (~3-4 hr)
SKIP_PHASE_2=1 SKIP_PHASE_3=1 SKIP_PHASE_4=1 bash run_adaptive_attacks.sh

# If C4 results look reasonable (recall drops from ~99% to ~0%), proceed to Phase 2:
SKIP_PHASE_1=1 SKIP_PHASE_3=1 SKIP_PHASE_4=1 bash run_adaptive_attacks.sh

# Then transfer + budget scaling:
SKIP_PHASE_1=1 SKIP_PHASE_2=1 bash run_adaptive_attacks.sh
```

## What Each Phase Does

### Phase 1 (~3-4 hr): C4 attack — sanity check

Trains a 16-token suffix to fool C4 while jailbreaking. Should reproduce Bailey's "recall → ~0%" result. If this fails (e.g., suffix doesn't converge or recall doesn't drop), there's a bug in the pipeline before Phase 2 is meaningful.

### Phase 2 (~6-8 hr): HPS attack — the key experiment

Same attack against HPS. Three possible outcomes:

| Outcome | What it means | Impact for paper |
|---------|---------------|------------------|
| HPS recall drops similarly to C4 (~0%) | Field-wide vulnerability extends to hyperbolic | Confirms Bailey for HPS. TMLR submission with adversarial robustness section. |
| HPS recall partially drops (e.g., 0.3-0.5) | Geometric prior provides marginal robustness | Mild positive contribution: "geometric methods more robust by Xpp" |
| HPS recall stays high (>0.7) | HPS resists Bailey-style attacks | Major positive finding. Possible USENIX submission. |

### Phase 3 (~1-2 hr): Cross-method transfer

Tests:
- Suffix trained against C4 → does it bypass HPS?
- Suffix trained against HPS → does it bypass C4?

Per Bailey et al.: attacks transfer between architecturally different probes. If HPS proves robust to its own targeted attack, the C4-transfer attack tests whether HPS resists attacks designed for other architectures.

### Phase 4 (~3-4 hr): Attack budget scaling

If HPS shows any robustness in Phase 2, this measures *how much*. Plots HPS recall vs. attack steps (256, 512, 1024 vs the 2048 default). If HPS requires 4× the attack budget to break, that's a quantifiable advantage even if it eventually breaks.

## Memory Requirements

Single A100 80GB is sufficient:
- Llama-3-8B in fp16: ~16 GB
- Activation memory during backward pass: ~10-15 GB
- Suffix gradients: <100 MB
- Probe modules: <500 MB
- Total: ~30 GB of 80 GB available

If you have 2 GPUs, you could run Phases 1 and 2 in parallel:
```bash
CUDA_VISIBLE_DEVICES=0 SKIP_PHASE_2=1 bash run_adaptive_attacks.sh &
CUDA_VISIBLE_DEVICES=1 SKIP_PHASE_1=1 bash run_adaptive_attacks.sh &
wait
```

## Output Files

After completion, you'll have in `results/adaptive_attacks/`:

```
attack_c4.json              ← Phase 1: standard vs adaptive recall on C4
attack_hps.json             ← Phase 2: standard vs adaptive recall on HPS
transfer_c4_to_hps.json     ← Phase 3a: C4-suffix tested on HPS
transfer_hps_to_c4.json     ← Phase 3b: HPS-suffix tested on C4
attack_hps_256steps.json    ← Phase 4: HPS at different attack budgets
attack_hps_512steps.json
attack_hps_1024steps.json
suffix_c4.pt                ← The trained adversarial suffix (for inspection)
suffix_hps.pt
log_phase{1-4}_*.txt        ← Per-phase logs
```

The bash script prints a summary at the end with the headline numbers.

## Interpreting Results

The key comparison is `recall_reduction` across the JSON files:

```bash
python3 -c "
import json, glob
for f in sorted(glob.glob('results/adaptive_attacks/attack_*.json')):
    d = json.load(open(f))
    print(f'{f:60s}  std={d[\"standard_attack\"][\"recall_at_fpr\"]:.3f}  adap={d[\"adaptive_attack\"][\"recall_at_fpr\"]:.3f}  drop={d[\"recall_reduction\"]:.3f}')
"
```

Expected patterns:
- If C4 drops from ~0.99 to ~0.0 (drop ≈ 0.99): pipeline works, Bailey reproduced
- If HPS drops from ~0.99 to ~0.0 (drop ≈ 0.99): HPS no more robust than C4
- If HPS drops from ~0.99 to ~0.5 (drop ≈ 0.5): geometric prior provides ~50pp robustness margin

## Tuning If Phase 1 Doesn't Reproduce Bailey

If C4 attack doesn't drive recall down to near-zero, try:

```bash
# More steps:
STEPS=4096 SKIP_PHASE_2=1 SKIP_PHASE_3=1 SKIP_PHASE_4=1 bash run_adaptive_attacks.sh

# Higher obfuscation weight:
LAMBDA_OBF=5.0 SKIP_PHASE_2=1 SKIP_PHASE_3=1 SKIP_PHASE_4=1 bash run_adaptive_attacks.sh

# Higher learning rate:
# Edit adaptive_attack.py: --lr 0.05 (default 0.01)
```

If C4 recall doesn't drop after these adjustments, the attack implementation has an issue and should be debugged before Phase 2.

## Time Budget

| Phase | Steps | Queries | Approx hours (1xA100) |
|-------|-------|---------|----------------------|
| 1: C4 | 2048 | 20 train | 3-4 |
| 2: HPS | 2048 | 20 train | 6-8 |
| 3: Transfer | eval only | 30 each | 1-2 |
| 4: HPS budget | 256/512/1024 | 20 train | 3-4 |
| **Total** | | | **13-18 hours** |

Plan for ~18 hours with margin. Run overnight Friday for results Saturday.

## What If A Phase Crashes

The phases write logs continuously, so you can resume any phase by re-running with `SKIP_*` for the completed ones:

```bash
# If phase 2 crashed midway:
SKIP_PHASE_1=1 bash run_adaptive_attacks.sh
```

The JSON output is only written at the end, so if a phase crashed, that phase needs to re-run.

## Open Questions To Discuss After Results

1. **If HPS holds up:** Is it the geometry, or the multi-layer trajectory features? (Compare to HPS-Euclidean — but HPS-Euc training isn't yet implemented in the script. Add it for Phase 5.)

2. **If HPS breaks:** What does the attack budget scaling show? Is there a meaningful difference at low attack budget?

3. **What's the practical threat model?** Bailey's attack is white-box with full probe access. In production, attackers usually don't have probe gradients. Does this matter for the paper's framing?

4. **Should we run Vicuna too?** HPS already fails on Vicuna without adaptive attacks. Adding adaptive attacks would tell us about robustness of the failure mode itself — not the most interesting question.
