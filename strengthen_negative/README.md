# Strengthening the Negative Result

This folder contains experiments to strengthen the paper's negative claim from
"HPS doesn't help" to "no hyperbolic method helps, and linear features capture
nearly all available information."

## Why These Experiments

The reviewer's critique was that we can't distinguish between:
1. Hyperbolic geometry doesn't help in general
2. Our specific HPS implementation doesn't help

These experiments address that ambiguity:
- **Experiment 1:** Tests *multiple* hyperbolic methods. If they all fail, the
  claim "hyperbolic doesn't help" is much stronger.
- **Experiment 4:** Tests an information-theoretic upper bound. If linear
  features capture all the discriminative information, no nonlinear method
  (hyperbolic or otherwise) can improve over linear baselines.

## Files

| File | Purpose | Time on DGX |
|------|---------|-------------|
| `experiment1_hyperbolic_methods.py` | Tests Hyperbolic AE + Hyperbolic NN + Lorentz centroid distance | ~30-60 min |
| `experiment4_information_theoretic.py` | MI estimation, PCA scan, saturation argument | ~15-30 min |
| `helpers/lorentz_ops.py` | Lorentz manifold primitives | n/a |

## Running

Both scripts use the existing activation caches:
- `results/llama3_activations_cache.npz` (Llama-3-8B)
- `results/vicuna_activations_cache.npz` (Vicuna-13B)

```bash
# From the main hps directory
cd /mnt/lab/Mo/hps/hps2/hps

python strengthen_negative/experiment1_hyperbolic_methods.py 2>&1 | \
    tee results/log_strengthen_exp1.txt

python strengthen_negative/experiment4_information_theoretic.py 2>&1 | \
    tee results/log_strengthen_exp4.txt
```

## Expected Outputs

### Experiment 1 — Hyperbolic Methods

Compares 4-5 hyperbolic methods + C4 baseline on Llama-3 and Vicuna:
- `HPS-Lorentz` (existing baseline, for comparison)
- `Hyperbolic-AE` (autoencoder with reconstruction loss in Lorentz space)
- `Hyperbolic-NN` (stacked Lorentz layers with nonlinearity)
- `Lorentz-Centroid` (distance to class centroids on hyperboloid)
- `Euclidean (matched)` (parameter-matched Euclidean variant)
- `C4` (linear probe on mean-pooled activations)

**Expected outcome:** All hyperbolic methods either match or underperform C4.

### Experiment 4 — Information-Theoretic

Three sub-tests:
- **4A. Saturation argument** — confirms C4 reaches the maximum possible AUROC
- **4B. Mutual information** — compares I(raw; labels) vs I(linear; labels)
- **4C. PCA dimensionality scan** — shows that 1-64 dims captures essentially
  all the available signal
- **4D. Random projection scan** — shows linear baseline isn't dependent on
  specific dimensionality reduction

**Expected outcome:** Linear features capture nearly all discriminative
information; the gap between linear and "any nonlinear method that could
exist" is bounded by something close to zero.

## What These Don't Address

- **Theoretical proof** (Experiment 3) — we explicitly don't attempt this.
  A formal proof that "hyperbolic geometry cannot help under linear
  separability" would be a separate paper-length theory project.
- **Multiple hierarchy definitions** (Experiment 2) — we already have
  δ-hyperbolicity from `vicuna_diagnostic.py`. Adding more measures has
  low value-to-effort ratio.

## Integration with Main Paper

After running, the results integrate into the paper as:
1. New table in Section 5: "All hyperbolic variants vs C4" (from Exp 1)
2. New section: "An information-theoretic upper bound" (from Exp 4)
3. Strengthens the abstract claim from "HPS doesn't help" to "no hyperbolic
   method we tested helps; linear features are information-theoretically
   sufficient for this task on these models"
