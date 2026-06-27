# old-tries — superseded early iterations

These are exploratory iterations that **predate** the current `src/` implementation. The whole
project pivoted away from them; kept here for provenance, not as runnable code. Each file below was
read and judged individually — the rest of those folders (scratch viz, near-duplicate variants,
single-feature probes, the old 4-test validation suite) was deleted (recoverable via git history).

## What this lineage was
The early approach tried to detect jailbreaks (esp. GCG suffixes) on Vicuna/Llama hidden states using
**hand-engineered trajectory "physics"** (Menger curvature, CLDP, velocity, attention entropy) +
**Mahalanobis anomaly scoring**, and a naïve (untrained) Poincaré projection. It hit FPR walls and was
abandoned for the current learned-Lorentz-projection + hierarchical-retrieval framing in `src/`.

## Kept files (representative milestones)
- `zero-try/shadow3.py` — the canonical "five-pillar" physics experiment (κ/CLDP/velocity/entropy/ADD + Mahalanobis); the ancestor of all the shadow variants.
- `zero-try/shadow2.py` — the most-evolved GCG detector (adds PPL-ratio, rare-token runs, tiered voting); its comments narrate *why* feature-engineering was dropped.
- `zero-try/01_ground_truth_labeler.py` — the refusal-prefix labeling oracle that produced `validated_datasets.json`.
- `zero-try/step4_massive_fpr_telemetry.py` — the only mass-throughput (10k–50k prompt) benign-FPR harness.
- `first-try/test1_gromov_delta.py` — Gromov δ-hyperbolicity of the residual stream; the "is the space intrinsically hyperbolic" check with no counterpart in `src/`.
- `second-try/HPS_Document.md` — the original Llama-3-8B design doc (pre-validation framing, the early optimistic ~1.0 AUROC results).
- `paper_draft_v_before_rerun.md` — an earlier (RTV-framed) paper draft, superseded by `markdowns/hps_preprint.md`.
- `loose-data/` — root-level datasets from the DGX (data_harmful/harmless CSVs, llama3_attacks*.json) that no current `src/` script references; kept as data.

Everything that survives and matters now lives in `src/` (see the top-level README).
