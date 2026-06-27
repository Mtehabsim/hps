# Restructure Plan — `hps-geometry`

> **Status:** Plan only. No file changes have been made. Execute on your go-ahead.
> **Decisions locked in:**
> 1. **Full `src/hps/` package** — pip-installable library + thin CLI drivers (de-duped geometry & config).
> 2. **Git history rewrite** — `git filter-repo` to purge the 1.3 GB of run logs from all past commits (DESTRUCTIVE, final phase).
> 3. **All work on branches — `main` is never touched directly.** (See Branch Strategy.)

---

## Why restructure

The project currently follows zero software-engineering structure. Six concrete problems:

| # | Problem (today) | Consequence |
|---|---|---|
| 1 | 28 scripts dumped flat in root, no grouping | Can't tell the positive line from the dead negative line from plotting |
| 2 | **Every file is both a library and a CLI** (argparse + `__main__` in all 22 runnable scripts) | Importing one function (`ProtoNet`) executes a 760-line script's module load |
| 3 | Geometry math duplicated across `hps_core.py`, `utils.py`, `hyperbolic_retrieval.py` | No single source of truth; a bug fixed in one isn't fixed in the others |
| 4 | Config duplicated (`config.py` says `PROJECTION_DIM=64`, `hps_core.py` re-hardcodes `d_proj=64`) | `config.py` is dead but looks authoritative |
| 5 | Inputs, outputs, logs, binaries all mixed in `results/` (470 files) + 1.3 GB committed | Can't find the canonical figure; clone is ~1.4 GB |
| 6 | Docs untracked (`markdowns/`), README describes a deleted codebase | A new reader is actively misled |

**The core fix** (resolves #1–#4 at once): a **pure `src/hps/` library** (importable, no side effects) plus **thin `experiments/` drivers** (argparse only, importing from `hps`).

---

## Branch strategy (main is never touched)

```
main ─────────────────────────────────────────────────────●  (untouched until final approved merge)
  │                                                        ▲
  └─► restructure  (long-lived integration branch off main)│
        │                                                  │
        ├─► restructure/phase-0-safety   ──merge──►        │
        ├─► restructure/phase-1-library  ──merge──►  restructure
        ├─► restructure/phase-2-folders  ──merge──►  (accumulates all phases)
        ├─► restructure/phase-3-io       ──merge──►        │
        ├─► restructure/phase-4-vendor   ──merge──►        │
        ├─► restructure/phase-5-docs     ──merge──►        │
        └─► restructure/phase-6-rewrite  ──merge──►  ──────┘  ⚠ DESTRUCTIVE
```

Rules:
- **`main` receives nothing until the very end**, and only via an explicit, separately-approved merge/PR after every phase is verified.
- A long-lived **`restructure`** integration branch is created off `main`.
- **Each phase is done on its own short-lived branch** (`restructure/phase-N-...`) cut from `restructure`, and merged back into `restructure` only after its green checkpoint passes. This keeps every phase independently revertable.
- The history-rewrite (Phase 6) is performed on the `restructure` branch (with a `.git` mirror backup first). Because it rewrites SHAs, it is the last thing done before the final merge to `main`.
- Before starting: `git clone --mirror` backup of the repo so the rewrite is always recoverable.

---

## Target structure

```
hps-geometry/
├── README.md              ← rewritten for the CURRENT project (not the stale "Sentinel" one)
├── pyproject.toml         ← declares `hps` package + console entry points + deps
├── .gitignore             ← adds *.npz *.pt runs/ obf_env/ results/**/*.log
│
├── src/hps/               ←─── THE LIBRARY (pure functions/classes, zero argparse, no import-time work)
│   ├── geometry.py        ← Lorentz/Poincaré ops (merge hps_core + dupes in utils/hyperbolic_retrieval)
│   ├── protonet.py        ← ProtoNet, train_eval, tpr_at_fpr, extract_benign  (from hierarchical_detector)
│   ├── retrieval.py       ← load_reps + mAP/recall metrics  (extracted from hyperbolic_retrieval)
│   ├── curvature.py       ← fit_all, _distortion, graph_orc  (from embedding_distortion + helm_token_curvature)
│   ├── taxonomy.py        ← TAXONOMY/SUBTAXONOMY constants + assign()  (from harm_taxonomy)
│   ├── extraction.py      ← load_model + hook extraction  (rescued from utils.py + archive/extraction)
│   ├── caching.py         ← npz load/save + norm-confound gate  (from inspect_cache)
│   └── config.py          ← ONE dataclass of paths + hyperparams (replaces config.py + scattered hardcodes)
│
├── experiments/           ←─── THIN DRIVERS (argparse + call into hps; grouped by science line)
│   ├── retrieval/         (the POSITIVE result)  hyperbolic_retrieval, mmlu_taxonomy,
│   │                       data_driven_hierarchy, curvature_sweep, openset_detection, openset_attacks
│   ├── detection/         (the NEGATIVE result)  statistical_tests, radial_distribution_check, norm_controlled_eval*
│   ├── methods/           (curvature measurement + arbiters/controls)  embedding_distortion,
│   │                       helm_token_curvature, verify_curvature_claim, rpf_on_cache,
│   │                       harm_vs_dataset_eval, label_agreement, inspect_cache
│   └── data/              harm_taxonomy.py  (build/label/deepen/extract)
│
├── figures/               ←─── FIGURE GENERATORS  calibration_panel, plot_leaves, plot_lambda_sweep,
│                                                   plot_masking, plot_obfuscation, attack_cost_curve
├── data/                  ←─── INPUTS only (prompt CSVs, taxonomy JSON)
├── results/
│   ├── caches/   *.npz  (gitignored + MANIFEST.md: how to regenerate)
│   ├── metrics/  *.json (kept — small, text)
│   ├── figures/  *.png  (CURRENT only: mentor_summary.png …; stale → archive)
│   └── logs/            (gitignored)
│
├── docs/                  ←─── was markdowns/, now COMMITTED
│   ├── reconciliation_memo.md   (read-first)
│   ├── preprint.md / reference.md / walkthrough.md / prereg.md / presentation.md
│   └── history/   (research_journey, evaluation_report, PROJECT_INVENTORY — older, contradicted)
│
├── third_party/obfuscated-activations/   ←─── vendored, pruned; HPSMetric refactored into src/hps/
└── archive/               ←─── unchanged, EXCEPT archive/extraction → src/hps/extraction.py
```

---

## Phases

Each phase = its own branch off `restructure`, ends at a **green checkpoint**
(`pip install -e . && python -c "import hps"` + a smoke-run of `hyperbolic_retrieval --help`), then merges into `restructure`. The history rewrite goes **last**.

```
Phase 0  Safety net        → commit docs/, write .gitignore, git rm --cached the bloat
Phase 1  Extract src/hps/  → the real engineering: lib out of CLIs, de-dupe geometry+config   ★ riskiest
Phase 2  Folder the drivers→ git mv into experiments/{retrieval,detection,methods,data}, figures/
Phase 3  Inputs/outputs    → split results/{caches,metrics,figures,logs}, fix 2 broken scripts
Phase 4  Vendor hygiene    → third_party/, refactor HPSMetric out of obf_reps
Phase 5  Docs + ergonomics → rewrite README, annotate contradicted numbers, console scripts
Phase 6  HISTORY REWRITE   → git filter-repo (DESTRUCTIVE — final, after everything verified)  ⚠
```

### Phase 0 — Safety net (non-destructive, ~15 min)
- `git add docs/` — rescue the 12 untracked write-ups now in `markdowns/` (problem 6).
- Write the real `.gitignore`; `git rm -r --cached` the 1.3 GB `runs/`, the `obf_env/` venv, and `*.pt`/`*.npz`. *(Working tree untouched; this only stops tracking. Deep history shrink is Phase 6.)*

### Phase 1 — Extract the library ★ (the real work; fixes problems 2/3/4)
- Create `src/hps/` + `pyproject.toml` (`pip install -e .`).
- Move shared primitives out of the CLI scripts into `hps/` modules (see migration table below). De-duplicate the Lorentz math (one copy in `hps/geometry.py`).
- Collapse `config.py` + hardcoded constants into one `hps/config.py` dataclass.
- **Leave each old script in root, importing from `hps`, so it still runs.** Verify green *before* moving anything. This separates "did I break the logic" (Phase 1) from "did I break a path" (Phase 2).

### Phase 2 — Folder the drivers (`git mv`, preserves history)
- Move drivers into `experiments/{retrieval,detection,methods,data}/` and painters into `figures/`.
- Rewrite the now-relocated cross-imports. **Bounded: exactly 14 cross-module import sites** (mapped below).

### Phase 3 — Inputs/outputs split + fix broken scripts
- Sort `results/` into `caches/ metrics/ figures/ logs/`; move inputs to `data/`; archive stale figure dirs (`figures/`, `plots/`, `results/figs/`, `results/adaptive_attacks/`).
- Fix the two known-broken scripts: `calibration_panel.py` (imports `delta_rel`/`graph_orc` from archived module) and `norm_controlled_eval.py` (broken `hps_core` API) — or formally retire them.
- Add `results/caches/MANIFEST.md` documenting that the `.npz` caches are regenerated via `hps/extraction.py` (they do **not** exist on disk — the biggest reproducibility risk).

### Phase 4 — Vendored-repo hygiene
- Move `obfuscated-activations/` → `third_party/`; refactor your `HPSMetric` out of their `obf_reps/metrics/__init__.py` into `src/hps/` (keep a thin shim) so the vendor copy stays pristine/updatable.

### Phase 5 — Docs & ergonomics
- Rewrite `README.md` for the current project (kill the "Sentinel Validation Suite" description).
- Annotate the contradicted numbers in `docs/history/research_journey.md` (the "phantom" p=0.082 vs verified p=0.036).
- Add a `Makefile` / `hps` console-script so experiments run as `hps retrieval ...` not `python deep/path.py`.

### Phase 6 — History rewrite ⚠ DESTRUCTIVE (final, re-confirm before running)
- `git clone --mirror` backup first.
- `git filter-repo --path obfuscated-activations/inference_time_experiments/runs --invert-paths` (+ venv, big binaries) to purge ~1.3 GB from all history. Shrinks `.git` from 107 MB to a few MB.
- **Done last, on the `restructure` branch, because it rewrites every commit SHA** — doing it earlier would force re-filtering all subsequent work and break phase checkpoints. Invalidates existing clones/forks → explicit re-confirmation required at run time.

---

## Phase 1 import-migration table (the load-bearing detail)

The 14 cross-module import sites become these moves. Exhaustive ⇒ migration is bounded.

| Shared symbol(s), today buried in a CLI script | New home | Imported by |
|---|---|---|
| `ProtoNet, train_eval, tpr_at_fpr, extract_benign` (`hierarchical_detector.py`) | `hps/protonet.py` | data_driven_hierarchy, plot_leaves, openset_detection, openset_attacks |
| `expmap0_c, lorentz_dist_c, ProtoNet, load_reps` (`hyperbolic_retrieval.py`) | `hps/retrieval.py` + `hps/geometry.py` | mmlu_taxonomy, openset_detection, openset_attacks |
| `fit_all, _distortion` (`embedding_distortion.py`) | `hps/curvature.py` | harm_taxonomy, hyperbolic_retrieval |
| `graph_orc, load_token_embeddings` (`helm_token_curvature.py`) | `hps/curvature.py` | calibration_panel, embedding_distortion |
| `assign` + `TAXONOMY/SUBTAXONOMY` (`harm_taxonomy.py`) | `hps/taxonomy.py` | label_agreement |
| Lorentz math (`hps_core.py` ⊕ duplicate in `utils.py`) | `hps/geometry.py` (one copy) | statistical_tests, radial_distribution_check, norm_controlled_eval |
| `config.py` consts ⊕ `d_proj=64` hardcodes | `hps/config.py` (one dataclass) | (everywhere) |

---

## Risk register

| Risk | Mitigation |
|---|---|
| **Missing `.npz` caches** (confirmed: 0 on disk) → can't smoke-test imports against real data | Phase 1 verification uses tiny synthetic arrays; `MANIFEST.md` records the `hps/extraction.py` regen recipe. Restructure cannot fix absent data — flagged loudly. |
| Hidden import via string/`exec`/notebook not caught by grep | Phase 1 runs each driver with `--help` to force import; the 7 `.ipynb` files are scanned too |
| `filter-repo` breaks the only copy | Full `git clone --mirror` backup of `.git` before Phase 6; explicit re-confirm at the prompt |
| `obf_reps/metrics/__init__.py` HPSMetric refactor changes results | Phase 4 keeps a thin shim importing from `hps`; diff a run's output before/after |
| Work accidentally lands on `main` | All phases on `restructure/phase-*` branches; `main` only via final approved merge |

---

## What this buys you

- **Readable in 30 seconds:** `experiments/retrieval/` *is* the paper; `experiments/detection/` is the negative half; `archive/` is the past.
- **`import hps` works** without executing a 760-line script.
- **One config, one geometry implementation** — fix a bug once.
- **A clone is ~10 MB instead of ~1.4 GB.**

---

## Effort estimate

Phase 0 ~15 min · Phase 1 the bulk (careful, verified) · Phases 2–3 mechanical, bounded · Phase 4 moderate · Phase 5 ~30 min · Phase 6 minutes (irreversible).
