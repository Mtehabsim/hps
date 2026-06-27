# HPS / Hyperbolic Jailbreak-Detection — Project Reference

**Model:** Llama-3-8B-Instruct. **Goal:** determine whether hyperbolic geometry on hidden
states helps jailbreak *detection* and *adaptive robustness*, vs a linear baseline.

---

## 1. Glossary (architectures, attacks, metrics, geometry)

### Probes / detectors

| Term | What it is | Key config | Notes |
|---|---|---|---|
| **C4** | `LogisticRegressionMetric`. Linear probe baseline. **Independent logistic-regression probe per layer** (late fusion), scores combined. | layers `[1,7,13,19,25,32]`, Euclidean | Strong clean detector; linear decision direction. |
| **HPS-12** (`HPSMetric`, `feature_set=all`) | Hyperbolic Projection Sentinel. Projects activations onto a **Lorentz hyperboloid** (proj_dim 64), computes **12 cross-layer trajectory features** (radial/curvature/displacement), scores with LR. | `proj_dim=64, kappa=0.1, layers[1,7,13,19,25,32]` | The "main" HPS. Weak clean detector (the 12 features are the bottleneck). |
| **raw_proj** | HPS variant: keep the 64-dim Lorentz projection, **drop the 12 features**; mean-pool projected points over layers. | `feature_set=raw_proj` | Isolates the compression step. Clean recall ≈ C4. |
| **rpf** (`raw_proj_full`) | Lift the **full 4096-dim** activation onto the hyperboloid (adds the `x0` Lorentz radial coordinate), **mean-pool over the 6 layers** (early fusion), single LR. | `feature_set=raw_proj_full` | "Purest hyperbolic-on-hidden-states" probe. |
| **rpfnox0** (`raw_proj_full_nox0`) | **Identical to rpf but WITHOUT the `x0` hyperbolic lift** → Euclidean. Mean-pool 4096-dim over layers, single LR. | `feature_set=raw_proj_full_nox0` | **The control:** rpf vs rpfnox0 isolates *exactly* the hyperbolic lift. |
| **mean_r / radial / radial_disp_8** | Subsets of the 12 HPS trajectory features (e.g. `mean_r` = mean radial coordinate only). | `feature_set=...` | Ablations: `mean_r` alone ≈ all 12 on clean detection. |

**Layer-fusion difference (important):** C4 = per-layer probes (late fusion); rpf/rpfnox0 = mean-pool layers then one probe (early fusion). This is the *only* architectural difference between C4 and rpfnox0 — both are **Euclidean**. rpf adds the hyperbolic `x0` on top of rpfnox0.

### Attacks (from Bailey et al., `obf_reps`)

| Term | What it is |
|---|---|
| **Soft-prompt attack** | Continuous/embedding attack (`Llama3SoftPrompted`, `AdamContinuousOptimizer`). Optimizes a continuous prompt to jailbreak **and** evade the probe. Strong/omnipotent threat model. |
| **GCG (hard-token)** | Discrete token-search attack (`GCGOptimizer`, search_width 512, topk 256). Realistic threat model (attacker controls tokens, not embeddings). Much weaker. |
| **λ (lambda)** | Attack loss weight: `total = λ·behavior_loss + (1−λ)·obfuscation_loss`. **λ = weight on jailbreak; (1−λ) = weight on evading the probe.** Low λ = max evasion pressure. |
| **behavior_loss** | Cross-entropy that makes the model emit the harmful output (drives jailbreak). |
| **obfuscation_loss** | The probe's own score on the attacked reps (`ProbePredictLoss` uses `metric.predict`). Minimizing it → probe says "benign" → evasion. |
| **attacker / defender innings** | Attacker optimizes vs the current probe; defender then retrains the probe on the attack (adversarial training round). |

### Metrics

| Term | Meaning |
|---|---|
| **AUROC** | Ranking quality of harmful-vs-benign scores. `standard_auc` = clean; `obfuscated_auc` (obf_auc) = on attacked inputs. ~1 detect, ~0.5 chance, ~0 fully inverted (harmful look *more* benign than benign). |
| **TPR @1%FPR** (`recall@1%FPR`) | **True-positive rate at 1% false-positive rate** = fraction of harmful caught when only 1% of benign are falsely flagged. The **operational** metric. `standard_recall` = clean; `obfuscated_recall` = attacked. |
| **jailbreak_behavior** | StrongREJECT-style score: fraction of attacks that actually produced harmful output (did the attack succeed at jailbreaking, independent of the probe). |

### Geometry / diagnosis

| Term | Meaning |
|---|---|
| **Hyperbolic space / Lorentz model / hyperboloid** | Negatively-curved space; volume grows exponentially with radius → embeds **trees/hierarchies** with low distortion. |
| **κ (kappa)** | Curvature parameter of the HPS hyperboloid (0.1). |
| **δ-hyperbolicity** | Gromov 4-point measure of tree-likeness. Low = tree-like. **Unreliable at LLM dimension** (high-D concentration). |
| **Ollivier-Ricci curvature (ORC)** | Discrete curvature on a kNN graph. **Failed calibration** (a sphere scored most "hyperbolic") → not used for verdicts. |
| **Embedding distortion (H/E/S)** | Fit the data's metric into Hyperbolic / Euclidean / Spherical space; lowest distortion = best-fit geometry. **Validated on known shapes (tree→H, sphere→S, flat→E).** The trustworthy geometry test. |
| **Dimension-matched random baseline / floor** | A random Gaussian of the same dimension through the *same* pipeline. Mandatory: high-D pipelines make anything look hyperbolic, so only the **margin below the floor** counts. |
| **Gradient masking / vanishing** | A defense looks robust because the attacker's gradient is ≈0 (e.g. a saturated sigmoid: slope `p(1−p)→0` when confident), not because it's secure. **False robustness** — defeated by gradient-free/transfer/BPDA attacks. |
| **logit / sigmoid / softplus** | logit = raw pre-squash score. `sigmoid` saturates (masks gradients); `softplus` is non-saturating (slope = `sigmoid`, never 0) and monotonic (detection metrics unchanged). |
| **Causal separability** | The attacker can change the probe's feature without changing the harmful behavior → why activation probes (any geometry) are evadable. |
| **Harm taxonomy / hierarchical-hyperbolic** | Reframe harm as a tree (category→subcategory→behavior); the only setting where hyperbolic geometry has a legitimate role. |

---

## 2. Clean (non-adaptive) detection — train/test

**Dataset:** `harmful_dataset` (`HIDHarmfulConceptDataModule`, 800 harmful behaviors + matched benign); obfuscation data `JailbreakCircuitBreakersObfusDataModule`. Evaluation set per run: **n_harmful=200, n_benign=200**. (Geometry analyses additionally use the JBShield Llama-3 activation cache: ~4.7k/5.2k train ben/atk, 1.2k/1.3k test, last-token per layer, 32 layers.)

Clean detection = `standard_auc` / `standard_recall@1%FPR` (harmful vs benign, **no attack**):

| Probe | Clean AUROC | Clean TPR@1%FPR | Comment |
|---|---|---|---|
| **C4** | ~1.000 | ~0.99–1.00 | strong linear detector |
| **HPS-12** | ~0.92 | ~0.35 (range 0.27–0.60) | **weak** — the 12 trajectory features hurt clean detection |
| **rpf** | 1.000 | 1.000 | recovers C4-level clean detection |
| **rpfnox0** | 1.000 | ~1.000 | = rpf (hyperbolic lift makes no difference) |

**Takeaway:** dropping HPS's 12 trajectory features (→ rpf/rpfnox0) recovers C4-level clean detection; the hyperbolic lift (rpf vs rpfnox0) changes nothing.

---

## 3. Adaptive-attack results (soft-prompt λ-sweep, attacker inning)

Lower `obf_*` = more evaded. **TPR@1%FPR is the operational metric.**

### C4 (linear)
| λ | clean AUROC | clean TPR | obf AUROC | **obf TPR@1%FPR** | jailbreak |
|---|---|---|---|---|---|
| 0.0 | 1.000 | 1.000 | 0.000 | 0.000 | 0.004 |
| 0.1 | 1.000 | 0.995 | 0.000 | 0.000 | 0.833 |
| 0.3 | 1.000 | 1.000 | 0.000 | 0.000 | 0.868 |
| 0.5 | 1.000 | 1.000 | 0.011 | 0.000 | 0.869 |
| 0.7 | 1.000 | 0.985 | 0.015 | 0.000 | 0.850 |
| 0.9 | 1.000 | 1.000 | 0.070 | 0.000 | 0.862 |
| 1.0 | 1.000 | 0.995 | 0.975 | 0.330 | 0.865 |

### HPS-12 (hyperbolic, 12 features)
| λ | clean AUROC | clean TPR | obf AUROC | **obf TPR@1%FPR** | jailbreak |
|---|---|---|---|---|---|
| 0.0 | 0.926 | 0.560 | 1.000 | 1.000 | 0.014 |
| 0.1 | 0.918 | 0.350 | **1.000** | **1.000** | 0.871 |
| 0.3 | 0.923 | 0.410 | 0.002 | 0.000 | 0.853 |
| 0.5 | 0.929 | 0.600 | 0.001 | 0.000 | 0.869 |
| 0.7 | 0.928 | 0.390 | 0.009 | 0.000 | 0.863 |
| 0.9 | 0.907 | 0.265 | 0.033 | 0.000 | 0.846 |
| 1.0 | 0.936 | 0.560 | 0.086 | 0.000 | 0.860 |

> The λ=0.1 "resistance" (obf AUROC/TPR = 1.0) is **gradient masking** (sigmoid saturation), **not robustness** — see §4 Q2.

### rpf (full 4096 hyperbolic)
| λ | clean AUROC | clean TPR | obf AUROC | **obf TPR@1%FPR** | jailbreak |
|---|---|---|---|---|---|
| 0.0 | 1.000 | 1.000 | 0.000 | 0.000 | 0.003 |
| 0.1 | 1.000 | 1.000 | 0.344 | 0.005 | 0.837 |
| 0.3 | 1.000 | 1.000 | 0.465 | 0.000 | 0.873 |
| 0.5 | 1.000 | 1.000 | 0.825 | 0.055 | 0.859 |
| 0.7 | 1.000 | 1.000 | 0.682 | 0.010 | 0.873 |
| 0.9 | 1.000 | 1.000 | 0.916 | 0.100 | 0.868 |
| 1.0 | 1.000 | 1.000 | 0.985 | 0.645 | 0.865 |

### rpfnox0 (Euclidean twin of rpf — control)
| λ | obf AUROC | **obf TPR@1%FPR** | jailbreak |
|---|---|---|---|
| 0.0 | 0.000 | 0.000 | 0.006 |
| 0.1 | 0.576 | 0.000 | 0.828 |
| 0.3 | 0.842 | 0.045 | 0.860 |
| 0.5 | 0.545 | 0.015 | 0.868 |
| 0.7 | 0.869 | 0.040 | 0.843 |
| 0.9 | 0.872 | 0.085 | 0.863 |
| 1.0 | 0.984 | 0.620 | 0.839 |
(clean AUROC/TPR ≈ 1.000 across λ)

**Reading of the adaptive results:**
- By **TPR@1%FPR** (operational): **all probes are bypassed** at the λ an attacker would pick (0.1–0.9). C4 ≈ 0; rpf/rpfnox0 ≈ 0–0.10 (marginally less bypassed).
- The small rpf/rpfnox0 edge over C4 is **architectural** (mean-pool early fusion), **not hyperbolic** (rpf ≈ rpfnox0).
- rpf's higher **obf AUROC** (0.34–0.98) vs C4 (~0) is a *ranking* artifact (C4's single linear score fully inverts; rpf's high-dim score doesn't) + likely partial gradient masking — it does **not** translate to TPR.

### Hard-token (GCG, 500 steps, single behavior `example_idx=0`)
| Probe | clean AUROC | clean TPR | obf AUROC | obf TPR@1%FPR | jailbreak |
|---|---|---|---|---|---|
| C4 | 0.9999 | 1.000 | 0.9996 | 0.985 | 0.008 |
| HPS-12 | 0.927 | 0.380 | 0.606 | 0.010 | 0.005 |

> **Caveat:** GCG did **not** jailbreak (jailbreak ≈ 0.005–0.008) → not yet a valid adaptive-evasion test. rpf/rpfnox0 GCG + multi-behavior runs pending.

---

## 4. Questions — answers & validation plans

### Post-meeting deliverables — status
| # | Meeting item | Status | Where |
|---|---|---|---|
| 1 | Hard-token (GCG) adaptive attacks | **PARTIAL — only open experiment.** Infra fixed; C4 ≫ HPS-12 on detection; but GCG didn't jailbreak; rpf/rpfnox0 + multi-behavior unrun | Q4, §3 |
| 2 | Explain the λ=0.1 HPS-resisted result | ✅ **DONE** — gradient masking, causally confirmed (softplus) | Q2 |
| 3 | Visualizations | ✅ **mostly done** — calibration, distortion, λ-sweep, masking, obfuscation-PCA, leaves, curvature sweep; only `OBF_DUMP` arrow viz optional-pending | §5 |
| 4 | HELM curvature ("why" + can hyperbolic be effective) | ✅ **comprehensively done** — token subspace hyperbolic; harm decision linear; robust negative across hierarchy + curvature + model | Q3, Q5/b/c/d |

**Net: 3 of 4 fully/mostly answered; #1 (GCG real adaptive comparison) is the remaining experiment.** Conclusion does not hinge on #1.

**Q1. Does hyperbolic geometry help jailbreak detection / robustness?**
**Confident: No.** Functional controls: `rpf ≡ rpfnox0` at every λ (the hyperbolic lift is inert); HPS-12 ≤ C4 (weaker clean, equally bypassed). The small rpf-over-C4 edge is architectural (shared by the Euclidean twin).

**Q2. Why did HPS "resist" at λ=0.1?**
**CONFIRMED (causal): gradient masking from sigmoid saturation.** With the original sigmoid, `obf_loss` was pinned at exactly 1.0 for all 800 steps (saturated at logit ≈ +50 → slope `p(1−p)≈0`) → obf_auc=1.000 (not evaded). Re-running with `obf_signal=softplus` (non-saturating, monotonic → detection unchanged): `obf_loss` descends to ~2.7e-10 and **obf_auc drops 1.000 → 0.001 (HPS fully evaded at λ=0.1)**, clean AUROC unchanged (0.918 → 0.909). Interventional proof: removing the saturation removes the "resistance." *Not* robustness, *not* hyperbolic — HPS has no advantage over C4 at any λ.
**Remaining cross-check (optional):** gradient-free GCG at λ=0.1 should also evade HPS.

**Q3. Are the hidden states hyperbolic?**
**Confident, nuanced:** **Token/vocabulary subspace = decisively hyperbolic** (embedding distortion H = 0.003 = 0.02× the random floor; confirms HELM, via a validated method). **Harm-decision reps = only weakly hyperbolic** (H ≈ 0.53× floor) and the **harm decision is linear** (C4 ≈ 1.0). So hyperbolicity lives in language structure, not the harm direction.
**Validation done:** embedding-distortion test passes the known-geometry calibration gate; δ and ORC do **not** (high-D artifacts) — only baseline-relative, calibration-passing measures are trusted.
**Faithful ≠ discriminative (reconciles "hyperbolic fits best" vs "doesn't help"):** the per-layer "hyperbolic = lowest distortion" measures *distance-preservation* (faithful embedding of the whole manifold), **not class separability**. Hyperbolic preserves the gross/radial manifold shape but its exponential metric *compresses the angular directions that separate categories* — so a faithful hyperbolic embedding still classifies no better, and at high curvature worse (curvature sweep: Euclidean best; see Q5d). A low-distortion container ≠ a discriminative coordinate system.

**Q4. Hard-token (GCG) adaptive comparison?**
**PARTIAL — the one open experiment.** Infrastructure fixed (OOM was a missing `no_grad`); C4 and HPS-12 ran (§3 — on detection **C4 ≫ HPS-12**). **But GCG did not jailbreak** (jailbreak ≈ 0.005–0.008, single `example_idx=0`) → not yet a valid adaptive-evasion test. rpf/rpfnox0 GCG configs built but unrun.
**To close:** run `c4 / rpf / rpfnox0` GCG looping several `example_idx` (real jailbreaks) at 500 steps; aggregate with `attack_cost_curve.py`; report TPR/AUROC + the gradient-masking cross-check. Expected (consistent with soft-prompt): hard-token weak, **rpf ≈ rpfnox0**, no hyperbolic difference.

**Q5. Does the hierarchical-hyperbolic (harm-taxonomy) idea have traction?**
**GREEN LIGHT (clean-label gate, layer 24, 650 behaviors).** Harm-category centroids: **H=0.017 vs E=0.184 (~10× better hyperbolic fit), H/floor=0.55** (below the random floor), and **taxonomy alignment Spearman=0.244 vs shuffle-null 0.077, p=0.003** (3σ). So the harm *category hierarchy* is genuinely hyperbolic + tree-aligned — the structure a hierarchical-hyperbolic detector needs **exists**. (First-pass keyword labels gave borderline 0.67 / p=0.057; LLM labels strengthened both, as predicted.)
**Caveat — what's NOT yet shown:** that a hyperbolic *detector* beats Euclidean or raises attack cost. Next: build the detector **with a Euclidean-taxonomy control** (isolate the hyperbolic contribution; note E is also below its floor → some structure is generic) and a **gradient-masking-safe** adaptive eval. Effect sizes are moderate; labels are from an imperfect 8B classifier; taxonomy is coarse (2-level, 14 leaves).

**Q5b. Stage-1 detector test (layer 24, 14 leaves+benign, 3 embed_dims × 5 seeds):**
**Negative — no net hyperbolic advantage.** typed_acc: Euclidean 0.735 > hyperbolic 0.701 (Δ=−0.034); binary TPR@1%FPR: **C4 linear 0.987 > hyperbolic 0.929 > Euclidean-proto 0.241**. So hyperbolic loses the hierarchical task to Euclidean and loses binary to the linear baseline. *Nuance:* hyperbolic geometry is **not inert** here (unlike rpf≡rpfnox0) — it produces a **tighter, cleanly-bimodal harm-vs-benign score separation** (minimal overlap), which markedly beats the Euclidean-proto on binary (whose benign score-tail overlaps harmful → TPR@1%FPR collapses). Diagnostic viz: separation is **angular** (2D-PCA left/right lobes) in both geometries; **embedding radius does NOT separate** benign/harm in either (the earlier radial-severity hypothesis was refuted by the plots). So hyperbolic helps *score tightness*, not via a radius axis — but still loses binary to linear C4 and typed to Euclidean. Real geometric effect, **no practical payoff** (baselines win). → Do not pursue the hierarchical-hyperbolic detector for performance; Stage 2 (adaptive) only as a masking-safe confirmation.

**Q5c. Robustness to hierarchy choice (β=0 ablation + data-driven hierarchy).**
**Negative is ROBUST.** (i) `β=0` (remove imposed tree regularizer) → unchanged (H still loses typed to E, binary to C4) → the imposed tree wasn't the cause. (ii) **Data-driven hierarchy** (cluster 650 harm + 800 benign into 30 data clusters — multi-benign + multi-harm, purity 0.99 — dendrogram tree): the **single-benign-blob was indeed the flaw** (Euclidean-proto binary TPR 0.24→**0.984** once benign is multi-cluster), **but fixing it equalized H and E** (binary TPR H=0.989 ≈ E=0.984, Δ=+0.004) with both **< C4 0.996**. So a deeper/fairer/data-optimal hierarchy makes the *Euclidean baseline catch up* — it does **not** make hyperbolic win. **Conclusion robust to hierarchy choice:** hyperbolic provides no advantage even with the data's own hierarchy; harm-vs-benign is linearly separable (~0.99 for all), so no geometry beats a linear probe.

**Q5d. Robustness to hyperbolic configuration (curvature sweep).**
**Negative robust to curvature/projection.** Curvature-c Lorentz (c→0 = Euclidean → strongly hyperbolic), typed_acc by c: **0.05→0.753, 0.25→0.739, 0.5→0.726, 1→0.702, 2→0.704, 4→0.645, 8→0.548** — **best at c→0 (Euclidean), monotonically WORSE with more curvature.** No curvature helps the hierarchical task; strong curvature actively hurts (compresses angular/category resolution). Binary: C4 linear (0.987) not beaten at any c (apparent c=0.5 peak 0.963 is the single-benign-proto helped by mild curvature, still < C4, settled by Q5c). Model choice (Lorentz vs Poincaré/Klein) is isometric → irrelevant. So the negative holds across **model + curvature** too — the limitation is the (linearly-separable) task, not the hyperbolic flavor. (`curvature_sweep.py`)

**Q6. Can hyperbolic raise the adaptive-attack budget?**
**Confident for binary task: No** (rpf ≡ rpfnox0 across the whole cost curve). For a taxonomy: **unknown**, gated on Q5 + a gradient-free check (the exponential-metric "cost" must not be gradient masking).

**Q7. Is the small rpf/rpfnox0 edge over C4 hyperbolic?**
**Confident: No — architectural.** rpfnox0 (Euclidean) matches rpf; the edge comes from mean-pool early fusion vs C4's per-layer probes. Both remain operationally bypassed.

**Q8. Can activation-probe adaptive robustness be "solved"?**
**Confident framing:** not by probe geometry. The limiter is causal separability + threat model. Levers that actually move it (all known in literature): output-level monitoring, attack-surface constraints (token-only), randomized/ensemble probes, causally-necessary features. Hyperbolic is the wrong axis.

---

## 5. Visualizations (what + purpose)

| # | Visualization | Purpose / finding it supports | Status |
|---|---|---|---|
| V1 | **Known-geometry calibration panel** (tree/sphere/flat/Gaussian at matched dim) for δ, ORC, embedding-distortion | Show δ/ORC are unreliable at high dim (invert known geometries); embedding-distortion passes → justifies the methodology | done (`calibration_panel.py`) |
| V2 | **Embedding distortion H/E/S** for token embeddings vs harm reps vs random floor | "Where hyperbolicity lives": token subspace decisively hyperbolic; harm reps only weakly; harm decision linear | done (`embedding_distortion.py`) |
| V3 | **Per-layer distortion (H/E/S vs layer)** | The hyperbolic→flat transition across depth; which layer (if any) is most hyperbolic | done |
| V4 | **λ-sweep curves: obf AUROC & TPR@1%FPR vs λ** for C4/HPS-12/rpf/rpfnox0 (overlaid) | Adaptive robustness; rpf≈rpfnox0 (hyperbolic inert); all bypassed by TPR | data ready, plot to assemble |
| V5 | **Attacker-cost curve** (evasion vs budget) + gradient-free (GCG) cross-check | Does any probe raise the attack budget; is "resistance" real or masking | partial (soft done; GCG pending) |
| V6 | **obf_loss trajectory: HPS vs C4 at λ=0.1** | Smoking gun for gradient masking (HPS pinned at 1.0, C4 descends) | data in logs, plot to assemble |
| V7 | **PCA of harmful vs benign (clean) + obfuscated/attacked harmful** | Visualize obfuscation: clean harm/benign cleanly separated; attack **drags harmful off its cluster into the probe-benign region** → overlap is *adversarially induced*, not natural/partitionable | **done** (`plot_obfuscation.py` → `obfuscation_pca.png`); `OBF_DUMP_ACTS` arrow version optional-pending |
| V8 | **Taxonomy gate plot** (category-centroid distortion H/E/S + rep↔tree correlation vs shuffle null) | Whether harm categories are hyperbolic + taxonomy-aligned → go/no-go on hierarchical detector | **done — GREEN (H/floor 0.55, p=0.003)** |
| V9 | **Softplus vs sigmoid obf_loss & obf_auc at λ=0.1** | Confirms the λ=0.1 resistance was masking: obf_loss 1.0→~0, obf_auc 1.000→0.001 once non-saturating | **done** (`hps_lam0.1_softplus.log`) |
| V10 | **Harm-taxonomy leaves** (PCA + t-SNE + learned hyperbolic Poincaré disk, colored by category, benign overlay) | Where the 14 harm categories live; sibling adjacency = real hierarchy; Poincaré = the learned tree layout | **done** (`plot_leaves.py` → `leaves.png`) |
| V11 | **Curvature sweep** (typed_acc & binary TPR vs curvature c; c→0 = Euclidean, + C4 line) | Robustness to curvature/projection: typed_acc best at c→0 (Euclidean), monotonically worse with curvature; no c beats C4 | **done** (`curvature_sweep.py` → `curv_sweep.png/.json`) |

---

## 6. Implications & applications (reframing the negative as a contribution)

**Organizing principle:** hyperbolic geometry helps when the **task objective itself is a hierarchy** (tree/taxonomy/abstraction), not when it's a flat decision. Match geometry to *task structure*, not just to the data's measured curvature.

**Where hyperbolic genuinely pays off (consistent with our findings):**
- **Hierarchical/taxonomic embedding & retrieval** — knowledge graphs, ontologies (WordNet, ICD codes, product/biological taxonomies), entailment/"is-a", hierarchical search/recommendation (the classic Nickel–Kiela sweet spot).
- **LLM architecture for language modeling (HELM)** — our validated result (token subspace 0.02× the random floor) is the green light for hyperbolic components *inside* the model.
- **Deep hierarchical *classification*** with many-level label trees (ours failed only because it was 2-level/flat) — parameter-efficient.
- **Compact low-dim embeddings of hierarchies** — edge/compression/visualization (this is exactly the low-distortion "faithful container" property we measured).
- **Interpretability** — hyperbolic probes/Poincaré viz to reveal a model's *concept hierarchy* (structure, not a decision).
- **(Speculative)** abstraction-level control via radius (radius ≈ generality) for summarization / granularity steering.

**Transferable methodology contributions (any representation-geometry study — vision, graph ML, neuroscience):**
- **Baseline-correct all curvature statistics** — δ-hyperbolicity, Ollivier-Ricci, and kNN-geodesic embedding-distortion *all read "hyperbolic" for random high-D data*; require a dimension-matched random null through the identical pipeline.
- **Faithful ≠ discriminative** — low embedding distortion does not imply usefulness for a classification task; choose geometry by task structure.

**Security-specific takeaway (the useful negative):** don't build geometric probes for harm detection — the harm decision is **linearly separable**, so a linear probe is the right tool; adaptive robustness is governed by **causal separability + threat model** (output monitoring, attack-surface constraints, randomized/ensemble probes, causally-necessary features), not probe geometry.

---

## References / provenance
- Bailey et al., *Obfuscated Activations* — attack framework (`obf_reps`), attacker/defender game, StrongREJECT judge.
- HELM (arXiv:2505.24722), *Hyperbolic LLMs via Mixture-of-Curvature Experts* — token-subspace hyperbolicity (we reproduce via validated embedding distortion).
- Athalye, Carlini, Wagner (2018), *Obfuscated Gradients Give a False Sense of Security* — gradient masking.
- Sala et al. (2018), Gu et al. (2019), Nickel & Kiela (2017) — constant-curvature embedding distortion.
- All numbers from runs in `obfuscated-activations/inference_time_experiments/runs/` and `results/`; model Llama-3-8B-Instruct.
</content>
