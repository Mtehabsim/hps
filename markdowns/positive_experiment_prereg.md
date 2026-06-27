# Pre-registration: Does hyperbolic geometry help *hierarchical harm-taxonomy modeling* at low dimension?

**Status:** pre-registered design (write-up before running). **Companion code:** `hyperbolic_retrieval.py`.
**Relationship to the main paper:** this is a *sibling* study to the negative result in `hps_preprint.md`. It does **not** revisit binary jailbreak detection (which is linearly separable, C4 ≈ 1.0, and where no geometry can win). It tests a *different capability* — faithful, compact embedding of the harm **hierarchy** — on the exact regime the negative paper named as its own boundary (§7) but never entered.

---

## 1. Motivation — why this can be positive when binary detection cannot

The negative result rests on two defeaters, both of which this experiment is designed to **escape by construction**:

1. **Linear separability.** The harm-vs-benign *decision* is flat and linearly separable, so no curved geometry improves it. → We do **not** test a binary decision. We test *retrieval, parent-accuracy, and zero-shot placement on the harm tree* — distance/containment tasks with no linear-separability ceiling (14-way leaf typing already tops out at ~0.70, not ~1.0).
2. **Faithful ≠ discriminative.** Hyperbolic embeds distances faithfully but compresses the angular directions that separate classes, so faithful ≠ useful *for classification*. → Retrieval and tree-distance preservation **reward faithfulness directly**; here faithfulness is the objective, not a liability.

The project already measured the prerequisite and it **passed**: harm-category centroids fit hyperbolic ~10× better than Euclidean (`harm_taxonomy_llm.json` / `hps_project_reference.md:171`: H=0.017 vs E=0.184, H/floor=0.55, Spearman(rep-dist, tree-dist)=0.244 vs shuffle-null 0.077, **p=0.003**). The design doc `hierarchical_hyperbolic_design.md:48-50` *proposed mAP-of-the-taxonomy in H vs E as the go/no-go* — then the project substituted argmax classification and **never ran retrieval**. This experiment runs the test that was designed and skipped.

It also closes the clearest **fairness gap** in the negative: the hierarchical detector and curvature sweep only swept embedding dimension `d ∈ {8,16,32}` (`hierarchical_detector.py:18,146`) — the *over-parameterized* side where Euclidean is guaranteed to catch up. Hyperbolic's textbook advantage (Nickel & Kiela 2017; Sala et al. 2018) is **"match accuracy at far smaller `d`"** (e.g. 5-D hyperbolic ≈ 200-D Euclidean). We add the **`d ∈ {2,3,4,6}` frontier**.

---

## 2. Hypotheses (falsifiable, directional)

- **H1 (low-dim frontier).** As `d → 2`, the hyperbolic encoder retains **tree-distance fidelity** (lower distortion vs the taxonomy tree) and **mAP / parent-accuracy** that the isometry-matched Euclidean encoder loses. Operational: `Δ(H−E)` on mAP and parent-acc is ≥ 0 and **increases monotonically as `d` decreases** over {32,16,8,6,4,3,2}.
- **H2 (retrieval).** At the best low `d`, hyperbolic beats Euclidean on **mean average precision** for sibling/same-category retrieval and on **1-NN parent (super-category) accuracy**.
- **H3 (zero-shot novel-leaf placement).** When an entire leaf is held out of training, its examples land nearest to a **sibling leaf of the same parent** more often under the hyperbolic embedding than the Euclidean one (placeable leaves only — those whose parent has ≥2 leaves: `violence`, `cyber`, `fraud_theft`, `hate_harassment`).

**Primary endpoint:** `Δ(H−E)` on mAP@all and parent-accuracy at `d=2,3`.
**Secondary:** tree-distance distortion vs `d`; recall@k; novel-leaf parent-rank; typed/leaf accuracy (reference only).

---

## 3. Design — the fair, isometry-controlled contrast

Identical to the legitimate part of the project's own code, so any difference is *exactly* the geometry:

| Element | Choice | Rationale |
|---|---|---|
| **Architecture** | `ProtoNet` from `hierarchical_detector.py`: `Linear(d_in→d)` encoder → (H) `expmap0` to Lorentz hyperboloid + `lorentz_dist` to learned prototypes, or (E) plain `cdist`. | H and E differ **only** in the distance metric — the rpf-vs-rpfnox0 fairness principle at the detector level. Curvature is *load-bearing* here (acosh is nonlinear, prototypes live on the manifold), unlike the rpf `x0` tautology. |
| **Training objective** | Cross-entropy on leaf classification (+ optional tree regularizer β on prototype distances). Same loop as `train_eval`. | Keeps the embedding space comparable to the project's detector; retrieval is evaluated *on the resulting space*. `--loss contrastive` offered as a robustness variant. |
| **Data** | The 650 LLM-labeled harm reps (`harm_taxonomy_llm.json`, 9 categories / 14 leaves), regenerated as `harm_taxonomy_llm_reps.npz` at **layer 24**. Benign is **excluded** from retrieval (it has no position in the harm tree). | Retrieval/entailment is *within* the harm hierarchy. |
| **Independent variable** | `d ∈ {2, 3, 4, 6, 8, 16, 32}`. | The frontier the project never tested. |
| **Standardization** | mu/sd fit on **train rows only** (per split). | Fixes the train+test standardization leak present in `hierarchical_detector.py:162`, `curvature_sweep.py:90`, `data_driven_hierarchy.py:100` — small but a clean experiment shouldn't inherit it. |
| **Seeds / CIs** | ≥ 5 seeds; report mean ± std and `Δ(H−E)` with its std. | Matches the project's existing rigor (σ-reporting). |
| **References** | C4 / Euclidean shown for context; the *competitor for H is E* (not C4 — C4 cannot do retrieval). | Avoids the category error of pitting a retrieval embedding against a binary linear probe. |

---

## 4. Metrics (all in `hyperbolic_retrieval.py`)

- **mAP@all** — for each held-out query, rank all other held-out points by manifold distance; relevant = same leaf (and a same-category variant). Mean average precision.
- **recall@k** (k=1,5,10).
- **1-NN leaf accuracy** and **1-NN parent (super-category) accuracy** — geometry-faithful, no centroid estimation.
- **tree-distance distortion** — `fit_all`-style: distortion of the embedding's pairwise distances vs the taxonomy tree distances (0/2/4), reported per `d`.
- **novel-leaf parent placement accuracy** — hold out each placeable leaf, train, then check whether held-out examples' nearest *train* neighbor shares the true parent. Random baseline = 1/(#categories with the held-out point's structure).

**Geometry knob check:** also sweep curvature `c` at the best `d` (reuse `curvature_sweep` style) to confirm whether retrieval peaks at `c>0` (genuine hyperbolic) or `c→0` (would reproduce the negative).

---

## 5. Falsification criteria (decided in advance)

| Outcome | Interpretation |
|---|---|
| `Δ(H−E)` mAP & parent-acc **> 0 and grows as `d↓`**, surviving 5-seed CIs | **POSITIVE** — hyperbolic helps hierarchical harm modeling at low `d`. Publishable as the sibling capability. |
| H wins on **tree-distance distortion** but **ties on mAP/parent-acc** | Partial — "more faithful container, no task payoff." Confirms `faithful ≠ discriminative` *extends* to retrieval. Honest, weaker. |
| H ≈ E at all `d`, or both peak at high `d` / `c→0` | **NEGATIVE extends** to hierarchical tasks — also publishable; strengthens the main paper's boundary claim. |
| H wins only at `d=2` and only on 1-2 rarest leaves | Sub-regime caveat, not a headline. Report as such. |

No outcome is unpublishable; the framing is fixed in advance to prevent post-hoc spin.

---

## 6. Known risk — taxonomy depth is the binding constraint

The existing taxonomy is **2-level / 14-leaf** with weak alignment (ρ=0.244 ≈ 6% of variance). Two-level trees are exactly where hyperbolic's advantage is smallest, and sibling (leaf) retrieval needs *angular* resolution that curvature compresses — so leaf-mAP may track `typed_acc` and peak at `c→0`, reproducing the negative. **Mitigations, in order of preference:**

1. **Build a deeper (3-4 level) taxonomy** from the 650 prompts (sub-sub-categories, e.g. `cyber/intrusion/{credential-theft, network-breach, privilege-escalation}`), via the existing `harm_taxonomy.py label` LLM-labeling path with a deeper option list. Depth is what gives hyperbolic room to win. **This is step 0 for a convincing result.**
2. Emphasize **parent-accuracy and zero-shot novel-leaf** (super-category level), where coarse tree structure dominates and angular compression matters less — the most likely place for a genuine positive even on the shallow tree.
3. Treat **tree-distance distortion** as the safest endpoint (it's what the green-light gate already favored H on).

Most-likely true outcome, stated honestly: a clean **tree-distance / parent-accuracy** win at low `d` and a **leaf-typed tie** — a genuine but *narrow* positive ("hyperbolic embeds the harm hierarchy more faithfully and far more compactly"), not "hyperbolic improves jailbreak detection."

---

## 7. Run plan (see §8 of the chat / README for DGX specifics)

1. (optional, GPU) deepen taxonomy → `harm_taxonomy_deep_llm.json`
2. (GPU) extract reps at layer 24 → `results/harm_taxonomy_llm_reps.npz`
3. (CPU/GPU) `hyperbolic_retrieval.py` sweep over `d` and geometry → `results/hyperbolic_retrieval.json` + `.png`
4. (CPU) read the JSON; apply the §5 decision table; write up under the §5 framing.

**Provenance note:** as of this writing there is **no activation cache on disk** (`results/` has zero `.npz`), so steps 1-2 must run on the DGX before step 3 sees real data. Step 3 has a `--selftest` mode that validates the full metric pipeline on synthetic tree-structured data with no GPU.
