# Plan A — Strengthen the Paper Before Submission

**Goal:** Address the AI reviewer's most critical gaps before submitting. Result: paper is mechanistically sound, statistically rigorous, and defensible against the standard reviewer critiques.

**Estimated time:** 2-3 weeks

**Status as of 2026-05-27:**
- [x] Statistical tests (HPS vs C4) — done, p=0.082 confirmed
- [x] Radial distribution multi-config check — done, 13/13 confirmed
- [x] Vicuna failure diagnostic — done, alignment-correlation observed
- [x] Hyperbolic methods comparison (Experiment 1) — done, key insight: Hyperbolic MLP works on Vicuna
- [ ] Vicuna sample size fix (this plan)
- [ ] Pre-training radial distribution control (this plan)
- [ ] Base vs Instruct alignment ablation (this plan)
- [ ] Briefing/outline updates (this plan)

---

## Phase 1 — Use JBShield's Public Vicuna Data (1-2 days)

**Goal:** Fix the small-sample (n=16 GCG) issue without generating new attacks.

### Step 1.1 — Verify dataset availability (5 min)

```bash
# On the DGX:
git clone https://github.com/NISPLab/JBShield /tmp/jbshield
ls /tmp/jbshield/dataset/
ls /tmp/jbshield/dataset/vicuna-13b-v1.5/ 2>/dev/null
```

**Expected:** Per-LLM directory with 9 attack JSON files including `gcg.json`. JBShield's full dataset has 850 GCG attacks per LLM.

**Decision point:**
- If Vicuna-13B GCG data exists in their dataset → proceed to Step 1.2
- If only Vicuna-7B → use that as a third LLM data point instead of Vicuna-13B
- If not available → fall back to Phase 2 alternative

### Step 1.2 — Extract activations on Vicuna-13B (1-2 hours compute)

Adapt `cross_model_compare.py --extract` to point at JBShield's GCG prompt list:

```python
# Pseudocode
jbshield_gcg = json.load(open("jbshield/dataset/vicuna-13b-v1.5/gcg.json"))
prompts = jbshield_gcg["prompts"]  # ~850 GCG attacks
# For each prompt: forward pass through Vicuna-13B, capture activations at layers [0,2,22,31,35,39]
# Append to existing vicuna_activations_cache.npz
```

**Output:** `results/vicuna_activations_cache_v2.npz` with ~850 GCG samples instead of ~80.

### Step 1.3 — Re-run Vicuna analyses with larger N (1 day)

```bash
python vicuna_imbalance_test.py --vicuna_cache results/vicuna_activations_cache_v2.npz
python gcg_specific_test.py --vicuna_cache results/vicuna_activations_cache_v2.npz
python vicuna_diagnostic.py --vicuna_cache results/vicuna_activations_cache_v2.npz
```

**Expected outcome:** GCG test set goes from 16 → ~170 samples. Confidence intervals tighten substantially.

**Decision point:**
- If alignment-mediated finding holds with n=170 → proceed to Phase 2
- If it disappears (HPS catches GCG fine on Vicuna) → reframe paper, drop alignment-mediation claim

---

## Phase 2 — Pre-training Radial Distribution Control (half-day)

**Goal:** Address reviewer #16: confirm the radial inversion is a TRAINED phenomenon, not an inherent property.

### Step 2.1 — Modify radial_distribution_check.py

Add an "epoch_0" condition that captures the radial distribution at random initialization (BEFORE any training):

```python
# In radial_distribution_check.py:
def measure_radial_pretraining(X_te_ben, X_te_atk, n_layers, d_hidden,
                                kappa_init, device):
    """Random init projection — no training."""
    proj = LorentzProjection(d_hidden, 64, kappa_init, n_layers).to(device)
    proj.log_k.requires_grad = False
    return compute_radial_for_state(
        np.concatenate([X_te_ben, X_te_atk]), proj.state_dict(),
        d_hidden, n_layers, kappa_init, device,
    )
```

### Step 2.2 — Run and compare

Compare:
- Pre-training (random init): expected to show no clear separation
- Post-training (learned): observed inversion (benign higher, attack lower)

**Expected outcome:** Pre-training shows random radial distribution. Post-training shows the inversion. **This proves the inversion is learned, not inherent**, addressing reviewer #16.

### Step 2.3 — Add to research_journey.md

Add 1 paragraph + table to Section 3 (radial distribution) showing the pre/post comparison.

---

## Phase 3 — Base Llama-3 vs Llama-3-Instruct Ablation (1-2 weeks)

**Goal:** Address reviewer #17 and #21: directly test the alignment hypothesis with same architecture, only alignment differs.

### Step 3.1 — Determine prompt format strategy (half-day decision + 1 day testing)

Base Llama-3 has no chat template. Two options:

**Option α (recommended):** Use Llama-3-Instruct's chat template even for base model
- Pro: Same activation extraction protocol → fair comparison
- Pro: Both models receive identical prompts
- Con: Base model wasn't trained with this template, may produce different activation patterns

**Option β:** Use raw text (no template) for base, chat template for Instruct
- Pro: Each model uses its "natural" format
- Con: Different prompt structure → different activations → confounded comparison

**Decision rule:** Run both as supplementary; report Option α as the main result with Option β as an ablation.

### Step 3.2 — Extract activations on base Llama-3-8B (2-3 hours compute)

```bash
# Download base model
huggingface-cli download meta-llama/Meta-Llama-3-8B

# Adapt hps_llama3.py to load base instead of Instruct
python strengthen_negative/extract_base_llama_activations.py \
    --model meta-llama/Meta-Llama-3-8B \
    --test-attacks llama3_attacks.json \
    --harmless data_harmless_100.csv \
    --harmful data_harmful_100.csv \
    --output results/llama3_base_activations_cache.npz
```

### Step 3.3 — Train HPS, HPS-Euclidean, C4 on base activations (2 hours)

Reuse statistical_tests.py + relevant pipeline scripts. All three methods on base Llama-3 activations.

### Step 3.4 — Per-attack comparison (1 day analysis)

Run `gcg_specific_test.py` adapted for three LLMs:
- Llama-3-base (no alignment)
- Llama-3-Instruct (SFT + RLHF)
- Vicuna-13B (SFT only)

**Predicted outcomes (and what each means):**

| Outcome | Interpretation |
|---|---|
| HPS catches GCG: Instruct=100%, base≈100%, Vicuna=37.5% | **NOT alignment-mediated** — confound is something else (model size, layer choice, data quality) |
| HPS catches GCG: Instruct=100%, base<60%, Vicuna=37.5% | **CONFIRMS alignment-mediated** — gradient with alignment strength |
| HPS catches GCG: Instruct=100%, base<60%, Vicuna≈100% | Mixed picture — may be specific to base model or specific to Vicuna |
| HPS catches GCG: All three at ~100% | The alignment-mediation hypothesis is **wrong**; something else explains the original Vicuna observation |

### Step 3.5 — Update paper with three-LLM analysis

If the alignment hypothesis is confirmed: lead with the three-LLM mechanism finding.
If not: reframe as "HPS has architecture-specific failure modes that we don't fully understand; future work needed."

---

## Phase 4 — Address Hyperbolic MLP finding (half-day analysis)

**New finding from Experiment 1:** Hyperbolic MLP achieves TPR=1.000 on Vicuna where standard HPS gets 0.703.

### Step 4.1 — Update narrative

The paper now has a more nuanced story:

1. **Geometric priors don't help over C4** (still true — no hyperbolic method beats C4)
2. **Vicuna failure is HPS-architecture-specific, not geometry-fundamental** (Hyperbolic MLP works fine)
3. **The "alignment-mediated → compression filters signatures" mechanism may be wrong** — Hyperbolic MLP also compresses but works

### Step 4.2 — Add to research_journey.md and paper outline

Add a subsection: "Why HPS specifically fails on Vicuna while other hyperbolic methods don't"

Possible explanations:
- Contrastive loss objective (vs supervised classification in MLP)
- Trajectory feature reduction (12 features collapse to 1)
- Single learned linear projection (vs multi-layer stacked Lorentz)

This would be the **strongest possible** version of the paper: not "HPS fails because alignment is weak" but "HPS's specific architecture has fragility that other hyperbolic architectures avoid; nonetheless, no hyperbolic method beats simpler linear probes."

---

## Phase 5 — Update Documents (3-4 days)

### Step 5.1 — Update research_journey.md
- Add Phase 1 results (larger Vicuna N)
- Add Phase 2 results (pre-training radial)
- Add Phase 3 results (alignment ablation)
- Add Phase 4 reframing (Hyperbolic MLP nuance)
- Address all 19 reviewer Tier-1 issues from clarifications

### Step 5.2 — Update mentor_briefing.md
- Replace single-LLM caveats with three-LLM evidence
- Update Three Confirmed Findings with stronger evidence
- Add Hyperbolic MLP finding as nuance to Finding #3
- Address reviewer's methodology clarity points (layer selection, contrastive loss specifics, feature definitions)

### Step 5.3 — Update paper_outline.md
- Methods section: full contrastive loss equation, full feature definitions
- Reframe abstract with three-LLM evidence
- Add Phase 4 finding to discussion
- Update related work to include the hyperbolic methods comparison

### Step 5.4 — Run final validation
- Re-run all key experiments with finalized data
- Generate updated plots
- Write reproduction documentation

---

## Decision Tree

```
Step 1.1 (verify JBShield data):
  AVAILABLE → proceed Phase 1
  NOT AVAILABLE → skip to Phase 3 only

Step 1.3 (re-run with larger N):
  Alignment finding holds → continue to Phase 2 + 3
  Alignment finding evaporates → skip Phase 3, reframe paper as "Hyperbolic MLP works
                                  but standard HPS doesn't; here's the architectural lesson"

Step 3.4 (alignment ablation result):
  HPS fails on base, works on Instruct → STRONG paper, alignment hypothesis confirmed
  HPS works on base AND Instruct → reframe — "HPS doesn't depend on alignment strength;
                                   the original Vicuna observation was something else"
  All three LLMs differ → publish as "HPS exhibits LLM-specific failures we don't fully
                          understand; future work"
```

---

## Total Time Budget

| Phase | Time | Critical Path? |
|---|---|---|
| Phase 1 (JBShield Vicuna data) | 1-2 days | Yes |
| Phase 2 (pre-training radial) | 0.5 day | No (parallelize) |
| Phase 3 (base vs Instruct) | 1-2 weeks | Yes |
| Phase 4 (Hyperbolic MLP nuance) | 0.5 day analysis | No |
| Phase 5 (documents) | 3-4 days | Yes |

**Total:** 2-3 weeks of work. **Earliest submission:** End of June 2026 if started immediately.

---

## What If You Have Less Time

### 1 week available
- Skip Phase 3 (base vs Instruct ablation)
- Do Phase 1 + 2 + 4 + 5
- Paper has stronger Vicuna evidence (n=170) but no clean alignment test
- Acknowledge in limitations: "Direct test of alignment-strength hypothesis (base vs aligned LLM) deferred to future work"

### 3 days available
- Skip Phase 1 and 3
- Do Phase 2 + 4 + 5 only
- Paper has same Vicuna n=16 issue but better methodology framing
- Drop the strong "alignment-mediated" claim; frame as "preliminary observation"

### Less than 3 days
- Drop Vicuna entirely (Option C from earlier discussion)
- Single-LLM paper with caveats
- Submit to TMLR or NeurIPS Safety Workshop

---

## Files to Create/Modify

### New scripts to write
- [ ] `strengthen_negative/extract_jbshield_vicuna_data.py` — Step 1.2
- [ ] `strengthen_negative/extract_base_llama_activations.py` — Step 3.2
- [ ] `strengthen_negative/alignment_ablation.py` — Step 3.4 (already started by other AI)

### Existing scripts to update
- [ ] `radial_distribution_check.py` — add pre-training control (Step 2.1)
- [ ] `vicuna_imbalance_test.py` — accept new cache path (Step 1.3)
- [ ] `gcg_specific_test.py` — extend to three LLMs (Step 3.4)

### Documents to update
- [ ] `research_journey.md` — full update with all phases
- [ ] `mentor_briefing.md` — three-LLM evidence + methodology clarity
- [ ] `paper_outline.md` — abstract, methods, results, discussion

---

## Notes

1. **Phase 1 is highest priority** — it's cheapest and most directly addresses the reviewer's strongest critique (n=16 sample size).

2. **Phase 3 is highest impact** — base vs Instruct is the cleanest possible alignment test.

3. **Phase 4 (Hyperbolic MLP nuance) was an unexpected finding.** It actually strengthens the negative result claim ("no hyperbolic method beats C4") but weakens the alignment-mediation mechanism story.

4. **If Phase 3 results contradict the alignment hypothesis**, the paper should pivot to the safer "geometric priors don't help across multiple architectures" framing.

5. **Reviewer's methodology critiques (#1-7) are all answerable in writing** — don't need new experiments to address.
