#!/usr/bin/env python3
"""
mmlu_taxonomy.py — build a GENUINE deep, ground-truth benign hierarchy from MMLU, to test whether
"deeper hierarchy -> bigger hyperbolic retrieval advantage" is a GENERAL law or a harm-data quirk.

Why MMLU: the harm taxonomy's "depth helps" result (gap +0.10 at 2-level -> +0.19 at 3-level) needs
a control on a hierarchy that is (a) NOT harm, (b) genuinely multi-level, (c) ground-truth-labeled
(no LLM labeling, no 'general' fallback). MMLU is ideal: 57 subjects roll up into 4 categories, a
real 2-level tree (category/subject), and we can optionally cut a 3rd level by sub-grouping subjects.
The benign topic control we already ran (code/math/wiki sources) had NO natural depth — MMLU does.

Labels are emitted as '/'-separated paths matching hyperbolic_retrieval.load_reps:
   level 2 : "category/subject"            (e.g. "stem/college_physics")
   level 3 : "category/group/subject"      (e.g. "stem/physics/college_physics")
so ALL existing machinery (load_reps shared-prefix tree distance, --stats bootstrap, --baselines,
--dims/--curvatures sweep) works unchanged on the output reps.

Commands:
  extract : pull MMLU (via `datasets`), forward each question, save last-token reps + path labels.  (GPU)
  (then run hyperbolic_retrieval.py on the produced *_reps.npz exactly like the harm tree.)

Usage:
  python mmlu_taxonomy.py extract --model_path $MP --layer 24 \
    --per_subject 40 --level 3 --output results/mmlu_taxonomy
  # -> results/mmlu_taxonomy_reps.npz  (reps + labels)
  # then:
  python hyperbolic_retrieval.py --harmful_npz results/mmlu_taxonomy_reps.npz \
    --dims 16 32 --curvatures 1.0 --seeds 5 --stats --baselines cosine,whitened \
    --output results/mmlu_retr_stats

NOTE: needs the `datasets` package (commented out in requirements.txt) -> `pip install datasets` on
the DGX. Use --selftest (no GPU, no datasets) to validate the label/tree construction.
"""
import argparse, json, os, sys
import numpy as np

# Canonical MMLU 57-subject -> 4-category mapping (the standard hendrycks grouping).
SUBJECT_CATEGORY = {
    # STEM
    "abstract_algebra": "stem", "anatomy": "stem", "astronomy": "stem", "college_biology": "stem",
    "college_chemistry": "stem", "college_computer_science": "stem", "college_mathematics": "stem",
    "college_physics": "stem", "computer_security": "stem", "conceptual_physics": "stem",
    "electrical_engineering": "stem", "elementary_mathematics": "stem", "high_school_biology": "stem",
    "high_school_chemistry": "stem", "high_school_computer_science": "stem",
    "high_school_mathematics": "stem", "high_school_physics": "stem", "high_school_statistics": "stem",
    "machine_learning": "stem",
    # humanities
    "formal_logic": "humanities", "high_school_european_history": "humanities",
    "high_school_us_history": "humanities", "high_school_world_history": "humanities",
    "international_law": "humanities", "jurisprudence": "humanities", "logical_fallacies": "humanities",
    "moral_disputes": "humanities", "moral_scenarios": "humanities", "philosophy": "humanities",
    "prehistory": "humanities", "professional_law": "humanities", "world_religions": "humanities",
    # social sciences
    "econometrics": "social_sciences", "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences", "high_school_macroeconomics": "social_sciences",
    "high_school_microeconomics": "social_sciences", "high_school_psychology": "social_sciences",
    "human_sexuality": "social_sciences", "professional_psychology": "social_sciences",
    "public_relations": "social_sciences", "security_studies": "social_sciences",
    "sociology": "social_sciences", "us_foreign_policy": "social_sciences",
    # other
    "business_ethics": "other", "clinical_knowledge": "other", "college_medicine": "other",
    "global_facts": "other", "human_aging": "other", "management": "other", "marketing": "other",
    "medical_genetics": "other", "miscellaneous": "other", "nutrition": "other",
    "professional_accounting": "other", "professional_medicine": "other", "virology": "other",
}

# Optional 3rd level: group subjects within a category into mid-level fields (gives real depth so we
# can test whether the curved advantage GROWS from level 2 -> level 3 on benign data).
SUBJECT_GROUP = {
    # stem groups
    "physics": ["college_physics", "high_school_physics", "conceptual_physics", "astronomy"],
    "math": ["abstract_algebra", "college_mathematics", "elementary_mathematics",
             "high_school_mathematics", "high_school_statistics"],
    "cs_eng": ["college_computer_science", "high_school_computer_science", "machine_learning",
               "computer_security", "electrical_engineering"],
    "biology": ["college_biology", "high_school_biology", "anatomy"],
    "chemistry": ["college_chemistry", "high_school_chemistry"],
    # humanities groups
    "history": ["high_school_european_history", "high_school_us_history", "high_school_world_history",
                "prehistory"],
    "law": ["international_law", "jurisprudence", "professional_law"],
    "philosophy_ethics": ["philosophy", "moral_disputes", "moral_scenarios", "logical_fallacies",
                          "world_religions", "formal_logic"],
    # social science groups
    "economics": ["econometrics", "high_school_macroeconomics", "high_school_microeconomics"],
    "politics_geo": ["high_school_geography", "high_school_government_and_politics",
                     "us_foreign_policy", "security_studies", "public_relations"],
    "psych_socio": ["high_school_psychology", "professional_psychology", "human_sexuality", "sociology"],
    # other groups
    "medicine": ["clinical_knowledge", "college_medicine", "professional_medicine", "medical_genetics",
                 "virology", "human_aging", "nutrition"],
    "business": ["business_ethics", "management", "marketing", "professional_accounting"],
    "misc": ["global_facts", "miscellaneous"],
}
SUBJECT_TO_GROUP = {s: g for g, subs in SUBJECT_GROUP.items() for s in subs}


def path_for(subject, level):
    cat = SUBJECT_CATEGORY.get(subject, "other")
    if level >= 3:
        grp = SUBJECT_TO_GROUP.get(subject, "misc")
        return f"{cat}/{grp}/{subject}"
    return f"{cat}/{subject}"


def cmd_extract(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("[mmlu] needs the `datasets` package: pip install datasets")
    # MMLU 'all' config has every subject; use the test split (largest).
    print("[mmlu] loading MMLU via datasets (cais/mmlu, config 'all')...", flush=True)
    ds = load_dataset("cais/mmlu", "all", split="test")
    rng = np.random.default_rng(0)
    # group row indices by subject, cap per subject for balance
    from collections import defaultdict
    by_subj = defaultdict(list)
    for i, subj in enumerate(ds["subject"]):
        by_subj[subj].append(i)
    chosen = []
    for subj, idxs in by_subj.items():
        if subj not in SUBJECT_CATEGORY:
            continue
        if len(idxs) > args.per_subject:
            idxs = [idxs[j] for j in rng.choice(len(idxs), args.per_subject, replace=False)]
        chosen += [(i, subj) for i in idxs]
    print(f"[mmlu] {len(chosen)} questions across {len(by_subj)} subjects "
          f"(cap {args.per_subject}/subject); level={args.level}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16,
                                                 device_map="cuda").eval()
    reps, labels = [], []
    questions = ds["question"]; choices = ds["choices"]
    with torch.no_grad():
        for n, (i, subj) in enumerate(chosen):
            q = questions[i]
            ch = choices[i]
            text = q + "\n" + "\n".join(f"{chr(65+j)}. {c}" for j, c in enumerate(ch))
            ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                          return_tensors="pt", add_generation_prompt=True).to("cuda")
            hs = model(ids, output_hidden_states=True).hidden_states[args.layer][0, -1].float().cpu().numpy()
            reps.append(hs); labels.append(path_for(subj, args.level))
            if (n + 1) % 200 == 0:
                print(f"[mmlu] {n+1}/{len(chosen)}", flush=True)
    reps = np.array(reps); labels = np.array(labels)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(args.output + "_reps.npz", reps=reps, labels=labels)
    from collections import Counter
    print(f"[mmlu] wrote {args.output}_reps.npz: {len(reps)} reps, "
          f"{len(set(labels.tolist()))} leaf paths, {len(set(l.split('/')[0] for l in labels))} top categories", flush=True)
    print("[mmlu] depth check — distinct paths per level:")
    print("   L1 categories:", sorted(set(l.split('/')[0] for l in labels)))
    if args.level >= 3:
        print("   L2 groups:", len(set('/'.join(l.split('/')[:2]) for l in labels)))
    print(f"[mmlu] NEXT: python hyperbolic_retrieval.py --harmful_npz {args.output}_reps.npz "
          f"--dims 16 32 --curvatures 1.0 --seeds 5 --stats --baselines cosine,whitened "
          f"--output results/mmlu_retr_stats", flush=True)


def cmd_selftest(args):
    """No GPU / no datasets: validate the taxonomy + path construction + that load_reps parses it."""
    print("[mmlu] selftest — taxonomy construction + load_reps compatibility (no GPU)")
    # sanity: every subject maps to a category and (for L3) a group
    miss_cat = [s for s in SUBJECT_CATEGORY if s not in SUBJECT_CATEGORY]
    miss_grp = [s for s in SUBJECT_CATEGORY if s not in SUBJECT_TO_GROUP]
    print(f"  subjects: {len(SUBJECT_CATEGORY)} | categories: {len(set(SUBJECT_CATEGORY.values()))} "
          f"| groups: {len(SUBJECT_GROUP)}")
    print(f"  subjects with no L3 group (fall to nothing): {miss_grp}")
    # build a fake reps npz from the taxonomy and confirm load_reps gives 2/4/6 tree distances at L3
    subs = list(SUBJECT_CATEGORY)[:20]
    labels = np.array([path_for(s, 3) for s in subs for _ in range(5)])
    reps = np.random.RandomState(0).randn(len(labels), 16).astype("float32")
    np.savez("/tmp/mmlu_self.npz", reps=reps, labels=labels)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hyperbolic_retrieval import load_reps
    Xh, leaf, parent, la, tree_D, leaves, parents = load_reps("/tmp/mmlu_self.npz")
    print(f"  load_reps OK: {len(leaves)} leaves, {len(parents)} parents (=L2 groups)")
    print(f"  tree_D distinct values: {sorted(set(tree_D.flatten().tolist()))}  (expect 0,2,4,6 at L3)")
    print(f"  sample leaf paths: {leaves[:3]}")
    os.remove("/tmp/mmlu_self.npz")
    print("[mmlu] selftest PASSED")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--model_path", required=True); e.add_argument("--layer", type=int, default=24)
    e.add_argument("--per_subject", type=int, default=40, help="cap questions per subject (balance)")
    e.add_argument("--level", type=int, default=3, choices=[2, 3], help="2=category/subject, 3=category/group/subject")
    e.add_argument("--output", default="results/mmlu_taxonomy")
    s = sub.add_parser("selftest")
    args = ap.parse_args()
    {"extract": cmd_extract, "selftest": cmd_selftest}[args.cmd](args)


if __name__ == "__main__":
    main()
