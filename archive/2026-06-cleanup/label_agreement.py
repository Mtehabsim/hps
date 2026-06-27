#!/usr/bin/env python3
"""
label_agreement.py — inter-labeler agreement check (#2): does the harm-leaf labeling reflect real
structure, or is the 8B-model labeler systematically biased?

The label-noise sweep (hyperbolic_retrieval.py --label_noise) tests robustness to RANDOM mislabeling.
It does NOT catch SYSTEMATIC bias — e.g. the labeler consistently confusing fraud↔theft. This script
catches that by cross-checking two INDEPENDENT labelers on the same prompts:

  Labeler A = the LLM labels we actually used        (results/harm_taxonomy_llm.json)
  Labeler B = the keyword assigner                   (harm_taxonomy.assign)

Agreement rate is a real label-quality estimate; the confusion matrix shows WHICH leaf pairs are
systematically confused (off-diagonal mass) vs random scatter. High agreement + diagonal confusion
=> labels are trustworthy. Low agreement concentrated on a few pairs => those leaves are unreliable
and should be merged / re-labeled / flagged.

CPU-only. Usage:
  python label_agreement.py --llm_json results/harm_taxonomy_llm.json --output results/label_agreement
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
from harm_taxonomy import assign  # keyword labeler: prompt -> (category, leaf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_json", default="results/harm_taxonomy_llm.json")
    ap.add_argument("--output", default="results/label_agreement")
    ap.add_argument("--level", choices=["category", "leaf"], default="leaf",
                    help="compare at top-category level or full category/leaf level")
    args = ap.parse_args()

    A = json.load(open(args.llm_json))["assignments"]
    rows = []
    for r in A:
        llm = (r["category"], r["leaf"])
        kw = assign(r["prompt"])                  # (category, leaf) or ('unassigned','unassigned')
        rows.append({"prompt": r["prompt"], "llm": llm, "kw": kw})

    def key(t):
        return t[0] if args.level == "category" else f"{t[0]}/{t[1]}"

    # Agreement only where BOTH labelers commit (keyword 'unassigned' = abstain, excluded from the
    # agreement rate but counted separately — it is coverage, not disagreement).
    committed = [x for x in rows if x["kw"][0] != "unassigned"]
    abstain = len(rows) - len(committed)
    agree = sum(1 for x in committed if key(x["llm"]) == key(x["kw"]))
    n = len(committed)
    rate = agree / n if n else float("nan")

    # Cohen's kappa over committed pairs (chance-corrected agreement)
    labels = sorted(set(key(x["llm"]) for x in committed) | set(key(x["kw"]) for x in committed))
    ix = {l: i for i, l in enumerate(labels)}
    K = len(labels)
    M = np.zeros((K, K), dtype=int)              # rows = LLM, cols = keyword
    for x in committed:
        M[ix[key(x["llm"])], ix[key(x["kw"])]] += 1
    po = np.trace(M) / M.sum()
    pe = (M.sum(1) @ M.sum(0)) / (M.sum() ** 2)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else float("nan")

    # Top systematic confusions (off-diagonal cells, largest first)
    confusions = []
    for i in range(K):
        for j in range(K):
            if i != j and M[i, j] > 0:
                confusions.append((int(M[i, j]), labels[i], labels[j]))
    confusions.sort(reverse=True)

    # Per-LLM-leaf agreement (which leaves are individually unreliable)
    per_leaf = {}
    for i, l in enumerate(labels):
        tot = M[i].sum()
        if tot > 0:
            per_leaf[l] = {"n": int(tot), "agree": int(M[i, i]), "rate": float(M[i, i] / tot)}

    print(f"[agree] level={args.level}; {len(rows)} prompts, {n} committed by keyword labeler, "
          f"{abstain} keyword-abstain ({100*abstain/len(rows):.0f}% coverage gap)", flush=True)
    print(f"[agree] raw agreement (LLM vs keyword, committed) = {rate:.3f}  ({agree}/{n})", flush=True)
    print(f"[agree] Cohen's kappa = {kappa:.3f}  "
          f"({'strong' if kappa>0.6 else 'moderate' if kappa>0.4 else 'weak'} chance-corrected agreement)", flush=True)
    print(f"\n[agree] top systematic confusions (LLM-label -> keyword-label, count):", flush=True)
    for cnt, a, b in confusions[:12]:
        print(f"    {cnt:>3}   {a:30s} -> {b}", flush=True)
    print(f"\n[agree] least-reliable LLM leaves (lowest self-agreement, n>=10):", flush=True)
    for l, d in sorted(per_leaf.items(), key=lambda kv: kv[1]["rate"]):
        if d["n"] >= 10:
            print(f"    {l:32s} agree {d['rate']:.2f}  (n={d['n']})", flush=True)

    out = {"level": args.level, "n_prompts": len(rows), "n_committed": n, "keyword_abstain": abstain,
           "raw_agreement": rate, "cohens_kappa": kappa,
           "top_confusions": [{"count": c, "llm": a, "keyword": b} for c, a, b in confusions[:20]],
           "per_leaf_agreement": per_leaf,
           "interpretation": (
               "High kappa (>0.6) + confusions clustered on semantically-adjacent leaves => labels "
               "reflect real structure, disagreements are boundary cases. Low kappa or confusions "
               "concentrated on a few pairs => those leaves are systematically conflated and the "
               "retrieval ground-truth there is unreliable; consider merging or re-labeling them.")}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump(out, open(args.output + ".json", "w"), indent=2)
    print(f"\n[agree] wrote {args.output}.json", flush=True)


if __name__ == "__main__":
    main()
