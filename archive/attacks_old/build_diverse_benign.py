"""
build_diverse_benign.py — Construct a diverse benign prompt set to break
length / format confounds in jailbreak detection evaluation.

Combines benign prompts from multiple sources to span:
  - Real user conversations (WildChat)
  - Edge-case benign that LOOKS suspicious (OR-Bench Hard)
  - Factual queries across domains (MMLU)
  - Math word problems (GSM8K)
  - Coding requests (HumanEval, MBPP)
  - Long-form natural text (WikiText)
  - Multi-language benign (FLORES, Aya)

Goal: Match attack-set diversity in length, format, language, and domain so
the classifier can't shortcut via "is this prompt unusual?" instead of
"is this prompt harmful?"

Following Anthropic Cheap Monitors' protocol (Cunningham et al. 2025):
  "Harmless test examples comprise: Single-turn inputs from WildChat
   (Zhao et al., 2024); the 'Hard' subset of OR-bench (Cui et al., 2024)."

Output:
  results/data_harmless_diverse.csv  — single column "prompt", N rows
  results/diverse_benign_stats.json   — per-source statistics

Usage:
  pip install datasets
  python build_diverse_benign.py
  python build_diverse_benign.py --n_samples 6000
  python build_diverse_benign.py --no_multilingual
"""

import argparse
import json
import os
import random
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Sample budgets per source (default)
# ---------------------------------------------------------------------------

DEFAULT_BUDGETS = {
    "alpaca_control":   500,   # preserve some old samples for back-compat
    "wildchat":        1500,   # real user diversity
    "or_bench_hard":   1000,   # tricky benign that looks suspicious
    "mmlu":            1000,   # factual cross-domain
    "gsm8k":            500,   # math problems
    "humaneval":        164,   # coding (HumanEval has only 164)
    "mbpp":             500,   # more coding
    "wikitext_short":   400,   # natural text 200-1500 chars
    "wikitext_long":    600,   # natural text 1500-15000 chars (CRITICAL: matches drattack/ijp/saa/puzzler)
    "multilingual":     500,   # non-English benign
}


# ---------------------------------------------------------------------------
# Quality filters
# ---------------------------------------------------------------------------

def is_clean_text(text, min_chars=15, max_chars=20000):
    """Reject empty, whitespace-only, URL-only, or extreme-length text.

    NOTE: max_chars=20000 (was 4000) so we can match very long attacks
    (puzzler attacks are up to 17,000 chars; drattack/ijp/saa are 2-9K chars).
    """
    if not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < min_chars:
        return False
    if len(text) > max_chars:
        return False
    # Reject prompts that are >50% URL
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    url_chars = sum(len(m.group()) for m in url_pattern.finditer(text))
    if url_chars / max(len(text), 1) > 0.5:
        return False
    # Reject pure-encoding (looks like base64 or hex)
    if re.fullmatch(r'[A-Za-z0-9+/=]{50,}', text):
        return False
    return True


def is_likely_safe(text):
    """Lightweight filter: reject prompts containing obvious harmful content
    keywords that might have leaked into 'benign' datasets like WildChat."""
    if not isinstance(text, str):
        return False
    bad_keywords = [
        "how to make a bomb", "how to kill", "child porn",
        "make explosive", "how to commit suicide",
        "recipe for", "synthesize ricin", "manufacture meth",
    ]
    lower = text.lower()
    return not any(kw in lower for kw in bad_keywords)


# ---------------------------------------------------------------------------
# Source loaders (all return list of strings, with quality filtering)
# ---------------------------------------------------------------------------

def load_alpaca_control(n, seed=42, existing_csv="data_harmless_6500.csv"):
    """Take some samples from existing Alpaca-style benign for back-compat."""
    print(f"  [alpaca_control] loading from {existing_csv}...")
    if not os.path.exists(existing_csv):
        print(f"    WARNING: {existing_csv} not found; using empty list")
        return []
    df = pd.read_csv(existing_csv)
    prompts = df.iloc[:, 0].dropna().astype(str).tolist()
    rng = random.Random(seed)
    rng.shuffle(prompts)
    keep = [p for p in prompts if is_clean_text(p)][:n]
    print(f"    kept {len(keep)} prompts")
    return keep


def load_wildchat(n, seed=42):
    """Real user conversations from Allen Institute's WildChat."""
    print(f"  [wildchat] loading...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("    WARNING: 'datasets' library not installed; skipping")
        return []
    try:
        # Allen-AI's filtered subset
        ds = load_dataset(
            "allenai/WildChat-1M",
            split="train",
            streaming=True,  # avoid full download
        )
    except Exception as e:
        print(f"    WARNING: WildChat load failed ({e}); skipping")
        return []

    rng = random.Random(seed)
    candidates = []
    for i, row in enumerate(ds):
        if i > 50000:  # don't process more than 50K
            break
        # Filter: English only, single-turn first message
        if row.get("language") != "English":
            continue
        if row.get("toxic", False):
            continue
        msgs = row.get("conversation", [])
        if not msgs or msgs[0].get("role") != "user":
            continue
        text = msgs[0].get("content", "")
        if is_clean_text(text) and is_likely_safe(text):
            candidates.append(text)
        if len(candidates) >= n * 3:  # collect 3x for sampling
            break

    rng.shuffle(candidates)
    keep = candidates[:n]
    print(f"    kept {len(keep)} from {len(candidates)} candidates")
    return keep


def load_or_bench_hard(n, seed=42):
    """OR-Bench Hard: tricky benign prompts that LOOK suspicious."""
    print(f"  [or_bench_hard] loading...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("    skipping (no datasets library)")
        return []
    try:
        ds = load_dataset("bench-llm/or-bench", "or-bench-hard-1k",
                          split="train")
    except Exception as e:
        print(f"    WARNING: OR-Bench load failed ({e}); skipping")
        return []
    prompts = []
    for row in ds:
        text = row.get("prompt", "") or row.get("question", "")
        if is_clean_text(text):
            prompts.append(text)
    rng = random.Random(seed)
    rng.shuffle(prompts)
    keep = prompts[:n]
    print(f"    kept {len(keep)}")
    return keep


def load_mmlu(n, seed=42):
    """MMLU questions across diverse subjects."""
    print(f"  [mmlu] loading...")
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    try:
        ds = load_dataset("cais/mmlu", "all", split="test")
    except Exception as e:
        print(f"    WARNING: MMLU load failed ({e}); skipping")
        return []
    prompts = []
    for row in ds:
        q = row.get("question", "")
        choices = row.get("choices", [])
        if q and choices:
            text = q + "\n" + "\n".join(
                f"{chr(65+i)}. {c}" for i, c in enumerate(choices)
            )
            if is_clean_text(text):
                prompts.append(text)
    rng = random.Random(seed)
    rng.shuffle(prompts)
    keep = prompts[:n]
    print(f"    kept {len(keep)}")
    return keep


def load_gsm8k(n, seed=42):
    """GSM8K math word problems."""
    print(f"  [gsm8k] loading...")
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    try:
        ds = load_dataset("gsm8k", "main", split="train")
    except Exception as e:
        print(f"    WARNING: GSM8K load failed ({e}); skipping")
        return []
    prompts = []
    for row in ds:
        q = row.get("question", "")
        if is_clean_text(q):
            prompts.append(q)
    rng = random.Random(seed)
    rng.shuffle(prompts)
    keep = prompts[:n]
    print(f"    kept {len(keep)}")
    return keep


def load_humaneval(n, seed=42):
    """HumanEval coding prompts."""
    print(f"  [humaneval] loading...")
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    try:
        ds = load_dataset("openai_humaneval", split="test")
    except Exception:
        try:
            ds = load_dataset("openai/openai_humaneval", split="test")
        except Exception as e:
            print(f"    WARNING: HumanEval load failed ({e}); skipping")
            return []
    prompts = []
    for row in ds:
        text = row.get("prompt", "")
        if is_clean_text(text):
            prompts.append(text)
    rng = random.Random(seed)
    rng.shuffle(prompts)
    keep = prompts[:n]
    print(f"    kept {len(keep)}")
    return keep


def load_mbpp(n, seed=42):
    """MBPP coding tasks (more diversity than HumanEval)."""
    print(f"  [mbpp] loading...")
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    try:
        ds = load_dataset("mbpp", split="train")
    except Exception as e:
        print(f"    WARNING: MBPP load failed ({e}); skipping")
        return []
    prompts = []
    for row in ds:
        text = row.get("text", "") or row.get("prompt", "")
        if is_clean_text(text):
            prompts.append(text)
    rng = random.Random(seed)
    rng.shuffle(prompts)
    keep = prompts[:n]
    print(f"    kept {len(keep)}")
    return keep


def load_wikitext(n, seed=42, min_chars=200, max_chars=1500, name="wikitext"):
    """WikiText: shorter natural text passages (200-1500 chars)."""
    print(f"  [{name}] loading...")
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    try:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    except Exception as e:
        print(f"    WARNING: WikiText load failed ({e}); skipping")
        return []
    prompts = []
    for row in ds:
        text = row.get("text", "")
        if is_clean_text(text, min_chars=min_chars, max_chars=max_chars):
            prompts.append(text)
        if len(prompts) >= n * 3:
            break
    rng = random.Random(seed)
    rng.shuffle(prompts)
    keep = prompts[:n]
    print(f"    kept {len(keep)} (target length {min_chars}-{max_chars})")
    return keep


def load_wikitext_long(n, seed=42, target_lengths=None):
    """
    WikiText LONG: concatenate passages to produce long benign text matching
    the lengths of long attacks (drattack ~1958, ijp ~2030, saa ~2197,
    puzzler ~11,684 chars).

    Target distribution: 1500-15000 chars, weighted toward common attack lengths.
    """
    print(f"  [wikitext_long] building long-text benign to match attack lengths...")
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    if target_lengths is None:
        # Mix of lengths matching attack distributions
        target_lengths = [
            1500, 1700, 2000, 2200,         # match drattack/ijp/saa
            2500, 3000, 4000,
            5000, 7000, 10000, 12000, 15000  # match puzzler
        ]

    try:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    except Exception as e:
        print(f"    WARNING: WikiText load failed ({e}); skipping")
        return []

    rng = random.Random(seed)
    prompts = []
    buffer = ""
    next_target = rng.choice(target_lengths)

    for row in ds:
        text = row.get("text", "")
        if not text or text.strip().startswith("="):
            continue
        buffer += text + " "
        if len(buffer) >= next_target:
            cleaned = buffer.strip()
            # Don't exceed max
            if len(cleaned) > 20000:
                cleaned = cleaned[:20000]
            if is_clean_text(cleaned, min_chars=1500, max_chars=20000):
                prompts.append(cleaned)
            buffer = ""
            next_target = rng.choice(target_lengths)
            if len(prompts) >= n * 2:
                break

    rng.shuffle(prompts)
    keep = prompts[:n]
    if keep:
        ls = [len(p) for p in keep]
        print(f"    kept {len(keep)}: char lengths "
              f"min={min(ls)}, median={int(np.median(ls))}, "
              f"mean={int(np.mean(ls))}, max={max(ls)}")
    else:
        print(f"    WARNING: no long passages constructed")
    return keep


def load_multilingual(n, seed=42):
    """Multi-language benign prompts to counterbalance Zulu attacks."""
    print(f"  [multilingual] loading...")
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    prompts = []

    # Try Aya dataset (multi-language instruction tuning)
    try:
        ds = load_dataset(
            "CohereForAI/aya_dataset",
            split="train",
            streaming=True,
        )
        # Sample diverse languages
        languages_wanted = {"Arabic", "Chinese", "French", "German",
                             "Hindi", "Japanese", "Russian", "Spanish",
                             "Swahili", "Turkish", "Vietnamese"}
        per_lang = n // len(languages_wanted) + 1
        lang_counts = Counter()
        for i, row in enumerate(ds):
            if i > 30000:
                break
            lang = row.get("language", "")
            if lang in languages_wanted and lang_counts[lang] < per_lang:
                text = row.get("inputs", "") or row.get("prompt", "")
                if is_clean_text(text):
                    prompts.append(text)
                    lang_counts[lang] += 1
            if sum(lang_counts.values()) >= n:
                break
        print(f"    kept {len(prompts)} from {dict(lang_counts)}")
    except Exception as e:
        print(f"    WARNING: Aya load failed ({e}); trying FLORES")
        try:
            ds = load_dataset("facebook/flores", "all", split="dev")
            for row in ds:
                for lang_key in row:
                    if lang_key.startswith("sentence_"):
                        text = row[lang_key]
                        if is_clean_text(text):
                            prompts.append(text)
                            if len(prompts) >= n:
                                break
                if len(prompts) >= n:
                    break
            print(f"    kept {len(prompts)} from FLORES")
        except Exception as e2:
            print(f"    WARNING: FLORES also failed ({e2}); skipping")

    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts[:n]


# ---------------------------------------------------------------------------
# Combine, deduplicate, save
# ---------------------------------------------------------------------------

def build_diverse_benign(budgets, seed=42, existing_csv="data_harmless_6500.csv",
                          skip_multilingual=False):
    """Load from each source, deduplicate, return combined list with metadata."""
    print("\n" + "=" * 70)
    print("BUILDING DIVERSE BENIGN SET")
    print("=" * 70)

    sources = []
    for name, n in budgets.items():
        if n <= 0:
            continue
        if skip_multilingual and name == "multilingual":
            continue

        if name == "alpaca_control":
            prompts = load_alpaca_control(n, seed, existing_csv)
        elif name == "wildchat":
            prompts = load_wildchat(n, seed)
        elif name == "or_bench_hard":
            prompts = load_or_bench_hard(n, seed)
        elif name == "mmlu":
            prompts = load_mmlu(n, seed)
        elif name == "gsm8k":
            prompts = load_gsm8k(n, seed)
        elif name == "humaneval":
            prompts = load_humaneval(n, seed)
        elif name == "mbpp":
            prompts = load_mbpp(n, seed)
        elif name == "wikitext_short":
            prompts = load_wikitext(n, seed, min_chars=200, max_chars=1500,
                                    name="wikitext_short")
        elif name == "wikitext_long":
            prompts = load_wikitext_long(n, seed)
        elif name == "wikitext":  # back-compat alias
            prompts = load_wikitext(n, seed, name="wikitext")
        elif name == "multilingual":
            prompts = load_multilingual(n, seed)
        else:
            print(f"  [unknown source: {name}]")
            continue

        for p in prompts:
            sources.append({"text": p, "source": name})

    return sources


def deduplicate(prompts):
    """Remove exact duplicates (preserve order, first occurrence wins)."""
    seen = set()
    out = []
    for p in prompts:
        key = p["text"][:200]  # use first 200 chars as dedup key
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def compute_stats(prompts, tokenizer=None):
    """Compute per-source statistics."""
    by_source = {}
    for p in prompts:
        src = p["source"]
        if src not in by_source:
            by_source[src] = {"count": 0, "char_lens": [], "token_lens": []}
        by_source[src]["count"] += 1
        by_source[src]["char_lens"].append(len(p["text"]))
        if tokenizer is not None:
            try:
                tokens = tokenizer.encode(p["text"])
                by_source[src]["token_lens"].append(len(tokens))
            except Exception:
                pass

    stats = {}
    for src, s in by_source.items():
        stats[src] = {
            "count": s["count"],
            "char_len_mean": float(np.mean(s["char_lens"])),
            "char_len_median": float(np.median(s["char_lens"])),
            "char_len_min": int(np.min(s["char_lens"])),
            "char_len_max": int(np.max(s["char_lens"])),
        }
        if s["token_lens"]:
            stats[src]["token_len_mean"] = float(np.mean(s["token_lens"]))
            stats[src]["token_len_median"] = float(np.median(s["token_lens"]))
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Override total sample budget (else uses defaults)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--existing_csv", default="data_harmless_6500.csv",
                        help="Existing benign CSV to source 'alpaca_control' from")
    parser.add_argument("--output_csv", default="results/data_harmless_diverse.csv")
    parser.add_argument("--output_stats",
                        default="results/diverse_benign_stats.json")
    parser.add_argument("--no_multilingual", action="store_true",
                        help="Skip multilingual sources (faster)")
    parser.add_argument("--tokenizer",
                        default="gpt2",
                        help="Tokenizer for stats (default: gpt2, non-gated). "
                             "Use meta-llama/Meta-Llama-3-8B-Instruct if you "
                             "have HF token.")
    args = parser.parse_args()

    # Determine budgets
    budgets = dict(DEFAULT_BUDGETS)
    if args.n_samples is not None:
        # Scale all budgets proportionally to hit n_samples total
        scale = args.n_samples / sum(budgets.values())
        budgets = {k: max(1, int(v * scale)) for k, v in budgets.items()}
        print(f"Scaled budgets: {budgets}")

    print(f"  Total target: {sum(budgets.values())} samples")
    print(f"  Per-source budgets: {budgets}")
    print()

    # Build
    prompts = build_diverse_benign(
        budgets, seed=args.seed,
        existing_csv=args.existing_csv,
        skip_multilingual=args.no_multilingual,
    )

    # Deduplicate
    print(f"\n  Total before dedup: {len(prompts)}")
    prompts = deduplicate(prompts)
    print(f"  Total after dedup:  {len(prompts)}")

    # Shuffle for output
    rng = random.Random(args.seed)
    rng.shuffle(prompts)

    # Tokenizer for stats
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        print(f"  Loaded tokenizer: {args.tokenizer}")
    except Exception as e:
        print(f"  Tokenizer load failed: {e}; skipping token stats")

    # Compute stats
    stats = compute_stats(prompts, tokenizer=tokenizer)

    # Print summary
    print("\n" + "=" * 70)
    print("DIVERSE BENIGN STATS")
    print("=" * 70)
    print(f"\n  {'Source':<22}  {'N':>5}  {'char_mean':>10}  "
          f"{'char_median':>11}  {'char_max':>9}  {'token_mean':>10}")
    print("  " + "-" * 80)
    total = 0
    for src, s in sorted(stats.items()):
        token_str = f"{s.get('token_len_mean', '-'):>10.1f}" \
            if 'token_len_mean' in s else f"{'-':>10s}"
        print(f"  {src:<22}  {s['count']:>5}  "
              f"{s['char_len_mean']:>10.1f}  "
              f"{s['char_len_median']:>11.0f}  "
              f"{s['char_len_max']:>9}  "
              f"{token_str}")
        total += s["count"]
    print("  " + "-" * 80)
    print(f"  {'TOTAL':<22}  {total:>5}")

    # Save
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "prompt": [p["text"] for p in prompts],
        "source": [p["source"] for p in prompts],
    })
    df.to_csv(output_path, index=False)
    print(f"\n  Saved CSV → {output_path}")

    stats_path = Path(args.output_stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved stats → {stats_path}")

    # Compare against original
    if os.path.exists(args.existing_csv):
        df_orig = pd.read_csv(args.existing_csv)
        orig_lens = [len(str(p)) for p in df_orig.iloc[:, 0].dropna()]
        new_lens = [len(p["text"]) for p in prompts]
        print(f"\n  Length comparison (chars):")
        print(f"    Original ({args.existing_csv}): "
              f"mean={np.mean(orig_lens):.1f}, median={np.median(orig_lens):.0f}, "
              f"max={max(orig_lens)}")
        print(f"    Diverse ({output_path}): "
              f"mean={np.mean(new_lens):.1f}, median={np.median(new_lens):.0f}, "
              f"max={max(new_lens)}")


if __name__ == "__main__":
    main()
