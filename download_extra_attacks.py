"""
Download emerging attacks + multilingual jailbreaks and merge into llama3_attacks.json format.

Sources:
  1. MultiJail (DAMO-NLP-SG, ICLR 2024) — 315 prompts × 9 languages (Arabic, Bengali, Swahili, Javanese, Thai, Italian, Vietnamese, Chinese, Korean)
  2. SequentialBreak  — sequential prompt chain attacks (try GitHub mirror)
  3. arXiv:2411.03343 — 35-method dataset (if released)

Output: data/extra_attacks.json  with same shape as llama3_attacks.json
        { "method_name": [prompt1, prompt2, ...], ... }
"""
import os
import json
import urllib.request
import urllib.error
from pathlib import Path

OUT_DIR = Path("data/extra_attacks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = "data/extra_attacks.json"


def download(url: str, dest: Path) -> bool:
    """Download URL to dest. Returns True on success."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ✓ Already have {dest.name}")
        return True
    try:
        print(f"  Fetching {url} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  ✓ Saved → {dest}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def fetch_multijail():
    """MultiJail dataset from DAMO-NLP-SG repo."""
    print("\n── 1. MultiJail (multilingual) ──")
    base = "https://raw.githubusercontent.com/DAMO-NLP-SG/multilingual-safety-for-LLMs/main/data"

    # The repo has data/MultiJail.csv with all language columns
    csv_path = OUT_DIR / "MultiJail.csv"
    if not download(f"{base}/MultiJail.csv", csv_path):
        return {}

    import csv
    by_lang = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Columns are typically: id, source, tags, en, zh, it, vi, ar, ko, th, bn, sw, jv
        lang_keys = ["ar", "bn", "sw", "jv", "th", "it", "vi", "zh", "ko"]
        for lang in lang_keys:
            by_lang[f"multijail_{lang}"] = []
        for row in reader:
            for lang in lang_keys:
                if lang in row and row[lang].strip():
                    by_lang[f"multijail_{lang}"].append(row[lang].strip())

    for k in list(by_lang.keys()):
        if not by_lang[k]:
            del by_lang[k]
        else:
            print(f"  {k}: {len(by_lang[k])} prompts")
    return by_lang


def fetch_sequentialbreak():
    """SequentialBreak — try multiple possible repo locations."""
    print("\n── 2. SequentialBreak ──")
    candidates = [
        "https://raw.githubusercontent.com/SequentialBreak/SequentialBreak/main/dataset/sequential_attacks.json",
        "https://raw.githubusercontent.com/Bishan-Sub/SequentialBreak/main/data/attacks.json",
    ]
    dest = OUT_DIR / "sequentialbreak.json"
    for url in candidates:
        if download(url, dest):
            try:
                with open(dest) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return {"sequentialbreak": [str(x) for x in data]}
                elif isinstance(data, dict):
                    flat = []
                    for v in data.values():
                        if isinstance(v, list):
                            flat.extend([str(x) for x in v])
                    return {"sequentialbreak": flat}
            except Exception as e:
                print(f"  ✗ Parse failed: {e}")
    print("  ✗ SequentialBreak data not directly downloadable — needs manual collection")
    return {}


def fetch_features_dataset():
    """arXiv:2411.03343 — 35 attack methods, 10800 prompts."""
    print("\n── 3. 35-method dataset (arXiv:2411.03343) ──")
    candidates = [
        "https://raw.githubusercontent.com/jordan-taylor1/jailbreak-features/main/data/jailbreaks.json",
        "https://raw.githubusercontent.com/jordan-taylor1/jailbreak-features/main/data/all_attacks.json",
    ]
    dest = OUT_DIR / "features_35.json"
    for url in candidates:
        if download(url, dest):
            try:
                with open(dest) as f:
                    data = json.load(f)
                grouped = {}
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            method = item.get("attack", item.get("method", "unknown"))
                            prompt = item.get("prompt", item.get("text", ""))
                            if prompt:
                                grouped.setdefault(method, []).append(prompt)
                elif isinstance(data, dict):
                    grouped = {k: [str(x) for x in v] for k, v in data.items() if isinstance(v, list)}
                for k, v in grouped.items():
                    print(f"  {k}: {len(v)} prompts")
                return grouped
            except Exception as e:
                print(f"  ✗ Parse failed: {e}")
    print("  ✗ 35-method dataset not directly downloadable")
    return {}


def fetch_artprompt():
    """ArtPrompt (ASCII art-based) — bundled with code."""
    print("\n── 4. ArtPrompt (ASCII art) ──")
    base = "https://raw.githubusercontent.com/uw-nsl/ArtPrompt/main/Vitc/ArtPrompt"
    candidates = [
        f"{base}/data/AdvBench_ArtPrompt.csv",
        f"{base}/data/MaliciousInstruct_ArtPrompt.csv",
    ]
    out = []
    for url in candidates:
        dest = OUT_DIR / Path(url).name
        if download(url, dest):
            import csv
            try:
                with open(dest, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Look for 'prompt' or first non-id column
                        for k in ["prompt", "art_prompt", "attack_prompt", "Prompt"]:
                            if k in row and row[k]:
                                out.append(row[k])
                                break
            except Exception as e:
                print(f"  ✗ Parse failed: {e}")
    if out:
        print(f"  artprompt: {len(out)} prompts")
        return {"artprompt": out}
    return {}


def fetch_drattack():
    """DrAttack — prompt decomposition."""
    print("\n── 5. DrAttack ──")
    candidates = [
        "https://raw.githubusercontent.com/xirui-li/DrAttack/main/data/harmful_prompts.json",
        "https://raw.githubusercontent.com/xirui-li/DrAttack/main/data/dr_prompts.json",
    ]
    dest = OUT_DIR / "drattack.json"
    for url in candidates:
        if download(url, dest):
            try:
                with open(dest) as f:
                    data = json.load(f)
                prompts = []
                if isinstance(data, list):
                    for x in data:
                        if isinstance(x, str):
                            prompts.append(x)
                        elif isinstance(x, dict):
                            for k in ["prompt", "attack", "text"]:
                                if k in x:
                                    prompts.append(x[k])
                                    break
                elif isinstance(data, dict):
                    for v in data.values():
                        if isinstance(v, list):
                            prompts.extend([str(x) for x in v])
                if prompts:
                    print(f"  drattack_extra: {len(prompts)} prompts")
                    return {"drattack_extra": prompts}
            except Exception as e:
                print(f"  ✗ Parse failed: {e}")
    return {}


def main():
    print("=" * 60)
    print("  Downloading extra attack datasets")
    print("=" * 60)

    all_attacks = {}
    all_attacks.update(fetch_multijail())
    all_attacks.update(fetch_sequentialbreak())
    all_attacks.update(fetch_features_dataset())
    all_attacks.update(fetch_artprompt())
    all_attacks.update(fetch_drattack())

    if not all_attacks:
        print("\n⚠ No data successfully downloaded.")
        print("   Manually clone the repos and place data files in data/extra_attacks/")
        return

    # Filter: drop empty / very short / extremely long
    filtered = {}
    for method, prompts in all_attacks.items():
        clean = [p for p in prompts if isinstance(p, str) and 5 < len(p) < 5000]
        if clean:
            filtered[method] = clean

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in filtered.values())
    print(f"\n{'═'*60}")
    print(f"  SUMMARY")
    print(f"{'═'*60}")
    print(f"  Method                  | #prompts")
    print(f"  ────────────────────────┼─────────")
    for method, prompts in sorted(filtered.items()):
        print(f"  {method:<23} | {len(prompts):>8}")
    print(f"  ────────────────────────┼─────────")
    print(f"  {'TOTAL':<23} | {total:>8}")
    print(f"\n  Saved → {OUTPUT_FILE}")
    print(f"\nNext step: merge with llama3_attacks.json or use directly")
    print(f"  python -c \"import json; a=json.load(open('llama3_attacks.json')); b=json.load(open('{OUTPUT_FILE}')); a.update(b); json.dump(a, open('llama3_attacks_extended.json','w'))\"")


if __name__ == "__main__":
    main()
