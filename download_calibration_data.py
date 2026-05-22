"""Download 100 harmful + 100 harmless calibration prompts for RTV/HPS comparison."""
import json
import random
import pandas as pd

random.seed(42)

# ── Harmless: Alpaca instructions ──
print("Loading Alpaca (harmless)...")
from datasets import load_dataset
ds = load_dataset("tatsu-lab/alpaca", split="train")
harmless = [row["instruction"].strip() for row in ds if len(row["instruction"].strip()) > 20]
random.shuffle(harmless)
harmless = harmless[:100]

# ── Harmful: HarmBench behaviors ──
print("Loading HarmBench (harmful)...")
try:
    df = pd.read_csv("https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv")
    harmful = df["Behavior"].dropna().tolist()
except:
    # Fallback: use AdvBench from JBShield if available
    print("  HarmBench download failed, trying JBShield harmful...")
    df = pd.read_csv("JBShield/data/harmful_calibration.csv")
    for col in ["prompt", "goal", "text", "instruction", "query"]:
        if col in df.columns:
            harmful = df[col].dropna().tolist(); break
    else:
        harmful = df.iloc[:, 0].dropna().tolist()

random.shuffle(harmful)
harmful = harmful[:100]

print(f"  Harmless: {len(harmless)}")
print(f"  Harmful:  {len(harmful)}")

# Save
pd.DataFrame({"prompt": harmless}).to_csv("data_harmless_100.csv", index=False)
pd.DataFrame({"prompt": harmful}).to_csv("data_harmful_100.csv", index=False)
print("Saved → data_harmless_100.csv, data_harmful_100.csv")
