"""
step4_massive_fpr_telemetry.py
==============================
Shadow 3: High-Throughput Benign Telemetry Extractor

Purpose:
    Processes tens of thousands of benign prompts to establish an
    irrefutable, large-scale False Positive Rate (FPR) baseline.
    
Optimization:
    Uses Prefill-Only Inference (no text generation) to process 
    prompts in milliseconds instead of seconds.
"""

import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =============================================================================
# 0. CONFIGURATION & CACHE (Must be set before importing HF)
# =============================================================================
MODEL_NAME = "lmsys/vicuna-13b-v1.5"
NUM_SAMPLES = 10000  # Change this to 50000 for the final paper run!
OUTPUT_FILE = "shadow3_massive_benign_telemetry.csv"

# Force lab cache to avoid home directory storage crash
lab_cache = "/mnt/lab/Mo/cache/"
os.makedirs(lab_cache, exist_ok=True)
os.environ["HF_HOME"] = lab_cache
os.environ["HF_DATASETS_CACHE"] = lab_cache
os.environ["TRANSFORMERS_CACHE"] = lab_cache

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset

# =============================================================================
# 1. THE PHYSICS MATH
# =============================================================================

def compute_kappa_signal(states: list, band_start: int, band_end: int, top_k: int) -> float:
    kappas = []
    for l in range(band_start, band_end):
        if l == 0 or l >= len(states) - 1: continue
        h_prev, h_curr, h_next = states[l-1], states[l], states[l+1]
        
        u, v, w = h_curr - h_prev, h_next - h_curr, h_next - h_prev
        a, b, c = torch.norm(u).item(), torch.norm(v).item(), torch.norm(w).item()
        s = (a + b + c) / 2.0
        area = max(s * (s - a) * (s - b) * (s - c), 0.0) ** 0.5
        denom = a * b * c
        kappas.append((4.0 * area) / denom if denom > 1e-8 else 0.0)
    return float(np.mean(sorted(kappas, reverse=True)[:min(top_k, len(kappas))])) if kappas else 0.0

def compute_cldp_depth_profile(states: list, band_start: int, band_end: int, top_k: int) -> float:
    layer_distances = [(l, 1.0 - F.cosine_similarity(states[l].unsqueeze(0), states[l+1].unsqueeze(0)).item()) 
                       for l in range(len(states) - 1)]
    band_dists = [d for l, d in layer_distances if band_start <= l < band_end]
    return float(np.mean(sorted(band_dists, reverse=True)[:min(top_k, len(band_dists))])) if band_dists else 0.0

# =============================================================================
# 2. MASSIVE EXTRACTION LOOP
# =============================================================================

def main():
    print(f"\n[System] Booting {MODEL_NAME} into DGX VRAM...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    m_config = AutoConfig.from_pretrained(MODEL_NAME)
    
    # We load the model. We don't need our custom hooks here because HF's
    # output_hidden_states=True is perfectly fine for prefill-only extraction.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, config=m_config, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    
    print("\n[Data] Downloading/Loading massive Alpaca dataset...")
    # Alpaca contains 52k diverse instructions
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.shuffle(seed=42).select(range(NUM_SAMPLES))
    
    print(f"\n[Extraction] Processing {NUM_SAMPLES} prompts...")
    
    # Open CSV in append mode so if the script crashes, you don't lose data
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w") as f:
            f.write("prompt_id,split,label,kappa_signal,cldp_k1_band_peak\n")

    for idx, item in enumerate(tqdm(dataset, desc="Extracting Topology")):
        # Construct the Vicuna prompt
        prompt_text = item['instruction']
        if item.get('input', ""):
            prompt_text += f"\n{item['input']}"
            
        full_prompt = f"USER: {prompt_text}\nASSISTANT:"
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            # output_hidden_states=True gives us a tuple of all layer states
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            
        # Extract the hidden states for the LAST token of the prompt (the decision point)
        all_hidden_states = outputs.hidden_states
        L = len(all_hidden_states)
        
        # Format states for our math functions: list of 1D tensors
        states = [all_hidden_states[l][0, -1, :].float().cpu() for l in range(L)]
        
        band_start, band_end = int(0.35 * L), int(0.65 * L)
        
        kappa = compute_kappa_signal(states, band_start, band_end, top_k=3)
        cldp = compute_cldp_depth_profile(states, band_start, band_end, top_k=3)
        
        # Write directly to disk
        with open(OUTPUT_FILE, "a") as f:
            # We label these all as 'benign_massive' and label=0
            f.write(f"{idx},benign_massive,0,{kappa:.6f},{cldp:.6f}\n")

    print(f"\n[Success] Extracted {NUM_SAMPLES} benign trajectories to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()