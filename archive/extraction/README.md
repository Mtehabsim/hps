# extraction — activation-cache extraction scripts

Scripts that forwarded prompts through Llama-3/Vicuna and wrote the activation caches the analyses
consume (the `*.npz` in `src/results/`). Superseded/archived now that the canonical caches exist.

Includes: `extract_all_layers.py`, `extract_diverse_benign_activations.py`,
`extract_jbshield_activations.py`, `extract_generation_activations.py`,
`extract_llama3_base_activations.py`, `extract_token_curvature.py`, plus cache fixers
(`fix_cache_max_length.py`, `filter_diverse_cache.py`). Kept for reproducibility of the caches.
