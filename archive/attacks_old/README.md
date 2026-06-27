# attacks_old — earlier attack & data-build scripts

Attack implementations and dataset-construction scripts predating the current `src/obfuscated-activations/`
(obf_reps) framework and the `openset_*` experiments in `src/`.

Includes: attack drivers (`adaptive_attack.py`, `flrt_attack.py`, `gcg_specific_test.py`,
`evaluate_ood_attacks.py`), dataset builders (`build_diverse_benign.py`, `build_jbshield_attacks.py`,
`build_novel_attacks.py`, `download_*`), and jailbreak validation (`validate_attacks.py`,
`validate_benign.py`, `verify_jailbreak.py`, `hard_benign.py`). Kept for provenance.
