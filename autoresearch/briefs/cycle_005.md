# Item
- Item 10: Single-block swap localization curve

# Context
- Item 9 is now complete: exact-open target is nanochat + climbmix, but the current machine only has a runnable `gpt2` proxy path.
- `aleph/lm_geometric_compression.py` already supports `layer_ids` for geometric compression and the matched-budget distillation baseline.
- Existing proxy artifact shows the key failure mode: tiny local block MSE can coexist with catastrophic full-model degradation.

# Dependencies
- Item 9 met. Proxy execution path is explicit and locally runnable.
- No exact-open nanochat assets are present locally, so this cycle must use proxy `gpt2`.

# Key Files
- `aleph/lm_geometric_compression.py`
- `experiments/nanochat_ffn_compression.py`
- `experiments/nanochat_single_block_localization.py`
- `data/shakespeare.txt`
- `runs/nanochat_ffn_compression/gpt2_shakespeare_real.json`

# Constraints
- Measure both local block metrics and full-model metrics after each single-block replacement.
- Include geometric compression and matched-budget distillation.
- Stay proxy-labeled; do not overclaim exact-open conclusions.

# Handoff
- API check confirmed `run_ffn_geometric_compression(..., layer_ids=(layer_id,))` and `run_ffn_distillation_baseline(..., layer_ids=(layer_id,))` already work for single-block swaps.
- Smoke run of the new runner succeeded on `gpt2`, layer `0`, hidden fraction `0.25`, with both geometric and distilled methods producing valid records.
- Next action: launch the continuous sweep over depth-stratified or all-layer `gpt2` blocks and append results incrementally to `runs/nanochat_single_block_localization_overnight/`.
