# Item
- Item 9: Benchmark lock-in and asset audit

# Context
- Goal: document what benchmark is actually runnable now for transformer FFN compression.
- Local code path exists in `aleph/lm_geometric_compression.py` and `experiments/nanochat_ffn_compression.py`.
- Existing proxy artifact exists at `runs/nanochat_ffn_compression/gpt2_shakespeare_real.json`.
- `docs/nanochat_ffn_compression.md` already describes both `hf_pretrained` and `nanochat_local` paths.

# Dependencies
- None. Item 9 is the prerequisite for items 10 and 11.
- PAST confirms transformer-first direction and exact pruning literature relevance.

# Key Files
- `aleph/lm_geometric_compression.py`
- `experiments/nanochat_ffn_compression.py`
- `docs/nanochat_ffn_compression.md`
- `runs/nanochat_ffn_compression/gpt2_shakespeare_real.json`
- Official nanochat sources: repo README, `nanochat/checkpoint_manager.py`, `nanochat/dataset.py`

# Constraints
- Must distinguish exact-open benchmark vs proxy benchmark vs smoke/debug.
- Must audit local availability, not assume nanochat assets exist.
- Must provide concrete local paths or public acquisition procedure.

# Handoff
- Local audit found no nearby `nanochat` checkout, no `climbmix` parquet dir, and no local `d24`/`d26` checkpoints under `/Users/mihai` (searched to depth 4).
- Hugging Face cache contains `models--gpt2`, so proxy teacher is locally cached.
- Official nanochat code uses `~/.cache/nanochat/base_checkpoints/{model_tag}` and `~/.cache/nanochat/base_data_climbmix`; dataset code can download shards from HF on demand, and repo docs give concrete `d24`/`d26` training paths.
