# Item
- Item 373: Port compression recipe to nanochat teacher model — end-to-end validation

# Context
Objective: validate topk+lstsq+adapter recipe (k=512, h=128) on nanochat teacher. Requirements: compressed nanochat PPL within 1.2× float baseline, task accuracy within 5% relative.

# Assessment
**PLAN INVALID (P2: missing assets)**

Item requires nanochat teacher checkpoint to exist locally. Audit findings:
- Cycle 004 (item 9) established: no nanochat d24/d26 checkpoints exist locally
- `/Users/mihai/.cache/nanochat/base_checkpoints/` does not contain required model files
- Existing nanochat runs use gpt2 as proxy teacher: `runs/nanochat_ffn_compression/gpt2_shakespeare_real.json`
- Nanochat architecture code exists (`transformers/models/nanochat/modeling_nanochat.py`) but no trained checkpoint is available
- Nanochat checkpoint acquisition requires external download from official repo or HF hub (no local availability path)

Asset blockers:
1. No nanochat teacher checkpoint (.pt, .ckpt, or HF model ID) available locally
2. Dataset (`climbmix-400b-shuffle`) not available locally
3. Item depends on external nanochat assets not provided by current environment

# Recommendation
Defer to cycle requiring nanochat asset acquisition first (not currently in scope). Current program uses gpt2 proxy for all compression validation (cycle 363 kd_polish confirms recipe, cycle 370 measures margins). Nanochat validation requires separate upstream task: obtain official nanochat d24/d26 checkpoint and climbmix dataset locally.
