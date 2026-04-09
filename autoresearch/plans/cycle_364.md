# Plan: Item 373 — Port compression recipe to nanochat teacher

**Status: PLAN INVALID**

## Reason: P2 (Missing Assets)

Item 373 requires nanochat teacher checkpoint to be available locally. However:

1. **No nanochat checkpoints exist locally** (confirmed cycle 004, audit cycle 146)
   - Expected paths: `/Users/mihai/.cache/nanochat/base_checkpoints/{d24,d26}` — empty
   - Local search: no .pt, .ckpt, or HF model files for nanochat

2. **Existing nanochat runs use gpt2 proxy** (not actual nanochat model)
   - `runs/nanochat_ffn_compression/gpt2_shakespeare_real.json` confirms proxy pattern
   - Confirms that current machine only has runnable gpt2 path

3. **Nanochat assets require external acquisition**
   - Nanochat d24/d26 checkpoints not in environment scope
   - Climbmix dataset similarly unavailable
   - No documented procedure for on-demand download in current setup

## Blockers
- External checkpoint dependency (nanochat repo or HF hub)
- Dataset dependency (climbmix-400b-shuffle)
- No local asset path; cannot proceed without upstream acquisition step

## Recommendation
Defer pending nanochat asset acquisition task. Current program uses gpt2 proxy for all compression validation (cycle 363, 370 confirm recipe on available assets). If nanochat validation becomes priority, first item must be: obtain official nanochat d24/d26 checkpoint and climbmix dataset locally.
