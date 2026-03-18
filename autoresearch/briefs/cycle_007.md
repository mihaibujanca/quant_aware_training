# Item
12 — Progressive multi-block accumulation curve

# Context
- Teacher ppl=122.350 on proxy gpt2+Shakespeare
- Cycle 005 per-layer damage at 16x (hf=0.0625): layer 0 worst (ppl=162.9, +33%), layer 7 best (ppl=121.6, -0.6%)
- Ranking easiest→hardest by 16x ppl: 7,5,1,4,3,6,2,8,9,10,11,0

# Dependencies
- Item 10 (cycle 005): DONE, confirmed — single-block swap data available in `runs/nanochat_single_block_localization_overnight_main/results.jsonl`

# Key Files
- `aleph/lm_geometric_compression.py` — `run_ffn_geometric_compression()`, `swap_compressed_ffns()`
- `experiments/nanochat_ffn_compression.py` — existing single-block experiment script

# Constraints
- Proxy-only (gpt2+Shakespeare); label clearly
- Use 16x budget (hf=0.0625) for accumulation to stress-test
- Run Python: `MPLCONFIGDIR=.mplconfig .venv/bin/python`
- 30-minute cap; 12 layers × 3 orderings × eval = must be efficient

# Handoff
Write `experiments/progressive_multiblock_accumulation.py`. Progressively compress 1→12 layers using easiest-first, hardest-first, random orderings. Measure full-model ppl at each step. Plot 3 curves. Evaluate against agenda criteria.
