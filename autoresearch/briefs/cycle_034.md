# Item
49. Block-specific compression ratio sweep — GPT-2 FFN

# Context
Cycle 033 confirmed 4× compression viable on 6 blocks (PPL +8.84%). Item 49 asks: **can we do better per-block?** Sweep ratios {2×, 3×, 4×, 5×, 6×} independently per block; find max ratio keeping single-block PPL ≤ baseline × 1.05 (+5% threshold).

# Dependencies
- **Item 5** (cycle 005, CONFIRMED): single-block swap established 6 easiest blocks: [7, 5, 4, 1, 3, 6]. Depth-sensitive: block 0 hard, middle/late blocks viable at 8×–16× compression.
- **Item 26** (cycle 015, INCONCLUSIVE): intrinsic dimension increases through depth (no structural simplification signal).
- **Item 27** (cycle 027, CONFIRMED): centroid similarity ρ=0.853 is dominant merge predictor (vs weight cosine ρ=0.041). Jacobian signal fails.

# Key Files
- **Library**: `/aleph/lm_geometric_compression.py` — ProbeCorpusSpec, CompressionBudgetSpec, derive_ffn_compression, swap_compressed_ffns, _evaluate_model
- **Baseline**: `/experiments/multiblock_accumulation_6block.py` — current method: topk channel selection + lstsq refit at 4× fixed ratio
- **Raw data**: `/runs/nanochat_single_block_localization_overnight_main/results.jsonl` — cycle 005 damage profile

# Constraints
- Use Shakespeare proxy (no nanochat access documented)
- Fixed probe: 200 train seqs, 50 holdout seqs, 128 seq len
- Single-block swaps only (no stacking yet — that's for multi-block accumulation later)
- Topk selection + lstsq refit (proven in cycles 026–027)
- Centroid signal already validated; use for merge heuristic if needed
- Report per-block max ratio that satisfies 5% PPL threshold

# Handoff
Sweep compression ratios {2×, 3×, 4×, 5×, 6×} on blocks [7, 5, 4, 1, 3, 6]. Per block, measure full-model PPL at each ratio; find max ratio ≤ baseline × 1.05. Output: 6-row table (block, max_ratio, ppl@ratio). Test CONFIRMED if ≥4 blocks tolerate ≥5×. Script template: extend multiblock_accumulation_6block.py with nested ratio loop. No cross-block interaction (cycle 033 stacking is separate).
