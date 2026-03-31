# Plan — Cycle 163 (Item 173)

## Goal
Profile per-block marginal PPL contribution under progressive compression, then test whether concentrating LoRA corrections on the highest-impact blocks recovers at least 1 PPL point vs uniform correction.

## Phase 1 — Stacking profile (13 evals × 3 seeds)
Compress blocks incrementally: for i=0..11, compress blocks {0..i}, evaluate PPL.
- Marginal contribution of block i = PPL(0..i) − PPL(0..i−1).
- 3 seeds (42, 137, 7). Report mean ± std.
- Unpolished model (post-lstsq, pre-corrections).

## Phase 2 — Correction concentration (key addition)
After profiling, take the top-3 blocks by marginal PPL contribution.

**Condition A — budget-matched (fewer blocks, same total params):**
Train LoRA r=8 corrections on ONLY those top-3 blocks. Total param budget ~37K (3 blocks × ~12.3K). Compare against the uniform baseline: LoRA r=8 on all 4 standard blocks [1,4,7,11] (~49K params, PPL ~59).

**Condition B — budget-matched (same total budget, higher rank):**
Distribute the full 49K param budget across only the top-3 blocks (LoRA r≈11 per block). Compare against uniform baseline.

Both conditions use the standard recipe: topk+lstsq k=4@6x compression + LoRA correction. 3 seeds each.

## Implementation — `experiments/per_block_stacking_profile_cycle163.py`

1. **Setup**: `load_nanochat_teacher`, `build_probe_corpus` (OWT), `probe_ffn_geometry` for all 12 layers. `CompressionBudgetSpec(hidden_fraction=1/6, selection_mode="topk")`.

2. **Phase 1 loop** — per seed: baseline PPL, then for i=0..11 build `CompressedFfnPack` for layers {0..i}, swap, evaluate PPL. Record marginals.

3. **Phase 2 loop** — per seed:
   - Identify top-3 blocks from Phase 1 marginals (use per-seed ranking).
   - Condition A: compress + LoRA r=8 on top-3 only → PPL.
   - Condition B: compress + LoRA r=11 on top-3 only → PPL.
   - Uniform baseline: compress + LoRA r=8 on [1,4,7,11] → PPL.

4. **Output**: `autoresearch/results/cycle_163_per_block_profile.json`
   - Fields: `baseline_ppl`, `per_block_marginals`, `top3_fraction`, `correction_results` (conditions A/B/uniform), `verdict`.

## Verdict logic
- **Confirmed**: top-3 blocks show Pareto structure (>60% of total degradation) AND either condition A or B improves PPL by ≥1 point over uniform correction baseline.
- **Refuted**: no Pareto structure (no block >15% of total), OR concentrated corrections perform ≥1 PPL worse than uniform.
- **Inconclusive**: Pareto structure exists but correction concentration neither helps nor hurts (within 1 PPL of uniform).

## Files touched
- NEW: `experiments/per_block_stacking_profile_cycle163.py`
- READ: `aleph/lm_geometric_compression.py` (imports only)
- OUT: `autoresearch/results/cycle_163_per_block_profile.json`
