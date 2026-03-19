# Task
Run easiest-first multi-block accumulation (k=1..6) with per-block 6x compression on GPT-2 FFN.
Compare PPL curve against cycle 033's uniform 4x curve.

# Hypothesis
Per-block 6x ratios (all blocks tolerated 6x in cycle 035) compose safely at k=4, achieving <=8% relative PPL with 6x average compression (beating uniform 4x's +8.84%).

# Pre-flight Results
- P1: Not duplicate — 033 used uniform 4x, this uses 6x per-block.
- P2: Deps satisfied — item 48 CONFIRMED, item 52 INCONCLUSIVE (all blocks safe at 6x).
- P3: Concrete — accumulation sweep with clear thresholds.

# Steps
1. Copy `experiments/multiblock_accumulation_6block.py` to `experiments/multiblock_accumulation_6x.py`.
2. Change `HIDDEN_FRACTION = 0.25` to `HIDDEN_FRACTION = 1/6` (~0.1667, i.e. 6x compression).
3. Keep everything else identical: TARGET_BLOCKS=[7,5,4,1,3,6], topk, 200 train + 50 eval, seed=42.
4. Run the script. It produces accumulation PPL at k=1..6.
5. Load cycle 033 results from `autoresearch/results/cycle_033_accumulation.json`.
6. Plot both curves (uniform 4x vs heterogeneous 6x) on the same axes.
7. Save results to `autoresearch/results/cycle_036_accumulation_6x.json` with fields:
   baseline_ppl, per_k (list of {k, blocks, ppl, rel_pct}), uniform_4x_ref (from 033).
8. Save plot to `autoresearch/results/cycle_036_accumulation_6x.png`.

# Validation
- Baseline PPL must be within 1% of cycle 033 baseline (same model, same corpus size).
- k=1 relative PPL should be close to cycle 035's single-block 6x result for block 7 (+0.51%).
- If k=1 PPL diverges >3% from 035 single-block, investigate before continuing.

# Artifacts
- `experiments/multiblock_accumulation_6x.py`
- `autoresearch/results/cycle_036_accumulation_6x.json`
- `autoresearch/results/cycle_036_accumulation_6x.png`

# Fallbacks
- If k=4 PPL >12% (REFUTED): report super-additive breakdown at 6x. Record the k at which PPL crosses 8% and 12% thresholds.
- If 8-12% (INCONCLUSIVE): note that 6x single-block safety doesn't compose linearly; suggest binary search on per-block ratios between 4x and 6x.
