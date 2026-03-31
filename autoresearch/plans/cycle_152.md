# Cycle 152 Plan: Progressive Stacking — Train-then-Freeze (Item 162)

## Task
Train compressed blocks one-at-a-time in upstream-to-downstream order. Each block trains under stacked CE loss with all previously compressed blocks already frozen and swapped in. Compare against: (a) joint training (cycle 149: 69.76 random init), (b) independent training (cycle 138: 74.4 stacked), (c) topk+lstsq+polish baseline (70.7).

## Hypothesis
Progressive training lets each downstream block adapt to the compression error introduced by upstream blocks. This should beat independent training (74.4) because each block sees the actual compressed upstream, and may approach joint training (69.76) by giving each block focused optimization time.

## Pre-flight
- P1 (duplicate): not a duplicate. Joint training (cycle 149) trains all blocks simultaneously. Independent training (cycle 138) trains each block in isolation. Progressive stacking is the intermediate protocol — each block trains under stacked context.
- P2 (deps): none.
- P3 (scope): 1 seed, ~30-40 min. 4 sequential phases of 500 steps.
- P4 (null baseline): float PPL ~57.65. Topk baseline ~70.7. Independent stacking ~74.4. If progressive PPL > 80, training is broken. If progressive PPL < 65, something is suspiciously good — verify.
- P5 (param accounting): 4 blocks × (encoder 3072×64 + decoder 64×768 + bias 768) = 4 × 246,528 = 986,112 params. Same as cycle 149. Assert at init.

## Method

### Step 0: Setup
Fork `experiments/joint_stacking_training_cycle149.py` to `experiments/progressive_stacking_cycle152.py`.

Key changes from cycle 149:
1. BLOCKS order: `[1, 4, 7, 11]` (upstream to downstream, NOT the `[7, 5, 4, 1]` from cycle 149).
2. Replace single `train_joint()` call with 4 sequential `train_single_block()` calls.
3. Use `init_mode="random"` (not SVD — cycle 149 showed SVD init gets stuck at 78.2).
4. Cosine schedule restarts per phase (each phase gets its own 500-step cosine).

### Step 1: Load model, eval float PPL
- Load GPT-2 from HF. Eval float PPL on 200 OWT sequences. Abort if > 60.
- Tokenize probe corpus (same as cycle 149).

### Step 2: Phase 1 — Compress block 1
- Replace block 1 MLP with `LinearFactorFFN(original_mlp, d_bn=64, init_mode="random")`.
- Freeze all model params except block 1's encoder + decoder.
- Train 500 steps, AdamW lr=1e-4, cosine schedule (T_max=500).
- Eval stacked PPL (only block 1 compressed). Record as `phase_1_ppl`.
- Freeze block 1's encoder + decoder permanently.

### Step 3: Phase 2 — Compress block 4
- Replace block 4 MLP with `LinearFactorFFN(original_mlp, d_bn=64, init_mode="random")`.
- Block 1 is already compressed and frozen. Only block 4's params are trainable.
- Train 500 steps, fresh AdamW + cosine schedule.
- Eval stacked PPL (blocks 1+4 compressed). Record as `phase_2_ppl`.
- Freeze block 4.

### Step 4: Phase 3 — Compress block 7
- Same protocol. Blocks 1+4 frozen, block 7 trainable.
- Eval stacked PPL (blocks 1+4+7 compressed). Record as `phase_3_ppl`.
- Freeze block 7.

### Step 5: Phase 4 — Compress block 11
- Same protocol. Blocks 1+4+7 frozen, block 11 trainable.
- Eval final stacked PPL (all 4 blocks compressed). Record as `phase_4_ppl` = final PPL.

### Step 6: Topk+lstsq+polish baseline
- Reuse cycle 149's topk baseline code. Blocks [1,4,7,11] at 6x, 500-step Adam polish.
- Eval stacked PPL. Should reproduce ~70.7.

### Step 7: Save results + verdict

Output JSON structure:
```json
{
  "cycle": 152,
  "item": 162,
  "seed": 42,
  "float_ppl": ...,
  "phases": {
    "phase_1": {"block": 1, "ppl": ..., "steps": 500},
    "phase_2": {"block": 4, "ppl": ..., "steps": 500},
    "phase_3": {"block": 7, "ppl": ..., "steps": 500},
    "phase_4": {"block": 11, "ppl": ..., "steps": 500}
  },
  "final_ppl": ...,
  "comparisons": {
    "joint_random_cycle149": 69.76,
    "independent_cycle138": 74.4,
    "topk_baseline": ...
  },
  "params": 986112,
  "verdict": "...",
  "verdict_detail": "..."
}
```

Verdict logic (from agenda item 162):
- **Confirmed** if progressive PPL beats independent stacking (74.4) by >= 3% relative → PPL <= 72.2.
- **Refuted** if progressive and independent within 1.5% → PPL >= 73.3.
- **Inconclusive** if 1.5-3% improvement (PPL 72.2-73.3).

## Implementation Notes

### `train_single_block()` function
Adaptation of cycle 149's `train_joint()`:
- Takes `block_id` parameter. Only that block's encoder/decoder params are optimized.
- Fresh AdamW optimizer and cosine scheduler per phase.
- Use the same RNG seed offset for reproducibility: `rng.manual_seed(SEED + phase_idx)`.
- Print loss every 100 steps (5 prints per phase).

### Block ordering
The agenda says blocks [1,4,7,11]. These are the same blocks as prior cycles but in UPSTREAM order. This is critical — block 1 must train first (earliest in the forward pass), then block 4 (sees compressed block 1), etc.

### Random init rationale
Cycle 149 showed random init (69.76) massively outperforms SVD init (78.2) under joint training. SVD init appears to trap the model in a local minimum near the original projection. Use random init here too.

## Output
- `experiments/progressive_stacking_cycle152.py`
- `autoresearch/results/cycle_152_progressive_stacking.json`

## Validation
- Float PPL: 57.65 ± 1. Abort if > 60.
- Phase 1 PPL (single block compressed): should be < 65 (single-block damage is small).
- Final PPL: expected range [68, 76]. If > 80, training failed. If < 66, suspiciously good.
- Topk baseline: should reproduce ~70.7 ± 1.
- Param count assertion: 986,112 total (4 × 246,528).
- Each phase's PPL should be weakly increasing (more blocks compressed = higher PPL), though downstream blocks may partially compensate.

## Fallbacks
- If phase 1 PPL > 70 (single block shouldn't cause this): check that only block 1 is compressed. Possible bug: wrong block index.
- If final PPL > 80: try 1000 steps per phase instead of 500 (double the budget, same total as joint training's 2000 steps but 4000 total).
- If final PPL is nearly identical to independent training (within 0.5 PPL): the progressive context isn't helping. The composition barrier may be fundamentally non-addressable by sequential adaptation.

## Wall time estimate
- Model load + float eval: ~2 min.
- 4 phases × (500 steps × ~1.5 sec/step + eval): ~4 × 14 min = ~28 min.
  (Actually faster — 500 steps with only 1 block's params should be ~0.8 sec/step → ~7 min/phase → ~28 min.)
- Topk baseline: ~5 min.
- Total: ~35 min.
