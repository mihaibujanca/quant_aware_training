# Cycle 141 Plan: Graduated Importance Pruning (Item 153)

## Task
Does importance-weighted channel pruning beat topk's activation-magnitude
selection? Rank channels by importance = ||W_down[:,j]||_2 * rms(GELU(z_j)),
prune bottom 5%/10%/20%, refit W_down via lstsq. Compare to topk at matched k.

## Hypothesis
Importance pruning selects channels that contribute most to the output (combining
weight magnitude and activation statistics), while topk selects by activation
magnitude alone on each input. If channel importance is structured (cycle 129:
isotropy is learned, not random), importance pruning should retain higher-value
channels than topk. Null hypothesis: topk's per-input adaptivity compensates for
its lack of global importance information; post-polish PPL is indistinguishable.

## Pre-flight
- P1 (duplicate): no prior cycle ran item 153.
- P2 (deps): cycle 128 completed (census). No hard blockers.
- P3 (scope): 3 pruning levels x 4 blocks x 2 methods x 3 seeds = 72 single-block
  evals + 3 pruning levels x 2 methods x 3 seeds = 18 stacked evals. Each eval is
  a forward pass (~5s). lstsq refit ~2s per block. Polish: 500 steps x 6 configs
  x 3 seeds = ~10 min. Total ~15-20 min.
- P4 (null baseline): topk k=4 single-block PPL known from prior cycles.
  Float PPL = 57.65. Abort if float PPL > 60 (pipeline broken).
- P5 (param accounting): both methods use the same number of active channels at
  each pruning level (matched k). The only difference is WHICH channels are kept.
  No param confound.

## Methods

### Importance computation (once per block)
For each block b in [1, 4, 7, 11]:
1. Collect pre-GELU activations z_j for 2048 probe tokens (32 random OWT sequences).
2. Compute per-channel importance: imp_j = ||W_down[:,j]||_2 * rms(GELU(z_j)).
3. Rank channels by importance (descending). Store ranking.

### Condition 1: importance pruning
For each pruning level p in [5%, 10%, 20%] (keep k = 2918, 2765, 2458):
1. Keep top-k channels by importance ranking.
2. Refit W_down_compressed via lstsq: minimize ||Y_float - X_kept @ W_down_kept||.
   X_kept = GELU(z)[:, kept_channels] on 2048 probe tokens.
3. Measure single-block swap PPL (pre-polish).

### Condition 2: topk baseline (matched k)
For each pruning level (k = 2918, 2765, 2458):
1. Use derive_ffn_compression with topk at matched k.
2. lstsq refit on same 2048 probe tokens.
3. Measure single-block swap PPL (pre-polish).

### Polish (both conditions)
For each condition at each pruning level:
1. 500-step Adam polish (CE loss, lr=1e-4, batch_size=4).
2. Measure single-block swap PPL (post-polish).

### Stacked evaluation
At the 20% pruning level (k=2458, most aggressive):
1. Stack all 4 blocks compressed with importance pruning.
2. Stack all 4 blocks compressed with topk.
3. Measure full-model PPL pre-polish and post-polish (joint 500-step polish).

## Steps
1. Create `experiments/importance_pruning_k4_cycle141.py`.
2. Load GPT-2 + tokenizer. Verify float PPL = 57.65 +/- 0.3. Abort if not.
3. Compute importance rankings for blocks [1, 4, 7, 11].
4. Run single-block evaluations for both conditions, 3 pruning levels, 3 seeds.
5. Run 500-step polish for each. Record pre-polish AND post-polish PPL.
6. Run stacked evaluation at 20% pruning level.
7. Save results JSON + comparison table. Print summary.

## Output Format
```
importance_pruning_results = {
    "float_ppl": float,
    "single_block": {
        block_id: {
            "prune_5pct": {"importance": {"pre": ..., "post": ...},
                           "topk": {"pre": ..., "post": ...}},
            "prune_10pct": {...},
            "prune_20pct": {...},
        }
    },
    "stacked_20pct": {
        "importance": {"pre": ..., "post": ...},
        "topk": {"pre": ..., "post": ...}
    },
    "seeds": [42, 137, 271]
}
```

## Abort / Pivot
- Abort if float PPL > 60 (pipeline issue).
- Abort if importance computation takes > 3 min per block (unexpected).
- If pre-polish importance == topk within noise at all levels, skip polish and
  report REFUTED early (don't waste 10 min on polish that will also match).
