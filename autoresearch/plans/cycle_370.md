# Cycle 370 Plan — Selective Compression (b10+b11 Uncompressed)

**Item**: 378
**Hypothesis**: The composition barrier is driven by b10/b11 quantization error. Skipping these two blocks while compressing all other 10 should yield PPL close to float.

## Pre-flight

- **P1 (duplicate)**: No prior plan tests this exact configuration (leave b10+b11 uncompressed, compress rest). Prior "selective" plans (cycles 97, 123, 169, etc.) address different block subsets, correction mechanisms, or activation selection — not selective skip of the two highest-error blocks.
- **P2 (dependencies)**: c366 CONFIRMED (b10+b11 = 77% local FFN error, hockey-stick explosion). c367 data usable (b11 2nd most important, b10 7th). Both available.
- **P3 (concrete)**: Yes — block set, k, adapter config, training steps, seeds, thresholds all specified.
- **P4 (program.md)**: No modifications needed.
- **P5 (null baseline)**: Float PPL ~53.89 is the primary comparison. 12-block all-compressed PPL 186.38 already known. No additional null baseline needed — the delta from float directly measures whether skipping b10+b11 breaks the barrier.
- **P6 (method alignment)**: Standard topk+lstsq+adapter recipe, same as all recent cycles. Batch compression from float model state (all 10 blocks at once), consistent with cycle_374 template.
- **P7 (statistical power)**: 3 seeds. Confirmed/refuted thresholds (55.0 / 60.0) are well-separated from float (53.89). Even with typical variance (~1-2 PPL across seeds), the thresholds give clear separation.

## Method

**Template**: `experiments/gpt2_adaptive_k_cycle374.py` (batch multi-block compression + joint adapter training).

**Changes from template**:
1. `ALL_BLOCKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` — 10 blocks, omitting 10 and 11.
2. `K = 512` uniform for all 10 blocks (no adaptive allocation — that's a separate variable).
3. `N_STEPS = 2500` — 10 blocks have ~67% more adapter params than 6-block (10 vs 6 adapters). Increase from 1500 to 2500 steps. Eval every 100 steps, best-checkpoint restore.
4. Remove adaptive/sqrt k logic — uniform k only.
5. Output path: `autoresearch/results/cycle_370_selective_compression.json`.
6. Report: effective compression ratio = (10 compressed blocks) / (12 total blocks) = 83.3% of FFN parameters compressed at k=512/3072 ratio. Uncompressed blocks retain full 3072 hidden dim.

**Config summary**:
- Blocks compressed: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
- Blocks at float: {10, 11}
- k = 512, adapter_hidden = 64
- N_CALIB = 200, N_EVAL = 100, SEQ_LEN = 128
- LR = 1e-4, N_STEPS = 2500, EVAL_EVERY = 100
- Seeds: [42, 137, 2024]
- Device: MPS

**Compression**: Batch — collect activations from float model for all 10 blocks simultaneously, run topk+lstsq, install compressed modules, then joint adapter training. This is identical to the 6-block recipe except with 10 blocks and b10/b11 untouched.

**Key implementation detail**: The model's `transformer.h[10].mlp` and `transformer.h[11].mlp` must NOT be touched — they remain as original `GPT2MLP` modules with all parameters frozen (requires_grad=False along with everything else non-adapter).

## Metrics

1. Per-seed final PPL (best checkpoint, restored)
2. Mean +/- std PPL across 3 seeds
3. Per-block reconstruction MSE and Frobenius error
4. Total trainable adapter params count (expect ~10 * (512*64 + 64 + 768*64 + 64) = ~816K)

## Verdict

- **CONFIRMED**: mean PPL < 55.0 — barrier broken, b10/b11 error was the bottleneck
- **REFUTED**: mean PPL > 60.0 — composition collapses even without late-block quantization; barrier is not b10/b11-driven
- **INCONCLUSIVE**: PPL in [55, 60] — partial improvement, b10/b11 contribute but are not the sole cause

## Script outline

```python
#!/usr/bin/env python3
"""Cycle 370 / Item 378 — Selective compression: compress blocks 0-9, leave b10+b11 at float.

Hypothesis: composition barrier is driven by b10/b11 quantization error (77% of
total local FFN error per c366). Skipping these two blocks should yield PPL near float.

Config: topk+lstsq+adapter (k=512, h=64) on blocks {0..9}. Blocks 10,11 at float32.
Joint adapter training, 2500 steps, 3 seeds.

Verdict:
  CONFIRMED  if mean PPL < 55.0
  REFUTED    if mean PPL > 60.0
  INCONCLUSIVE if PPL in [55, 60]
"""
# Based on experiments/gpt2_adaptive_k_cycle374.py
# Key changes:
#   - ALL_BLOCKS = [0,1,2,3,4,5,6,7,8,9]  (skip 10,11)
#   - Uniform k=512 for all 10 blocks
#   - N_STEPS = 2500 (more adapters to train)
#   - Remove adaptive/sqrt k logic
#   - Output: autoresearch/results/cycle_370_selective_compression.json

ALL_BLOCKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
K = 512
ADAPTER_HIDDEN = 64
N_STEPS = 2500
LR = 1e-4
SEEDS = [42, 137, 2024]
# ... (rest follows cycle_374 template exactly)
# compress_blocks() and install_compressed_blocks() operate only on ALL_BLOCKS
# blocks 10,11 are never touched — remain as original GPT2 MLP modules
# train_joint() trains adapter params for the 10 compressed blocks only
# Verdict logic: mean_ppl < 55 → CONFIRMED, > 60 → REFUTED
```

## Runtime estimate

- 10-block activation collection: ~2x of 6-block (~4 min)
- topk+lstsq for 10 blocks: ~2 min
- Joint training 2500 steps: ~25-30 min per seed (10 blocks = more backward passes through adapters)
- Total: ~100-110 min for 3 seeds

## Failure modes to watch

1. **Memory**: 10 blocks of activations in RAM simultaneously. At 200 seqs * 128 tokens * 3072 hidden * 4 bytes = ~300MB per block, 10 blocks = ~3GB. Should fit on MPS with 16GB+.
2. **Underfitting**: 2500 steps may not be enough for 10 adapters. If PPL is still decreasing at step 2500, the experiment is inconclusive due to insufficient training, not composition failure. The eval-every-100 + best-checkpoint pattern will show the training curve.
3. **Overfitting on calib set**: 200 calib seqs for training 816K params. Watch for calib loss decreasing while eval PPL increases.
