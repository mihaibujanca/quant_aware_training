# Cycle 221 Audit — Item 226

**Status: PASS**
**Old outcome: inconclusive**
**New outcome: inconclusive**

## Verdict check
Criteria: Confirmed ≤58.0, Refuted >62.0, else Inconclusive.
best_mean = 59.515 ± 0.094 → strictly inside (58.0, 62.0] → INCONCLUSIVE confirmed.

## Quality checks
- G1 Implementation: BLOCKS=[1,2,4,5,7,8], K=512, N_CALIB=200, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100 — all match cycle 211 setup. Best-state snapshot stores all 6 (W_out, b_out) tensors on CPU; restore + assert `abs(restored - best_ppl) < 1e-3` enforces correctness. 3 seeds {42, 137, 2024} as planned.
- G2 Reproducibility: 3 seeds, std 0.094 PPL on best, std 170 on best step. Init PPL identical to 7 decimals across seeds (deterministic topk+lstsq, as expected). Strong reproducibility.
- G3 Interpretation: best_mean 59.515 vs cycle 211 single-seed 59.680 = -0.165 PPL. The framing "6-block ceiling at ~59.5 PPL" is correct — formal early stopping does not break the ceiling.
- G4 Counterintuitive: Not surprising; cycle 211 already foreshadowed this.
- G5 Learning: Manifest's `learned` line clearly states the 59.5-PPL ceiling and the 0.094 std. Good. `next` correctly points to W_fc retraining / LoRA / item 228 K-sweep.
- G6 Method failure vs genuine: Genuine inconclusive — the method works (overfitting trajectory matches cycle 211: ~70-72 PPL at step 4000), early stopping caught the optimum (best_step ~1300-1600), and the metric reproducibly lands in the inconclusive band across seeds.

## Cross-cycle consistency
- Cycle 206 (3-block joint) Adam: 46.556 PPL.
- Cycle 211 (6-block, single seed, no early stop): 59.680 best, 72.250 final.
- Cycle 221 (6-block, 3 seeds, early stop): 59.515 ± 0.094 best, 71.054 ± 0.850 final.
The 13-PPL gap between 3-block and 6-block joint Adam is the structural finding — output-head Adam alone cannot dissolve the cascade beyond ~59.5 PPL.

## Recommendations for next
1. The cycle 215 sequential approach (3 blocks: 47 sequential vs 46.5 joint) showed sequential ≈ joint at 3 blocks, so sequential is unlikely to help at 6 blocks for the same reason — the bottleneck is representational capacity of W_out/b_out, not optimization order.
2. Real progress requires expanding the trainable surface: W_fc retraining, LoRA on c_fc/c_proj, or higher K. Item 228 (lower K with Adam) probes the K dependence; pair it with a higher-K test to map the K → ceiling curve.
3. Consider an analytical baseline check: compute the rank of the residual error subspace at the 6 blocks. If it exceeds K=512, the ceiling is informational, not optimization-related.
