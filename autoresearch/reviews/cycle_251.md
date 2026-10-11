# Cycle 251 Design Review — Item 261

## Verdict: APPROVE

## Summary

Trivial K swap on cycle 245's 5-block [1,2,4,7,8] template: K=512 → K=128. All other hyperparameters (BLOCKS, N_CALIB, SEQ_LEN, LR, N_STEPS, EVAL_EVERY, SEEDS, disjoint splits, best-state restore) unchanged. Isolates the K=128 penalty at the 5-block cascade-tolerant configuration.

## Scientific Soundness

- Clean single-variable change (K only); isolates channel-count effect from cycle 245's near-float baseline.
- Anchors are well-chosen: cycle 245 (53.826, 5-block K=512) and cycle 231 (67.562, 6-block K=128) bracket the expected range.
- Open question is genuine: does the K=128 penalty (+8 PPL at 6 blocks) scale with block count or with total capacity?

## Methodology

- Same disjoint calib/eval split (all_ids[:200] / all_ids[200:300]) as cycle 245 — directly comparable.
- 3 seeds + best-state restore consistent with recent cadence.
- topk+lstsq init → joint Adam on W_out+b_out: the working recipe.

## Thresholds

- CONFIRMED ≤56.0 (+2.2 PPL from cycle 245, smaller than 6-block K=128 penalty of +8).
- REFUTED ≥62.0 (penalty scales like 6-block case).
- 6-PPL inconclusive band is appropriate given noise and the uncertainty about how K-penalty interacts with block count.

## Risks

- INCONCLUSIVE band is plausible (~61 if penalty is roughly linear in blocks).
- No execution risk: 5-block K=128 footprint is strictly smaller than cycle 221.

## Bookkeeping

- New output paths (`cycle_251_*`) avoid collision with cycle 245.
- Cycle/item stamps present.

## Decision

APPROVE. Proceed to execution.
