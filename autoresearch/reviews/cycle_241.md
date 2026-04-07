# Cycle 241 Review — Item 251 (W_out + b_out + c_fc bias joint Adam, 6 blocks)

## Decision

**APPROVED**

## Summary

The revised plan correctly constrains the trainable surface to `W_out + b_out + c_fc BIAS` only, with `W_fc` explicitly frozen. Param count ~2.38M matches the agenda formula via interpretation (B) (full 3072-d `b_fc` as a Parameter, topk selection at forward time). Scaffolding reuses cycle 221 (BLOCKS=[1,2,4,5,7,8], K=512, N_CALIB=200, N_EVAL=100, SEQ_LEN=128, LR=1e-4, 4000 steps, eval-every-100, best-state restore, seeds {42, 137, 2024}).

## Strengths

- **W_fc freeze is explicit and load-bearing.** Plan calls out the cycle 213 script as a gotcha ("do NOT copy that block verbatim") and mandates `W_fc.requires_grad_(False)` with a runtime assert. This is the correct way to prevent accidental scope expansion to 4.7M params.
- **Param count assertion at startup** (`2,382,336`) catches any interpretation drift before training burns compute.
- **Interpretation (B) is faithful to the agenda arithmetic.** `397,056 × 6 = 2,382,336 ≈ 2.38M` matches the agenda claim exactly. The plan acknowledges that 2560 of the 3072 `b_fc` entries are dead (non-topk-selected) and receive zero gradient, but counts them per the agenda — this is honest bookkeeping.
- **Baselines are specified concretely**: cycle 221 W_out-only 6-block 59.515 ± 0.094 and cycle 213 W_out+c_fc-bias 3-block 46.424. The 3-block vs 6-block contrast is exactly the structural question the item asks.
- **Verdict thresholds** (`≤58.0` CONFIRMED / `[59.0, 60.5]` REFUTED / else INCONCLUSIVE) are tight enough to distinguish signal from noise given cycle 221 std 0.094.
- **Best-state restore with assert** (`|restored − best| < 1e-3`) is already a known-good pattern from cycle 221.
- **Init PPL sanity check**: init should match cycle 221 (same lstsq, pretrained `b_fc` init = frozen value cycle 221 used). Good trip-wire for implementation bugs.

## Concerns (minor)

- **Dead params in the count**: interpretation (B) counts 2560/3072 `b_fc` entries that cannot affect the loss (no gradient path through topk selection). This is cosmetic — the effective trainable surface is interpretation (A)'s 2.37M. Agenda arithmetic wins here, but worth noting in the write-up so future readers don't mistake 2.38M for effective surface.
- **b_fc gradient path**: the plan says "forward takes `b_fc[topk_idx]` at runtime" — confirm this indexing preserves gradient flow to the selected 512 entries. Using fancy indexing on an `nn.Parameter` in PyTorch does propagate gradients correctly, so this is fine, but an explicit test (one backward step, check `b_fc.grad[topk_idx].abs().sum() > 0` and `b_fc.grad[~topk_mask].abs().sum() == 0`) would be reassuring and cheap.
- **Runtime budget**: +18K effective params vs cycle 221's 2.36M — overhead should be <5%. Plan's ≤1.05× estimate is reasonable.

## Risks

- **If REFUTED**, this closes off the "c_fc bias unfreeze" direction at both 3 and 6 blocks. The 59.5 floor survives another attack.
- **If CONFIRMED**, the delta relative to cycle 213's 3-block noop would be informative: it would mean the 3-block surface is already flat w.r.t. bias, but the 6-block surface has enough residual curvature for bias training to matter. Either way the result is clean.

## Approval

Approved as written. Execute as planned.
