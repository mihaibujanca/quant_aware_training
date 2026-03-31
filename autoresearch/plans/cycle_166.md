# Cycle 166 Plan: Constant Negative Bias (Item 176)

## Pre-flight
- P1 (duplicate): no prior cycle tested ReLU + b_leak replacement. Cycle 164 killed positives; cycle 165 decomposed z_mean. This tests the complementary direction.
- P2 (deps): none. Standalone interpretability item.
- P3 (scope): 12 per-block + 1 full-model eval. ~10 min compute.
- P4 (program.md): no modifications needed.

## Task
Test if the GELU negative regime functions as a constant bias: replace GELU→ReLU (kills negative leak) and add b_leak = E[y_neg] as a fixed constant per block. Compare PPL against float baseline.

## Hypothesis
The negative regime is 87-98% token-invariant (cycle 165 brief context). If this invariance means the remaining 2-13% token-dependent signal is dispensable, then ReLU + b_leak recovers near-float PPL — the negative regime is absorbable as 768 bias params/block.

## Steps
1. Load GPT-2, tokenizer, 200 OWT sequences (seed 42). Verify baseline PPL in [15, 65].
2. Collect b_leak per block: run 4096 tokens through float model, hook each mlp.act to record y_neg = gelu(z) where z <= 0. b_leak[i] = mean over tokens and batch, shape (3072,).
3. Implement ReLUPlusBiasWrapper: forward(z) = ReLU(z) + b_leak (broadcast over batch/seq).
4. Per-block eval: for each block i, replace only block i's activation, eval PPL. Compute delta_pct = (ppl_mod - ppl_base) / ppl_base * 100.
5. Full-model eval: replace ALL 12 blocks simultaneously, eval PPL.
6. Save JSON + per-block bar plot.

## Validation
- Baseline PPL must be in [15, 65] (standard GPT-2 sanity).
- b_leak must have shape (3072,) per block and negative mean (consistent with cycle 165: z_mean ≈ -1 → gelu(z<0) is small negative).
- Per-block delta should be small (<5%) for most blocks if confirmed; large (>30%) if refuted.

## Criteria
- **Confirmed**: ReLU + b_leak within +5% PPL in ≥8/12 blocks.
- **Refuted**: >+30% PPL degradation.
- **Inconclusive**: +5-30%.

## Artifacts
- `experiments/constant_neg_bias_gpt2_cycle166.py`
- `autoresearch/results/cycle_166_neg_bias.json`
- `autoresearch/results/cycle_166_neg_bias.png`

## Fallbacks
- If b_leak collection OOMs: reduce to 2048 tokens.
- If per-block deltas are bimodal (some <5%, some >30%): report block-by-block, note which blocks are absorbable.
