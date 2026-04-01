# Cycle 186 Brief — Item 195: Fewer-block compression — GPT-2 FFN k=4 on 2 blocks vs 4 blocks

## Hypothesis
Block count is a primary cost driver in cascade error (cycle 171 result: 12@3x=183 PPL vs 4@6x=59 PPL). Compressing fewer blocks with the same k=4 compression budget will reduce cascade depth and achieve better PPL/compression tradeoff than spreading the same compression across 4 blocks.

## Method
1. Use the working recipe: topk (k=4) + lstsq + 500-step Adam polish + LoRA r=8 corrections
2. Test all 6 possible 2-block pairs from blocks {1, 4, 7, 11}:
   - Pairs: (1,4), (1,7), (1,11), (4,7), (4,11), (7,11)
3. For each pair, apply topk+lstsq+polish+LoRA corrections as in cycle 182 recipe
4. Evaluate on OWT probe corpus (same as 4-block baseline)
5. Record PPL for each pair and select the best
6. Compare best 2-block PPL against 4-block baseline (PPL ~59)

## Success criteria
- **Confirmed**: best 2-block pair achieves PPL < 42
- **Refuted**: all 2-block pairs give PPL > 48
- **Inconclusive**: 42–48 PPL range

## Expected outcome
The hypothesis predicts that reducing cascade depth (2 vs 4 blocks) will lower PPL, likely achieving < 45 PPL on the best pair. The rationale: cascade error compounds multiplicatively through block boundaries; halving block count should significantly reduce this propagation. However, single-block pairs (1 alone, 4 alone, etc.) may be too constrained by hidden_dim=512 expansion, so 2-block combinations that span early–middle–late layers (e.g., (1,11)) may perform best.

## Runtime estimate
~2–3 hours (6 pairs × ~20–25 min per eval). Sequential execution of pairs on single GPU sufficient.
