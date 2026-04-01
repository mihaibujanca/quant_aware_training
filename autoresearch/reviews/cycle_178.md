# Cycle 178 — Self-Review

## R1: Does the plan test the agenda item's hypothesis?
YES. Item 188 asks: can we absorb the constant negative bias, then compress only positive-regime channels? The plan directly measures this against standard topk+lstsq and computes the absorption cascade cost.

## R2: Are the confirmed/refuted criteria falsifiable?
YES. Confirmed requires PPL <= 68 (a concrete number) AND a 2-PPL improvement over standard topk. Refuted has two clear failure modes: absorption cascade too costly (>5% over baseline) or positive-only worse than full topk.

## R3: Is the comparison fair?
CONCERN: The positive-only path absorbs b_leak into all 12 blocks before compressing 4 blocks. The standard topk control compresses 4 blocks but leaves 8 blocks as float GELU. This means the positive-only path starts from a degraded baseline (~42 PPL from cycle 166 all-block absorption). This is an unfair comparison.

**Fix:** For the positive-only condition, only absorb b_leak in the 4 compressed blocks [1,4,7,11]. The remaining 8 blocks stay float GELU. This makes the starting point comparable: both conditions modify exactly 4 blocks.

## R4: Could the result be an artifact?
CONCERN: If n_positive_channels < 512 in some blocks, the positive-only path uses fewer channels than standard topk. That's a param-count difference, not a method difference. We must report param counts and ideally match them.

**Fix:** Report per-block n_positive and the effective k used. If n_positive < 512, the positive-only path uses fewer params — note this as a confound or increase k to match.

## R5: Is the experiment scoped correctly?
YES. 3 seeds, 200 eval sequences, 4 blocks, clear metrics. Runtime should be ~15-20 minutes. No scope creep.

## Action items from review
1. Only absorb b_leak in the 4 compressed blocks, not all 12 (fixes R3)
2. Report n_positive_channels per block and param counts for both conditions
3. If n_positive < 512 in any block, note the param asymmetry
