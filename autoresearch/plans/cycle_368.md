# Cycle 368 — Item 377: Mixed compression (importance-based k allocation)

## Verdict: PLAN INVALID — effective duplicate of refuted cycle 365

## Reasoning

Item 377 proposes allocating k per block based on importance ranking instead of Frobenius norms. On the test set [b2, b1, b10, b11, b7, b6], the two rankings produce nearly identical allocations:

| Block | Importance rank (asc) | Frobenius rank (asc) | Both methods |
|-------|----------------------|---------------------|--------------|
| b1    | 1 (lowest)           | low                 | starved      |
| b7    | 2                    | low-mid             | starved      |
| b2    | 3                    | low                 | starved      |
| b6    | 4                    | mid                 | mid k        |
| b10   | 5                    | high                | boosted      |
| b11   | 6 (highest)          | highest             | boosted      |

Both methods concentrate channels at b10/b11 and starve b1/b2/b7. This is because later GPT-2 blocks are both higher Frobenius error AND higher importance (removal cost), so the two signals correlate within this block set.

Cycle 365 showed this exact allocation pattern (starve early, boost late) produces PPL 67.63, worse than uniform k=512 at 66.34. The failure mechanism — early-block starvation propagates compound errors — applies identically regardless of whether the allocation signal is Frobenius norm or importance rank.

The item text's other proposal (zeroed FFN for easy blocks) was ruled out by cycle 367/376: no blocks are prunable below +0.5 PPL.

## What would be different

The only scenario where importance-based differs from Frobenius-based is if importance and Frobenius *diverge* — i.e., a block has high Frobenius error but low importance, or vice versa. Within the 6-block test set, they don't diverge meaningfully. For the full 12-block case, block 0 (highest importance, +3021 PPL) would get maximum k under importance-based but might not under Frobenius-based — but block 0 is not in the 6-block test set, and the 12-block case hasn't been reached yet.

## Recommendation

Mark item 377 as PLAN INVALID (effective duplicate of refuted 374/c365). The next productive direction should avoid per-block k reallocation entirely, since both available ranking signals (Frobenius, importance) produce correlated allocations that suffer from the same early-block starvation mechanism.
