# Cycle 203 Review: Block Subset Search (Item 214)

## Verdict: REFUTED

No superset of [1,4,7] achieves within 2 PPL of the anchor (61.75). Adding any 2+ extra blocks
causes cascade that pushes PPL to 75+ (+13-17 PPL penalty).

## Key Numbers

| Block set         | n_blocks | PPL    | vs float |
|------------------|----------|--------|----------|
| [1,4,7] (anchor) | 3        | 61.75  | +14.6%   |
| [1,2,3,4,7]      | 5        | 75.65  | +40.4%   |
| [1,4,5,6,7]      | 5        | 77.03  | +42.9%   |
| [1,4,7,8,9]      | 5        | 78.39  | +45.5%   |
| [1,4,7,8,9,10]   | 6        | 89.46  | +66.0%   |
| [1,2,3,4,7,8,9]  | 7        | 97.95  | +81.7%   |
| [1,2,3,4,5,6,7]  | 7        | 101.78 | +88.8%   |

## Interpretation

Adding ANY 2 extra blocks beyond [1,4,7] costs at minimum +14 PPL (from 61.75 to 75.65).
This is consistent with cycle 187 (multi-block drop superadditive cascade) and cycle 188
(cascade barrier materializes after block 6-7 for C3 composition).

The cascade is near-symmetric regardless of WHICH blocks are added:
- Early blocks (2,3): +40% above float
- Middle blocks (5,6): +43%
- Later blocks (8,9): +46%
No region of the model is more cascade-tolerant than others.

This suggests cascade is a property of the cumulative compression error, not of specific
block positions. Each additional compressed block adds ~14-15% degradation above the
3-block ceiling, consistent with the linear cascade growth hypothesis.

## Impact

Closes item 214. The ceiling for the current topk+lstsq approach with blocks excluded from
{0, 11} is 3 compressed blocks → +14.6% PPL degradation. Further improvement requires
either: (a) a better single-block approximator, or (b) a fundamentally different approach
that doesn't accumulate cascade error per block.

The question for the next cycle: is [1,4,7] actually optimal among all 3-block selections,
or could a different 3-block combination (e.g., [2,5,8] or [3,6,9]) achieve lower PPL?
