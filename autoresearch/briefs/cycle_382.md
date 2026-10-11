# Cycle 382 Brief — Minimal-Penalty Compression: Least-Important Blocks [b1,b7,b2] + Adapter (Item 390)

## Context
- c367: block importance ranking [1,7,2,6,9,10,...] with deltas b1=+1.60, b7=+2.35, b2=+3.85 PPL.
- c378: 3-block [b2,b1,b10] compressed+adapter PPL 47.22 vs float+adapter PPL 39.73. Gap = 7.49 PPL.
- c378 used b10 (importance 4.58), not the optimal least-important set.
- Item 390 hypothesis: compressing only the 3 truly least-important blocks [b1,b7,b2] will yield a smaller penalty than 7.49 PPL, since b7 (2.35) replaces b10 (4.58).

## Hypothesis
Compressing [b1,b7,b2] + adapter yields a penalty (compressed+adapter minus float+adapter) < 3 PPL on held-out test. This would demonstrate viable selective compression on easy blocks.

## Design
Two configs, c378 protocol (train on wikitext-2-val, eval on wikitext-2-test):
- **Config A**: Float+adapter [b1,b7,b2], h=64, adapter params only trainable.
- **Config B**: Compressed+adapter [b1,b7,b2], topk+lstsq k=512, h=128, proj+adapter trainable.
- 3 seeds (42, 137, 2024). 2000 steps, Adam lr=1e-4. Best checkpoint by val eval, single test eval.

## Success Criteria
- **Confirmed**: penalty < 3 PPL (viable selective deployment).
- **Refuted**: penalty >= 5 PPL (even easiest blocks suffer).
- **Inconclusive**: penalty in [3, 5).

## Key References
- c367 block importance: `autoresearch/results/cycle_367_block_importance.json`
- c378 3-block baseline: `autoresearch/results/cycle_378_heldout_test.json`
- Template: `experiments/gpt2_heldout_test_cycle378.py`
