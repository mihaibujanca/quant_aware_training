# Cycle 007: Progressive Multi-Block Accumulation Curve

## Verdict

Outcome: **confirmed (proxy)**

Easiest-first cumulative compression materially outperforms hardest-first ordering, meeting the ≥25% perplexity reduction threshold at k≥6 blocks and growing to 36% at k=9. Single-block localization from cycle 005 correctly identifies the safe compression order.

## Setup

- Teacher: `gpt2` (proxy), ppl=122.350 on Shakespeare
- Compression: 16x (hidden_fraction=0.0625), geometric method
- Orderings tested:
  - Easiest-first: [7, 5, 4, 1, 3, 6, 2, 9, 10, 8, 11, 0]
  - Hardest-first: [0, 11, 8, 10, 9, 2, 6, 3, 1, 4, 5, 7]
  - Random (seed=42): [7, 5, 2, 8, 9, 6, 11, 3, 4, 0, 1, 10]

## Key Results

| k | easiest_first | hardest_first | random | easy_vs_hard reduction |
|---|--------------|--------------|--------|----------------------|
| 1 | 119.8 (-2.1%) | 151.0 (+23.4%) | 119.8 (-2.1%) | 20.6% |
| 3 | 129.4 (+5.8%) | 166.0 (+35.6%) | 127.7 (+4.4%) | 22.0% |
| 6 | 166.7 (+36.2%) | 229.7 (+87.7%) | 168.9 (+38.1%) | **27.4%** |
| 9 | 222.2 (+81.7%) | 348.1 (+184.5%) | 227.0 (+85.5%) | **36.2%** |
| 12 | 403.5 (+229.8%) | 403.5 (+229.8%) | 403.5 (+229.8%) | 0.0% |

## Criteria Evaluation

- **Confirmed**: easiest-first outperforms hardest-first by ≥25% at k=6 (27.4%) and k=9 (36.2%). The advantage grows with the number of compressed blocks, not shrinks.
- NOT refuted: the three curves are clearly distinguishable — degradation is not nearly identical across orderings.
- NOT inconclusive: the ordering advantage persists and grows through k=9, not just at small k.

## Nuance: easiest-first vs random

Easiest-first barely outperforms random (1-2% at most). This is because a random ordering doesn't systematically pick the worst layers — and random happened to start with layer 7 (the easiest). The real insight is that hardest-first is catastrophically worse, meaning **avoiding the hard layers early is the critical decision**. Single-block localization is informative primarily as a signal of which blocks to compress last, not as a fine-grained optimizer of ordering.

## Interpretation

1. **Ordering matters enormously for accumulation**: 36% perplexity gap at 9/12 blocks compressed.
2. **Damage is super-additive for hard blocks**: layer 0 alone adds +23% ppl, and adding it early poisons all subsequent compressions.
3. **Single-block ranking predicts safe accumulation order**: the cycle 005 damage curve is a valid priority signal.
4. **All-block 16x compression fails regardless of order**: ppl=403 (3.3x teacher) at k=12 for all orderings.
5. **Sweet spot is k=4-5 blocks (easiest-first)**: ppl stays below 150 (< 25% degradation) when compressing the 4-5 easiest blocks.

## Caveats

- Proxy-only (gpt2+Shakespeare); does not answer the nanochat exact-open benchmark.
- Single random seed for the random ordering; a proper test would sample multiple random orderings.
- The geometric method may interact with ordering differently than distillation would.

## Artifacts

- `experiments/progressive_multiblock_accumulation.py`
- `autoresearch/results/cycle_007_accumulation.json`
- `autoresearch/results/cycle_007_accumulation.png`
