# Cycle 374 Brief — Error Budget Test (Item 382)

## Context
- **Working recipe**: topk (k=512 of 3072) + lstsq + MLP adapter (h=64). 3-block PPL: 47.43, 6-block PPL: 65.54.
- **Composition barrier**: going from 3→6 blocks is catastrophic (~18 PPL jump) despite spectrally flat per-block error.

## Hypothesis
Composition barrier is error-budget-driven. At k=512 (6x compression), 3 blocks work, 6 fail. If per-block error is the bottleneck, k=768 (4x compression, 33% less error) should extend the working frontier past 3 blocks.

## Task
- Run 6-block topk+lstsq (k=768) + MLP adapter (h=64) on [b2, b1, b10, b4, b3, b11], joint training 2000 steps, 3 seeds.
- Also run 3-block k=768 as sanity check (should beat k=512's 47.43).

## Success Criteria
- **Confirmed**: 6-block PPL < 55.0 (breaks barrier)
- **Refuted**: PPL > 60.0 (barrier persists)
- **Inconclusive**: 55–60 range

## Key References
- Baseline float: 53.89
- 3-block k=512: 47.43
- 6-block k=512: 65.54
