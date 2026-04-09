# Cycle 343 Brief — Item 352

## Question
GPT-2 multi-block LASER accumulation curve — do compressed blocks compose?

## Prior art
- Cycle 280: b1 rank-2 LASER PPL 49.03 (float 53.90, -4.87)
- Cycle 285: b2 rank-2 LASER PPL 49.33
- Cycle 289: b10 rank-5 LASER PPL 51.47
- Cycle 270: b11 rank-22 LASER PPL 54.47
- Cycle 305: 5-block all-at-once LASER PPL 73.01 — REFUTED (composition destroyed quality)
- Cycle 341: 6/12 blocks LASER-viable by output-cov concentration

## What's new
Cycle 305 tried all 5 blocks at once with ranks from prior single-block experiments.
This cycle builds an **accumulation curve**: start with the strongest block, add one at a time
(ordered by output-covariance concentration), retrain jointly after each addition.

Block ordering (ascending eff_rank = most concentrated first):
  b2 (1.05) → b1 (1.94) → b10 (16.70) → b11 (22.49)

Ranks = ceil(output-covariance eff_rank):
  b2: rank 2, b1: rank 2, b10: rank 17, b11: rank 23

## Verdict criteria
- CONFIRMED: 2+ blocks compose with PPL <= 54.90 (float + 1.0)
- REFUTED: single-block is best; adding a 2nd block always degrades
- INCONCLUSIVE: otherwise
