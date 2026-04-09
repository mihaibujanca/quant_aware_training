# Cycle 379 Brief — Compression-Accuracy Pareto Curve: k x Adapter h Sweep (Item 387)

## Context
- c377: Float+adapter (h=64) PPL 38.94 on val; compressed+adapter (k=512, h=128) PPL 47.43.
- c378: Held-out test confirms: float+adapter 39.73, compressed+adapter 47.22, gap=7.49 PPL.
- Compressed blocks use far fewer FFN params: k=512 block = ~786K vs float block = ~4.72M.
- Missing question: on a total-params vs PPL plot, is any compressed config Pareto-optimal? Compression loses on PPL but wins on params — where do the tradeoff curves cross?

## Hypothesis
At least one compressed config (some k, some adapter h) achieves PPL within 2 of the float+adapter config at matched total params, OR within 2 PPL at <= 80% of its params.

## Design
Reduced sweep (1 seed, 24 configs total):
- **Compressed**: k in {256, 384, 512, 768, 1024} x adapter h in {0, 32, 64, 128} = 20 configs.
- **Float+adapter**: h in {0, 32, 64, 128} = 4 configs.
- All use blocks [2, 1, 10], same training setup as c378 (2000 steps, Adam lr=1e-4, val selection, test eval).
- Seed 42 only for initial sweep. If a Pareto-interesting point emerges, validate with seeds 137 and 2024.
- Output: JSON with all 24 results + Pareto frontier plot.

## Success Criteria
- **Confirmed**: at least one compressed config within 2 PPL of float+adapter at <= 80% total params.
- **Refuted**: all compressed configs are Pareto-dominated by float+adapter.

## Key References
- c378 script: `experiments/gpt2_heldout_test_cycle378.py` (fork base)
- c378 results: `autoresearch/results/cycle_378_heldout_test.json`
- Agenda item 387
