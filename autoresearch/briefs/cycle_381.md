# Cycle 381 Brief — 6-Block Compressed+Adapter Composition Penalty (Item 389)

## Context
- c378 (3-block, test data): float+adapter PPL 39.73, compressed+adapter PPL 47.22. Gap = 7.49 PPL.
- c357/c361: 6-block compressed (no adapter) PPL ~65.54. Float baseline test PPL 56.52.
- No 6-block float+adapter or 6-block compressed+adapter measurements exist yet.
- Blocks: [2, 1, 10, 4, 3, 11] (the established 6-block set).

## Hypothesis
Composition penalty (compressed+adapter minus float+adapter) scales sublinearly from 3 to 6 blocks. 3-block penalty was 7.49 PPL; if 6-block penalty < 15, sublinear growth is confirmed.

## Design
Two configs, both trained on wikitext-2-val, evaluated on wikitext-2-test (c378 protocol):
- **Config A**: 6-block float+adapter (h=64, all GPT-2 weights frozen, adapter params only).
- **Config B**: 6-block compressed+adapter (topk+lstsq k=512, adapter h=128, proj+adapter trainable).
- 3 seeds (42, 137, 2024). 2000 steps, Adam lr=1e-4. Best checkpoint by val eval, single test eval.

## Success Criteria
- **Confirmed** (penalty < 15 PPL): sublinear composition penalty growth.
- **Refuted** (penalty > 25 PPL): superlinear, composition barrier persists.
- **Inconclusive**: penalty in [15, 25].

## Key References
- c378 results (3-block baseline): `autoresearch/results/cycle_378_heldout_test.json`
- Template: `experiments/gpt2_heldout_test_cycle378.py`
- Agenda item 389 in `autoresearch/agenda.md`
