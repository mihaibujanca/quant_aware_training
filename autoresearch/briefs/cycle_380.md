# Cycle 380 Brief — Full-Model Float+Adapter Ceiling (Item 388)

## Context
- c377: 3-block float+adapter (blocks [2,1,10], h=64, 297K params) PPL 38.94 (val), 39.73 (test via c378).
- c378: held-out verification confirmed val ordering on test. Float baseline test PPL 56.52.
- This experiment: attach h=64 MLP adapters to ALL 12 GPT-2 blocks (uncompressed). Train jointly, same schedule as c377 (2000 steps, Adam lr=1e-4). This measures the adapter capacity ceiling — the best PPL achievable by adding adapter capacity to float GPT-2 without any compression.

## Hypothesis
12-block adapters have headroom beyond 3-block (PPL < 36.0). If 12-block is only marginally better than 3-block (PPL > 38.0), the 3-block config already saturates adapter capacity.

## Design
- Float GPT-2 with h=64 MLP adapters on blocks 0-11 (all 12). All GPT-2 weights frozen; only adapter params trained.
- Train on wikitext-2-val (200 seqs), eval on wikitext-2-test (200 seqs). Best checkpoint by val eval (50 seqs).
- 3 seeds (42, 137, 2024). Also eval float baseline on test for sanity.
- Total adapter params: 12 × 99,136 = 1,189,632 (~4× the 3-block budget).

## Success Criteria
- **Confirmed** (PPL < 36.0): adapters have headroom; 3-block adapter is far from ceiling.
- **Refuted** (PPL > 38.0): 3-block adapter already near ceiling; more blocks don't help.
- **Inconclusive**: PPL in [36, 38].

## Key References
- c377 results: `autoresearch/results/cycle_377_adapter_only.json`
- c378 results: `autoresearch/results/cycle_378_heldout_test.json`
- Template: `experiments/gpt2_adapter_only_cycle377.py`
- Agenda item 388 in `autoresearch/agenda.md`
