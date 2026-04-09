# Cycle 338 Brief

## Item
Item 347: GPT-2 full 12-block weight spectrum census — why does LASER work on GPT-2?

## Context
- Cycle 280 measured GPT-2 b1 c_proj weight SVD: top-1 energy = 4.59%, top-2 = 7.57%, top-5 = 11.67%, top-10 = 15.23%
- Cycle 337 (pgolf census): all 13 blocks flat — max top-1 = 1.66% (b12), typical ~0.5%. REFUTED LASER viability for pgolf.
- GPT-2 b1 is ~7-9x more concentrated than pgolf typical blocks in top-1 energy
- We only measured b1 in GPT-2. The other 11 blocks are unmeasured. Is b1 an outlier or representative?
- GPT-2 small: 12 blocks, c_proj uses HF Conv1D, weight shape (3072, 768) per block

## Key Question
Is GPT-2's concentrated weight spectrum a widespread pretraining phenomenon (explaining why LASER works) or a b1 outlier (meaning the cycle 280 LASER success was a lucky block pick)?

## Key Files
- `autoresearch/results/cycle_280_block1_lowrank.json` — b1 weight SVD data
- `autoresearch/results/cycle_337_spectrum_census.json` — pgolf census for comparison
- `experiments/pgolf_weight_spectrum_census_cycle346.py` — template pattern (numpy-only SVD)

## Verdict Thresholds (from agenda)
The agenda doesn't specify explicit thresholds. Propose:
- CONFIRMED (widespread): median block top-5 energy >= 8% (2x pgolf max)
- REFUTED (b1 outlier): only b1 has top-5 >= 8%, all others below 5%
- INCONCLUSIVE: mixed — some blocks concentrated, some flat
