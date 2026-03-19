# Cycle 045 Plan — Practical Compression Recipe

## Goal
Write a standalone recipe documenting the best known GPT-2 FFN geometric compression pipeline, then run it end-to-end on Shakespeare (replication) and WikiText-2 (reality check).

## Step 1 — Write recipe document
Create `autoresearch/results/cycle_045_recipe.md` containing:
- **Blocks to compress**: [7, 5, 4, 1] (easiest-first from cycle 005/033)
- **Skip block 0**: catastrophic on all model sizes (cycles 005, 041)
- **Per-block ratio**: 6x (or 10x -- cycles 040/042 show no difference with correction)
- **Channel selection**: topk by activity x output-norm on probe tokens (cycle 026)
- **Refit**: lstsq W_down refit on probe activations (cycle 026)
- **Correction**: single affine lstsq at block 8 input (cycle 039). NOT multi-point (cycle 043).
- **Probe corpus**: 200 sequences, seq_len=128. Must match eval distribution (cycle 044).
- **Seeds**: 3 seeds, 200 eval sequences each
- **CAVEAT (prominent)**: method is corpus-specific. +6-9% on Shakespeare, +32% on WikiText-2.
- **Failure modes**: block 0, k>=5 wall (+16% min), cross-domain transfer catastrophic

## Step 2 — Replication script
New file: `experiments/compression_recipe_validation.py`
Adapt from `experiments/domain_robustness_wikitext.py` (cycle 044).

### Condition A: Shakespeare end-to-end
Load GPT-2, compress blocks [7,5,4,1] at 6x with topk+lstsq on Shakespeare probe,
fit correction at block 8, eval on Shakespeare. 3 seeds. Target: +8.35% +/- 1pp.

### Condition B: WikiText-2 corpus-specific end-to-end
Same pipeline but probe AND eval on WikiText-2. Target: ~+31.86% (cycle 044 match).

## Step 3 — Verdict
- **CONFIRMED**: Shakespeare replication within 1pp of cycle 039 (+8.35%) AND recipe document is self-contained (no hidden dependencies).
- **REFUTED**: Shakespeare replication differs by >3pp from documented numbers.
- **INCONCLUSIVE**: Numbers match but recipe requires corpus-specific tuning (cycle 044 finding). This is the expected outcome -- document it honestly.

## Outputs
- `autoresearch/results/cycle_045_recipe.md` (standalone recipe)
- `autoresearch/results/cycle_045_validation.json` (replication numbers)
- `autoresearch/results/cycle_045_validation.png` (bar chart A vs B)

## Estimated cost
~10 min: 2 conditions x 3 seeds x (compress + correct + eval). No new downloads needed (WikiText-2 cached from cycle 044).
