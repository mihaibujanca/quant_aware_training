# Cycle 062 Plan: Reduced Compression Ratio for Stacking (Item 78)

## Goal
Test k=4 stacking at per-block ratios {2x, 3x, 4x, 6x} WITHOUT gradient polish. Pure analytic (topk + lstsq). 6x is replication baseline from cycle 058.

## File
Create `experiments/stacking_ratio_sweep_gpt2.py` based on `experiments/gradient_polish_stacked_gpt2.py`.

## Key changes from cycle 059/061 template

1. **Sweep variable**: `RATIOS = [2, 3, 4, 6]` (hidden_fraction = 1/ratio). Remove all polish code.
2. **max_tokens_per_layer = 8192** (not 2048 or 4096). At 2x, lstsq has 1537 columns; 8192 tokens gives 5.3x overdetermination. Cycle 056 audit showed 2048 tokens caused underdetermined lstsq at 2x — this is the critical fix.
3. **PROBE_HOLDOUT_SEQS = 200** (25600 tokens available, well above 8192 needed).
4. **No polish**: remove `_polish_block_v2`, `_polish_joint`, `POLISH_STEPS_LIST`, `POLISH_LR`. Outer loop is `for seed / for ratio`.

## Config
```
TARGET_BLOCKS = [7, 5, 4, 1]
RATIOS = [2, 3, 4, 6]
SEEDS = [42, 123, 7]
MAX_TOKENS_PER_LAYER = 8192
PROBE_TRAIN_SEQS = 200
PROBE_HOLDOUT_SEQS = 200
EVAL_VAL_SEQS = 200
SEQUENCE_LENGTH = 128
BATCH_SIZE = 4
OWT_PATH = data/openwebtext_sample.txt
```

## Experiment loop
```
for seed in SEEDS:
    corpus = build_probe_corpus(..., shuffle_seed=seed)
    for ratio in RATIOS:
        budget = CompressionBudgetSpec(hidden_fraction=1/ratio, ...)
        atlas = probe_ffn_geometry(teacher, corpus, layer_ids=TARGET_BLOCKS,
                                   max_tokens_per_layer=8192)
        pack = derive_ffn_compression(atlas, budget)
        student = swap_compressed_ffns(teacher, pack, inplace=False)
        ppl = _evaluate_model(teacher, student, corpus, batch_size=4)["perplexity"]
        results[seed][ratio] = ppl
```

## Sanity check
- 6x baseline must match cycle 058 (~75.06 mean PPL, +30.2%). If >3pp deviation, flag and investigate (note: 8192 vs 4096 probe tokens may slightly change 6x results — acceptable if within 3pp).

## Aggregation & verdict
Mean/std over 3 seeds for each ratio. Report PPL and delta_pct vs baseline.
- **CONFIRMED** if k=4@3x <= +5% PPL degradation.
- **REFUTED** if k=4@3x >= +20%.
- **INCONCLUSIVE** if +5-20%.

## Plot
Single figure (`cycle_062_stacking_ratio_sweep.png`): PPL delta (%) vs compression ratio, with error bars (std over seeds). Horizontal lines at +5% and +20% thresholds.

## Output artifacts
- `autoresearch/results/cycle_062_stacking_ratio_sweep.json`
- `autoresearch/results/cycle_062_stacking_ratio_sweep.png`
