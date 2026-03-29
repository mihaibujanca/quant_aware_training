# Cycle 106 Plan: Gradient Polish on k=5 Stacking — GPT-2 FFN

**Item:** 101
**Script:** `experiments/gradient_polish_k5_stacking_gpt2.py` (new, derived from `gradient_polish_stacked_gpt2.py`)

## Pre-flights

**P1 -- Duplicate:** No. k=5 polish never tested.
**P2 -- Dependencies:** None.
**P3 -- Concrete:** Two configs, four polish budgets, three seeds, one script.
**P4 -- Prompt/program edits:** None required.
**P5 -- Null baseline:** 0-step polish on OWT (measured in Phase 1 of this experiment).
**P6 -- Method fit:** W_down-only polish. If wall is attention-driven, refutation is informative but does NOT close item 101 (see Verdict).
**P7 -- Power:** 2 configs × 4 step budgets × 3 seeds = 24 runs. Seed variance ~2pp (cycle 061).

## Hypothesis

Polish (Adam on W_down) reduces k=5 stacking PPL delta on OWT by ≥50% relative.

- **Confirmed if:** best polish reduces OWT delta by ≥50% for ≥1 config.
- **Refuted if:** best polish reduces delta by <20% for both configs.
- **Inconclusive if:** 20-50% reduction range.

Note: refutation of W_down-only polish does NOT close item 101. It motivates attention-inclusive polish (joint W_down + W_V or full-block fine-tune) as follow-up.

## Steps

### Phase 1: OWT Calibration (1 seed)

1. Eval float GPT-2 on OWT (200 seqs, seq_len=128). Abort if outside [25, 65].
2. Compress k=5 with configs `[7,5,1,3,6]` and `[7,5,4,1,6]`, 0 polish steps, seed=42.
3. Record `owt_0step_delta_pct` for each config. This is the unpolished OWT baseline.
4. Log to stdout. Expected range: +30-40% based on k=4 OWT-vs-Shakespeare scaling.

### Phase 2: Polish Sweep (all seeds)

5. **Copy and adapt** `gradient_polish_stacked_gpt2.py`. Changes:
   - `TARGET_BLOCKS_LIST = [[7,5,1,3,6], [7,5,4,1,6]]`
   - `POLISH_STEPS_LIST = [0, 50, 200, 500]`
   - `POLISH_LR = 1e-4`, `SEEDS = [42, 123, 7]`
   - OWT data, `PROBE_TRAIN_SEQS=200`, `EVAL_VAL_SEQS=200`, seq_len=128
   - Joint polish on all 5 W_down simultaneously.

6. **Per-step logging (A2):** Every 10 Adam steps, log both train loss and eval PPL on a held-out 50-seq split. Flag if eval PPL increases >2% from its minimum (overfitting signal).

7. For each config × seed × step budget: eval PPL, compute delta_pct.

### Phase 3: Verdict

8. Compute `reduction_pct = (delta_0step - delta_polished) / delta_0step * 100` per config, using mean over 3 seeds. Report per-seed results alongside means.
9. Apply thresholds: ≥50% reduction → confirmed, <20% → refuted, else inconclusive.

## Output

- `autoresearch/results/cycle_106_k5_polish.json` — all runs + per-step logs.
- `autoresearch/results/cycle_106_k5_polish.png` — line plot: steps vs PPL delta (%), both configs, error bars = seed std.

## Validation

- Phase 1 0-step must produce finite, reasonable PPL (no NaN/inf, delta >5%).
- Polish train loss monotonically decreasing for first 10 steps minimum.

## Fallbacks

- If Phase 1 gives delta <10%: config or compression mismatch — re-derive atlas, abort.
- If all budgets <20% reduction: REFUTED. Note attention-cascade (cycle 104) as hypothesis; propose attention-inclusive polish as next cycle.
