# Task
Multi-seed confirmation of single-block compression tolerance at 6x — GPT-2 FFN (item 52).
Replicate cycle 034's 6x single-block result on 3 seeds with 200 eval sequences each.

# Hypothesis
- **Confirmed if**: for >=5 of 6 blocks, mean relative PPL at 6x across 3 seeds is <=5%, AND no block shows sub-baseline mean PPL (non-monotonicity was noise).
- **Refuted if**: for >=3 of 6 blocks, mean relative PPL at 6x exceeds 5%, indicating 034's tolerance was a single-seed artifact.
- **Inconclusive if**: mean PPL is within threshold but cross-seed std exceeds 2% for a majority of blocks, indicating high measurement noise even with more sequences.

# Pre-flight Results
- P1 PASS: No prior multi-seed 6x experiment in PAST EXPERIMENTS.
- P2 PASS: Depends on item 49 — INCONCLUSIVE (not refuted). Item 52 is the follow-up.
- P3 PASS: Concrete — sweep 6 blocks x 3 seeds x 1 ratio, report mean/std.
- P5: Baseline PPL computed fresh per seed (each seed rebuilds corpus with its own shuffle_seed).
- P6: Method uses topk + lstsq refit — GELU-compatible (confirmed cycles 026-034).

# Steps
1. Create `experiments/multiseed_6x_confirmation_gpt2.py` adapted from `experiments/block_ratio_sweep_gpt2.py`:
   - `SEEDS = [42, 123, 7]`, `RATIO = 6`, `PROBE_HOLDOUT_SEQ = 200`, `TARGET_BLOCKS = [7, 5, 4, 1, 3, 6]`.
   - **Critical fix (review issue 1)**: For each seed, pass `shuffle_seed=seed` to `ProbeCorpusSpec`.
     This varies the train/val document split, changing both the probe geometry input and eval windows.
     `torch.manual_seed(seed)` is also set for any stochastic PyTorch ops but is NOT the primary variance source.
   - For each seed: build corpus with `ProbeCorpusSpec(shuffle_seed=seed, ...)`, compute baseline PPL, probe geometry, then run single-block 6x swap and eval for each block.
   - **Null baseline (review issue 2)**: For each seed, also run a `selection_mode="random"` compression at 6x per block.
     Check if `CompressionBudgetSpec` supports `selection_mode="random"`. If not, implement in-script:
     randomly permute channel indices (seeded by `seed`) and take first 1/6, then run lstsq refit.
     This bounds the noise floor — if topk is no better than random, the compression is meaningless.
   - Collect per-block per-seed: baseline_ppl, compressed_ppl, relative_ppl_pct, random_ppl, random_relative_pct.
   - After all seeds: compute per-block mean/std of relative PPL (topk) and mean of random baseline.
   - Save JSON to `autoresearch/results/cycle_035_multiseed_6x.json`.
   - Bar chart: blocks x-axis, mean relative PPL with std error bars, horizontal line at 5% threshold,
     overlay random-baseline mean as grey markers. Save to `autoresearch/results/cycle_035_multiseed_6x.png`.
2. Run: `python experiments/multiseed_6x_confirmation_gpt2.py`.
3. Read JSON. Apply verdict criteria: count blocks with mean <=5%, check sub-baseline mean, check std >2%.
   Also verify topk beats random for all blocks (if not, flag in summary).
4. Update `autoresearch/agenda.md`: move item 52 to PAST EXPERIMENTS with verdict and cycle number.
5. Write `autoresearch/results/cycle_035_summary.md` with verdict, per-block table, random-baseline comparison.
6. Commit all artifacts.

# Validation
- JSON must contain 6 blocks x 3 seeds x 2 modes (topk, random) = 36 data points.
- Baseline PPL should differ across seeds (confirms shuffle_seed is working). If identical, the seed variation is broken — abort and debug.
- Seed 42 topk results should approximately match cycle 034 values (within noise from 200 vs 50 eval seqs).
- Random-baseline PPL should be strictly worse than topk for all blocks. If not, topk selection adds no value.
- No NaN/Inf in any PPL value.

# Artifacts
- `experiments/multiseed_6x_confirmation_gpt2.py`
- `autoresearch/results/cycle_035_multiseed_6x.json`
- `autoresearch/results/cycle_035_multiseed_6x.png`
- `autoresearch/results/cycle_035_summary.md`

# Fallbacks
- If `selection_mode="random"` not in API: implement random channel selection in-script using `torch.randperm` with seed.
- If baseline PPL is identical across all 3 seeds: shuffle_seed is not propagating. Debug `build_probe_corpus` to find where seeding happens and patch.
- If one seed OOMs: reduce batch_size to 2, retry.
