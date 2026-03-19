# Task
Measure how centroid-merge channel selection quality varies with probe corpus size {64, 128, 256, 512, 1024, 2048} tokens on a single GPT-2 FFN block.

# Hypothesis
Centroid-based channel selection is stable under small probe corpora, requiring <= 256 tokens to match the 2048-token reference.

**Confirmed if**: Jaccard >= 0.85 and PPL within 2% of the 2048-token reference at probe size <= 256 tokens.
**Refuted if**: Jaccard < 0.6 or PPL degrades > 5% relative at probe size 256, indicating the signal needs large data.
**Inconclusive if**: Jaccard is stable but PPL variance across seeds is high (> 3%), suggesting instability from a different source.

# Pre-flight Results
- P1 PASS: No prior experiment tests channel selection vs probe size. Cycle 027 measured affinity signals at fixed 1024 tokens.
- P2 PASS: Deps 26 (INCONCLUSIVE cycle 015), 27 (REFUTED cycle 047). No skip-if rule on item 50.
- P3 PASS: Single concrete experiment with clear metrics.
- P4 PASS: No modifications to program.md needed.
- P5 (null baseline): Random channel selection Jaccard at k=512 from 3072 = C(3072,512)^{-1} overlap ~ 0.17. Any Jaccard > 0.5 is well above chance.
- P6 PASS: GPT-2 uses GELU; centroid-merge uses activation statistics (mean, std, derivative), compatible with GELU.
- P7 PASS: 3 seeds x 6 probe sizes = 18 runs. Jaccard is a set statistic (no sampling noise). PPL eval on fixed 200-seq OWT holdout.

# Steps
1. Create `experiments/centroid_probe_stability_gpt2.py`:
   - Load GPT-2 via `load_nanochat_teacher(TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2"))`.
   - Use block 7 (easiest to compress per cycle 005/058). `CompressionBudgetSpec(selection_mode="topk", hidden_fraction=1/6)` (6x, k~512).
   - Fixed OWT eval corpus: `ProbeCorpusSpec(parquet_dir=..., eval_val_sequences=200, sequence_length=128)`.
   - For each probe_size in {64, 128, 256, 512, 1024, 2048} tokens and each seed in {42, 123, 7}:
     a. Build probe corpus with `probe_holdout_sequences = ceil(probe_size / 128)`, `shuffle_seed=seed`.
     b. Build atlas for block 7 with `max_tokens_per_layer=probe_size`.
     c. Derive compression pack, record `selected_channels` (sorted tensor of chosen channel indices).
     d. Swap FFN, evaluate PPL on the fixed eval corpus via `_evaluate_model`.
   - Compute per probe_size: Jaccard = |intersection| / |union| of channel sets across all 3 seeds. Mean PPL and std across seeds. PPL % degradation vs baseline GPT-2 PPL.
   - Reference: 2048-token mean PPL is the baseline for relative PPL comparison.
   - Save JSON to `autoresearch/results/cycle_067_centroid_stability.json` with per-size-seed results.
   - Plot: 2-panel figure. Left: Jaccard vs probe_size with 0.85 threshold line. Right: PPL % degradation vs probe_size with 2% and 5% threshold lines. Error bars = seed std. Save to `autoresearch/results/cycle_067_centroid_stability.png`.

2. Run: `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/centroid_probe_stability_gpt2.py`

3. Adjudicate outcome by checking JSON: Jaccard@256 vs 0.85, PPL@256 vs 2%/5% thresholds, PPL std vs 3%.

4. Update `autoresearch/agenda.md`: move item 50 from NEXT to PAST EXPERIMENTS with outcome, cycle, artifacts.

5. Commit: `[autoresearch] cycle 067: centroid-merge probe stability — GPT-2 FFN`

# Validation
- Baseline GPT-2 PPL should be ~30 (WikiText) or ~58 (OWT) — sanity check against known values.
- Jaccard at probe_size=2048 across seeds should be high (>0.9) — if not, the method itself is unstable.
- Channel counts: all runs should select exactly k=512 channels. Assert in code.
- PPL at 2048 tokens should roughly match cycle 058 single-block values (~-3 to -5% for block 7).

# Artifacts
- `experiments/centroid_probe_stability_gpt2.py`
- `autoresearch/results/cycle_067_centroid_stability.json`
- `autoresearch/results/cycle_067_centroid_stability.png`

# Fallbacks
- If OWT download fails: fall back to Shakespeare probe (as in cycle 027), note corpus difference.
- If block 7 shows degenerate behavior (all seeds identical at all sizes): repeat with block 4.
- If Jaccard is perfect (1.0) at all sizes: the channel scores may be dominated by weight norms (data-free). Check by comparing topk channels with and without activation statistics (weight-only scores). Report as a separate finding.
