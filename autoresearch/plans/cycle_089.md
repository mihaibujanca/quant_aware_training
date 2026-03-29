# Task
Item 104: Compositional error accumulation curve (k=1..8 blocks) — GPT-2 FFN. Compress blocks at 6x using topk+lstsq+500 Adam polish. Accumulate k=1..8 in two orderings (least-damage-first and reverse). Measure stacked PPL at each k. Fit scaling law. 3 seeds, OWT eval.

# Hypothesis
- **Confirmed if**: error is sub-multiplicative (PPL(k) < baseline · (1+δ)^k) — stacking is viable beyond k=4.
- **Refuted if**: error is super-multiplicative (cross-block amplification) — hard ceiling exists.
- **Inconclusive if**: additive for k<=4 but regime changes at k>4.

# Pre-flight Results
- P1: Not duplicate. Cycle 036 ran k=1..6 at 6x but WITHOUT polish (pure lstsq). Cycle 059 ran k=4 with polish but only tested polish step counts {0,10,25,50}, not accumulation curve. Cycle 087 tested 4-block stacking with mixed corpus but not the full k=1..8 polished curve. This is the first polished accumulation sweep to k=8.
- P2: No dependencies. Item says "Depends on: none".
- P3: Concrete — single script, 8 blocks, two orderings, 3 seeds, clear metric (PPL vs k curve + scaling law fit).
- P4: No program.md changes needed.
- P5 (null baseline): If each block's single-swap damage δ_i is independent and additive, PPL(k) ≈ baseline + Σ_{i=1}^{k} δ_i (linear in k). If errors multiply through the residual stream, PPL(k) ≈ baseline · Π_{i=1}^{k} (1 + δ_i/baseline) ≈ baseline · (1+δ̄)^k (exponential). Cycle 036 (unpolished k=6) hit +36% at k=6, roughly consistent with multiplicative compounding of ~5% per block. Polish reduces single-block damage from ~5% to ~1% (cycle 059/061), so polished multiplicative null predicts PPL(8) ≈ baseline · 1.01^8 ≈ +8%. Anything worse than +8% at k=8 signals super-multiplicative interaction. Report both additive and multiplicative fits.
- P6 (methods): Compress each block independently (probe geometry, topk selection, lstsq W_down, 500 Adam polish on W_down only — lr=1e-4, batch_size=4, probe_holdout as polish data). Then swap all k blocks simultaneously and evaluate PPL. Polish is per-block (not joint) to isolate composition effects. Use `_polish_block_v2` from `experiments/gradient_polish_stacked_gpt2.py`. Two orderings test whether damage order matters for error propagation through the residual stream.
- P7 (power): 200 eval sequences × 128 tokens = 25,600 eval tokens per measurement. 3 seeds. With baseline PPL ~30 on OWT, a +5% shift is ~1.5 PPL points — well above noise floor (cycle 059 seed std was ~0.3 PPL). 8 k-values × 2 orderings × 3 seeds = 48 measurements total. Note: brief specifies blocks [7,5,4,1,3,6,8,9] but cycle 007's full damage order is [7,5,4,1,3,6,2,9,10,8,11,0] — block 2 is easier than block 8. Use the brief's order as given (it may intentionally skip block 2 to test harder blocks earlier).

# Steps
Script: `experiments/compositional_accumulation_curve_gpt2.py`. Adapt from `experiments/multiblock_accumulation_6x.py` (accumulation loop) + `experiments/gradient_polish_stacked_gpt2.py` (polish infrastructure, OWT corpus, seed handling).

1. **Setup.** Load GPT-2 (float32, CPU). Build OWT probe corpus: 200 probe_train, 200 probe_holdout (polish data), 200 eval_val sequences, seq_len=128. Seeds = [42, 123, 7]. Forward ordering = [7,5,4,1,3,6,8,9]. Reverse ordering = [9,8,6,3,1,4,5,7].

2. **Per-block compression.** For each seed: probe FFN geometry for all 8 blocks simultaneously (max_tokens_per_layer=4096, batch_size=4). Derive compression at hidden_fraction=1/6, selection_mode="topk". This gives a CompressedFfnPack with 8 layers.

3. **Per-block polish.** For each of the 8 compressed blocks: swap ONLY that block into a fresh copy, run 500 Adam steps (lr=1e-4) on W_down using probe_holdout tokens. Extract the polished W_down weights. Store polished CompressedFfnLayer objects indexed by block_id. (Polish is per-block to avoid joint optimization confounds.)

4. **Accumulation sweep.** For each ordering (forward, reverse), for k=1..8: take the first k blocks from the ordering, build a CompressedFfnPack from the polished layers, swap into teacher, evaluate PPL on eval_val. Record: k, ordering, blocks_active, PPL, relative_delta_pct.

5. **Scaling law fit.** For each ordering: fit PPL(k) = a + b·k (additive) and PPL(k) = a · (1+c)^k (multiplicative) via least squares. Report R² for both. If multiplicative R² > 0.95 and c < single-block-avg-delta/baseline, verdict is sub-multiplicative. Also compute residuals: if PPL(k) consistently exceeds multiplicative fit at high k, that's super-multiplicative.

6. **Aggregate.** Mean and std across 3 seeds for each (k, ordering) pair.

7. **Plot.** `autoresearch/results/cycle_089_accumulation_curve.png` — (a) PPL vs k for both orderings (mean ± std), with additive and multiplicative fit curves overlaid, (b) per-k residual from multiplicative fit (positive = super-multiplicative).

8. **Save.** JSON: `autoresearch/results/cycle_089_accumulation_curve.json` — per-seed raw PPL, aggregates, fit parameters, verdict.

9. **Bookkeeping.** Update agenda item 104 with outcome, cycle tag, artifact paths. Commit: `[autoresearch] cycle 089: compositional error accumulation curve — GPT-2 FFN`.

# Validation
- Baseline GPT-2 PPL on OWT should be ~30 (sanity check — if wildly different, corpus issue).
- k=1 forward (block 7 alone) should give <+2% degradation (block 7 is easiest, cycle 005: -0.63% uncompressed).
- k=4 forward polished should be ~+5% (cycle 059/061 established this).
- PPL must be monotonically non-decreasing with k for each ordering (adding more compressed blocks cannot help).
- Forward ordering PPL(k) <= reverse ordering PPL(k) for all k (least-damage-first is always better or equal).
- Multiplicative fit R² > 0.9 (if not, the model is neither additive nor multiplicative — report raw curve).

# Artifacts
- `experiments/compositional_accumulation_curve_gpt2.py`
- `autoresearch/results/cycle_089_accumulation_curve.json`
- `autoresearch/results/cycle_089_accumulation_curve.png`
- Agenda update: item 104 outcome + cycle tag.
- Commit: `[autoresearch] cycle 089: compositional error accumulation curve — GPT-2 FFN`

# Fallbacks
- OOM with 8 blocks probed simultaneously → probe in two batches of 4, merge atlas.
- Per-block polish at 500 steps × 8 blocks × 3 seeds = 12,000 total polish steps → ~20 min on CPU. If too slow, reduce to 250 steps (cycle 061 showed 250 captures 90% of polish benefit).
- Non-monotonic PPL (k=n+1 < k=n) → likely seed noise. Check per-seed. If persistent, flag as anomaly.
- Block 8 or 9 geometry probe fails (out-of-range layer_id) → GPT-2 small has 12 blocks (0-11), blocks 8 and 9 are valid.
