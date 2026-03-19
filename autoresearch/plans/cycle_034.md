# Task
Block-specific compression ratio sweep on GPT-2 FFN blocks [7, 5, 4, 1, 3, 6]. Sweep ratios {2x, 3x, 4x, 5x, 6x} per block (single-block swaps), find max ratio within 5% PPL threshold.

# Hypothesis
From agenda item 49 (verbatim + expanded inconclusive per design review):
- **Confirmed if**: at least 4 blocks tolerate >=5x compression within the 5% PPL threshold, suggesting headroom for tighter multi-block stacking.
- **Refuted if**: no block tolerates >4x compression within threshold, indicating the current merge algorithm saturates before structural gains.
- **Inconclusive if**: results are bimodal (some blocks tolerate 6x, others collapse at 3x) without a clear predictor. Also inconclusive if 1-3 blocks tolerate >=5x (partial headroom, insufficient for broad stacking).

# Pre-flight Results
- **P1 (duplicate)**: PASS. No prior experiment swept per-block ratios independently.
- **P2 (dependencies)**: PASS. Items 5 (CONFIRMED), 26 (INCONCLUSIVE), 27 (CONFIRMED). No skip-if-refuted triggered.
- **P3 (concrete)**: PASS. 6 blocks x 5 ratios = 30 evaluations, clear table output.
- **P4 (prompt modification)**: PASS. No changes to program.md or prompts.
- **P5 (null baseline)**: Threshold computed dynamically: `threshold = baseline_ppl * 1.05`. SVD-norm baseline (cycle 026) gives +2.2-7.2% at 4x; topk+lstsq should beat that.
- **P6 (methods)**: Topk channel selection + lstsq refit is GELU-appropriate (validated cycles 026-027).
- **P7 (statistical power)**: 200 train + 50 holdout seqs. Deterministic pipeline (no stochastic training).

# Steps
1. Create `experiments/block_ratio_sweep_gpt2.py` extending cycle 033 script:
   - Config: `TARGET_BLOCKS = [7, 5, 4, 1, 3, 6]`, `RATIOS = [2, 3, 4, 5, 6]`
   - Shakespeare probe: 200 train, 50 holdout, seq_len=128, seed=42
   - Load GPT-2, measure baseline PPL once. Compute `threshold = baseline_ppl * 1.05` dynamically (R1 fix: no hardcoded baseline).
   - Outer loop: blocks. Inner loop: ratios. Each: fresh model copy, compress ONE block, measure full-model PPL.
   - **R2 fix**: After lstsq refit, compute reconstruction MSE = `||X_out_original - X_out_compressed||^2 / n_tokens` on probe train data. Log alongside PPL.
   - Output: 30-row table (block, ratio, ppl, ppl_pct_change, lstsq_recon_mse). Per-block max_ratio where ppl <= threshold.
   - Save JSON to `autoresearch/results/cycle_034_ratio_sweep.json` (includes baseline_ppl, threshold, and per-entry mse).
   - Save heatmap plot (blocks x ratios, colored by PPL%) to `autoresearch/results/cycle_034_ratio_sweep.png`.
2. Run: `python experiments/block_ratio_sweep_gpt2.py`
3. Evaluate verdict using **R3 expanded criteria**:
   - >=4 blocks with max_ratio >=5: CONFIRMED
   - 0 blocks with max_ratio >4: REFUTED
   - 1-3 blocks with max_ratio >=5: INCONCLUSIVE (partial headroom)
   - Bimodal without predictor: INCONCLUSIVE
   - If non-monotonic PPL observed, check lstsq MSE to diagnose refit artifact.
4. Update `autoresearch/agenda.md`: move item 49 to PAST EXPERIMENTS with outcome, commit hash, artifacts.
5. Commit: `[autoresearch] cycle 034: block-specific compression ratio sweep — GPT-2 FFN`

# Validation
- Baseline PPL must be in ~120-130 range (consistent with prior Shakespeare cycles)
- At ratio=4x, block 7 PPL should roughly match cycle 033 single-block result
- PPL must be monotonically non-decreasing with ratio per block; if violated, check lstsq MSE for that entry
- lstsq MSE should be monotonically increasing with ratio (more compression = worse reconstruction)

# Artifacts
- `experiments/block_ratio_sweep_gpt2.py`
- `autoresearch/results/cycle_034_ratio_sweep.json`
- `autoresearch/results/cycle_034_ratio_sweep.png`

# Fallbacks
- OOM: reduce to {3, 5, 6} (skip 2x/4x since 4x known from cycle 033)
- Baseline PPL >5% off prior cycles: check Shakespeare path and probe config; abort if unresolvable
- All blocks collapse at 5x: try 4.5x intermediate to narrow saturation point
