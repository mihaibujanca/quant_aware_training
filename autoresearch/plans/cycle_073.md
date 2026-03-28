# Task
Item 84: Rank-matched adaptive compression — GPT-2 FFN. Per-block channels proportional to rank_99 (cycle 068), total budget fixed to match uniform 6x. Stacked k=4 with/without gradient polish.

# Hypothesis
- **Confirmed if**: rank-matched achieves ≤+20% unpolished OR ≤+3% polished (vs +30.2% / +4.9% uniform baselines).
- **Refuted if**: rank-matched performs worse than uniform 6x at both conditions.
- **Inconclusive if**: improvement <3pp unpolished without polish gain.

# Disambiguation (cycle 050)
This experiment does NOT use lstsq boundary correction. Cycle 050 showed variable ratios break lstsq correction MSE; that failure mode is irrelevant here. We measure raw PPL only — compression + optional Adam polish, no analytic correction step.

# Rank source
Cycle 068 (3-seed mean, 200 tokens/seed): B1=588.4, B4=460.0, B5=461.4, B7=464.0. Cycle 072 (single probe, different token sample) reported B1=640 — the discrepancy is sampling variance (cycle 068 cross-seed std=1.1, but max_rank_99=635 within one seed). We use cycle 068 means (3-seed averaged) as the more robust estimate.

# Budget arithmetic
Uniform 6x total: 4×512=2048 channels. Scale = 2048/1973.8 = 1.0376.
Rank-matched: B7=481, B5=479, B4=477, B1=611. Total=2048 (exact match).

# Steps
Script: `experiments/rank_matched_compression_gpt2.py`

1. **Setup**: Load GPT-2, OWT corpus (200 train / 400 probe / 200 eval seqs, seq_len=128). Seeds [42,123,7]. Blocks=[7,5,4,1]. Two configs: `uniform={b:512}`, `rank_matched={7:481,5:479,4:477,1:611}`. Assert both sum to 2048. Probe tokens increased to 400 seqs (51K tokens) to avoid underdetermined gradient polish for B1 (611 channels × 768 ≈ 469K params; 51K tokens × 128 dims gives sufficient gradient signal).

2. **Per config, per seed**: Probe geometry. Build per-block `CompressedFfnPack` with `CompressionBudgetSpec(hidden_fraction=k/3072, selection_mode="topk")`. Stack all 4. Evaluate unpolished PPL. Log **per-block reconstruction error norms** (Frobenius of W_compressed − W_original restricted to kept channels) to track asymmetric benefit.

3. **Polish**: 500 Adam steps (lr=1e-4) on all 4 W_down. Track held-out loss every 50 steps on probe split (first 200 seqs train, next 200 held-out) — abort if held-out loss increases 3 consecutive checks (overfitting guard). Re-evaluate polished PPL on eval split.

4. **Save JSON**: `autoresearch/results/cycle_073_rank_matched_compression.json` — per-seed PPL for both configs at 0/500 polish steps, means/stds, delta_pct vs baseline, per-block channel counts, per-block error norms, held-out loss trajectory.

5. **Plot**: Bar chart — 4 bars (uniform-0, rank-0, uniform-500, rank-500). Error bars from 3 seeds. Thresholds at +20%/+3%. Second subplot: per-block error norms (grouped by config). Save `autoresearch/results/cycle_073_rank_matched_compression.png`.

Run: `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/rank_matched_compression_gpt2.py`

# Validation
- Total channels: both configs sum to 2048 (assert in script). Param counts within 0.5%.
- Uniform replication: 0-step uniform within 5pp of cycle 061 (+30.2%). If off, abort.
- Cross-seed std < 3pp per condition.
- Polish held-out trajectory must be non-increasing (±noise). If early-stopped, report step count.

# Artifacts
- `experiments/rank_matched_compression_gpt2.py`
- `autoresearch/results/cycle_073_rank_matched_compression.json`
- `autoresearch/results/cycle_073_rank_matched_compression.png`

# Fallbacks
- If uniform replication fails (>5pp from +30.2%), debug probe token count before proceeding.
- If polish overfits (held-out increases), reduce to 300 steps or lower lr to 5e-5 and re-run.
