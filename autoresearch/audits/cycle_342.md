# Audit — Cycle 342

**Item**: 351 — LASER with output-covariance gating on pgolf
**Provisional verdict**: REFUTED

## Checks

1. **Baseline consistency**: baseline_bpb=1.362078 matches cycle 327 exactly. Same checkpoint, same eval. PASS.

2. **Retrain budget was actually extended**: 2000 steps confirmed in stderr logs (step 100/2000 through 2000/2000). Wall time ~195s per config vs ~49s for 500-step configs. PASS.

3. **Results match criteria**: Item 351 says "Refuted if output-covariance-gated LASER still gives BPB delta > 0.01". Best delta = +0.034847, clearly > 0.01. PASS.

4. **Comparison to cycle 327**:
   - rank=5: +0.037780 (c327, 500 steps) → +0.037669 (c342, 2000 steps, lr=3e-5) — improvement: 0.000111 BPB
   - rank=10: +0.034984 (c327, 500 steps) → +0.034847 (c342, 2000 steps, lr=3e-5) — improvement: 0.000137 BPB
   - Essentially zero improvement. The retrain budget was never the bottleneck. PASS.

5. **lr control**: rank=5 at lr=1e-4 with 2000 steps gave +0.037037 — marginally better than lr=3e-5 (+0.037669) but still +0.037. Neither lr helps. PASS.

6. **Interpretation**: The B_init BPB (SVD, no retrain) for rank=5 is 1.403734, delta=+0.041656. After 2000 steps retrain, delta falls to +0.037669 — only 0.004 BPB recovered by retrain. The SVD truncation error is structural: rank-5 captures only 2.62% of weight Frobenius energy. Even the output-covariance eff_rank=4.89 is ~2.5x less concentrated than GPT-2 b1's 1.94. LASER needs near-rank-1 concentration to work.

## Status: PASS

Verdict confirmed: **REFUTED**. Extended retrain does not rescue LASER on pgolf b3. The bottleneck is insufficient output-covariance concentration, not retrain budget.
