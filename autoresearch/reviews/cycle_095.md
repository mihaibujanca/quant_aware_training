verdict: APPROVE
round: 2

## R1-R5 checklist

**R1 (meaningful question):** Pass. Corrects item 105's compression math. Per-head [768,64] rank-11 SVD = 5.4x is the right granularity. Block-level design (all 12 heads compressed simultaneously) directly tests the operationally relevant scenario. Confirmation/refutation thresholds (+5% / +15%) are concrete and discriminating.

**R2 (null baseline):** Pass. P1 states the null (full-rank at d_head=64, rank-11 destroys function). Baseline PPL evaluated with abort bounds [25, 65], which bracket GPT-2-small's expected ~29-35 range.

**R3 (3 seeds):** Pass. Round 1 blocking issue resolved. Step 3 now explicitly evaluates all block-level swaps under all 3 seeds [42, 123, 7]. Cost estimate updated to 36 evals (12 blocks x 3 seeds). Mean and std of delta% recorded per block. Verdict logic operates on mean delta, so seed-to-seed noise (~1-3%) is averaged out.

**R4 (cost estimate):** Pass. 36 evals x 200 seqs each, estimated 10-15 min on MPS. Reasonable.

**R5 (known pitfalls):** Pass. MP8 (representation dimension vs functional rank) is marginally relevant: weight-space SVD is a stricter test than data-dependent functional rank, so if weight SVD compression preserves PPL, data-dependent compression would work at least as well. No other pitfalls triggered.

## Round 1 issue resolution

**R3 blocking (3 seeds per swap):** Fixed. The design was also improved from per-head to block-level swaps, which better matches the stated question (block-level compressibility). 3 seeds are now applied at every block swap.

**A1 (W_O omission):** Addressed. P4 now explicitly notes that QKV-only 5.4x yields ~3.1x actual block-level compression because W_O is untouched. Scope limited to QKV feasibility.

**A2 (Phase 2 conditional execution):** Addressed. Phase 2 now always runs on top-3 most compressible blocks regardless of Phase 1 outcome.

## Remaining notes (non-blocking)

- Phase 2 has no formal confirmation/refutation criterion. Acceptable for exploratory multi-block evidence, but results should be interpreted qualitatively.
- Results JSON should record achieved compression ratio (5.4x per-projection, ~3.1x per-block) alongside PPL deltas to avoid inflated claims.
