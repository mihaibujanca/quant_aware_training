# Cycle 194 Brief: Self-Correction Depth Analysis (Item 204)

## Item

204 — Self-correction depth analysis — where and why blocks 8-10 partially compensate

## Context

Cycles 190 and 193 both show non-monotone progressive PPL: in cycle 190 (hybrid), PPL
peaks at k=7 (28.484), decreases through k=10 (27.851), then rises at k=11 (28.980).
In cycle 193 (C3+Hybrid), PPL peaks at k=9 (38.393), decreases at k=10 (38.283), then
jumps at k=11 (41.447). This pattern of "partial self-correction at blocks 8-10" recurs
across different activation modifications.

## Hypothesis

Two competing mechanisms:
1. **Error magnitude decreases**: later hybrid blocks actually reduce ||e_l||_F (partial
   cancellation). The modified activation at later blocks corrects accumulated earlier errors.
2. **Error redirected**: ||e_l||_F keeps growing, but the error moves to directions that
   don't affect PPL much (low-PPL subspace). The error is present but harmless.

## Experiment Design

Rerun the hybrid (cycle 190 config: SEQ_LEN=1024, N_SEQS=200, SEED=42) progressive sweep
but add residual error collection at each k. For each k:
1. Build float+hybrid(k) model pair
2. Collect residuals from both at analysis tokens
3. Compute e_l = res_float[l] - res_hybrid_k[l] for all 12 layers
4. Measure ||e_l||_F per layer and mean_cos(e_l, e_l_k7) (alignment with k=7 errors)

Key comparison: at k=8,9,10 vs k=7:
- Does ||e_l||_F decrease? → Confirmed (partial cancellation)
- Does cos(e_l, e_7) decrease while ||e_l||_F grows? → Refuted (redirected)
