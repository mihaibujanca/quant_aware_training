# Cycle 264 Plan — Item 274 (P2)

## Item
274. Pullback κ_l on per-block partial swaps — heatmap of where the bump is load-bearing.

## Hypothesis (verbatim)
Swapping block k shifts κ at k−1, k, k+1 strongly (local effect) AND at every later layer weakly (downstream propagation), with the heatmap showing a clear diagonal band plus an upper-triangle tail; the upper-triangle tail intensity is monotonically increasing as k decreases (early swaps have more downstream amplification).

## Verdict criteria (from agenda item 274)
- CONFIRMED: max(|Δκ|) on the diagonal band ≥ 3× the off-diagonal median AND late-layer columns show monotonically increasing |Δκ| as swap location k moves earlier.
- REFUTED: Δκ heatmap is structureless (no diagonal band, no upper-triangle pattern) OR anti-diagonal.
- INCONCLUSIVE otherwise.

## Setup
- Reuse cycle 258 loader: 200 wikitext-2 val × 128, seed 42, MPS.
- Reuse cycle 258 b_leak per-block per-channel precompute (E_x[gelu(z) − relu(z)] in 3072-d intermediate space).
- Reuse cycle 258 ReluPlusBleak module exactly.
- Curvature via aleph.pullback_geometry.compute_curvature(reduce='mean'), 11-vector for layers 1..11.

## Steps
1. Create experiments/per_block_swap_curvature_cycle264.py.
2. Load gpt2 + 200×128 calib slice; precompute b_leak per block (cycle 258 hook recipe).
3. Compute baseline κ_baseline (all-GELU) once via collect_residual_trajectory + compute_curvature.
4. For k ∈ {0..11}:
   - Patch block k act with ReluPlusBleak(b_leak[k]); other 11 blocks untouched.
   - Collect trajectory; compute κ_swap_at_k (length 11).
   - Restore block k to original GELU.
5. Build Δκ matrix shape (12, 11): rows = swap location k, columns = κ index l ∈ {0..10} (layers 1..11).
6. Compute ‖Δκ‖₁ per row; rank swaps.
7. Verdict logic: diagonal band = entries where κ index l ∈ {k−1, k, k+1} (k=0 → l ∈ {0,1}); off-diagonal = the rest. Compute max|Δκ|_diag vs median|Δκ|_offdiag and ratio. Late-layer monotonicity: for each late l ∈ {7..10}, check whether |Δκ[k,l]| increases as k decreases through k=0..6.
8. Save JSON with κ_baseline, κ_per_swap (12×11), Δκ matrix (12×11), L1 ranking, verdict logic numbers.
9. Save heatmap PNG to autoresearch/results/cycle_264_swap_kappa_heatmap.png.
10. Save summary md, manifest, log.

## Sanity / restore
- After every swap restore originals[k]; final assert eval_ppl matches baseline within 1e-4.
- 12 forward passes for trajectories + 1 baseline + 1 b_leak hook pass ≈ 14 passes.

## Trainable params
0.
