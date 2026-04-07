# Item
274 (P2): Pullback curvature κ_l on per-block partial swaps — heatmap.
For each l ∈ {0..11}: install ReLU+b_leak ONLY at block l (other 11 stay GELU). Compute κ profile via pullback geometry. Report Δκ_l profile vs baseline (12-vector per swap). Heatmap: rows = swap location k, columns = layer l where κ moves.

# Context
- Cycle 258: all-12 ReLU+b_leak swap → κ kinks in layers 2-6 by 14-48%. We don't know if this is local (layer k swaps move only κ_k) or distributed.
- Cycle 173: single-block C3 swaps cost 0.4-1.4% PPL each (single-block local effect).

# Dependencies
- aleph/pullback_geometry.py (compute_curvature with reduce='none' or default).
- Cycle 258's b_leak per-block computation.
- Cycle 258's ReLU+b_leak patching code.

# Key Files
- experiments/pullback_geom_variants_cycle258.py — has per-block b_leak compute and the patching logic.

# Constraints
- Calib: 200 wikitext-2 val × 128, seed 42 (matches cycle 258).
- Compute b_leak per block once (reuse cycle 258).
- For each swap_location k ∈ {0..11}:
  - Install ReLU + b_leak[k] ONLY at block k. Other 11 blocks stay GELU.
  - Run collect_residual_trajectory + compute_curvature(reduce='mean').
  - Get κ profile (length 11 — for layers 1..11).
  - Save as κ_swap_at_k.
- Compute Δκ_swap_at_k = κ_swap_at_k − κ_baseline_GELU.
- Build 12 × 11 matrix Δκ[k, l] (rows = swap, cols = layer where κ moved).
- Compute ‖Δκ‖_1 per swap to rank "how disruptive each swap is".
- Plot heatmap and per-row profile.
- Verify: per-block local effect (rows show diagonal band at k-1, k, k+1) PLUS upper-triangle tail (k → l > k+1 weak propagation).
- 12 forward passes total. Each ~1 min. Total ~15 min.
- Trainable params: 0.

# Handoff
Plan agent: design via reusing cycle 258 patches. New script `experiments/per_block_swap_curvature_cycle264.py`. Output JSON + summary + heatmap PNG.
