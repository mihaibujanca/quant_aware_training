# Item
269 (P0): Pullback curvature & salience — GELU vs ReLU+bias trajectories.
For (a) baseline GELU, (b) ReLU+precomputed b_leak (cycle 181 patch), (c) gelu_noneg+μ (cycle 228 patch): collect residual trajectory via aleph.pullback_geometry, compute κ_l and S_l per layer, plot, report ρ(S, κ) per variant.

# Context
- aleph/pullback_geometry.py provides collect_residual_trajectory, compute_salience, compute_curvature.
- Cycle 181 patch (ReLU + per-block precomputed b_leak): need to find or recreate. Cycle 184 used "ReLU+bias" — likely the same pattern.
- Cycle 228 patch (gelu_noneg + per-channel μ_l): cycle 228 has GeluNonegCorrected class.
- The agenda item references "cycle 252" as having measured pullback κ but cycle 252 was the block 5 geometric deep-dive — there's likely a reference mismatch. The cycle 252 reference is probably to a parallel pullback experiment that may or may not exist on disk. Treat the absence of a baseline as expected; this cycle generates the GELU baseline along with the two variants.

# Dependencies
- aleph/pullback_geometry.py (exists, 229 lines, has all needed functions).
- Cycle 181 ReLU+bias patch: search for it. Likely at experiments/relu_bias_*.py.
- Cycle 228 gelu_noneg+μ patch: experiments/gelu_noneg_corrected_cycle228.py has GeluNonegCorrected.

# Key Files
- aleph/pullback_geometry.py
- experiments/gelu_noneg_corrected_cycle228.py
- experiments related to cycle 181 (search for relu_bias / activation_swap)

# Constraints
- Calib slice: 200 wikitext-2 val seqs × 128 tokens, seed 42 (matches cycle 252-era convention).
- For each variant, collect_residual_trajectory → 13-tuple of (B, T, 768) tensors.
- Compute salience and curvature with reduce='none' to keep per-token; aggregate to mean / p10 / p90 over (B, T).
- Compute ρ(S_l, κ_l) per variant (Pearson correlation across layers).
- Plot per-variant κ_l and S_l profiles overlaid.
- Confirmed if max_l |κ_GELU − κ_ReLU| / κ_GELU ≥ 0.30 in at least one of layers 0-6 AND |ρ_ReLU(S, κ)| ≤ 0.5.
- Refuted if all per-layer relative κ differences < 0.10 AND ρ_ReLU ∈ [-0.85, -0.65].
- Inconclusive otherwise.
- Trainable params: 0 (analysis only).

# Handoff
Plan agent: locate the cycle 181 ReLU+bias patch (or recreate from ReLU(z) + b_leak with b_leak chosen to match GELU's first moment). Build the 3 variants: baseline GELU (untouched model), ReLU+bias patch (replace mlp.act in all 12 blocks with ReLU + add b_leak as a post-relu offset), gelu_noneg+μ (cycle 228's GeluNonegCorrected with μ_l per block). Run pullback geometry on each. Output JSON + summary + matplotlib plot saved to autoresearch/results/.
