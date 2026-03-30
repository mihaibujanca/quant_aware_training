# Cycle 132 Plan: Attention Projection Pruning (Item 126) — REVISED

## Hypothesis
Constraining the rank of the composed W_V @ W_O map in attention heads of high-amplification layers reduces cascade amplification without destroying task performance. Applied uniformly to all heads per layer (cycle 113 refuted per-head concentration).

## Pre-flights
- P1 (null): Rank-16 retains 25% of 64 dims. If cascade scales with rank, expect ~75% reduction — but attention may re-route.
- P2 (deps): Cycle 104 (attn_frac), cycle 113 (per-head uniformity) both complete.
- P3 (cost): 3 ranks × 3 seeds × ~7 layers + 9 stacked PPL evals ≈ 72 forward passes. ~30 min MPS.
- P4 (abort): Baseline PPL outside [25, 65]. Any SVD-modified model produces NaN.

## Target Layers
From cycle 104 attn_frac:
- **Group A** (attn_frac > 1.0): layers 0,1,3,4,5,10
- **Group B** (attn_frac 0.9–1.0): layers 2,6
- **Group C** (control): layer 10→11 — only layer with concentrated heads (cycle 113)

## Implementation

**Script**: `experiments/attn_proj_pruning_cycle132.py`

1. **Baseline.** Load GPT-2, eval PPL on OWT (200 seqs), seeds [42, 123, 7]. Abort if outside [25, 65].

2. **Composed SVD replacement.** For each target layer L, head h, rank r ∈ {8, 11, 16}:
   - Extract V_h (768→64) from `c_attn.weight[:, 1536+h*64:1536+(h+1)*64]`.
   - Extract O_h (64→768) from `c_proj.weight[:, h*64:(h+1)*64]`.
   - Compose M_h = V_h @ O_h (768×768, rank 64). SVD: U, S, Vt = torch.linalg.svd(M_h).
   - Truncate to rank r: M_h_r = U[:,:r] @ diag(S[:r]) @ Vt[:r,:].
   - Factor back: V_h' = U[:,:r] @ diag(sqrt(S[:r])) → shape (768, r), O_h' = diag(sqrt(S[:r])) @ Vt[:r,:] → shape (r, 768). Inject as rank-r replacement: store M_h_r and apply via forward hook that replaces the head's V@O contribution in the residual stream.

3. **Cascade measurement.** Hook points: **post-FFN, pre-LayerNorm** (raw residual stream before next block's LN1). Compress source block FFN (k=4, topk+lstsq+500 Adam, 6×). Measure ||delta_downstream|| / ||delta_source|| with and without SVD pruning. Report per-layer amplification ratio.

4. **Stacked PPL.** Apply SVD pruning to all Group A layers + compress FFN blocks [7,5,4,1] at k=4. Eval PPL. Compare to FFN-only stacked PPL.

5. **Group C test.** Layer 10→11: SVD top-3 heads (cycle 113 data) vs all 12 heads. Compare cascade reduction.

## Verdict Logic
- Mean amp reduction across Group A layers, averaged over 3 seeds.
- Stacked PPL delta = PPL(FFN+SVD) − PPL(FFN-only).
- **CONFIRMED**: amp reduction ≥ 20% relative AND stacked PPL improves by ≥ 2% of baseline PPL (i.e., ≥ ~2.5 points at baseline ~125).
- **REFUTED**: stacked PPL degrades (PPL(FFN+SVD) > PPL(FFN-only) + 1).
- **INCONCLUSIVE**: amp reduced but PPL within ±1 of FFN-only.

## Artifacts
- `experiments/attn_proj_pruning_cycle132.py`
- `autoresearch/results/cycle_132_attn_pruning.json`
- `autoresearch/results/cycle_132_amp_reduction.png`
- `autoresearch/results/cycle_132_manifest.txt`
