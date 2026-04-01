# Cycle 179 Review — Spectral Channel Selection (Item 189)

Item 189. Reviewer: self-review agent.

## R1: Premise validity — OK

Spectral preservation of W_down is a reasonable proxy for maintaining the block's linear mapping structure. The PruneNet idea (selecting channels to preserve singular value distribution) has theoretical grounding: if the pruned submatrix preserves the spectral profile, the resulting linear map (post-activation) is closer to the original in operator norm. The comparison against activation-norm topk is fair — both select 512 channels, same lstsq fitting, same polish.

## R2: Measurement soundness — OK

Both conditions share identical lstsq + polish pipeline, differing only in channel selection. Baseline PPL check gates against bad data. KS-distance is a standard nonparametric statistic for comparing distributions. Per-block channel overlap metric quantifies whether spectral and topk select similar channels (high overlap would make any PPL difference suspicious).

## R3: Capacity and statistical adequacy — OK

3 seeds, 200 eval sequences each. The spectral selection is deterministic (greedy on weights, no data randomness), so seed variation comes only from corpus shuffling and polish optimization. This is sufficient for a ±2 PPL verdict threshold.

## R4: Method-architecture alignment — MINOR NOTE

The greedy SVD loop (512 steps × ~1024 candidate SVDs of 768×|S| matrices) is computationally expensive. For |S|=512, each SVD is O(768² × 512). With 1024 candidates per step, that's ~500K SVDs total per block. This could take 10+ minutes per block.

**Mitigation:** The plan already notes restricting candidates to top-1024 by column norm. Additionally, a cheaper proxy for KS-distance — using the Frobenius norm ratio ||W[:,S]||_F / ||W||_F — could be used for the first 256 channels, switching to exact KS only for the final 256. However, since this is a one-time computation and we only have 4 blocks, runtime is acceptable for a research experiment (estimate: 30-60 min total).

**Alternative acceleration:** Instead of per-candidate SVD, use the approximation that adding column c to S changes the spectrum approximately by the projection of W[:,c] onto the current singular vectors. Rank-1 update of SVD (Brand's method) would make this O(768 × |S|) per candidate. But for a first test, brute force is fine.

## R5: Known pitfall scan — OK

- **MP3 (Underdetermined):** Not applicable — channel selection is weight-only, lstsq is the same as baseline.
- **MP5 (Insufficient capacity):** k=512 same as topk baseline, fair comparison.
- **MP1 (Dead channels):** Spectral selection could pick dead channels (high W_down norm but never activated). The plan should exclude dead channels (same dead_mask as topk). This is handled by using the atlas's dead_mask.
- No other pitfall applies.

## Summary

No blocking issues. One advisory: exclude dead channels from spectral candidates (use atlas dead_mask). Runtime may be long but acceptable for 4 blocks.

**Verdict: ACCEPT.**
