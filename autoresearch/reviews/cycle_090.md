verdict: REJECT — 1 new blocking issue
round: 2
issues: 1
notes: 1

## Round 1 fix verification

**I1 (blocking — V-projection ambiguity): FIXED.** Step 4 now explicitly states "compress the full concatenated attention weight matrices" with shape [768, 768] per projection. The method is unambiguous: rank-128 truncated SVD on each of W_V, W_Q, W_K, W_O. Clear.

**I2 (non-blocking — 144-head ablation): FIXED.** Step 3 now performs 12-block ablation (zero all heads in one block), not 144-head ablation. Faster, directly relevant. Step 2 still measures per-head SVD rank (all 144 heads) which is fine — it's one forward pass with hooks, cheap.

**Note 1 (SVD baseline): Addressed.** Step 4 now uses truncated SVD directly as the primary method, superseding the random-projection + lstsq approach. However, this creates an inconsistency (see Issue 1 below).

**Note 2 (quantify consistency): FIXED.** Validation section now specifies "within 2× of cycle 022 at the same layer."

## R1: Premise validity — PASS with concern

The experiment scope has expanded: the revision compresses all four projection matrices (W_Q, W_K, W_V, W_O) rather than V-only as in round 1. Compressing Q and K changes the attention *pattern* (softmax(Q K^T / sqrt(d))), not just the values being aggregated. R1's analysis that "the attention pattern Q@K^T is unchanged" (R5 MP4) no longer holds. This is a substantively different and more aggressive experiment. That said, compressing the full block is a legitimate feasibility scan — it just tests a different hypothesis than V-only compression.

## R2: Null baseline — ISSUE 1 (blocking)

**The "6× compression" claim is numerically wrong.** The plan says "single-block low-rank compression at 6×" throughout (title, hypothesis, steps). It justifies this as 768/128 ≈ 6. But that is a *rank reduction ratio*, not a *parameter compression ratio*.

For a [768, 768] matrix, rank-128 truncated SVD produces factors U [768, 128] and V^T [128, 768], totaling 2 × 768 × 128 = 196,608 parameters vs the original 768 × 768 = 589,824. That is **3.0× parameter compression**, not 6×.

The hypothesis criteria ("≤ +5% single-block swap PPL at 6×") and null baseline (">+50% PPL per block" for random projection at "6×") are calibrated to a 6× number that does not match the actual compression achieved. At the true 3× compression, the null baseline prediction should be much lower (discarding ~67% of rank, not ~83% of information), and the +5% success criterion may be too lenient.

This matters because:
1. The verdict (confirmed/refuted/inconclusive) hinges on PPL thresholds calibrated to the wrong compression ratio.
2. Comparison to FFN compression cycles (which presumably use consistent compression accounting) will be apples-to-oranges.
3. If the plan means 6× parameter compression, the target rank should be ~64 (768²/(2×768×64) ≈ 6), which is much more aggressive.

**Fix:** Decide whether "6×" means rank-reduction (rank 128, ~3× params) or parameter-compression (rank ~64, ~6× params). Recalibrate hypothesis thresholds to match whichever definition is chosen. State the definition explicitly.

## R3: Statistical/sample adequacy — PASS

3 seeds for probe sampling, full eval set for PPL, 12 blocks × 4 matrices. Adequate for a feasibility scan.

## R4: Method correctness — PASS

Truncated SVD on the full concatenated matrices is well-defined. Replacing original weights with the low-rank approximation and measuring PPL is a clean single-block swap experiment. The validation checks (baseline PPL ≈ 57.65, ablation should show significant damage, consistency with cycle 022 ranks) are appropriate.

## R5: Known pitfall scan — PASS

- The P5 null baseline text still references "random V-projection + lstsq" but Step 4 uses truncated SVD. Minor inconsistency — should update P5 to match the actual method. Non-blocking.
- Compressing Q and K simultaneously means attention patterns change, which could interact non-linearly with V compression. A single-block swap still isolates this to one block, so the measurement is interpretable even if the effects are entangled within the block.

## Note 1 (informational): P5 text inconsistency

P5 says "random V-projection + lstsq should yield >+50% PPL per block" but the experiment now uses truncated SVD on all four matrices, not random projection + lstsq on V alone. The null baseline description should be updated to match the revised method: e.g., "random rank-128 projection (no SVD) of all four attention matrices should yield significantly worse PPL than truncated SVD."
