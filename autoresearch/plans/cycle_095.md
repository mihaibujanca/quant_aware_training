# Cycle 095 Plan: Attention Compression v2 — GPT-2

**Item:** 109
**Script:** `experiments/attention_compression_v2_gpt2.py` (new)

## Pre-flights

**P1 — Null hypothesis:** GPT-2 attention QKV projections are full-rank at d_head=64. Rank-11 SVD destroys head function, PPL blows up for most blocks.

**P2 — Criteria (verbatim from agenda):**
- Confirmed if: rank-11 per-head swap PPL ≤ +5% for ≥ 6/12 blocks (all heads in block compressed simultaneously).
- Refuted if: rank-11 swap PPL > +15% for majority of blocks (attention matrices are not low-rank enough for 6× compression).
- Inconclusive if: works for some blocks but not others.

**P3 — Novelty vs replication:** Item 105 (cycle 090) was rejected twice for wrong compression ratio (rank-128 on [768,768] = 3x, not 6x). This corrects the math: per-head rank-11 on [768,64] projections = 5.4x.

**P4 — Confounds:** Block-level swap underestimates multi-block damage (cross-layer interactions). Mitigated: Phase 2 tests simultaneous compression of top-3 blocks. Output projection (W_O) not compressed here — scope limited to QKV feasibility. **Note:** W_O has 589,824 params per block, so even with 5.4× QKV compression, the real block-level compression is ~3.1× when W_O is untouched. True 6× attention compression would require compressing W_O as well; this experiment establishes QKV feasibility only.

**P5 — Cost estimate:** 12 block-level swaps × 3 seeds × 200 eval seqs = 36 evals. ~10-15 min on MPS.

**P6 — Data dependencies:** `data/openwebtext_sample.txt` (same as FFN compression cycles).

**P7 — Abort conditions:** Baseline PPL outside [25, 65]. Any block swap PPL > 10× baseline (diverged, likely bug).

## Implementation

1. **Load GPT-2** via `transformers` (same pattern as FFN compression scripts).
2. **Baseline PPL** on OWT, 200 eval seqs, 3 seeds [42, 123, 7]. Report mean ± std.
3. **Block-level SVD swap loop:** For each layer L (0..11):
   - Compress **all 12 heads simultaneously** in block L: for each head H (0..11), for each projection in {Q, K, V}:
     - Extract weight slice: `W[:, H*64:(H+1)*64]` shape [768, 64].
     - SVD: `U, S, Vt = torch.linalg.svd(W, full_matrices=False)`. Truncate to rank-11.
     - Reconstruct: `W_approx = U[:,:11] @ diag(S[:11]) @ Vt[:11,:]`.
   - Patch all 36 projections (12 heads × 3 QKV) into model.
   - Eval PPL under all 3 seeds. Record mean and std of delta%.
   - Restore original weights before next block.
4. **Bar chart:** 12 bars (one per block), height = mean PPL delta%, error bars = ±1 std across seeds. Horizontal lines at +5% and +15% thresholds. Save as PNG.
5. **Phase 2 — multi-block compression:** Take the top-3 most compressible blocks (lowest mean PPL delta). Compress all three simultaneously (rank-11 on all heads in all 3 blocks). Eval PPL with 3 seeds. This always runs regardless of Phase 1 outcome, giving multi-block evidence in every scenario.
6. **Verdict logic:**
   ```
   pass_count = sum(1 for L in range(12) if mean_delta[L] <= 5.0)
   if pass_count >= 6: CONFIRMED
   elif sum(1 for L in range(12) if mean_delta[L] > 15.0) > 6: REFUTED
   else: INCONCLUSIVE
   ```
7. **Output:** `autoresearch/results/cycle_095_attention_compression_v2.json`, `..._barchart.png`.
8. **Agenda update + commit** per standard autoresearch flow.
