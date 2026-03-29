# Cycle 098 Plan: Joint FFN+Attention Compression (GPT-2)

**Item:** 113
**Script:** `experiments/joint_ffn_attn_compression_gpt2.py` (new)

## Pre-flights

**P1 -- Null hypothesis:** FFN and attention compression errors interact super-additively through the shared residual stream. Joint PPL > sum of individual deltas.

**P2 -- Criteria (verbatim from agenda):**
- Confirmed if: joint compression on k=4 blocks achieves ≤+20% PPL (FFN-only was +5%, so attention adds ≤+15%).
- Refuted if: joint k=2 already >+20% (joint compression is strictly worse than sum of parts).
- Inconclusive if: joint k=4 is +20-40% (viable but cascade worsens).

**P3 -- Novelty vs replication:** FFN-only k=4 (+5%) and attention-only 3-block (+13%) established in cycles 089 and 095. This is the first test combining both compression modes on the same blocks.

**P4 -- Confounds:** Block overlap — FFN blocks [7,5,4,1] and attention viable blocks [1,4,9] share blocks 1 and 4. Joint compression on overlapping blocks tests worst-case interaction (same block doubly compressed). This is a feature, not a bug: real deployment would compress both in the same block.

**P5 -- Cost estimate:** 4 conditions × 3 seeds × 200 eval seqs + 3 FFN polish runs = ~20 min on MPS.

**P6 -- Data dependencies:** `data/openwebtext_sample.txt` (same as cycles 089/095/097).

**P7 -- Abort conditions:** Baseline PPL outside [25, 65]. Any condition PPL > 10× baseline.

## Implementation

1. **Baseline.** Load GPT-2, eval PPL on OWT, 200 seqs, 3 seeds [42, 123, 7]. Verify ~51.66.

2. **Condition A: FFN-only (replication).** Compress FFN in blocks [7,5,4,1] at 6x (topk+lstsq + 500 Adam steps, lr=1e-4, W_down only). Eval PPL. Expected: ~+5% (cycle 089).

3. **Condition B: Attention-only (replication).** Compress attention QKV in blocks [1,4,9] at rank-11 SVD (using `_compress_attn_block_inplace` from cycle 095). Eval PPL. Expected: ~+13% (cycle 095).

4. **Condition C: Joint FFN+attention.** Apply BOTH compressions simultaneously: FFN topk+lstsq+polish on blocks [7,5,4,1] AND attention rank-11 SVD on blocks [1,4,9]. Polish FFN W_down only (attention SVD is fixed, not polished). Eval PPL.

5. **Interaction analysis.**
   ```
   additive_prediction = ppl_A + ppl_B - baseline
   interaction = ppl_C - additive_prediction
   if interaction > 0: super-additive (amplifying)
   if interaction ≈ 0: orthogonal (additive)
   if interaction < 0: sub-additive (cancellation)
   ```

6. **Verdict logic:**
   ```
   delta_joint_k4 = (ppl_C - baseline) / baseline * 100
   # Also compute delta for k=2 subset (blocks [7,5] FFN + blocks [1,4] attn)
   if delta_joint_k4 <= 20.0:    CONFIRMED
   elif delta_joint_k2 > 20.0:   REFUTED
   else:                         INCONCLUSIVE
   ```
   Note: "joint k=2" in the refutation criterion means the 2 most-compressed blocks from each modality, not 2 total blocks. Use blocks [7,5] (FFN) + [1,4] (attention) — the 4 blocks with lowest individual damage.

7. **Output:**
   - JSON: `autoresearch/results/cycle_098_joint_compression.json` — baseline, conditions A/B/C PPLs (mean ± std), interaction term, per-seed values.
   - Plot: grouped bar chart — baseline, FFN-only, attn-only, additive prediction, joint actual. Error bars from 3 seeds. Save as `..._joint_compression.png`.

8. **Agenda update + commit** per standard autoresearch flow.
