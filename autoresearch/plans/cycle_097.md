# Cycle 097 Plan: Block-0-Skip Attention Stacking (GPT-2)

**Item:** 114
**Script:** `experiments/block0_skip_attention_gpt2.py` (new)

## Pre-flights

**P1 -- Null hypothesis:** Skipping block 0 does not help; 5-block simultaneous attention compression exceeds +20% PPL, because the cascade ceiling is structural (c≈0.41) and unrelated to block 0.

**P2 -- Criteria (verbatim from agenda):**
- Confirmed if: 5-block attention-only (skip block 0) achieves ≤+10% PPL.
- Refuted if: 5-block attention-only (skip block 0) >+20% PPL (block 0 isn't the only bottleneck).
- Inconclusive if: +10-20%.

**P3 -- Novelty vs replication:** Cycle 095 tested 3-block simultaneous (+13%). This is the first 5-block test with an explicit block-0 skip strategy.

**P4 -- Confounds:** Going from 3 to 5 blocks increases cascade exposure. If result is >+20%, it could mean (a) block 0 isn't the bottleneck, or (b) 5 blocks is simply too many regardless. Mitigated: also test 3-block-skip-0 (same 3 blocks as cycle 095 minus block 0 if it was included, else the top-3 from the viable set) to isolate the block-count effect from the block-0 effect.

**P5 -- Cost estimate:** 3 conditions × 3 seeds × 200 eval seqs = 9 evals. Position probe: 2 conditions × 3 seeds = 6 evals. ~15 min on MPS.

**P6 -- Data dependencies:** `data/openwebtext_sample.txt` (same corpus as cycle 095).

**P7 -- Abort conditions:** Baseline PPL outside [25, 65]. Any compressed PPL > 10× baseline (diverged).

## Implementation

1. **Baseline.** Load GPT-2, eval PPL on OWT, 200 seqs, 3 seeds [42, 123, 7]. Verify ≈51.66.

2. **Condition A: 5-block skip-0.** Compress blocks [1, 3, 4, 9, 10] simultaneously at rank-11 QKV SVD (using `_compress_attn_block_inplace` from cycle 095). Eval PPL.

3. **Condition B: 3-block skip-0 control.** Compress top-3 of the viable set (blocks [1, 3, 4] — lowest single-block deltas from cycle 095). Eval PPL. Compare to cycle 095's 3-block +13% (which may have included block 0 neighbors). This isolates block-count vs block-0 effects.

4. **Condition C: position probe.** Compress only block 0 at rank-11. Eval PPL on two token subsets: positions 0-16 vs positions 112-128. Report attention output L2 delta per subset (hook after `attn.c_proj`). This tests whether block-0 sensitivity is position-embedding-driven.

5. **Verdict logic:**
   ```
   delta_5block = (ppl_A - baseline) / baseline * 100
   if delta_5block <= 10.0:    CONFIRMED
   elif delta_5block > 20.0:   REFUTED
   else:                       INCONCLUSIVE
   ```
   Secondary diagnostics (not for verdict): 3-block-skip-0 delta, position probe L2 ratios.

6. **Output:**
   - JSON: `autoresearch/results/cycle_097_block0_skip_attention.json` — baseline PPL, all condition PPLs (mean ± std over seeds), per-seed values, position probe L2 deltas.
   - Plot: grouped bar chart — baseline, 3-block-skip-0, 5-block-skip-0, plus cycle-095 3-block reference line. Save as `..._block0_skip.png`.

7. **Agenda update + commit** per standard autoresearch flow.
