# Cycle 114 Plan: Linearity-Adaptive Compression Ratio — GPT-2 FFN

**Item:** 125 | **Script:** `experiments/linearity_adaptive_ratio_gpt2.py` (new)

## Criteria (verbatim from agenda)
- **Confirmed if:** non-uniform (b) outperforms uniform (a) by >=3% relative PPL at matched total params.
- **Refuted if:** uniform within 1% of best non-uniform.
- **Inconclusive if:** 1-3% or dependent on block selection.

## Budget-Matching Correction

Agenda says "blocks 4,7 at 8x, blocks 1,11 at 4x" but 8x (k=384) + 4x (k=768) sums
to 2×384+2×768=2304, not 2048. To match uniform budget (4×512=2048):

| Condition     | Blocks 4,7 (k) | Blocks 1,11 (k) | Total | Eff. ratio mid | Eff. ratio edge |
|---------------|-----------------|------------------|-------|----------------|-----------------|
| uniform       | 512             | 512              | 2048  | 6.0x           | 6.0x            |
| middle-heavy  | 384             | 640              | 2048  | 8.0x           | 4.8x            |
| edge-heavy    | 640             | 384              | 2048  | 4.8x           | 8.0x            |

All conditions sum to exactly 2048 channels. Ratio spread: 8.0x vs 4.8x (~1.67 ratio).

## Pre-flights
**P1 -- Prior:** Cycle 094 (damage-based allocation, different blocks [7,5,4,1]) = REFUTED
(+0.24%). Linearity-based is a different hypothesis but same mechanism — risk of repeat.
**P2 -- Cost:** 3 allocations × 3 seeds × 4 blocks = 36 block fits + 9 stacked evals. ~45 min MPS.
**P3 -- Abort:** Baseline PPL outside [25, 65]. Any stacked PPL > 500.
**P4 -- Budget:** Assert sum(k_values) == 2048 in code for every allocation.

## Implementation

1. **Baseline.** Load GPT-2, eval 200 seqs OWT. Abort if outside [25, 65].

2. **Atlas.** Per seed, probe FFN geometry for blocks [1,4,7,11] using 4096 tokens.

3. **Per-allocation compression.** For each allocation (uniform, middle-heavy, edge-heavy):
   Build `CompressionBudgetSpec` with `per_layer_hidden={1:k_1, 4:k_4, 7:k_7, 11:k_11}`.
   Use `_derive_block_pack_from_atlas` (reuse from cycle 094 pattern).

4. **Polish.** Per block, 500 Adam steps on W_down, lr=1e-4. Polish each block in
   single-block mode, then inject polished weights into stacked model (same as cycle 094).

5. **Eval.** Stacked 4-block PPL on 200 OWT eval seqs.

6. **Param verification.** For each allocation, log:
   - Per-block k and compressed_param_count
   - Total compressed params across 4 blocks
   Assert all allocations have identical total param count (tolerance < 0.1%).

7. **Verdict logic.**
   - uniform_ppl = mean across 3 seeds of condition (a).
   - best_nonunif_ppl = min(mean(b), mean(c)).
   - delta = (uniform_ppl - best_nonunif_ppl) / uniform_ppl × 100.
   - If delta >= 3%: CONFIRMED. If delta < 1%: REFUTED. Else: INCONCLUSIVE.

8. **Output.**
   - JSON: `autoresearch/results/cycle_114_linearity_adaptive_ratio.json`
   - Bar chart: 3 conditions × 3 seeds + mean.
   - Manifest: `autoresearch/results/cycle_114_manifest.txt`
