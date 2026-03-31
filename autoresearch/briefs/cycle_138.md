# Item 152: Bottleneck AE k=4 Stacking — GPT-2 FFN

## Item
Cycle 137 CONFIRMED param-matched AE (d_bn=64) beats topk+lstsq by 3-3.4%
on single-block at 6x with 3x fewer learnable params. Now test the critical
stacking case: compress blocks [1, 4, 7, 11] simultaneously with AE, stack
all 4, measure full-model PPL. Compare to working recipe (topk+lstsq +
500-step polish, PPL ~70). Also test AE + LoRA corrections (r=8).

- **Confirmed** if AE stacking PPL < 67 (>4% improvement over topk ~70),
  OR AE+corrections PPL < 57 (>3% over topk+corrections ~59).
- **Refuted** if AE stacking PPL >= 69 (within 2% of topk, cascade dominates).
- **Inconclusive** if AE stacking PPL 67-69 (marginal, within seed noise).

## Context
- **Cycle 137** (item 151): param-matched AE CONFIRMED. B4 ae=53.65 vs topk=55.32
  (-3.0%), B7 ae=54.46 vs topk=56.36 (-3.4%). AE uses 247K params vs topk 788K.
  Advantage is architectural (nonlinear encoder captures channel correlations).
- **Cycle 131** (item 88): original AE POC, INCONCLUSIVE (2.5x param confound).
- **Working recipe**: topk+lstsq+500 Adam at k=4@6x -> PPL ~70 post-polish.
  + LoRA corrections (r=8, 49K params): PPL ~59.
- **Cycle 133**: polish erases pre-polish differences in hybrid methods. Both
  pre- and post-polish comparison is essential.

## Dependencies
- Cycle 137 (hard): confirmed AE advantage at single-block. Unblocks stacking test.
- BottleneckFFN class from `experiments/ae_param_match_k512_cycle137.py` (reuse).
- LoRA correction infrastructure from existing recipe (if testing AE+corrections).

## Key Files
- `experiments/ae_param_match_k512_cycle137.py` — BottleneckFFN class, AE train loop
- `aleph/lm_geometric_compression.py` — swap_compressed_ffns, derive_ffn_compression

## Constraints
- CPU-only, ~5h wall time for full matrix. Use seed 42 only for initial sweep.
  Escalate to 3 seeds only if AE stacking PPL < 67 (CONFIRMED threshold).
- Float PPL baseline: 57.65 (from cycle 115 manifest).
