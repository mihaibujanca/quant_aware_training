# Item 154: Correction Subspace Analysis — What Do LoRA Corrections Learn? — GPT-2 FFN k=4

## Item
Cycle 127 found corrections partially align with compression error (max cosine
0.46) but mostly learn a novel subspace — alignment with both error and attention
subspaces was near null baseline. Item 154 does sharper characterization via
projection decomposition + ablation: train LoRA corrections (r=8), SVD the
correction matrix, project onto 3 orthogonal subspaces (compression error PCs,
attention amplification PCs, novel complement), and measure each component's
contribution to PPL recovery via zeroing ablation. Cross-block sharing test
probes whether corrections are block-specific or generic.

- **Confirmed** if a single subspace accounts for >=60% of correction PPL recovery.
- **Refuted** if all three subspaces contribute roughly equally (diffuse).
- **Inconclusive** if dominant subspace varies by block or seed.

## Context
- **Cycle 127** (item 144): SVD alignment analysis. Max alignment with error
  subspace: 0.18 (block 1), max with attention: 0.05. Both near null baseline
  (p99=0.13). INCONCLUSIVE verdict — corrections don't strongly align with
  either reference. The "novel subspace" question is unresolved.
- **Cycle 138** (item 152): AE+corrections WORSENED PPL (76.7 vs 74.4 AE-only).
  Corrections trained for topk error patterns may actively interfere when the
  compression method changes. This strongly suggests corrections learn something
  specific to topk error structure, not a generic compensation.
- **Cycle 120** (item 119): LoRA corrections recover ~74% of k=4 degradation
  (PPL 69.4 vs float 57.6, uncorrected 95.9). r=8, 49K params.
- **Cycle 104**: cascade is attention-driven (70%+ in 8/11 block transitions).
- **Working recipe**: topk k=4 + lstsq + 500 Adam -> PPL ~70. + LoRA -> ~59.

## Dependencies
- Cycle 127 (soft): provides alignment baseline and code patterns.
  `experiments/svd_alignment_cycle127.py` — LowRankCorrection, P1Model classes.
- `aleph/lm_geometric_compression.py` — compression pipeline.
- Float PPL baseline: 57.65. Corrected PPL baseline: 69.4 (cycle 127).

## Key Files
- `experiments/svd_alignment_cycle127.py` — correction training + SVD extraction
- `aleph/lm_geometric_compression.py` — derive_ffn_compression, swap_compressed_ffns

## Constraints
- CPU-only. Correction training ~15 min (2000 steps). SVD/projection fast.
- 3 seeds x 4 blocks = moderate combinatorial load; keep ablation to 3 subspaces only.
- Must re-train corrections per seed (cannot reuse cycle 127's single-seed corrections).
