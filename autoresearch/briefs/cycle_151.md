# Item
161 — Correction subspace analysis on unpolished model — GPT-2 FFN k=4

# Context
- Cycle 142 (item 154): attempted subspace decomposition but trained corrections on a POLISHED model. PPL worsened 71->88.7 because corrections and polish compete for the same error budget. Verdict: INCONCLUSIVE (method failure).
- Cycle 115 (item 118): the working correction protocol. Corrections trained on the UNPOLISHED topk+lstsq model (PPL ~85 pre-polish). r=8 LoRA recovered PPL to ~59. This is the correct baseline.
- Cycle 127 (item 144): SVD alignment found max cosine 0.18 with error subspace — near null. Corrections mostly learn a "novel" subspace. But alignment != ablation importance.
- Cycle 138 (item 152): corrections trained for topk error worsen AE models (76.7 vs 74.4). Corrections are topk-specific, not generic.
- The composition barrier (cycle 138): independently trained blocks don't compose. If corrections primarily compensate for cascade propagation (not local error), that's evidence the barrier is about inter-block error propagation.

# Dependencies
- None. Reuses cycle 142 code with polish removed.

# Key Files
- `experiments/correction_subspace_cycle142.py` — fork this, remove polish (lines 108-131)
- `experiments/svd_alignment_cycle127.py` — LowRankCorrection, P1Model classes
- `aleph/lm_geometric_compression.py` — compression pipeline, swap_compressed_ffns

# Constraints
- MUST skip polish entirely. The compressed model should have PPL ~85, not ~70.
- LoRA r=8, 2000 steps, lr=1e-4 (same as cycle 115/142).
- Blocks [7, 5, 4, 1] at 6x. Seed 42 only (screening).
- 200 OWT eval sequences.
- Decomposition: 2-way (local error vs cascade/novel), not 3-way. Cycle 127 showed attention subspace is near-null; merge with novel.
- Corrected PPL should land in [60, 75] range (cycle 115 got ~59 with r=8).

# Handoff
Fork `experiments/correction_subspace_cycle142.py` to `experiments/correction_subspace_unpolished_cycle151.py`. Remove `build_compressed_model`'s polish loop (lines 108-131). Keep everything else: decomposition, projection, ablation. Save JSON to `autoresearch/results/cycle_151_correction_subspace_unpolished.json`.
