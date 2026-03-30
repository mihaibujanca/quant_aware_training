# Cycle 130 — Item 87: Cross-block Procrustes Alignment

## Item
**Cross-block weight sharing via Procrustes alignment — GPT-2 FFN [MOONSHOT]**

## Context
Item 80 (FFN Jacobian effective rank census, cycle 068, CONFIRMED) established that 10/12 blocks have rank_99 ≤ 500 vs 592 random baseline. Item 87 hypothesizes: if blocks have similar spectral structure, their compressed W_down matrices (k=4, from prior topk+lstsq pipeline) may be related by orthogonal rotation. Procrustes alignment finds rotation R minimizing ||W_down_i − R·W_down_j||_F for each block pair.

## Dependencies
- Does NOT depend on item 80 (but benefits from it conceptually).
- Requires: topk+lstsq compression baseline (k=4) already trained from prior cycles.
- Baseline PPL: ~80 (stacked k=4 independent weights); correction LoRA adds ~14% gain (cycle 125 recipe).

## Key Files
- `experiments/procrustes_linearity_gpt2.py` (cycle 018, early work on Procrustes, pre-topk+lstsq)
- `experiments/svd_alignment_cycle127.py` (cycle 127, subspace alignment methods, reference template)
- `experiments/ffn_jacobian_subspace_alignment.py` (cycle recent, alignment exploration)
- Compression baseline: `aleph/lm_geometric_compression.py` (CompressedFfnPack, _evaluate_model)

## Constraints
1. **Procrustes metric**: all 6 pairwise distances (4 blocks → C(4,2) pairs). Residual threshold < 10% for CONFIRM, > 30% for REFUTE.
2. **Test protocol**: if pairs align, build shared-W_down + per-block rotations R_i (4×768×768 params), measure stacked PPL. Must be ≤ 2pp degradation vs independent.
3. **Extreme case**: ONE shared W_down + 4 learned R_i. Optional: if residuals very small, test low-rank R (e.g., rotation parametrized via Cayley).
4. **Data**: OWT 1024-token batches (standard probe corpus).

## Handoff
Compute pairwise Procrustes residuals on the 4 topk+lstsq-compressed k=4 blocks. If any pair < 10%, implement shared-weight variant and measure PPL. Report: (a) all 6 pairwise distances, (b) PPL gap (independent vs shared), (c) parameter savings.
