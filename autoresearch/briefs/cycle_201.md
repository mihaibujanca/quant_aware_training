# Cycle 201 Brief: Block 11 Geometry Deep-dive (Item 211)

## Item

211 — Block 11 geometry deep-dive — anomalous delta_norm and rank drop

## Context

Anomalies from cycle 198:
- Block 11 delta_norm=335.7 (vs median ~30, 11× larger than other blocks)
- Block 11 is borderline droppable (PPL 41.4 per cycle 187; highest of non-block-0 blocks)
- Cycle 184: rank_90 drops 434→215 at final block (consistent with self-correction)
- Cycle 200: block 11 Frob div decreases (0.211) vs block 10 (0.211) — slight self-correction

The question: is block 11 performing a "final projection" role analogous to block 0's initial
role? Block 0 handles high-diversity input (raw embeddings, rank=244); does block 11 push
the residual stream into a constrained output distribution?

## Experiment Design

1. **Block input/output rank**: measure rank_90 of block 11's output (not just input)
2. **Token-type breakdown**: does the large delta cluster on specific token types? Check:
   - Position in sequence (early vs late tokens)
   - Token frequency (common vs rare tokens)
   - Mean delta by top-50 most common tokens
3. **Delta direction analysis**: does block 11's delta point in a specific direction?
   Measure cos(delta_11, top_singular_vector_of_delta_11) across tokens.
4. **Weight analysis**: compare block 11's c_fc and c_proj weight SVD to other blocks.

## Key Files

- `experiments/block0_criticality_cycle198.py` — identical hook pattern, reuse
- `autoresearch/results/cycle_198_block0_criticality.json` — per-block metrics for comparison
- GPT-2 tokenizer for token-type analysis

## Constraints

- N_SEQS=50, SEQ_LEN=64 (same as cycle 198 for direct comparison)
- Compute block OUTPUT rank using block.register_forward_hook's out tensor
- Focus on blocks 10, 11, and for comparison blocks 3, 7 (mid-model)
