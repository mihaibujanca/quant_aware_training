# Cycle 127 — Item 144: SVD of Learned Corrections: Geometric Alignment

## Task
Extract trained P1 LoRA corrections (r=8, blocks [1,4,5,7]), SVD them, and measure cosine alignment against three reference subspaces: (a) compression error E_l, (b) attention amplification directions (cycle 104), (c) bump channel directions (cycle 110).

## Hypothesis
Learned corrections align primarily with attention amplification directions (cosine > 0.5 for top-4 singular vectors in >= 3/4 blocks), indicating corrections undo what attention does to the compression error rather than directly correcting the raw error.

## Pre-flight Results
- **P1 (duplicate)**: No duplicate. Item 91 analyzed compression bases, not learned corrections.
- **P2 (dependency)**: Item 142 done (cycle 125). Must re-train P1 (seed 42, 2000 steps, lr=1e-4).
- **P3 (concrete)**: Criteria fully specified. **P5 (null)**: subspace-vs-subspace protocol (see step 7).
- **P6 (architecture)**: Pure analysis. **P7 (power)**: 4 blocks x 4 vectors = 16 alignment checks per ref.

## Steps
1. Re-use cycle 125 setup: load GPT-2, compress blocks [1,4,5,7] k=4 topk+lstsq+polish. Build P1Model(rank=8). Train 2000 steps (seed 42, lr=1e-4). Extract U (768x8) and V (8x768) per block.
2. Per block: SVD of C = U @ V to get top-8 left/right singular vectors + singular values.
3. Compression error E_l: run float and compressed forward on holdout tokens. **Hook placement: post-FFN, pre-LayerNorm of next block** (raw residual-stream sum after `x = x + ffn(ln2(x))`). Compute E_l = X_float - X_compressed at these hooks. SVD of E_l to get top-8 directions. This ensures E_l lives in the same pre-LayerNorm residual space where C = UV^T is applied.
4. Attention amplification directions: re-derive from float vs compressed forward (cycle 104 data has only scalars). Hook attention output delta per block transition at the same post-FFN/pre-LN point. SVD to get top-8 directions.
5. Bump channel directions: recompute from cycle 110 protocol. Per block, identify bump-dominated channels (bump_frac > 0.5), construct axis-aligned indicator vectors.
6. Alignment: per block, compute |cos(u_i, v_j)| for top-4 correction SVD vectors vs top-4 of each reference. Report max-over-pairs per block.
7. **Null baseline (subspace-vs-subspace)**: draw two independent random rank-8 matrices in R^768 (Gaussian entries), SVD each to get top-4 vectors, compute max |cos(u_i, v_j)| over the 4x4 pairs. Repeat 1000 times. Report mean and 99th percentile of max-over-pairs. This calibrates chance alignment under the exact same measurement protocol used in step 6.
8. **Secondary (non-blocking)**: compute data-weighted SVD of C @ X_holdout^T per block to check whether geometrically prominent correction directions also carry energy on real data.
9. Verdict per agenda criteria on top-4 alignment across 4 blocks.

## Validation
- Re-trained P1 PPL within 1 of cycle 125 (~69.5). If gap > 2: debug before proceeding.
- Null baseline max-over-pairs 99th percentile should be well below 0.5 (expected ~0.1-0.15 for 4-d in 768-d).
- Singular values of C non-degenerate (no collapsed rank).

## Artifacts
- `experiments/svd_alignment_cycle127.py` — single self-contained script.
- `autoresearch/results/cycle_127_svd_alignment.json` — per-block alignment matrices + singular values + null stats + verdict.
- `autoresearch/results/cycle_127_alignment_heatmap.png` — 4-block x 3-reference alignment heatmap.

## Fallbacks
- If all alignments < 0.3: INCONCLUSIVE, check propagated error (E at block l from earlier compression) as secondary.
- If attention perturbation SVD is expensive: process one block at a time on CPU.
