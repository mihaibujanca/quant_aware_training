# Cycle 267 — Item 277: Block 11 c_proj output rank decomposition

**Verdict: CONFIRMED** — block 11 eff_rank = 22.49 ≤ 50 (concentration story holds).

## Config
- GPT-2 small, wikitext-2 val, 200 × 128 seed 42, MPS
- Hook: `model.transformer.h[b].mlp.c_proj` OUTPUT → (25600, 768)
- Blocks: [1, 5, 11], trainable params = 0
- Cov = mlp_out.T @ mlp_out / N (768×768); eff_rank via eigh entropy
- V_mlp: top-20 eigenvectors of cov; V_wp: top-20 right SVs of c_proj.weight (3072, 768)
- lm_head.weight (50257, 768); top-200 rows by L2 norm → top-200 lm_head rows

## Per-block results

| block | eff_rank | top-1 energy | top-5 | top-20 | top-50 | top-200 | med top-5 cos V_mlp·top200_lm_head | med top-5 cos V_mlp·ALL_lm_head |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  |   1.94 | 0.9294 | 0.9415 | 0.9482 | 0.9559 | 0.9761 | 0.1586 | 0.1968 |
| 5  | 179.19 | 0.2462 | 0.3079 | 0.3750 | 0.4636 | 0.7135 | 0.1295 | 0.2825 |
| 11 |  22.49 | 0.4895 | 0.6818 | 0.7487 | 0.7936 | 0.8959 | 0.1231 | 0.2391 |

## Verdict logic
- CONFIRMED if eff_rank ≤ 50 OR med top-5 cos (top-200 lm_head) ≥ 0.5
- REFUTED if eff_rank ≥ 200 AND med top-5 cos (top-200 lm_head) ≤ 0.2
- Block 11: eff_rank 22.49 ≤ 50 → **CONFIRMED** (concentration clause satisfied, alignment clause not satisfied — 0.123 < 0.5)

## Geometric reading
- Block 11 c_proj output lives in a ~22-dimensional effective subspace out of 768 — top-1 eigenvector carries 48.95% of variance, top-5 carries 68.18%. Most of the 768-D output is dead.
- Block 5 is the anti-pattern: eff_rank 179.19 (≈1/4 of 768), top-1 only 24.62%. Block 5's c_proj output fills a moderate-rank subspace; no single direction dominates.
- Block 1 is the extreme case: eff_rank = 1.94, top-1 = 92.94%. Block 1's c_proj output is essentially a rank-1 operator at the output (one huge direction plus tiny residual).
- The eff_rank ordering block 1 << block 11 << block 5 (1.94, 22.49, 179.19) inverts the mid-block "structurally normal" picture: both the earliest and latest blocks have concentrated c_proj outputs; the mid block is the high-rank one.
- Alignment story is NEGATIVE: none of the top-5 V_mlp directions for any block align with the top-200 lm_head rows at > 0.35. The PPL-load of block 11 is NOT from reading into the high-norm unembedding rows. When restricted to top-200 lm_head rows, max cos for any block is ≤ 0.347; median of top-5 is ≤ 0.159.
- When we open cosine search to ALL 50257 lm_head rows, block 11's first 2 V_mlp directions hit 0.586 and 0.623 — so there ARE specific low-norm lm_head rows that block 11's top directions align with, but they're not in the top-200 by row norm, and even then only the top 2/5 V_mlp directions find them (3rd onward ≤ 0.24), so the median stays at 0.239.
- Operator-level V_wp alignment (SVD of c_proj.weight directly, ignoring activations) mirrors the output-level finding: block 11's top V_wp direction has max cos 0.353 with top-200 lm_head rows, median top-5 = 0.147.

## Compression implication
- Block 11's ~22-dim concentration opens a targeted probe: compress block 11 on a rank-22 (or rank-5 — top-5 already captures 68.18% of energy) subspace of its c_proj output. A K=22 or even K=5 compression of the OUTPUT side of block 11 could plausibly preserve most of its PPL contribution.
- Block 5 is the opposite — no targeted subspace compression at the output will work because the energy is distributed across ~179 dimensions. If block 5 compresses well (which it does in the 4-block / 2-block sweeps), the compression must be along the 3072-dim post-GELU INPUT side, not the 768-dim output side.
- Block 1's rank-1.94 output is the clean signature of the "early block does almost nothing new" pattern from cycles 218 / 260: the MLP at block 1 is essentially applying a single direction on top of whatever LN_2 produces.

## Raw numbers reference
- Block 11 top-5 max-cos per-V_mlp with top-200 lm_head rows: [0.347, 0.118, 0.066, 0.193, 0.123]
- Block 11 top-5 max-cos per-V_mlp with ALL lm_head rows: [0.586, 0.623, 0.162, 0.239, 0.210]
- Block 11 c_proj.weight top SV = 60.99, 20th SV = 17.76 (ratio ~3.4); Block 5: 27.62 → 9.93 (ratio 2.78); Block 1: 28.71 → 7.57 (ratio 3.79). Block 11's operator has ~2× the overall scale of blocks 1/5 but similar SV decay shape.

## Files
- `experiments/block11_cproj_decomp_cycle267.py`
- `autoresearch/plans/cycle_267.md`
- `autoresearch/briefs/cycle_267.md`
- `autoresearch/logs/cycle_267_run.log`
- `autoresearch/results/cycle_267_block11_cproj_decomp.json`
- `autoresearch/results/cycle_267_summary.md`
