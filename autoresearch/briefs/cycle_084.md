# Cycle 084 — Item 95: Data-Conditioned Jacobian Clustering for Merge Candidates

## Item
**95. Data-conditioned Jacobian clustering for merge candidates — GPT-2 FFN**

Probe FFN input-output behavior to find mergeable structure. For block 7, compute per-token diagonal GELU'(z) patterns on 1024 mixed-corpus tokens. Cluster with k-means, k in {8, 16, 32, 64}. Within each cluster, tokens share a similar effective linear map. Measure: (a) cluster-centroid reconstruction MSE vs full-precision output, (b) K* for <5% output norm error, (c) whether centroid Jacobians are low-rank.

## Context
- **Antecedent**: Item 97 (GELU activation pattern diversity) provides cluster analyses; item 95 adds explicit merge-candidate identification and Jacobian properties.
- **Motivation**: Items 98–99 (centroid affine map compression) depend on this analysis for real compression implementation.
- **Status**: Item 97 is P1 (earlier priority, not yet run); item 95 is P2 moonshot.

## Dependencies
- **Item 80** (CONFIRMED, cycle 068): FFN Jacobian effective rank census. **Outcome**: rank_99 per block 434–510 (vs 592 Gaussian), genuinely low-rank. No single-block Jacobian rank metric predicts compression difficulty; anti-correlated with P1 variation.
- **Item 97** (PENDING): GELU activation pattern diversity analysis — must run first to define clusters K* and continuous GELU'(z) vectors.

## Key Files
- `experiments/ffn_jacobian_rank_census.py` — reference Jacobian SVD pipeline (cycle 068)
- `experiments/gpt2_local_complexity.py` — torchfunc.jacrev(ffn_forward) pattern (lines 152–154)
- `experiments/mixed_corpus_lstsq_refit_gpt2.py` — mixed-corpus probe loading (line 222)

## Constraints
- **Block 7 only** (single-block analysis, no stacking)
- **1024 mixed-corpus tokens** (Shakespeare, WikiText-2, OWT, C4)
- **K-means on 3072-dim soft activation patterns** [0,1] (per-token GELU'(z))
- **K in {8, 16, 32, 64}** — find K* where centroid MSE < 5% output norm
- **Success**: K* ≤ 32; **Refute**: K* > 128; **Inconclusive**: 32 < K* ≤ 128
- **No experiments run** — context gathering only.
